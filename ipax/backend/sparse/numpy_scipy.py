# Copyright 2026 Niklas Wahl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SciPy sparse adapter (ADAPTER — concrete-library import allowed here).

This is the CPU-side sparse-direct wrapper for the NumPy/SciPy backend. The
core emits structure as Array-API integer/value
vectors ``(row, col, value)``; this module is the **only** place under
``backend/`` that converts them into a concrete ``scipy.sparse`` matrix and
factors it (invariants #1/#4). The IPM never sees a ``scipy.sparse`` object — it
only ever holds the :class:`SparseOperator` wrapper and the returned Array-API
arrays.

Two factorization routes are exposed through :class:`SciPySparseAdapter.solver`:

* **Symmetric CPU default:** ``feral-solver`` (imported as ``feral``) factors
  symmetric KKT-shaped systems with a sparse LDLᵀ factorization and certified
  inertia counts. It is the preferred path when installed and the assembled
  matrix is symmetric.
* **General fallback:** ``scipy.sparse.linalg.splu`` remains available for any
  square nonsingular sparse matrix, including non-symmetric systems and
  environments without Feral. If inertia is requested on the fallback path, the
  adapter asks Feral/MUMPS first and only densifies at small scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Allowed import boundary (invariants #1, #4): backend/sparse/ adapters only.
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from ipax.backend.operators import LinearOperator
from ipax.backend.sparse._routing import to_xp_array
from ipax.linalg.solver import LinearSolveError

if TYPE_CHECKING:
    from ipax.typing import Array, Namespace

# Above this size we refuse to densify for the inertia fallback; a real sparse
# inertia provider (MUMPS/PARDISO) is required instead.
_DENSE_INERTIA_MAX = 1000


def _to_numpy(arr: Array) -> np.ndarray:
    """Bring an Array-API array onto the host as a NumPy ndarray.

    NumPy/``array_api_compat.numpy`` arrays pass through; other CPU backends
    (e.g. Torch) cross the boundary zero-copy via DLPack where possible.
    """
    if isinstance(arr, np.ndarray):
        return arr
    try:
        return np.from_dlpack(arr)
    except (TypeError, RuntimeError, BufferError, ValueError):
        return np.asarray(arr)


def _to_index(arr: Array) -> np.ndarray:
    """Host integer-index vector for COO assembly."""
    return _to_numpy(arr).astype(np.int64, copy=False)


def _import_feral() -> Any:
    """Import the optional Feral binding lazily at the sparse adapter boundary."""
    try:
        import feral
    except ImportError as exc:
        raise ImportError(
            "feral-solver is not installed; install `ipax[sparse-cpu]` to enable "
            "the CPU sparse LDL^T solver"
        ) from exc
    return feral


def _is_symmetric(matrix: scipy.sparse.csc_matrix) -> bool:
    """Return whether ``matrix`` is numerically symmetric."""
    if matrix.shape[0] != matrix.shape[1]:
        return False
    diff = (matrix - matrix.T).tocoo()
    diff.eliminate_zeros()
    if diff.nnz == 0:
        return True
    matrix_scale = float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0
    diff_scale = float(np.max(np.abs(diff.data)))
    n = int(matrix.shape[0])
    tol = max(1.0, matrix_scale) * n * float(np.finfo(np.float64).eps)
    return diff_scale <= tol


def _inertia_tuple(inertia: Any) -> tuple[int, int, int]:
    """Normalize Feral's ``Inertia`` value object to a plain tuple."""
    return int(inertia.n_pos), int(inertia.n_neg), int(inertia.n_zero)


def _feral_factor_succeeded(feral: Any, status: object) -> bool:
    """Return whether a Feral factor status represents success."""
    return bool(status == feral.FactorStatus.SUCCESS)


def _feral_status_name(feral: Any, status: object) -> str:
    """Human-readable Feral factor status for error messages."""
    try:
        return str(feral.FactorStatus(status).name)
    except Exception:
        return str(status)


class SparseOperator(LinearOperator):
    """``LinearOperator`` backed by a ``scipy.sparse`` matrix.

    ``matvec``/``rmatvec`` accept and return arrays in the *original* namespace
    ``xp`` (the matrix lives on the host), so the operator is a drop-in
    replacement for any other :class:`LinearOperator` in the IPM.

    The assembled matrix is kept in its given (COO) form. The sparse-direct route
    consumes the **CSC** form (:attr:`csc_matrix`) and never calls ``matvec``, so
    the CSR matvec format is materialized lazily only when a matvec-family method
    is actually used — both forms are cached, and a fresh operator is built each
    IPM iteration, so the cache lifetime is naturally one factorization.
    """

    def __init__(
        self,
        matrix: scipy.sparse.spmatrix,
        xp: Namespace,
        *,
        symmetric: bool | None = None,
    ) -> None:
        self._matrix = matrix
        self._xp = xp
        # Structural symmetry hint from the assembler (None ⇒ test numerically).
        self._symmetric_hint = symmetric
        self._csr: scipy.sparse.csr_matrix | None = None
        self._csc: scipy.sparse.csc_matrix | None = None
        self._symmetric: bool | None = None

    @property
    def _csr_matrix(self) -> scipy.sparse.csr_matrix:
        """The CSR (matvec) form, built and cached on first use."""
        if self._csr is None:
            self._csr = self._matrix.tocsr()
        return self._csr

    @property
    def csc_matrix(self) -> scipy.sparse.csc_matrix:
        """The CSC (factorization) form, built and cached on first use."""
        if self._csc is None:
            self._csc = self._matrix.tocsc()
        return self._csc

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self._matrix.shape
        return int(rows), int(cols)

    @property
    def scipy_matrix(self) -> scipy.sparse.spmatrix:
        """The wrapped host matrix in CSR (matvec) form."""
        return self._csr_matrix

    def is_symmetric(self) -> bool:
        """Whether ``A == Aᵀ``: honor the assembler's hint, else test numerically.

        The structural hint (set by the KKT condensed/saddle assembler, which
        mirrors one value array into ``C``/``Cᵀ``) is authoritative when present,
        so the sparse-direct route skips the O(nnz) numerical test every iteration.
        """
        if self._symmetric_hint is not None:
            return self._symmetric_hint
        if self._symmetric is None:
            self._symmetric = _is_symmetric(self.csc_matrix)
        return self._symmetric

    def matvec(self, v: Array) -> Array:
        out = self._csr_matrix @ _to_numpy(v)
        return to_xp_array(out, self._xp)

    def rmatvec(self, v: Array) -> Array:
        out = self._csr_matrix.T @ _to_numpy(v)
        return to_xp_array(out, self._xp)

    def matmat(self, V: Array) -> Array:
        out = self._csr_matrix @ _to_numpy(V)
        return to_xp_array(out, self._xp)

    def diagonal(self, like: Array | None = None) -> Array:
        del like
        return to_xp_array(self._csr_matrix.diagonal(), self._xp)

    def row_inf_norms(self, like: Array | None = None) -> Array:
        del like
        m, n = self.shape
        if n == 0:
            return to_xp_array(np.zeros((m,), dtype=self._matrix.dtype), self._xp)
        norms = abs(self._csr_matrix).max(axis=1).toarray().reshape((m,))
        return to_xp_array(norms, self._xp)

    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        del like
        coo = self._csr_matrix.tocoo()  # canonical (duplicates summed)
        return (
            to_xp_array(coo.row, self._xp),
            to_xp_array(coo.col, self._xp),
            to_xp_array(coo.data, self._xp),
            self.shape,
        )


def _require_csc(K: LinearOperator) -> tuple[scipy.sparse.csc_matrix, Namespace]:
    """Extract the host CSC matrix + namespace from a :class:`SparseOperator`.

    The sparse-direct route only factors operators that carry real sparse
    structure. A generic matrix-free operator has no triplets to assemble, so we
    reject it with a clear message rather than silently densifying. The CSC form
    is cached on the operator, so repeated calls within one factorization (router
    → dispatched inner solver) convert at most once.
    """
    if not isinstance(K, SparseOperator):
        raise TypeError(
            "SciPy sparse solver requires a SparseOperator built from COO "
            f"triplets; got {type(K).__name__}"
        )
    matrix = K.csc_matrix
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("sparse solver requires a square operator")
    return matrix, K._xp


def _dense_inertia(matrix: scipy.sparse.csc_matrix) -> tuple[int, int, int]:
    """Inertia of a symmetric matrix via a dense symmetric eigensolve.

    Sylvester's law of inertia: the signs of the eigenvalues give ``(n₊, n₋,
    n₀)``. Used only as a small-scale fallback when no sparse inertia-revealing
    factorization (MUMPS/PARDISO) is available.
    """
    a = np.asarray(matrix.toarray(), dtype=np.float64)
    a = 0.5 * (a + a.T)  # symmetrize defensively
    n = a.shape[0]
    eig = np.linalg.eigvalsh(a)
    scale = float(np.max(np.abs(eig))) if n else 0.0
    tol = max(1.0, scale) * n * float(np.finfo(np.float64).eps)
    pos = int(np.count_nonzero(eig > tol))
    neg = int(np.count_nonzero(eig < -tol))
    return pos, neg, n - pos - neg


def _feral_inertia(matrix: scipy.sparse.csc_matrix) -> tuple[int, int, int] | None:
    """Inertia from Feral's sparse LDL^T factorization, or ``None``."""
    try:
        feral = _import_feral()
    except ImportError:
        return None

    try:
        feral_matrix = feral.from_scipy(matrix, symmetric="full")
        status, inertia = feral.Solver().factor(feral_matrix)
    except Exception:
        return None

    if not _feral_factor_succeeded(feral, status) or inertia is None:
        return None
    return _inertia_tuple(inertia)


def _mumps_inertia(matrix: scipy.sparse.csc_matrix) -> tuple[int, int, int] | None:
    """Inertia from MUMPS' symmetric factorization, or ``None`` if unavailable.

    MUMPS reports the number of negative pivots in ``INFOG(12)`` and null pivots
    in ``INFOG(28)`` after a symmetric (``SYM=2``) factorization; the positive
    count is the remainder. Requires the ``PyMUMPS`` binding (``mumps`` →
    ``DMumpsContext``). Any import/runtime/API mismatch returns ``None`` so the
    caller transparently falls back to the dense computation — MUMPS is a
    performance accelerator here, never a correctness dependency.

    .. note::
       This path is gated behind an optional native dependency that has no
       Windows wheels; it is exercised only where MUMPS is installed.
    """
    try:  # optional dependency, isolated import
        from mumps import DMumpsContext
    except ImportError:
        return None

    n = int(matrix.shape[0])
    ctx = None
    try:
        # SYM=2: general symmetric; MUMPS wants the lower triangle, 1-based.
        lower = scipy.sparse.tril(matrix).tocoo()
        ctx = DMumpsContext(sym=2)
        ctx.set_shape(n)
        ctx.set_centralized_assembled(
            lower.row.astype(np.int32) + 1,
            lower.col.astype(np.int32) + 1,
            lower.data.astype(np.float64),
        )
        ctx.run(job=4)  # analyze + factor (no solve needed for inertia)
        infog = ctx.id.infog
        neg = int(infog[12 - 1])
        zero = int(infog[28 - 1])
    except Exception:  # any binding/API mismatch ⇒ transparent dense fallback
        return None
    finally:
        if ctx is not None:
            try:
                ctx.destroy()
            except Exception:  # best-effort cleanup
                pass

    if neg < 0 or zero < 0 or neg + zero > n:
        return None  # implausible ⇒ distrust and fall back
    return n - neg - zero, neg, zero


def _symmetric_inertia(matrix: scipy.sparse.csc_matrix) -> tuple[int, int, int]:
    """Inertia ``(n₊, n₋, n₀)`` of a symmetric matrix (Feral, MUMPS, dense)."""
    if not _is_symmetric(matrix):
        raise ValueError("inertia is defined only for symmetric matrices")
    inertia = _feral_inertia(matrix)
    if inertia is not None:
        return inertia
    inertia = _mumps_inertia(matrix)
    if inertia is not None:
        return inertia
    if int(matrix.shape[0]) > _DENSE_INERTIA_MAX:
        raise NotImplementedError(
            "sparse LDLᵀ inertia requires feral-solver or a MUMPS/PARDISO "
            "binding; the dense "
            f"fallback is capped at n={_DENSE_INERTIA_MAX}"
        )
    return _dense_inertia(matrix)


class FeralSparseSolver:
    """Sparse symmetric-indefinite ``LinearSolver`` backed by Feral LDL^T.

    Feral is a CPU sparse adapter dependency, not a core dependency: it is
    imported lazily here and consumes the already materialized SciPy CSC matrix.
    It solves only symmetric systems; the CPU default wrapper falls back to
    SuperLU for general sparse matrices.
    """

    def __init__(self, *, require_inertia: bool = False) -> None:
        self._require_inertia = require_inertia
        self._feral: Any = None
        self._feral_matrix: Any = None
        self._solver: Any = None
        self._xp: Namespace | None = None
        self._inertia: tuple[int, int, int] | None = None

    def describe(self) -> str:
        return "Feral LDL^T (CPU)"

    def factor(self, K: LinearOperator) -> None:
        matrix, xp = _require_csc(K)
        assert isinstance(K, SparseOperator)  # narrowed by _require_csc
        if not K.is_symmetric():
            raise ValueError("Feral sparse solver requires a symmetric operator")

        feral = _import_feral()
        errors = (RuntimeError, feral.FeralError)
        try:
            feral_matrix = feral.from_scipy(matrix, symmetric="full")
            # Reuse a persistent ``Solver`` so its symbolic analysis (fill-reducing
            # ordering + elimination tree) is cached across IPM iterations. Feral's
            # ``factor`` reuses it when the sparsity pattern is unchanged — the
            # common case: the KKT pattern is fixed in an interior-point method, and
            # only diagonal values (Σ_x, Σ_s, δ_w escalation) and Jacobian entries
            # move — and re-analyzes automatically when it changes (e.g. the L-BFGS
            # border growing during warm-up). Only the numeric factorization is
            # redone each step, as in IPOPT's structure/values split.
            if self._solver is None:
                self._solver = feral.Solver()
            status, inertia = self._solver.factor(feral_matrix)
        except errors as exc:
            raise LinearSolveError("feral LDL^T factorization failed") from exc

        if not _feral_factor_succeeded(feral, status):
            status_name = _feral_status_name(feral, status)
            raise LinearSolveError(
                f"feral LDL^T factorization failed with status {status_name}"
            )
        if self._require_inertia and inertia is None:
            raise LinearSolveError("feral LDL^T factorization did not report inertia")

        self._feral = feral
        self._feral_matrix = feral_matrix
        self._xp = xp
        # Feral returns the LDLᵀ inertia for free on every symmetric factor, so
        # keep it whenever available (not only under ``require_inertia``): the IPM
        # uses it best-effort for inertia-guided δ_w correction.
        self._inertia = _inertia_tuple(inertia) if inertia is not None else None

    def solve(self, rhs: Array) -> Array:
        if self._solver is None or self._xp is None:
            raise RuntimeError("factor() must be called before solve()")
        b = np.ascontiguousarray(_to_numpy(rhs), dtype=np.float64)
        errors = (RuntimeError, self._feral.FeralError)
        try:
            if bool(getattr(self._solver, "needs_refinement", False)):
                x = self._solver.solve_refined(self._feral_matrix, b)
            else:
                x = self._solver.solve(b)
        except errors as exc:
            raise LinearSolveError("feral LDL^T solve failed") from exc
        return to_xp_array(x, self._xp)

    @property
    def inertia(self) -> tuple[int, int, int]:
        """The factored operator's inertia ``(n₊, n₋, n₀)``."""
        if self._inertia is None:
            raise RuntimeError(
                "inertia is unavailable: create the solver with "
                "require_inertia=True and call factor() first"
            )
        return self._inertia

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """Best-effort inertia (free from the LDLᵀ factor); ``None`` if absent."""
        return self._inertia


class SuperLUSparseSolver:
    """General sparse ``LinearSolver`` backed by SciPy SuperLU.

    This is the compatibility fallback for non-symmetric sparse systems and for
    environments where the optional Feral binding is unavailable.
    """

    def __init__(self, *, require_inertia: bool = False) -> None:
        self._require_inertia = require_inertia
        self._lu: scipy.sparse.linalg.SuperLU | None = None
        self._xp: Namespace | None = None
        self._inertia: tuple[int, int, int] | None = None

    def describe(self) -> str:
        return "SciPy SuperLU (CPU)"

    def factor(self, K: LinearOperator) -> None:
        matrix, xp = _require_csc(K)
        self._xp = xp
        try:
            self._lu = scipy.sparse.linalg.splu(matrix)
        except RuntimeError as exc:  # singular factor ⇒ numerical failure
            raise LinearSolveError("sparse LU factorization failed") from exc
        if self._require_inertia:
            self._inertia = _symmetric_inertia(matrix)

    def solve(self, rhs: Array) -> Array:
        if self._lu is None or self._xp is None:
            raise RuntimeError("factor() must be called before solve()")
        b = _to_numpy(rhs)
        try:
            x = self._lu.solve(b)
        except RuntimeError as exc:
            raise LinearSolveError("sparse solve failed") from exc
        return to_xp_array(x, self._xp)

    @property
    def inertia(self) -> tuple[int, int, int]:
        """The factored operator's inertia ``(n₊, n₋, n₀)``."""
        if self._inertia is None:
            raise RuntimeError(
                "inertia is unavailable: create the solver with "
                "require_inertia=True and call factor() first"
            )
        return self._inertia

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """SuperLU is not inertia-revealing; only set under ``require_inertia``."""
        return self._inertia


class SciPySparseSolver:
    """CPU sparse-direct solver preferring Feral and falling back to SuperLU."""

    def __init__(
        self, *, require_inertia: bool = False, prefer_feral: bool = True
    ) -> None:
        self._require_inertia = require_inertia
        self._prefer_feral = prefer_feral
        self._inner: FeralSparseSolver | SuperLUSparseSolver | None = None
        self._feral_unavailable = False

    def describe(self) -> str:
        return self._inner.describe() if self._inner is not None else "SciPy (CPU)"

    def factor(self, K: LinearOperator) -> None:
        _require_csc(K)  # validate: SparseOperator + square (CSC cached for reuse)
        assert isinstance(K, SparseOperator)  # narrowed by _require_csc
        # Reuse the chosen inner solver across factorizations so its symbolic
        # cache survives — the route (Feral for symmetric, SuperLU otherwise) is
        # stable across IPM iterations, so this rebuilds only on the rare flip.
        if self._prefer_feral and not self._feral_unavailable and K.is_symmetric():
            if not isinstance(self._inner, FeralSparseSolver):
                self._inner = FeralSparseSolver(require_inertia=self._require_inertia)
            try:
                self._inner.factor(K)
            except ImportError:
                # Feral binding genuinely missing: remember and use SuperLU.
                self._feral_unavailable = True
                self._inner = None
            else:
                return

        if not isinstance(self._inner, SuperLUSparseSolver):
            self._inner = SuperLUSparseSolver(require_inertia=self._require_inertia)
        self._inner.factor(K)

    def solve(self, rhs: Array) -> Array:
        if self._inner is None:
            raise RuntimeError("factor() must be called before solve()")
        return self._inner.solve(rhs)

    @property
    def inertia(self) -> tuple[int, int, int]:
        """The factored operator's inertia ``(n₊, n₋, n₀)``."""
        if self._inner is None:
            raise RuntimeError("factor() must be called before reading inertia")
        return self._inner.inertia

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """Best-effort inertia of the dispatched inner solver; ``None`` if absent."""
        if self._inner is None:
            return None
        return self._inner.inertia_or_none()


class SciPySparseAdapter:
    """Factory pairing COO assembly with the SciPy sparse-direct solver."""

    def from_coo(
        self,
        rows: Array,
        cols: Array,
        values: Array,
        *,
        shape: tuple[int, int],
        symmetric: bool | None = None,
    ) -> SparseOperator:
        """Build a :class:`SparseOperator` from Array-API COO triplets.

        ``symmetric`` is an optional structural hint from the assembler; ``None``
        leaves the operator to test symmetry numerically when first asked.
        """
        from ipax.backend.namespace import array_namespace

        xp = array_namespace(values)
        matrix = scipy.sparse.coo_matrix(
            (_to_numpy(values), (_to_index(rows), _to_index(cols))),
            shape=shape,
        )
        return SparseOperator(matrix, xp, symmetric=symmetric)

    def solver(self, *, require_inertia: bool = False) -> SciPySparseSolver:
        """Return the CPU sparse-direct solver (Feral default, SuperLU fallback)."""
        return SciPySparseSolver(require_inertia=require_inertia)


__all__ = [
    "FeralSparseSolver",
    "SciPySparseAdapter",
    "SciPySparseSolver",
    "SparseOperator",
    "SuperLUSparseSolver",
]
