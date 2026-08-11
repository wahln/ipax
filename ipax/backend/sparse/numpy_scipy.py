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
from scipy.linalg.blas import get_blas_funcs

from ipax.backend.operators import LinearOperator
from ipax.backend.sparse._canonical import CompiledCompressed, compile_compressed
from ipax.backend.sparse._routing import to_xp_array
from ipax.linalg.solver import LinearSolveError

if TYPE_CHECKING:
    from ipax.typing import Array, Namespace

# Above this size we refuse to densify for the inertia fallback; a real sparse
# inertia provider (MUMPS/PARDISO) is required instead.
_DENSE_INERTIA_MAX = 1000

# Density at or above which the weighted Gram Aᵀ diag(w) A is accumulated by
# chunked dense GEMM instead of SpGEMM. Flop model: the dense path does m·n²
# MACs at BLAS speed (O(100) GFLOP/s multithreaded), while scipy's SpGEMM does
# ~2·Σ nnz_row² hash-probe ops at O(1) Gop/s *plus* a symbolic pass of the same
# order every call — so dense wins once the mean row density exceeds a few
# percent. Measured on TROTS Prostate_CK (28% dense, n=2266, m=8.9e4): SpGEMM
# 90 s vs chunked GEMM 3.4 s per Gram. The Gram result is a dense n×n either
# way, so nothing is lost structurally.
_GRAM_DENSE_MIN_DENSITY = 0.05
# Row-chunk cap (elements) for the dense-GEMM path: two m_chunk×n float64
# buffers ≈ 800 MB at this cap, independent of m.
_GRAM_DENSE_CHUNK_ELEMENTS = 50_000_000

# Sampling caps for the Gram fill-in estimate (``gram_fill_estimate``): the
# per-column union size is exact, so a modest evenly spaced column sample
# already tracks structured (banded/block) patterns closely, and the per-column
# row cap only truncates columns whose Gram fill is far past any sparse-route
# threshold anyway (a few hundred scattered rows saturate the union).
_GRAM_FILL_SAMPLE_COLUMNS = 128
_GRAM_FILL_SAMPLE_ROWS = 512


def _estimate_gram_fill(
    csr_indptr: np.ndarray,
    csr_indices: np.ndarray,
    csc_indptr: np.ndarray,
    csc_indices: np.ndarray,
    n_cols: int,
) -> float:
    """Estimated density of the Gram pattern ``nnz(AᵀA)/n²`` in ``[0, 1]``.

    Exact-per-sample column overlap: Gram entry ``(j, k)`` is structurally
    nonzero iff columns ``j`` and ``k`` share a row, so the pattern of Gram
    column ``j`` is exactly the union of the CSR column patterns of the rows
    listed in CSC column ``j``. Averaging that union size over an evenly
    spaced column sample estimates the mean Gram-column fill at
    O(sample · nnz-per-column) cost — no SpGEMM, no Gram allocation. This is
    the cheap probe the sparse normal-equations route selection needs
    (scattered rows of low nnz-density still saturate the union, which is
    precisely what the estimate must reveal). Measured 2026-07 on banded and
    scattered tall matrices (n=10k–20k, m=10n, 20 nnz/row): within 1%
    relative of the exact ``nnz(AᵀA)/n²`` at 25–60 ms per probe.
    """
    if n_cols == 0:
        return 0.0
    sample = min(n_cols, _GRAM_FILL_SAMPLE_COLUMNS)
    columns = np.unique(np.linspace(0, n_cols - 1, sample).astype(np.int64))
    total = 0
    for j in columns:
        rows = csc_indices[csc_indptr[j] : csc_indptr[j + 1]]
        if rows.shape[0] == 0:
            continue
        if rows.shape[0] > _GRAM_FILL_SAMPLE_ROWS:
            step = np.linspace(0, rows.shape[0] - 1, _GRAM_FILL_SAMPLE_ROWS)
            rows = rows[np.unique(step.astype(np.int64))]
        union = np.unique(
            np.concatenate(
                [csr_indices[csr_indptr[i] : csr_indptr[i + 1]] for i in rows]
            )
        )
        total += int(union.shape[0])
    return total / (int(columns.shape[0]) * n_cols)


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


def _scatter_add(out: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    """Unbuffered scatter-add (sums duplicate targets) for the canonical map."""
    np.add.at(out, idx, vals)


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
        pattern_signature: object | None = None,
        csc: scipy.sparse.csc_matrix | None = None,
    ) -> None:
        self._matrix = matrix if csc is None else csc
        self._xp = xp
        # Structural symmetry hint from the assembler (None ⇒ test numerically).
        self._symmetric_hint = symmetric
        self._pattern_signature = pattern_signature
        self._csr: scipy.sparse.csr_matrix | None = None
        # A pre-canonicalized CSC from the compiled COO map (values-only refactor
        # fast path) is the factorization form directly — no per-iteration tocsc.
        self._csc: scipy.sparse.csc_matrix | None = csc
        self._symmetric: bool | None = None
        # Gram-path caches (the condensed n ≪ m route calls ``gram(Σ_s)`` on
        # every KKT factor — per IPM iteration *and* per δ_w retry / SOC /
        # Mehrotra re-solve with bit-identical weights, so an O(m) value compare
        # amortizes the Σ nnz_row² SpGEMM):
        self._gram_transpose: scipy.sparse.csr_matrix | None = None  # Aᵀ as CSR
        self._gram_scaled: scipy.sparse.csr_matrix | None = None  # diag(w)A buffer
        self._gram_row_index: np.ndarray | None = None  # nnz → row map for w
        # One memo slot per accumulate_dtype: the mixed-precision dense route
        # legitimately alternates reduced and exact requests with identical
        # weights (e.g. a mixed PD failure re-materializing exactly inside a
        # δ_w retry), and a single slot would evict on every alternation —
        # exactly where the memo is meant to pay off. At most 2 keys × n².
        self._gram_memo: dict[str | None, tuple[np.ndarray, np.ndarray]] = {}
        self._gram_compute_count = 0  # observability/testing: actual SpGEMM runs
        self._reduced_csr: scipy.sparse.csr_matrix | None = None  # cast-once data
        self._native_streak = False  # last gram() request was native (see gram)
        self._squared: scipy.sparse.csr_matrix | None = None  # A∘A (Gram diagonals)
        # Sparse-Gram (COO) memo for the sparse normal-equations route.
        self._gram_coo_weights: np.ndarray | None = None
        self._gram_coo_value: scipy.sparse.coo_matrix | None = None

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

    def coo_pattern_signature(self) -> object | None:
        return self._pattern_signature

    def matvec(self, v: Array) -> Array:
        out = self._csr_matrix @ _to_numpy(v)
        return to_xp_array(out, self._xp)

    def rmatvec(self, v: Array) -> Array:
        out = self._csr_matrix.T @ _to_numpy(v)
        return to_xp_array(out, self._xp)

    def matmat(self, V: Array) -> Array:
        out = self._csr_matrix @ _to_numpy(V)
        return to_xp_array(out, self._xp)

    def rmatmat(self, V: Array) -> Array:
        out = self._csr_matrix.T @ _to_numpy(V)
        return to_xp_array(out, self._xp)

    def diagonal(self, like: Array | None = None) -> Array:
        del like
        return to_xp_array(self._csr_matrix.diagonal(), self._xp)

    @property
    def _squared_csr(self) -> scipy.sparse.csr_matrix:
        """The elementwise square ``A∘A``, built once (shared by both Gram
        diagonals, which are recomputed every IPM iteration)."""
        if self._squared is None:
            self._squared = self._csr_matrix.multiply(self._csr_matrix).tocsr()
        return self._squared

    def gram_diagonal(self, weights: Array) -> Array:
        # diag(Aᵀ diag(w) A)_k = Σ_i w_i A_ik² = (A∘A)ᵀ w (column energies).
        out = self._squared_csr.T @ _to_numpy(weights)
        return to_xp_array(np.asarray(out).reshape(-1), self._xp)

    def gram(
        self,
        weights: Array,
        *,
        accumulate_dtype: str | None = None,
        hinted_only: bool = False,
    ) -> Array:
        if (
            hinted_only
            and accumulate_dtype is not None
            and self.gram_accumulate_dtype_hint() != accumulate_dtype
        ):
            accumulate_dtype = None  # this matrix's data does not support it
        # Aᵀ diag(w) A as a dense n×n matrix, formed by sparse arithmetic: scale
        # rows by w (still sparse) then a sparse Aᵀ·(diag(w)A) product, densifying
        # only the small n×n result — never the m×n matrix A itself (the point at
        # RT scale, where m ≫ n).
        #
        # The wrapped matrix is fixed for this operator's lifetime, so two levels
        # of caching amortize the per-iteration cost (the SpGEMM arithmetic floor
        # is Σ_i nnz_row_i²): a last-weights memo — δ_w retries, SOC and Mehrotra
        # re-solves within one IPM iteration re-request the *same* Σ_s — and,
        # on a miss, a cached Aᵀ CSR + same-pattern ``diag(w)A`` buffer so only
        # value work (no CSC→CSR conversion, no ``multiply`` allocation) remains
        # besides the product itself. Callers must treat the returned array as
        # read-only; the condensed route only ever adds it out-of-place.
        w = _to_numpy(weights).reshape(-1)
        if accumulate_dtype is None:
            # Release the reduced-data copy only on the SECOND consecutive
            # native request. A single one does *not* mean the mixed route is
            # done: ``DenseSolver`` answers a failed refinement from an exact
            # rebuild but re-engages mixed on the next factorization, until
            # ``refine_failure_limit`` *consecutive* failures disable it. Freeing
            # on the first would force a full nnz recast on every intermittent
            # hard iteration — the cast-once optimization defeated exactly where
            # it is most needed. Two in a row means no consumer came back.
            if self._native_streak:
                self._reduced_csr = None
            self._native_streak = True
        else:
            self._native_streak = False
        memo = self._gram_memo.get(accumulate_dtype)
        if memo is not None and np.array_equal(w, memo[0]):
            return to_xp_array(memo[1], self._xp)

        a = self._csr_matrix
        m, n = a.shape
        size = int(m) * int(n)
        if size > 0 and a.nnz / size >= _GRAM_DENSE_MIN_DENSITY:
            # Dense-ish rows (e.g. RT dose matrices): accumulate by chunked
            # dense GEMM — SpGEMM's Σ nnz_row² hash arithmetic (plus its
            # per-call symbolic pass) is the wrong algorithm here (see
            # _GRAM_DENSE_MIN_DENSITY for the model and measurements).
            if accumulate_dtype is not None:
                # Mixed-precision accumulate (``DenseOptions.gram_dtype``):
                # run the chunked accumulation in the reduced dtype through a
                # cached same-structure CSR (data cast once, indices shared),
                # then upcast only the small n×n result. Only this dense
                # strategy honors the request — the SpGEMM branch below is
                # memory-bound, so reduced arithmetic buys little there.
                acc = np.dtype(accumulate_dtype)
                gram = self._gram_dense_accumulate(
                    self._reduced_data_csr(acc), w.astype(acc, copy=False)
                )
                gram = gram.astype(np.result_type(a.dtype, w.dtype), copy=False)
            else:
                gram = self._gram_dense_accumulate(a, w)
        else:
            if self._gram_transpose is None:
                # One-time symbolic work: the n×m transpose CSR (scipy would
                # otherwise re-derive it inside every ``Aᵀ @ ·`` product), the
                # scaled same-pattern buffer, and the nnz→row map expanding w
                # to A's data.
                self._gram_transpose = a.T.tocsr()
                self._gram_scaled = a.copy()
                self._gram_row_index = np.repeat(
                    np.arange(a.shape[0]), np.diff(a.indptr)
                )
            scaled = self._gram_scaled
            assert scaled is not None and self._gram_row_index is not None
            scaled.data = a.data * w[self._gram_row_index]
            gram = np.asarray((self._gram_transpose @ scaled).toarray())
        self._gram_memo[accumulate_dtype] = (np.array(w, copy=True), gram)
        self._gram_compute_count += 1
        return to_xp_array(gram, self._xp)

    def _reduced_data_csr(self, dtype: np.dtype) -> scipy.sparse.csr_matrix:
        """Same-structure CSR with the data cast once to ``dtype``.

        Index arrays are shared with the wrapped matrix (no copy); only the
        value array is duplicated (e.g. +4 bytes/nnz for float32) — so every
        later reduced-precision accumulate reads half the bytes instead of
        paying a per-chunk cast."""
        if self._reduced_csr is None or self._reduced_csr.dtype != dtype:
            a = self._csr_matrix
            self._reduced_csr = scipy.sparse.csr_matrix(
                (a.data.astype(dtype), a.indices, a.indptr), shape=a.shape
            )
        return self._reduced_csr

    def gram_coo(self, weights: Array) -> tuple[Array, Array, Array, tuple[int, int]]:
        # ``Aᵀ diag(w) A`` kept SPARSE: the sparse normal-equations route
        # factors the n×n condensed matrix directly, so the product is never
        # densified. Reuses the cached transpose/scaled-buffer machinery of
        # :meth:`gram`. Pattern stability across weight changes holds because
        # SciPy's SpGEMM structure depends only on the operand patterns (no
        # numerical pruning) and ``tocoo`` of the canonical CSR product is
        # deterministic — callers may cache the (rows, cols) and reuse
        # symbolic factorizations.
        w = _to_numpy(weights).reshape(-1)
        if (
            self._gram_coo_value is not None
            and self._gram_coo_weights is not None
            and np.array_equal(w, self._gram_coo_weights)
        ):
            product = self._gram_coo_value
        else:
            a = self._csr_matrix
            if self._gram_transpose is None:
                self._gram_transpose = a.T.tocsr()
                self._gram_scaled = a.copy()
                self._gram_row_index = np.repeat(
                    np.arange(a.shape[0]), np.diff(a.indptr)
                )
            scaled = self._gram_scaled
            assert scaled is not None and self._gram_row_index is not None
            scaled.data = a.data * w[self._gram_row_index]
            product = (self._gram_transpose @ scaled).tocoo()
            self._gram_coo_weights = np.array(w, copy=True)
            self._gram_coo_value = product
            self._gram_compute_count += 1
        n = int(self.shape[1])
        return (
            to_xp_array(np.asarray(product.row), self._xp),
            to_xp_array(np.asarray(product.col), self._xp),
            to_xp_array(np.asarray(product.data), self._xp),
            (n, n),
        )

    def gram_coo_capable(self) -> bool:
        return True

    def gram_accumulate_dtype_hint(self) -> str | None:
        # Storage-level metadata only (see the LinearOperator docstring):
        # float32-stored data may be accumulated reduced without losing data
        # information.
        return "float32" if self._matrix.dtype == np.float32 else None

    def gram_fill_estimate(self) -> float | None:
        """Estimated Gram-pattern density (sampled column overlap; see
        :func:`_estimate_gram_fill`). One-time selection probe, never on the
        per-iteration path."""
        m, n = self.shape
        if m == 0 or n == 0:
            return 0.0
        csr = self._csr_matrix
        csc = self.csc_matrix
        return _estimate_gram_fill(csr.indptr, csr.indices, csc.indptr, csc.indices, n)

    @staticmethod
    def _gram_dense_accumulate(a: scipy.sparse.csr_matrix, w: np.ndarray) -> np.ndarray:
        """``Aᵀ diag(w) A`` by BLAS over zero-copy row windows of ``a``.

        Each chunk densifies ``m_chunk × n`` rows (bounded by
        ``_GRAM_DENSE_CHUNK_ELEMENTS``) — O(m·n²) FLOPs total at dense-BLAS
        speed, peak extra memory two chunk buffers regardless of ``m``.

        For finite nonnegative real weights (the IPM's Σ weights are positive
        by construction) the chunk is scaled in place by ``√w`` and
        accumulated with the *symmetric* rank-k update (``syrk``): one
        triangle, so half the GEMM FLOPs, and no ``w ∘ block`` temporary. The
        triangle is mirrored by pure copies at the end, making the result
        bitwise symmetric as a side effect. Mixed-sign, non-finite, or
        non-float weights fall back to the general
        ``blockᵀ @ (w_chunk ∘ block)`` accumulation.

        Numerical caveat: the two forms differ in their overflow/underflow
        *envelope*, not just rounding — syrk multiplies ``fl(√w·a)·fl(√w·b)``
        where GEMM computes ``a·fl(w·b)``. The symmetric split *halves the
        dynamic range* of the intermediates (favorable for the common case),
        but for extreme ``w`` paired with strongly asymmetric ``|a|, |b|`` it
        can saturate where the asymmetric grouping does not; relevant mostly
        for float32 data, whose exponent headroom is ~1e38.
        """
        m, n = a.shape
        dtype = np.result_type(a.dtype, w.dtype)
        chunk = max(1, _GRAM_DENSE_CHUNK_ELEMENTS // max(int(n), 1))
        use_syrk = dtype in (np.float32, np.float64) and bool(
            np.all((w >= 0.0) & np.isfinite(w))
        )
        if use_syrk:
            syrk = get_blas_funcs("syrk", dtype=dtype)
            sqrt_w = np.sqrt(w.astype(dtype, copy=False))
            # syrk accumulates in place into an F-ordered target.
            out = np.zeros((n, n), dtype=dtype, order="F")
        else:
            out = np.zeros((n, n), dtype=dtype)
        for start in range(0, m, chunk):
            end = min(m, start + chunk)
            lo, hi = int(a.indptr[start]), int(a.indptr[end])
            window = scipy.sparse.csr_matrix(
                (a.data[lo:hi], a.indices[lo:hi], a.indptr[start : end + 1] - lo),
                shape=(end - start, n),
            )
            block = np.asarray(window.toarray(), dtype=dtype)
            if use_syrk:
                block *= sqrt_w[start:end, None]  # in place — block is fresh
                # ``block.T`` is an F-contiguous (n, m_chunk) view (no copy);
                # syrk adds block.T @ block into ``out``'s lower triangle.
                out = syrk(
                    1.0, block.T, beta=1.0, c=out, trans=0, lower=1, overwrite_c=1
                )
            else:
                out += block.T @ (w[start:end, None] * block)
        if use_syrk:
            # Mirror the computed lower triangle by row-slice copies — no
            # arithmetic (bitwise-exact incl. -0.0, unlike `out += triu(out.T)`)
            # and no n²-sized index temporaries (a `triu_indices` gather peaks
            # at ~2.5× the output size, which can OOM at the top of the dense
            # route's range).
            out = np.ascontiguousarray(out)
            for j in range(n):
                out[j, j + 1 :] = out[j + 1 :, j]
        return out

    def row_gram_diagonal(self, weights: Array) -> Array:
        # diag(A diag(w) Aᵀ)_j = Σ_k w_k A_jk² = (A∘A) w (row energies).
        out = self._squared_csr @ _to_numpy(weights)
        return to_xp_array(np.asarray(out).reshape(-1), self._xp)

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


def _reject_non_finite(matrix: scipy.sparse.csc_matrix) -> None:
    """Raise :class:`LinearSolveError` if any stored value is inf/NaN.

    A non-finite KKT entry means an upstream Hessian/Jacobian/Σ evaluated at a
    bad trial iterate overflowed (e.g. a ``1/0`` in the problem's element
    functions). The dense route already treats such a factorization as a
    numerical failure the IPM recovers from (δ_w escalation, then step-failure
    classification); the sparse route must behave identically rather than let a
    backend-specific exception escape as an uncaught crash — Feral's numeric
    factorization raises a bare ``ValueError`` here, and SuperLU would otherwise
    factor to a non-finite solution. Checking the ``O(nnz)`` value array up front
    is negligible beside the factorization it guards, and gives both inner
    solvers one backend-neutral failure signal.
    """
    if matrix.nnz and not bool(np.isfinite(matrix.data).all()):
        raise LinearSolveError(
            "sparse KKT matrix has non-finite entries (inf/NaN); the upstream "
            "derivatives overflowed at this iterate"
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
        _reject_non_finite(matrix)

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
        _reject_non_finite(matrix)
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
    """Factory pairing COO assembly with the SciPy sparse-direct solver.

    The adapter persists across an interior-point solve (the facade caches one
    instance), so it memoizes the COO→canonical-CSC map for the most recent
    ``pattern_signature``. On a fixed KKT pattern this turns the per-iteration
    ``coo_matrix(...).tocsc()`` sort into a single scatter-add (see
    :mod:`ipax.backend.sparse._canonical`).
    """

    def __init__(self) -> None:
        self._signature: object | None = None
        self._compiled: CompiledCompressed | None = None
        self._meta: tuple[tuple[int, int], int] | None = None

    def from_coo(
        self,
        rows: Array,
        cols: Array,
        values: Array,
        *,
        shape: tuple[int, int],
        symmetric: bool | None = None,
        pattern_signature: object | None = None,
    ) -> SparseOperator:
        """Build a :class:`SparseOperator` from Array-API COO triplets.

        ``symmetric`` is an optional structural hint from the assembler; ``None``
        leaves the operator to test symmetry numerically when first asked.
        ``pattern_signature`` keys the compiled-map cache: a stable signature
        (the KKT block's fixed pattern) reuses the canonical CSC structure and
        recomputes only the value array; ``None`` (value-dependent patterns)
        always rebuilds from scratch.
        """
        from ipax.backend.namespace import array_namespace

        xp = array_namespace(values)
        np_rows = _to_index(rows)
        np_cols = _to_index(cols)
        np_values = _to_numpy(values)
        n_triplets = int(np_values.shape[0])

        if pattern_signature is not None:
            csc = self._canonical_csc(
                np_rows, np_cols, np_values, shape, pattern_signature, n_triplets
            )
            return SparseOperator(
                csc,
                xp,
                symmetric=symmetric,
                pattern_signature=pattern_signature,
                csc=csc,
            )

        matrix = scipy.sparse.coo_matrix((np_values, (np_rows, np_cols)), shape=shape)
        return SparseOperator(
            matrix,
            xp,
            symmetric=symmetric,
            pattern_signature=pattern_signature,
        )

    def _canonical_csc(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        values: np.ndarray,
        shape: tuple[int, int],
        signature: object,
        n_triplets: int,
    ) -> scipy.sparse.csc_matrix:
        """Canonical CSC via the cached compiled map, recompiling on a new pattern.

        The cache is keyed on ``(signature, shape, n_triplets)``: a caller that
        reuses a signature key for a structurally different system (a grown
        L-BFGS border) recompiles rather than misapplying the stale map.
        """
        meta = (shape, n_triplets)
        if self._compiled is None or self._signature != signature or self._meta != meta:
            # major=col, minor=row ⇒ canonical CSC structure directly.
            self._compiled = compile_compressed(
                np,
                _scatter_add,
                major=cols,
                minor=rows,
                n_major=shape[1],
                n_minor=shape[0],
            )
            self._signature = signature
            self._meta = meta
        compiled = self._compiled
        data = compiled.data(values)
        csc = scipy.sparse.csc_matrix(
            (data, compiled.indices, compiled.indptr), shape=shape
        )
        # The compiled structure is sorted and duplicate-free by construction, so
        # SuperLU/Feral can consume it without a defensive re-canonicalization.
        csc.has_canonical_format = True
        return csc

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
