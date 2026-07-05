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

"""``DenseSolver`` — Array-API reference solver.

Materializes the condensed/saddle KKT matrix and solves it with
``xp.linalg.solve``. Because ``xp.linalg.solve`` is an LU factorization it
*succeeds* on an indefinite system, which would let a non-descent Newton step
through on a nonconvex problem; so before solving it probes the condensed block
``N`` (the part that must be positive definite) with ``xp.linalg.cholesky`` and
raises :class:`LinearSolveError` when ``N`` is not PD, driving the IPM's δ_w
escalation — the dense analog of the sparse route's inertia check. Both
``cholesky`` and ``solve`` are in the Array-API ``linalg`` extension, so this
stays pure Array API. Target ≤ ~1e4 variables; the reference implementation every
other solver is checked against.

``DenseOptions(kkt_route="augmented")`` selects an alternative route: instead
of condensing the inequality Gram term into ``N`` (a normal-equations step),
it keeps ``∇g``/``−Σ_s⁻¹`` as an explicit border (``operator.
augmented_dense_matrix()``, see ``ipax.ipm.kkt``) and factors that with a
pivoted Bunch-Kaufman LDLᵀ where a backend adapter is registered
(``ipax.backend.dense``), falling back to ``xp.linalg.eigh`` (Array-API pure,
always available) otherwise. Either way it exposes real inertia via
``inertia_or_none()``, so the IPM's inertia-guided δ_w correction (already
wired generically in ``ipm/driver.py``) engages for the dense route too.
Falls back to the condensed route automatically when the operator can't
expose the bordered matrix (e.g. an L-BFGS Hessian, already PD by damping).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ipax.backend.namespace import array_namespace
from ipax.linalg.solver import LinearSolveError

if TYPE_CHECKING:
    from ipax.backend.operators import LinearOperator
    from ipax.options import DenseOptions
    from ipax.typing import Array


def _eigenvalue_inertia(eigenvalues: Array, xp: Any) -> tuple[int, int, int]:
    """Sign-count eigenvalues into ``(n_pos, n_neg, n_zero)`` (Sylvester's law).

    Mirrors ``ipax.backend.sparse.numpy_scipy._dense_inertia``'s tolerance
    convention (scaled by the largest-magnitude eigenvalue and problem size),
    kept Array-API pure here since this runs in the dense route's core path.
    """
    n = int(eigenvalues.shape[0])
    if n == 0:
        return 0, 0, 0
    scale = float(xp.max(xp.abs(eigenvalues)))
    eps = float(xp.finfo(eigenvalues.dtype).eps)
    tol = max(1.0, scale) * n * eps
    pos = int(xp.sum(xp.astype(eigenvalues > tol, xp.int64)))
    neg = int(xp.sum(xp.astype(eigenvalues < -tol, xp.int64)))
    return pos, neg, n - pos - neg


class DenseSolver:
    """Cholesky/solve on the materialized condensed KKT block."""

    def __init__(self, options: DenseOptions | None = None) -> None:
        if options is None:
            from ipax.options import DenseOptions as _DenseOptions

            options = _DenseOptions()
        self._options = options
        self._operator: LinearOperator | None = None
        self._matrix: Array | None = None
        self._logical_size = 0
        # The backend LDL^T adapter (e.g. a persistent cuSOLVER handle) is
        # looked up once and reused for the solver's whole lifetime — like
        # SparseDirectSolver's cached inner solver — since re-creating a GPU
        # handle every Newton iteration would be wasteful. NOT reset by
        # factor(); a solver instance is expected to stay on one backend.
        self._augmented_adapter: Any = None
        self._adapter_checked = False
        # Per-factorization augmented (LDL^T-or-eigh) route state, lazily
        # populated on the first solve() after factor(); _augmented_tried
        # guards a single attempt per factorization (mirrors _matrix's
        # caching). _using_adapter selects which of the two mechanisms
        # populated _inertia (the adapter's own solve, or the eigh path).
        self._augmented_tried = False
        self._using_adapter = False
        self._augmented_size = 0
        self._eigenvalues: Array | None = None
        self._eigenvectors: Array | None = None
        self._inertia: tuple[int, int, int] | None = None

    def describe(self) -> str:
        """Human-readable label for diagnostics."""
        return "dense (augmented)" if self._inertia is not None else "dense"

    def set_outer_residual(self, residual: float) -> None:
        """No-op: a direct factorization has no inner tolerance to adapt."""
        del residual

    def factor(self, K: LinearOperator) -> None:
        if K.shape[0] != K.shape[1]:
            raise ValueError("DenseSolver requires a square operator")
        self._operator = K
        self._matrix = None
        self._logical_size = int(K.shape[0])
        self._augmented_tried = False
        self._using_adapter = False
        self._eigenvalues = None
        self._eigenvectors = None
        self._inertia = None

    def solve(self, rhs: Array) -> Array:
        if self._operator is None:
            raise RuntimeError("factor() must be called before solve()")

        n = self._logical_size
        if len(rhs.shape) not in (1, 2):
            raise ValueError("right-hand side must be a vector or matrix")
        if int(rhs.shape[0]) != n:
            raise ValueError("right-hand side dimension does not match operator")

        xp = array_namespace(rhs)

        if self._options.kkt_route == "augmented" and not self._augmented_tried:
            self._augmented_tried = True
            self._try_factor_augmented(xp)

        if self._eigenvalues is not None or self._using_adapter:
            return self._solve_augmented(rhs, xp)

        # Optional fast path: operators with exploitable structure (an L-BFGS
        # condensed/saddle block) solve via Woodbury without the n×n
        # materialization; the default raises NotImplementedError, so plain
        # operators fall through to materializing and factoring below.
        if self._matrix is None:
            structured_solve = getattr(self._operator, "dense_structured_solve", None)
            if structured_solve is not None:
                try:
                    return structured_solve(rhs)
                except NotImplementedError:
                    pass
                except LinearSolveError:
                    raise
                except Exception as exc:
                    raise LinearSolveError("dense structured solve failed") from exc

            self._matrix = self._materialize_dense_matrix(rhs, xp, n)
            self._guard_positive_definite(self._matrix, xp)

        matrix = self._matrix
        assert matrix is not None
        try:
            return xp.linalg.solve(matrix, rhs)
        except Exception as exc:
            raise LinearSolveError("dense linear solve failed") from exc

    def _try_factor_augmented(self, xp: Any) -> None:
        """Attempt the augmented route; no-op (silent fallback) if unsupported."""
        assert self._operator is not None
        augmented_dense_matrix = getattr(self._operator, "augmented_dense_matrix", None)
        if augmented_dense_matrix is None:
            return
        try:
            matrix = augmented_dense_matrix()
        except NotImplementedError:
            return
        except Exception as exc:
            raise LinearSolveError(
                "augmented dense matrix materialization failed"
            ) from exc

        assembled_size = int(matrix.shape[0])

        if not self._adapter_checked:
            from ipax.backend.dense import get_dense_symmetric_indefinite_adapter

            self._augmented_adapter = get_dense_symmetric_indefinite_adapter(xp)
            self._adapter_checked = True

        adapter = self._augmented_adapter
        if adapter is not None:
            adapter.factor(matrix)
            inertia = adapter.inertia_or_none()
            if inertia is not None:
                self._using_adapter = True
                self._augmented_size = assembled_size
                self._inertia = inertia
                return
            # Adapter reported no inertia (shouldn't happen in practice, but
            # stay honest): fall through to the eigh path below.

        eigh = getattr(xp.linalg, "eigh", None)
        if eigh is None:
            return  # backend lacks eigh; fall back to the condensed route

        try:
            result = eigh(matrix)
        except Exception as exc:
            raise LinearSolveError("augmented dense eigendecomposition failed") from exc

        eigenvalues = result.eigenvalues
        pos, neg, zero = _eigenvalue_inertia(eigenvalues, xp)
        if zero > 0:
            raise LinearSolveError(
                "augmented KKT block is numerically singular (a near-zero eigenvalue)"
            )
        self._eigenvalues = eigenvalues
        self._eigenvectors = result.eigenvectors
        self._augmented_size = assembled_size
        self._inertia = (pos, neg, zero)

    def _solve_augmented(self, rhs: Array, xp: Any) -> Array:
        n = self._logical_size
        pad = self._augmented_size - n
        padded_rhs = self._pad_rhs(rhs, xp, pad)

        if self._using_adapter:
            assert self._augmented_adapter is not None
            try:
                solution = self._augmented_adapter.solve(padded_rhs)
            except LinearSolveError:
                raise
            except Exception as exc:
                raise LinearSolveError("augmented dense LDL^T solve failed") from exc
            return self._truncate_solution(solution, n)

        assert self._eigenvalues is not None and self._eigenvectors is not None
        eigenvalues = self._eigenvalues
        eigenvectors = self._eigenvectors

        vt = xp.permute_dims(eigenvectors, (1, 0))
        projected = xp.matmul(vt, padded_rhs)
        if len(padded_rhs.shape) == 1:
            projected = projected / eigenvalues
        else:
            projected = projected / xp.expand_dims(eigenvalues, axis=1)
        solution = xp.matmul(eigenvectors, projected)
        return self._truncate_solution(solution, n)

    @staticmethod
    def _pad_rhs(rhs: Array, xp: Any, pad: int) -> Array:
        if pad <= 0:
            return rhs
        if len(rhs.shape) == 1:
            zero = xp.zeros((pad,), dtype=rhs.dtype)
        else:
            zero = xp.zeros((pad, rhs.shape[1]), dtype=rhs.dtype)
        return xp.concat((rhs, zero), axis=0)

    @staticmethod
    def _truncate_solution(solution: Array, n: int) -> Array:
        if int(solution.shape[0]) == n:
            return solution
        return solution[:n] if len(solution.shape) == 1 else solution[:n, :]

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """Inertia from the augmented route, or ``None`` (condensed / unsupported).

        Mirrors ``SparseDirectSolver.inertia_or_none`` so
        ``driver._inertia_acceptable`` engages for the dense route too, with
        zero changes to ``ipm/driver.py`` (invariant #3).
        """
        return self._inertia

    def _materialize_dense_matrix(self, rhs: Array, xp: Any, n: int) -> Array:
        assert self._operator is not None
        dense_matrix = getattr(self._operator, "dense_matrix", None)
        if dense_matrix is not None:
            try:
                return dense_matrix(rhs)
            except NotImplementedError:
                pass
            except Exception as exc:
                raise LinearSolveError("dense matrix materialization failed") from exc

        identity = xp.eye(n, dtype=rhs.dtype)
        try:
            return self._operator.matmat(identity)
        except Exception as exc:
            raise LinearSolveError("dense matrix materialization failed") from exc

    def _guard_positive_definite(self, matrix: Array, xp: Any) -> None:
        """Reject a non-PD condensed block so the IPM escalates δ_w.

        ``xp.linalg.solve`` (LU) would silently accept an indefinite ``N`` and
        return a non-descent step; a Cholesky on the condensed block fails exactly
        when ``N`` is not positive definite. Skips the probe when the operator
        exposes no condensed block (e.g. an L-BFGS low-rank Hessian, PD by
        damping) or when the backend lacks ``cholesky``.
        """
        primal_block = getattr(self._operator, "primal_block", None)
        block = primal_block() if primal_block is not None else None
        if block is None:
            return
        cholesky = getattr(xp.linalg, "cholesky", None)
        if cholesky is None:
            return
        n = block.shape[0]
        # For the condensed (no-equality) operator the materialized matrix *is*
        # ``N``; equality saddles store ``N`` in the leading primal block.
        primal = matrix if n == matrix.shape[0] else matrix[:n, :n]
        try:
            cholesky(primal)
        except Exception as exc:
            raise LinearSolveError("condensed block is not positive definite") from exc


__all__ = ["DenseSolver"]
