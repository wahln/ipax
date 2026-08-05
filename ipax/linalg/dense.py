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

When the probed block spans the whole system (the condensed route), the probe's
Cholesky factor is kept and every solve back-substitutes it through the
``get_dense_cholesky_solve`` backend gap-filler (the Array-API ``linalg``
extension has no triangular solve), replacing the redundant O(n³) LU refactor
with an O(n²) solve — this also makes corrector/SOC back-solves against the
same factorization cheap. Backends without the gap-filler (array-api-strict,
JAX) keep the original LU path.

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

import math
from typing import TYPE_CHECKING, Any

from ipax.backend.namespace import array_namespace
from ipax.linalg.solver import LinearSolveError

if TYPE_CHECKING:
    from collections.abc import Callable

    from ipax.backend.operators import LinearOperator
    from ipax.options import DenseOptions
    from ipax.typing import Array


# Floor for the refinement stopping tolerance, in units of the rhs dtype's
# machine epsilon: ``DenseOptions.refine_tol`` assumes float64 working
# precision, so at a coarser working dtype the absolute default (1e-10) would
# be unreachable and every solve would burn the whole refinement budget. The
# achievable limiting residual of fixed-precision refinement is a small
# multiple of u·‖N‖‖x‖ (Carson & Higham 2018), hence a small constant times
# eps. A backstop today — ``_resolved_gram_dtype`` already refuses to reduce
# to anything not strictly narrower than the working dtype, so a float32
# solve never engages the mixed route in the first place.
_REFINE_TOL_EPS_FACTOR = 16.0


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
        # Cholesky factor kept from the PD probe when it spans the whole
        # system, plus the once-per-instance backend back-substitution lookup
        # (mirrors _adapter_checked; a solver stays on one backend).
        self._cholesky_factor: Array | None = None
        self._cholesky_solve: Callable[[Array, Array], Array] | None = None
        self._cholesky_solve_checked = False
        # Mixed-precision Gram state (DenseOptions.gram_dtype): whether the
        # *current* factorization was materialized with a reduced-precision
        # Gram (so solves must be refined), and the instance-wide kill switch
        # set the first time reduced precision demonstrably fails (refinement
        # stall, or a PD failure the exact matrix does not reproduce) — the
        # κ(N)·u32 ≳ 1 endgame, after which every factorization stays native.
        self._mixed_engaged = False
        self._mixed_disabled = False
        self._mixed_ever_engaged = False
        # Consecutive mixed-route failures (refinement rejection or a
        # precision-caused PD mismatch). κ(N) is NOT monotone along an IPM run
        # (μ jumps, δ_w regularization, re-centering), so one hard iteration
        # must not permanently forfeit the reduced-precision savings; only
        # ``refine_failure_limit`` failures in a row flip the kill switch.
        # Any certified mixed solve resets the counter.
        self._mixed_failures = 0
        self._mixed_label = ""  # resolved reduced dtype of the engaged route
        self.refine_iterations = 0  # corrections applied by the last solve()

    def describe(self) -> str:
        """Human-readable label for diagnostics.

        The mixed marker is sticky: a run that used the reduced-precision
        Gram and later self-disabled reports ``gram=float32->native`` rather
        than pretending it ran native throughout (``Result.routes`` captures
        this label once, after the driver returns).
        """
        if self._inertia is not None:
            return "dense (augmented)"
        if self._mixed_engaged:
            return f"dense (gram={self._mixed_label})"
        if self._mixed_disabled and self._mixed_ever_engaged:
            return f"dense (gram={self._mixed_label}->native)"
        return "dense"

    def kkt_form(self) -> str:
        """The KKT assembly actually factored (``Result.routes.kkt_form``).

        ``"augmented"`` only when the bordered route engaged; a configured
        augmented route that fell back (L-BFGS Hessian, oversized border)
        honestly reports ``"condensed"``.
        """
        return "augmented" if self._inertia is not None else "condensed"

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
        self._cholesky_factor = None
        self._mixed_engaged = False
        self.refine_iterations = 0

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

            self._materialize_and_guard(rhs, xp, n)

        matrix = self._matrix
        assert matrix is not None
        self.refine_iterations = 0
        x = self._solve_factored(matrix, rhs, xp)
        if not self._mixed_engaged:
            return x

        refined = self._refine(x, rhs, xp)
        if refined is not None:
            self._mixed_failures = 0
            return refined
        # Refinement rejected the solve (stall or budget exhaustion above the
        # acceptance level): the reduced-precision factor is not accurate
        # enough for THIS system. Rebuild the exact matrix and answer from
        # the exact factorization; only ``refine_failure_limit`` consecutive
        # failures disable the mixed route for good — conditioning along an
        # IPM run is not monotone (μ jumps, δ_w, re-centering), so one hard
        # iteration must not forfeit the savings on every later one.
        self._register_mixed_failure()
        self._mixed_engaged = False
        self._cholesky_factor = None
        self._matrix = self._materialize_dense_matrix(rhs, xp, n)
        self._guard_positive_definite(self._matrix, xp)
        return self._solve_factored(self._matrix, rhs, xp)

    def _register_mixed_failure(self) -> None:
        self._mixed_failures += 1
        if self._mixed_failures >= self._options.refine_failure_limit:
            self._mixed_disabled = True

    def _materialize_and_guard(self, rhs: Array, xp: Any, n: int) -> None:
        """Materialize + PD-probe ``self._matrix``, preferring the mixed route.

        With ``gram_dtype != "native"`` (and no prior failure) the condensed
        matrix is materialized through the operator's ``dense_matrix_mixed``
        hook — the Gram term accumulated in reduced precision. A PD failure of
        the mixed matrix that the exact matrix does *not* reproduce is
        precision noise: the solver keeps the exact matrix and permanently
        disables the mixed route; a failure the exact matrix reproduces is
        genuine (propagates, mixed stays enabled).

        Note the PD probe runs on the *approximate* matrix, so the converse
        masking — an indefinite exact ``N`` whose mixed materialization is PD —
        passes the guard here. That masking requires a negative eigenvalue of
        size ≲ u32·‖N‖, which forces the refinement contraction ρ ≈ κ·u32 ≳ 1:
        the stall detector in :meth:`_refine` then rejects the solve and the
        exact rebuild re-probes (and fails) on the exact matrix. The guard and
        the stall detector are a *pair* — weakening either breaks the dense
        route's indefiniteness detection under mixed precision (pinned by
        ``test_masked_indefinite_exact_matrix_still_escalates``).
        """
        mixed_hook = getattr(self._operator, "dense_matrix_mixed", None)
        gram_dtype = self._resolved_gram_dtype(rhs, xp)
        if (
            gram_dtype is not None
            and not self._mixed_disabled
            and mixed_hook is not None
        ):
            try:
                matrix = mixed_hook(
                    rhs, gram_dtype, hinted_only=self._options.gram_dtype == "auto"
                )
            except NotImplementedError:
                matrix = None
            except Exception as exc:
                raise LinearSolveError("dense matrix materialization failed") from exc
            if matrix is not None:
                self._mixed_ever_engaged = True
                try:
                    self._guard_positive_definite(matrix, xp)
                except LinearSolveError:
                    self._cholesky_factor = None
                    exact = self._materialize_dense_matrix(rhs, xp, n)
                    self._guard_positive_definite(exact, xp)  # genuine ⇒ raises
                    self._register_mixed_failure()
                    self._mixed_engaged = False
                    self._matrix = exact
                    return
                self._mixed_engaged = True
                self._matrix = matrix
                return
        self._mixed_engaged = False
        self._matrix = self._materialize_dense_matrix(rhs, xp, n)
        self._guard_positive_definite(self._matrix, xp)

    def _resolved_gram_dtype(self, rhs: Array, xp: Any) -> str | None:
        """The concrete reduced dtype to accumulate the Gram in, or ``None``.

        ``"native"`` never reduces; ``"float32"`` forces it; ``"auto"`` (the
        default) asks the operator whether its constraint data carries only
        reduced-precision information (``gram_accumulate_dtype_hint`` —
        declared metadata, e.g. float32 dose matrices upcast at load) and is
        a strict no-op when it does not, so fully-float64 problems never pay
        a refinement pass. Resolution happens once, here: everything
        downstream sees a concrete dtype name.

        Either way the candidate must be *strictly narrower* than the working
        dtype the solve runs in. Reducing to the working precision is the
        native arithmetic bit-for-bit, so engaging the mixed route there
        would buy nothing and cost a refinement pass — a float32 solve of
        float32 data is simply a float32 solve.
        """
        mode = self._options.gram_dtype
        if mode == "native":
            return None
        if mode == "float32":
            candidate = "float32"
        else:
            hint_fn = getattr(self._operator, "gram_accumulate_dtype_hint", None)
            if hint_fn is None:
                return None
            try:
                candidate = hint_fn()
            except NotImplementedError:
                return None
            if candidate is None:
                return None
        reduced = getattr(xp, candidate, None)
        if reduced is None:
            return None
        # Coarser precision ⇔ wider eps, so a real reduction needs the
        # candidate's eps strictly above the working dtype's.
        if float(xp.finfo(reduced).eps) <= float(xp.finfo(rhs.dtype).eps):
            return None
        self._mixed_label = candidate if mode == "float32" else f"auto:{candidate}"
        return candidate

    def _solve_factored(self, matrix: Array, rhs: Array, xp: Any) -> Array:
        """Back-substitute the kept Cholesky factor, falling back to LU."""
        if self._cholesky_factor is not None:
            solve_cholesky = self._lookup_cholesky_solve(xp)
            if solve_cholesky is not None:
                try:
                    return solve_cholesky(self._cholesky_factor, rhs)
                except Exception:
                    # The reuse is purely an optimization: the materialized
                    # matrix is still in hand, so an unexpected trsm/potrs
                    # failure falls back to LU instead of escalating delta_w
                    # (the factor is dropped so later solves skip the retry).
                    self._cholesky_factor = None
        return self._solve_lu(matrix, rhs, xp)

    def _refine(self, x: Array, rhs: Array, xp: Any) -> Array | None:
        """Fixed-precision iterative refinement against the exact operator.

        The factorization approximates ``N`` only to the reduced-precision
        Gram's rounding, but the residual is evaluated with the operator's
        exact float64 ``matvec`` — classic fixed-precision refinement
        (Wilkinson; Carson & Higham 2018, SIAM J. Sci. Comput. 40(2)): each
        correction contracts the error by ρ ≈ κ(N)·u32, so a handful of
        O(n²)-solve + matvec steps restores working accuracy whenever
        ρ < 1. The target is ``refine_tol``; when the budget runs out or the
        contraction plateaus (which includes plateauing at the achievable
        rounding floor ~κ·u64), the solve is still *accepted* if its
        measured exact residual is within ``refine_accept_tol`` — an honest
        certificate, just a looser one. Returns ``None`` only when even that
        level is missed — the caller's signal to rebuild in native precision.
        """
        assert self._operator is not None and self._matrix is not None
        apply = self._operator.matvec if len(rhs.shape) == 1 else self._operator.matmat
        bnorm = float(xp.max(xp.abs(rhs)))
        if bnorm == 0.0 or not math.isfinite(bnorm):
            return x
        # Floor the stopping tolerance on the working dtype's eps: the
        # configured default assumes float64 and would be unreachable on a
        # float32 working dtype (see _REFINE_TOL_EPS_FACTOR).
        tol = max(
            self._options.refine_tol,
            _REFINE_TOL_EPS_FACTOR * float(xp.finfo(rhs.dtype).eps),
        )
        accept_tol = max(self._options.refine_accept_tol, tol)
        stall_ratio = self._options.refine_stall_ratio
        max_iters = self._options.refine_max_iters
        previous = math.inf
        # Track the best iterate: when κ·u32 > 1 the sequence diverges, and
        # because the fp32 Gram's rounding concentrates in the small-eigenvalue
        # subspace the *initial* iterate is typically the best one — accepting
        # the minimum-residual iterate instead of the last keeps those solves.
        best = x
        best_rnorm = math.inf
        for iteration in range(max_iters + 1):
            self.refine_iterations = iteration
            try:
                residual = rhs - apply(x)
            except Exception:
                return None  # no exact matvec ⇒ a mixed factor is uncertifiable
            rnorm = float(xp.max(xp.abs(residual)))
            if rnorm < best_rnorm:
                best = x
                best_rnorm = rnorm
            if rnorm <= tol * bnorm:
                return x
            if rnorm >= stall_ratio * previous or iteration == max_iters:
                # Plateaued, diverging, or out of budget: accept the best
                # iterate on its measured exact residual if it clears the
                # (looser) acceptance certificate.
                return best if best_rnorm <= accept_tol * bnorm else None
            previous = rnorm
            x = x + self._solve_factored(self._matrix, residual, xp)
        return None

    def _solve_lu(self, matrix: Array, rhs: Array, xp: Any) -> Array:
        try:
            return xp.linalg.solve(matrix, rhs)
        except Exception as exc:
            raise LinearSolveError("dense linear solve failed") from exc

    def _lookup_cholesky_solve(self, xp: Any) -> Callable[[Array, Array], Array] | None:
        if not self._cholesky_solve_checked:
            from ipax.backend.dense import get_dense_cholesky_solve

            self._cholesky_solve = get_dense_cholesky_solve(xp)
            self._cholesky_solve_checked = True
        return self._cholesky_solve

    def _try_factor_augmented(self, xp: Any) -> None:
        """Attempt the augmented route; no-op (silent fallback) if unsupported."""
        assert self._operator is not None
        augmented_dense_matrix = getattr(self._operator, "augmented_dense_matrix", None)
        if augmented_dense_matrix is None:
            return
        # Size guard for tall problems: the bordered matrix is
        # (n + m_eq + m_ineq)² dense, so with m ≫ n materializing it would
        # allocate gigabytes for a system whose condensed form is n × n. Checked
        # via the operator's size hook *before* materialization; fall back to
        # the condensed route silently (losing only the inertia diagnostic).
        assembled_size_hook = getattr(self._operator, "augmented_assembled_size", None)
        if (
            assembled_size_hook is not None
            and int(assembled_size_hook()) > self._options.augmented_max_size
        ):
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

        When the probed block spans the whole system (the condensed route),
        the factor is kept so ``solve`` back-substitutes it instead of paying
        a second O(n³) LU factorization of the same matrix. An equality
        saddle's probe covers only the leading ``N`` block of the indefinite
        bordered matrix, so nothing is kept there.
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
            factor = cholesky(primal)
        except Exception as exc:
            raise LinearSolveError("condensed block is not positive definite") from exc
        # Keep the factor only where a backend back-substitution exists;
        # otherwise it would be dead n×n memory next to the LU path.
        if primal is matrix and self._lookup_cholesky_solve(xp) is not None:
            self._cholesky_factor = factor


__all__ = ["DenseSolver"]
