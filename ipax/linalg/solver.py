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

"""The ``LinearSolver`` protocol and strategy selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from ipax.backend.namespace import Capabilities
    from ipax.backend.operators import LinearOperator
    from ipax.options import Options
    from ipax.typing import Array

# Auto-selection thresholds. Below ``_DENSE_AUTO_MAX_VARS`` the dense route is
# the reference choice regardless of constraint shape. Between that and
# ``_TALL_DENSE_MAX_VARS`` the dense condensed (normal-equations) route is still
# preferred *when the problem is tall*: with ``m ≥ _TALL_ROW_EXCESS · n``
# inequality rows and a Gram-capable Jacobian, one n×n Cholesky per iteration
# beats Krylov's repeated O(nnz(∇g)) matvecs through the huge Jacobian
# (Breedveld 2017 §2: the condensed system is n×n however large m grows).
# The tall extension is additionally gated on Jacobian *density*: the dense
# win in this zone comes from the adapters' dense-GEMM Gram (engages at
# ≥ ~5% density; the measured 13–19× TROTS per-iteration wins), while at ~1%
# density the SpGEMM Gram + O(n³) Cholesky LOSE to Krylov (tall-crossover
# measurement 2026-07: n=10k, m=10n — dense 75.5 s/iter vs Krylov 46.5).
_DENSE_AUTO_MAX_VARS = 10_000
_TALL_DENSE_MAX_VARS = 20_000
_TALL_ROW_EXCESS = 10
# Mirrors the adapters' dense-GEMM Gram crossover (``_GRAM_DENSE_MIN_DENSITY``
# in ``backend/sparse/numpy_scipy.py`` — kept as a separate constant so the
# core does not import an adapter).
_TALL_DENSE_MIN_DENSITY = 0.05
# Estimated Gram-pattern density (``gram_fill_estimate``) at or below which a
# tall sparse-Jacobian problem takes the sparse normal-equations route instead
# of Krylov. The regimes are far apart, so the exact value is uncritical:
# the banded NE validation case (n=20k, m=10n, 20 nnz/row) has Gram fill
# ≈ 2e-3 and solved in 62 s where Krylov ran 50+ min unconverged, while
# scattered sparsity of the same nnz saturates the Gram (fill → 1), where a
# sparse factorization of the near-dense n×n block would be hopeless.
_TALL_SPARSE_NE_MAX_FILL = 0.01


class LinearSolveError(RuntimeError):
    """Backend-neutral numerical failure from a linear solve.

    The IPM regularization loop catches this type and retries with a larger
    primal regularization. Configuration errors, unsupported features, shape
    errors, and user/operator callback bugs should propagate as their original
    exceptions instead of being reclassified as numerical trouble.

    ``iterate`` optionally carries the solver's last approximate solution when
    the failure is a work limit rather than a breakdown (an iterative solver's
    truncated iterate). The KKT driver ignores it and escalates ``δ_w``;
    feasibility restoration uses it as a truncated-Newton trial direction.
    """

    def __init__(self, message: str = "", *, iterate: Array | None = None) -> None:
        super().__init__(message)
        self.iterate = iterate


@runtime_checkable
class LinearSolver(Protocol):
    """Solve ``K x = rhs`` for a KKT or condensed operator ``K``."""

    def factor(self, K: LinearOperator) -> None:
        """Prepare/factor the operator."""
        ...

    def solve(self, rhs: Array) -> Array:
        """Return ``x`` such that ``K x = rhs`` to the configured tolerance.

        One ``factor`` must serve *multiple* ``solve`` calls with different
        right-hand sides, reusing whatever the factor step prepared — the
        driver relies on this on its default path: second-order corrections
        and the centrality correctors re-solve the search direction's system
        (Wächter & Biegler 2006, §2.4, eq. (26)) without re-factoring.
        """
        ...

    def set_outer_residual(self, residual: float) -> None:
        """Hint the current outer KKT residual for an adaptive inner tolerance.

        Iterative solvers use it to drive an inexact-Newton forcing sequence
        (solve loosely while the IPM is far from optimal, tighten as it converges);
        direct solvers ignore it. Optional — the driver calls it once per iteration
        and a solver may no-op.
        """
        ...

    def is_direct(self) -> bool:
        """Whether ``factor`` does the work and ``solve`` back-substitutes it.

        ``True`` for direct solvers (Cholesky/LU/LDLᵀ: a fresh factorization
        costs one factor, a re-solve almost nothing), ``False`` for iterative
        ones (``factor`` only binds the operator; every ``solve`` is a full
        Krylov run). The driver reads it to choose between re-solving a
        *regularized* retained system and re-solving fresh at ``δ_w = 0``
        for the second-order corrections: fresh where a factorization is
        cheap, reuse where each solve is the cost. Optional, like
        ``set_outer_residual`` (the driver reads both through duck typing,
        so a solver may omit them; ``isinstance`` checks against this
        protocol do require them) — a solver without it is treated as
        direct. Must be a method, not a class attribute.
        """
        ...


def select_solver(
    *,
    n_vars: int,
    has_equalities: bool,
    capabilities: Capabilities,
    options: Options,
    m_ineq: int = 0,
    ineq_gram_capable: Callable[[], bool] | None = None,
    ineq_density: Callable[[], float | None] | None = None,
    ineq_gram_fill: Callable[[], float | None] | None = None,
) -> LinearSolver:
    """Pick a concrete linear solver from user preference and backend features.

    ``m_ineq`` (total inequality rows), ``ineq_gram_capable``, ``ineq_density``
    and ``ineq_gram_fill`` (all *lazy* structural probes of the inequality
    Jacobian — they may evaluate it at ``x0``, so they are only called when the
    tall-problem heuristic actually needs them) extend auto-selection to tall
    ``n ≪ m`` problems. ``ineq_density`` returns ``nnz/(m·n)`` or ``None``
    when the operator exposes no COO structure; ``ineq_gram_fill`` returns the
    estimated Gram-pattern density ``nnz(∇gᵀ∇g)/n²`` or ``None`` when it
    cannot be estimated (or when the caller knows the sparse normal-equations
    form is unusable for the problem, e.g. a non-L-BFGS Hessian).
    """
    from ipax.linalg.dense import DenseSolver
    from ipax.linalg.krylov import KrylovSolver

    def has_dense_solve() -> bool:
        return capabilities.has_linalg and "solve" in capabilities.linalg_functions

    mode = options.linsolve
    if mode == "dense":
        if not has_dense_solve():
            raise RuntimeError("dense linear solving requires xp.linalg.solve")
        return DenseSolver(options.dense)

    if mode == "krylov":
        return KrylovSolver(options.krylov)

    if mode == "sparse":
        if not capabilities.has_sparse_adapter:
            raise RuntimeError("sparse linear solving is unavailable for this backend")
        from ipax.linalg.sparse import SparseDirectSolver

        form = options.sparse.kkt_route
        if form == "auto":
            # Same tall gate and measured thresholds as the linsolve="auto"
            # heuristic below. The caller withholds ``ineq_gram_fill`` whenever
            # the NE form is unusable (non-L-BFGS Hessian, no gram_coo,
            # non-COO equality Jacobians), so auto then stays on "augmented".
            form = "augmented"
            if (
                n_vars < _TALL_DENSE_MAX_VARS
                and m_ineq >= _TALL_ROW_EXCESS * n_vars
                and ineq_gram_fill is not None
            ):
                fill = ineq_gram_fill()
                if fill is not None and fill <= _TALL_SPARSE_NE_MAX_FILL:
                    # Localized rows: the Gram provably stays sparse.
                    form = "normal_equations"
                elif fill is not None:
                    # The Gram fills in — but with ∇g itself past the dense
                    # crossover the bordered factor is effectively a dense
                    # (n+m) factorization through sparse machinery (TROTS
                    # dose matrices: 100% dense rows, ~8x slower end-to-end
                    # than the n×n condensation), so tall aspect decides.
                    density = ineq_density() if ineq_density is not None else None
                    if density is not None and density >= _TALL_DENSE_MIN_DENSITY:
                        form = "normal_equations"
        return SparseDirectSolver(form=form)

    dense_viable = has_dense_solve() and (
        not has_equalities or "cholesky" in capabilities.linalg_functions
    )

    # Tall n ≪ m with a provably-sparse Gram *pattern* (localized/banded rows —
    # the ``gram_fill_estimate`` sampled-column-overlap probe): one small sparse
    # factorization of the condensed n×n block per iteration beats BOTH the
    # dense route (which would form and then densely factor the same block,
    # ignoring the sparsity it certifiably has — banded tall QP with m = 10n:
    # sparse-NE 16–49× faster per iteration at every n from 250 to 9000,
    # 2026-08 crossover measurement) and Krylov (iteration counts blow up on
    # the ill-conditioned late-IPM Σ; Breedveld 2017 §2; banded validation
    # n=20k: NE optimal in 62 s vs Krylov 50+ min unconverged). This gate
    # therefore outranks the small-n dense rule below. The fill probe is only
    # consulted when the rows are NOT dense-ish (a density at or past the
    # dense-GEMM crossover fills the Gram anyway, and the probe evaluates the
    # Jacobian — don't pay it when the dense route is already the winner).
    # Equalities border into the factored saddle explicitly; whether ∇c can
    # emit that border is the caller's probe's concern (it withholds
    # ``ineq_gram_fill`` when it cannot).
    if (
        n_vars < _TALL_DENSE_MAX_VARS
        and m_ineq >= _TALL_ROW_EXCESS * n_vars
        and capabilities.has_sparse_adapter
        and ineq_gram_fill is not None
    ):
        density = ineq_density() if ineq_density is not None else None
        if density is None or density < _TALL_DENSE_MIN_DENSITY:
            fill = ineq_gram_fill()
            if fill is not None and fill <= _TALL_SPARSE_NE_MAX_FILL:
                from ipax.linalg.sparse import SparseDirectSolver

                return SparseDirectSolver(form="normal_equations")

    if n_vars < _DENSE_AUTO_MAX_VARS and dense_viable:
        return DenseSolver(options.dense)

    # Tall n ≪ m: the condensed block stays n×n however many inequality rows
    # exist, and a Gram-capable Jacobian forms it without densifying m×n —
    # prefer the direct dense route over Krylov through the huge Jacobian,
    # but only when the rows are dense enough for the dense-GEMM Gram (see
    # the threshold comments above; sparse tall Jacobians with a *filling*
    # Gram measure faster on Krylov in this zone, and the provably-sparse-Gram
    # case was already routed to sparse-NE above).
    if n_vars < _TALL_DENSE_MAX_VARS and m_ineq >= _TALL_ROW_EXCESS * n_vars:
        if dense_viable and ineq_gram_capable is not None and ineq_gram_capable():
            density = ineq_density() if ineq_density is not None else None
            if density is None or density >= _TALL_DENSE_MIN_DENSITY:
                return DenseSolver(options.dense)

    return KrylovSolver(options.krylov)


def select_restoration_solver(
    options: Options, *, n_vars: int
) -> Callable[[], LinearSolver] | None:
    """Factory for the iterative restoration solver; ``None`` keeps dense.

    ``"auto"`` applies the same size cutoff as the main route's
    ``linsolve="auto"`` (``_DENSE_AUTO_MAX_VARS``): the dense restoration
    materializes two ``n × n`` arrays and solves them per damping trial, so
    it stops being viable exactly where the dense KKT route does, while the
    v29 paired S2MPJ sweep showed the matrix-free route losing only on small
    problems (every flipped row had ``n ≤ 1247``).

    A factory rather than an instance: the driver builds a fresh solver per
    restoration entry, so the operator a solver retains after ``factor()`` —
    and through it the last restoration point's ``m × n`` Jacobians — is
    released when restoration returns instead of living for the rest of the run.
    """
    mode = options.restoration.linear_solver
    if mode == "dense" or (mode == "auto" and n_vars < _DENSE_AUTO_MAX_VARS):
        return None
    from ipax.linalg.krylov import KrylovSolver

    krylov_options = options.restoration.krylov
    return lambda: KrylovSolver(krylov_options)


__all__ = [
    "LinearSolveError",
    "LinearSolver",
    "select_restoration_solver",
    "select_solver",
]
