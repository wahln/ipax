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

"""The main IPM loop: convergence test, logging, callbacks.

``IPMDriver`` ties the components together but owns no linear-algebra strategy:
the ``LinearSolver`` is injected (invariant #3), so new solve routes never touch
this file. Strict convergence is the scaled KKT ∞-norm ≤ ``tol`` (IPOPT
scaling ``s_d, s_c``; Wächter & Biegler 2006, eqs. 4–6); an explicit checker
handles optional guarded acceptable-stagnation and wall-time exits.

Handles bounds + (nonlinear) inequalities + equalities, with the Lagrangian
Hessian resolved by precedence — analytic operator, autodiff HVP, or the
persistent Powell-damped L-BFGS approximation whose curvature pairs this loop
updates from each accepted step. Inequalities/bounds use the condensed
normal-equations route (Breedveld 2017, eq. 18); equalities border it into the
Friedlander–Orban quasidefinite saddle. Globalization is the filter line-search
(default) or the Breedveld step controller, with a Gauss-Newton feasibility
restoration phase. The injected ``LinearSolver`` is the dense reference solver,
the matrix-free Krylov solver, or the sparse-direct route; restoration keeps
its dense reference solve unless a second solver factory is injected for it
(``RestorationOptions``).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeVar

from ipax._logging import (
    HEADER_REPEAT_INTERVAL,
    ITERATION,
    format_header,
    format_record,
    logger,
)
from ipax.backend.operators import (
    Dense,
    Diagonal,
    LinearOperator,
    MatrixFreeJacobian,
    VStack,
    as_operator,
)
from ipax.ipm.barrier import (
    FreeModeMonitor,
    adaptive_mu,
    boundary_ratio,
    breedveld_mu,
    complementarity_measures,
    fallback_mu,
    update_mu,
)
from ipax.ipm.breedveld_ls import BreedveldController
from ipax.ipm.corrections import (
    CenteringOnly,
    CorrectionContext,
    HigherOrderCorrection,
    probing_mu,
    quality_mu,
    select_corrector,
)
from ipax.ipm.filter_ls import Filter, FilterLineSearch
from ipax.ipm.hessian import LBFGSOperator
from ipax.ipm.init import (
    apply_warm_start,
    initialize,
    least_squares_duals,
    recenter_slacks_duals,
)
from ipax.ipm.kkt import build_condensed_operator, build_saddle_operator
from ipax.ipm.restoration import RestorationExit, feasible_theta_tol, restore
from ipax.ipm.step import NewtonStep, recover_eliminated
from ipax.ipm.termination import ConditionChecker
from ipax.linalg.regularize import (
    RegularizationState,
    escalate_delta_c,
    escalate_delta_w,
)
from ipax.linalg.solver import LinearSolveError
from ipax.problem.autodiff import get_autodiff_adapter
from ipax.result import (
    DerivativeSources,
    IterationInfo,
    IterationRecord,
    KKTResiduals,
    Result,
    Status,
)

if TYPE_CHECKING:
    from ipax.linalg.solver import LinearSolver
    from ipax.options import MuSchedule, OptimalityConditionOptions, Options
    from ipax.problem.base import Problem
    from ipax.result import IterationCallback, WarmStart
    from ipax.typing import Array, Namespace


T = TypeVar("T")


# IPOPT (Wächter & Biegler 2006) constants kept out of the loop body.
_S_MAX = 100.0  # eq. (5): cap on the dual/complementarity scaling factors
_KAPPA_EPSILON = 10.0  # eq. (7): barrier sub-problem tolerance factor κ_ε
# Relative tolerance for the stall detector's "KKT error unchanged" test.
_STALL_REL_TOL = 1e-12
# Failure statuses that return the best accepted iterate instead of the last.
_FAILURE_RETURNS_BEST = frozenset(
    {
        Status.INFEASIBLE,
        Status.STALLED,
        Status.MAX_ITER,
        Status.MAX_TIME,
        Status.NUMERICAL_ERROR,
        Status.RESTORATION_FAILED,
    }
)
# δ_w escalation attempts for the feasible-point descent enforcement.
_MAX_DESCENT_ATTEMPTS = 25
# dφ must exceed this fraction of the barrier objective's scale to count as a
# *meaningful* ascent direction: near convergence dφ is float noise around
# zero (and the near-zero step passes Armijo anyway), so escalating δ_w there
# would only distort an already-converged step with futile re-solves.
_DESCENT_NOISE_FACTOR = 1e-12
_MAX_REG_ATTEMPTS = 40
_MAX_MU_REDUCTIONS = 64
# A step solve that fails at a point already within this multiple of the
# optimality tolerances is reported ACCEPTABLE rather than NUMERICAL_ERROR: near
# a solution the condensed system is ill-conditioned, so the step can be
# non-finite even though the iterate is essentially optimal (IPOPT
# ``acceptable_tol`` ≈ 1e2 × ``tol``).
_STEP_FAILURE_ACCEPT_FACTOR = 1e2
# eq. (18): the filter guard region is {θ ≥ θ_max} with θ_max = 1e4·max(1, θ(x0)).
# Bounds how infeasible an accepted trial may be, so an f-type step whose barrier
# objective collapses cannot be taken while the constraint violation explodes.
_THETA_MAX_FACTOR = 1e4
# The eq. (23) constraint-violation threshold θ_min = fac·max(1, θ(x_0)), the
# mirror of θ_max above (IPOPT's theta_min_fact/theta_max_fact defaults).
_THETA_MIN_FACTOR = 1e-4
# A restoration that signals local infeasibility is only believed when the
# returned iterate is *actually* infeasible by the driver's own θ metric — its
# violation must exceed this multiple of the constraint-violation tolerance.
# Restoration's raw ℓ∞ measure can stall a hair above its own (differently
# scaled) tolerance at a point the solver considers feasible; declaring such a
# point "locally infeasible" is a contradiction (S2MPJ Task 1: HS13/HS56/HS72).
# The multiple is generous but far below any genuinely-infeasible stationary
# point (whose violation is bounded away from zero, ≫ tol). It gates BOTH the
# "resume vs believe the claim" decision and the x0-anchored second-chance
# probe, so it must stay tight enough to keep triggering the rescue path that
# recovers problems like HS111/OET7 (a looser value here reroutes them into a
# worse basin — S2MPJ v12 regression). The *terminal* near-feasible downgrade
# is a separate, looser threshold below (`_NEAR_FEASIBLE_FACTOR`).
_RESTORATION_INFEASIBLE_FACTOR = 1e3
# Terminal near-feasibility band (IPOPT `constr_viol_tol` level, 1e-4 at the
# default): a restored point that *would* be declared locally infeasible but is
# itself feasible to this band is reported STALLED instead — a point feasible to
# ~1e-4 is not distinguishable from a degenerate near-feasible optimum, so
# "locally infeasible" is the wrong verdict (S2MPJ v11 item 3: LEWISPOL — 9
# nonlinear eqs with a multiplicity-3 degenerate root — and the
# ARGAUSS/LANCZOS/MISRA1B NLS cluster floor at θ ~ 1e-5, the float64 limit).
# Applied only at the terminal INFEASIBLE emission (NOT the resume/second-chance
# decision above), so it never reroutes a rescuable problem. Genuinely
# infeasible problems keep their verdict (violation ≫ 1e-4: BURKEHAN 1.0,
# PDE1 2.5).
_NEAR_FEASIBLE_FACTOR = 1e4
# An *uncertified* restoration stall (window/budget exit, no stationarity
# certificate) resumes the main loop only while restoration keeps reducing the
# violation by at least this factor between stalls; otherwise the run ends as
# RESTORATION_FAILED. Bounds the resume loop: θ must shrink geometrically, so
# at most ~log(θ0/tol)/log(1/factor) resumes can occur.
_RESTORATION_PROGRESS_FACTOR = 0.9
# Barrier escalation on a *repeated* feasible-point re-center. The first
# re-center repairs slacks/duals at the current μ (the HS101 fix); when that
# repeats, the barrier problem itself is what cannot be solved — observed on an
# RT fluence case where a free-mode μ oracle pinned μ at its floor and every
# re-center re-derived the same rejected direction for hundreds of iterations
# (at θ0 = 0 the W&B eq. (19) switching condition holds for every descent
# direction, so each trial faces the full Armijo test against the stiff
# barrier). The repeat raises μ to the scale of the stationarity residual the
# barrier must absorb — the same "iterate outran the barrier" response as
# IPOPT's adaptive oracle raising μ (cf. Nocedal, Wächter & Waltz 2009, §5.1
# re-initialization) — by at least this factor (a smaller raise changes
# nothing against an α-halving line search) and at most back to ``mu_init``
# (never worse-centered than a cold start).
_RECENTER_MU_RAISE_FACTOR = 10.0
_RECENTER_MU_ERROR_FRACTION = 0.1  # μ_new target: fraction of the stall's KKT error


def _norm_inf(xp: Namespace, v: Array) -> float:
    if int(v.shape[0]) == 0:
        return 0.0
    return float(xp.max(xp.abs(v)))


def _norm1(xp: Namespace, v: Array) -> float:
    if int(v.shape[0]) == 0:
        return 0.0
    return float(xp.sum(xp.abs(v)))


def _clamped_min(xp: Namespace, ratios: list[Array]) -> float:
    """Fraction-to-boundary step from per-block ratios: ``min(1, min ratios)``,
    floored at zero, in a single host sync (1.0 when nothing can block)."""
    if not ratios:
        return 1.0
    alpha = float(ratios[0]) if len(ratios) == 1 else float(xp.min(xp.stack(ratios)))
    return max(0.0, min(1.0, alpha))


def _max_of(xp: Namespace, parts: list[Array]) -> float:
    """``max`` of device-side scalars with a single host sync (0.0 if empty)."""
    if not parts:
        return 0.0
    if len(parts) == 1:
        return float(parts[0])
    return float(xp.max(xp.stack(parts)))


def _sum_of(xp: Namespace, parts: list[Array]) -> float:
    """``sum`` of device-side scalars with a single host sync (0.0 if empty)."""
    if not parts:
        return 0.0
    if len(parts) == 1:
        return float(parts[0])
    return float(xp.sum(xp.stack(parts)))


def _within_relaxed_tol(
    optimality: OptimalityConditionOptions, record: IterationRecord
) -> bool:
    """Whether every enabled scaled KKT component is within the relaxed tolerance.

    "Relaxed" is :data:`_STEP_FAILURE_ACCEPT_FACTOR` × the optimality tolerance
    (IPOPT ``acceptable_tol``). Used to decide whether a stall — a failed step
    solve, or a line search handing off to restoration — sits at an essentially
    optimal iterate that should be accepted rather than discarded.
    """
    factor = _STEP_FAILURE_ACCEPT_FACTOR
    checks = (
        (optimality.dual_inf_tol, record.dual_infeasibility),
        (optimality.constr_viol_tol, record.primal_infeasibility),
        (optimality.compl_inf_tol, record.complementarity),
    )
    enabled = [(tol, value) for tol, value in checks if tol is not None]
    return bool(enabled) and all(value <= factor * tol for tol, value in enabled)


def _feasible_evidence_tol(optimality: OptimalityConditionOptions) -> float:
    """The θ level below which an iterate counts as evidence of feasibility.

    The same threshold the local-infeasibility verdict uses: a restored point
    is believed infeasible only above it, so an *accepted* iterate below it
    directly contradicts any later infeasibility claim.
    """
    tol = optimality.constr_viol_tol
    return _RESTORATION_INFEASIBLE_FACTOR * (
        tol if tol is not None else optimality.kkt_tol
    )


def _near_feasible_tol(optimality: OptimalityConditionOptions) -> float:
    """The θ band (IPOPT ``constr_viol_tol`` level) below which a restored point
    is treated as near-feasible: it is reported STALLED rather than declared
    locally infeasible, since a point feasible to ~1e-4 is not distinguishable
    from a degenerate near-feasible optimum (S2MPJ item 3: LEWISPOL cluster)."""
    tol = optimality.constr_viol_tol
    return _NEAR_FEASIBLE_FACTOR * (tol if tol is not None else optimality.kkt_tol)


def _restoration_reports_infeasible(
    theta_restored: float, optimality: OptimalityConditionOptions
) -> bool:
    """Whether a restoration "infeasible" signal should be believed.

    Restoration minimizes the constraint infeasibility and reports local
    infeasibility when it stalls with the violation above its own tolerance. That
    verdict is only trustworthy when the returned iterate is *genuinely*
    infeasible by the driver's own θ — its violation must exceed
    :data:`_RESTORATION_INFEASIBLE_FACTOR` × the constraint-violation tolerance.
    Restoration's raw ℓ∞ measure is scaled differently and can stall a hair above
    its threshold at a point that is feasible here (a degenerate optimum where
    constraint qualification fails, or a limit cycle that keeps re-reaching
    feasibility); reporting such a point as infeasible would contradict the fact
    that it is feasible (S2MPJ Task 1). A genuinely infeasible stationary point
    has a violation bounded well away from zero, far above this threshold.
    """
    return theta_restored > _feasible_evidence_tol(optimality)


def _classify_step_failure(
    optimality: OptimalityConditionOptions, record: IterationRecord
) -> tuple[Status, str]:
    """Classify a failed step solve as ACCEPTABLE (near-optimal) or an error.

    Near a solution the condensed system becomes ill-conditioned and the Newton
    step can be non-finite even when the iterate is essentially optimal (e.g. μ
    driven well below the achieved KKT residual). Salvage such an iterate as
    ACCEPTABLE rather than discarding a usable solution; otherwise the failure is
    a genuine numerical error.
    """
    if _within_relaxed_tol(optimality, record):
        return (
            Status.ACCEPTABLE,
            "acceptable: step solve failed at a point within the relaxed KKT tolerance",
        )
    return (
        Status.NUMERICAL_ERROR,
        "condensed factorization failed despite regularization",
    )


@dataclass
class _RestorationState:
    """Loop-persistent state for the feasibility-restoration handler.

    ``second_chance_used`` fires the x0-anchored infeasibility probe at most
    once; ``uncertified_stall_theta`` tracks the θ at the last uncertified
    restoration stall so a resume is only justified while restoration keeps
    reducing the violation between stalls; ``feasible_recenters`` counts the
    feasible-point re-centers so a *repeat* escalates the barrier (raises μ)
    instead of treadmilling at the same one. Held in an explicit object (not
    module state, invariant #5) and threaded through the loop.
    """

    second_chance_used: bool = False
    uncertified_stall_theta: float = float("inf")
    feasible_recenters: int = 0


@dataclass(frozen=True)
class _RestorationOutcome:
    """Result of :meth:`IPMDriver._handle_restoration`, applied by the caller.

    ``resume=True`` restarts the main loop from the updated ``(x, s, y_ineq)``;
    otherwise the run terminates with ``status``/``message``. Keeping the
    control-flow decision in the return value lets the two entry points — a
    filter line-search failure and a step-solve failure the δ_w ladder could
    not repair — share one handler. A non-``None`` ``mu`` carries a barrier
    escalation (a repeated feasible-point re-center raised μ); the caller
    applies it to the loop's μ and suspends any free-mode μ oracle so the
    raise is not immediately re-targeted away. A non-``None`` ``y_eq`` carries
    repaired equality multipliers (the analogue of the inequality central-band
    clip, for problems whose only constraints are equalities).
    """

    resume: bool
    x: Array | None = None
    s: Array | None = None
    y_ineq: Array | None = None
    y_eq: Array | None = None
    status: Status | None = None
    message: str | None = None
    mu: float | None = None


_MISS = object()


class _PointCache:
    """Memoized problem callbacks at the most recently visited primal points.

    The globalization evaluates ``f``/``c``/``g`` (and, on the L-BFGS route, the
    gradient) at every trial point; the accepted trial then becomes the next
    iterate, whose loop-top evaluation would repeat all of them — as would
    ``phi0``/``theta0`` at the current iterate. Keying by array *identity*
    keeps the lookup O(1) with no device sync: the driver adopts the trial
    array object itself on acceptance (see ``run``), so the hit is exact. A
    fresh array — a restoration or re-centered point — is simply a miss.

    A tiny LRU of ``capacity`` points (current iterate, trial, SOC point) is
    enough; each entry keeps a strong reference to its array so ``id`` cannot
    be recycled while cached.
    """

    __slots__ = ("_capacity", "_entries")

    def __init__(self, capacity: int = 4) -> None:
        self._capacity = capacity
        self._entries: dict[int, tuple[Any, dict[str, Any]]] = {}

    def get(self, x: Array, key: str) -> Any:
        entry = self._entries.get(id(x))
        if entry is None or entry[0] is not x:
            return _MISS
        return entry[1].get(key, _MISS)

    def retain(self, x: Array) -> None:
        """Drop every cached point except ``x`` (the iterate the loop adopted).

        Called at the top of each iteration so stale Jacobians/operators of
        rejected trials or previous iterates are released immediately rather
        than living on until the LRU cycles them out — at 1e5 variables a
        backlog of explicit Jacobians is a multiplier on peak memory.
        """
        entry = self._entries.get(id(x))
        self._entries.clear()
        if entry is not None and entry[0] is x:
            self._entries[id(x)] = entry

    def put(self, x: Array, key: str, value: Any) -> None:
        entry = self._entries.get(id(x))
        if entry is None or entry[0] is not x:
            if len(self._entries) >= self._capacity:
                del self._entries[next(iter(self._entries))]
            entry = (x, {})
            self._entries[id(x)] = entry
        entry[1][key] = value


class IPMDriver:
    """Primal–dual interior-point iteration (condensed normal-equations route)."""

    def __init__(
        self,
        problem: Problem,
        *,
        xp: Namespace,
        solver: LinearSolver,
        options: Options,
        lower: Array | None,
        upper: Array | None,
        has_ineq: bool,
        has_eq: bool = False,
        callback: IterationCallback | None = None,
        record_transform: Callable[[IterationRecord], IterationRecord] | None = None,
        warm_start: WarmStart | None = None,
        restoration_solver_factory: Callable[[], LinearSolver] | None = None,
    ) -> None:
        self._problem = problem
        self._xp = xp
        self._solver = solver
        # A factory, built per restoration entry: the solver's retained operator
        # (and the Jacobians behind it) must not outlive the restoration call.
        self._restoration_solver_factory = restoration_solver_factory
        # Whether the solver's retained factorization is the iteration's
        # *unregularized* KKT matrix (δ_w = 0, base δ_c). Recorded by
        # ``_solve_step`` at factor time — the only place the main solver is
        # factored — so every consumer (the SOC reuse gate) reads the truth
        # rather than reconstructing it from flags kept in step with each
        # refactoring site.
        self._factor_unregularized = False
        self._options = options
        self._lower = lower
        self._upper = upper
        self._has_ineq = has_ineq
        self._has_eq = has_eq
        self._callback = callback
        self._record_transform = record_transform
        self._warm_start = warm_start
        # Higher-order step correction strategy. The default is a no-op
        # (``active`` is False), so the predictor direction is used unchanged.
        corrections = options.corrections
        assert not isinstance(corrections, str)  # normalized by Options.__post_init__
        self._corrector: HigherOrderCorrection = select_corrector(corrections)
        # Standalone "probing"/"quality" (no corrector requested) run the
        # corrector path with a plain centered re-solve, since both oracles
        # need the affine direction (Nocedal, Wächter & Waltz 2009, §3).
        self._mu_schedule: MuSchedule = options.mu_schedule
        if self._mu_schedule in ("probing", "quality") and not self._corrector.active:
            self._corrector = CenteringOnly()
        self._problem_time_total = 0.0
        self._cache = _PointCache()
        # Whether any finite lower/upper bound exists; refined in ``run`` from
        # the masks so bound-free problems skip the bound arithmetic entirely.
        self._has_lower = lower is not None
        self._has_upper = upper is not None
        self._linear_eq_data = self._normalize_linear_eq(problem.linear_eq())
        self._n = int(problem.n_vars)
        # Derivative provenance from the resolver (§3.3); surfaced in Result.
        self._sources: DerivativeSources = getattr(
            problem, "sources", DerivativeSources(gradient="analytic", hessian="exact")
        )
        self._has_analytic_hessian: bool = getattr(
            problem, "has_analytic_hessian", True
        )

    @staticmethod
    def _normalize_linear_eq(
        data: tuple[Array | LinearOperator, Array] | None,
    ) -> tuple[LinearOperator, Array] | None:
        if data is None:
            return None
        matrix, rhs = data
        return as_operator(matrix), rhs

    # -- model evaluation -------------------------------------------------

    def _time_problem_call(self, callback: Callable[[], T]) -> T:
        """Run a user problem callback and accumulate its wall-clock seconds."""
        start = perf_counter()
        try:
            return callback()
        finally:
            self._problem_time_total += perf_counter() - start

    def _cached(self, x: Array, key: str, compute: Callable[[], T]) -> T:
        """Memoize ``compute()`` at the point ``x`` (see :class:`_PointCache`)."""
        hit = self._cache.get(x, key)
        if hit is not _MISS:
            return hit  # type: ignore[no-any-return]
        value = compute()
        self._cache.put(x, key, value)
        return value

    def _gradient(self, x: Array) -> Array:
        return self._cached(
            x,
            "grad",
            lambda: self._time_problem_call(lambda: self._problem.gradient(x)),
        )

    def _objective(self, x: Array) -> float:
        return self._cached(
            x,
            "f",
            lambda: float(self._time_problem_call(lambda: self._problem.objective(x))),
        )

    def _ineq(self, x: Array) -> Array:
        if not self._has_ineq:
            return self._xp.zeros((0,), dtype=x.dtype)
        return self._cached(
            x,
            "g",
            lambda: self._time_problem_call(lambda: self._problem.ineq_constraints(x)),
        )

    def _ineq_jac(self, x: Array) -> LinearOperator:
        if not self._has_ineq:
            return Dense(self._xp.zeros((0, self._n), dtype=x.dtype))
        return self._cached(
            x,
            "ineq_jac",
            lambda: as_operator(
                self._time_problem_call(lambda: self._problem.ineq_jacobian(x))
            ),
        )

    def _eq(self, x: Array) -> Array:
        """Combined equalities ``c(x)``: nonlinear stacked with ``A x − b``."""
        if not self._has_eq and self._linear_eq_data is None:
            return self._xp.zeros((0,), dtype=x.dtype)
        return self._cached(x, "c", lambda: self._eq_uncached(x))

    def _eq_nonlinear(self, x: Array) -> Array:
        """Nonlinear equalities only (cached; the setup counts them separately)."""
        return self._cached(
            x,
            "c_nonlinear",
            lambda: self._time_problem_call(lambda: self._problem.eq_constraints(x)),
        )

    def _eq_uncached(self, x: Array) -> Array:
        xp = self._xp
        parts: list[Array] = []
        if self._has_eq:
            parts.append(self._eq_nonlinear(x))
        if self._linear_eq_data is not None:
            a_op, b = self._linear_eq_data
            parts.append(a_op.matvec(x) - b)
        if not parts:
            return xp.zeros((0,), dtype=x.dtype)
        if len(parts) == 1:
            return parts[0]
        return xp.concat(tuple(parts))

    def _eq_jac(self, x: Array) -> LinearOperator:
        """Combined equality Jacobian ``∇c(x)`` as a ``LinearOperator``."""
        if not self._has_eq and self._linear_eq_data is None:
            return Dense(self._xp.zeros((0, self._n), dtype=x.dtype))
        return self._cached(x, "eq_jac", lambda: self._eq_jac_uncached(x))

    def _eq_jac_uncached(self, x: Array) -> LinearOperator:
        ops: list[LinearOperator] = []
        if self._has_eq:
            jacobian = self._time_problem_call(lambda: self._problem.eq_jacobian(x))
            ops.append(as_operator(jacobian))
        if self._linear_eq_data is not None:
            ops.append(self._linear_eq_data[0])
        if not ops:
            return Dense(self._xp.zeros((0, self._n), dtype=x.dtype))
        if len(ops) == 1:
            return ops[0]
        return VStack(tuple(ops))

    def _lagrangian_gradient(
        self,
        grad: Array,
        ineq_jac: LinearOperator,
        eq_jac: LinearOperator,
        y_eq: Array,
        y_ineq: Array,
        m: int,
        m_eq: int,
    ) -> Array:
        """``∇_x L = ∇f + ∇cᵀ y_eq + ∇gᵀ y_ineq`` — the L-BFGS curvature source."""
        lag = grad
        if m_eq > 0:
            lag = lag + eq_jac.rmatvec(y_eq)
        if m > 0:
            lag = lag + ineq_jac.rmatvec(y_ineq)
        return lag

    def _hessian(
        self,
        x: Array,
        y_eq_nonlinear: Array,
        y_ineq: Array,
        lbfgs: LBFGSOperator,
    ) -> LinearOperator:
        """Lagrangian Hessian operator, by the resolved source (§3.2, §4.3).

        The source was decided once by ``derivatives.resolve`` (honoring
        ``options.hessian``) and recorded in ``self._sources.hessian``:

        - ``"exact"``: an analytic ``lagrangian_hessian``, with the current
          nonlinear equality/inequality multipliers (so ``W`` carries the
          constraint curvature ``Σ y·∇²c`` for nonconvex problems). Linear
          equalities are intentionally sliced away because their Hessian term is
          zero by contract.
        - ``"autodiff-hvp"``: exact Hessian-vector products of the Lagrangian via
          the backend autodiff adapter (no matrix formed).
        - ``"lbfgs"``: the persistent Powell-damped L-BFGS approximation, whose
          curvature pairs the driver updates each step.
        """
        if self._has_analytic_hessian:
            hessian = self._time_problem_call(
                lambda: self._problem.lagrangian_hessian(
                    x, y_eq_nonlinear, y_ineq, sigma=1.0
                )
            )
            return as_operator(hessian)
        if self._sources.hessian == "autodiff-hvp":
            return self._autodiff_hvp_operator(x, y_eq_nonlinear, y_ineq)
        if self._sources.hessian == "lbfgs":
            return lbfgs
        raise RuntimeError("no Hessian operator is available for this solve")

    def _autodiff_hvp_operator(
        self, x: Array, y_eq_nonlinear: Array, y_ineq: Array
    ) -> LinearOperator:
        """Matrix-free Hessian of the Lagrangian via autodiff HVPs (§4.3)."""
        adapter = get_autodiff_adapter(self._xp)
        if adapter is None:
            raise NotImplementedError(
                "autodiff-hvp Hessian requires a PyTorch/JAX backend"
            )
        xp = self._xp
        problem = self._problem
        has_eq = self._has_eq
        has_ineq = self._has_ineq

        def lagrangian(z: Array) -> Array:
            value = self._time_problem_call(lambda: problem.objective(z))
            if has_eq:
                constraints = self._time_problem_call(lambda: problem.eq_constraints(z))
                value = value + xp.sum(y_eq_nonlinear * constraints)
            if has_ineq:
                constraints = self._time_problem_call(
                    lambda: problem.ineq_constraints(z)
                )
                value = value + xp.sum(y_ineq * constraints)
            return value

        def matvec(v: Array) -> Array:
            return adapter.hvp(lagrangian, x, v)  # type: ignore[attr-defined]

        return MatrixFreeJacobian((self._n, self._n), matvec, matvec)

    # -- masks / bound bookkeeping ---------------------------------------

    def _masks(self, dtype: object) -> tuple[Array, Array, Array, Array]:
        xp = self._xp
        n = self._n
        if self._lower is None:
            mask_l = xp.zeros((n,), dtype=xp.bool)
            lower_safe = xp.zeros((n,), dtype=dtype)
        else:
            mask_l = xp.isfinite(self._lower)
            lower_safe = xp.where(mask_l, self._lower, xp.zeros((n,), dtype=dtype))
        if self._upper is None:
            mask_u = xp.zeros((n,), dtype=xp.bool)
            upper_safe = xp.zeros((n,), dtype=dtype)
        else:
            mask_u = xp.isfinite(self._upper)
            upper_safe = xp.where(mask_u, self._upper, xp.zeros((n,), dtype=dtype))
        return mask_l, mask_u, lower_safe, upper_safe

    # -- KKT error -------------------------------------------------------

    def kkt_error(
        self,
        *,
        grad: Array,
        ineq_jac: LinearOperator,
        m: int,
        g: Array,
        s: Array,
        y_ineq: Array,
        z_lower: Array,
        z_upper: Array,
        x_minus_l: Array,
        u_minus_x: Array,
        mask_l: Array,
        mask_u: Array,
        n_bounds: int,
        mu: float,
        c: Array,
        eq_jac: LinearOperator,
        m_eq: int,
        y_eq: Array,
        lagrangian_grad: Array | None = None,
    ) -> KKTResiduals:
        """Scaled KKT ∞-norm with IPOPT ``s_d, s_c`` scaling (§4.5).

        ``lagrangian_grad`` optionally supplies ``∇f + ∇cᵀy + ∇gᵀλ`` already
        formed by the caller (the L-BFGS curvature pair needs the same vector),
        saving the two Jacobian-transpose products.
        """
        xp = self._xp
        if lagrangian_grad is None:
            lagrangian_grad = self._lagrangian_gradient(
                grad, ineq_jac, eq_jac, y_eq, y_ineq, m, m_eq
            )
        r_d = lagrangian_grad
        if self._has_lower:
            r_d = r_d - z_lower
        if self._has_upper:
            r_d = r_d + z_upper
        dual_inf = _norm_inf(xp, r_d)

        # Each block's reduction stays on the device; one sync per quantity.
        prim_parts: list[Array] = []
        if m > 0:
            prim_parts.append(xp.max(xp.abs(g + s)))
        if m_eq > 0:
            prim_parts.append(xp.max(xp.abs(c)))
        prim = _max_of(xp, prim_parts)

        compl_parts: list[Array] = []
        if m > 0:
            compl_parts.append(xp.max(xp.abs(s * y_ineq - mu)))
        if self._has_lower:
            zero = xp.zeros_like(x_minus_l)
            compl_parts.append(
                xp.max(xp.abs(xp.where(mask_l, x_minus_l * z_lower - mu, zero)))
            )
        if self._has_upper:
            zero = xp.zeros_like(u_minus_x)
            compl_parts.append(
                xp.max(xp.abs(xp.where(mask_u, u_minus_x * z_upper - mu, zero)))
            )
        compl = _max_of(xp, compl_parts)

        n_dual = m + n_bounds + m_eq
        if n_dual > 0:
            dual_parts: list[Array] = []
            if m > 0:
                dual_parts.append(xp.sum(xp.abs(y_ineq)))
            if m_eq > 0:
                dual_parts.append(xp.sum(xp.abs(y_eq)))
            if self._has_lower:
                dual_parts.append(xp.sum(xp.abs(z_lower)))
            if self._has_upper:
                dual_parts.append(xp.sum(xp.abs(z_upper)))
            sum_dual = _sum_of(xp, dual_parts)
            s_d = max(_S_MAX, sum_dual / n_dual) / _S_MAX
            s_c = max(_S_MAX, sum_dual / n_dual) / _S_MAX
        else:
            s_d = 1.0
            s_c = 1.0
        return KKTResiduals(
            dual_infeasibility=dual_inf / s_d,
            primal_infeasibility=prim,
            complementarity=compl / s_c,
        )

    def _terminal_certificate(
        self,
        *,
        x: Array,
        g: Array,
        c: Array,
        s: Array,
        m: int,
        m_eq: int,
        mask_l: Array,
        mask_u: Array,
        n_bounds: int,
        x_minus_l: Array,
        u_minus_x: Array,
        recorded_primal: float,
        recorded_objective: float,
    ) -> tuple[KKTResiduals, Array, Array, Array, Array] | None:
        """Certify the returned iterate with multipliers it can justify.

        KKT satisfaction is an existence claim over the multipliers, but the
        per-iteration test (W&B 2006, eq. (5) ``E_0``) evaluates whatever
        duals the trajectory happens to carry. When those drifted — a
        rank-deficient ``∇c`` under-determines ``y``; μ-complementarity bound
        duals stay where initialization put them once the line search freezes
        — a certifiable point reports a large residual and the run ends in a
        failure status while parked at the answer.

        Candidate multipliers: the least-squares equality estimate (the duals
        the point itself justifies — exactly 0 on an objective-free system)
        and inequality/bound duals at the zero the ``E_0`` complementarity
        term demands at a strictly interior point (interiority of ``s`` and
        the bound gaps is maintained by the fraction-to-boundary rule, so
        zero duals are always sign-feasible here). A genuinely active
        constraint needs a nonzero multiplier, so its dropped dual leaves the
        raw gradient in the residual and the certificate *declines*:
        exhibited duals upper-bound the achievable dual infeasibility within
        the ``E_0`` test's own IPOPT ``s_d`` scaling semantics (§4.5) — the
        same scaled measure every in-loop test uses — so the certificate can
        under-certify but not certify a point the loop's own test would
        reject with the same multipliers.

        The gate is the acceptable-stopping family (IPOPT ``acceptable_tol``
        semantics — "what the caller is willing to accept"): every enabled
        residual tolerance plus ``f_tol`` must pass. ``dual_inf_tol`` is
        **required** to be enabled — with the candidate multipliers the
        complementarity residual is zero by construction, so without an
        enabled dual test a legal partial-``None`` configuration could
        certify a non-stationary point. ``f_rel_change_tol`` compares
        consecutive iterates and has no one-shot terminal analogue, so an
        enabled setting declines the certificate rather than silently
        weakening the caller's acceptance test.

        Returns ``(residuals, y_eq, y_ineq, z_lower, z_upper)`` on success,
        else ``None``.
        """
        opts = self._options.acceptable
        if opts.dual_inf_tol is None:
            return None
        if opts.f_rel_change_tol is not None:
            return None
        # Record-level early-outs before any problem evaluation: the primal
        # residual and the objective are independent of the multipliers, so a
        # returned iterate that already fails those tolerances cannot be
        # certified — and this method runs on the common failure exits.
        if opts.constr_viol_tol is not None and not (
            recorded_primal <= opts.constr_viol_tol
        ):
            return None
        if opts.f_tol is not None and not abs(recorded_objective) <= opts.f_tol:
            return None
        xp = self._xp
        dtype = x.dtype
        try:
            grad = self._gradient(x)
            eq_jac = self._eq_jac(x)
            ineq_jac = self._ineq_jac(x)
            y_c = (
                least_squares_duals(eq_jac, grad, xp=xp, m_eq=m_eq)
                if m_eq > 0
                else xp.zeros((0,), dtype=dtype)
            )
        except Exception:  # a certificate must never be the thing that fails
            logger.debug("terminal certificate: evaluation failed", exc_info=True)
            return None
        if m_eq > 0 and not bool(xp.all(xp.isfinite(y_c))):
            logger.debug("terminal certificate: non-finite dual estimate")
            return None
        lam_c = xp.zeros((m,), dtype=dtype)
        z_zero = xp.zeros_like(x)
        try:
            # Also best-effort: ``kkt_error`` applies the Jacobians (``rmatvec``
            # on a possibly lazy or user-supplied operator), so it can fail for
            # exactly the reasons the evaluation above can. Outside a guard it
            # would turn an ordinary STALLED/budget exit into an exception —
            # the certificate declining must never be worse than not having one.
            residuals = self.kkt_error(
                mu=0.0,
                grad=grad,
                ineq_jac=ineq_jac,
                m=m,
                g=g,
                s=s,
                y_ineq=lam_c,
                z_lower=z_zero,
                z_upper=z_zero,
                x_minus_l=x_minus_l,
                u_minus_x=u_minus_x,
                mask_l=mask_l,
                mask_u=mask_u,
                n_bounds=n_bounds,
                c=c,
                eq_jac=eq_jac,
                m_eq=m_eq,
                y_eq=y_c,
            )
        except Exception:
            logger.debug("terminal certificate: residual failed", exc_info=True)
            return None
        checks = (
            (opts.dual_inf_tol, residuals.dual_infeasibility),
            (opts.constr_viol_tol, residuals.primal_infeasibility),
            (opts.compl_inf_tol, residuals.complementarity),
        )
        for tol, value in checks:
            # ``not (value <= tol)`` rather than ``value > tol``: a NaN
            # residual must decline the certificate, not slip past it.
            if tol is not None and not value <= tol:
                return None
        # Distinct zero arrays for the two bound-dual blocks: returning one
        # shared object would alias ``Result.z_lower`` and ``Result.z_upper``.
        return residuals, y_c, lam_c, z_zero, xp.zeros_like(x)

    # -- main loop -------------------------------------------------------

    def run(self, x0: Array) -> Result:
        """Iterate to convergence and return the :class:`Result`."""
        run_start = perf_counter()
        xp = self._xp
        opts = self._options
        dtype = x0.dtype
        n = self._n

        mask_l, mask_u, lower_safe, upper_safe = self._masks(dtype)
        n_lower = int(xp.sum(xp.astype(mask_l, xp.int64)))
        n_upper = int(xp.sum(xp.astype(mask_u, xp.int64)))
        n_bounds = n_lower + n_upper
        self._has_lower = n_lower > 0
        self._has_upper = n_upper > 0

        m = int(self._ineq(x0).shape[0]) if self._has_ineq else 0
        if self._has_eq:
            m_nonlinear_eq = int(self._eq_nonlinear(x0).shape[0])
        else:
            m_nonlinear_eq = 0
        m_eq = int(self._eq(x0).shape[0])
        delta_c = opts.regularization.delta_c if m_eq > 0 else 0.0
        ineq_fn: Callable[[Array], Array] | None = self._ineq if m > 0 else None

        start = initialize(
            xp=xp,
            x0=x0,
            lower_safe=lower_safe,
            upper_safe=upper_safe,
            mask_l=mask_l,
            mask_u=mask_u,
            ineq_fn=ineq_fn,
            mu_init=opts.barrier.mu_init,
            m=m,
            slack_init_scale=opts.barrier.slack_init_scale,
        )
        x, s = start.x, start.s
        y_ineq, z_lower, z_upper = start.y_ineq, start.z_lower, start.z_upper
        y_eq = xp.zeros((m_eq,), dtype=dtype)
        if self._restoration_solver_factory is not None:
            self._check_restoration_adjoints(x, m, m_eq)

        if self._warm_start is not None:
            s, y_eq, y_ineq, z_lower, z_upper = apply_warm_start(
                xp=xp,
                warm=self._warm_start,
                s=s,
                y_eq=y_eq,
                y_ineq=y_ineq,
                z_lower=z_lower,
                z_upper=z_upper,
                m=m,
                m_eq=m_eq,
                n=n,
                mask_l=mask_l,
                mask_u=mask_u,
            )

        mu = opts.barrier.mu_init
        history: list[IterationRecord] = []
        alpha_p = 1.0
        alpha_d = 1.0
        # Last *accepted* combined steplength (Breedveld 2017, eq. (11)); the
        # steplength-driven μ schedule must not consume the α = 1 sentinel
        # above before a first step has actually been taken.
        last_alpha: float | None = None
        # Free-mode safeguard (NWW 2009, §5.1): suspends a non-monotone μ
        # oracle while the KKT error fails to make sufficient progress. Inert
        # for the monotone schedule (its μ handling is already monotone).
        mu_monitor = FreeModeMonitor(opts.barrier)
        free_mode = True
        # Stall detector: consecutive iterations with no accepted step and a
        # bit-frozen KKT error mean the loop is re-deriving the same rejected
        # direction — terminate honestly instead of burning the budget.
        stalled_iters = 0
        prev_e0: float | None = None
        # Best-iterate bookkeeping: the accepted iterate with the lowest scaled
        # KKT error, returned instead of the final wreckage when the run ends
        # in a failure status, plus the feasibility evidence for the
        # local-infeasibility veto (an accepted iterate below the verdict's own
        # believe-threshold contradicts any later "infeasible" claim).
        theta_best = float("inf")
        best_state: (
            tuple[IterationRecord, Array, Array, Array, Array, Array, Array] | None
        ) = None
        reg_applied = 0.0
        status = Status.MAX_ITER
        message = "maximum iterations reached"
        optimality = ConditionChecker.for_optimality(opts.optimality)
        acceptable = ConditionChecker.for_acceptable(opts.acceptable)

        filt = Filter()
        # μ the filter's φ entries were recorded at. The filter is
        # re-initialized whenever the barrier parameter changes — its φ_μ
        # coordinates are specific to that barrier problem (W&B 2006 filter
        # re-initialization on a barrier update; IPOPT ``MonotoneMuUpdate`` /
        # ``AdaptiveMuUpdate`` both call ``FilterLSAcceptor::Reset``; NWW 2009
        # §5: "the history in the filter [is] reset at every free iteration
        # because the barrier problem itself changes").
        filter_mu = mu
        # Free-mode acceptance filter (NWW 2009, §5 obj-constr variant): the
        # ``(θ, f)`` history of free-mode iterates, with the RAW objective —
        # unlike φ_μ it is comparable across the oracle's per-iteration μ
        # re-targets, which is why it (and not the W&B filter) is the
        # free-mode merit memory. Never cleared during the run (IPOPT clears
        # its globalization filter only at initialization); inert unless a
        # non-monotone μ oracle is in free mode with
        # ``LineSearchOptions.free_mode_acceptance="obj-constr-filter"``.
        free_filt = Filter()
        line_search = FilterLineSearch(opts.line_search)
        # eq. (18): θ_max guard, fixed from the initial constraint violation.
        theta0_init = self._theta_l1(x, s, m, m_eq)
        theta_max = _THETA_MAX_FACTOR * max(1.0, theta0_init)
        # eq. (23): θ_min, likewise fixed from the initial iterate.
        theta_min = _THETA_MIN_FACTOR * max(1.0, theta0_init)
        # Second-chance restoration anchor (S2MPJ 2026-07 audit): restoration
        # from a wandered-off iterate often converges to a nonzero LOCAL
        # minimizer of the infeasibility even though feasibility is directly
        # reachable from the user's starting point (28 of the 52 falsely
        # INFEASIBLE problems). A local-infeasibility claim therefore gets one
        # extra probe anchored here before it is believed.
        x_restore_anchor = x
        # Loop-persistent restoration bookkeeping (one-shot x0 probe + the θ of
        # the last uncertified stall), shared by the line-search and step-solve
        # restoration entry points via ``_handle_restoration``.
        rstate = _RestorationState()
        breedveld = BreedveldController(opts.breedveld)
        use_breedveld = opts.globalization == "breedveld"

        ones = xp.ones((n,), dtype=dtype)

        # Persistent L-BFGS history (§4.3). Curvature pairs are pushed at the
        # top of each iteration from the just-completed step, so the Hessian
        # used this iteration reflects every accepted step so far.
        lbfgs = LBFGSOperator(n, opts.lbfgs)
        use_lbfgs = (
            not self._has_analytic_hessian
        ) and self._sources.hessian == "lbfgs"
        prev_x: Array | None = None
        prev_grad: Array | None = None
        prev_ineq_jac: LinearOperator | None = None
        prev_eq_jac: LinearOperator | None = None
        problem_time_mark = self._problem_time_total
        last_step_solve_time = 0.0
        # Line-search backtracking count and restoration marker, both computed
        # at the tail of an iteration (once the search/restoration outcome is
        # known) and surfaced on the *next* row — mirroring how
        # ``last_step_solve_time`` reports the cost of the step that produced
        # that row.
        last_line_search_iters = 0
        pending_restored = False

        has_lower, has_upper = self._has_lower, self._has_upper

        def bound_gaps(x: Array) -> tuple[Array, Array]:
            x_minus_l = xp.where(mask_l, x - lower_safe, ones) if has_lower else ones
            u_minus_x = xp.where(mask_u, upper_safe - x, ones) if has_upper else ones
            return x_minus_l, u_minus_x

        # Count of logged rows, used to reprint the header periodically.
        rows_logged = 0

        for it in range(opts.max_iter + 1):
            self._step_solve_seconds = 0.0
            self._cache.retain(x)
            x_minus_l, u_minus_x = bound_gaps(x)
            g = self._ineq(x)
            c = self._eq(x)
            grad = self._gradient(x)
            ineq_jac = self._ineq_jac(x)
            eq_jac = self._eq_jac(x)

            # ``∇_x L`` at the current multipliers — shared by the KKT residual
            # and the L-BFGS curvature pair (one pair of Jᵀ products, not two).
            lagrangian_grad = self._lagrangian_gradient(
                grad, ineq_jac, eq_jac, y_eq, y_ineq, m, m_eq
            )
            if use_lbfgs:
                if prev_x is not None:
                    # prev_* are set together with prev_x at the loop tail.
                    assert prev_grad is not None
                    assert prev_ineq_jac is not None and prev_eq_jac is not None
                    delta = x - prev_x
                    gamma = lagrangian_grad - self._lagrangian_gradient(
                        prev_grad, prev_ineq_jac, prev_eq_jac, y_eq, y_ineq, m, m_eq
                    )
                    lbfgs.update(delta, gamma)
                prev_x, prev_grad = x, grad
                prev_ineq_jac, prev_eq_jac = ineq_jac, eq_jac

            err_kwargs = {
                "grad": grad,
                "ineq_jac": ineq_jac,
                "m": m,
                "g": g,
                "s": s,
                "y_ineq": y_ineq,
                "z_lower": z_lower,
                "z_upper": z_upper,
                "x_minus_l": x_minus_l,
                "u_minus_x": u_minus_x,
                "mask_l": mask_l,
                "mask_u": mask_u,
                "n_bounds": n_bounds,
                "c": c,
                "eq_jac": eq_jac,
                "m_eq": m_eq,
                "y_eq": y_eq,
            }
            residuals = self.kkt_error(
                mu=0.0, lagrangian_grad=lagrangian_grad, **err_kwargs
            )
            e0 = residuals.error
            # Feed the current outer KKT residual to the linear solver so an
            # iterative route can drive an inexact-Newton forcing sequence (loose
            # early, tight near convergence); direct solvers ignore it.
            self._solver.set_outer_residual(e0)
            if self._mu_schedule != "monotone":
                free_mode, entered_monotone = mu_monitor.observe(e0)
                if entered_monotone and m + n_bounds > 0:
                    # NWW 2009, §5.1: restart the monotone strategy from a
                    # fraction of the current complementarity — floored by the
                    # centrality condition (El-Bakry et al. 1996) on the
                    # *current* primal/dual residual, matching the free-mode
                    # paths: a collapsed complementarity must not re-enter
                    # monotone mode at a μ the decentered iterate cannot solve.
                    avg_compl, _ = complementarity_measures(
                        s=s,
                        y_ineq=y_ineq,
                        z_lower=z_lower,
                        z_upper=z_upper,
                        x_minus_l=x_minus_l,
                        u_minus_x=u_minus_x,
                        mask_l=mask_l,
                        mask_u=mask_u,
                        m=m,
                        n_bounds=n_bounds,
                    )
                    mu = fallback_mu(
                        avg_compl,
                        opts.barrier,
                        opts.optimality.kkt_tol,
                        infeasibility=max(
                            residuals.dual_infeasibility,
                            residuals.primal_infeasibility,
                        ),
                    )
                    logger.debug(
                        "iter %d: free-mode safeguard tripped (e0=%.3e); "
                        "monotone mode from mu=%.3e",
                        it,
                        e0,
                        mu,
                    )
            theta = self._theta(
                x, g, s, c, m, m_eq, mask_l, mask_u, lower_safe, upper_safe
            )
            objective = self._objective(x)
            record_problem_time = self._problem_time_total - problem_time_mark
            record = IterationRecord(
                iteration=it,
                objective=objective,
                mu=mu,
                theta=theta,
                kkt_error=e0,
                alpha_primal=alpha_p,
                alpha_dual=alpha_d,
                regularization=reg_applied,
                dual_infeasibility=residuals.dual_infeasibility,
                primal_infeasibility=residuals.primal_infeasibility,
                complementarity=residuals.complementarity,
                problem_time=record_problem_time,
                step_solve_time=last_step_solve_time,
                line_search_iters=last_line_search_iters,
                restored=pending_restored,
            )
            pending_restored = False
            if self._record_transform is not None:
                record = self._record_transform(record)
            problem_time_mark = self._problem_time_total
            history.append(record)
            theta_best = min(theta_best, theta)
            if best_state is None or e0 < best_state[0].kkt_error:
                best_state = (record, x, s, y_eq, y_ineq, z_lower, z_upper)
            if logger.isEnabledFor(ITERATION):
                if rows_logged % HEADER_REPEAT_INTERVAL == 0:
                    logger.log(ITERATION, format_header())
                logger.log(
                    ITERATION,
                    format_record(
                        record, acceptable=acceptable.conditions_hold(record)
                    ),
                )
                rows_logged += 1

            stop_requested = self._invoke_callback(
                record, x, s, y_eq, y_ineq, z_lower, z_upper, m, m_eq
            )

            decision = optimality.observe(record) or acceptable.observe(record)
            if decision is not None:
                status = decision.status
                message = decision.message
                break
            # IPOPT-style diverging-iterates test: an unbounded-below problem
            # drives ‖x‖ off to infinity while the objective diverges below
            # (e.g. INDEF: f → −1e155). BOTH signals are required — the
            # iterate norm alone false-positives on problems whose iterates
            # wander astronomically far before converging (S2MPJ KOEBHELB:
            # ‖x‖ grows monotonically past 1e22 across ~30 iterations, then
            # returns and converges to f = 112 — a *bounded-below* objective
            # throughout, which is exactly what "unbounded" must not claim).
            # Report honest UNBOUNDED rather than waiting for the runaway
            # iterate to overflow into a NUMERICAL_ERROR.
            if (
                opts.diverging_iterates_tol is not None
                and float(xp.max(xp.abs(x))) > opts.diverging_iterates_tol
                and float(objective) <= -opts.diverging_iterates_tol
            ):
                status = Status.UNBOUNDED
                message = (
                    "iterates and objective diverging; the problem appears "
                    "unbounded below"
                )
                break
            if (
                opts.max_time is not None
                and perf_counter() - run_start >= opts.max_time
            ):
                status = Status.MAX_TIME
                message = "stopped: maximum wall time reached"
                break
            if stop_requested:
                status = Status.STOPPED
                message = "stopped: iteration callback requested termination"
                break
            # A frozen iterate (zero accepted steplength + unchanged KKT error)
            # cannot recover by repetition: with identical state every derived
            # quantity — and thus the rejected direction — is identical too.
            # Genuine limit cycles (restoration jumps that keep *moving* the
            # iterate) change e0 and reset the counter, so they still run to
            # the ordinary budgets rather than being misreported here.
            frozen = (
                prev_e0 is not None
                and alpha_p == 0.0
                and abs(e0 - prev_e0) <= _STALL_REL_TOL * max(1.0, abs(prev_e0))
            )
            stalled_iters = stalled_iters + 1 if frozen else 0
            prev_e0 = e0
            if opts.max_stall_iter is not None and stalled_iters >= opts.max_stall_iter:
                if _within_relaxed_tol(opts.optimality, record):
                    status = Status.ACCEPTABLE
                    message = (
                        "acceptable: iteration stalled within the relaxed KKT tolerance"
                    )
                else:
                    status = Status.STALLED
                    message = (
                        f"stalled: no accepted step and no KKT-error progress "
                        f"for {stalled_iters} consecutive iterations"
                    )
                break
            if it == opts.max_iter:
                break

            sigma_s = y_ineq / s if m > 0 else xp.zeros((0,), dtype=dtype)
            zeros_n = xp.zeros_like(x)
            sigma_l = (
                xp.where(mask_l, z_lower / x_minus_l, zeros_n) if has_lower else zeros_n
            )
            sigma_u = (
                xp.where(mask_u, z_upper / u_minus_x, zeros_n) if has_upper else zeros_n
            )
            sigma_x = (
                sigma_l + sigma_u
                if (has_lower and has_upper)
                else (sigma_l if has_lower else sigma_u)
            )

            r_pi = g + s if m > 0 else xp.zeros((0,), dtype=dtype)
            w = self._hessian(x, y_eq[:m_nonlinear_eq], y_ineq, lbfgs)

            def solve_step_timed(
                w: LinearOperator,
                sigma_x: Array,
                sigma_s: Array,
                ineq_jac: LinearOperator,
                rhs_x: Array,
                eq_jac: LinearOperator,
                m_eq: int,
                r_y: Array,
                delta_c: float,
                delta_w_floor: float = 0.0,
            ) -> tuple[Array, Array, float, bool]:
                problem_before = self._problem_time_total
                start = perf_counter()
                result = self._solve_step(
                    w,
                    sigma_x,
                    sigma_s,
                    ineq_jac,
                    rhs_x,
                    eq_jac,
                    m_eq,
                    r_y,
                    delta_c,
                    delta_w_floor,
                )
                elapsed = perf_counter() - start
                problem_elapsed = self._problem_time_total - problem_before
                self._step_solve_seconds += max(0.0, elapsed - problem_elapsed)
                return result

            rhs_kwargs: dict[str, Any] = {
                "grad": grad,
                "s": s,
                "ineq_jac": ineq_jac,
                "eq_jac": eq_jac,
                "y_eq": y_eq,
                "sigma_s": sigma_s,
                "r_pi": r_pi,
                "mask_l": mask_l,
                "mask_u": mask_u,
                "x_minus_l": x_minus_l,
                "u_minus_x": u_minus_x,
                "m": m,
                "m_eq": m_eq,
            }
            recover_kwargs: dict[str, Any] = {
                "xp": xp,
                "ineq_jac": ineq_jac,
                "m": m,
                "s": s,
                "y_ineq": y_ineq,
                "r_pi": r_pi,
                "sigma_s": sigma_s,
                "z_lower": z_lower,
                "z_upper": z_upper,
                "sigma_l": sigma_l,
                "sigma_u": sigma_u,
                "x_minus_l": x_minus_l,
                "u_minus_x": u_minus_x,
                "mask_l": mask_l,
                "mask_u": mask_u,
                "has_lower": has_lower,
                "has_upper": has_upper,
            }

            step_solve_failed = False
            if self._corrector.active:
                # Mehrotra/Gondzio/probing: the affine and correction directions
                # share one KKT operator. Sparse-direct reuses its factorization;
                # dense/Krylov perform another solve against that operator.
                corrected = self._corrected_step(
                    w=w,
                    sigma_x=sigma_x,
                    sigma_s=sigma_s,
                    ineq_jac=ineq_jac,
                    eq_jac=eq_jac,
                    c=c,
                    delta_c=delta_c,
                    m_eq=m_eq,
                    solve_step_timed=solve_step_timed,
                    rhs_kwargs=rhs_kwargs,
                    recover_kwargs=recover_kwargs,
                    mu=mu,
                    last_alpha=last_alpha,
                    free_mode=free_mode,
                    dual_infeasibility=residuals.dual_infeasibility,
                    primal_infeasibility=residuals.primal_infeasibility,
                    err_kwargs=err_kwargs,
                )
                if corrected is None:
                    step_solve_failed = True
                else:
                    mu, step, reg_applied = corrected
                    # A step must have descent properties for the barrier problem
                    # at the current μ (NWW 2009, §5). A corrected direction built
                    # on a low-quality (quasi-Newton) affine probe can lose them —
                    # the complementarity-target perturbation is amplified through
                    # the near-singular condensed operator — leaving the line
                    # search only rejectable ascent trials. The corrector is a
                    # step-quality mechanism, not part of the μ selection, so fall
                    # back to the plain centered step toward the same μ.
                    if (
                        self._dphi(
                            x,
                            s,
                            step,
                            grad,
                            mu,
                            m,
                            mask_l,
                            mask_u,
                            x_minus_l,
                            u_minus_x,
                        )
                        >= 0.0
                    ):
                        rhs = self._condensed_rhs(mu, mu, mu, **rhs_kwargs)
                        dx, dy_eq, reg_fallback, ok = solve_step_timed(
                            w,
                            sigma_x,
                            sigma_s,
                            ineq_jac,
                            rhs,
                            eq_jac,
                            m_eq,
                            -c,
                            delta_c,
                        )
                        if ok:
                            step = recover_eliminated(
                                dx, mu=mu, dy_eq=dy_eq, **recover_kwargs
                            )
                            reg_applied = max(reg_applied, reg_fallback)
                        # else: the failed fallback ladder left the solver
                        # factored past the matrix that produced the (kept)
                        # corrected direction; ``_solve_step`` recorded that,
                        # so SOC will not re-solve against it.
            else:
                mu = self._next_mu(
                    mu,
                    last_alpha=last_alpha,
                    free_mode=free_mode,
                    infeasibility=max(
                        residuals.dual_infeasibility, residuals.primal_infeasibility
                    ),
                    **err_kwargs,
                )
                rhs = self._condensed_rhs(mu, mu, mu, **rhs_kwargs)
                dx, dy_eq, reg_applied, ok = solve_step_timed(
                    w, sigma_x, sigma_s, ineq_jac, rhs, eq_jac, m_eq, -c, delta_c
                )
                if not ok:
                    step_solve_failed = True
                else:
                    step = recover_eliminated(dx, mu=mu, dy_eq=dy_eq, **recover_kwargs)

            # Every μ-update path is complete for this iteration (fallback
            # re-entry, free-mode oracle, monotone reduction, or a recenter
            # escalation applied on the previous iteration's ``continue``):
            # re-initialize the filter if the barrier problem changed, before
            # anything consumes or augments it (see ``filter_mu`` above).
            if mu != filter_mu:
                filt.clear()
                filter_mu = mu

            if step_solve_failed:
                # Classify the failure: a solve that fails at an essentially
                # optimal iterate is salvaged ACCEPTABLE (ill-conditioning near
                # the solution), exactly as before. Otherwise, a Newton step the
                # δ_w ladder could not complete (δ_w driven past δ_w^max in the
                # inertia-correction algorithm, W&B 2006 §3.1) triggers the
                # feasibility restoration phase (§3.3) rather than an outright
                # numerical_error — the same globalization fallback a line-search
                # failure takes, so a diverging-multiplier feasibility system
                # (S2MPJ v11 exact-route *NE cluster) resolves to an honest
                # infeasibility verdict or is rescued, instead of crashing.
                status, message = _classify_step_failure(
                    self._options.optimality, record
                )
                if status is Status.ACCEPTABLE:
                    break
                theta0 = self._theta_l1(x, s, m, m_eq)
                phi0 = self._phi(x, s, mu, m, mask_l, mask_u, lower_safe, upper_safe)
                outcome = self._handle_restoration(
                    x=x,
                    s=s,
                    y_ineq=y_ineq,
                    y_eq=y_eq,
                    g=g,
                    mu=mu,
                    m=m,
                    m_eq=m_eq,
                    mask_l=mask_l,
                    mask_u=mask_u,
                    lower_safe=lower_safe,
                    upper_safe=upper_safe,
                    theta0=theta0,
                    theta_inf=self._theta_linf(x, s, m, m_eq),
                    phi0=phi0,
                    record=record,
                    filt=filt,
                    theta_best=theta_best,
                    x_restore_anchor=x_restore_anchor,
                    rstate=rstate,
                    it=it,
                )
                if not outcome.resume:
                    assert outcome.status is not None and outcome.message is not None
                    status, message = outcome.status, outcome.message
                    break
                assert (
                    outcome.x is not None
                    and outcome.s is not None
                    and outcome.y_ineq is not None
                )
                x, s, y_ineq = outcome.x, outcome.s, outcome.y_ineq
                if outcome.y_eq is not None:
                    y_eq = outcome.y_eq
                if outcome.mu is not None:
                    # Barrier escalation from a repeated feasible re-center;
                    # suspend the free-mode μ oracle or it would re-target μ
                    # from the stale complementarity on the next iteration.
                    mu = outcome.mu
                    mu_monitor.suspend(e0)
                alpha_p = 0.0
                last_alpha = 0.0
                prev_x = None
                last_step_solve_time = self._step_solve_seconds
                pending_restored = True
                continue

            # Descent enforcement at feasible iterates. At θ = 0 only
            # f-type (Armijo) acceptance exists (W&B 2006, §2.3), which no
            # ascent trial can pass — and an *iterative* KKT solve can return
            # a non-descent direction without failing (CG on an indefinite
            # condensed operator "succeeds" with garbage; success is not a
            # descent certificate, unlike the dense route's Cholesky PD-probe).
            # Left alone, the rejected step changes nothing and the identical
            # direction is recomputed forever (S2MPJ POWELLBSLS burned the
            # whole 10k budget this way). Escalating δ_w until the direction
            # is descent is the same inertia-correction response W&B 2006 §3.1
            # prescribe when the (1,1) block is not positive definite.
            theta0 = self._theta_l1(x, s, m, m_eq)
            phi0 = self._phi(x, s, mu, m, mask_l, mask_u, lower_safe, upper_safe)
            # The probe's directional derivative is the line search's ``dphi``
            # whenever the loop leaves the step unchanged; keep it to avoid a
            # second evaluation (and host sync).
            dphi_probe: float | None = None
            if theta0 == 0.0:
                ascent_noise = _DESCENT_NOISE_FACTOR * max(1.0, abs(phi0))
                descent_floor = reg_applied
                for _ in range(_MAX_DESCENT_ATTEMPTS):
                    dphi_probe = self._dphi(
                        x, s, step, grad, mu, m, mask_l, mask_u, x_minus_l, u_minus_x
                    )
                    if dphi_probe <= ascent_noise:
                        break
                    dphi_rejected = dphi_probe
                    dphi_probe = None  # the step is about to change
                    descent_floor = (
                        opts.regularization.delta_w_init
                        if descent_floor <= 0.0
                        else descent_floor * opts.regularization.delta_w_factor
                    )
                    if descent_floor > opts.regularization.delta_w_max:
                        break
                    logger.debug(
                        "iter %d: non-descent direction at feasible iterate "
                        "(dphi=%.3e); re-solving with delta_w >= %.2e",
                        it,
                        dphi_rejected,
                        descent_floor,
                    )
                    rhs = self._condensed_rhs(mu, mu, mu, **rhs_kwargs)
                    dx, dy_eq, reg_descent, ok = solve_step_timed(
                        w,
                        sigma_x,
                        sigma_s,
                        ineq_jac,
                        rhs,
                        eq_jac,
                        m_eq,
                        -c,
                        delta_c,
                        delta_w_floor=descent_floor,
                    )
                    if not ok:
                        # The failed escalation refactored the solver past the
                        # matrix that produced the (kept) earlier direction;
                        # ``_solve_step`` recorded that for the SOC gate.
                        break
                    step = recover_eliminated(dx, mu=mu, dy_eq=dy_eq, **recover_kwargs)
                    reg_applied = max(reg_applied, reg_descent)
                    descent_floor = max(descent_floor, reg_descent)

            if use_breedveld:
                tau = opts.breedveld.tau
            else:
                # Predictor-corrector targets can be far below machine epsilon
                # (or zero after an affine fallback). Use the configured barrier
                # floor for the boundary fraction so ``1 - μ`` never rounds to
                # exactly one and slacks/bound gaps remain strictly positive.
                mu_fraction = max(
                    mu, opts.barrier.mu_min, opts.optimality.kkt_tol / 10.0
                )
                tau = max(opts.barrier.tau_min, 1.0 - mu_fraction)
            alpha_p_max = self._alpha_primal(
                step, s, m, x_minus_l, u_minus_x, mask_l, mask_u, tau
            )
            alpha_d = self._alpha_dual(
                step, y_ineq, z_lower, z_upper, m, mask_l, mask_u, tau
            )

            # Trial points by step length, so the overshoot guard, the SOC and
            # the accepted-step update reuse the very array the line search
            # evaluated (the point cache is keyed by identity; see _PointCache).
            trial_points: dict[float, tuple[Array, Array]] = {}

            def trial_point(
                alpha: float,
                x: Array = x,
                s: Array = s,
                step: NewtonStep = step,
                trial_points: dict[float, tuple[Array, Array]] = trial_points,
            ) -> tuple[Array, Array]:
                point = trial_points.get(alpha)
                if point is None:
                    x_t = x + alpha * step.dx
                    s_t = s + alpha * step.ds if m > 0 else s
                    point = (x_t, s_t)
                    trial_points[alpha] = point
                return point

            # Bind the loop-varying state as defaults (the closure is consumed
            # within this iteration, but this keeps it explicit and lint-clean).
            def eval_point(
                alpha: float,
                mu: float = mu,
            ) -> tuple[float, float]:
                x_t, s_t = trial_point(alpha)
                return (
                    self._theta_l1(x_t, s_t, m, m_eq),
                    self._phi(x_t, s_t, mu, m, mask_l, mask_u, lower_safe, upper_safe),
                )

            def grad_finite(alpha: float) -> bool:
                """Whether the objective gradient at the trial point is finite.

                A quasi-Newton step can overshoot into a region where θ/φ are
                still finite but the derivatives overflow (inf/NaN) — accepting it
                poisons the next KKT solve. Used only on the L-BFGS route, where
                the overshoot occurs; the exact route's scaled steps do not need
                the extra gradient evaluation.
                """
                return bool(xp.all(xp.isfinite(self._gradient(trial_point(alpha)[0]))))

            def kkt_progress(
                alpha: float,
                x: Array = x,
                s: Array = s,
                step: NewtonStep = step,
                alpha_d: float = alpha_d,
                y_eq: Array = y_eq,
                y_ineq: Array = y_ineq,
                z_lower: Array = z_lower,
                z_upper: Array = z_upper,
                e0: float = e0,
            ) -> bool:
                """Certify scaled-KKT-error decrease at the trial point.

                The feasible-point rescue's acceptance certificate (see
                ``LineSearchOptions.feasible_kkt_progress``): evaluates the
                scaled KKT error at ``(x + α Δx, s + α Δs)`` with the duals at
                the iteration's fraction-to-boundary dual step — the state the
                driver would adopt on acceptance — and requires it to fall by
                the configured fraction of the current error ``e0``. Costs one
                gradient + Jacobian evaluation; consulted on the first trial
                of a feasible-point search only.
                """
                gamma_e = opts.line_search.feasible_kkt_progress
                assert gamma_e is not None  # closure only wired when enabled
                x_t, s_t = trial_point(alpha)
                x_minus_l_t, u_minus_x_t = bound_gaps(x_t)
                residuals_t = self.kkt_error(
                    mu=0.0,
                    grad=self._gradient(x_t),
                    ineq_jac=self._ineq_jac(x_t),
                    m=m,
                    g=self._ineq(x_t),
                    s=s_t,
                    y_ineq=y_ineq + alpha_d * step.dy_ineq,
                    z_lower=z_lower + alpha_d * step.dz_lower,
                    z_upper=z_upper + alpha_d * step.dz_upper,
                    x_minus_l=x_minus_l_t,
                    u_minus_x=u_minus_x_t,
                    mask_l=mask_l,
                    mask_u=mask_u,
                    n_bounds=n_bounds,
                    c=self._eq(x_t),
                    eq_jac=self._eq_jac(x_t),
                    m_eq=m_eq,
                    y_eq=y_eq + alpha_d * step.dy_eq,
                )
                e_t = residuals_t.error
                return math.isfinite(e_t) and e_t <= (1.0 - gamma_e) * e0

            soc_point: tuple[Array, Array] | None = None

            def is_strictly_interior(x_t: Array, s_t: Array) -> bool:
                if m > 0 and not bool(xp.all(s_t > 0.0)):
                    return False
                inside_l = xp.logical_or(xp.logical_not(mask_l), x_t > lower_safe)
                inside_u = xp.logical_or(xp.logical_not(mask_u), x_t < upper_safe)
                return bool(xp.all(xp.logical_and(inside_l, inside_u)))

            def soc(
                alpha: float,
                x: Array = x,
                s: Array = s,
                step: NewtonStep = step,
                ineq_jac: LinearOperator = ineq_jac,
                sigma_s: Array = sigma_s,
                mu: float = mu,
                w: LinearOperator = w,
                sigma_x: Array = sigma_x,
                eq_jac: LinearOperator = eq_jac,
            ) -> tuple[float, float] | None:
                """Second-order correction for nonlinear constraint residuals.

                While the solver's retained factorization is the iteration's
                *unregularized* matrix (``δ_w = 0``, base ``δ_c`` — the common
                case; ``_factor_unregularized``, recorded at factor time) each
                correction re-solves it: that matrix is exactly Wächter &
                Biegler 2006's choice (§2.4, eq. (26): "the same matrix as in
                (13)", "to avoid additional matrix factorizations"), so
                conformance and reuse coincide. (The right-hand side keeps
                ipax's pre-existing residual convention — ``c`` at the
                accumulated corrected point — rather than the eq. (27) blend.)

                Whenever the retained factorization is *regularized* (a
                ``δ_w > 0`` step, a δ_c-escalated saddle, a failed re-solve
                ladder, or a fallback solve inside this loop that ended
                regularized) ipax deliberately deviates and re-solves fresh at
                ``δ_w = 0``, as it always has: reusing the δ_w-inflated matrix
                degrades the feasibility correction enough to reroute whole
                runs — measured 2026-09-02, ZAMB2/ZAMB2m11 (exact/dense) and
                ACOPP30/TWIRIMD1 (lbfgs/dense) all left their baseline
                trajectories for restoration-heavy paths 10-60× more expensive
                per iteration when SOC used the step's δ_w. The fresh solve is
                also the fallback when the re-solve fails, so trajectories are
                unchanged relative to the pre-reuse code in every case.
                """
                nonlocal soc_point
                if opts.line_search.max_soc <= 0:
                    return None

                base_x, base_s = trial_point(alpha)
                corr_x = xp.zeros_like(x)
                corr_s = xp.zeros_like(s)
                empty_ineq = xp.zeros((0,), dtype=dtype)

                for _ in range(opts.line_search.max_soc):
                    x_c = base_x + corr_x
                    s_c = base_s + corr_s if m > 0 else s
                    if not is_strictly_interior(x_c, s_c):
                        return None

                    c_c = self._eq(x_c)
                    g_c = self._ineq(x_c)
                    r_pi_c = g_c + s_c if m > 0 else empty_ineq
                    rhs_soc = xp.zeros_like(step.dx)
                    if m > 0:
                        rhs_soc = rhs_soc - ineq_jac.rmatvec(sigma_s * r_pi_c)

                    sol_c = None
                    if self._factor_unregularized:
                        # W&B-exact fast path (see the docstring): the retained
                        # matrix IS the δ_w = 0 matrix here, so the re-solve is
                        # bitwise the fresh solve below, minus the rebuild and
                        # factorization. Read per round: the fresh solve below
                        # may leave the solver factored regularized.
                        sol_c = self._resolve_reused_factorization(rhs_soc, -c_c, m_eq)
                    if sol_c is not None:
                        dx_c = sol_c[: self._n]
                    else:
                        dx_c, _, _, ok = solve_step_timed(
                            w,
                            sigma_x,
                            sigma_s,
                            ineq_jac,
                            rhs_soc,
                            eq_jac,
                            m_eq,
                            -c_c,
                            delta_c,
                        )
                        if not ok:
                            return None

                    corr_x = corr_x + dx_c
                    if m > 0:
                        corr_s = corr_s - r_pi_c - ineq_jac.matvec(dx_c)

                x_soc = base_x + corr_x
                s_soc = base_s + corr_s if m > 0 else s
                if not is_strictly_interior(x_soc, s_soc):
                    return None
                # Reject a corrected trial whose derivatives overflow (see
                # ``grad_finite``): the SOC point has its own gradient, distinct
                # from ``x + α d``, so the line search cannot check it for us.
                if use_lbfgs and not bool(xp.all(xp.isfinite(self._gradient(x_soc)))):
                    return None

                soc_point = (x_soc, s_soc)
                return (
                    self._theta_l1(x_soc, s_soc, m, m_eq),
                    self._phi(
                        x_soc,
                        s_soc,
                        mu,
                        m,
                        mask_l,
                        mask_u,
                        lower_safe,
                        upper_safe,
                    ),
                )

            # θ0/φ0 were computed above for the descent enforcement; x, s and
            # μ are unchanged since.
            dphi = (
                dphi_probe
                if dphi_probe is not None
                else self._dphi(
                    x, s, step, grad, mu, m, mask_l, mask_u, x_minus_l, u_minus_x
                )
            )

            used_soc = False
            used_free_acceptance = False
            if use_breedveld:
                alpha_p, restoration, last_line_search_iters = breedveld.search(
                    alpha_max=alpha_p_max,
                    theta0=theta0,
                    phi0=phi0,
                    dphi=dphi,
                    eval_point=eval_point,
                    iteration=it,
                )
            elif (
                self._mu_schedule != "monotone"
                and free_mode
                and m + n_bounds > 0
                and opts.line_search.free_mode_acceptance == "obj-constr-filter"
            ):
                # Free-mode acceptance (NWW 2009, §5; see ``search_free``): the
                # oracle re-targets μ every iteration, so the weak (θ, f)
                # margin-filter test governs; the KKT-error monitor above trips
                # this branch off (``free_mode`` False ⇒ rigorous W&B search)
                # on stagnation. Remember the current — accepted — iterate
                # first, mirroring IPOPT's RememberCurrentPointAsAccepted: the
                # margins are what keep the vs-current comparison from
                # degenerating into a monotone requirement.
                def eval_point_free(alpha: float) -> tuple[float, float]:
                    x_t, s_t = trial_point(alpha)
                    return (self._theta_l1(x_t, s_t, m, m_eq), self._objective(x_t))

                used_free_acceptance = True
                free_filt.augment(theta0, objective)
                result = line_search.search_free(
                    alpha_max=alpha_p_max,
                    theta_max=theta_max,
                    eval_point=eval_point_free,
                    entries=free_filt.entries,
                    # IPOPT AdaptiveMuUpdate::CheckSufficientProgress margins.
                    margin=opts.line_search.free_filter_margin_fact
                    * min(opts.line_search.free_filter_max_margin, e0),
                    grad_finite=grad_finite if use_lbfgs else None,
                )
                alpha_p = result.alpha
                restoration = result.restoration
                last_line_search_iters = result.n_trials
            else:
                result = line_search.search(
                    alpha_max=alpha_p_max,
                    theta0=theta0,
                    phi0=phi0,
                    dphi=dphi,
                    theta_max=theta_max,
                    theta_min=theta_min,
                    eval_point=eval_point,
                    entries=filt.entries,
                    soc=soc,
                    grad_finite=grad_finite if use_lbfgs else None,
                    # Feasible-point rescue: only meaningful where the eq. (19)
                    # switching condition degenerates — numerically feasible
                    # iterates (round-off keeps θ0 at ~1e-16, not exactly 0);
                    # same feasibility notion as the re-center path. See
                    # LineSearchOptions.feasible_kkt_progress.
                    kkt_progress=(
                        kkt_progress
                        if (
                            opts.line_search.feasible_kkt_progress is not None
                            and theta0 <= feasible_theta_tol(opts.optimality.kkt_tol)
                        )
                        else None
                    ),
                )
                alpha_p = result.alpha
                restoration = result.restoration
                used_soc = result.used_soc
                last_line_search_iters = result.n_trials
                if result.accepted and result.augment:
                    filt.augment(theta0, phi0)

            if (
                restoration
                and used_free_acceptance
                and opts.barrier.fallback != "never"
            ):
                # A failed free-mode line search is the switch-to-monotone
                # trigger, not a restoration entry (NWW 2009, §5; IPOPT's
                # skipped-line-search signal in AdaptiveMuUpdate: sufficient
                # progress is denied, the mode flips, and the *rigorous* W&B
                # machinery retries from the same iterate — restoration stays
                # available to that path if it fails too). No step is taken;
                # μ is re-seeded like the monitor's own trip at the loop top
                # (monotone restart from the complementarity, floored by the
                # El-Bakry centrality condition).
                logger.debug(
                    "iter %d: free-mode line search failed; switching to "
                    "monotone mode (rigorous acceptance)",
                    it,
                )
                mu_monitor.suspend(e0)
                avg_compl, _ = complementarity_measures(
                    s=s,
                    y_ineq=y_ineq,
                    z_lower=z_lower,
                    z_upper=z_upper,
                    x_minus_l=x_minus_l,
                    u_minus_x=u_minus_x,
                    mask_l=mask_l,
                    mask_u=mask_u,
                    m=m,
                    n_bounds=n_bounds,
                )
                mu = fallback_mu(
                    avg_compl,
                    opts.barrier,
                    opts.optimality.kkt_tol,
                    infeasibility=max(
                        residuals.dual_infeasibility,
                        residuals.primal_infeasibility,
                    ),
                )
                alpha_p = 0.0
                last_alpha = 0.0
                prev_x = None
                last_step_solve_time = self._step_solve_seconds
                continue

            if restoration:
                outcome = self._handle_restoration(
                    x=x,
                    s=s,
                    y_ineq=y_ineq,
                    y_eq=y_eq,
                    g=g,
                    mu=mu,
                    m=m,
                    m_eq=m_eq,
                    mask_l=mask_l,
                    mask_u=mask_u,
                    lower_safe=lower_safe,
                    upper_safe=upper_safe,
                    theta0=theta0,
                    theta_inf=self._theta_linf(x, s, m, m_eq),
                    phi0=phi0,
                    record=record,
                    filt=filt,
                    theta_best=theta_best,
                    x_restore_anchor=x_restore_anchor,
                    rstate=rstate,
                    it=it,
                )
                if not outcome.resume:
                    assert outcome.status is not None and outcome.message is not None
                    status, message = outcome.status, outcome.message
                    break
                assert (
                    outcome.x is not None
                    and outcome.s is not None
                    and outcome.y_ineq is not None
                )
                x, s, y_ineq = outcome.x, outcome.s, outcome.y_ineq
                if outcome.y_eq is not None:
                    y_eq = outcome.y_eq
                if outcome.mu is not None:
                    # Barrier escalation from a repeated feasible re-center;
                    # suspend the free-mode μ oracle or it would re-target μ
                    # from the stale complementarity on the next iteration.
                    mu = outcome.mu
                    mu_monitor.suspend(e0)
                # A restoration jump / barrier re-center is a blocked Newton step:
                # σ(0) = 0.01 re-centers on the next iteration (Breedveld eq. (12)),
                # and the jump is not a Newton step, so the next curvature pair
                # would be meaningless — drop the history anchor.
                alpha_p = 0.0
                last_alpha = 0.0
                prev_x = None
                last_step_solve_time = self._step_solve_seconds
                pending_restored = True
                continue

            # Adopt the evaluated trial arrays themselves (bitwise the same
            # values as ``x + α Δx``) so the loop-top callbacks hit the cache.
            if used_soc and soc_point is not None:
                x, s_next = soc_point
            else:
                x, s_next = trial_point(alpha_p)
            if m > 0:
                s = s_next
                y_ineq = y_ineq + alpha_d * step.dy_ineq
            if m_eq > 0:
                y_eq = y_eq + alpha_p * step.dy_eq
                # A rank-deficient ∇c leaves this update free to inject
                # ‖c‖/δ_c of null-space noise that the (θ, φ) filter cannot
                # see. Opt-in, and gated on gross divergence so ordinary
                # primal-dual coupling is untouched.
                repair_factor = opts.regularization.equality_dual_repair
                if repair_factor is not None:
                    y_eq = self._repair_equality_duals(
                        x, y_eq, m_eq, factor=repair_factor
                    )
            z_lower = xp.where(
                mask_l, z_lower + alpha_d * step.dz_lower, xp.zeros_like(x)
            )
            z_upper = xp.where(
                mask_u, z_upper + alpha_d * step.dz_upper, xp.zeros_like(x)
            )
            # NOTE: the W&B 2006 eq. (16) κ_Σ dual clip was prototyped here and
            # measured on the S2MPJ false-infeasible subset (2026-07): it fixed
            # nothing (the DEGENLPA-class divergence lives in the *equality*
            # multipliers, which eq. (16) does not touch) and broke CRESC4
            # exact/krylov (optimal@59 → max_iter). Deliberately not shipped.
            # Combined steplength over all blocks (Breedveld 2017, eq. (11)),
            # feeding the σ(α) duality-gap reduction on the next iteration.
            last_alpha = min(alpha_p, alpha_d)
            last_step_solve_time = self._step_solve_seconds

        final_record = history[-1]
        # On a failure status, return the best accepted iterate instead of
        # whatever state the failing tail left behind (e.g. DEGENLPA's
        # diverged endgame after an essentially optimal iterate). Success
        # statuses keep the terminating iterate; UNBOUNDED keeps the diverging
        # one (it *is* the diagnosis), and a user STOP returns the current
        # point (least surprise).
        # ``<=`` matters: a restoration jump reassigns x *after* the final
        # record was written, so on a tie the snapshot restores a state
        # consistent with that record instead of the jumped-to point.
        if (
            best_state is not None
            and status in _FAILURE_RETURNS_BEST
            and best_state[0].kkt_error <= final_record.kkt_error
        ):
            final_record, x, s, y_eq, y_ineq, z_lower, z_upper = best_state
            message = (
                f"{message}; returning the best accepted iterate "
                f"(KKT {final_record.kkt_error:.3e} at iteration "
                f"{final_record.iteration})"
            )
        # Budget exhaustion — or a stall — at an essentially optimal returned
        # iterate reports ACCEPTABLE, mirroring the step-failure salvage
        # paths: a stall at KKT 6e-7 already reports ACCEPTABLE through the
        # relaxed tolerance, so running out of iterations or clock at the same
        # quality must not read as a harsher failure (S2MPJ budget-cluster
        # audit: DIAMON2DLS oscillates at the acceptable level without holding
        # it for the acceptable-iter window, then reported MAX_TIME from a
        # 6.7e-7 best iterate). STALLED is re-judged here too because the
        # in-loop stall check saw only the frozen *tail* iterate, while the
        # salvage above may have swapped in a better best iterate (S2MPJ
        # WEEDS: best 3.8e-7 — inside the relaxed band — frozen tail 6.2e-6).
        if status in (
            Status.MAX_ITER,
            Status.MAX_TIME,
            Status.STALLED,
        ) and _within_relaxed_tol(self._options.optimality, final_record):
            status = Status.ACCEPTABLE
            message = (
                "acceptable: the run ended at an iterate within the relaxed "
                f"KKT tolerance ({message})"
            )
        x_minus_l, u_minus_x = bound_gaps(x)
        g = self._ineq(x)
        c = self._eq(x)
        final_kkt = final_record.kkt_error
        final_dual_inf = final_record.dual_infeasibility
        final_primal_inf = final_record.primal_infeasibility
        final_compl = final_record.complementarity
        # Terminal KKT certificate (IPOPT-triage class B — "reached the
        # answer, won't certify"): a run can end STALLED/MAX_ITER/MAX_TIME
        # *at* an acceptable KKT point whose recorded residual is polluted by
        # multipliers the point never asked for — a rank-deficient ``∇c``
        # under-determines ``y`` (S2MPJ NONSCOMPNE: recorded KKT 6.8e-5 at a
        # point whose least-squares multipliers give dual infeasibility
        # exactly 0), and initialized bound duals never re-converge once the
        # line search freezes. KKT satisfaction is an existence claim over the
        # multipliers, so the returned iterate is re-judged with candidate
        # duals it can justify; exhibited duals only ever *upper-bound* the
        # achievable dual infeasibility, so this can under-certify but never
        # falsely certify. Terminal-only: the loop, restoration and rescue
        # paths are untouched (the 0.6.1 near-feasible-band precedent).
        if status in (Status.STALLED, Status.MAX_ITER, Status.MAX_TIME):
            certificate = self._terminal_certificate(
                x=x,
                g=g,
                c=c,
                s=s,
                m=m,
                m_eq=m_eq,
                mask_l=mask_l,
                mask_u=mask_u,
                n_bounds=n_bounds,
                x_minus_l=x_minus_l,
                u_minus_x=u_minus_x,
                recorded_primal=final_record.primal_infeasibility,
                recorded_objective=final_record.objective,
            )
            if certificate is not None:
                residuals_c, y_eq, y_ineq, z_lower, z_upper = certificate
                final_kkt = residuals_c.error
                final_dual_inf = residuals_c.dual_infeasibility
                final_primal_inf = residuals_c.primal_infeasibility
                final_compl = residuals_c.complementarity
                status = Status.ACCEPTABLE
                message = (
                    "acceptable: terminal KKT certificate at the returned "
                    f"iterate — repaired multipliers give dual infeasibility "
                    f"{final_dual_inf:.3e}, feasibility {final_primal_inf:.3e}, "
                    f"complementarity {final_compl:.3e} ({message})"
                )
        final_theta = self._theta(
            x, g, s, c, m, m_eq, mask_l, mask_u, lower_safe, upper_safe
        )

        # The human-readable result/timing summary is emitted by ``solve`` once
        # the solution has been unscaled; the driver only assembles the Result.
        logger.debug(
            "driver finished: status=%s after %d iteration(s)",
            status.value,
            len(history),
        )

        return Result(
            status=status,
            x=x,
            objective=self._objective(x),
            y_eq=y_eq if m_eq > 0 else None,
            y_ineq=y_ineq if m > 0 else None,
            z_lower=z_lower if self._lower is not None else None,
            z_upper=z_upper if self._upper is not None else None,
            n_iter=len(history),
            kkt_error=final_kkt,
            dual_infeasibility=final_dual_inf,
            primal_infeasibility=final_primal_inf,
            complementarity=final_compl,
            constraint_violation=final_theta,
            derivative_sources=self._sources,
            history=tuple(history),
            message=message,
        )

    # -- helpers ---------------------------------------------------------

    def _invoke_callback(
        self,
        record: IterationRecord,
        x: Array,
        s: Array,
        y_eq: Array,
        y_ineq: Array,
        z_lower: Array,
        z_upper: Array,
        m: int,
        m_eq: int,
    ) -> bool:
        """Call the user iteration hook; ``True`` means a stop was requested.

        Optional blocks collapse to ``None`` (matching :class:`Result`) so the
        callback only sees state that actually exists for this problem.
        """
        if self._callback is None:
            return False
        info = IterationInfo(
            record=record,
            x=x,
            s=s if m > 0 else None,
            y_eq=y_eq if m_eq > 0 else None,
            y_ineq=y_ineq if m > 0 else None,
            z_lower=z_lower if self._lower is not None else None,
            z_upper=z_upper if self._upper is not None else None,
        )
        return bool(self._callback(info))

    def _theta(
        self,
        x: Array,
        g: Array,
        s: Array,
        c: Array,
        m: int,
        m_eq: int,
        mask_l: Array,
        mask_u: Array,
        lower_safe: Array,
        upper_safe: Array,
    ) -> float:
        """Original-problem violation: ``c``, ``max(g, 0)`` and bound overshoot."""
        xp = self._xp
        parts: list[Array] = []
        if m > 0:
            parts.append(xp.max(xp.maximum(g, xp.zeros_like(g))))
        if m_eq > 0:
            parts.append(xp.max(xp.abs(c)))
        if self._has_lower or self._has_upper:
            zero = xp.zeros_like(x)
        if self._has_lower:
            parts.append(xp.max(xp.where(mask_l, lower_safe - x, zero)))
        if self._has_upper:
            parts.append(xp.max(xp.where(mask_u, x - upper_safe, zero)))
        # Bound overshoots are clamped at zero by the max with the empty case.
        return max(0.0, _max_of(xp, parts))

    def _next_mu(
        self,
        mu: float,
        *,
        last_alpha: float | None,
        free_mode: bool = True,
        infeasibility: float = 0.0,
        **err_kwargs: Any,
    ) -> float:
        """Advance μ by the configured schedule (``Options.mu_schedule``).

        ``"monotone"`` runs the guarded Fiacco–McCormick reduction loop below;
        the free-mode schedules instead re-target μ from the current
        complementarity on every iteration: ``"adaptive"`` via the LOQO
        centrality rule (Nocedal, Wächter & Waltz 2009, eqs. (3.1)/(3.6)),
        ``"breedveld"`` via the steplength-driven duality-gap reduction
        (Breedveld 2017, eqs. (10)–(12)). ``"probing"`` and ``"quality"``
        never reach this method in free mode — they need the affine direction,
        so they are applied on the corrector path (see ``_corrected_step``);
        with ``free_mode`` ``False`` they fall to the monotone reduction like
        every oracle. Problems without
        complementarity pairs — and, for the steplength rule, iterations before
        the first accepted step — fall back to keeping/reducing μ monotonically.
        With ``free_mode`` ``False`` (the NWW §5.1 safeguard tripped) every
        oracle is suspended in favor of the monotone reduction.
        """
        opts = self._options
        schedule = self._mu_schedule
        m = int(err_kwargs["m"])
        n_bounds = int(err_kwargs["n_bounds"])
        if not free_mode or schedule == "monotone" or m + n_bounds == 0:
            return self._reduce_mu(mu, **err_kwargs)
        if schedule == "breedveld" and last_alpha is None:
            # σ(α) needs an accepted steplength; keep μ_init for the first solve.
            return mu
        avg_compl, min_compl = complementarity_measures(
            s=err_kwargs["s"],
            y_ineq=err_kwargs["y_ineq"],
            z_lower=err_kwargs["z_lower"],
            z_upper=err_kwargs["z_upper"],
            x_minus_l=err_kwargs["x_minus_l"],
            u_minus_x=err_kwargs["u_minus_x"],
            mask_l=err_kwargs["mask_l"],
            mask_u=err_kwargs["mask_u"],
            m=m,
            n_bounds=n_bounds,
        )
        tol = opts.optimality.kkt_tol
        if schedule == "adaptive":
            mu_next = adaptive_mu(avg_compl, min_compl, opts.barrier, tol)
        else:
            # Only the steplength rule remains: probing/quality are routed to
            # the corrector path in free mode (guaranteed by the CenteringOnly
            # installation in __init__), monotone/suspended exited above.
            assert schedule == "breedveld" and last_alpha is not None
            mu_next = breedveld_mu(avg_compl, last_alpha, opts.barrier, tol)
        # Centrality floor (El-Bakry et al. 1996): μ must not vanish faster
        # than the primal/dual infeasibility, or the barrier decenters at a
        # still-unsolved iterate (see BarrierOptions.kappa_centrality).
        return max(mu_next, opts.barrier.kappa_centrality * infeasibility)

    def _reduce_mu(self, mu: float, **err_kwargs: object) -> float:
        opts = self._options
        for _ in range(_MAX_MU_REDUCTIONS):
            e_mu = self.kkt_error(mu=mu, **err_kwargs).error  # type: ignore[arg-type]
            if mu <= opts.barrier.mu_min or e_mu > _KAPPA_EPSILON * mu:
                return mu
            new_mu = update_mu(mu, opts.barrier, opts.optimality.kkt_tol)
            if new_mu >= mu:
                return mu
            mu = new_mu
        return mu

    # -- condensed RHS / corrector primitives ----------------------------

    def _condensed_rhs(
        self,
        target_s: Array | float,
        target_l: Array | float,
        target_u: Array | float,
        *,
        grad: Array,
        s: Array,
        ineq_jac: LinearOperator,
        eq_jac: LinearOperator,
        y_eq: Array,
        sigma_s: Array,
        r_pi: Array,
        mask_l: Array,
        mask_u: Array,
        x_minus_l: Array,
        u_minus_x: Array,
        m: int,
        m_eq: int,
    ) -> Array:
        """Right-hand side of the condensed x-system for complementarity targets.

        ``target_*`` is the per-component complementarity target ``τ`` (scalar
        ``μ`` for the standard step; vectors for the corrector). The terms
        ``τ/s``, ``τ_L/(x−x_L)``, ``τ_U/(x_U−x)`` are the only places ``τ``
        enters the condensed reduction (§2.3).
        """
        xp = self._xp
        rhs = -grad
        if m > 0:
            rhs = rhs - ineq_jac.rmatvec(sigma_s * r_pi + target_s / s)
        if m_eq > 0:
            rhs = rhs - eq_jac.rmatvec(y_eq)
        if self._has_lower:
            rhs = rhs + xp.where(mask_l, target_l / x_minus_l, xp.zeros_like(grad))
        if self._has_upper:
            rhs = rhs - xp.where(mask_u, target_u / u_minus_x, xp.zeros_like(grad))
        return rhs

    def _resolve_reused_factorization(
        self, rhs_x: Array, r_y: Array, m_eq: int
    ) -> Array | None:
        """Re-solve the most recently factored KKT system for a new RHS.

        Second-order corrections and the centrality correctors solve the same
        linear system as the search direction, only with a different
        right-hand side; Wächter & Biegler 2006 (§2.4, eq. (26)) use "the
        same matrix as in (13)" — same ``δ_w``/``δ_c``, same factorization —
        explicitly "to avoid additional matrix factorizations". The
        sparse-direct route back-solves its LDLᵀ; the dense route
        back-substitutes the retained Cholesky factor (an LU re-solve where
        the backend lacks the triangular-solve gap-filler); the Krylov route
        reuses the operator — the memoized Woodbury factors on structured
        bound-only L-BFGS blocks, a fresh iterative solve otherwise. Returns
        ``None`` on a failed or non-finite solve — these corrections are
        opportunistic, so the caller simply proceeds without one.
        """
        xp = self._xp
        rhs = xp.concat((rhs_x, r_y)) if m_eq > 0 else rhs_x
        # Same accounting as ``solve_step_timed``: a matrix-free re-solve can
        # call back into problem derivatives (autodiff-HVP matvecs), whose
        # time is already charged to the problem — don't double-count it.
        problem_before = self._problem_time_total
        start = perf_counter()
        try:
            sol = self._solver.solve(rhs)
        except LinearSolveError:
            sol = None
        elapsed = perf_counter() - start
        problem_elapsed = self._problem_time_total - problem_before
        self._step_solve_seconds += max(0.0, elapsed - problem_elapsed)
        if sol is None or not bool(xp.all(xp.isfinite(sol))):
            return None
        return sol

    def _solve_targets_reused_operator(
        self,
        comp_s: Array,
        comp_l: Array,
        comp_u: Array,
        *,
        c: Array,
        m_eq: int,
        rhs_kwargs: dict[str, Any],
        recover_kwargs: dict[str, Any],
    ) -> NewtonStep | None:
        """Solve the current condensed system for new complementarity targets.

        A :meth:`_resolve_reused_factorization` on the system established by
        the affine solve. Returns ``None`` on a failed/non-finite solve so
        the corrector can fall back to the affine step.
        """
        xp = self._xp
        rhs_x = self._condensed_rhs(comp_s, comp_l, comp_u, **rhs_kwargs)
        sol = self._resolve_reused_factorization(rhs_x, -c, m_eq)
        if sol is None:
            return None
        dx_t = sol[: self._n]
        dy_t = sol[self._n :] if m_eq > 0 else xp.zeros((0,), dtype=rhs_x.dtype)
        return recover_eliminated(
            dx_t,
            mu=0.0,
            comp_s=comp_s,
            comp_l=comp_l,
            comp_u=comp_u,
            dy_eq=dy_t,
            **recover_kwargs,
        )

    def _corrected_step(
        self,
        *,
        w: LinearOperator,
        sigma_x: Array,
        sigma_s: Array,
        ineq_jac: LinearOperator,
        eq_jac: LinearOperator,
        c: Array,
        delta_c: float,
        m_eq: int,
        solve_step_timed: Callable[..., tuple[Array, Array, float, bool]],
        rhs_kwargs: dict[str, Any],
        recover_kwargs: dict[str, Any],
        mu: float,
        last_alpha: float | None,
        free_mode: bool,
        dual_infeasibility: float,
        primal_infeasibility: float,
        err_kwargs: dict[str, Any],
    ) -> tuple[float, NewtonStep, float] | None:
        """Affine predictor + μ oracle + higher-order corrector.

        The affine direction doubles as the probe for the ``"probing"`` and
        ``"quality"`` oracles (the latter also scores the centering family it
        spans — NWW 2009, §3.3); any other ``mu_schedule`` picks the target via
        ``_next_mu`` and the corrector merely aims at it (Nocedal, Wächter &
        Waltz 2009: the corrector is not part of the barrier-parameter
        selection). Returns ``(μ, step, δ_w)`` where ``μ`` is the barrier
        target the corrected ``step`` aims at (adopted by the line search), or
        ``None`` if the affine operator setup/solve failed despite
        regularization.
        """
        infeasibility = max(dual_infeasibility, primal_infeasibility)
        affine_rhs = self._condensed_rhs(0.0, 0.0, 0.0, **rhs_kwargs)
        dx_aff, dy_aff, reg_applied, ok = solve_step_timed(
            w, sigma_x, sigma_s, ineq_jac, affine_rhs, eq_jac, m_eq, -c, delta_c
        )
        if not ok:
            return None
        affine_step = recover_eliminated(dx_aff, mu=0.0, dy_eq=dy_aff, **recover_kwargs)
        solve_targets = functools.partial(
            self._solve_targets_reused_operator,
            c=c,
            m_eq=m_eq,
            rhs_kwargs=rhs_kwargs,
            recover_kwargs=recover_kwargs,
        )
        alpha_primal = functools.partial(
            self._alpha_primal,
            s=recover_kwargs["s"],
            m=recover_kwargs["m"],
            x_minus_l=recover_kwargs["x_minus_l"],
            u_minus_x=recover_kwargs["u_minus_x"],
            mask_l=recover_kwargs["mask_l"],
            mask_u=recover_kwargs["mask_u"],
        )
        alpha_dual = functools.partial(
            self._alpha_dual,
            y_ineq=recover_kwargs["y_ineq"],
            z_lower=recover_kwargs["z_lower"],
            z_upper=recover_kwargs["z_upper"],
            m=recover_kwargs["m"],
            mask_l=recover_kwargs["mask_l"],
            mask_u=recover_kwargs["mask_u"],
        )
        context = CorrectionContext(
            affine=affine_step,
            s=recover_kwargs["s"],
            y_ineq=recover_kwargs["y_ineq"],
            x_minus_l=recover_kwargs["x_minus_l"],
            u_minus_x=recover_kwargs["u_minus_x"],
            z_lower=recover_kwargs["z_lower"],
            z_upper=recover_kwargs["z_upper"],
            mask_l=recover_kwargs["mask_l"],
            mask_u=recover_kwargs["mask_u"],
            solve=solve_targets,
            alpha_primal=alpha_primal,
            alpha_dual=alpha_dual,
            mu_min=max(
                self._options.barrier.mu_min, self._options.optimality.kkt_tol / 10.0
            ),
        )
        if free_mode and self._mu_schedule in ("probing", "quality"):
            oracle_mu = (
                quality_mu(context, dual_infeasibility, primal_infeasibility)
                if self._mu_schedule == "quality"
                else probing_mu(context)
            )
            # Centrality floor (El-Bakry et al. 1996) — see BarrierOptions.
            mu_target = max(
                oracle_mu,
                self._options.barrier.kappa_centrality * infeasibility,
            )
        else:
            mu_target = self._next_mu(
                mu,
                last_alpha=last_alpha,
                free_mode=free_mode,
                infeasibility=infeasibility,
                **err_kwargs,
            )
        result = self._corrector.correct(context, mu_target)
        return result.mu, result.step, reg_applied

    def _inertia_acceptable(self, operator: LinearOperator) -> bool:
        """IPOPT inertia check on a successful sparse LDLᵀ factorization.

        Best-effort and solver-agnostic (invariant #3): it only engages when the
        injected solver reports the factor's inertia (a sparse LDLᵀ backend such
        as Feral / cuDSS) *and* the KKT operator knows its target inertia. For a
        diagonal-plus-low-rank (L-BFGS) Hessian the target folds the compact
        ``M``-block signature into the bordered inertia (see
        :meth:`~ipax.ipm.kkt._CondensedOperator.expected_inertia`), so the check
        now covers that route too. In every other case — dense Cholesky / Krylov
        (no inertia, but Cholesky already fails on indefiniteness), a
        non-inertia-revealing factorization, or a singular/oversized ``M`` block
        (no reliable target) — it returns ``True`` and leaves correction to the
        factorization-failure escalation.
        """
        # Ask the solver first: dense/Krylov solvers never report an inertia,
        # and the target below costs an eigensolve of the L-BFGS middle block.
        inertia_fn = getattr(self._solver, "inertia_or_none", None)
        actual = inertia_fn() if inertia_fn is not None else None
        if actual is None:
            return True
        target_fn = getattr(operator, "expected_inertia", None)
        target = target_fn() if target_fn is not None else None
        if target is None:
            return True
        return bool(actual == target)

    def _solve_step(
        self,
        w: LinearOperator,
        sigma_x: Array,
        sigma_s: Array,
        ineq_jac: LinearOperator,
        rhs_x: Array,
        eq_jac: LinearOperator,
        m_eq: int,
        r_y: Array,
        delta_c: float,
        delta_w_floor: float = 0.0,
    ) -> tuple[Array, Array, float, bool]:
        """Factor/solve the condensed (or bordered saddle) step (§2.3, §4.4).

        Returns ``(Δx, Δy_eq, δ_w, ok)``. With equalities present the condensed
        block is bordered into the quasidefinite saddle (Friedlander–Orban) and
        solved through the same injected ``LinearSolver``; Breedveld δ_w
        escalation handles a failed factorization. ``delta_w_floor`` starts the
        ladder above zero — the descent-enforcement re-solve uses it to demand
        a more strongly regularized (more convex) system than the last attempt.
        """
        xp = self._xp
        reg = RegularizationState(delta_w=delta_w_floor)
        sigma_x_op = Diagonal(sigma_x)
        sigma_s_op = Diagonal(sigma_s)
        empty = xp.zeros((0,), dtype=rhs_x.dtype)
        # δ_c is escalated on a failed saddle solve only as a *last resort*: δ_w
        # cannot repair a rank-deficient equality Jacobian (which leaves the
        # saddle singular in the (2,2) dual block), only δ_c can (W&B 2006,
        # §3.1). It is threaded locally, so it grows only within this failing
        # solve.
        current_delta_c = delta_c

        reg_opts = self._options.regularization
        # Two-phase ladder. Phase 1 escalates δ_w alone: an indefinite (1,1)
        # block — including one carrying exploded-multiplier curvature ~1e8–1e9
        # (HS61) — is repaired by δ_w by itself, and any δ_c mixed into that
        # repair distorts the dual step exactly when it must correct the
        # multipliers (HS61 cycled forever on a (2.3e9, 2.6e-3) step where the
        # (2.3e9, 1e-8) step recovered). Only once δ_w has grown past
        # `delta_c_trigger` without success is the failure attributed to the
        # dual block: phase 2 then *resets δ_w to the floor* — a huge δ_w left
        # in place would poison the very solve δ_c is meant to rescue — and
        # escalates both from small values.
        in_dual_phase = False

        def escalate() -> None:
            nonlocal current_delta_c, in_dual_phase
            if (
                m_eq > 0
                and not in_dual_phase
                and reg.delta_w >= reg_opts.delta_c_trigger
            ):
                in_dual_phase = True
                reg.delta_w = 0.0  # restart the δ_w ladder at its floor
                current_delta_c = escalate_delta_c(current_delta_c, reg_opts)
                return
            escalate_delta_w(reg, reg_opts)
            if in_dual_phase:
                current_delta_c = escalate_delta_c(current_delta_c, reg_opts)

        for _ in range(_MAX_REG_ATTEMPTS):
            condensed = build_condensed_operator(
                w,
                sigma_x_op,
                sigma_s_op,
                ineq_jac,
                reg,
                sigma_x_zero=not (self._has_lower or self._has_upper),
            )
            if m_eq > 0:
                operator = build_saddle_operator(condensed, eq_jac, current_delta_c)
                rhs = xp.concat((rhs_x, r_y))
            else:
                operator = condensed
                rhs = rhs_x
            try:
                self._solver.factor(operator)
                sol = self._solver.solve(rhs)
            except LinearSolveError:
                escalate()
                logger.debug(
                    "factorization failed; escalating delta_w to %.2e, delta_c to %.2e",
                    reg.delta_w,
                    current_delta_c,
                )
                continue
            if not self._inertia_acceptable(operator):
                # A symmetric-indefinite LDLᵀ can succeed with the *wrong* inertia
                # (a non-descent step the failure path never sees); IPOPT bumps
                # δ_w until the (1,1) block is PD (Wächter & Biegler 2006, §3.1).
                escalate()
                logger.debug(
                    "KKT inertia mismatch; escalating delta_w to %.2e", reg.delta_w
                )
                continue
            if bool(xp.all(xp.isfinite(sol))):
                dx = sol[: self._n]
                dy = sol[self._n :] if m_eq > 0 else empty
                # The returned δ_w alone does not identify the retained matrix:
                # a phase-2 success reports δ_w = 0 with an escalated δ_c.
                self._factor_unregularized = (
                    reg.delta_w == 0.0 and current_delta_c == delta_c
                )
                return dx, dy, reg.delta_w, True
            escalate()
            logger.debug("non-finite step; escalating delta_w to %.2e", reg.delta_w)
        self._factor_unregularized = False  # left factored past the ladder's end
        return rhs_x, empty, reg.delta_w, False

    def _alpha_primal(
        self,
        step: NewtonStep,
        s: Array,
        m: int,
        x_minus_l: Array,
        u_minus_x: Array,
        mask_l: Array,
        mask_u: Array,
        tau: float,
    ) -> float:
        xp = self._xp
        parts: list[Array] = []
        if m > 0:
            parts.append(boundary_ratio(s, step.ds, tau))
        if self._has_lower:
            v_l = xp.where(mask_l, x_minus_l, xp.ones_like(x_minus_l))
            dv_l = xp.where(mask_l, step.dx, xp.zeros_like(x_minus_l))
            parts.append(boundary_ratio(v_l, dv_l, tau))
        if self._has_upper:
            v_u = xp.where(mask_u, u_minus_x, xp.ones_like(u_minus_x))
            dv_u = xp.where(mask_u, -step.dx, xp.zeros_like(u_minus_x))
            parts.append(boundary_ratio(v_u, dv_u, tau))
        return _clamped_min(xp, parts)

    def _alpha_dual(
        self,
        step: NewtonStep,
        y_ineq: Array,
        z_lower: Array,
        z_upper: Array,
        m: int,
        mask_l: Array,
        mask_u: Array,
        tau: float,
    ) -> float:
        xp = self._xp
        parts: list[Array] = []
        if m > 0:
            parts.append(boundary_ratio(y_ineq, step.dy_ineq, tau))
        if self._has_lower:
            v_l = xp.where(mask_l, z_lower, xp.ones_like(z_lower))
            dv_l = xp.where(mask_l, step.dz_lower, xp.zeros_like(z_lower))
            parts.append(boundary_ratio(v_l, dv_l, tau))
        if self._has_upper:
            v_u = xp.where(mask_u, z_upper, xp.ones_like(z_upper))
            dv_u = xp.where(mask_u, step.dz_upper, xp.zeros_like(z_upper))
            parts.append(boundary_ratio(v_u, dv_u, tau))
        return _clamped_min(xp, parts)

    def _phi(
        self,
        x: Array,
        s: Array,
        mu: float,
        m: int,
        mask_l: Array,
        mask_u: Array,
        lower_safe: Array,
        upper_safe: Array,
    ) -> float:
        """Barrier objective ``φ_μ`` (Wächter & Biegler 2006, §2)."""
        xp = self._xp
        val = self._objective(x)
        parts: list[Array] = []
        if m > 0:
            parts.append(xp.sum(xp.log(s)))
        if self._has_lower:
            x_minus_l = xp.where(mask_l, x - lower_safe, xp.ones_like(x))
            parts.append(xp.sum(xp.where(mask_l, xp.log(x_minus_l), xp.zeros_like(x))))
        if self._has_upper:
            u_minus_x = xp.where(mask_u, upper_safe - x, xp.ones_like(x))
            parts.append(xp.sum(xp.where(mask_u, xp.log(u_minus_x), xp.zeros_like(x))))
        return val - mu * _sum_of(xp, parts)

    def _theta_l1(self, x: Array, s: Array, m: int, m_eq: int) -> float:
        """Constraint violation ``θ = ‖(c, g+s)‖₁`` used by the filter."""
        xp = self._xp
        parts: list[Array] = []
        if m > 0:
            parts.append(xp.sum(xp.abs(self._ineq(x) + s)))
        if m_eq > 0:
            parts.append(xp.sum(xp.abs(self._eq(x))))
        val = _sum_of(xp, parts)
        return val

    def _theta_linf(self, x: Array, s: Array, m: int, m_eq: int) -> float:
        """Constraint violation ``θ = ‖(c, g+s)‖_∞`` — restoration's own measure.

        The feasible-point guard must predict whether :func:`restore` would
        no-op, so it has to ask the question the way restoration answers it.
        The filter's ``θ`` is the ℓ1 norm, which on a wide problem exceeds the
        tolerance that the ℓ∞ norm passes: S2MPJ ``DRCAVTY1`` sums 100 residuals
        to ``1.2e-8`` where ``max|c|`` is ``7.1e-10``, so the guard declared the
        point infeasible and handed it to a restoration that immediately
        declared it feasible and returned it unchanged.
        """
        xp = self._xp
        parts: list[Array] = []
        if m > 0:
            parts.append(xp.max(xp.abs(self._ineq(x) + s)))
        if m_eq > 0:
            parts.append(xp.max(xp.abs(self._eq(x))))
        return _max_of(xp, parts)

    def _dphi(
        self,
        x: Array,
        s: Array,
        step: NewtonStep,
        grad: Array,
        mu: float,
        m: int,
        mask_l: Array,
        mask_u: Array,
        x_minus_l: Array,
        u_minus_x: Array,
    ) -> float:
        """Directional derivative ``∇φ_μᵀ d`` along the step."""
        xp = self._xp
        parts: list[Array] = [xp.sum(grad * step.dx)]
        if m > 0:
            parts.append(-mu * xp.sum(step.ds / s))
        if self._has_lower:
            parts.append(
                -mu * xp.sum(xp.where(mask_l, step.dx / x_minus_l, xp.zeros_like(x)))
            )
        if self._has_upper:
            parts.append(
                mu * xp.sum(xp.where(mask_u, step.dx / u_minus_x, xp.zeros_like(x)))
            )
        return _sum_of(xp, parts)

    def _repair_equality_duals(
        self, x: Array, y_eq: Array, m_eq: int, *, factor: float = 1.0
    ) -> Array:
        """Repair the equality multipliers at a feasible-point failure.

        The counterpart of :func:`recenter_slacks_duals`, which clips the
        *inequality* multipliers into the central band (W&B eq. (16)). An
        equality-only problem has no slacks to re-floor, so without this the
        re-center is a no-op and the driver re-derives the same rejected
        direction (S2MPJ ``COOLHANS``: ``‖y_eq‖∞`` drifts to 1.1e6 on a problem
        whose objective is identically zero, which forces ``y* = 0``).

        The estimate is the least-squares one, ``argmin_y ‖∇f + Aᵀy‖`` — the
        multipliers this iterate can actually justify — solved matrix-free so
        the sparse and matrix-free routes are unaffected (see
        :func:`least_squares_duals`). On an objective-free system it returns
        exactly zero, which is the correct multiplier there.

        Measured on the 12 class-A reproducers: 0/12 reach feasibility without
        this repair, 6/12 with it (METHANL8, HYDCAR6, DRCAVTY1, DRCAVTY3,
        CYCLOOCF, ROBOT, all to 1e-9 or better). A drifted estimate is only
        replaced when the new one is finite and actually closer to
        stationarity, so a repair can never make the iterate worse.

        ``factor`` scales the estimate's residual in that comparison, so the
        repair fires only when the current multipliers are worse by more than
        this ratio. The default ``1.0`` *is* the "strictly closer" rule above —
        the restoration path's contract. The accepted-step call site
        (``RegularizationOptions.equality_dual_repair``) passes a large factor so
        only genuine null-space divergence is corrected; see that option for why
        the gate has to be conservative.
        """
        if m_eq == 0:
            return y_eq
        xp = self._xp
        try:
            eq_jac = self._eq_jac(x)
            grad = self._gradient(x)
            estimate = least_squares_duals(eq_jac, grad, xp=xp, m_eq=m_eq)
        except Exception:  # a repair must never be the thing that fails
            return y_eq
        if not bool(xp.all(xp.isfinite(estimate))):
            return y_eq

        def _stationarity(mult: Array) -> float:
            return float(xp.max(xp.abs(grad + eq_jac.rmatvec(mult))))

        if _stationarity(y_eq) <= factor * _stationarity(estimate):
            return y_eq
        logger.debug(
            "iter: repairing equality multipliers, ||y||inf %.2e -> %.2e",
            float(xp.max(xp.abs(y_eq))) if y_eq.shape[0] else 0.0,
            float(xp.max(xp.abs(estimate))) if estimate.shape[0] else 0.0,
        )
        return estimate

    def _handle_restoration(
        self,
        *,
        x: Array,
        s: Array,
        y_ineq: Array,
        y_eq: Array,
        g: Array,
        mu: float,
        m: int,
        m_eq: int,
        mask_l: Array,
        mask_u: Array,
        lower_safe: Array,
        upper_safe: Array,
        theta0: float,
        theta_inf: float,
        phi0: float,
        record: IterationRecord,
        filt: Filter,
        theta_best: float,
        x_restore_anchor: Array,
        rstate: _RestorationState,
        it: int,
    ) -> _RestorationOutcome:
        """Recover a globalization failure by re-centering or restoration.

        Shared by the two failures that cannot make progress with the current
        barrier state: a filter line search that hands off to restoration, and
        a step solve the inertia correction (δ_w ladder) could not complete
        (Wächter & Biegler 2006: §3.1 escalates δ_w and reverts to the §3.3
        restoration phase once it exceeds δ_w^max — a failed regularization is
        a restoration trigger, not an outright failure). Returns an outcome the
        caller applies: ``resume`` restarts the main loop from the updated
        ``(x, s, y_ineq)``; otherwise the run terminates with the returned
        ``status``/``message``.
        """
        xp = self._xp
        opts = self._options
        # A stall at an already near-optimal iterate (ill-conditioning near the
        # solution) is accepted rather than restored, to avoid a false
        # "infeasible".
        if _within_relaxed_tol(opts.optimality, record):
            return _RestorationOutcome(
                resume=False,
                status=Status.ACCEPTABLE,
                message=(
                    "acceptable: globalization stalled at a point within the "
                    "relaxed KKT tolerance"
                ),
            )
        # Restoration entered at an already-feasible point cannot move it:
        # ``restore()`` exits immediately at the same ``x``, and resuming with
        # the stale barrier state re-derives the same rejected direction forever
        # (S2MPJ v11, HS101 exact routes: boundary-floor slacks against
        # multipliers grown to ~1e6 give Σ_s ~ 1e18, δ_w escalates to ~1e5 every
        # iteration, and the fraction-to-boundary rule caps every step at
        # ~1e-11 — a limit cycle the stall detector ends at an *infeasible* best
        # iterate). Repair the barrier state instead: re-floor the slacks on the
        # current μ and clip the multipliers to the central band (Wächter &
        # Biegler 2006, §3.3 / eq. (16)).
        #
        # The test is deliberately independent of whether the problem *has*
        # inequality constraints. Re-centering repairs slacks, so requiring
        # ``m > 0`` looked natural when this guard was written for the HS101
        # class (inequalities, no equalities) — but the harm being prevented is
        # entering a restoration that cannot move the iterate, and that does not
        # depend on slacks existing. On the equality-only CUTEst systems the
        # guard never fired and the driver livelocked exactly as described above
        # (S2MPJ v3: METHANL8/DRCAVTY1/CYCLOOCF each entered restoration 26
        # times, every one exiting FEASIBLE at an unchanged ``x``, and reported
        # `stalled` at a violation IPOPT clears in a handful of iterations).
        # ``m + m_eq > 0`` keeps the guard to problems that *have* barrier state
        # tied to constraints. With neither block there is nothing to re-center
        # and nothing for restoration to do either, so the branch would only
        # choose between two no-ops — and it chose worse: S2MPJ ``HADAMALS``
        # (bounds only, so θ ≡ 0 and this fires on every failure) went from
        # `optimal` to `stalled` at a worse objective on two routes.
        if m + m_eq > 0 and theta_inf <= feasible_theta_tol(opts.optimality.kkt_tol):
            logger.debug(
                "iter %d: globalization failed at a feasible point; "
                "re-centering slacks/duals instead of restoration",
                it,
            )
            filt.augment(theta0, phi0)
            rstate.feasible_recenters += 1
            # A *repeated* feasible re-center means repairing slacks/duals at
            # the current μ already failed once — the barrier problem itself is
            # what cannot be solved. Raise μ to the scale of the stall's KKT
            # error (capped at mu_init, raised by at least the escalation
            # factor); the caller applies it and suspends any free-mode μ
            # oracle. See _RECENTER_MU_RAISE_FACTOR for the full rationale.
            mu_raised: float | None = None
            if rstate.feasible_recenters >= 2:
                target = min(
                    opts.barrier.mu_init,
                    max(
                        _RECENTER_MU_RAISE_FACTOR * mu,
                        _RECENTER_MU_ERROR_FRACTION * record.kkt_error,
                    ),
                )
                if target > mu:
                    mu_raised = target
                    logger.debug(
                        "iter %d: repeated feasible re-center; raising the "
                        "barrier mu from %.2e to %.2e",
                        it,
                        mu,
                        mu_raised,
                    )
            # ``g`` is the inequality residual at this iterate, computed at the
            # top of the loop; the failure did not move ``x``, so it is current.
            # Re-center on the μ the loop will actually resume with, so the
            # slack floor and the multiplier central band match that barrier.
            mu_eff = mu_raised if mu_raised is not None else mu
            s, y_ineq = recenter_slacks_duals(xp, g, y_ineq, mu_eff)
            y_eq = self._repair_equality_duals(x, y_eq, m_eq)
            return _RestorationOutcome(
                resume=True, x=x, s=s, y_ineq=y_ineq, y_eq=y_eq, mu=mu_raised
            )
        logger.debug("iter %d: entering feasibility restoration", it)
        filt.augment(theta0, phi0)
        x, s, rest_exit = self._restore(
            x, s, m, m_eq, mask_l, mask_u, lower_safe, upper_safe
        )
        theta_restored = self._theta_l1(x, s, m, m_eq)
        still_infeasible = _restoration_reports_infeasible(
            theta_restored, opts.optimality
        )
        # A local-infeasibility *claim* needs both the driver's own θ to be
        # genuinely large (restoration's raw ℓ∞ measure can stall marginally
        # above its differently scaled tolerance at a point that is feasible
        # here — S2MPJ Task 1) AND a stationarity-type exit. An *uncertified*
        # stall (window/budget exit) is not evidence of infeasibility (S2MPJ
        # restfix audit: LAKES/NASH/SWOPF were honest out-of-budget failures
        # relabeled "infeasible" by the early window exit): it resumes the main
        # loop while restoration keeps reducing θ between stalls, and a repeat
        # without progress terminates below — after the one-shot x0 probe had
        # its say — as RESTORATION_FAILED, never as INFEASIBLE.
        claim_certified = still_infeasible and rest_exit.certifies_infeasibility
        uncertified_stall = still_infeasible and not rest_exit.certifies_infeasibility
        stall_progressed = (
            theta_restored
            <= _RESTORATION_PROGRESS_FACTOR * rstate.uncertified_stall_theta
        )
        if uncertified_stall and stall_progressed:
            rstate.uncertified_stall_theta = theta_restored
        if claim_certified or (uncertified_stall and not stall_progressed):
            # Second chance: restoration is a LOCAL method, so from a wandered-off
            # iterate it can converge to a nonzero local minimizer of the
            # infeasibility on a perfectly feasible problem (S2MPJ 2026-07 audit:
            # 28 of 52 falsely INFEASIBLE problems are restorable directly from
            # x0). Probe once from the starting point before terminating on
            # either a certified claim or an exhausted uncertified stall; a
            # believable feasible outcome resumes the main loop there.
            if not rstate.second_chance_used:
                rstate.second_chance_used = True
                logger.debug(
                    "iter %d: probing the infeasibility claim with a "
                    "restoration anchored at the starting point",
                    it,
                )
                x_r, s_r, exit_r = self._restore(
                    x_restore_anchor,
                    s,
                    m,
                    m_eq,
                    mask_l,
                    mask_u,
                    lower_safe,
                    upper_safe,
                )
                theta_probe = self._theta_l1(x_r, s_r, m, m_eq)
                if not (
                    exit_r.certifies_infeasibility
                    and _restoration_reports_infeasible(theta_probe, opts.optimality)
                ):
                    # A probe that stayed infeasible is itself the next stall
                    # baseline; a believably feasible rescue is a fresh basin, so
                    # the stall streak restarts.
                    rstate.uncertified_stall_theta = (
                        theta_probe
                        if _restoration_reports_infeasible(theta_probe, opts.optimality)
                        else float("inf")
                    )
                    return _RestorationOutcome(resume=True, x=x_r, s=s_r, y_ineq=y_ineq)
                # The anchored probe itself reached a stationary point of the
                # infeasibility: a certificate even when the triggering stall
                # carried none.
                claim_certified = True
            if not claim_certified:
                return _RestorationOutcome(
                    resume=False,
                    status=Status.RESTORATION_FAILED,
                    message=(
                        "restoration failed: the infeasibility minimization "
                        "stalled without a stationarity certificate and without "
                        f"reducing the constraint violation (theta={theta_restored:.3e})"
                    ),
                )
            # Veto: a local-infeasibility claim is contradicted by the run's own
            # history whenever an *accepted* iterate already reached the
            # near-feasible band — a diverged endgame (degenerate duals, tiny μ)
            # is a stall at a feasible problem, not evidence of infeasibility
            # (S2MPJ v10: DEGENLPA reached θ = 1.7e-7 before the collapse that
            # used to be reported as INFEASIBLE; v11 item 3: the ARGAUSS/LANCZOS/
            # MISRA1B NLS cluster reaches an accepted θ ~ 5e-5 before restoration
            # floors just above it). Uses the ~1e-4 near-feasible band, not the
            # tighter believe-threshold: it fires only after the resume/second-
            # chance path, so a rescuable problem (HS111) never reaches it.
            if theta_best <= _near_feasible_tol(opts.optimality):
                return _RestorationOutcome(
                    resume=False,
                    status=Status.STALLED,
                    message=(
                        "stalled: restoration could not re-reduce the constraint "
                        f"violation, but an accepted iterate already reached "
                        f"theta={theta_best:.3e} — the problem is not locally "
                        "infeasible"
                    ),
                )
            # Terminal near-feasible downgrade: a certified "infeasible" point
            # that is itself feasible to IPOPT's constr_viol_tol band (~1e-4) is
            # not distinguishable from a degenerate near-feasible optimum, so
            # report the honest STALLED instead of a wrong "locally infeasible"
            # (S2MPJ item 3: LEWISPOL floors at θ ~ 1e-5, the float64 limit).
            # This is checked only here, after the resume/second-chance path has
            # already run at the tighter threshold, so it never reroutes a
            # rescuable problem — it only softens the final verdict.
            if theta_restored <= _near_feasible_tol(opts.optimality):
                return _RestorationOutcome(
                    resume=False,
                    status=Status.STALLED,
                    message=(
                        "stalled: restoration floored at a near-feasible point "
                        f"(theta={theta_restored:.3e}, within the ~1e-4 "
                        "feasibility band) it could not improve — the problem is "
                        "not locally infeasible"
                    ),
                )
            return _RestorationOutcome(
                resume=False,
                status=Status.INFEASIBLE,
                message=(
                    "locally infeasible: restoration could not reduce the "
                    "constraint violation"
                ),
            )
        return _RestorationOutcome(resume=True, x=x, s=s, y_ineq=y_ineq)

    def _restore(
        self,
        x: Array,
        s: Array,
        m: int,
        m_eq: int,
        mask_l: Array,
        mask_u: Array,
        lower_safe: Array,
        upper_safe: Array,
    ) -> tuple[Array, Array, RestorationExit]:
        """Run the Gauss-Newton restoration phase (delegates to ``restore``)."""
        return restore(
            xp=self._xp,
            x=x,
            s=s,
            m=m,
            m_eq=m_eq,
            eq_fn=self._eq,
            eq_jac_fn=self._eq_jac,
            ineq_fn=self._ineq,
            ineq_jac_fn=self._ineq_jac,
            mask_l=mask_l,
            mask_u=mask_u,
            lower_safe=lower_safe,
            upper_safe=upper_safe,
            tol=self._options.optimality.kkt_tol,
            linear_solver=(
                None
                if self._restoration_solver_factory is None
                else self._restoration_solver_factory()
            ),
        )

    def _check_restoration_adjoints(self, x: Array, m: int, m_eq: int) -> None:
        """Refuse a matvec-only Jacobian at entry, not at the first restoration.

        The iterative restoration route applies ``Jᵀ Σ J`` through ``matvec``
        and ``rmatvec``; a Jacobian without an adjoint would otherwise abort the
        run — with no ``Result`` — hundreds of iterations in, when restoration
        is first entered. One adjoint product with a zero vector per Jacobian
        at the first iterate (whose Jacobians the point cache reuses) is the
        price.
        """
        xp = self._xp
        probes: list[tuple[str, LinearOperator, int]] = []
        if m_eq > 0:
            probes.append(("equality", self._eq_jac(x), m_eq))
        if m > 0:
            probes.append(("inequality", self._ineq_jac(x), m))
        for name, jacobian, rows in probes:
            try:
                jacobian.rmatvec(xp.zeros((rows,), dtype=x.dtype))
            except NotImplementedError as exc:
                raise ValueError(
                    "RestorationOptions(linear_solver='krylov') applies the "
                    f"{name} Jacobian through matvec and rmatvec, but this "
                    f"problem's {name} Jacobian has no adjoint: {exc}"
                ) from exc


__all__ = ["IPMDriver"]
