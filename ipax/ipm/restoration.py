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

"""Feasibility restoration phase (Wächter & Biegler §3.3, §4.1).

Entered when the filter line search cannot find an acceptable ``α ≥ α_min``.
This is a **damped Gauss-Newton / Levenberg–Marquardt** minimization of the
ℓ2 constraint infeasibility

    F(x) = ½‖c(x)‖² + ½‖max(g(x), 0)‖²

(the slacks are recovered as ``s = max(-g(x), floor)`` afterwards, which is the
minimizer of ``‖g+s‖`` over ``s ≥ 0``). It returns the reached point together
with a :class:`RestorationExit` reason; only a stationarity-type exit with
``θ`` above tolerance is evidence of *local infeasibility* — a stall or an
exhausted budget merely says the minimization gave up.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

from ipax.backend.operators import LinearOperator, VStack
from ipax.linalg.solver import LinearSolveError

if TYPE_CHECKING:
    from ipax.linalg.solver import LinearSolver
    from ipax.typing import Array, Namespace

_MAX_ITER = 200  # outer (Jacobian-rebuild) iterations; stalls exit via the window
_STALL_WINDOW = 12  # exit when f fails to improve by _STALL_RTOL over this window
_STALL_RTOL = 1e-3
_LM_INIT = 1e-8  # Levenberg–Marquardt damping seed
_LM_GROW = 10.0
_LM_SHRINK = 0.1
_LM_MAX = 1e16  # ceiling on the damping before declaring no further progress
_GRAD_TOL = 1e-10  # stationarity test for the infeasibility objective
_SLACK_FLOOR = 1e-12
# One-shot escape probe before certifying local infeasibility (see
# ``_escape_direction``): relative size of the kick, and the golden angle that
# makes the fixed probe direction aperiodic in every component.
_ESCAPE_SCALE = 1e-6
_GOLDEN_ANGLE = 2.399963229728653


class RestorationExit(Enum):
    """How the infeasibility minimization ended.

    Only :attr:`STATIONARY` and :attr:`NO_DESCENT` are certificates of local
    infeasibility (a first-order stationary point of the bound-constrained
    infeasibility with ``θ > 0``; Wächter & Biegler 2006, §3.3). A window or
    budget exit is a mere stall: the S2MPJ restfix audit (2026-07) showed the
    trailing-window guard exiting *early* on slow problems (LAKES/NASH/SWOPF),
    and treating that as an infeasibility verdict relabels an honest
    out-of-budget failure as a false claim about the problem. Likewise an
    iterative inner solve that never produced a finite direction
    (:attr:`LINEAR_SOLVE_FAILED`) says nothing about local feasibility.
    """

    FEASIBLE = "feasible"  # θ_∞ reached the feasibility tolerance
    STATIONARY = "stationary"  # projected-gradient stationary point of F
    NO_DESCENT = "no_descent"  # no LM damping in [init, max] yields descent
    LINEAR_SOLVE_FAILED = "linear_solve_failed"  # no finite LM direction was obtained
    STALL_WINDOW = "stall_window"  # trailing-window plateau (uncertified)
    BUDGET = "budget"  # iteration budget exhausted (uncertified)

    @property
    def certifies_infeasibility(self) -> bool:
        """Whether this exit is first-order evidence of local infeasibility."""
        return self in (RestorationExit.STATIONARY, RestorationExit.NO_DESCENT)


def feasible_theta_tol(tol: float) -> float:
    """The θ threshold below which restoration considers a point feasible.

    Shared with the driver's feasible-entry guard: a line-search failure at a
    point already below this threshold must not enter restoration at all
    (``restore()`` would exit immediately at the same ``x``), it re-centers
    the barrier state instead.
    """
    return max(tol, 1e-8)


def _dense(op: LinearOperator, xp: Namespace, dtype: object) -> Array:
    return op.matmat(xp.eye(op.shape[1], dtype=dtype))


class _RestorationNormalOperator(LinearOperator):
    """Reduced damped Gauss-Newton matrix ``F (Jᵀ Σ J) F + B + λI`` by products.

    ``J`` stacks the equality and inequality Jacobians and ``Σ`` their row
    weights (ones for equalities, the ``g > 0`` indicator for inequalities);
    ``F``/``B`` are the free/blocked 0-1 masks of the projected-Newton
    reduction. One instance serves a whole outer iteration: only ``damping``
    moves along the Levenberg–Marquardt ladder, so the Jacobi diagonal is
    computed once and re-shifted per rung.
    """

    def __init__(
        self,
        *,
        jacobian: LinearOperator,
        weights: Array,
        free: Array,
        blocked: Array,
        damping: float,
    ) -> None:
        self._jacobian = jacobian
        self._weights = weights
        self._free = free
        self._blocked = blocked
        self.damping = damping
        self._reduced_diagonal: Array | None = None

    @property
    def shape(self) -> tuple[int, int]:
        n = int(self._free.shape[0])
        return n, n

    def matvec(self, v: Array) -> Array:
        jv = self._jacobian.matvec(self._free * v)
        gram_v = self._jacobian.rmatvec(self._weights * jv)
        return self._free * gram_v + (self._blocked + self.damping) * v

    def diagonal(self, like: Array | None = None) -> Array:
        # ``gram_diagonal`` raises NotImplementedError for Jacobians without a
        # cheap column-energy; that propagates, and the Krylov Jacobi
        # preconditioner treats a missing diagonal as "no preconditioning".
        del like
        if self._reduced_diagonal is None:
            gram_diagonal = self._jacobian.gram_diagonal(self._weights)
            self._reduced_diagonal = self._free * gram_diagonal + self._blocked
        return self._reduced_diagonal + self.damping


def _escape_direction(xp: Namespace, x: Array) -> Array:
    """A fixed, component-wise aperiodic probe direction for the escape kick.

    The first-order certificate is blind to saddles: on a symmetry-invariant
    subspace (equal components at a permutation-symmetric start such as S2MPJ
    POWERSUMNE/HADAMARD, a cyclic ring in CYCLOOCT) every Gauss-Newton direction
    stays in the subspace, and a critical point of ½‖c‖² restricted to it is a
    critical point of the full infeasibility (Palais 1979, principle of
    symmetric criticality) — a saddle, not local infeasibility. An exact solve
    never leaves the subspace; the dense LU reference only did so through
    round-off in near-null directions amplified by the rank deficiency. Before
    certifying, ``restore`` therefore kicks ``x`` once along this deterministic,
    backend-agnostic direction and lets the damped Gauss-Newton loop continue:
    a saddle's unstable manifold amplifies the kick, a true local minimizer
    re-certifies at the same point.
    """
    k = xp.arange(int(x.shape[0]), dtype=x.dtype)
    return xp.sin(1.0 + _GOLDEN_ANGLE * k)


def _stacked_jacobian(
    xp: Namespace,
    eq_jac: LinearOperator | None,
    c: Array,
    ineq_jac: LinearOperator | None,
    gpos: Array,
    active: Array | None,
) -> tuple[LinearOperator, Array, Array] | None:
    """Stack ``(J, Σ, r)`` of the infeasibility model ``½‖Σ^½ r‖²``, or None.

    Returns the stacked Jacobian, its row weights and the residual it acts on
    (``c`` for equalities, the positive part of ``g`` for inequalities).
    """
    if eq_jac is not None and ineq_jac is not None:
        assert active is not None
        return (
            VStack((eq_jac, ineq_jac)),
            xp.concat((xp.ones_like(c), active)),
            xp.concat((c, gpos)),
        )
    if eq_jac is not None:
        return eq_jac, xp.ones_like(c), c
    if ineq_jac is not None:
        assert active is not None
        return ineq_jac, active, gpos
    return None


def restore(
    *,
    xp: Namespace,
    x: Array,
    s: Array,
    m: int,
    m_eq: int,
    eq_fn: Callable[[Array], Array],
    eq_jac_fn: Callable[[Array], LinearOperator],
    ineq_fn: Callable[[Array], Array],
    ineq_jac_fn: Callable[[Array], LinearOperator],
    mask_l: Array,
    mask_u: Array,
    lower_safe: Array,
    upper_safe: Array,
    tol: float,
    linear_solver: LinearSolver | None = None,
) -> tuple[Array, Array, RestorationExit]:
    """Minimize the constraint infeasibility; return ``(x, s, exit_reason)``."""
    dtype = x.dtype
    n = int(x.shape[0])
    identity = xp.eye(n, dtype=dtype) if linear_solver is None else None
    feasible_tol = feasible_theta_tol(tol)

    margin = feasible_tol
    both = xp.logical_and(mask_l, mask_u)
    narrow = xp.logical_and(both, upper_safe - lower_safe <= 2.0 * margin)
    midpoint = 0.5 * (lower_safe + upper_safe)
    lower_target = xp.where(narrow, midpoint, lower_safe + margin)
    upper_target = xp.where(narrow, midpoint, upper_safe - margin)

    def project(x: Array) -> Array:
        x = xp.where(mask_l, xp.maximum(x, lower_target), x)
        return xp.where(mask_u, xp.minimum(x, upper_target), x)

    def infeasibility(x: Array) -> tuple[float, Array, Array, Array]:
        c = eq_fn(x) if m_eq > 0 else xp.zeros((0,), dtype=dtype)
        g = ineq_fn(x) if m > 0 else xp.zeros((0,), dtype=dtype)
        gpos = xp.maximum(g, xp.zeros_like(g)) if m > 0 else g
        f = 0.5 * (float(xp.sum(c * c)) + float(xp.sum(gpos * gpos)))
        return f, c, g, gpos

    def recover_slack(g: Array) -> Array:
        if m == 0:
            return s
        floor = xp.full((m,), _SLACK_FLOOR, dtype=dtype)
        return xp.maximum(-g, floor)

    def filter_theta(c: Array, g: Array, s_out: Array) -> float:
        theta = float(xp.sum(xp.abs(c))) if m_eq > 0 else 0.0
        if m > 0:
            theta += float(xp.sum(xp.abs(g + s_out)))
        return theta

    def escape(x: Array) -> Array:
        scale = _ESCAPE_SCALE * (1.0 + float(xp.max(xp.abs(x))))
        return project(x + scale * _escape_direction(xp, x))

    x = project(x)
    lam = _LM_INIT
    f_window: list[float] = []
    escape_used = False  # the saddle probe fires at most once per call
    exit_reason = RestorationExit.BUDGET
    for _ in range(_MAX_ITER):
        f, c, g, gpos = infeasibility(x)

        theta = float(xp.max(xp.abs(c))) if m_eq > 0 else 0.0
        if m > 0:
            theta = max(theta, float(xp.max(xp.abs(gpos))))
        if theta <= feasible_tol:
            exit_reason = RestorationExit.FEASIBLE
            break

        # Trailing-window progress guard: a plateau (an LM accept/reject limit
        # cycle near a nonzero local minimizer of F) exits here instead of
        # consuming the full budget, which is what pays for the larger
        # _MAX_ITER that genuinely converging runs (CORE1-class) need.
        f_window.append(f)
        if len(f_window) > _STALL_WINDOW:
            del f_window[0]
            if f > (1.0 - _STALL_RTOL) * f_window[0]:
                exit_reason = RestorationExit.STALL_WINDOW
                break

        eq_jac = eq_jac_fn(x) if m_eq > 0 else None
        ineq_jac = ineq_jac_fn(x) if m > 0 else None
        active = xp.astype(g > 0.0, dtype) if m > 0 else None
        stacked = (
            None
            if linear_solver is None
            else _stacked_jacobian(xp, eq_jac, c, ineq_jac, gpos, active)
        )
        hessian: Array | None = None
        if linear_solver is None:
            # Dense reference route: materialize ``Jᵀ Σ J`` block by block
            # (kept verbatim so its round-off, and the sweep it was tuned on,
            # stay put).
            hessian = xp.zeros((n, n), dtype=dtype)
            grad = xp.zeros((n,), dtype=dtype)
            if eq_jac is not None:
                jc = _dense(eq_jac, xp, dtype)
                jc_t = xp.permute_dims(jc, (1, 0))
                hessian = hessian + xp.matmul(jc_t, jc)
                grad = grad + xp.matmul(jc_t, c)
            if ineq_jac is not None:
                assert active is not None
                jg = _dense(ineq_jac, xp, dtype)
                jg_w = jg * xp.expand_dims(active, axis=1)
                jg_t = xp.permute_dims(jg, (1, 0))
                hessian = hessian + xp.matmul(jg_t, jg_w)
                grad = grad + xp.matmul(jg_t, gpos)
        elif stacked is None:
            grad = xp.zeros((n,), dtype=dtype)
        else:
            jacobian, _, residual = stacked
            grad = jacobian.rmatvec(residual)

        # First-order stationarity for the BOUND-CONSTRAINED infeasibility
        # problem: a component whose descent direction points out of the box is
        # blocked and carries no reducibility information (projected-gradient
        # optimality; Bertsekas 1999, prop. 2.1.2). Testing the raw gradient
        # here misses active-bound stalls (MANNE, S2MPJ 2026-07 audit) and
        # grinds the damping to its ceiling with one Jacobian rebuild per
        # projection-swallowed trial.
        blocked_lo = xp.logical_and(
            mask_l, xp.logical_and(x <= lower_target, grad > 0.0)
        )
        blocked_hi = xp.logical_and(
            mask_u, xp.logical_and(x >= upper_target, grad < 0.0)
        )
        blocked = xp.logical_or(blocked_lo, blocked_hi)
        pg = xp.where(blocked, xp.zeros_like(grad), grad)
        grad_norm = float(xp.max(xp.abs(pg))) if n > 0 else 0.0
        if grad_norm <= _GRAD_TOL:
            # Stationary point of the infeasibility with θ > 0 ⇒ a local-
            # infeasibility certificate — once the saddle probe has had its
            # say (``_escape_direction``).
            if not escape_used:
                escape_used = True
                x = escape(x)
                f_window.clear()
                continue
            s_out = recover_slack(g)
            return x, s_out, RestorationExit.STATIONARY

        # Bound-blocked variables are fixed for this step: zero their rows and
        # columns of the normal matrix (unit diagonal, zero rhs) so the free
        # block solves the *reduced* Gauss-Newton system — projected Newton on
        # the binding set (Bertsekas 1999, §2.3). On strongly coupled
        # Jacobians the full-space step is dominated by the blocked
        # components; projection swallows it and the LM loop degrades into a
        # microscopic gradient crawl (S2MPJ DRUGDIS: θ 0.19 → 0.16 in a full
        # 200-iteration budget, vs 8e-4 with the reduction).
        blocked_f = xp.astype(blocked, dtype)
        free_f = 1.0 - blocked_f
        normal: _RestorationNormalOperator | None = None
        if linear_solver is None:
            assert hessian is not None and identity is not None
            hessian = (
                hessian
                * (xp.expand_dims(free_f, axis=1) * xp.expand_dims(free_f, axis=0))
                + identity * blocked_f
            )
        else:
            # ``stacked`` is set: a missing Jacobian means a zero gradient,
            # which exited as STATIONARY above.
            assert stacked is not None
            jacobian, weights, _ = stacked
            normal = _RestorationNormalOperator(
                jacobian=jacobian,
                weights=weights,
                free=free_f,
                blocked=blocked_f,
                damping=lam,
            )
        grad = pg

        # Damped Gauss-Newton step with an INNER damping loop: the normal
        # matrix, gradient and residuals belong to the unchanged iterate, so a
        # rejected trial retries with a larger λ without rebuilding them
        # (Levenberg–Marquardt damping control; Nocedal & Wright 2006, §10.3).
        # A rank-deficient or extreme-scale normal matrix (e.g. the (1+x1²)²
        # Jacobian of HS7 reaching ~1e201 at a bad iterate) can make the
        # backend's solve raise or return a non-finite step; both count as a
        # rejected trial. The dense exception type is backend-specific (numpy
        # ``LinAlgError``, torch ``_LinAlgError``, …) and cannot be named
        # without importing a concrete library (invariant #1), so it is caught
        # broadly. The iterative route raises ``LinearSolveError`` only for
        # numerical trouble — any other exception is an operator-callback or
        # configuration bug and propagates — and a work-capped solve hands
        # back its truncated iterate, which is a descent direction for the SPD
        # Gauss-Newton model (Steihaug 1983): it is tried like any direction
        # rather than discarded, so the ladder is not climbed on Krylov work
        # limits alone.
        accepted = False
        direction_tried = False
        while lam <= _LM_MAX:
            dx: Array | None
            try:
                if normal is None:
                    assert hessian is not None and identity is not None
                    dx = xp.linalg.solve(hessian + lam * identity, -grad)
                else:
                    assert linear_solver is not None
                    normal.damping = lam
                    linear_solver.factor(normal)
                    dx = linear_solver.solve(-grad)
            except MemoryError:  # a genuine resource failure must propagate
                raise
            except LinearSolveError as exc:
                dx = exc.iterate
            except Exception:
                if normal is not None:
                    raise
                dx = None  # backend-specific dense singular-solve error
            if dx is not None and bool(xp.all(xp.isfinite(dx))):
                direction_tried = True
                x_trial = project(x + dx)
                f_trial, _, _, _ = infeasibility(x_trial)
                if f_trial < f:
                    x = x_trial
                    lam = max(_LM_INIT, lam * _LM_SHRINK)
                    accepted = True
                    break
            lam = lam * _LM_GROW
        if not accepted:
            # A finite direction rejected at every damping is numerical
            # stationarity, which certifies like the gradient test above. On
            # the iterative route, never obtaining a direction at all is not a
            # certificate (a Krylov breakdown says nothing about local
            # feasibility). The dense route keeps its established
            # exception-to-NO_DESCENT semantics: its λ = 1e16 rung is always
            # solvable, so that case is a non-finite Jacobian.
            certified = normal is None or direction_tried
            if certified and not escape_used:
                # Same saddle probe as the gradient test: a symmetric trap
                # rejects every in-subspace direction too.
                escape_used = True
                x = escape(x)
                f_window.clear()
                lam = _LM_INIT
                continue
            exit_reason = (
                RestorationExit.NO_DESCENT
                if certified
                else RestorationExit.LINEAR_SOLVE_FAILED
            )
            break

    _, c, g, _ = infeasibility(x)
    s_out = recover_slack(g)
    if filter_theta(c, g, s_out) <= feasible_tol:
        # The final point is feasible by the driver's own (ℓ1 filter) measure,
        # whatever ended the loop — never report a stall from a feasible point.
        exit_reason = RestorationExit.FEASIBLE
    return x, s_out, exit_reason


__all__ = ["RestorationExit", "feasible_theta_tol", "restore"]
