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

if TYPE_CHECKING:
    from ipax.backend.operators import LinearOperator
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


class RestorationExit(Enum):
    """How the infeasibility minimization ended.

    Only :attr:`STATIONARY` and :attr:`NO_DESCENT` are certificates of local
    infeasibility (a first-order stationary point of the bound-constrained
    infeasibility with ``θ > 0``; Wächter & Biegler 2006, §3.3). A window or
    budget exit is a mere stall: the S2MPJ restfix audit (2026-07) showed the
    trailing-window guard exiting *early* on slow problems (LAKES/NASH/SWOPF),
    and treating that as an infeasibility verdict relabels an honest
    out-of-budget failure as a false claim about the problem.
    """

    FEASIBLE = "feasible"  # θ_∞ reached the feasibility tolerance
    STATIONARY = "stationary"  # projected-gradient stationary point of F
    NO_DESCENT = "no_descent"  # no LM damping in [init, max] yields descent
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
) -> tuple[Array, Array, RestorationExit]:
    """Minimize the constraint infeasibility; return ``(x, s, exit_reason)``."""
    dtype = x.dtype
    n = int(x.shape[0])
    identity = xp.eye(n, dtype=dtype)
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

    x = project(x)
    lam = _LM_INIT
    f_window: list[float] = []
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

        hessian = xp.zeros((n, n), dtype=dtype)
        grad = xp.zeros((n,), dtype=dtype)
        if m_eq > 0:
            jc = _dense(eq_jac_fn(x), xp, dtype)
            jc_t = xp.permute_dims(jc, (1, 0))
            hessian = hessian + xp.matmul(jc_t, jc)
            grad = grad + xp.matmul(jc_t, c)
        if m > 0:
            jg = _dense(ineq_jac_fn(x), xp, dtype)
            active = xp.astype(g > 0.0, dtype)
            jg_w = jg * xp.expand_dims(active, axis=1)
            hessian = hessian + xp.matmul(xp.permute_dims(jg, (1, 0)), jg_w)
            grad = grad + xp.matmul(xp.permute_dims(jg, (1, 0)), gpos)

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
            # infeasibility certificate.
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
        hessian = (
            hessian * (xp.expand_dims(free_f, axis=1) * xp.expand_dims(free_f, axis=0))
            + identity * blocked_f
        )
        grad = pg

        # Damped Gauss-Newton step with an INNER damping loop: the normal
        # matrix, gradient and residuals belong to the unchanged iterate, so a
        # rejected trial retries with a larger λ without rebuilding them
        # (Levenberg–Marquardt damping control; Nocedal & Wright 2006, §10.3).
        # A rank-deficient or extreme-scale normal matrix (e.g. the (1+x1²)²
        # Jacobian of HS7 reaching ~1e201 at a bad iterate) can make the
        # backend's solve raise or return a non-finite step; both count as a
        # rejected trial. The exception type is backend-specific (numpy
        # ``LinAlgError``, torch ``_LinAlgError``, …) and cannot be named
        # without importing a concrete library (invariant #1), so it is caught
        # broadly.
        accepted = False
        while lam <= _LM_MAX:
            try:
                dx = xp.linalg.solve(hessian + lam * identity, -grad)
                step_ok = bool(xp.all(xp.isfinite(dx)))
            except MemoryError:  # a genuine resource failure must propagate
                raise
            except Exception:  # backend-specific singular-solve error
                step_ok = False
            if step_ok:
                x_trial = project(x + dx)
                f_trial, _, _, _ = infeasibility(x_trial)
                if f_trial < f:
                    x = x_trial
                    lam = max(_LM_INIT, lam * _LM_SHRINK)
                    accepted = True
                    break
            lam = lam * _LM_GROW
        if not accepted:
            # No damping in [_LM_INIT, _LM_MAX] yields descent: numerically
            # stationary, which certifies like the gradient test above.
            exit_reason = RestorationExit.NO_DESCENT
            break

    _, c, g, _ = infeasibility(x)
    s_out = recover_slack(g)
    if filter_theta(c, g, s_out) <= feasible_tol:
        # The final point is feasible by the driver's own (ℓ1 filter) measure,
        # whatever ended the loop — never report a stall from a feasible point.
        exit_reason = RestorationExit.FEASIBLE
    return x, s_out, exit_reason


__all__ = ["RestorationExit", "feasible_theta_tol", "restore"]
