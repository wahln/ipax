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
minimizer of ``‖g+s‖`` over ``s ≥ 0``). It returns a point with reduced
violation, or declares local infeasibility when the infeasibility minimization
stalls at a stationary point with ``θ`` above tolerance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipax.backend.operators import LinearOperator
    from ipax.typing import Array, Namespace

_MAX_ITER = 80
_LM_INIT = 1e-8  # Levenberg–Marquardt damping seed
_LM_GROW = 10.0
_LM_SHRINK = 0.1
_GRAD_TOL = 1e-10  # stationarity test for the infeasibility objective
_SLACK_FLOOR = 1e-12


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
) -> tuple[Array, Array, bool]:
    """Minimize the constraint infeasibility; return ``(x, s, infeasible)``."""
    dtype = x.dtype
    n = int(x.shape[0])
    identity = xp.eye(n, dtype=dtype)
    feasible_tol = max(tol, 1e-8)

    def project(x: Array) -> Array:
        margin = feasible_tol
        both = xp.logical_and(mask_l, mask_u)
        narrow = xp.logical_and(both, upper_safe - lower_safe <= 2.0 * margin)
        midpoint = 0.5 * (lower_safe + upper_safe)
        lower_target = xp.where(narrow, midpoint, lower_safe + margin)
        upper_target = xp.where(narrow, midpoint, upper_safe - margin)
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
    for _ in range(_MAX_ITER):
        f, c, g, gpos = infeasibility(x)
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

        theta = float(xp.max(xp.abs(c))) if m_eq > 0 else 0.0
        if m > 0:
            theta = max(theta, float(xp.max(xp.abs(gpos))))
        grad_norm = float(xp.max(xp.abs(grad))) if n > 0 else 0.0

        if theta <= feasible_tol:
            break
        if grad_norm <= _GRAD_TOL:
            # Stationary point of the infeasibility with θ > 0 ⇒ infeasible.
            s_out = recover_slack(g)
            return x, s_out, True

        dx = xp.linalg.solve(hessian + lam * identity, -grad)
        x_trial = project(x + dx)
        f_trial, _, _, _ = infeasibility(x_trial)
        if f_trial < f:
            x = x_trial
            lam = max(_LM_INIT, lam * _LM_SHRINK)
        else:
            lam = lam * _LM_GROW

    _, c, g, _ = infeasibility(x)
    s_out = recover_slack(g)
    return x, s_out, filter_theta(c, g, s_out) > feasible_tol


__all__ = ["restore"]
