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

"""Primal/dual initialization (Breedveld §3.1, generalized).

- Slacks/duals floored to a positive constant; ``x_0`` projected strictly inside
  ``[x_L, x_U]`` (Wächter & Biegler 2006, §3.6).
- Optional least-squares dual initialization ``∇g(x_0)ᵀ y_0 = −∇f(x_0)`` (LSQR
  in the matrix-free path) — not currently implemented.
- Warm-start hook (the RT layer may inject an application-specific start).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipax.result import WarmStart
    from ipax.typing import Array, Namespace

# Wächter & Biegler 2006, §3.6: relative/absolute push keeping x_0 interior.
_KAPPA1 = 1e-2
_KAPPA2 = 1e-2
# IPOPT ``bound_relax_factor`` (fixed_variable_treatment='relax_bounds'): widen a
# fixed / near-degenerate bound pair so the strict interior is non-empty.
_BOUND_RELAX = 1e-8
# Floor so slacks/duals start strictly positive (Breedveld 2017, §3.1).
_SLACK_FLOOR = 1e-2
# Floor for warm-started slacks/duals: just strictly interior, so supplied
# near-active values are preserved rather than pushed back toward μ-scale.
_WARM_FLOOR = 1e-8
# Re-centering for a line search that fails at an already-feasible point
# (Wächter & Biegler 2006, §3.3: the multipliers belong to the abandoned
# iterate and are re-initialized after a restoration jump). The slack floor is
# μ-scaled so the repair is a small perturbation near convergence, and the
# multiplier clip mirrors the Σ safeguard of W&B eq. (16) with a much tighter
# band (κ_Σ = 1e10 there): this is a one-shot repair of a provably stuck
# state, not an every-iteration guard, so it may be aggressive — a component
# already consistent with the central path (sλ ≈ μ) always passes untouched.
_RECENTER_SLACK_FRACTION = 0.1
_RECENTER_KAPPA = 1e2


@dataclass(frozen=True, slots=True)
class InitialPoint:
    """Strictly-feasible-interior starting point for the IPM."""

    x: Array
    s: Array
    y_eq: Array
    y_ineq: Array
    z_lower: Array
    z_upper: Array


def project_interior(
    xp: Namespace,
    x0: Array,
    lower_safe: Array,
    upper_safe: Array,
    mask_l: Array,
    mask_u: Array,
) -> Array:
    """Push ``x0`` strictly inside the bounds (Wächter & Biegler 2006, §3.6)."""
    both = xp.logical_and(mask_l, mask_u)
    span = xp.where(both, upper_safe - lower_safe, xp.ones_like(x0))

    push_l = _KAPPA1 * xp.maximum(xp.ones_like(x0), xp.abs(lower_safe))
    push_l = xp.where(both, xp.minimum(push_l, _KAPPA2 * span), push_l)
    push_u = _KAPPA1 * xp.maximum(xp.ones_like(x0), xp.abs(upper_safe))
    push_u = xp.where(both, xp.minimum(push_u, _KAPPA2 * span), push_u)

    x = xp.where(mask_l, xp.maximum(x0, lower_safe + push_l), x0)
    x = xp.where(mask_u, xp.minimum(x, upper_safe - push_u), x)
    return x


def relax_fixed_bounds(
    xp: Namespace,
    lower: Array | None,
    upper: Array | None,
) -> tuple[Array | None, Array | None]:
    """Widen fixed / near-degenerate finite bound pairs to admit an interior.

    A fixed variable (``x_L == x_U``) — common in CUTEst problems — has no strict
    interior, so the barrier dual ``z = μ/(x − x_L)`` is singular and the first
    Newton step is non-finite. Relax only pairs whose gap is within
    :data:`_BOUND_RELAX` of degenerate, symmetrically about their midpoint
    (IPOPT ``fixed_variable_treatment='relax_bounds'``); well-separated bounds are
    returned untouched. One-sided bounds (the other side ``±inf``) are unaffected.
    """
    if lower is None or upper is None:
        return lower, upper
    both = xp.logical_and(xp.isfinite(lower), xp.isfinite(upper))
    # Finite stand-ins off the two-sided pairs so ±inf bounds don't produce
    # inf/nan arithmetic (the results are masked out, but would still warn).
    zero = xp.zeros_like(lower)
    safe_lower = xp.where(both, lower, zero)
    safe_upper = xp.where(both, upper, zero)
    mid = 0.5 * (safe_lower + safe_upper)
    scale = xp.maximum(xp.ones_like(mid), xp.abs(mid))
    needs = xp.logical_and(both, (safe_upper - safe_lower) <= _BOUND_RELAX * scale)
    relax = _BOUND_RELAX * scale
    lower = xp.where(needs, mid - relax, lower)
    upper = xp.where(needs, mid + relax, upper)
    return lower, upper


def initialize(
    *,
    xp: Namespace,
    x0: Array,
    lower_safe: Array,
    upper_safe: Array,
    mask_l: Array,
    mask_u: Array,
    ineq_fn: Callable[[Array], Array] | None,
    mu_init: float,
    m: int,
    slack_init_scale: float = 0.0,
) -> InitialPoint:
    """Build the strictly-interior initial point for the condensed route.

    Slacks satisfy ``g(x) + s = 0`` where that keeps ``s`` positive, otherwise
    they are floored (Breedveld 2017, §3.1). Duals start at ``μ`` complementarity
    so ``S Λ e ≈ μ e`` and the bound complementarities match at iteration 0.

    With ``slack_init_scale > 0`` the flat slack floor :data:`_SLACK_FLOOR` is
    raised to ``max(_SLACK_FLOOR, slack_init_scale·max|g(x_0)|)`` so a
    deeply-infeasible start does not pin every violated-constraint slack against
    a fixed constant (which forces a ~1e-3 fraction-to-boundary step); the
    coupled ``y = μ_init/s`` then starts the multipliers scaled to the constraint
    magnitude too (``BarrierOptions.slack_init_scale``).
    """
    dtype = x0.dtype
    x = project_interior(xp, x0, lower_safe, upper_safe, mask_l, mask_u)

    if m > 0 and ineq_fn is not None:
        g = ineq_fn(x)
        floor = xp.full((m,), _SLACK_FLOOR, dtype=dtype)
        if slack_init_scale > 0.0:
            # Room proportional to the constraint magnitude (0-d, broadcast).
            floor = xp.maximum(floor, slack_init_scale * xp.max(xp.abs(g)))
        s = xp.maximum(-g, floor)
        y_ineq = mu_init / s
    else:
        s = xp.zeros((0,), dtype=dtype)
        y_ineq = xp.zeros((0,), dtype=dtype)

    x_minus_l = xp.where(mask_l, x - lower_safe, xp.ones_like(x))
    u_minus_x = xp.where(mask_u, upper_safe - x, xp.ones_like(x))
    zero = xp.zeros_like(x)
    z_lower = xp.where(mask_l, mu_init / x_minus_l, zero)
    z_upper = xp.where(mask_u, mu_init / u_minus_x, zero)

    return InitialPoint(
        x=x,
        s=s,
        y_eq=xp.zeros((0,), dtype=dtype),
        y_ineq=y_ineq,
        z_lower=z_lower,
        z_upper=z_upper,
    )


def recenter_slacks_duals(
    xp: Namespace, g: Array, y_ineq: Array, mu: float
) -> tuple[Array, Array]:
    """Re-center slacks and inequality multipliers on the current barrier problem.

    Used when the filter line search fails at an **already-feasible** iterate:
    restoration cannot move such a point (it exits immediately at the same
    ``x``), and resuming with the stale state re-derives the same rejected
    direction forever (S2MPJ v11: the HS101 limit cycle, where boundary-floor
    slacks against multipliers grown to ~1e6 gave ``Σ_s = λ/s ~ 1e18``).

    Slacks are re-floored at a fraction of the current ``μ`` (interior again,
    a vanishing perturbation near convergence) and the multipliers are clipped
    into the central band ``[μ/(κ·s), κ·μ/s]`` — the analogue of the ``κ_Σ``
    dual safeguard (Wächter & Biegler 2006, eq. (16)) applied once, at the
    repair point, with a tight band. Components already consistent with the
    central path (``s·λ ≈ μ``) pass through unchanged.
    """
    floor = xp.full(g.shape, _RECENTER_SLACK_FRACTION * mu, dtype=g.dtype)
    s = xp.maximum(-g, floor)
    y = xp.clip(y_ineq, mu / (_RECENTER_KAPPA * s), _RECENTER_KAPPA * mu / s)
    return s, y


def _floor_positive(xp: Namespace, arr: Array) -> Array:
    """Push ``arr`` up to the warm-start interiority floor where it falls below."""
    return xp.maximum(arr, xp.full_like(arr, _WARM_FLOOR))


def apply_warm_start(
    *,
    xp: Namespace,
    warm: WarmStart,
    s: Array,
    y_eq: Array,
    y_ineq: Array,
    z_lower: Array,
    z_upper: Array,
    m: int,
    m_eq: int,
    n: int,
    mask_l: Array,
    mask_u: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Override the default start with user-supplied slacks/multipliers.

    Each provided block must match the problem's dimensions. Slacks and bound /
    inequality multipliers are floored strictly positive (interiority); bound
    multipliers are re-masked to zero off their active bounds; equality
    multipliers pass through. Returns ``(s, y_eq, y_ineq, z_lower, z_upper)``.
    """

    def _check(name: str, arr: Array | None, length: int) -> None:
        if arr is not None and int(arr.shape[0]) != length:
            raise ValueError(
                f"warm-start {name} has length {int(arr.shape[0])}, expected {length}"
            )

    _check("s", warm.s, m)
    _check("y_ineq", warm.y_ineq, m)
    _check("y_eq", warm.y_eq, m_eq)
    _check("z_lower", warm.z_lower, n)
    _check("z_upper", warm.z_upper, n)

    zero = xp.zeros((n,), dtype=z_lower.dtype)
    if warm.s is not None:
        s = _floor_positive(xp, warm.s)
    if warm.y_ineq is not None:
        y_ineq = _floor_positive(xp, warm.y_ineq)
    if warm.y_eq is not None:
        y_eq = warm.y_eq
    if warm.z_lower is not None:
        z_lower = xp.where(mask_l, _floor_positive(xp, warm.z_lower), zero)
    if warm.z_upper is not None:
        z_upper = xp.where(mask_u, _floor_positive(xp, warm.z_upper), zero)
    return s, y_eq, y_ineq, z_lower, z_upper


__all__ = [
    "InitialPoint",
    "apply_warm_start",
    "initialize",
    "project_interior",
    "recenter_slacks_duals",
    "relax_fixed_bounds",
]
