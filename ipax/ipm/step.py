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

"""Newton step assembly + analytic slack/bound-dual elimination (§2.3).

Given the condensed solve ``Δx``, recover the eliminated increments ``Δs``,
``Δλ``, ``Δz_L``, ``Δz_U`` by the cheap diagonal back-substitutions (Breedveld
2017, eqs. 14–17; Wächter & Biegler 2006, eqs. 11–13).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipax.backend.operators import LinearOperator
    from ipax.typing import Array, Namespace


@dataclass(frozen=True, slots=True)
class NewtonStep:
    """A full primal–dual search direction."""

    dx: Array
    ds: Array
    dy_eq: Array
    dy_ineq: Array
    dz_lower: Array
    dz_upper: Array


def recover_eliminated(
    dx: Array,
    *,
    xp: Namespace,
    ineq_jac: LinearOperator,
    m: int,
    s: Array,
    y_ineq: Array,
    r_pi: Array,
    sigma_s: Array,
    z_lower: Array,
    z_upper: Array,
    sigma_l: Array,
    sigma_u: Array,
    x_minus_l: Array,
    u_minus_x: Array,
    mask_l: Array,
    mask_u: Array,
    mu: float,
    comp_s: Array | float | None = None,
    comp_l: Array | float | None = None,
    comp_u: Array | float | None = None,
    dy_eq: Array | None = None,
) -> NewtonStep:
    """Back-substitute slack / bound-dual increments after the condensed solve.

    The condensed system solved for ``Δx``; here we recover the rest from the
    perturbed-KKT relations (the residuals folded into the diagonals, §2.3):

    - ``Δs  = -(g + s) - J Δx``
    - ``Δλ  = Σ_s (J Δx + (g + s)) - λ + τ_s/s``         with ``Σ_s = Λ/S``
    - ``Δz_L = -Σ_L Δx - z_L + τ_L/(x - x_L)``           with ``Σ_L = Z_L/(X-X_L)``
    - ``Δz_U =  Σ_U Δx - z_U + τ_U/(x_U - x)``           with ``Σ_U = Z_U/(X_U-X)``

    ``τ`` is the complementarity *target* for each block. By default it is the
    scalar barrier parameter ``μ`` (the standard centered Newton step). Passing
    the per-component ``comp_s``/``comp_l``/``comp_u`` vectors overrides it — the
    mechanism the Mehrotra/Gondzio corrections use to inject adaptive centering
    and the second-order complementarity correction (``τ = σμ − ΔΔ``) while
    reusing this same recovery (``ipax/ipm/corrections.py``).
    """
    dtype = dx.dtype
    target_s = mu if comp_s is None else comp_s
    target_l = mu if comp_l is None else comp_l
    target_u = mu if comp_u is None else comp_u
    if m > 0:
        jdx = ineq_jac.matvec(dx)
        ds = -r_pi - jdx
        dy_ineq = sigma_s * (jdx + r_pi) - y_ineq + target_s / s
    else:
        ds = xp.zeros((0,), dtype=dtype)
        dy_ineq = xp.zeros((0,), dtype=dtype)

    zero = xp.zeros_like(dx)
    dz_lower = xp.where(mask_l, -sigma_l * dx - z_lower + target_l / x_minus_l, zero)
    dz_upper = xp.where(mask_u, sigma_u * dx - z_upper + target_u / u_minus_x, zero)

    return NewtonStep(
        dx=dx,
        ds=ds,
        dy_eq=xp.zeros((0,), dtype=dtype) if dy_eq is None else dy_eq,
        dy_ineq=dy_ineq,
        dz_lower=dz_lower,
        dz_upper=dz_upper,
    )


__all__ = ["NewtonStep", "recover_eliminated"]
