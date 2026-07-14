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

"""Barrier parameter schedules, free-mode safeguard, fraction-to-boundary."""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

from ipax.backend.namespace import array_namespace

if TYPE_CHECKING:
    from ipax.options import BarrierOptions
    from ipax.typing import Array


# LOQO centrality rule constants (Nocedal, Wächter & Waltz 2009, eq. (3.6)):
# σ = _LOQO_SCALE · min(_LOQO_WEIGHT · (1−ξ)/ξ, _LOQO_CLIP)³, so σ ≤ 0.8 and the
# next μ never exceeds 0.8× the current average complementarity.
_LOQO_SCALE = 0.1
_LOQO_WEIGHT = 0.05
_LOQO_CLIP = 2.0


def _mu_floor(options: BarrierOptions, tol: float) -> float:
    """Common schedule floor ``max(μ_min, ε_tol/10)`` (Wächter & Biegler, eq. (7))."""
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    return max(options.mu_min, tol / 10.0)


def update_mu(mu: float, options: BarrierOptions, tol: float) -> float:
    """Monotone Fiacco-McCormick barrier reduction.

    ``μ⁺ = max(ε_tol/10, κ_μ·min(μ, μ^θ_μ))`` — an aggressive variant of
    Wächter & Biegler 2006, eq. (7) with ``κ_μ`` multiplying *both* branches.
    For ``μ ≤ 1`` this is the historical ipax pace ``κ_μ·μ^θ_μ`` (the paper's
    plain ``min(κ_μ·μ, μ^θ_μ)`` reduces μ 3–5× slower there, which cost
    HS268/S268 in the S2MPJ v9 sweep); the ``min`` still matters for ``μ > 1``,
    where the superlinear branch *increases* and, unguarded, locked forever
    any μ an adaptive oracle had inflated
    (regression: tests/regression/test_mu_oracle_inflation.py).
    """
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    return float(
        max(
            _mu_floor(options, tol),
            options.kappa_mu * min(mu, mu**options.theta_mu),
        )
    )


def adaptive_mu(
    avg_compl: float, min_compl: float, options: BarrierOptions, tol: float
) -> float:
    """LOQO centrality rule (Nocedal, Wächter & Waltz 2009, eqs. (3.1), (3.6)).

    ``μ = σ · avg_compl`` with ``σ = 0.1·min(0.05·(1−ξ)/ξ, 2)³`` where
    ``ξ = min_compl / avg_compl`` measures the deviation of the smallest
    complementarity product from the average: a centered iterate (ξ → 1)
    yields an aggressive reduction, an uncentered one (ξ → 0) caps σ at 0.8.
    """
    floor = _mu_floor(options, tol)
    if avg_compl <= 0.0:
        return floor
    xi = min(min_compl / avg_compl, 1.0)
    if xi <= 0.0:
        sigma = _LOQO_SCALE * _LOQO_CLIP**3
    else:
        sigma = _LOQO_SCALE * min(_LOQO_WEIGHT * (1.0 - xi) / xi, _LOQO_CLIP) ** 3
    return float(max(floor, sigma * avg_compl))


def breedveld_mu(
    avg_compl: float, alpha: float, options: BarrierOptions, tol: float
) -> float:
    """Steplength-driven duality-gap reduction (Breedveld 2017, eqs. (10)–(12)).

    The next barrier parameter is ``σ(α)`` times the current duality-gap
    estimate ``avg_compl`` (eq. (10)), with ``σ = ((α−1)/(α+10))²`` (eq. (12))
    built from the steplength α actually taken: a full step (α = 1) drives μ
    superlinearly toward zero, a blocked step (α → 0) re-centers with σ = 0.01.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    sigma = ((alpha - 1.0) / (alpha + 10.0)) ** 2
    return float(max(_mu_floor(options, tol), sigma * avg_compl))


def fallback_mu(avg_compl: float, options: BarrierOptions, tol: float) -> float:
    """Monotone-mode re-entry μ (Nocedal, Wächter & Waltz 2009, §5.1).

    When the free-mode safeguard trips, the monotone strategy restarts from a
    fraction of the current complementarity: ``μ = 0.8·(average complementarity)``
    in the paper's implementations, floored like every schedule.
    """
    return float(max(_mu_floor(options, tol), options.fallback_mu_factor * avg_compl))


class FreeModeMonitor:
    """KKT-error globalization for free-mode μ oracles (NWW 2009, §5.1, Alg. A).

    In *free mode* the oracle re-targets μ every iteration; that is safe only
    while the iterates keep making progress. This monitor requires the KKT
    error to stay below ``κ·M_k`` where ``M_k = max{Φ_{k−l}, …, Φ_k}`` with
    ``l = min(k, l_max)`` over the free-mode iterates; on failure the driver
    suspends the oracle (*monotone mode*) until the error re-crosses ``κ·M_k``
    of the switch point, checked at every intermediate monotone iterate.

    One instance per solve — the mode is explicit per-run state, not module
    state. Iterates are never rolled back: ipax's filter/Breedveld line search
    already globalizes each step, so the safeguard gates only the μ rule
    (matching IPOPT's ``adaptive_mu_globalization="kkt-error"`` behavior).
    """

    def __init__(self, options: BarrierOptions) -> None:
        self._enabled = options.fallback == "kkt-error"
        self._kappa = options.fallback_kappa
        # M_k spans the last l_max+1 free-mode errors (l = min(k, l_max)).
        self._errors: deque[float] = deque(maxlen=options.fallback_window + 1)
        self._free = True
        self._m_switch = math.inf

    def observe(self, error: float) -> tuple[bool, bool]:
        """Record this iterate's KKT error; return ``(free, entered_monotone)``.

        ``entered_monotone`` is ``True`` exactly on the switching iteration so
        the driver can re-initialize μ (see :func:`fallback_mu`) once.
        """
        if not self._enabled:
            return True, False
        if self._free:
            if self._errors and error > self._kappa * max(self._errors):
                self._free = False
                self._m_switch = max(self._errors)
                return False, True
            self._errors.append(error)
            return True, False
        if error <= self._kappa * self._m_switch:
            # Sufficient progress for the original problem — resume free mode
            # with a fresh reference window seeded at the re-entry error.
            self._free = True
            self._errors.clear()
            self._errors.append(error)
            return True, False
        return False, False

    def suspend(self, error: float) -> None:
        """Force monotone mode from outside the KKT-error test.

        Used when a repeated feasible-point re-center *raised* μ (the barrier
        escalation): in free mode the oracle re-targets μ from the current
        complementarity on the very next iteration — exactly the stale,
        near-floor value the raise is escaping — silently undoing it.
        Suspension reuses the NWW §5.1 re-entry rule of :meth:`observe`: free
        mode resumes once the KKT error drops below ``κ`` of the suspension
        point, i.e. once the raised barrier has produced real progress. When
        already suspended the tighter switch point wins. A no-op when the
        fallback safeguard is disabled.
        """
        if not self._enabled:
            return
        self._m_switch = error if self._free else min(self._m_switch, error)
        self._free = False


def complementarity_measures(
    *,
    s: Array,
    y_ineq: Array,
    z_lower: Array,
    z_upper: Array,
    x_minus_l: Array,
    u_minus_x: Array,
    mask_l: Array,
    mask_u: Array,
    m: int,
    n_bounds: int,
) -> tuple[float, float]:
    """``(average, minimum)`` complementarity products over slack/bound pairs.

    The pairs are the slack products ``sᵢλᵢ`` plus the active-bound products
    ``(x−l)ᵢ z_{L,i}`` and ``(u−x)ᵢ z_{U,i}`` — the duality-gap estimate of
    Breedveld 2017, eq. (10), and the centrality inputs of the LOQO rule
    (Nocedal, Wächter & Waltz 2009, eq. (3.6)). Vectorized to two host syncs.
    """
    n_pairs = m + n_bounds
    if n_pairs <= 0:
        raise ValueError("no complementarity pairs (m + n_bounds == 0)")
    xp = array_namespace(x_minus_l)
    inf = math.inf

    prod_l = x_minus_l * z_lower
    prod_u = u_minus_x * z_upper
    total = xp.sum(xp.where(mask_l, prod_l, xp.zeros_like(prod_l))) + xp.sum(
        xp.where(mask_u, prod_u, xp.zeros_like(prod_u))
    )
    smallest = xp.min(
        xp.minimum(
            xp.where(mask_l, prod_l, xp.full_like(prod_l, inf)),
            xp.where(mask_u, prod_u, xp.full_like(prod_u, inf)),
        )
    )
    if m > 0:
        prod_s = s * y_ineq
        total = total + xp.sum(prod_s)
        smallest = xp.minimum(smallest, xp.min(prod_s))
    minimum = float(smallest)
    return float(total) / n_pairs, (0.0 if math.isinf(minimum) else minimum)


def fraction_to_boundary(v: Array, dv: Array, tau: float) -> float:
    """Largest ``alpha`` in ``[0, 1]`` preserving ``v + alpha * dv >= (1 - tau) * v``."""
    if len(v.shape) != 1 or len(dv.shape) != 1:
        raise ValueError("fraction-to-boundary expects rank-1 vectors")
    if v.shape != dv.shape:
        raise ValueError("v and dv must have the same shape")
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")

    if int(v.shape[0]) == 0:
        return 1.0

    # Only components moving toward the boundary (dv < 0) limit the step; for
    # those alpha_i = tau * v_i / (-dv_i) (Wachter & Biegler 2006, eq. 15).
    # Vectorized so the whole rule costs a single host<->device sync (the final
    # float()) rather than one per element — the element-wise Python loop made
    # this O(n) syncs/call and dominated GPU iteration time.
    xp = array_namespace(v, dv)
    blocking = dv < 0.0
    safe_denom = xp.where(blocking, -dv, xp.ones_like(dv))
    ratios = xp.where(blocking, tau * v / safe_denom, xp.full_like(v, math.inf))
    alpha = float(xp.min(ratios))
    return max(0.0, min(1.0, alpha))


__all__ = [
    "FreeModeMonitor",
    "adaptive_mu",
    "breedveld_mu",
    "complementarity_measures",
    "fallback_mu",
    "fraction_to_boundary",
    "update_mu",
]
