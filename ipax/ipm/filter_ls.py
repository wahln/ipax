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

"""Filter line-search globalization — default (Wächter & Biegler §2–3, §4.1).

- Filter ``F`` of ``(θ, φ_μ)`` pairs, ``θ(x) = ‖(c, g+s)‖₁``.
- Acceptance by sufficient decrease in ``θ`` **or** barrier objective ``φ_μ``;
  switching condition + Armijo for f-type steps.
- Second-order correction (SOC) when a full step increases ``θ``.
- Hands off to the feasibility restoration phase when no ``α ≥ α_min`` is
  acceptable.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipax.options import LineSearchOptions

# Switching-condition constant δ (Wächter & Biegler 2006, eq. 19).
_SWITCH_DELTA = 1.0


def _safe_pow(base: float, exponent: float) -> float:
    """``base ** exponent`` with IEEE overflow semantics (→ ``inf``, never raise).

    Python's ``float.__pow__`` raises ``OverflowError`` instead of returning
    ``inf`` when the result exceeds the double range, which crashes the switching
    test on a badly-scaled iterate (e.g. an enormous directional derivative
    ``dphi``). Treat such an overflow as ``+inf`` so the comparison still has a
    meaningful, finite-safe answer.
    """
    try:
        return float(base**exponent)
    except OverflowError:
        return float("inf")


@dataclass(slots=True)
class Filter:
    """The ``(θ, φ)`` filter set."""

    entries: list[tuple[float, float]] = field(default_factory=list)

    def is_acceptable(self, theta: float, phi: float) -> bool:
        """Pareto non-domination: not dominated by any existing entry."""
        return all(theta < tj or phi < fj for tj, fj in self.entries)

    def augment(self, theta: float, phi: float) -> None:
        """Add ``(θ, φ)`` and drop entries it dominates."""
        self.entries = [
            (tj, fj) for tj, fj in self.entries if not (theta <= tj and phi <= fj)
        ]
        self.entries.append((theta, phi))


@dataclass(frozen=True, slots=True)
class LineSearchResult:
    """Outcome of one filter line search."""

    alpha: float
    accepted: bool
    augment: bool  # θ-type accepted steps augment the filter
    restoration: bool
    used_soc: bool = False
    n_trials: int = 1  # number of backtracking trial step sizes evaluated


class FilterLineSearch:
    """Backtracking filter line search with optional SOC."""

    def __init__(self, options: LineSearchOptions) -> None:
        self._o = options

    def _filter_acceptable(
        self,
        theta_t: float,
        phi_t: float,
        entries: list[tuple[float, float]],
    ) -> bool:
        """Sufficient decrease w.r.t. every filter entry (W&B eq. 18)."""
        o = self._o
        for tj, fj in entries:
            if not (
                theta_t <= (1.0 - o.gamma_theta) * tj or phi_t <= fj - o.gamma_phi * tj
            ):
                return False
        return True

    def search(
        self,
        *,
        alpha_max: float,
        theta0: float,
        phi0: float,
        dphi: float,
        theta_max: float,
        eval_point: Callable[[float], tuple[float, float]],
        entries: list[tuple[float, float]],
        soc: Callable[[float], tuple[float, float] | None] | None = None,
        grad_finite: Callable[[float], bool] | None = None,
    ) -> LineSearchResult:
        """Return the accepted ``(α, …)`` or signal restoration.

        ``eval_point(α)`` returns ``(θ, φ)`` at ``x + α d`` (and ``s + α ds``).
        ``soc(α)`` optionally returns ``(θ, φ)`` for a second-order-corrected
        trial when the full step increases θ (W&B §2.3, eq. 27). ``theta_max`` is
        the W&B eq. (18) guard: trials with ``θ ≥ θ_max`` (or non-finite θ) are
        never acceptable.

        ``grad_finite(α)``, when supplied, reports whether the Lagrangian
        gradient at the trial point is finite. A step whose ``θ``/``φ`` are finite
        can still overshoot into a region where the *derivatives* overflow to
        inf/NaN (e.g. an exp/rational element function); the line search only
        evaluates ``θ``/``φ``, so such a point would be accepted and then poison
        the next KKT solve. Treating a non-finite-gradient trial as unacceptable
        keeps backtracking to a damped step that stays in the finite region,
        reusing the existing α-reduction (and restoration hand-off if the whole
        ray is bad).
        """
        o = self._o
        alpha = alpha_max
        alpha_min = o.alpha_min_frac
        first = True
        trials = 0
        while alpha >= alpha_min:
            trials += 1
            theta_t, phi_t = eval_point(alpha)

            # SOC on the first (full-ish) trial that worsens feasibility.
            if first and soc is not None and theta_t > theta0:
                corrected = soc(alpha)
                if corrected is not None:
                    theta_c, phi_c = corrected
                    if self._accept(
                        theta_c, phi_c, theta0, phi0, dphi, alpha, theta_max, entries
                    ):
                        # The SOC point differs from ``x + α d``; its own gradient
                        # finiteness is checked inside ``soc`` (which returns None
                        # to reject a non-finite-derivative corrected trial).
                        switching = self._switching(dphi, alpha, theta0)
                        return LineSearchResult(
                            alpha, True, not switching, False, True, trials
                        )
            first = False

            if self._accept(
                theta_t, phi_t, theta0, phi0, dphi, alpha, theta_max, entries
            ) and (grad_finite is None or grad_finite(alpha)):
                switching = self._switching(dphi, alpha, theta0)
                return LineSearchResult(
                    alpha, True, not switching, False, n_trials=trials
                )
            alpha *= 0.5
        return LineSearchResult(alpha_min, False, False, True, n_trials=trials)

    def _switching(self, dphi: float, alpha: float, theta0: float) -> bool:
        o = self._o
        return dphi < 0.0 and alpha * _safe_pow(
            -dphi, o.s_phi
        ) > _SWITCH_DELTA * _safe_pow(theta0, o.s_theta)

    def _accept(
        self,
        theta_t: float,
        phi_t: float,
        theta0: float,
        phi0: float,
        dphi: float,
        alpha: float,
        theta_max: float,
        entries: list[tuple[float, float]],
    ) -> bool:
        o = self._o
        # W&B eq. (18): the filter is initialized to the guard region {θ ≥ θ_max}.
        # Reject wildly infeasible (or non-finite) trials outright, before the
        # f-type switching/Armijo test — otherwise a step whose barrier objective
        # φ collapses toward -∞ could be accepted while θ explodes. A non-finite
        # φ_t itself must also be rejected: a trial that overshoots into a region
        # where the objective evaluates to ±∞/NaN (e.g. an overflowing exp/rational
        # element function) would otherwise pass the Armijo test — ``φ_t = -∞`` is
        # trivially below any finite bound — instead of backtracking to a finite,
        # usable iterate.
        if (
            not math.isfinite(theta_t)
            or not math.isfinite(phi_t)
            or theta_t >= theta_max
        ):
            return False
        if not self._filter_acceptable(theta_t, phi_t, entries):
            return False
        if self._switching(dphi, alpha, theta0):
            # f-type step: require Armijo decrease on the barrier objective.
            return phi_t <= phi0 + o.eta_phi * alpha * dphi
        # θ-type step: sufficient decrease in θ or φ vs the current point
        # (W&B eq. 20). At a feasible iterate (θ0 = 0) no θ-progress step
        # exists — the branch would degenerate to "0 ≤ 0" and accept an
        # arbitrary ascent direction (W&B §2.3: only f-type/φ-decrease
        # acceptance applies there), so it requires θ0 > 0.
        theta_progress = theta0 > 0.0 and theta_t <= (1.0 - o.gamma_theta) * theta0
        return theta_progress or phi_t <= phi0 - o.gamma_phi * theta0


__all__ = ["Filter", "FilterLineSearch", "LineSearchResult"]
