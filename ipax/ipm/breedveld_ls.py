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

"""Breedveld step controller — alternative globalization mode (§4.2).

A lighter controller for convex/RT-like problems, selectable via
``Options.globalization == "breedveld"``:

- Max step to keep ``(z, w, x, y) > 0``, scaled by ``τ = 0.995``.
- Markov filter (barrier-or-infeasibility Armijo, eqs. 34–35) for nonconvex
  problems; ratio control (``f⁺γ_y⁺ / fγ_y < 20``, eq. 36) for early iterations
  of convex problems. Backtrack by ``0.9`` up to 10–20 times.
- Optional (``BreedveldOptions.max_soc > 0``, an ipax addition — the paper
  only backtracks): a second-order correction of a rejected full step
  (Wächter & Biegler 2006, §2.4), judged by the same Markov test.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ipax._logging import logger

if TYPE_CHECKING:
    from ipax.options import BreedveldOptions

# Below this iteration index the controller uses the looser ratio control
# (Breedveld 2017 §2, eq. 36) suited to the early iterates of convex problems.
_RATIO_CONTROL_ITERS = 5


@dataclass(frozen=True, slots=True)
class BreedveldSearchResult:
    """Outcome of one Breedveld step-length search.

    ``used_soc`` marks an accepted second-order-corrected trial: the driver
    then moves to the corrected point (not ``x + α d``) at ``alpha``.
    ``n_trials`` counts every evaluated trial, an evaluated correction
    included (the filter line search does not count its SOC trials).
    """

    alpha: float
    restoration: bool
    n_trials: int
    used_soc: bool = False


class BreedveldController:
    """Markov-filter + ratio-control step acceptance."""

    def __init__(self, options: BreedveldOptions) -> None:
        self._o = options

    def markov_accept(
        self,
        theta_t: float,
        phi_t: float,
        theta0: float,
        phi0: float,
        dphi: float,
        alpha: float,
    ) -> bool:
        """Accept on barrier-objective Armijo **or** infeasibility decrease.

        Breedveld 2017, eqs. (34)–(35): a trial is acceptable if it reduces the
        barrier objective (Armijo) or the constraint infeasibility.
        """
        o = self._o
        armijo = phi_t <= phi0 + o.armijo_c * alpha * dphi
        feasibility = theta_t <= (1.0 - o.armijo_c) * theta0
        return armijo or feasibility

    def search(
        self,
        *,
        alpha_max: float,
        theta0: float,
        phi0: float,
        dphi: float,
        eval_point: Callable[[float], tuple[float, float]],
        iteration: int,
        soc: Callable[[float], tuple[float, float] | None] | None = None,
        soc_theta_min: float = 0.0,
    ) -> BreedveldSearchResult:
        """Search a step length (Breedveld 2017 §2) or signal restoration.

        ``eval_point(α)`` returns ``(θ, φ)`` at ``x + α d``. ``soc(α)``, when
        supplied and ``BreedveldOptions.max_soc > 0``, returns ``(θ, φ)`` of a
        second-order-corrected trial (or ``None`` if no correction could be
        formed); it is consulted once, on a rejected first Markov trial that
        did not reduce θ and whose own θ exceeds ``soc_theta_min`` (see
        ``BreedveldOptions.max_soc``). A correction that returns a point
        counts as one trial; a ``None`` (nothing evaluated) does not.
        """
        o = self._o
        alpha = alpha_max
        trials = 0

        # Ratio control: in early iterations of (near-)convex problems accept the
        # aggressive fraction-to-boundary step when the barrier objective does not
        # inflate beyond the ratio bound (eq. 36).
        ratio_control = iteration < _RATIO_CONTROL_ITERS
        if ratio_control:
            trials += 1
            theta_t, phi_t = eval_point(alpha)
            inflate = abs(phi_t) <= o.ratio_limit * max(abs(phi0), 1.0)
            if inflate and (theta_t <= theta0 or phi_t <= phi0):
                return BreedveldSearchResult(alpha, False, trials)

        # SOC is a Markov-acceptance addition only: the ratio-control window is
        # the paper's deliberately loose early-iterate test, left as published.
        try_soc = soc is not None and o.max_soc > 0 and not ratio_control
        for k in range(o.max_backtrack):
            trials += 1
            theta_t, phi_t = eval_point(alpha)
            # Breedveld 2017, eqs. (34)–(35): barrier Armijo or θ decrease.
            if self.markov_accept(theta_t, phi_t, theta0, phi0, dphi, alpha):
                return BreedveldSearchResult(alpha, False, trials)
            # Wächter & Biegler 2006, §2.4 (A-5.5): correct only the first,
            # full-length trial, and skip it when that trial reduced θ.
            # Deviation: also skip it when the trial is itself feasible to
            # ``soc_theta_min`` — the correction's right-hand side is the
            # trial's constraint residual, so a round-off θ_t (linear
            # constraints: θ_t ≥ θ0 ≈ 1e-15 at random) has nothing to correct
            # and would cost a solve per rejected full step. (θ_t = 0 never
            # gets here: eq. (35) accepts it.)
            if try_soc and k == 0 and theta_t >= theta0 and theta_t > soc_theta_min:
                assert soc is not None
                corrected = soc(alpha)
                if corrected is not None:
                    trials += 1
                # A-5.7–A-5.8 with the Markov test (Breedveld 2017, eqs.
                # (34)–(35)): the corrected point is judged at the original α.
                accepted = corrected is not None and self.markov_accept(
                    corrected[0], corrected[1], theta0, phi0, dphi, alpha
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "  breedveld trial %d: alpha=%.3e soc-%s",
                        trials,
                        alpha,
                        "accept"
                        if accepted
                        else ("none" if corrected is None else "reject"),
                    )
                if accepted:
                    return BreedveldSearchResult(alpha, False, trials, used_soc=True)
                # A-5.10: fall back to the plain backtracking sequence. (The
                # driver's closure stacks a fixed ``max_soc`` rounds and is
                # judged once, in place of A-5.9's κ_soc-guarded loop.)
            alpha *= o.backtrack
        return BreedveldSearchResult(alpha, True, trials)


__all__ = ["BreedveldController", "BreedveldSearchResult"]
