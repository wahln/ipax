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
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipax.options import BreedveldOptions

# Below this iteration index the controller uses the looser ratio control
# (Breedveld 2017 §2, eq. 36) suited to the early iterates of convex problems.
_RATIO_CONTROL_ITERS = 5


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
    ) -> tuple[float, bool, int]:
        """Return ``(alpha, restoration_needed, n_trials)`` (Breedveld 2017 §2)."""
        o = self._o
        alpha = alpha_max
        trials = 0

        # Ratio control: in early iterations of (near-)convex problems accept the
        # aggressive fraction-to-boundary step when the barrier objective does not
        # inflate beyond the ratio bound (eq. 36).
        if iteration < _RATIO_CONTROL_ITERS:
            trials += 1
            theta_t, phi_t = eval_point(alpha)
            inflate = abs(phi_t) <= o.ratio_limit * max(abs(phi0), 1.0)
            if inflate and (theta_t <= theta0 or phi_t <= phi0):
                return alpha, False, trials

        for _ in range(o.max_backtrack):
            trials += 1
            theta_t, phi_t = eval_point(alpha)
            if self.markov_accept(theta_t, phi_t, theta0, phi0, dphi, alpha):
                return alpha, False, trials
            alpha *= o.backtrack
        return alpha, True, trials


__all__ = ["BreedveldController"]
