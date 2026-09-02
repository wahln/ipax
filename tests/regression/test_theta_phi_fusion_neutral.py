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

"""The fused (θ, φ) merit read is bitwise the separate evaluations.

``_theta_phi`` batches θ's ℓ1 sums, the raw objective, and φ's barrier sum
into one host transfer at every hot merit site (loop top, each line-search
trial, the SOC point). Trajectory neutrality of that fusion rests on the
returned floats being *bitwise* what the unfused reads produced — this
shadows every call during a real constrained solve and pins exactly that,
including the write-back of the objective float into the point cache.
"""

from __future__ import annotations

from ipax import Options, Status, solve
from ipax.ipm.driver import IPMDriver
from ipax.testing.problems import HS71
from tests._helpers import array


def test_theta_phi_matches_the_unfused_components(namespace, monkeypatch):
    checked = {"n": 0}
    orig = IPMDriver._theta_phi

    def checking(self, x, s, mu, m, m_eq, mask_l, mask_u, lower_safe, upper_safe):
        theta, phi = orig(
            self, x, s, mu, m, m_eq, mask_l, mask_u, lower_safe, upper_safe
        )
        # θ: bitwise the standalone ℓ1 evaluation.
        assert theta == self._theta_l1(x, s, m, m_eq)
        # The objective float written back to the point cache must be bitwise
        # the lazily-derived one ...
        f = self._objective(x)
        assert f == float(self._objective_raw(x))
        # ... and φ must decompose exactly over it with the barrier sum the
        # unfused read would have produced (the formula is duplicated here on
        # purpose: this is the regression pin for the fusion).
        xp = self._xp
        parts = []
        if m > 0:
            parts.append(xp.sum(xp.log(s)))
        if self._has_lower:
            x_minus_l = xp.where(mask_l, x - lower_safe, xp.ones_like(x))
            parts.append(xp.sum(xp.where(mask_l, xp.log(x_minus_l), xp.zeros_like(x))))
        if self._has_upper:
            u_minus_x = xp.where(mask_u, upper_safe - x, xp.ones_like(x))
            parts.append(xp.sum(xp.where(mask_u, xp.log(u_minus_x), xp.zeros_like(x))))
        if parts:
            barrier = (
                float(parts[0]) if len(parts) == 1 else float(xp.sum(xp.stack(parts)))
            )
        else:
            barrier = 0.0
        assert phi == f - mu * barrier
        checked["n"] += 1
        return theta, phi

    monkeypatch.setattr(IPMDriver, "_theta_phi", checking)
    result = solve(
        HS71(namespace),
        array(namespace, [1.0, 5.0, 5.0, 1.0]),
        options=Options(),
    )

    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    assert checked["n"] > 0  # the shadow actually ran (every merit site)
