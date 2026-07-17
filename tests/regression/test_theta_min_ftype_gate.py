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

"""Regression: Armijo is not demanded above θ_min (W&B Algorithm A, Step 4).

ipax keyed the f-type branch on the eq. (19) switching condition alone, so at an
*infeasible* iterate it required Armijo decrease on the barrier objective where
the filter method asks only for sufficient decrease in θ **or** φ (eq. 20). A
trial cutting the constraint violation decisively while φ rose was rejected and
backtracked — potentially all the way to the restoration phase.

The unit tests in ``tests/unit/test_theta_min_switching.py`` pin the branch logic
directly; this pins the wiring, on every backend — that the driver-supplied θ_min
actually reaches the acceptance test and changes real classifications on a real
solve. (HS71 landing on its published optimum is already covered by
``tests/integration/test_hock_schittkowski.py``.)
"""

from __future__ import annotations

from ipax import Options, Status, solve
from ipax.ipm.filter_ls import FilterLineSearch
from ipax.options import LineSearchOptions
from ipax.testing.problems import HS71
from tests._helpers import array


def test_ftype_branch_is_gated_by_the_driver_supplied_theta_min(namespace, monkeypatch):
    # The gate is only real if the driver's θ_min actually reaches it. Recording
    # the trials where the θ_min conjunct *flips* the verdict away from the old
    # switching-only answer is what makes this non-vacuous: an unwired θ_min
    # (or one left at +inf) would never flip anything and ``seen`` would be empty.
    seen: list[tuple[float, float]] = []
    original = FilterLineSearch._is_ftype

    def spy(self, dphi, alpha, theta0, theta_min):
        verdict = original(self, dphi, alpha, theta0, theta_min)
        if verdict != self._switching(dphi, alpha, theta0):
            seen.append((theta0, theta_min))
        return verdict

    monkeypatch.setattr(FilterLineSearch, "_is_ftype", spy)
    result = solve(
        HS71(namespace),
        array(namespace, [1.0, 5.0, 5.0, 1.0]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            line_search=LineSearchOptions(ftype_requires_theta_min=True),
        ),
    )

    assert result.status is Status.OPTIMAL
    # At least one trial was reclassified θ-type purely because θ0 > θ_min —
    # i.e. the gate changed a real decision on a real solve.
    assert seen, "theta_min never altered an f-type classification"
    assert all(theta0 > theta_min for theta0, theta_min in seen)
