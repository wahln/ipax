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

"""Regression: the driver wires the Wächter & Biegler eq. (23) α_min.

Before this, ``alpha_min_frac`` was a flat ``1e-8`` backtracking floor, so a
hopeless ray was halved 27 times before conceding to restoration regardless of
the iterate. eq. (23) derives the threshold from the current θ and ∇φᵀd, and
needs a θ_min the driver must compute from the *initial* constraint violation
(θ_min = 1e-4·max(1, θ(x_0)), mirroring the existing θ_max guard) and thread into
every line search. The unit tests in ``tests/unit/test_alpha_min.py`` pin the
formula; these pin the driver-side plumbing that feeds it, on every backend.
"""

from __future__ import annotations

from ipax import Options, Status, solve
from ipax.ipm.filter_ls import FilterLineSearch
from ipax.testing.problems import HS71, BoundConstrainedQP
from tests._helpers import array


def test_driver_derives_theta_min_from_the_initial_violation(namespace, monkeypatch):
    # HS71 starts infeasible, so θ(x_0) > 1 and θ_min scales with it rather than
    # sitting at the bare 1e-4·1. Capturing it at the seam is what proves the
    # driver computes it from the initial iterate instead of passing a constant.
    seen: list[float] = []
    original = FilterLineSearch.search

    def spy(self, **kwargs):
        seen.append(kwargs["theta_min"])
        return original(self, **kwargs)

    monkeypatch.setattr(FilterLineSearch, "search", spy)
    problem = HS71(namespace)
    x0 = array(namespace, [1.0, 5.0, 5.0, 1.0])
    result = solve(problem, x0, options=Options(hessian="exact", linsolve="dense"))

    assert result.status is Status.OPTIMAL
    assert seen, "the filter line search never ran"
    # Fixed for the whole run — derived once from x_0, never re-taken from the
    # current iterate (which would make α_min drift as θ collapses).
    assert len(set(seen)) == 1
    # Scaled by the initial violation rather than pinned at the 1e-4·1 floor.
    # Asserting the *shape* keeps the test from re-implementing the driver's own
    # θ metric; the first row of the history is θ(x_0), which anchors it.
    theta0 = result.history[0].theta
    assert theta0 > 1.0, "HS71 from this x_0 should start infeasible"
    assert seen[0] == 1e-4 * theta0


def test_theta_min_floors_at_one_for_a_feasible_start(namespace, monkeypatch):
    # max(1, θ(x_0)) — a feasible or near-feasible start must not collapse θ_min
    # toward 0, which would permanently disable the eq. (23) switching term.
    seen: list[float] = []
    original = FilterLineSearch.search

    def spy(self, **kwargs):
        seen.append(kwargs["theta_min"])
        return original(self, **kwargs)

    monkeypatch.setattr(FilterLineSearch, "search", spy)
    result = solve(
        BoundConstrainedQP(namespace),
        array(namespace, [0.25, 0.75]),
        options=Options(hessian="exact", linsolve="dense"),
    )

    assert result.status is Status.OPTIMAL
    assert seen and seen[0] == 1e-4
