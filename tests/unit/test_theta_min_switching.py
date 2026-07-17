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

"""The θ_min half of the f-type test (Wächter & Biegler 2006, Algorithm A).

A trial is f-type — and so subject to the Armijo condition — only when the
switching condition (eq. 19) holds **and** the current iterate is already nearly
feasible, θ ≤ θ_min. ipax previously keyed the branch on the switching condition
alone, so at an infeasible iterate it demanded Armijo decrease on φ where W&B
(and IPOPT, ``FilterLSAcceptor::CheckAcceptabilityOfTrialPoint``) ask only for
sufficient decrease in θ *or* φ. That made the search reject trials the filter
method accepts, backtracking further than the algorithm requires.
"""

from __future__ import annotations

from ipax.ipm.filter_ls import FilterLineSearch
from ipax.options import LineSearchOptions

_OPTS = LineSearchOptions()


def test_is_ftype_requires_both_switching_and_theta_below_theta_min():
    ls = FilterLineSearch(_OPTS)
    # A strong descent direction: the eq. (19) switching condition holds at α = 1.
    assert ls._switching(dphi=-1e6, alpha=1.0, theta0=1.0)

    # Near-feasible (θ ≤ θ_min) ⇒ f-type.
    assert ls._is_ftype(dphi=-1e6, alpha=1.0, theta0=1.0, theta_min=10.0)
    # Infeasible (θ > θ_min) ⇒ NOT f-type, despite switching holding.
    assert not ls._is_ftype(dphi=-1e6, alpha=1.0, theta0=1.0, theta_min=0.5)
    # Switching failing (ascent) ⇒ never f-type, regardless of θ_min.
    assert not ls._is_ftype(dphi=1.0, alpha=1.0, theta0=1.0, theta_min=10.0)


def test_armijo_is_not_demanded_above_theta_min():
    ls = FilterLineSearch(_OPTS)
    # φ rises well past the Armijo bound, but θ drops from 1.0 to 0.1 — decisive
    # sufficient decrease in θ (eq. 20). Above θ_min that is an acceptable
    # θ-type step; the old switching-only branch called it "armijo" and rejected.
    reason = ls._reject_reason(
        theta_t=0.1,
        phi_t=500.0,
        theta0=1.0,
        phi0=1.0,
        dphi=-1e6,
        alpha=1.0,
        theta_max=1e10,
        theta_min=0.5,  # θ0 = 1.0 > θ_min ⇒ θ-type branch
        entries=[],
    )
    assert reason is None

    # The same trial at an iterate *below* θ_min stays f-type and must still
    # face Armijo — this is the branch the change must not weaken.
    assert (
        ls._reject_reason(
            theta_t=0.1,
            phi_t=500.0,
            theta0=1.0,
            phi0=1.0,
            dphi=-1e6,
            alpha=1.0,
            theta_max=1e10,
            theta_min=10.0,  # θ0 = 1.0 ≤ θ_min ⇒ f-type branch
            entries=[],
        )
        == "armijo"
    )


def test_theta_type_branch_above_theta_min_still_demands_decrease():
    # Dropping the Armijo requirement above θ_min must not accept *anything*:
    # the eq. (20) sufficient-decrease test in θ or φ still governs.
    ls = FilterLineSearch(_OPTS)

    reason = ls._reject_reason(
        theta_t=2.0,  # θ worsens
        phi_t=500.0,  # φ worsens
        theta0=1.0,
        phi0=1.0,
        dphi=-1e6,
        alpha=1.0,
        theta_max=1e10,
        theta_min=0.5,
        entries=[],
    )
    assert reason == "no-decrease"


def test_accepted_step_augments_unless_switching_and_armijo_both_hold():
    # Filter bookkeeping is a *separate* predicate from the acceptance branch
    # (W&B Step 5; IPOPT ``UpdateForNextIteration``: augment on
    # ``!IsFtype(α) || !ArmijoHolds(α)``, where its IsFtype is switching-only).
    # It deliberately does NOT consult θ_min: a step accepted via eq. (20) above
    # θ_min that nonetheless satisfies switching *and* Armijo is not recorded.
    ls = FilterLineSearch(_OPTS)

    # switching ∧ Armijo ⇒ no augmentation, θ_min irrelevant.
    for theta_min in (0.5, 10.0):
        assert not ls._augments_filter(
            phi_t=-1e3, phi0=1.0, dphi=-1e6, alpha=1.0, theta0=1.0
        ), theta_min

    # switching ∧ ¬Armijo ⇒ augment. Reachable only above θ_min, where eq. (20)
    # can accept a step the Armijo test would have refused.
    assert ls._augments_filter(phi_t=500.0, phi0=1.0, dphi=-1e6, alpha=1.0, theta0=1.0)

    # ¬switching (ascent) ⇒ augment regardless of φ.
    assert ls._augments_filter(phi_t=-1e3, phi0=1.0, dphi=1.0, alpha=1.0, theta0=1.0)


def test_theta_progress_step_above_theta_min_augments_the_filter():
    # End-to-end through search(): a trial accepted above θ_min on θ-progress
    # while φ worsens fails Armijo, so the filter must record (θ0, φ0) —
    # otherwise the method loses the entry that keeps it from cycling.
    line_search = FilterLineSearch(_OPTS)

    result = line_search.search(
        alpha_max=1.0,
        theta0=1.0,
        phi0=1.0,
        dphi=-1e6,  # switching holds ...
        theta_max=1e10,
        theta_min=0.5,  # ... but θ0 > θ_min ⇒ accepted via eq. (20)
        eval_point=lambda alpha: (0.1, 500.0),  # θ-progress, φ worsens
        entries=[],
        soc=None,
    )

    assert result.accepted
    assert result.alpha == 1.0
    assert result.n_trials == 1
    assert result.augment  # Armijo fails ⇒ recorded


def test_genuine_ftype_step_does_not_augment_the_filter():
    # The mirror: switching + Armijo ⇒ f-type ⇒ the filter is left alone.
    line_search = FilterLineSearch(_OPTS)

    result = line_search.search(
        alpha_max=1.0,
        theta0=1.0,
        phi0=1.0,
        dphi=-1e6,
        theta_max=1e10,
        theta_min=10.0,  # θ0 ≤ θ_min ⇒ f-type branch
        eval_point=lambda alpha: (0.1, -1e3),  # Armijo satisfied
        entries=[],
        soc=None,
    )

    assert result.accepted
    assert not result.augment
