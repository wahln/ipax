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

"""The adaptive minimum step size — Wächter & Biegler 2006, eq. (23).

``FilterLineSearch`` operates on plain Python floats and callables, so these
tests carry no array namespace; the multi-backend obligation for this change is
discharged by the integration/regression layer.
"""

from __future__ import annotations

import pytest

from ipax.ipm.filter_ls import _ALPHA_MIN_FLOOR, FilterLineSearch
from ipax.options import LineSearchOptions

# Constants chosen so every eq. (23) branch lands well above _ALPHA_MIN_FLOOR,
# isolating the formula from the finite-termination backstop.
_ISOLATED = LineSearchOptions(
    alpha_min_frac=0.5,  # γ_α
    gamma_theta=0.1,
    gamma_phi=0.1,
    s_theta=2.0,
    s_phi=2.0,
)


def test_alpha_min_ascent_branch_is_gamma_alpha_times_gamma_theta():
    # eq. (23), third case: with ∇φᵀd ≥ 0 there is no Armijo-reachable step to
    # protect, so α_min collapses to γ_α·γ_θ regardless of θ.
    ls = FilterLineSearch(_ISOLATED)

    assert ls._alpha_min(dphi=2.0, theta0=1.0, theta_min=0.5) == pytest.approx(0.05)
    # dphi == 0.0 is not descent either (the guard is a strict `< 0`).
    assert ls._alpha_min(dphi=0.0, theta0=1.0, theta_min=0.5) == pytest.approx(0.05)


def test_alpha_min_descent_above_theta_min_uses_two_terms():
    # eq. (23), second case (∇φᵀd < 0, θ > θ_min): min{γ_θ, γ_φ·θ/(−∇φᵀd)}.
    # γ_φ·θ/(−∇φᵀd) = 0.1·1.0/2.0 = 0.05 < γ_θ = 0.1  ⇒  α_min = 0.5·0.05.
    ls = FilterLineSearch(_ISOLATED)

    assert ls._alpha_min(dphi=-2.0, theta0=1.0, theta_min=0.5) == pytest.approx(0.025)


def test_alpha_min_descent_below_theta_min_adds_the_switching_term():
    # eq. (23), first case (∇φᵀd < 0, θ ≤ θ_min): the δ·θ^{s_θ}/(−∇φᵀd)^{s_φ}
    # term joins the min. θ = 0.1, dphi = −2:
    #   γ_φ·θ/(−∇φᵀd)                = 0.1·0.1/2       = 0.005
    #   δ·θ^{s_θ}/(−∇φᵀd)^{s_φ}      = 0.1²/2²         = 0.0025   ← binds
    ls = FilterLineSearch(_ISOLATED)

    below = ls._alpha_min(dphi=-2.0, theta0=0.1, theta_min=1.0)
    assert below == pytest.approx(0.5 * 0.0025)

    # The same point with θ > θ_min drops the third term, so α_min is larger —
    # this is what makes θ_min load-bearing rather than decorative.
    above = ls._alpha_min(dphi=-2.0, theta0=0.1, theta_min=0.05)
    assert above == pytest.approx(0.5 * 0.005)
    assert above > below


def test_alpha_min_matches_eq23_at_the_shipped_defaults():
    # IPOPT parity at the shipped constants (γ_α = alpha_min_frac = 0.05,
    # γ_θ = 1e-5): a descent direction at θ = 1, dphi = −1 sits above the
    # default θ_min, so α_min = γ_α·min{1e-5, 1e-5·1/1} = 5e-7 — nearly two
    # decades above the 1e-8 floor this replaces.
    ls = FilterLineSearch(LineSearchOptions())

    assert ls._alpha_min(dphi=-1.0, theta0=1.0, theta_min=1e-4) == pytest.approx(5e-7)


def test_alpha_min_floors_at_a_feasible_iterate():
    # At θ0 = 0 with a descent direction, eq. (23) evaluates to *exactly* zero
    # (both θ-bearing terms vanish). Taken literally that makes the backtracking
    # loop non-terminating, so α_min is floored — which also keeps the feasible
    # iterate behaving exactly as it did before eq. (23) landed.
    ls = FilterLineSearch(LineSearchOptions())

    assert ls._alpha_min(dphi=-1.0, theta0=0.0, theta_min=1e-4) == _ALPHA_MIN_FLOOR


def test_alpha_min_survives_an_underflowing_directional_derivative():
    # (−∇φᵀd)^{s_φ} underflows to 0.0 for a tiny dphi; dividing by it would
    # raise ZeroDivisionError and take down the line search. With the switching
    # term dropped, γ_θ binds: α_min = 0.05·1e-5.
    ls = FilterLineSearch(LineSearchOptions())

    assert ls._alpha_min(dphi=-1e-200, theta0=1e-5, theta_min=1e-4) == pytest.approx(
        5e-7
    )


def test_alpha_min_survives_an_overflowing_directional_derivative():
    # The mirror case: (−∇φᵀd)^{s_φ} overflows, so the switching term → 0 and
    # the floor takes over. Must not raise OverflowError.
    ls = FilterLineSearch(LineSearchOptions())

    assert ls._alpha_min(dphi=-1e200, theta0=1e-5, theta_min=1e-4) == _ALPHA_MIN_FLOOR


def test_search_hands_off_to_restoration_at_the_eq23_alpha_min():
    # An unacceptable ray backtracks only down to the eq. (23) α_min = 5e-7
    # (see the defaults test above) rather than the old flat 1e-8, so the
    # restoration hand-off costs 21 trials instead of 27.
    line_search = FilterLineSearch(LineSearchOptions())

    result = line_search.search(
        alpha_max=1.0,
        theta0=1.0,
        phi0=1.0,
        dphi=-1.0,
        theta_max=1e10,
        theta_min=1e-4,
        eval_point=lambda alpha: (2.0, 5.0),  # never acceptable
        entries=[],
        soc=None,
    )

    assert not result.accepted
    assert result.restoration
    assert result.alpha == pytest.approx(5e-7)
    assert result.n_trials == 21


def test_free_search_keeps_the_flat_floor():
    # eq. (23) is defined through ∇φᵀd and the Armijo/switching tests, none of
    # which the free-mode search uses — so it keeps the flat backstop and its
    # behaviour is unchanged by this parity fix.
    line_search = FilterLineSearch(LineSearchOptions())

    result = line_search.search_free(
        alpha_max=1.0,
        theta_max=1e10,
        eval_point=lambda alpha: (2.0, 5.0),
        entries=[(1.0, 1.0)],
        margin=0.0,
    )

    assert not result.accepted
    assert result.restoration
    assert result.alpha == _ALPHA_MIN_FLOOR


def test_alpha_min_frac_is_validated_to_the_open_unit_interval():
    # γ_α ∈ (0, 1) (IPOPT bounds alpha_min_frac the same way): γ_α ≤ 0 would
    # zero out α_min (no restoration hand-off), and γ_α ≥ 1 would make α_min
    # exceed the switching-condition bound it is meant to sit safely under.
    for bad in (0.0, -0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="alpha_min_frac"):
            LineSearchOptions(alpha_min_frac=bad)

    LineSearchOptions(alpha_min_frac=0.05)  # the default is valid
