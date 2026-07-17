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

"""The two W&B line-search refinements are OPT-IN; the defaults are inert.

Both were measured on the full S2MPJ corpus and both *lost* against the shipped
baseline — eq. (23) α_min by −4, the θ_min f-type gate by −10 (attributed
independently; see docs/benchmarks/s2mpj.md). They stay in the tree because they
are faithful to the paper and may suit central-path-following (radiotherapy-like)
workloads, but they must not touch the default solver.

These tests pin exactly that: with a default ``LineSearchOptions`` the behaviour
is byte-for-byte the pre-eq.(23) behaviour.
"""

from __future__ import annotations

import pytest

from ipax.ipm.filter_ls import FilterLineSearch
from ipax.options import LineSearchOptions

_DEFAULT = LineSearchOptions()


def test_alpha_min_defaults_to_the_flat_floor():
    # gamma_alpha=None ⇒ α_min is the flat ``alpha_min_frac``, whatever θ0/dphi
    # are. The eq. (23) branch structure must not leak into the default.
    ls = FilterLineSearch(_DEFAULT)

    assert _DEFAULT.gamma_alpha is None
    assert _DEFAULT.alpha_min_frac == 1e-8
    for dphi, theta0 in [(-1.0, 1.0), (5.0, 0.0), (-1e-6, 1e-6), (-1.0, 0.0)]:
        assert ls._alpha_min(dphi=dphi, theta0=theta0, theta_min=1e-4) == 1e-8


def test_alpha_min_frac_keeps_its_original_meaning_as_an_absolute_floor():
    # It is a step size, not a safety factor: no (0,1) restriction, and it is
    # returned verbatim.
    ls = FilterLineSearch(LineSearchOptions(alpha_min_frac=1e-12))

    assert ls._alpha_min(dphi=-1.0, theta0=1.0, theta_min=1e-4) == 1e-12


def test_opting_in_to_gamma_alpha_enables_eq23():
    ls = FilterLineSearch(LineSearchOptions(gamma_alpha=0.05))

    # θ0 = 1 > θ_min, descent: γ_α·min{γ_θ, γ_φ·θ/(−∇φᵀd)} = 0.05·1e-5.
    assert ls._alpha_min(dphi=-1.0, theta0=1.0, theta_min=1e-4) == pytest.approx(5e-7)
    # ... and the floor still bounds it from below at a feasible iterate, where
    # eq. (23) is exactly 0 and the backtracking loop would not terminate.
    assert ls._alpha_min(dphi=-1.0, theta0=0.0, theta_min=1e-4) == 1e-8


def test_ftype_defaults_to_switching_only():
    # The θ_min conjunct must not apply by default: an infeasible iterate with a
    # descent direction stays f-type, i.e. Armijo-governed, exactly as in 0.7.0.
    ls = FilterLineSearch(_DEFAULT)

    assert _DEFAULT.ftype_requires_theta_min is False
    assert ls._is_ftype(dphi=-1e6, alpha=1.0, theta0=1.0, theta_min=0.5)
    assert ls._reject_reason(0.1, 500.0, 1.0, 1.0, -1e6, 1.0, 1e10, 0.5, []) == "armijo"


def test_opting_in_to_ftype_theta_min_enables_the_gate():
    ls = FilterLineSearch(LineSearchOptions(ftype_requires_theta_min=True))

    assert not ls._is_ftype(dphi=-1e6, alpha=1.0, theta0=1.0, theta_min=0.5)
    # θ-progress is now acceptable where Armijo was previously demanded.
    assert ls._reject_reason(0.1, 500.0, 1.0, 1.0, -1e6, 1.0, 1e10, 0.5, []) is None


def test_augmentation_fix_is_inert_while_the_theta_min_gate_is_off():
    # ``_augments_filter`` tests ¬(switching ∧ Armijo) where ipax previously
    # tested ¬switching. With the gate off these coincide *identically*: an
    # accepted trial for which switching holds must have passed Armijo (it took
    # the f-type branch), so the Armijo conjunct can never change the answer.
    # This is why the augmentation repair rides along un-gated.
    ls = FilterLineSearch(_DEFAULT)

    for dphi in (-1e6, -1.0, 1.0, 0.0):
        for phi_t in (-1e3, 0.5, 500.0):
            switching = ls._switching(dphi, 1.0, 1.0)
            accepted = (
                ls._reject_reason(0.1, phi_t, 1.0, 1.0, dphi, 1.0, 1e10, 1e10, [])
                is None
            )
            if not accepted:
                continue
            assert ls._augments_filter(
                phi_t=phi_t, phi0=1.0, dphi=dphi, alpha=1.0, theta0=1.0
            ) == (not switching)
