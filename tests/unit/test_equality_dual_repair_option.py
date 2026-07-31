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

"""``equality_dual_repair`` is OPT-IN and its default must be inert.

The accepted-step repair was validated on a 57-problem S2MPJ subset (6 gains, 0
regressions), but the full-corpus three-route sweep that decides defaults in
this project has not run for it yet. Until it does, a default ``Options()``
must behave exactly as before — these tests pin that.
"""

from __future__ import annotations

import pytest

from ipax.options import Options, RegularizationOptions


def test_the_repair_is_disabled_by_default():
    assert RegularizationOptions().equality_dual_repair is None
    assert Options().regularization.equality_dual_repair is None


def test_a_factor_below_one_is_rejected():
    # ``factor`` multiplies the least-squares residual in the ratio test, so a
    # value below 1 would repair multipliers that are already the better ones —
    # the eager behaviour the gate exists to prevent.
    with pytest.raises(ValueError, match="equality_dual_repair"):
        RegularizationOptions(equality_dual_repair=0.5)


def test_a_non_finite_factor_is_rejected():
    with pytest.raises(ValueError, match="equality_dual_repair"):
        RegularizationOptions(equality_dual_repair=float("inf"))


def test_a_factor_of_one_is_accepted_as_the_eager_setting():
    # The boundary is legal: it means "repair whenever the estimate is better",
    # the restoration path's own rule applied on every accepted step.
    assert RegularizationOptions(equality_dual_repair=1.0).equality_dual_repair == 1.0


def test_the_validated_threshold_is_accepted():
    opts = RegularizationOptions(equality_dual_repair=1e10)

    assert opts.equality_dual_repair == 1e10
