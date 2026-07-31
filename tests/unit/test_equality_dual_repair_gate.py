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

"""The divergence gate on the least-squares equality-multiplier repair.

``_repair_equality_duals`` was written for the restoration path, where it fires
whenever the least-squares estimate is *strictly* closer to stationarity. Firing
that eagerly on every accepted step is measurably harmful: it overwrites the
legitimate primal–dual coupling, and near convergence it prevents the dual
iterate from converging at all (measured: healthy S2MPJ equality problems
degrade from ``optimal`` to ``acceptable`` because the reset pins the dual
residual at ~1e-8).

The gate makes the repair fire only on *divergence* — when the current
multipliers are worse for stationarity by more than a factor ``factor`` — which
is self-limiting: once repaired, the ratio test stops tripping.

``factor=1.0`` reproduces the original "strictly better" rule exactly, so the
restoration call site is unaffected.
"""

from __future__ import annotations

from ipax.backend.namespace import array_namespace
from ipax.backend.operators import Dense
from ipax.ipm.driver import IPMDriver
from ipax.options import Options
from tests._helpers import array, norm_inf


def _driver(namespace, *, jac_rows, grad):
    """A driver stub: the repair only needs the namespace, Jacobian, gradient."""
    driver = IPMDriver.__new__(IPMDriver)  # bypass __init__
    driver._xp = array_namespace(array(namespace, [0.0]))
    driver._options = Options()
    operator = Dense(array(namespace, jac_rows))
    driver._eq_jac = lambda x: operator
    driver._gradient = lambda x: array(namespace, grad)
    return driver


def _overdetermined(namespace):
    """One multiplier against two gradient components.

    ``A = [[1, 0]]`` and ``∇f = [1, 5]`` give ``‖∇f + Aᵀy‖∞ = max(|1+y|, 5)``,
    minimized at ``y = -1`` with residual 5. A non-zero least-squares residual
    is what makes the *ratio* observable — on a consistent system the estimate
    is exact and every factor trips.
    """
    return _driver(namespace, jac_rows=[[1.0, 0.0]], grad=[1.0, 5.0])


def test_factor_one_reproduces_the_strictly_better_rule(namespace):
    # The restoration call site's contract: repair whenever the estimate is
    # closer to stationarity at all. y = 100 gives residual 101 > 5.
    driver = _overdetermined(namespace)
    x = array(namespace, [0.0, 0.0])

    repaired = driver._repair_equality_duals(
        x, array(namespace, [100.0]), 1, factor=1.0
    )

    assert abs(norm_inf(driver._xp, repaired) - 1.0) < 1e-6  # the estimate, -1


def test_a_large_factor_keeps_healthy_multipliers(namespace):
    # y = -1 is already the least-squares estimate: residual 5 either way, so
    # the ratio is 1 and a factor of 10 must leave it alone. This is the case
    # that firing unconditionally got wrong.
    driver = _overdetermined(namespace)
    x = array(namespace, [0.0, 0.0])
    healthy = array(namespace, [-1.0])

    repaired = driver._repair_equality_duals(x, healthy, 1, factor=10.0)

    assert repaired is healthy


def test_a_large_factor_still_repairs_gross_divergence(namespace):
    # The pathology: ‖y‖ orders of magnitude past what the iterate can justify
    # (S2MPJ NONSCOMPNE reaches 1.0e8 on a zero-objective system). Here the
    # irreducible residual is 5, so clearing a 1e10 ratio needs > 5e10.
    driver = _overdetermined(namespace)
    x = array(namespace, [0.0, 0.0])

    repaired = driver._repair_equality_duals(
        x, array(namespace, [1e12]), 1, factor=1e10
    )

    assert abs(norm_inf(driver._xp, repaired) - 1.0) < 1e-6


def test_the_gate_is_a_ratio_against_the_irreducible_residual(namespace):
    # Conservatism contract: a multiplier 1e9 from zero is *kept* when the
    # least-squares residual is itself 5, because 1e9 < 1e10 * 5. The gate
    # measures divergence relative to what the iterate can justify, not an
    # absolute magnitude — this is what leaves healthy trajectories alone.
    driver = _overdetermined(namespace)
    x = array(namespace, [0.0, 0.0])
    large_but_within_ratio = array(namespace, [1e9])

    repaired = driver._repair_equality_duals(x, large_but_within_ratio, 1, factor=1e10)

    assert repaired is large_but_within_ratio


def test_zero_objective_divergence_is_repaired_to_zero(namespace):
    # ∇f ≡ 0 forces y* = 0, so any non-zero multiplier is pure noise from the
    # rank-deficient dual system. The estimate is exactly zero.
    driver = _driver(namespace, jac_rows=[[2.0, 0.0], [0.0, 3.0]], grad=[0.0, 0.0])
    x = array(namespace, [0.0, 0.0])

    repaired = driver._repair_equality_duals(
        x, array(namespace, [1e8, -1e8]), 2, factor=1e10
    )

    assert norm_inf(driver._xp, repaired) < 1e-8


def test_no_equalities_is_a_no_op(namespace):
    driver = _overdetermined(namespace)
    x = array(namespace, [0.0, 0.0])
    empty = array(namespace, [])

    assert driver._repair_equality_duals(x, empty, 0, factor=1e10) is empty


def test_a_failing_estimate_never_replaces_the_multipliers(namespace):
    # A repair must never be the thing that fails the solve.
    driver = _overdetermined(namespace)

    def _boom(_x):
        raise RuntimeError("jacobian unavailable")

    driver._eq_jac = _boom
    x = array(namespace, [0.0, 0.0])
    current = array(namespace, [1e9])

    assert driver._repair_equality_duals(x, current, 1, factor=1e10) is current


def test_the_default_factor_is_the_restoration_rule(namespace):
    # Callers that do not pass a factor keep the pre-gate behaviour.
    driver = _overdetermined(namespace)
    x = array(namespace, [0.0, 0.0])

    repaired = driver._repair_equality_duals(x, array(namespace, [100.0]), 1)

    assert abs(norm_inf(driver._xp, repaired) - 1.0) < 1e-6


def test_a_non_finite_estimate_keeps_the_current_multipliers(namespace, monkeypatch):
    # ``least_squares_duals`` guarantees a finite return today, so this contract
    # is the driver refusing to *depend* on that: a repair compares stationarity
    # residuals, and NaN comparisons are all false, which would silently adopt
    # an unusable estimate. Stubbed at the module boundary because the real
    # estimator cannot produce the input.
    import ipax.ipm.driver as driver_module

    driver = _overdetermined(namespace)
    x = array(namespace, [0.0, 0.0])
    current = array(namespace, [1e12])
    nan = float("nan")

    monkeypatch.setattr(
        driver_module,
        "least_squares_duals",
        lambda *a, **k: array(namespace, [nan]),
    )

    assert driver._repair_equality_duals(x, current, 1, factor=1e10) is current
