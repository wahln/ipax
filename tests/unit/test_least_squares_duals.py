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

"""Least-squares estimate of the equality multipliers.

``min_y ‖∇f + Aᵀy‖`` — the multipliers that come closest to satisfying
stationarity at the current point. Used to repair multipliers that have drifted
(S2MPJ ``COOLHANS`` reaches ``‖y‖∞ ≈ 1.1e6`` on a problem whose objective is
identically zero, which forces ``y* = 0``), and the estimate ``init.py`` has
documented as "not currently implemented" since the module was written.

Solved matrix-free through the normal equations ``(A Aᵀ + δI) y = −A ∇f``: the
Array API has no ``lstsq`` (invariant #2), and forming ``A`` densely would break
the sparsity-is-an-adapter rule at radiotherapy scale (invariant #4). Only
``matvec``/``rmatvec`` are used, so every backend and every operator flavour
works the same way.
"""

from __future__ import annotations

import pytest

from ipax.backend.operators import Dense, LinearOperator
from ipax.ipm.init import least_squares_duals
from tests._helpers import array, assert_allclose


def _op(namespace, rows) -> LinearOperator:
    return Dense(array(namespace, rows))


def test_zero_objective_gradient_gives_zero_multipliers(namespace):
    # The class-A case: an objective-free system (min 0 s.t. c(x)=0) has
    # ∇f ≡ 0, so stationarity Aᵀy = 0 is solved exactly by y = 0.
    A = _op(namespace, [[2.0, 0.0], [0.0, 3.0]])
    grad = array(namespace, [0.0, 0.0])

    y = least_squares_duals(A, grad, xp=namespace)

    assert_allclose(namespace, y, array(namespace, [0.0, 0.0]), atol=1e-10)


def test_recovers_the_exact_multipliers_when_the_system_is_consistent(namespace):
    # A = I, so ∇f + Aᵀy = 0 has the exact solution y = −∇f.
    A = _op(namespace, [[1.0, 0.0], [0.0, 1.0]])
    grad = array(namespace, [3.0, -5.0])

    y = least_squares_duals(A, grad, xp=namespace)

    assert_allclose(namespace, y, array(namespace, [-3.0, 5.0]), rtol=1e-6)


def test_solves_the_least_squares_problem_when_overdetermined(namespace):
    # One equality in two variables: Aᵀy = [y, 2y] cannot match ∇f = [1, 0]
    # exactly. The least-squares solution minimises ‖∇f + Aᵀy‖:
    #   min_y (1 + y)² + (2y)²  ->  y = −1/5.
    A = _op(namespace, [[1.0, 2.0]])
    grad = array(namespace, [1.0, 0.0])

    y = least_squares_duals(A, grad, xp=namespace)

    assert_allclose(namespace, y, array(namespace, [-0.2]), rtol=1e-5)


def test_a_rank_deficient_jacobian_still_returns_finite_multipliers(namespace):
    # Duplicated rows make A Aᵀ singular; the regularisation has to carry it,
    # because a degenerate Jacobian is exactly when multipliers drift.
    A = _op(namespace, [[1.0, 1.0], [1.0, 1.0]])
    grad = array(namespace, [2.0, 2.0])

    y = least_squares_duals(A, grad, xp=namespace)

    assert namespace.all(namespace.isfinite(y))
    # Both rows are identical, so the two multipliers must be too.
    assert_allclose(namespace, y[0:1], y[1:2], rtol=1e-6)


def test_no_equalities_is_an_empty_estimate(namespace):
    A = _op(namespace, [[1.0, 2.0]])
    grad = array(namespace, [1.0, 0.0])

    y = least_squares_duals(A, grad, xp=namespace, m_eq=0)

    assert y.shape == (0,)


def test_the_estimate_beats_a_drifted_multiplier(namespace):
    # The property the repair relies on: whatever the incoming multipliers are,
    # the estimate cannot be worse at satisfying stationarity.
    A = _op(namespace, [[2.0, 0.0], [0.0, 3.0]])
    grad = array(namespace, [0.0, 0.0])
    drifted = array(namespace, [1.1e6, -4.0e5])

    y = least_squares_duals(A, grad, xp=namespace)

    def residual(mult):
        return float(namespace.max(namespace.abs(grad + A.rmatvec(mult))))

    assert residual(y) < residual(drifted)


def test_matrix_free_operators_are_supported(namespace):
    # Only matvec/rmatvec may be used: a dense A is unaffordable at RT scale.
    class _MatrixFree(LinearOperator):
        shape = (1, 2)

        def matvec(self, v):
            return namespace.reshape(v[0:1] + 2.0 * v[1:2], (1,))

        def rmatvec(self, v):
            return namespace.concat([v, 2.0 * v])

        def matmat(self, V):  # pragma: no cover - not used by the estimator
            raise AssertionError("the estimator must stay matrix-free")

    grad = array(namespace, [1.0, 0.0])

    y = least_squares_duals(_MatrixFree(), grad, xp=namespace)

    assert_allclose(namespace, y, array(namespace, [-0.2]), rtol=1e-5)


def test_a_huge_gradient_does_not_produce_non_finite_multipliers(namespace):
    A = _op(namespace, [[1e-8, 0.0]])
    grad = array(namespace, [1e12, 0.0])

    y = least_squares_duals(A, grad, xp=namespace)

    assert namespace.all(namespace.isfinite(y))


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_scale_invariance_of_the_estimate(namespace, scale):
    # Scaling the objective gradient scales the multipliers linearly.
    A = _op(namespace, [[2.0, 0.0], [0.0, 3.0]])
    base = least_squares_duals(A, array(namespace, [1.0, 1.0]), xp=namespace)
    scaled = least_squares_duals(A, array(namespace, [scale, scale]), xp=namespace)

    assert_allclose(namespace, scaled, base * scale, rtol=1e-4, atol=1e-8)
