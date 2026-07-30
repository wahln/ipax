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

"""Equality multipliers diverge on a rank-deficient dual system.

On a zero-objective feasibility problem (``min 0`` s.t. ``c(x) = 0``) the exact
multipliers are ``y* = 0``: stationarity is ``∇f + Aᵀy = 0`` with ``∇f ≡ 0``. Yet
S2MPJ ``NONSCOMPNE`` returns ``‖y_eq‖∞ = 1.0e8`` and ``VANDERM1`` ``3.0e4``.

The mechanism, measured 2026-07-30: ``NONSCOMPNE``'s equality Jacobian is
**rank 24 of 25** — exactly singular at the starting point — so the dual block is
under-determined and ``‖Δy‖`` is bounded only by the primal–dual regularization,
``‖c‖ / δ_c ≈ 1e-1 / 1e-8 = 1e7``. The trace shows ``y`` going from 0 to 7.9e7 in
the *single* step from iteration 0 to 1, and never recovering. The filter cannot
see it: acceptance tests (θ, φ) only, so a step that improves feasibility while
destroying stationarity is accepted at full length. The poisoned multipliers then
feed the L-BFGS Lagrangian pairs and every subsequent step system.

The reproducer below is that shape in miniature: duplicated equality rows make
the Jacobian exactly rank-deficient, and the objective is identically zero.
"""

from __future__ import annotations

import numpy as np

import ipax
from ipax.options import Options, RegularizationOptions
from ipax.problem.base import Problem

_REPAIR = 1e10  # the A/B-validated divergence factor


class _RankDeficientFeasibility(Problem):
    """``min 0`` subject to a duplicated equality — Jacobian rank 1 of 2.

    The duplicate row leaves the multiplier undetermined along ``(1, -1)``: any
    ``y`` with ``y0 + y1`` fixed satisfies the same stationarity residual, so the
    dual system is singular exactly as in ``NONSCOMPNE``.
    """

    n_vars = 2

    def objective(self, x):
        return 0.0 * x[0]

    def gradient(self, x):
        return 0.0 * x

    def eq_constraints(self, x):
        residual = x[0] ** 2 + x[1] - 2.0
        return np.asarray([residual, residual])

    def eq_jacobian(self, x):
        row = [2.0 * x[0], 1.0]
        return np.asarray([row, row])


def _solve(*, repair: float | None):
    problem = _RankDeficientFeasibility()
    options = Options(
        hessian="lbfgs",
        linsolve="dense",
        max_iter=200,
        regularization=RegularizationOptions(equality_dual_repair=repair),
    )
    return ipax.solve(problem, np.asarray([1.0, 0.0]), options=options)


def test_the_repair_keeps_the_multipliers_at_their_exact_value():
    # ∇f ≡ 0 forces y* = 0. With the gate enabled the returned multipliers must
    # stay near it instead of drifting into the Jacobian's null space.
    result = _solve(repair=_REPAIR)

    assert result.y_eq is not None
    assert float(np.max(np.abs(np.asarray(result.y_eq)))) < 1e-6


def test_the_repair_reaches_a_feasible_stationary_point():
    result = _solve(repair=_REPAIR)

    assert result.success, result.message
    assert result.primal_infeasibility < 1e-8
    assert result.dual_infeasibility < 1e-8


def test_the_option_is_inert_when_disabled():
    # The opt-in contract: leaving it unset must not change the solve. Both
    # arms are compared on the trajectory length and the returned point.
    disabled = _solve(repair=None)
    default = ipax.solve(
        _RankDeficientFeasibility(),
        np.asarray([1.0, 0.0]),
        options=Options(hessian="lbfgs", linsolve="dense", max_iter=200),
    )

    assert disabled.status is default.status
    assert disabled.n_iter == default.n_iter
    np.testing.assert_allclose(
        np.asarray(disabled.x), np.asarray(default.x), rtol=0, atol=0
    )
