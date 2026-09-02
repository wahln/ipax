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

"""The L-BFGS apply guard must not scan the full result.

``_apply`` is the matvec — the hottest path in the solver, applied many times
per iteration by the condensed build and the Krylov solver. The singular-middle
guard added with the ``LINSPANH`` fix checked ``xp.all(xp.isfinite(result))``,
which allocates an ``n``-sized temporary **and forces a host synchronisation on
every application**. That is the same pattern that cost ~23x on GPU in
``barrier.py`` (see the device-efficiency work); it does not show up on a CPU
profile because the condensed Gram build dominates there, which is exactly why
it needs a test rather than a benchmark.

The failure mode being guarded is a backend whose ``solve`` returns inf/nan
instead of raising on a singular middle matrix. That garbage originates in the
solve output ``z``, which is ``2k``-sized (``k`` = L-BFGS memory, ≤ 20) rather
than ``n``-sized — so checking ``z`` catches the same condition at a cost that
does not grow with the problem.
"""

from __future__ import annotations

import numpy as np

from ipax.backend.namespace import array_namespace
from ipax.ipm.hessian import LBFGSOperator
from ipax.options import LBFGSOptions


def _healthy_operator(n: int, k: int) -> LBFGSOperator:
    """A well-conditioned compact operator with ``n`` variables and ``k`` pairs.

    The compact state is held as the S/Y blocks (``U = [xi*S  Y]`` with
    ``xi = 1``), matching the operator's internal representation.
    """
    assert k % 2 == 0
    op = LBFGSOperator(n, LBFGSOptions())
    op._xi = 1.0
    rng = np.random.default_rng(0)
    u = np.asarray(rng.standard_normal((n, k)), dtype=np.float64)
    op._s = u[:, : k // 2]
    op._y = u[:, k // 2 :]
    op._m = np.asarray(2.0 * np.eye(k), dtype=np.float64)
    return op


def test_the_apply_guard_cost_does_not_grow_with_the_problem(monkeypatch):
    n, k = 4000, 6
    op = _healthy_operator(n, k)
    v = np.asarray(np.ones(n), dtype=np.float64)
    xp = array_namespace(v)

    scanned: list[tuple[int, ...]] = []
    real_isfinite = xp.isfinite

    def spy(a):
        scanned.append(tuple(a.shape))
        return real_isfinite(a)

    monkeypatch.setattr(xp, "isfinite", spy)
    out = op.matvec(v)

    assert out.shape == (n,)
    # Whatever the guard inspects, it must be bounded by the L-BFGS memory, not
    # by the number of variables.
    assert scanned, "the guard must still verify the solve output"
    assert all(shape[0] <= 2 * k for shape in scanned), (
        f"guard scanned an n-sized array: shapes={scanned}, n={n}, k={k}"
    )


def test_a_non_finite_solve_output_still_falls_back_to_the_seed():
    # The condition the guard exists for: a singular middle matrix whose solve
    # returns inf/nan instead of raising. The device-side guard zeroes the
    # solve output, so the correction is exactly zero and the result is the
    # identity seed without a host synchronization.
    op = LBFGSOperator(2, LBFGSOptions())
    op._xi = 1.0
    # U = [S Y] = I with one pair: S the first column, Y the second.
    op._s = np.asarray([[1.0], [0.0]], dtype=np.float64)
    op._y = np.asarray([[0.0], [1.0]], dtype=np.float64)
    op._m = np.asarray([[1e-300, 0.0], [0.0, 1e-300]], dtype=np.float64)
    v = np.asarray([1e50, 1e50], dtype=np.float64)

    out = op.matvec(v)

    assert bool(np.all(np.isfinite(out)))
    np.testing.assert_allclose(out, v)  # the xi = 1 seed
