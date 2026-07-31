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

"""A singular L-BFGS middle matrix must report "no diagonal", not crash.

``LBFGSOperator.diagonal`` forms ``diag(ξI − U M⁻¹ Uᵀ)`` by solving with the
compact middle matrix ``M``. When ``M`` is singular that solve raises a
*backend* exception — ``numpy.linalg.LinAlgError``, and a different type on
Torch — which escaped all the way out of ``ipax.solve``.

The contract around it already exists: ``diagonal`` raises
``NotImplementedError`` before the first curvature pair, the condensed KKT
operator's ``spd_preconditioner_diagonal`` propagates that, and the Krylov
solver treats it as "no preconditioner diagonal available" and carries on. A
singular ``M`` is the same situation — the diagonal cannot be formed — so it
must be reported the same way rather than as a hard failure.

Found on S2MPJ ``LINSPANH`` (``lbfgs/krylov``), where a change of trajectory
elsewhere in the driver was enough to reach a singular ``M`` mid-solve and turn
a converging run into an uncaught ``LinAlgError``.
"""

from __future__ import annotations

import pytest

from ipax.ipm.hessian import LBFGSOperator
from ipax.options import LBFGSOptions
from tests._helpers import array, assert_allclose


def _operator_with_middle(namespace, middle, u):
    op = LBFGSOperator(2, LBFGSOptions())
    op._xi = 1.0
    op._u = array(namespace, u)
    op._m = array(namespace, middle)
    return op


def test_singular_middle_matrix_reports_no_diagonal(namespace):
    # Two identical rows: M is exactly singular.
    op = _operator_with_middle(
        namespace, [[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]
    )

    with pytest.raises(NotImplementedError):
        op.diagonal()


def test_a_nonsingular_middle_matrix_still_returns_the_diagonal(namespace):
    # The guard must not swallow the working case: M = I, U = I gives
    # diag(B) = ξ − diag(U M⁻¹ Uᵀ) = 1 − 1 = 0.
    op = _operator_with_middle(
        namespace, [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]
    )

    diagonal = op.diagonal()

    assert diagonal.shape == (2,)
    assert bool(namespace.all(namespace.isfinite(diagonal)))


def test_a_near_singular_middle_matrix_producing_non_finites_reports_no_diagonal(
    namespace,
):
    # Some backends return inf/nan from a singular solve instead of raising;
    # an unusable diagonal must be reported either way.
    op = _operator_with_middle(
        namespace, [[1e-300, 0.0], [0.0, 1e-300]], [[1e200, 0.0], [0.0, 1e200]]
    )

    try:
        diagonal = op.diagonal()
    except NotImplementedError:
        return
    assert bool(namespace.all(namespace.isfinite(diagonal)))


def test_applying_the_operator_with_a_singular_middle_falls_back_to_the_seed(
    namespace,
):
    # matvec has no "unavailable" channel — callers need a vector back — so a
    # singular middle matrix drops the low-rank correction and applies the
    # identity seed. That keeps B positive definite, which is what the condensed
    # route requires, instead of aborting the solve.
    op = _operator_with_middle(
        namespace, [[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]
    )
    v = array(namespace, [2.0, -3.0])

    out = op.matvec(v)

    assert bool(namespace.all(namespace.isfinite(out)))
    assert_allclose(namespace, out, v)  # xi = 1 seed


def test_a_healthy_operator_still_applies_its_correction(namespace):
    # The guard must not swallow the working case.
    op = _operator_with_middle(
        namespace, [[2.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]]
    )
    v = array(namespace, [2.0, -3.0])

    out = op.matvec(v)

    # B = I - U (2I)^-1 U^T = I - 0.5 I = 0.5 I
    assert_allclose(namespace, out, 0.5 * v, rtol=1e-10)
