"""Bounds, scaled affine inequalities, and L-BFGS across the three solve routes."""

from __future__ import annotations

import pytest

from ipax import FunctionProblem, Options, solve
from ipax.backend.operators import COOOperator
from ipax.backend.sparse import get_sparse_adapter
from tests._helpers import array, assert_allclose


@pytest.mark.parametrize("route", ["dense", "krylov", "sparse"])
@pytest.mark.parametrize("representation", ["linear_ineq", "coo_callback"])
def test_affine_lbfgs_active_inequality_and_bound(namespace, route, representation):
    xp = namespace
    if (route == "sparse" or representation == "coo_callback") and get_sparse_adapter(
        xp
    ) is None:
        pytest.skip("backend has no sparse adapter")
    A = array(xp, [[200.0, 200.0, 0.0]])
    b = array(xp, [200.0])
    target = array(xp, [2.0, 1.5, -1.0])
    if representation == "linear_ineq":
        constraints = {"linear_ineq": (A, array(xp, [-float("inf")]), b)}
    else:
        op = COOOperator(
            xp.asarray([0, 0]), xp.asarray([0, 1]), array(xp, [200.0, 200.0]), (1, 3)
        )
        constraints = {
            "ineq_constraints": lambda x: op.matvec(x) - b,
            "ineq_jacobian": lambda x: op,
        }
    problem = FunctionProblem(
        3,
        lambda x: 0.5 * xp.sum((x - target) ** 2),
        gradient=lambda x: x - target,
        bounds=(xp.zeros((3,), dtype=A.dtype), xp.full((3,), 2.0, dtype=A.dtype)),
        **constraints,
    )
    result = solve(
        problem,
        array(xp, [0.2, 0.2, 0.2]),
        options=Options(
            linsolve=route, hessian="lbfgs", scaling="gradient-based", max_iter=100
        ),
    )
    assert result.success, result.message
    assert result.kkt_error <= 1e-8
    assert result.derivative_sources.hessian == "lbfgs"
    assert_allclose(xp, result.x, array(xp, [0.75, 0.25, 0.0]), atol=1e-6, rtol=1e-6)
    assert (
        result.y_ineq is not None
        and result.z_lower is not None
        and result.z_upper is not None
    )
    stationarity = (
        result.x
        - target
        + xp.matmul(xp.permute_dims(A, (1, 0)), result.y_ineq)
        - result.z_lower
        + result.z_upper
    )
    assert float(xp.max(xp.abs(stationarity))) < 1e-7
    assert float(xp.max(xp.abs(result.y_ineq * (xp.matmul(A, result.x) - b)))) < 1e-7
    assert float(xp.max(xp.abs(result.z_lower * result.x))) < 1e-7
