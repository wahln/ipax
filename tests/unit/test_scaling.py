"""Unit tests for gradient-based NLP auto-scaling."""

from __future__ import annotations

import math

import pytest

from ipax.backend.operators import MatrixFreeJacobian
from ipax.backend.sparse import get_sparse_adapter
from ipax.options import Options, ScalingOptions
from ipax.problem.function import FunctionProblem
from ipax.problem.scaling import ScaledProblem, compute_scaling
from tests._helpers import array, assert_allclose, assert_scalar_close


def _problem(xp, **kwargs):
    return FunctionProblem(2, lambda x: xp.sum(x * x), **kwargs)


def test_objective_factor_caps_gradient_infnorm(namespace):
    xp = namespace
    grad = array(xp, [60.0, -80.0])  # ∞-norm 80
    prob = _problem(xp, gradient=lambda x: grad)
    x0 = array(xp, [0.0, 0.0])
    scaling = compute_scaling(
        prob, x0, xp, has_eq=False, has_ineq=False, max_gradient=40.0
    )
    assert_scalar_close(scaling.obj, 0.5)  # 40 / 80


def test_objective_factor_is_one_when_already_small(namespace):
    xp = namespace
    prob = _problem(xp, gradient=lambda x: array(xp, [1.0, 2.0]))
    x0 = array(xp, [0.0, 0.0])
    scaling = compute_scaling(
        prob, x0, xp, has_eq=False, has_ineq=False, max_gradient=100.0
    )
    assert_scalar_close(scaling.obj, 1.0)


def test_constraint_row_factors(namespace):
    xp = namespace
    jac = array(xp, [[3.0, 4.0], [0.0, 0.0]])  # row ∞-norms 4, 0
    prob = _problem(
        xp,
        gradient=lambda x: array(xp, [1.0, 1.0]),
        ineq_constraints=lambda x: xp.matmul(jac, x),
        ineq_jacobian=lambda x: jac,
    )
    x0 = array(xp, [0.0, 0.0])
    scaling = compute_scaling(
        prob, x0, xp, has_eq=False, has_ineq=True, max_gradient=2.0
    )
    # row 0: min(1, 2/4) = 0.5; row 1 (zero row) is left unscaled at 1.
    assert_allclose(xp, scaling.ineq, array(xp, [0.5, 1.0]), rtol=1e-12, atol=1e-12)


def test_matrix_free_scaling_does_not_probe_every_adjoint_row(namespace):
    xp = namespace
    jac = array(xp, [[3.0, 4.0], [5.0, 6.0]])
    adjoint_calls = 0

    def rmatvec(v):
        nonlocal adjoint_calls
        adjoint_calls += 1
        return xp.matmul(xp.permute_dims(jac, (1, 0)), v)

    op = MatrixFreeJacobian(
        jac.shape,
        matvec=lambda v: xp.matmul(jac, v),
        rmatvec=rmatvec,
    )
    prob = _problem(
        xp,
        gradient=lambda x: array(xp, [1.0, 1.0]),
        ineq_constraints=lambda x: xp.matmul(jac, x),
        ineq_jacobian=lambda x: op,
    )

    scaling = compute_scaling(
        prob,
        array(xp, [0.0, 0.0]),
        xp,
        has_eq=False,
        has_ineq=True,
        max_gradient=2.0,
    )

    assert adjoint_calls == 0
    assert_allclose(xp, scaling.ineq, array(xp, [1.0, 1.0]))


def test_matrix_free_scaling_accepts_bulk_row_norm_callback(namespace):
    xp = namespace
    jac = array(xp, [[3.0, 4.0], [5.0, 6.0]])
    op = MatrixFreeJacobian(
        jac.shape,
        matvec=lambda v: xp.matmul(jac, v),
        rmatvec=lambda v: xp.matmul(xp.permute_dims(jac, (1, 0)), v),
        row_inf_norms=lambda: array(xp, [4.0, 6.0]),
    )
    prob = _problem(
        xp,
        gradient=lambda x: array(xp, [1.0, 1.0]),
        ineq_constraints=lambda x: xp.matmul(jac, x),
        ineq_jacobian=lambda x: op,
    )

    scaling = compute_scaling(
        prob,
        array(xp, [0.0, 0.0]),
        xp,
        has_eq=False,
        has_ineq=True,
        max_gradient=2.0,
    )

    assert_allclose(xp, scaling.ineq, array(xp, [0.5, 1.0 / 3.0]))


@pytest.mark.sparse
def test_sparse_constraint_row_factors_use_sparse_fast_path(namespace):
    xp = namespace
    adapter = get_sparse_adapter(xp)
    if adapter is None:
        pytest.skip(f"no sparse adapter for backend {xp.__name__!r}")
    jac = adapter.from_coo(
        xp.asarray((0, 0, 1)),
        xp.asarray((0, 1, 1)),
        array(xp, [3.0, -4.0, 6.0]),
        shape=(2, 2),
    )
    prob = _problem(
        xp,
        gradient=lambda x: array(xp, [1.0, 1.0]),
        ineq_constraints=lambda x: jac.matvec(x),
        ineq_jacobian=lambda x: jac,
    )

    scaling = compute_scaling(
        prob,
        array(xp, [0.0, 0.0]),
        xp,
        has_eq=False,
        has_ineq=True,
        max_gradient=2.0,
    )

    assert_allclose(xp, scaling.ineq, array(xp, [0.5, 1.0 / 3.0]))


def test_linear_eq_combined_factor_orders_nonlinear_then_linear(namespace):
    xp = namespace
    a = array(xp, [[0.0, 50.0]])  # row ∞-norm 50
    prob = _problem(
        xp,
        gradient=lambda x: array(xp, [1.0, 1.0]),
        linear_eq=(a, array(xp, [1.0])),
    )
    x0 = array(xp, [0.0, 0.0])
    scaling = compute_scaling(
        prob, x0, xp, has_eq=False, has_ineq=False, max_gradient=10.0
    )
    assert scaling.eq is None
    assert_allclose(xp, scaling.linear_eq, array(xp, [0.2]), rtol=1e-12, atol=1e-12)
    # no nonlinear equalities → combined equals the linear block
    assert_allclose(xp, scaling.combined_eq, array(xp, [0.2]), rtol=1e-12, atol=1e-12)


def test_scaled_problem_applies_factors(namespace):
    xp = namespace
    jac = array(xp, [[3.0, 4.0]])
    prob = _problem(
        xp,
        gradient=lambda x: 200.0 * x,
        ineq_constraints=lambda x: xp.matmul(jac, x),
        ineq_jacobian=lambda x: jac,
        lagrangian_hessian=lambda x, ye, yi, sigma: sigma * 200.0 * xp.eye(2),
    )
    x0 = array(xp, [1.0, 1.0])  # gradient ∞-norm 200
    scaling = compute_scaling(
        prob, x0, xp, has_eq=False, has_ineq=True, max_gradient=50.0
    )
    scaled = ScaledProblem(prob, scaling)

    x = array(xp, [2.0, -1.0])
    s_f = scaling.obj
    assert_scalar_close(s_f, 0.25)  # 50 / 200

    assert_scalar_close(scaled.objective(x), s_f * float(prob.objective(x)))
    assert_allclose(
        xp, scaled.gradient(x), s_f * prob.gradient(x), rtol=1e-12, atol=1e-12
    )
    assert_allclose(
        xp,
        scaled.ineq_constraints(x),
        scaling.ineq * prob.ineq_constraints(x),
        rtol=1e-12,
        atol=1e-12,
    )
    # row-scaled Jacobian: matvec rows carry the constraint factors.
    v = array(xp, [1.0, 1.0])
    assert_allclose(
        xp,
        scaled.ineq_jacobian(x).matvec(v),
        scaling.ineq * prob.ineq_jacobian(x) @ v,
        rtol=1e-12,
        atol=1e-12,
    )


def test_scaling_options_validation():
    with pytest.raises(ValueError, match="scaling method"):
        ScalingOptions(method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_gradient"):
        ScalingOptions(method="gradient-based", max_gradient=0.0)
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            ScalingOptions(method="gradient-based", max_gradient=value)
    with pytest.raises(ValueError, match="scaling method"):
        Options(scaling="bogus")  # type: ignore[arg-type]
