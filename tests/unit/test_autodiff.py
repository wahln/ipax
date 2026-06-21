"""Autodiff adapter + resolution tests (skipped on non-autodiff backends).

Covers the criterion that autodiff and analytic gradients agree to
finite-difference tolerance, and that the resolver routes to autodiff when a
problem omits its derivatives on a capable backend.
"""

from __future__ import annotations

import pytest

from ipax import FunctionProblem, Options, Status, solve
from ipax.problem.autodiff import get_autodiff_adapter
from ipax.problem.derivatives import resolve
from ipax.problem.finitediff import gradient_fd, jacobian_fd
from tests._helpers import array, assert_allclose


@pytest.fixture
def autodiff_namespace(namespace):
    if get_autodiff_adapter(namespace) is None:
        pytest.skip("backend has no autodiff adapter")
    return namespace


def _rosenbrock(xp):
    def objective(x):
        return 100.0 * (x[1] - x[0] * x[0]) ** 2 + (1.0 - x[0]) ** 2

    def gradient(x):
        return xp.stack(
            (
                -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
                200.0 * (x[1] - x[0] * x[0]),
            )
        )

    return objective, gradient


def test_autodiff_gradient_agrees_with_analytic_and_fd(autodiff_namespace):
    xp = autodiff_namespace
    objective, gradient = _rosenbrock(xp)
    resolved = resolve(FunctionProblem(2, objective), xp, Options())
    x = array(xp, [0.5, -0.3])

    assert resolved.sources.gradient == "autodiff"
    assert_allclose(xp, resolved.gradient(x), gradient(x), rtol=1e-10, atol=1e-10)
    assert_allclose(
        xp, resolved.gradient(x), gradient_fd(objective, x), rtol=1e-5, atol=1e-6
    )


def test_autodiff_jacobian_matches_finite_difference(autodiff_namespace):
    xp = autodiff_namespace

    def constraints(x):
        return xp.stack((x[0] * x[0] - x[1], x[0] + 2.0 * x[1]))

    problem = FunctionProblem(2, lambda x: xp.sum(x * x), ineq_constraints=constraints)
    resolved = resolve(problem, xp, Options())
    x = array(xp, [0.7, -1.1])

    assert resolved.sources.ineq_jacobian == "autodiff"
    assert_allclose(
        xp, resolved.ineq_jacobian(x), jacobian_fd(constraints, x), rtol=1e-5, atol=1e-6
    )


def test_lbfgs_solve_with_autodiff_gradient_reaches_optimum(autodiff_namespace):
    xp = autodiff_namespace
    objective, _ = _rosenbrock(xp)
    result = solve(
        FunctionProblem(2, objective),
        array(xp, [-1.2, 1.0]),
        options=Options(hessian="lbfgs", linsolve="dense"),
    )

    assert result.status is Status.OPTIMAL
    assert result.derivative_sources.gradient == "autodiff"
    assert result.derivative_sources.hessian == "lbfgs"
    assert_allclose(xp, result.x, array(xp, [1.0, 1.0]), rtol=1e-5, atol=1e-5)


def test_autodiff_hvp_solve_matches_exact_optimum(autodiff_namespace):
    xp = autodiff_namespace
    objective, _ = _rosenbrock(xp)
    result = solve(
        FunctionProblem(2, objective),
        array(xp, [-1.2, 1.0]),
        options=Options(hessian="autodiff-hvp", linsolve="dense"),
    )

    assert result.status is Status.OPTIMAL
    assert result.derivative_sources.hessian == "autodiff-hvp"
    assert_allclose(xp, result.x, array(xp, [1.0, 1.0]), rtol=1e-6, atol=1e-6)
