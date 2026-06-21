"""Integration tests for a small Hock-Schittkowski subset."""

from __future__ import annotations

import math

from ipax import Options, Status, solve
from ipax.problem.base import Problem
from ipax.testing.problems import HS6, HS7, HS8, HS9, HS21, HS28, HS35, HS43, HS71
from tests._helpers import array, assert_allclose, assert_scalar_close, implemented

_EXACT_DENSE = {"hessian": "exact", "linsolve": "dense"}


class HS1Rosenbrock(Problem):
    """HS1: unconstrained Rosenbrock, optimum f(1, 1) = 0."""

    def __init__(self, xp) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        return 100.0 * (x[1] - x[0] * x[0]) ** 2 + (1.0 - x[0]) ** 2

    def gradient(self, x):
        return self.xp.stack(
            (
                -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
                200.0 * (x[1] - x[0] * x[0]),
            )
        )


def test_hs1_rosenbrock_reaches_published_optimum(namespace):
    with implemented("HS subset"):
        result = solve(
            HS1Rosenbrock(namespace),
            array(namespace, [-1.2, 1.0]),
            options=Options(hessian="lbfgs", linsolve="dense"),
        )

    assert result.status is Status.OPTIMAL
    assert_allclose(
        namespace, result.x, array(namespace, [1.0, 1.0]), rtol=1e-6, atol=1e-6
    )
    assert_scalar_close(result.objective, 0.0, atol=1e-8)


def _assert_solved(result, namespace, known_x, known_f) -> None:
    assert result.status is Status.OPTIMAL
    assert result.kkt_error <= 1e-6
    assert result.constraint_violation <= 1e-6
    assert_allclose(namespace, result.x, known_x, rtol=1e-6, atol=1e-6)
    assert_scalar_close(result.objective, known_f, atol=1e-6)


def test_hs6_nonconvex_equality(namespace):
    problem = HS6(namespace)
    result = solve(
        problem, array(namespace, [-1.2, 1.0]), options=Options(**_EXACT_DENSE)
    )
    _assert_solved(result, namespace, problem.known_solution(), 0.0)


def test_hs7_log_objective_equality(namespace):
    problem = HS7(namespace)
    result = solve(
        problem, array(namespace, [2.0, 2.0]), options=Options(**_EXACT_DENSE)
    )
    _assert_solved(result, namespace, problem.known_solution(), -math.sqrt(3.0))


def test_hs8_two_quadratic_equalities(namespace):
    problem = HS8(namespace)
    result = solve(
        problem, array(namespace, [2.0, 1.0]), options=Options(**_EXACT_DENSE)
    )
    _assert_solved(result, namespace, problem.known_solution(), -1.0)


def test_hs35_inequality_qp_with_bounds(namespace):
    problem = HS35(namespace)
    result = solve(
        problem, array(namespace, [0.5, 0.5, 0.5]), options=Options(**_EXACT_DENSE)
    )
    _assert_solved(result, namespace, problem.known_solution(), 1.0 / 9.0)


def test_hs43_rosen_suzuki_inequalities(namespace):
    problem = HS43(namespace)
    result = solve(
        problem, array(namespace, [0.0, 0.0, 0.0, 0.0]), options=Options(**_EXACT_DENSE)
    )
    _assert_solved(result, namespace, problem.known_solution(), -44.0)


def test_hs21_active_bound_with_linear_inequality(namespace):
    problem = HS21(namespace)
    result = solve(
        problem, array(namespace, [3.0, 1.0]), options=Options(**_EXACT_DENSE)
    )
    _assert_solved(result, namespace, problem.known_solution(), -99.96)
    z_lower, z_upper = problem.known_bound_multipliers()
    assert_allclose(namespace, result.z_lower, z_lower, rtol=1e-6, atol=1e-6)
    assert_allclose(namespace, result.z_upper, z_upper, rtol=1e-6, atol=1e-6)


def test_hs28_equality_qp_degenerate_multiplier(namespace):
    problem = HS28(namespace)
    result = solve(
        problem, array(namespace, [-1.0, 0.5, 0.5]), options=Options(**_EXACT_DENSE)
    )
    _assert_solved(result, namespace, problem.known_solution(), 0.0)
    assert_allclose(
        namespace, result.y_eq, problem.known_multiplier(), rtol=1e-6, atol=1e-6
    )


def test_hs9_nonunique_periodic_optimum(namespace):
    # The optimum is non-unique (periodic); assert the optimal value and KKT
    # satisfaction rather than a specific x*.
    problem = HS9(namespace)
    result = solve(
        problem, array(namespace, [0.0, 0.0]), options=Options(**_EXACT_DENSE)
    )

    assert result.status is Status.OPTIMAL
    assert result.kkt_error <= 1e-6
    assert result.constraint_violation <= 1e-6
    assert_scalar_close(result.objective, -0.5, atol=1e-6)


def test_hs71_full_constraint_mix(namespace):
    # Published optimum carries ~9 significant figures; relax the primal/objective
    # tolerance accordingly while still gating KKT/feasibility tightly.
    problem = HS71(namespace)
    result = solve(
        problem, array(namespace, [1.0, 5.0, 5.0, 1.0]), options=Options(**_EXACT_DENSE)
    )

    assert result.status is Status.OPTIMAL
    assert result.kkt_error <= 1e-6
    assert result.constraint_violation <= 1e-6
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-5, atol=1e-5)
    assert_scalar_close(result.objective, 17.0140173, atol=1e-4)
