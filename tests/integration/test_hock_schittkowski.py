"""Integration tests for a small Hock-Schittkowski subset."""

from __future__ import annotations

import math

from ipax import Options, Status, solve
from ipax.problem.base import Problem
from ipax.testing.problems import HS6, HS7, HS8, HS35, HS43
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
