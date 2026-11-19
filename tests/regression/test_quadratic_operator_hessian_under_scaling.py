"""Regression: ``QuadraticProblem`` with an operator ``Q`` solves under scaling.

Through 0.13 it raised ``NotImplementedError: scaled operator Hessians are
not supported`` whenever gradient-based scaling activated (‖∇f(x0)‖∞ >
``ScalingOptions.max_gradient`` = 100, Wächter & Biegler 2006 §3.8), so the
same problem solved from one start and crashed from another. It now returns
``ScaledOperator(Q, sigma)``.
"""

from __future__ import annotations

import ipax
from ipax.backend.operators import as_operator
from tests._helpers import array, assert_allclose


def test_quadratic_problem_with_an_operator_hessian_under_scaling(namespace):
    xp = namespace
    Q = as_operator(array(xp, [[1e4, 0.0], [0.0, 2e4]]))
    c = array(xp, [1.0, -1.0])
    # ‖∇f(x0)‖∞ ≈ 2e4 > 100, so the Hessian is requested with sigma = s_f != 1.
    x0 = array(xp, [1.0, 1.0])
    result = ipax.solve(ipax.QuadraticProblem(Q, c), x0)
    assert result.status is ipax.Status.OPTIMAL
    assert result.routes.scaling == "gradient-based"
    assert_allclose(xp, result.x, array(xp, [-1e-4, 5e-5]), rtol=1e-6, atol=1e-10)


def test_quadratic_problem_scales_the_operator_hessian(namespace):
    xp = namespace
    Q = as_operator(array(xp, [[2.0, 1.0], [1.0, 3.0]]))
    problem = ipax.QuadraticProblem(Q, array(xp, [0.0, 0.0]))
    empty = array(xp, [])
    W = problem.lagrangian_hessian(array(xp, [0.0, 0.0]), empty, empty, 0.5)
    v = array(xp, [1.0, -2.0])
    assert_allclose(xp, W.matvec(v), 0.5 * Q.matvec(v))
    assert_allclose(xp, W.dense_matrix(), 0.5 * Q.dense_matrix())
