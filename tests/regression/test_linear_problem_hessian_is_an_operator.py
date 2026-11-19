"""Regression: ``LinearProblem`` no longer builds a dense n×n zero Hessian.

Through 0.13 its ``lagrangian_hessian`` returned ``xp.zeros((n, n))`` every
iteration — 80 GB per iteration at n = 1e5, against the "never form a dense
Hessian at scale" rule. It now returns a zero ``Diagonal`` operator.
"""

from __future__ import annotations

import ipax
from ipax.backend.operators import LinearOperator
from tests._helpers import array, to_float


def test_linear_problem_hessian_is_a_zero_operator_not_a_dense_matrix(namespace):
    xp = namespace
    n = 5
    problem = ipax.LinearProblem(
        array(xp, [1.0] * n), bounds=(array(xp, [0.0] * n), None)
    )
    empty = array(xp, [])
    W = problem.lagrangian_hessian(array(xp, [0.5] * n), empty, empty, 1.0)
    assert isinstance(W, LinearOperator)
    assert W.shape == (n, n)
    assert to_float(xp.sum(xp.abs(W.matvec(array(xp, [1.0] * n))))) == 0.0
