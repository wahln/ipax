"""Regression: ``Result.n_iter`` counts the iterations performed.

Through 0.13 it was ``len(history)``, which includes the start point as
iteration 0, so it was one too high — and 1 for a solve that stopped at the
start point. pyRadPlan compensated with ``n_iter - 1``; from 0.14 that
correction must go (CHANGELOG, *Changed*).
"""

from __future__ import annotations

import ipax
from ipax.problem.base import Problem
from tests._helpers import array


class _Shifted(Problem):
    """``sum((x - 1)^2)`` with optional bounds."""

    def __init__(self, xp, bounds=(None, None)):
        self._xp, self._bounds = xp, bounds

    @property
    def n_vars(self) -> int:
        return 2

    def bounds(self):
        return self._bounds

    def objective(self, x):
        return self._xp.sum((x - 1.0) * (x - 1.0))

    def gradient(self, x):
        return 2.0 * (x - 1.0)


def test_n_iter_counts_iterations_performed(namespace):
    result = ipax.solve(_Shifted(namespace), array(namespace, [0.0, 0.0]))
    assert result.history[0].iteration == 0  # the start point
    assert result.n_iter == result.history[-1].iteration
    assert result.n_iter == len(result.history) - 1


def test_n_iter_is_zero_when_the_solve_exits_at_the_start(namespace):
    bounds = (array(namespace, [1.0, 1.0]), array(namespace, [0.0, 0.0]))
    result = ipax.solve(
        _Shifted(namespace, bounds=bounds), array(namespace, [0.0, 0.0])
    )
    assert result.status is ipax.Status.INFEASIBLE
    assert result.n_iter == 0
