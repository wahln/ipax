"""Unit tests for two-sided linear-inequality lowering.

These check the algebra of :func:`ipax.problem.linear_ineq.lower_linear_inequalities`
directly (one-sided rows, signs, offsets, row selection) independent of a solve.
"""

from __future__ import annotations

import pytest

from ipax.problem.base import Problem
from ipax.problem.linear_ineq import lower_linear_inequalities
from tests._helpers import array, assert_allclose


class _LinearIneqOnly(Problem):
    """Minimal problem carrying only a two-sided ``linear_ineq`` block."""

    def __init__(self, xp, a, lower, upper) -> None:
        self.xp = xp
        self._a = a
        self._lower = lower
        self._upper = upper

    @property
    def n_vars(self) -> int:
        return int(self._a.shape[1])

    def objective(self, x):
        return self.xp.sum(x * x)

    def gradient(self, x):
        return 2.0 * x

    def linear_ineq(self):
        return self._a, self._lower, self._upper


def _lowered_g(namespace, a_rows, lower, upper, x_values):
    a = array(namespace, a_rows)
    problem = _LinearIneqOnly(
        namespace, a, array(namespace, lower), array(namespace, upper)
    )
    lowered = lower_linear_inequalities(
        problem, array(namespace, [0.0] * a.shape[1]), namespace
    )
    assert lowered.linear_ineq() is None  # lowered away
    x = array(namespace, x_values)
    return lowered.ineq_constraints(x)


def test_lower_only_row_is_negated(namespace):
    # 10 x1 - x2 ≥ 10  ⇒  g = 10 - 10 x1 + x2 ≤ 0.
    g = _lowered_g(namespace, [[10.0, -1.0]], [10.0], [float("inf")], [1.0, 2.0])
    assert int(g.shape[0]) == 1
    assert_allclose(namespace, g, array(namespace, [10.0 - 10.0 + 2.0]), atol=1e-9)


def test_upper_only_row_keeps_sign(namespace):
    # x1 ≤ 3  ⇒  g = x1 - 3 ≤ 0.
    g = _lowered_g(namespace, [[1.0, 0.0]], [float("-inf")], [3.0], [5.0, 0.0])
    assert int(g.shape[0]) == 1
    assert_allclose(namespace, g, array(namespace, [2.0]), atol=1e-9)


def test_two_sided_row_yields_lower_then_upper(namespace):
    # 1 ≤ x1 ≤ 4 at x1 = 2 ⇒ lower g = 1 - 2 = -1, upper g = 2 - 4 = -2.
    g = _lowered_g(namespace, [[1.0, 0.0]], [1.0], [4.0], [2.0, 0.0])
    assert int(g.shape[0]) == 2
    assert_allclose(namespace, g, array(namespace, [-1.0, -2.0]), atol=1e-9)


def test_free_row_is_dropped(namespace):
    # Row 0 has both bounds infinite ⇒ contributes nothing; row 1 (x2 ≤ 1) stays.
    g = _lowered_g(
        namespace,
        [[1.0, 0.0], [0.0, 1.0]],
        [float("-inf"), float("-inf")],
        [float("inf"), 1.0],
        [9.0, 3.0],
    )
    assert int(g.shape[0]) == 1
    assert_allclose(namespace, g, array(namespace, [2.0]), atol=1e-9)


def test_inverted_range_raises(namespace):
    problem = _LinearIneqOnly(
        namespace,
        array(namespace, [[1.0, 0.0]]),
        array(namespace, [5.0]),
        array(namespace, [1.0]),
    )
    with pytest.raises(ValueError, match="lower bound exceeds"):
        lower_linear_inequalities(problem, array(namespace, [0.0, 0.0]), namespace)


def test_bound_shape_mismatch_raises(namespace):
    problem = _LinearIneqOnly(
        namespace,
        array(namespace, [[1.0, 0.0], [0.0, 1.0]]),
        array(namespace, [0.0]),
        array(namespace, [1.0]),
    )
    with pytest.raises(ValueError, match="match the matrix row count"):
        lower_linear_inequalities(problem, array(namespace, [0.0, 0.0]), namespace)
