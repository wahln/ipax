"""Regression: the driver must not re-evaluate problem callbacks at a point it
already evaluated.

Before the point cache, every iteration evaluated ``f``/``c``/``g`` at the
current iterate at the loop top *and again* for ``phi0``/``theta0`` before the
line search, and the accepted trial point (already evaluated by the line search,
including its gradient on the L-BFGS route) was evaluated a third time at the
next loop top. Measured on 100-d Rosenbrock: 3.15 ``f`` and 2.0 ``∇f`` per
iteration against SciPy L-BFGS-B's 1.18/1.18 at the same iteration count; a
small constrained NLP showed 5.1 constraint evaluations per iteration.

The invariant pinned here: one gradient per iterate, and objective/constraint
evaluations bounded by the number of *distinct* points the globalization
visited (line-search trials), not by a multiple of the iteration count.
"""

from __future__ import annotations

from typing import Any

import pytest

from ipax import Options, Problem, Status, solve
from ipax.options import LineSearchOptions
from ipax.testing.problems import HS7, HS43, HS71
from tests._helpers import array


class _Counting(Problem):
    """Delegating wrapper that counts every callback invocation."""

    def __init__(self, inner: Problem) -> None:
        self._inner = inner
        self.calls: dict[str, int] = {
            "objective": 0,
            "gradient": 0,
            "eq_constraints": 0,
            "eq_jacobian": 0,
            "ineq_constraints": 0,
            "ineq_jacobian": 0,
        }

    @property
    def n_vars(self) -> int:
        return self._inner.n_vars

    def bounds(self) -> Any:
        return self._inner.bounds()

    def _count(self, name: str, x: Any) -> Any:
        self.calls[name] += 1
        return getattr(self._inner, name)(x)

    def objective(self, x: Any) -> Any:
        return self._count("objective", x)

    def gradient(self, x: Any) -> Any:
        return self._count("gradient", x)

    def eq_constraints(self, x: Any) -> Any:
        return self._count("eq_constraints", x)

    def eq_jacobian(self, x: Any) -> Any:
        return self._count("eq_jacobian", x)

    def ineq_constraints(self, x: Any) -> Any:
        return self._count("ineq_constraints", x)

    def ineq_jacobian(self, x: Any) -> Any:
        return self._count("ineq_jacobian", x)


class _Rosenbrock(Problem):
    """Extended Rosenbrock — an unconstrained problem needing many iterations."""

    def __init__(self, xp: Any, n: int) -> None:
        self.xp, self._n = xp, n

    @property
    def n_vars(self) -> int:
        return self._n

    def objective(self, x: Any) -> Any:
        xp = self.xp
        return xp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    def gradient(self, x: Any) -> Any:
        xp = self.xp
        xm, xp1 = x[:-1], x[1:]
        g = xp.zeros_like(x)
        inner = -400.0 * xm * (xp1 - xm**2) - 2.0 * (1.0 - xm)
        outer = 200.0 * (xp1 - xm**2)
        g[:-1] = inner
        g[1:] = g[1:] + outer
        return g


def _trials(result: Any) -> int:
    return sum(rec.line_search_iters for rec in result.history)


def test_unconstrained_lbfgs_evaluates_each_point_once(namespace):
    n = 20
    inner = _Rosenbrock(namespace, n)
    problem = _Counting(inner)
    x0 = array(namespace, [-1.2, 1.0] * (n // 2))
    result = solve(problem, x0, options=Options())
    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    assert result.n_iter > 20  # a real multi-iteration run
    calls = problem.calls
    # One gradient per iterate (x0 plus every accepted step); the L-BFGS
    # overshoot guard evaluates the trial gradient, which the next loop top
    # must reuse rather than recompute.
    assert calls["gradient"] <= result.n_iter + 3, calls
    # Every objective evaluation is at a distinct trial point (plus x0).
    assert calls["objective"] <= _trials(result) + 3, calls


_CASES = [
    (HS43, [0.0, 0.0, 0.0, 0.0]),
    (HS7, [2.0, 2.0]),
    (HS71, [1.0, 5.0, 5.0, 1.0]),
]
_IDS = ["HS43-ineq", "HS7-eq", "HS71-eq-ineq-bounds"]


@pytest.mark.parametrize(("factory", "x0"), _CASES, ids=_IDS)
def test_constrained_without_soc_evaluates_each_point_once(namespace, factory, x0):
    """With second-order corrections off, the only points visited are x0 and
    the line-search trials — so that is the evaluation budget."""
    problem = _Counting(factory(namespace))
    options = Options(line_search=LineSearchOptions(max_soc=0))
    result = solve(problem, array(namespace, x0), options=options)
    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    calls = problem.calls
    n_iter, trials = result.n_iter, _trials(result)
    restored = sum(1 for rec in result.history if rec.restored)
    slack = 3 + 4 * restored  # restoration visits a few points of its own
    assert calls["gradient"] <= n_iter + slack, calls
    assert calls["objective"] <= trials + slack, calls
    for name in ("eq_constraints", "ineq_constraints"):
        if calls[name]:
            assert calls[name] <= trials + slack, calls


@pytest.mark.parametrize(("factory", "x0"), _CASES, ids=_IDS)
def test_constrained_with_soc_bounded_by_visited_points(namespace, factory, x0):
    """Each second-order correction probes one new point (its residual forms
    the corrected RHS, W&B 2006 §2.4) and the accepted SOC point is one more;
    nothing beyond those and the trials may be evaluated."""
    problem = _Counting(factory(namespace))
    result = solve(problem, array(namespace, x0), options=Options())
    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    calls = problem.calls
    n_iter, trials = result.n_iter, _trials(result)
    max_soc = Options().line_search.max_soc
    restored = sum(1 for rec in result.history if rec.restored)
    slack = 3 + 4 * restored
    assert calls["gradient"] <= n_iter + slack, calls
    # f is evaluated at trials and accepted SOC points only (never at probes).
    assert calls["objective"] <= trials + n_iter + slack, calls
    for name in ("eq_constraints", "ineq_constraints"):
        if calls[name]:
            assert calls[name] <= trials + (max_soc + 1) * n_iter + slack, calls
