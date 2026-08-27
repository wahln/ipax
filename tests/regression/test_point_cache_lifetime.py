"""Regression: the driver's point cache must not keep stale derivatives alive.

Codex review of the 0.10.1 release PR (#12): with a four-entry LRU every
loop-top point cached its Jacobians, so up to four full explicit Jacobians (or
matrix-free operators retaining device graphs) stayed strongly referenced at
once — a 4× peak-memory multiplier at the 1e5-variable target. The cache now
retains only the current iterate when the loop advances; trial points are
evaluated afresh each iteration and evicted with it.

Also covers the descent-enforcement retry diagnostic: the probe's directional
derivative was cleared *before* the debug log formatted it (``%.3e`` of
``None`` → a logging error on every regularization retry).
"""

from __future__ import annotations

import gc
import logging
import weakref
from typing import Any

import pytest

from ipax import Options, Problem, Status, solve
from ipax.ipm.driver import _MISS, IPMDriver, _PointCache
from ipax.testing.problems import HS71
from tests._helpers import array


def test_point_cache_retain_keeps_only_the_named_point(namespace):
    cache = _PointCache(capacity=4)
    a = array(namespace, [1.0])
    b = array(namespace, [2.0])
    c = array(namespace, [3.0])
    for point, value in ((a, "fa"), (b, "fb"), (c, "fc")):
        cache.put(point, "f", value)
    cache.put(b, "eq_jac", "Jb")
    cache.retain(b)
    assert cache.get(b, "f") == "fb"
    assert cache.get(b, "eq_jac") == "Jb"
    assert cache.get(a, "f") is _MISS
    assert cache.get(c, "f") is _MISS
    # Retaining a point the cache never saw simply empties it.
    cache.retain(a)
    assert cache.get(b, "f") is _MISS


class _TrackedJacobians(Problem):
    """HS71 whose Jacobians are fresh arrays tracked by weak reference."""

    def __init__(self, xp: Any) -> None:
        self._inner = HS71(xp)
        self.eq_refs: list[weakref.ref[Any]] = []
        self.ineq_refs: list[weakref.ref[Any]] = []

    @property
    def n_vars(self) -> int:
        return self._inner.n_vars

    def bounds(self) -> Any:
        return self._inner.bounds()

    def objective(self, x: Any) -> Any:
        return self._inner.objective(x)

    def gradient(self, x: Any) -> Any:
        return self._inner.gradient(x)

    def eq_constraints(self, x: Any) -> Any:
        return self._inner.eq_constraints(x)

    def eq_jacobian(self, x: Any) -> Any:
        jac = self._inner.eq_jacobian(x)
        self.eq_refs.append(weakref.ref(jac))
        return jac

    def ineq_constraints(self, x: Any) -> Any:
        return self._inner.ineq_constraints(x)

    def ineq_jacobian(self, x: Any) -> Any:
        jac = self._inner.ineq_jacobian(x)
        self.ineq_refs.append(weakref.ref(jac))
        return jac


def test_at_most_two_jacobians_alive_during_a_solve(namespace):
    problem = _TrackedJacobians(namespace)
    x0 = array(namespace, [1.0, 5.0, 5.0, 1.0])
    try:
        weakref.ref(problem.eq_jacobian(x0))
    except TypeError:
        pytest.skip("backend arrays do not support weak references")
    problem.eq_refs.clear()
    peak = {"eq": 0, "ineq": 0}

    def callback(info: Any) -> bool:
        gc.collect()
        peak["eq"] = max(peak["eq"], sum(1 for r in problem.eq_refs if r() is not None))
        peak["ineq"] = max(
            peak["ineq"], sum(1 for r in problem.ineq_refs if r() is not None)
        )
        return False

    result = solve(problem, x0, options=Options(), callback=callback)
    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    assert result.n_iter >= 5
    # The current iterate's Jacobian (held by the cache and the loop locals)
    # plus at most the previous one still referenced by the L-BFGS curvature
    # pair bookkeeping — never a backlog of cached points.
    assert peak["eq"] <= 2, peak
    assert peak["ineq"] <= 2, peak


def test_descent_retry_diagnostic_formats(namespace, monkeypatch, caplog):
    """Force the non-descent retry at a feasible iterate and check the debug
    record still formats (the probe value must survive until it is logged)."""
    from ipax.testing.problems import UnconstrainedQuadratic

    xp = namespace
    problem = UnconstrainedQuadratic(
        array(xp, [[2.0, 0.0], [0.0, 4.0]]), array(xp, [-1.0, 1.0]), xp
    )
    original = IPMDriver._dphi
    state = {"calls": 0}

    def ascent_once(self: Any, *args: Any, **kwargs: Any) -> float:
        state["calls"] += 1
        value = original(self, *args, **kwargs)
        return abs(value) + 1.0 if state["calls"] == 1 else value

    monkeypatch.setattr(IPMDriver, "_dphi", ascent_once)
    caplog.set_level(logging.DEBUG, logger="ipax")
    result = solve(problem, array(xp, [3.0, -3.0]), options=Options())
    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    messages = [rec.getMessage() for rec in caplog.records]  # raises on bad args
    assert any("non-descent direction" in msg for msg in messages), messages
