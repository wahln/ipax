"""Regression: a ``FunctionProblem`` subclass's overridden derivative is used.

Through 0.13 the derivative resolver trusted ``FunctionProblem``'s record of
the callables passed to its constructor, so a subclass overriding
``gradient`` (or a Jacobian or the Hessian) was silently ignored and finite
differences ran instead. From 0.14 ``FunctionProblem.provides()`` reports
the given callables plus the subclass's overrides.
"""

from __future__ import annotations

import warnings

import ipax
from ipax.options import Options
from tests._helpers import array


def test_function_problem_subclass_override_is_used(namespace):
    calls = []

    class Sub(ipax.FunctionProblem):
        def gradient(self, x):
            calls.append(1)
            return 2.0 * (x - 1.0)

    xp = namespace
    problem = Sub(2, lambda x: xp.sum((x - 1.0) * (x - 1.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ipax.FiniteDifferenceWarning)
        result = ipax.solve(
            problem, array(xp, [0.0, 0.0]), options=Options(enable_autodiff=False)
        )
    assert result.derivative_sources.gradient == "analytic"
    assert calls
