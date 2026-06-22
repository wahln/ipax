"""Unit tests for the S2MPJ bridge adapter (no checkout required).

These drive ``_S2MPJProblem`` with a tiny fake CUTEst_problem so the bridge's
numerical-robustness behavior is testable without a real S2MPJ download.
"""

from __future__ import annotations

import numpy as np

import ipax.testing.backends as backends
from benchmarks.corpus.s2mpj import _S2MPJProblem
from tests._helpers import array


class _OverflowingInstance:
    """Minimal S2MPJ-like object whose evaluations overflow (Python ``float**``)."""

    n = 2
    m = 0
    objgrps = (0,)  # advertises an objective group
    x0 = np.array([[1.0], [1.0]])
    xlower = np.array([[-np.inf], [-np.inf]])
    xupper = np.array([[np.inf], [np.inf]])

    def fx(self, x):
        raise OverflowError(34, "Result too large")

    def fgx(self, x):
        raise OverflowError(34, "Result too large")


def test_adapter_objective_overflow_returns_inf():
    # Regression (LUKVLE4C): S2MPJ's generated ``float**`` can raise OverflowError
    # at a line-search trial point; the adapter must return +inf so the solver
    # rejects the trial instead of crashing.
    xp = backends.import_namespace("numpy")
    problem = _S2MPJProblem(_OverflowingInstance(), xp)
    x = array(xp, [1.0, 1.0])

    obj = problem.objective(x)
    assert not bool(xp.isfinite(obj))
    assert float(obj) > 0.0  # +inf, so φ rejects the trial

    grad = problem.gradient(x)
    assert grad.shape == (2,)
    assert not bool(xp.any(xp.isfinite(grad)))
