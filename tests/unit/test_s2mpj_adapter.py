"""Unit tests for the S2MPJ bridge adapter (no checkout required).

These drive ``_S2MPJProblem`` with a tiny fake CUTEst_problem so the bridge's
numerical-robustness behavior is testable without a real S2MPJ download.
"""

from __future__ import annotations

import numpy as np
import pytest

import ipax.testing.backends as backends
from benchmarks.corpus.s2mpj import _S2MPJExactProblem, _S2MPJProblem
from ipax.problem.base import Problem
from tests._helpers import array, assert_allclose


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


# -- exact Lagrangian Hessian bridge -----------------------------------------


class _QuadInstance:
    """Fake S2MPJ problem with a known Lagrangian Hessian.

    ``f = 0.5(a x0² + b x1²)`` (``∇²f = diag(a, b)``) with one quadratic
    constraint ``c0 = 0.5 x0²`` (``∇²c0 = diag(1, 0)``). ``LgHxy`` mirrors S2MPJ's
    ``evalLx``: ``∇²f + Σ Y_i ∇²c_i`` with no objective scaling — exactly what the
    adapter must compose ``(σ, y_eq, y_ineq)`` into.
    """

    n = 2
    m = 1
    a = 3.0
    b = 5.0
    objgrps = (0,)
    x0 = np.array([[1.0], [1.0]])
    xlower = np.array([[-np.inf], [-np.inf]])
    xupper = np.array([[np.inf], [np.inf]])

    def __init__(self, clower: float, cupper: float) -> None:
        self.clower = np.array([[clower]])
        self.cupper = np.array([[cupper]])

    def LgHxy(self, x, y):  # S2MPJ method name (CamelCase by convention)
        import scipy.sparse as sp

        hf = np.array([[self.a, 0.0], [0.0, self.b]])
        ch0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        yv = np.reshape(np.asarray(y, dtype=float), (-1,))
        coeff = float(yv[0]) if yv.shape[0] else 0.0
        return 0.0, np.zeros((2, 1)), sp.csr_matrix(hf + coeff * ch0)


def _dense_hessian(problem, xp, x, y_eq, y_ineq, sigma=1.0):
    op = problem.lagrangian_hessian(
        array(xp, x), array(xp, y_eq), array(xp, y_ineq), sigma=sigma
    )
    if hasattr(op, "matvec"):  # SparseOperator → materialize via matvec
        cols = [op.matvec(array(xp, e)) for e in ([1.0, 0.0], [0.0, 1.0])]
        return xp.stack(cols, axis=1)
    return op


def test_base_adapter_does_not_advertise_an_analytic_hessian():
    # ipax's ``_provides`` compares against ``Problem`` at the class level, so the
    # base class must inherit the (raising) base method to take the L-BFGS route,
    # while the exact subclass overrides it.
    assert _S2MPJProblem.lagrangian_hessian is Problem.lagrangian_hessian
    assert _S2MPJExactProblem.lagrangian_hessian is not Problem.lagrangian_hessian


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    ("clower", "cupper", "y_eq", "y_ineq", "extra_diag0"),
    [
        # equality (clower == cupper): curvature enters as +y_eq·∇²c
        (0.0, 0.0, [2.0], [], 2.0),
        # lower side clower ≤ c ⇒ g = clower − c, ∇²g = −∇²c ⇒ −y_ineq·∇²c
        (0.0, np.inf, [], [2.0], -2.0),
        # upper side c ≤ cupper ⇒ g = c − cupper, ∇²g = +∇²c ⇒ +y_ineq·∇²c
        (-np.inf, 0.0, [], [2.0], 2.0),
    ],
)
def test_exact_hessian_maps_multipliers_with_correct_signs(
    clower, cupper, y_eq, y_ineq, extra_diag0, sparse
):
    xp = backends.import_namespace("numpy")
    problem = _S2MPJExactProblem(_QuadInstance(clower, cupper), xp, sparse=sparse)

    hess = _dense_hessian(problem, xp, [1.0, 1.0], y_eq, y_ineq)

    expected = xp.asarray(
        [[_QuadInstance.a + extra_diag0, 0.0], [0.0, _QuadInstance.b]]
    )
    assert_allclose(xp, hess, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("sparse", [False, True])
def test_exact_hessian_scales_only_the_objective_term_by_sigma(sparse):
    # Under gradient-based scaling the driver passes σ = s_f ≠ 1; only ∇²f scales,
    # the constraint curvature keeps its (already-scaled) multiplier.
    xp = backends.import_namespace("numpy")
    problem = _S2MPJExactProblem(_QuadInstance(0.0, 0.0), xp, sparse=sparse)
    sigma = 0.25

    hess = _dense_hessian(problem, xp, [1.0, 1.0], [2.0], [], sigma=sigma)

    expected = xp.asarray(
        [[sigma * _QuadInstance.a + 2.0, 0.0], [0.0, sigma * _QuadInstance.b]]
    )
    assert_allclose(xp, hess, expected, rtol=1e-12, atol=1e-12)
