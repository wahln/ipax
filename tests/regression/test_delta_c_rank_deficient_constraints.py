"""Regression: rank-deficient equality Jacobians need δ_c escalation.

S2MPJ sweep — the lbfgs/krylov ``numerical_error`` cluster (ACOPP*/ACOPR*, AC
optimal-power-flow) died at iteration 1: their equality Jacobian is rank-deficient
(the reference-bus degeneracy), which leaves the bordered saddle singular in the
(2,2) **dual** block. The regularization loop only escalated ``δ_w`` (the (1,1)
primal block), which cannot repair a singular dual block — so it escalated to
``1e27`` uselessly (and into numerical singularity) and reported
``numerical_error``.

The fix escalates ``δ_c`` alongside ``δ_w`` on a failed KKT solve (Wächter &
Biegler 2006, §3.1). This reproduces the failure with a deliberately redundant
equality constraint (declared twice → rank-1 Jacobian) and checks the solve now
reaches the true optimum instead of failing.
"""

from __future__ import annotations

from ipax import Options, Status, solve
from ipax.problem.base import Problem
from tests._helpers import array, assert_allclose, assert_scalar_close


class _RedundantEquality(Problem):
    """``min ½(x1² + x2²)`` s.t. ``x1 + x2 = 1`` declared **twice**.

    The duplicated constraint makes the equality Jacobian ``[[1,1],[1,1]]`` rank
    1, so the saddle is singular in the dual block unless ``δ_c`` regularizes it.
    Optimum ``(0.5, 0.5)``, ``f* = 0.25``.
    """

    def __init__(self, xp) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        return 0.5 * (x[0] * x[0] + x[1] * x[1])

    def gradient(self, x):
        return x

    def eq_constraints(self, x):
        residual = x[0] + x[1] - 1.0
        return self.xp.stack((residual, residual))

    def eq_jacobian(self, x):
        one = self.xp.ones_like(x[0])
        row = self.xp.stack((one, one))
        return self.xp.stack((row, row))

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del y_eq, y_ineq
        return sigma * self.xp.eye(2, dtype=x.dtype)


def _assert_solved(result, namespace) -> None:
    assert result.status is Status.OPTIMAL, f"unexpected status {result.status}"
    assert_allclose(
        namespace, result.x, array(namespace, [0.5, 0.5]), rtol=1e-5, atol=1e-5
    )
    assert_scalar_close(result.objective, 0.25, atol=1e-5)


def test_dense_solves_rank_deficient_equality(namespace):
    result = solve(
        _RedundantEquality(namespace),
        array(namespace, [2.0, -1.0]),
        options=Options(hessian="exact", linsolve="dense"),
    )
    _assert_solved(result, namespace)


def test_lbfgs_solves_rank_deficient_equality(namespace):
    # The Hessian-less L-BFGS route on the same degenerate saddle.
    result = solve(
        _RedundantEquality(namespace),
        array(namespace, [2.0, -1.0]),
        options=Options(hessian="lbfgs", linsolve="dense"),
    )
    _assert_solved(result, namespace)
