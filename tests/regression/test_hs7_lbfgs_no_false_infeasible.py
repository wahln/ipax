"""Regression: the L-BFGS route must not blow up / falsely report infeasible on HS7.

S2MPJ sweep (2026-06-24) Task 1 — false infeasibility. On the S2MPJ formulation of
HS7 (``min ln(1+x1²) - x2`` s.t. ``(1+x1²)² + x2² = 4``) the L-BFGS routes diverged:
the filter line search accepted a catastrophic full step whose barrier objective
``φ ≈ -x2`` collapsed toward ``-∞`` while the constraint violation ``θ`` exploded to
``~1e134``, so the f-type (switching + Armijo) branch let it through. The solver then
reported ``INFEASIBLE``/``NUMERICAL_ERROR`` even though a feasible optimum exists.

Root cause: the filter was initialized empty, missing the Wächter & Biegler 2006
eq. (18) guard region ``{θ ≥ θ_max}`` that rejects wildly infeasible trial points
outright. The analytic :class:`~ipax.testing.problems.HS7` masked the bug because a
problem that supplies ``lagrangian_hessian`` silently runs the *exact* route even when
``hessian="lbfgs"`` is requested — so the L-BFGS path is exercised here by hiding the
Hessian.
"""

from __future__ import annotations

import math

from ipax import Options, Status, solve
from ipax.problem.base import Problem
from ipax.testing.problems import HS7
from tests._helpers import array, assert_allclose, assert_scalar_close


class _HS7NoHessian(Problem):
    """HS7 with the Lagrangian Hessian withheld, forcing the genuine L-BFGS route."""

    def __init__(self, xp) -> None:
        self._p = HS7(xp)

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        return self._p.objective(x)

    def gradient(self, x):
        return self._p.gradient(x)

    def eq_constraints(self, x):
        return self._p.eq_constraints(x)

    def eq_jacobian(self, x):
        return self._p.eq_jacobian(x)


def _assert_reached_optimum(result, namespace) -> None:
    assert result.status is Status.OPTIMAL, f"unexpected status {result.status}"
    assert result.constraint_violation <= 1e-6
    assert result.kkt_error <= 1e-6
    assert_allclose(
        namespace,
        result.x,
        array(namespace, [0.0, math.sqrt(3.0)]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert_scalar_close(result.objective, -math.sqrt(3.0), atol=1e-5)


def test_hs7_lbfgs_dense_does_not_diverge(namespace):
    result = solve(
        _HS7NoHessian(namespace),
        array(namespace, [2.0, 2.0]),
        options=Options(hessian="lbfgs", linsolve="dense"),
    )
    _assert_reached_optimum(result, namespace)


def test_hs7_lbfgs_krylov_does_not_diverge(namespace):
    result = solve(
        _HS7NoHessian(namespace),
        array(namespace, [2.0, 2.0]),
        options=Options(hessian="lbfgs", linsolve="krylov"),
    )
    _assert_reached_optimum(result, namespace)
