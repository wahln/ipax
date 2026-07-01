"""Regression: a non-finite sparse KKT matrix must not crash the solver.

S2MPJ sweep Task 2 — ``solve_error`` cluster. On EIGMAXA, EIGMINA, LUKVLE8,
MESH, BRATU1D, RAT42LS, RAT43 and INDEF the sparse-direct route raised an
uncaught ``ValueError`` (``matrix value at nnz index N is non-finite``) and the
whole solve aborted. The cause: an upstream Hessian/Jacobian evaluated at a bad
trial iterate overflowed to inf/NaN, and Feral's numeric factorization rejects a
non-finite entry with a bare ``ValueError`` that :class:`FeralSparseSolver` did
not translate — so it escaped instead of being handled like the dense route
already handles a non-finite factorization (δ_w escalation, then a graceful
step-failure classification).

The fix makes the sparse adapter raise :class:`LinearSolveError` for a
non-finite matrix, which the IPM regularization loop catches. This test pins the
end-to-end contract: a solve whose sparse KKT goes non-finite returns a
:class:`~ipax.Result` rather than propagating an exception.
"""

from __future__ import annotations

import pytest

from ipax import Options, Status, solve
from ipax.backend.sparse import get_sparse_adapter
from ipax.testing.problems import HS6
from tests._helpers import array

pytestmark = pytest.mark.sparse


class _HS6InfHessian(HS6):
    """HS6 with a non-finite Lagrangian Hessian, so the sparse KKT is non-finite."""

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        h = super().lagrangian_hessian(x, y_eq, y_ineq, sigma)
        return h + float("inf") * self.xp.ones_like(h)


def _solve_sparse(namespace):
    if get_sparse_adapter(namespace) is None:
        pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")
    return solve(
        _HS6InfHessian(namespace),
        array(namespace, [2.0, 2.0]),
        options=Options(hessian="exact", linsolve="sparse", max_iter=50),
    )


def test_sparse_non_finite_kkt_returns_result_without_raising(namespace):
    # The solve must not propagate the backend's non-finite ``ValueError``: the
    # non-finite factorization is a recoverable numerical failure, so the driver
    # exhausts δ_w escalation and terminates with a status object in hand.
    result = _solve_sparse(namespace)
    assert result.status in (
        Status.NUMERICAL_ERROR,
        Status.ACCEPTABLE,
        Status.MAX_ITER,
        Status.RESTORATION_FAILED,
    ), f"unexpected status {result.status}"
