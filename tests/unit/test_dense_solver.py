"""Unit tests for the dense reference linear solver."""

from __future__ import annotations

import pytest

from ipax.backend.namespace import array_namespace
from ipax.backend.operators import Dense, Diagonal, LinearOperator
from ipax.ipm.hessian import LBFGSOperator
from ipax.ipm.kkt import build_condensed_operator, build_saddle_operator
from ipax.linalg.dense import DenseSolver
from ipax.linalg.regularize import RegularizationState
from ipax.linalg.solver import LinearSolveError
from ipax.options import LBFGSOptions
from tests._helpers import array, assert_allclose


def test_dense_solver_wraps_backend_singular_failure(namespace):
    operator = Dense(array(namespace, [[1.0, 2.0], [2.0, 4.0]]))
    solver = DenseSolver()
    solver.factor(operator)

    with pytest.raises(LinearSolveError, match="dense linear solve failed"):
        solver.solve(array(namespace, [1.0, 2.0]))


def _condensed(namespace, w_dense, *, delta_w=0.0):
    dtype = array(namespace, [0.0]).dtype
    return build_condensed_operator(
        Dense(w_dense),
        Diagonal(array(namespace, [0.0, 0.0])),
        Diagonal(array(namespace, [])),
        Dense(namespace.zeros((0, 2), dtype=dtype)),
        RegularizationState(delta_w=delta_w),
    )


def test_dense_solver_rejects_indefinite_condensed_block(namespace):
    # N = diag(1, -1) is indefinite: xp.linalg.solve (LU) would accept it, but the
    # Cholesky PD guard must reject it so the IPM escalates δ_w.
    op = _condensed(namespace, array(namespace, [[1.0, 0.0], [0.0, -1.0]]))
    solver = DenseSolver()
    solver.factor(op)

    with pytest.raises(LinearSolveError, match="not positive definite"):
        solver.solve(array(namespace, [1.0, 1.0]))


def test_dense_solver_accepts_pd_condensed_block(namespace, tol):
    op = _condensed(namespace, array(namespace, [[2.0, 0.0], [0.0, 3.0]]))
    solver = DenseSolver()
    solver.factor(op)
    actual = solver.solve(array(namespace, [2.0, 3.0]))

    assert_allclose(namespace, actual, array(namespace, [1.0, 1.0]), **tol)


def test_dense_solver_prefers_structured_solve(namespace, tol):
    class _StructuredOnly:
        shape = (2, 2)

        def __init__(self):
            self.called = False

        def dense_structured_solve(self, rhs):
            self.called = True
            return 2.0 * rhs

        def matmat(self, V):
            raise AssertionError("structured solve should avoid materialization")

    op = _StructuredOnly()
    solver = DenseSolver()
    solver.factor(op)

    actual = solver.solve(array(namespace, [1.0, 2.0]))

    assert op.called
    assert_allclose(namespace, actual, array(namespace, [2.0, 4.0]), **tol)


def test_dense_solver_reuses_materialized_leading_primal_block(namespace, tol):
    class _ExplodingPrimal:
        shape = (2, 2)

        def matmat(self, V):
            raise AssertionError("primal block should already be materialized")

    class _LeadingPrimalSaddle(LinearOperator):
        shape = (3, 3)

        def __init__(self, matrix):
            self._matrix = matrix

        def matvec(self, v):
            xp = array_namespace(self._matrix, v)
            return xp.matmul(self._matrix, v)

        def matmat(self, V):
            xp = array_namespace(self._matrix, V)
            return xp.matmul(self._matrix, V)

        def primal_block(self):
            return _ExplodingPrimal()

    matrix = array(
        namespace,
        [[2.0, 0.25, 1.0], [0.25, 3.0, -1.0], [1.0, -1.0, -1.0]],
    )
    rhs = array(namespace, [1.0, 2.0, 3.0])
    solver = DenseSolver()
    solver.factor(_LeadingPrimalSaddle(matrix))

    actual = solver.solve(rhs)

    assert_allclose(namespace, namespace.matmul(matrix, actual), rhs, **tol)


def _lbfgs_operator(namespace, n):
    op = LBFGSOperator(n, LBFGSOptions(memory=5))
    delta = array(namespace, [1.0, 0.5] + [0.0] * (n - 2))
    gamma = array(namespace, [2.0, 1.0] + [0.5] * (n - 2))
    op.update(delta, gamma)
    return op


def test_dense_solver_falls_back_when_condensed_has_inequalities(namespace, tol):
    # An L-BFGS condensed block *with* an inequality Gram term has no exact
    # structured solve, so the solver must materialize and factor instead.
    W = _lbfgs_operator(namespace, 2)
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    jac = Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]]))
    op = build_condensed_operator(
        W, sigma_x, sigma_s, jac, RegularizationState(delta_w=1e-6)
    )
    rhs = array(namespace, [1.0, -2.0])
    solver = DenseSolver()
    solver.factor(op)

    with pytest.raises(NotImplementedError):
        op.dense_structured_solve(rhs)  # structured path unavailable -> fall back
    actual = solver.solve(rhs)

    expected = namespace.linalg.solve(op.matmat(namespace.eye(2, dtype=rhs.dtype)), rhs)
    assert_allclose(namespace, actual, expected, **tol)


def test_dense_solver_falls_back_for_saddle_without_structured_condensed(
    namespace, tol
):
    # A saddle whose condensed block is a plain (non-L-BFGS) Hessian cannot use the
    # Woodbury Schur path; the NotImplementedError propagates and the solver
    # materializes the bordered system.
    condensed = _condensed(
        namespace, array(namespace, [[2.0, 0.5], [0.5, 3.0]]), delta_w=1e-6
    )
    saddle = build_saddle_operator(
        condensed, Dense(array(namespace, [[1.0, -1.0]])), 1e-4
    )
    rhs = array(namespace, [1.0, -2.0, 0.5])
    solver = DenseSolver()
    solver.factor(saddle)

    with pytest.raises(NotImplementedError):
        saddle.dense_structured_solve(rhs)  # condensed has no compact form
    actual = solver.solve(rhs)

    expected = namespace.linalg.solve(
        saddle.matmat(namespace.eye(3, dtype=rhs.dtype)), rhs
    )
    assert_allclose(namespace, actual, expected, **tol)


def test_dense_solver_skips_guard_for_plain_operator(namespace, tol):
    # A plain Dense operator exposes no primal_block, so the PD guard is skipped
    # and an ordinary (indefinite-but-nonsingular) solve still succeeds.
    op = Dense(array(namespace, [[0.0, 1.0], [1.0, 0.0]]))
    solver = DenseSolver()
    solver.factor(op)
    actual = solver.solve(array(namespace, [2.0, 3.0]))

    assert_allclose(namespace, actual, array(namespace, [3.0, 2.0]), **tol)
