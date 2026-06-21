"""Unit tests for the dense reference linear solver."""

from __future__ import annotations

import pytest

from ipax.backend.operators import Dense, Diagonal
from ipax.ipm.kkt import build_condensed_operator
from ipax.linalg.dense import DenseSolver
from ipax.linalg.regularize import RegularizationState
from ipax.linalg.solver import LinearSolveError
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


def test_dense_solver_skips_guard_for_plain_operator(namespace, tol):
    # A plain Dense operator exposes no primal_block, so the PD guard is skipped
    # and an ordinary (indefinite-but-nonsingular) solve still succeeds.
    op = Dense(array(namespace, [[0.0, 1.0], [1.0, 0.0]]))
    solver = DenseSolver()
    solver.factor(op)
    actual = solver.solve(array(namespace, [2.0, 3.0]))

    assert_allclose(namespace, actual, array(namespace, [3.0, 2.0]), **tol)
