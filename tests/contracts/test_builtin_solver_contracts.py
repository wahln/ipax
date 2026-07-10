"""Contract coverage for built-in ``LinearSolver`` implementations."""

from __future__ import annotations

import pytest

from ipax.backend.operators import COOOperator, Dense, Diagonal
from ipax.backend.sparse import get_sparse_adapter
from ipax.ipm.kkt import build_condensed_operator
from ipax.linalg.dense import DenseSolver
from ipax.linalg.krylov import KrylovSolver
from ipax.linalg.regularize import RegularizationState
from ipax.linalg.sparse import SparseDirectSolver
from ipax.options import DenseOptions, KrylovOptions
from tests._helpers import array
from tests.contracts.test_solver_contract import LinearSolverContract


def _spd_system(namespace):
    A = array(namespace, [[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    x_exact = array(namespace, [1.0, -2.0, 0.5])
    rhs = namespace.matmul(A, x_exact)
    return Dense(A), rhs, x_exact


def _condensed_with_inequalities_system(namespace):
    # A genuine inequality border, so the augmented route actually engages
    # (as opposed to silently falling back to the condensed route).
    op = build_condensed_operator(
        Dense(array(namespace, [[4.0, 0.5], [0.5, 3.0]])),
        Diagonal(array(namespace, [0.25, 0.75])),
        Diagonal(array(namespace, [2.0, 0.5])),
        Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]])),
        RegularizationState(delta_w=1e-6),
    )
    x_exact = array(namespace, [1.0, -2.0])
    rhs = op.matvec(x_exact)
    return op, rhs, x_exact


class TestDenseSolver(LinearSolverContract):
    implementation_reason = "DenseSolver"

    def make_solver(self):
        return DenseSolver()

    def make_system(self, namespace):
        return _spd_system(namespace)


class TestDenseSolverAugmented(LinearSolverContract):
    implementation_reason = "DenseSolver (augmented route)"

    def make_solver(self):
        return DenseSolver(DenseOptions(kkt_route="augmented"))

    def make_system(self, namespace):
        return _condensed_with_inequalities_system(namespace)


class TestKrylovSolver(LinearSolverContract):
    implementation_reason = "KrylovSolver"

    def make_solver(self):
        # Tight rtol so the matrix-free iteration matches the reference optimum
        # to the contract battery's tolerance ladder.
        return KrylovSolver(KrylovOptions(method="cg", rtol=1e-12))

    def make_system(self, namespace):
        return _spd_system(namespace)


def _condensed_sparse_ne_system(namespace):
    # Sparse W and sparse ∇g so the normal-equations form actually engages.
    if get_sparse_adapter(namespace) is None:
        pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")
    W = COOOperator(
        namespace.asarray([0, 1, 0, 1]),
        namespace.asarray([0, 1, 1, 0]),
        array(namespace, [4.0, 3.0, 0.5, 0.5]),
        (2, 2),
        symmetric=True,
        pattern_key="W",
    )
    jac = COOOperator(
        namespace.asarray([0, 0, 1, 2]),
        namespace.asarray([0, 1, 0, 1]),
        array(namespace, [1.0, 2.0, -1.0, 0.5]),
        (3, 2),
        pattern_key="G",
    )
    op = build_condensed_operator(
        W,
        Diagonal(array(namespace, [0.25, 0.75])),
        Diagonal(array(namespace, [2.0, 0.5, 1.0])),
        jac,
        RegularizationState(delta_w=1e-6),
    )
    x_exact = array(namespace, [1.0, -2.0])
    rhs = op.matvec(x_exact)
    return op, rhs, x_exact


class TestSparseDirectSolverNormalEquations(LinearSolverContract):
    implementation_reason = "SparseDirectSolver (normal-equations form)"

    def make_solver(self):
        return SparseDirectSolver(form="normal_equations")

    def make_system(self, namespace):
        return _condensed_sparse_ne_system(namespace)
