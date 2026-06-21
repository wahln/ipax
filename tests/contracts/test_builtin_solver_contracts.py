"""Contract coverage for built-in ``LinearSolver`` implementations."""

from __future__ import annotations

from ipax.backend.operators import Dense
from ipax.linalg.dense import DenseSolver
from ipax.linalg.krylov import KrylovSolver
from ipax.options import KrylovOptions
from tests._helpers import array
from tests.contracts.test_solver_contract import LinearSolverContract


def _spd_system(namespace):
    A = array(namespace, [[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    x_exact = array(namespace, [1.0, -2.0, 0.5])
    rhs = namespace.matmul(A, x_exact)
    return Dense(A), rhs, x_exact


class TestDenseSolver(LinearSolverContract):
    implementation_reason = "DenseSolver"

    def make_solver(self):
        return DenseSolver()

    def make_system(self, namespace):
        return _spd_system(namespace)


class TestKrylovSolver(LinearSolverContract):
    implementation_reason = "KrylovSolver"

    def make_solver(self):
        # Tight rtol so the matrix-free iteration matches the reference optimum
        # to the contract battery's tolerance ladder.
        return KrylovSolver(KrylovOptions(method="cg", rtol=1e-12))

    def make_system(self, namespace):
        return _spd_system(namespace)
