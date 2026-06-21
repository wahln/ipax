"""Reusable contract battery for ``LinearSolver`` implementations."""

from __future__ import annotations

from tests._helpers import assert_allclose, implemented, norm_inf


class LinearSolverContract:
    """Mixin battery. Subclasses provide ``make_solver`` and ``make_system``."""

    implementation_reason = "solvers"

    def make_solver(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def make_system(self, namespace):  # pragma: no cover - overridden
        """Return ``(K, rhs, x_exact)``."""
        raise NotImplementedError

    def test_solves_system_to_expected_solution(self, namespace, tol):
        with implemented(self.implementation_reason):
            solver = self.make_solver()
            K, rhs, x_exact = self.make_system(namespace)
            solver.factor(K)
            actual = solver.solve(rhs)

        assert_allclose(namespace, actual, x_exact, **tol)

    def test_residual_within_tolerance(self, namespace):
        with implemented(self.implementation_reason):
            solver = self.make_solver()
            K, rhs, _ = self.make_system(namespace)
            solver.factor(K)
            actual = solver.solve(rhs)
            residual = K.matvec(actual) - rhs

        assert norm_inf(namespace, residual) <= 1e-8
