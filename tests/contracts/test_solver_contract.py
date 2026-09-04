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

    def test_reuses_factorization_for_a_second_rhs(self, namespace):
        # One factor() must serve multiple solve() calls with different
        # right-hand sides — the driver's default path relies on it (SOC and
        # the centrality correctors re-solve the step's system, W&B 2006
        # §2.4 eq. (26)).
        with implemented(self.implementation_reason):
            solver = self.make_solver()
            K, rhs, _ = self.make_system(namespace)
            solver.factor(K)
            first = solver.solve(rhs)
            second = solver.solve(2.0 * rhs)

        assert norm_inf(namespace, K.matvec(first) - rhs) <= 1e-8
        assert norm_inf(namespace, K.matvec(second) - 2.0 * rhs) <= 1e-7

    def test_is_direct_is_a_bool_method_when_exposed(self):
        # Optional capability hook: the driver's SOC reuse policy reads it
        # through duck typing (absent = direct). When present it must be a
        # method returning a plain bool, and it must not depend on factor().
        solver = self.make_solver()
        is_direct = getattr(solver, "is_direct", None)
        if is_direct is None:
            return
        assert callable(is_direct)
        assert is_direct() in (True, False)
        assert type(is_direct()) is bool
