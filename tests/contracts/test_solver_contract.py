"""Reusable contract battery for ``LinearSolver`` implementations."""

from __future__ import annotations

import pytest

from ipax.linalg.solver import LinearSolveError
from tests._helpers import assert_allclose, implemented, norm_inf


class _BuggyOperator:
    """Proxy whose every callable hook raises ``bug`` (``shape`` stays real).

    Attributes the wrapped operator lacks stay absent, so optional-capability
    probes (``getattr(K, name, None)``) behave as they would on ``K``.
    """

    def __init__(self, inner, bug: BaseException) -> None:
        self._inner = inner
        self._bug = bug

    @property
    def shape(self):
        return self._inner.shape

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr
        bug = self._bug

        def raising(*args, **kwargs):
            del args, kwargs
            raise bug

        return raising


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

    def make_singular_system(self, namespace):
        """Return ``(K, rhs)`` the solver must fail on numerically, or ``None``."""
        del namespace
        return None

    def test_numerical_failure_is_classified(self, namespace):
        # Failure contract (``LinearSolver`` docstring): a singular system
        # raises LinearSolveError, yields a non-finite solution, or (an
        # inertia-revealing LDLᵀ) reports a zero eigenvalue — the signals the
        # IPM answers with δ_w escalation (Wächter & Biegler 2006, §3.1).
        # Never a raw backend exception, which would abort the run.
        with implemented(self.implementation_reason):
            system = self.make_singular_system(namespace)
            if system is None:
                pytest.skip("no singular system provided for this solver")
            K, rhs = system
            solver = self.make_solver()
            try:
                solver.factor(K)
                x = solver.solve(rhs)
            except LinearSolveError:
                return

        inertia_fn = getattr(solver, "inertia_or_none", None)
        inertia = inertia_fn() if inertia_fn is not None else None
        singular_inertia = inertia is not None and inertia[2] > 0
        assert singular_inertia or not bool(namespace.all(namespace.isfinite(x)))

    def test_operator_callback_errors_propagate_unchanged(self, namespace):
        # ...and the dividing line: an exception from the operator's *own*
        # callbacks that is not a numerical failure (a TypeError from a user
        # Jacobian's gram, say) is a bug — it must leave factor/solve as
        # itself, never reclassified as numerical trouble for the ladder.
        with implemented(self.implementation_reason):
            K, rhs, _ = self.make_system(namespace)
            bug = TypeError("asarray() got an unexpected keyword argument 'device'")
            solver = self.make_solver()
            with pytest.raises(TypeError) as info:
                solver.factor(_BuggyOperator(K, bug))
                solver.solve(rhs)

        assert info.value is bug

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
