"""Reusable contract battery for ``Problem`` implementations."""

from __future__ import annotations

import pytest

from tests._helpers import array, assert_allclose, central_gradient


class ProblemContract:
    """Mixin battery. Subclasses provide ``make_problem(namespace)``.

    ``make_problem`` may return either ``problem`` or ``(problem, x_sample)``.
    """

    def make_problem(self, namespace):  # pragma: no cover - overridden by subclasses
        raise NotImplementedError

    def _problem_and_point(self, namespace):
        made = self.make_problem(namespace)
        if isinstance(made, tuple):
            return made
        return made, array(namespace, [0.0 for _ in range(made.n_vars)])

    def test_objective_returns_scalar(self, namespace):
        problem, x = self._problem_and_point(namespace)
        value = problem.objective(x)
        if hasattr(value, "shape"):
            assert value.shape == ()
        else:
            assert isinstance(value, float | int)

    def test_gradient_matches_central_difference(self, namespace, tol):
        problem, x = self._problem_and_point(namespace)
        try:
            gradient = problem.gradient(x)
        except NotImplementedError:
            pytest.xfail("Problem implementation leaves gradient to resolution")

        fd_gradient = central_gradient(namespace, problem.objective, x)
        assert_allclose(namespace, gradient, fd_gradient, **tol)

    def test_bounds_have_problem_dimension(self, namespace):
        problem, _ = self._problem_and_point(namespace)
        lower, upper = problem.bounds()
        if lower is not None:
            assert lower.shape == (problem.n_vars,)
        if upper is not None:
            assert upper.shape == (problem.n_vars,)

    def test_linear_constraints_have_static_shapes(self, namespace):
        problem, _ = self._problem_and_point(namespace)
        linear_eq = problem.linear_eq()
        if linear_eq is not None:
            A_eq, b_eq = linear_eq
            assert A_eq.shape[1] == problem.n_vars
            assert b_eq.shape == (A_eq.shape[0],)

        linear_ineq = problem.linear_ineq()
        if linear_ineq is not None:
            A_ineq, lower, upper = linear_ineq
            assert A_ineq.shape[1] == problem.n_vars
            assert lower.shape == (A_ineq.shape[0],)
            assert upper.shape == (A_ineq.shape[0],)
