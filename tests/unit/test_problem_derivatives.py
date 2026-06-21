"""Unit tests for finite differences and derivative-source resolution."""

from __future__ import annotations

import pytest

from ipax.options import Options
from ipax.problem.base import Problem
from ipax.problem.derivatives import resolve
from ipax.problem.finitediff import gradient_fd, jacobian_fd
from tests._helpers import array, assert_allclose, implemented


def test_gradient_fd_matches_quadratic_gradient(namespace, tol):
    center = array(namespace, [1.0, -2.0, 0.5])
    x = array(namespace, [0.25, -1.5, 2.0])

    def objective(value):
        diff = value - center
        return namespace.sum(diff * diff)

    with implemented("finite-difference gradient"):
        actual = gradient_fd(objective, x)

    expected = 2.0 * (x - center)
    assert_allclose(namespace, actual, expected, rtol=1e-5, atol=1e-6)


def test_jacobian_fd_matches_vector_function(namespace, tol):
    x = array(namespace, [0.5, -2.0])

    def constraints(value):
        return namespace.stack(
            (
                value[0] + 2.0 * value[1],
                value[0] * value[0] - value[1],
            )
        )

    with implemented("finite-difference Jacobian"):
        actual = jacobian_fd(constraints, x)

    expected = array(namespace, [[1.0, 2.0], [1.0, -1.0]])
    assert_allclose(namespace, actual, expected, rtol=1e-5, atol=1e-6)


def test_resolve_prefers_analytic_gradient(namespace):
    class AnalyticGradientProblem(Problem):
        @property
        def n_vars(self) -> int:
            return 2

        def objective(self, x):
            return namespace.sum(x * x)

        def gradient(self, x):
            return 2.0 * x

    problem = AnalyticGradientProblem()
    with implemented("derivative resolution"):
        resolved = resolve(problem, namespace, Options(enable_autodiff=True))

    assert resolved.sources.gradient == "analytic"
    assert_allclose(
        namespace,
        resolved.gradient(array(namespace, [1.0, -3.0])),
        array(namespace, [2.0, -6.0]),
    )


def test_resolve_raises_when_no_gradient_source_is_enabled(namespace):
    class ObjectiveOnlyProblem(Problem):
        @property
        def n_vars(self) -> int:
            return 1

        def objective(self, x):
            return namespace.sum(x * x)

    try:
        resolve(
            ObjectiveOnlyProblem(),
            namespace,
            Options(enable_autodiff=False, enable_finite_diff=False),
        )
    except NotImplementedError as exc:
        pytest.xfail(f"derivative resolution: {exc}")
    except RuntimeError as exc:
        assert "gradient" in str(exc).lower()
    else:
        pytest.fail("expected a RuntimeError when no gradient source is enabled")


def test_exact_hessian_requires_analytic_operator(namespace):
    class ObjectiveOnlyProblem(Problem):
        @property
        def n_vars(self) -> int:
            return 1

        def objective(self, x):
            return namespace.sum(x * x)

        def gradient(self, x):
            return 2.0 * x

    with pytest.raises(RuntimeError, match="hessian='exact'"):
        resolve(ObjectiveOnlyProblem(), namespace, Options(hessian="exact"))
