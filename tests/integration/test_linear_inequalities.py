"""Integration tests for two-sided linear inequalities (``l ≤ A x ≤ u``).

The solver lowers a constant-data ``linear_ineq`` block into the standard
one-sided inequality machinery, so these exercise the full IPM path (slacks,
multipliers, fraction-to-boundary, filter line search) across every backend.
"""

from __future__ import annotations

import pytest

from ipax import Options, Status, solve
from ipax.problem.base import Problem
from ipax.testing.problems import HS21
from tests._helpers import array, assert_allclose, assert_scalar_close

_EXACT_DENSE = Options(hessian="exact", linsolve="dense")


def _assert_optimal(result) -> None:
    assert result.status is Status.OPTIMAL
    assert result.kkt_error <= 1e-6
    assert result.constraint_violation <= 1e-6


class _LowerBoundedQP(Problem):
    """``min 0.5‖x - c‖²`` s.t. ``A x ≥ b`` (one-sided lower) via ``linear_ineq``.

    With ``c = (0, 0)``, ``A = [[1, 1]]`` and ``b = 2`` the unconstrained
    minimizer ``(0, 0)`` is infeasible; the optimum is the projection onto the
    line ``x1 + x2 = 2`` ⇒ ``(1, 1)``, ``f* = 1``.
    """

    def __init__(self, xp) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        return 0.5 * self.xp.sum(x * x)

    def gradient(self, x):
        return x

    def linear_ineq(self):
        xp = self.xp
        a = array(xp, [[1.0, 1.0]])
        return a, array(xp, [2.0]), array(xp, [float("inf")])

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del y_eq, y_ineq
        return sigma * self.xp.eye(2, dtype=x.dtype)

    def known_solution(self):
        return array(self.xp, [1.0, 1.0])


class _RangeBoxQP(Problem):
    """``min 0.5‖x - c‖²`` s.t. ``l ≤ A x ≤ u`` with both sides finite.

    ``c = (3, 0)``, ``A = [[1, 0]]`` (reads ``x1``), ``1 ≤ x1 ≤ 2``. The
    unconstrained min wants ``x1 = 3``; the active upper row drives ``x1 = 2``,
    ``x2 = 0``, ``f* = 0.5``.
    """

    def __init__(self, xp) -> None:
        self.xp = xp
        self.center = array(xp, [3.0, 0.0])

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        diff = x - self.center
        return 0.5 * self.xp.sum(diff * diff)

    def gradient(self, x):
        return x - self.center

    def linear_ineq(self):
        xp = self.xp
        a = array(xp, [[1.0, 0.0]])
        return a, array(xp, [1.0]), array(xp, [2.0])

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del y_eq, y_ineq
        return sigma * self.xp.eye(2, dtype=x.dtype)

    def known_solution(self):
        return array(self.xp, [2.0, 0.0])


class _MixedIneqQP(Problem):
    """A nonlinear inequality *and* a two-sided linear inequality together.

    ``min 0.5‖x - (3, 3)‖²`` s.t. ``‖x‖² ≤ 2`` (nonlinear) and ``x1 ≤ 5``
    (linear, ``-inf ≤ x1 ≤ 5``, deliberately slack). The disc constraint binds;
    the optimum lies on ``‖x‖² = 2`` nearest ``(3, 3)`` ⇒ ``(1, 1)`` (a single,
    non-degenerate active constraint), ``f* = 0.5·((1-3)² + (1-3)²) = 4``.
    """

    def __init__(self, xp) -> None:
        self.xp = xp
        self.center = array(xp, [3.0, 3.0])

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        diff = x - self.center
        return 0.5 * self.xp.sum(diff * diff)

    def gradient(self, x):
        return x - self.center

    def ineq_constraints(self, x):
        return self.xp.stack((x[0] * x[0] + x[1] * x[1] - 2.0,))

    def ineq_jacobian(self, x):
        xp = self.xp
        return xp.stack((xp.stack((2.0 * x[0], 2.0 * x[1])),))

    def linear_ineq(self):
        xp = self.xp
        a = array(xp, [[1.0, 0.0]])
        return a, array(xp, [float("-inf")]), array(xp, [5.0])

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del y_eq
        xp = self.xp
        # Only the nonlinear inequality contributes curvature (2·y·I); the linear
        # block must not be indexed here, so y_ineq is the nonlinear part only.
        assert int(y_ineq.shape[0]) == 1
        h = sigma + 2.0 * y_ineq[0]
        zero = xp.zeros_like(x[0])
        return xp.stack((xp.stack((h, zero)), xp.stack((zero, h))))

    def known_solution(self):
        return array(self.xp, [1.0, 1.0])


def test_lower_bounded_linear_inequality(namespace):
    problem = _LowerBoundedQP(namespace)
    result = solve(problem, array(namespace, [0.0, 0.0]), options=_EXACT_DENSE)
    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert_scalar_close(result.objective, 1.0, atol=1e-6)
    # One active lower row ⇒ a single positive inequality multiplier.
    assert result.y_ineq is not None
    assert int(result.y_ineq.shape[0]) == 1
    assert float(result.y_ineq[0]) > 0.0


def test_two_sided_range_upper_active(namespace):
    problem = _RangeBoxQP(namespace)
    result = solve(problem, array(namespace, [1.5, 0.0]), options=_EXACT_DENSE)
    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert_scalar_close(result.objective, 0.5, atol=1e-6)
    # Lowered to two rows (lower then upper); only the upper row is active.
    assert result.y_ineq is not None
    assert int(result.y_ineq.shape[0]) == 2


def test_mixed_nonlinear_and_linear_inequalities(namespace):
    problem = _MixedIneqQP(namespace)
    result = solve(problem, array(namespace, [0.5, 0.5]), options=_EXACT_DENSE)
    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert_scalar_close(result.objective, 4.0, atol=1e-6)
    # One nonlinear inequality + one lowered linear inequality.
    assert result.y_ineq is not None
    assert int(result.y_ineq.shape[0]) == 2


def test_linear_inequality_with_gradient_scaling(namespace):
    problem = _LowerBoundedQP(namespace)
    result = solve(
        problem,
        array(namespace, [0.0, 0.0]),
        options=Options(hessian="exact", linsolve="dense", scaling="gradient-based"),
    )
    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)


def test_hs21_via_linear_ineq_matches_known_optimum(namespace):
    problem = HS21(namespace)
    result = solve(problem, array(namespace, [3.0, 1.0]), options=_EXACT_DENSE)
    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert_scalar_close(result.objective, -99.96, atol=1e-6)
    z_lower, z_upper = problem.known_bound_multipliers()
    assert_allclose(namespace, result.z_lower, z_lower, rtol=1e-6, atol=1e-6)
    assert_allclose(namespace, result.z_upper, z_upper, rtol=1e-6, atol=1e-6)


def test_matrix_free_linear_ineq_is_rejected(namespace):
    from ipax.backend.operators import Identity

    class _MatrixFreeLinearIneq(Problem):
        @property
        def n_vars(self) -> int:
            return 2

        def objective(self, x):
            return namespace.sum(x * x)

        def gradient(self, x):
            return 2.0 * x

        def linear_ineq(self):
            return (
                Identity(2),
                array(namespace, [0.0, 0.0]),
                array(namespace, [1.0, 1.0]),
            )

    with pytest.raises(NotImplementedError, match="ineq_constraints"):
        solve(_MatrixFreeLinearIneq(), array(namespace, [0.5, 0.5]))
