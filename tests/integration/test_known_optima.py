"""Integration tests: solve curated problems and verify KKT conditions."""

from __future__ import annotations

from ipax import Options, Status, solve
from ipax.problem.base import Problem
from ipax.testing.problems import (
    BoundConstrainedQP,
    EqualityConstrainedQP,
    InfeasibleEqualities,
    UnconstrainedQuadratic,
)
from tests._helpers import array, assert_allclose, assert_scalar_close, implemented


def _assert_optimal_result(result) -> None:
    assert result.status is Status.OPTIMAL
    assert result.success
    assert result.kkt_error <= 1e-6
    assert result.constraint_violation <= 1e-6


def test_unconstrained_quadratic_hits_closed_form(namespace, tol):
    Q = array(namespace, [[4.0, 1.0], [1.0, 3.0]])
    b = array(namespace, [1.0, 2.0])
    problem = UnconstrainedQuadratic(Q, b, namespace)
    x0 = array(namespace, [0.0, 0.0])

    with implemented("dense solver"):
        result = solve(problem, x0, options=Options(hessian="exact", linsolve="dense"))

    _assert_optimal_result(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert_allclose(
        namespace, problem.gradient(result.x), array(namespace, [0.0, 0.0]), **tol
    )


def test_bound_constrained_qp_finds_known_active_bounds(namespace):
    problem = BoundConstrainedQP(namespace)
    x0 = array(namespace, [0.25, 0.75])

    with implemented("bound handling"):
        result = solve(problem, x0, options=Options(hessian="exact", linsolve="dense"))

    _assert_optimal_result(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    z_lower, z_upper = problem.known_bound_multipliers()
    assert_allclose(namespace, result.z_lower, z_lower, rtol=1e-6, atol=1e-6)
    assert_allclose(namespace, result.z_upper, z_upper, rtol=1e-6, atol=1e-6)


def test_inequality_qp_finds_known_active_set(namespace):
    class InequalityQP(Problem):
        @property
        def n_vars(self) -> int:
            return 1

        def objective(self, x):
            return 0.5 * (x[0] - 2.0) * (x[0] - 2.0)

        def gradient(self, x):
            return namespace.stack((x[0] - 2.0,))

        def ineq_constraints(self, x):
            return namespace.stack((x[0] - 1.0,))

        def ineq_jacobian(self, x):
            del x
            return array(namespace, [[1.0]])

        def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
            del y_eq, y_ineq
            return sigma * array(namespace, [[1.0 + 0.0 * x[0]]])

    problem = InequalityQP()
    x0 = array(namespace, [0.5])

    with implemented("inequality handling"):
        result = solve(problem, x0, options=Options(hessian="exact", linsolve="dense"))

    _assert_optimal_result(result)
    assert_allclose(namespace, result.x, array(namespace, [1.0]), rtol=1e-6, atol=1e-6)
    assert_allclose(
        namespace, result.y_ineq, array(namespace, [1.0]), rtol=1e-6, atol=1e-6
    )


def test_equality_constrained_qp_satisfies_kkt(namespace):
    problem = EqualityConstrainedQP(namespace)
    x0 = array(namespace, [0.9, 0.1])

    with implemented("equalities"):
        result = solve(problem, x0, options=Options(hessian="exact", linsolve="dense"))

    _assert_optimal_result(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert_allclose(
        namespace, result.y_eq, problem.known_multiplier(), rtol=1e-6, atol=1e-6
    )
    assert_allclose(
        namespace,
        problem.stationarity(result.x, result.y_eq),
        array(namespace, [0.0, 0.0]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_mixed_linear_and_nonlinear_equalities_slice_hessian_multipliers(namespace):
    class MixedEqualityProblem(Problem):
        @property
        def n_vars(self) -> int:
            return 2

        def objective(self, x):
            return 0.5 * ((x[0] - 2.0) * (x[0] - 2.0) + x[1] * x[1])

        def gradient(self, x):
            return namespace.stack((x[0] - 2.0, x[1]))

        def eq_constraints(self, x):
            return namespace.stack((x[0] * x[0] - 1.0,))

        def eq_jacobian(self, x):
            return namespace.stack(
                (namespace.stack((2.0 * x[0], namespace.zeros_like(x[1]))),)
            )

        def linear_eq(self):
            return array(namespace, [[0.0, 1.0]]), array(namespace, [0.0])

        def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
            del y_ineq
            assert int(y_eq.shape[0]) == 1
            zero = namespace.zeros_like(x[0])
            h00 = sigma + 2.0 * y_eq[0]
            h11 = sigma + zero
            return namespace.stack(
                (namespace.stack((h00, zero)), namespace.stack((zero, h11)))
            )

    result = solve(
        MixedEqualityProblem(),
        array(namespace, [1.2, 0.25]),
        options=Options(hessian="exact", linsolve="dense"),
    )

    _assert_optimal_result(result)
    assert_allclose(namespace, result.x, array(namespace, [1.0, 0.0]), atol=1e-6)


def test_infeasible_problem_reports_infeasible(namespace):
    class InfeasibleBounds(Problem):
        @property
        def n_vars(self) -> int:
            return 1

        def bounds(self):
            return array(namespace, [1.0]), array(namespace, [0.0])

        def objective(self, x):
            return namespace.sum(x * x)

        def gradient(self, x):
            return 2.0 * x

    with implemented("restoration"):
        result = solve(InfeasibleBounds(), array(namespace, [0.5]), options=Options())

    assert result.status is Status.INFEASIBLE
    assert not result.success
    assert "infeasible" in result.message.lower()
    # device must be reported on the early-return failure path too, consistent
    # with the main solve path (Copilot PR #2 review).
    assert result.device == str(array(namespace, [0.5]).device)


def test_inconsistent_equalities_report_infeasible(namespace):
    result = solve(InfeasibleEqualities(namespace), array(namespace, [0.5]))
    assert result.status is Status.INFEASIBLE
    assert not result.success
    assert "infeasible" in result.message.lower()
    assert result.constraint_violation > 1e-6


def test_result_history_tracks_monotone_kkt_progress(namespace):
    Q = array(namespace, [[2.0, 0.0], [0.0, 1.0]])
    b = array(namespace, [1.0, -1.0])
    problem = UnconstrainedQuadratic(Q, b, namespace)

    with implemented("driver iteration history"):
        result = solve(problem, array(namespace, [10.0, -10.0]), options=Options())

    errors = [record.kkt_error for record in result.history]
    assert result.n_iter == len(result.history)
    assert errors
    assert_scalar_close(errors[-1], result.kkt_error, rtol=0.0, atol=1e-12)
    assert errors[-1] <= errors[0]
