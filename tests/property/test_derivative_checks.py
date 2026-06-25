"""Property-based tests for derivatives, operators, and solver invariants."""

from __future__ import annotations

import pytest

from ipax.backend.operators import Dense
from ipax.problem.finitediff import gradient_fd, jacobian_fd
from ipax.testing.problems import (
    HS6,
    HS7,
    HS8,
    HS9,
    HS21,
    HS28,
    HS35,
    HS43,
    HS71,
    UnconstrainedQuadratic,
)
from tests._helpers import array, assert_allclose, implemented, transpose

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")


@hypothesis.given(
    a=st.floats(min_value=0.25, max_value=10.0, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=0.25, max_value=10.0, allow_nan=False, allow_infinity=False),
    x0=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    x1=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture]
)
def test_gradient_fd_matches_random_diagonal_quadratics(namespace, a, b, x0, x1):
    Q = array(namespace, [[a, 0.0], [0.0, b]])
    linear = array(namespace, [1.5, -0.25])
    problem = UnconstrainedQuadratic(Q, linear, namespace)
    x = array(namespace, [x0, x1])

    with implemented("derivative harness"):
        actual = gradient_fd(problem.objective, x)

    assert_allclose(namespace, actual, problem.gradient(x), rtol=1e-5, atol=1e-5)


@hypothesis.given(
    entries=st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=6,
        max_size=6,
    ),
    vec=st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=3,
    ),
    covec=st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=2,
    ),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture]
)
def test_operator_adjoint_identity_over_random_vectors(namespace, entries, vec, covec):
    A = array(namespace, [entries[:3], entries[3:]])
    v = array(namespace, vec)
    w = array(namespace, covec)

    with implemented("operators"):
        op = Dense(A)
        Av = op.matvec(v)
        Atw = op.rmatvec(w)

    left = namespace.sum(Av * w)
    right = namespace.sum(v * Atw)
    assert_allclose(namespace, left, right, rtol=1e-8, atol=1e-8)


# Interior test points, chosen away from singularities, for the FD oracle check.
_HS_DERIVATIVE_CASES = [
    (HS6, [-1.0, 1.5]),
    (HS7, [1.0, 2.0]),
    (HS8, [3.0, 2.0]),
    (HS9, [1.0, 2.0]),
    (HS21, [3.0, 1.0]),
    (HS28, [-1.0, 0.5, 0.5]),
    (HS35, [0.5, 0.5, 0.5]),
    (HS43, [0.5, -0.5, 1.0, -0.5]),
    (HS71, [1.5, 4.0, 3.5, 1.5]),
]


def _maybe(call):
    """Return ``call()`` or ``None`` if the optional method is not implemented."""
    try:
        return call()
    except NotImplementedError:
        return None


@pytest.mark.parametrize(
    "problem_cls, point",
    _HS_DERIVATIVE_CASES,
    ids=[cls.__name__ for cls, _ in _HS_DERIVATIVE_CASES],
)
def test_hs_oracle_derivatives_match_finite_differences(namespace, problem_cls, point):
    """Analytic grad/Jacobian/Lagrangian-Hessian agree with central differences.

    Guards the hand-coded HS derivatives (an error-prone surface) and doubles as
    the derivative-harness coverage for the analytic oracle problems.
    """
    xp = namespace
    problem = problem_cls(xp)
    x = array(xp, point)

    # Objective gradient.
    assert_allclose(
        xp, problem.gradient(x), gradient_fd(problem.objective, x), rtol=1e-5, atol=1e-5
    )

    # Nonlinear constraint Jacobians (linear constraints carry no Jacobian here).
    eq = _maybe(lambda: problem.eq_constraints(x))
    if eq is not None:
        assert_allclose(
            xp,
            problem.eq_jacobian(x),
            jacobian_fd(problem.eq_constraints, x),
            rtol=1e-5,
            atol=1e-5,
        )
    ineq = _maybe(lambda: problem.ineq_constraints(x))
    if ineq is not None:
        assert_allclose(
            xp,
            problem.ineq_jacobian(x),
            jacobian_fd(problem.ineq_constraints, x),
            rtol=1e-5,
            atol=1e-5,
        )

    # Lagrangian Hessian vs FD of the Lagrangian gradient. Only the nonlinear
    # constraints contribute curvature; linear blocks contribute nothing.
    sigma = 1.3
    n_eq = 0 if eq is None else int(eq.shape[0])
    n_ineq = 0 if ineq is None else int(ineq.shape[0])
    y_eq = array(xp, [0.7 + 0.1 * i for i in range(n_eq)])
    y_ineq = array(xp, [0.3 + 0.1 * i for i in range(n_ineq)])

    def lagrangian_gradient(z):
        grad = sigma * problem.gradient(z)
        if n_eq:
            grad = grad + xp.matmul(transpose(xp, problem.eq_jacobian(z)), y_eq)
        if n_ineq:
            grad = grad + xp.matmul(transpose(xp, problem.ineq_jacobian(z)), y_ineq)
        return grad

    hessian = problem.lagrangian_hessian(x, y_eq, y_ineq, sigma)
    assert_allclose(
        xp, hessian, jacobian_fd(lagrangian_gradient, x), rtol=1e-4, atol=1e-4
    )


@hypothesis.given(
    scale=st.floats(
        min_value=0.25, max_value=10.0, allow_nan=False, allow_infinity=False
    )
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture]
)
def test_quadratic_solution_is_invariant_to_positive_objective_scaling(
    namespace, scale
):
    Q = array(namespace, [[4.0, 1.0], [1.0, 3.0]])
    b = array(namespace, [1.0, 2.0])
    original = UnconstrainedQuadratic(Q, b, namespace)
    scaled = UnconstrainedQuadratic(scale * Q, scale * b, namespace)

    assert_allclose(
        namespace,
        original.known_solution(),
        scaled.known_solution(),
        rtol=1e-10,
        atol=1e-10,
    )
