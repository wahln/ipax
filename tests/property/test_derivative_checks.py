"""Property-based tests for derivatives, operators, and solver invariants."""

from __future__ import annotations

import pytest

from ipax.backend.operators import Dense
from ipax.problem.finitediff import gradient_fd
from ipax.testing.problems import UnconstrainedQuadratic
from tests._helpers import array, assert_allclose, implemented

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
