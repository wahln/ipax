"""Unit tests for the L-BFGS Hessian operator."""

from __future__ import annotations

import pytest

from ipax import FunctionProblem, Options, Status, solve
from ipax.ipm.hessian import LBFGSOperator
from ipax.options import LBFGSOptions
from ipax.testing.problems import BoundConstrainedQP
from tests._helpers import array, assert_allclose, implemented


def test_lbfgs_operator_has_square_shape(namespace):
    with implemented("L-BFGS"):
        op = LBFGSOperator(4, LBFGSOptions(memory=3))

    assert op.shape == (4, 4)


def test_lbfgs_initial_matvec_is_identity_scaled(namespace, tol):
    v = array(namespace, [1.0, -2.0, 3.0])
    with implemented("L-BFGS"):
        op = LBFGSOperator(3, LBFGSOptions(memory=3))
        actual = op.matvec(v)

    assert_allclose(namespace, actual, v, **tol)


def test_lbfgs_update_keeps_positive_curvature(namespace):
    delta = array(namespace, [1.0, 0.0])
    gamma = array(namespace, [2.0, 0.5])
    v = array(namespace, [0.5, -1.0])

    with implemented("L-BFGS"):
        op = LBFGSOperator(2, LBFGSOptions(memory=3, powell_damping=True))
        op.update(delta, gamma)
        Bv = op.matvec(v)

    curvature = namespace.sum(v * Bv)
    assert float(curvature) > 0.0


def test_lbfgs_initial_scaling_option_controls_seed_curvature(namespace):
    delta = array(namespace, [1.0, 0.0])
    gamma = array(namespace, [2.0, 0.0])
    orthogonal = array(namespace, [0.0, 1.0])

    scaled = LBFGSOperator(2, LBFGSOptions(memory=3, initial_scaling=True))
    unscaled = LBFGSOperator(2, LBFGSOptions(memory=3, initial_scaling=False))
    scaled.update(delta, gamma)
    unscaled.update(delta, gamma)

    assert_allclose(namespace, scaled.matvec(orthogonal), array(namespace, [0.0, 2.0]))
    assert_allclose(
        namespace, unscaled.matvec(orthogonal), array(namespace, [0.0, 1.0])
    )


def test_lbfgs_diagonal_matches_dense_reconstruction(namespace, tol):
    """The compact-form diagonal equals the diagonal of ``B`` applied to ``I``."""
    op = LBFGSOperator(3, LBFGSOptions(memory=5))
    op.update(array(namespace, [1.0, 0.5, -0.5]), array(namespace, [2.0, 1.0, 0.5]))
    op.update(array(namespace, [0.5, -1.0, 1.0]), array(namespace, [1.0, 1.5, 0.5]))

    identity = namespace.eye(3, dtype=array(namespace, [0.0]).dtype)
    dense_b = op.matmat(identity)
    assert_allclose(namespace, op.diagonal(), namespace.linalg.diagonal(dense_b), **tol)


def test_lbfgs_diagonal_unavailable_before_first_pair(namespace):
    op = LBFGSOperator(3, LBFGSOptions(memory=5))
    with pytest.raises(NotImplementedError):
        op.diagonal()


def test_lbfgs_initial_diagonal_uses_template(namespace, tol):
    op = LBFGSOperator(3, LBFGSOptions(memory=5))
    like = array(namespace, [0.0, 0.0, 0.0])

    assert_allclose(namespace, op.diagonal(like), namespace.ones_like(like), **tol)


def test_lbfgs_compact_form_reconstructs_operator(namespace, tol):
    """``compact_form`` returns ``(ξ, U, M)`` consistent with the matvec."""
    op = LBFGSOperator(3, LBFGSOptions(memory=5))
    op.update(array(namespace, [1.0, 0.5, -0.5]), array(namespace, [2.0, 1.0, 0.5]))
    op.update(array(namespace, [0.5, -1.0, 1.0]), array(namespace, [1.0, 1.5, 0.5]))

    xi, u, m = op.compact_form()
    v = array(namespace, [0.5, -1.0, 2.0])
    # B v = ξ v − U M⁻¹ Uᵀ v.
    u_t_v = namespace.matmul(namespace.permute_dims(u, (1, 0)), v)
    expected = xi * v - namespace.matmul(u, namespace.linalg.solve(m, u_t_v))
    assert_allclose(namespace, op.matvec(v), expected, **tol)


def test_lbfgs_matmat_uses_batched_compact_form(namespace, tol):
    """Multi-RHS application uses one compact solve, not column-wise matvecs."""
    op = LBFGSOperator(3, LBFGSOptions(memory=5))
    op.update(array(namespace, [1.0, 0.5, -0.5]), array(namespace, [2.0, 1.0, 0.5]))
    op.update(array(namespace, [0.5, -1.0, 1.0]), array(namespace, [1.0, 1.5, 0.5]))
    V = array(namespace, [[0.5, -1.0], [-1.0, 0.25], [2.0, 1.5]])

    xi, u, m = op.compact_form()
    u_t_v = namespace.matmul(namespace.permute_dims(u, (1, 0)), V)
    expected = xi * V - namespace.matmul(u, namespace.linalg.solve(m, u_t_v))

    def _explode(_):
        raise AssertionError("matmat should not fall back to column-wise matvec")

    op.matvec = _explode
    op.rmatvec = _explode

    assert_allclose(namespace, op.matmat(V), expected, **tol)
    assert_allclose(namespace, op.rmatmat(V), expected, **tol)


def test_lbfgs_dense_matrix_matches_batched_application(namespace, tol):
    op = LBFGSOperator(3, LBFGSOptions(memory=5))
    like = array(namespace, [0.0, 0.0, 0.0])
    identity = namespace.eye(3, dtype=like.dtype)

    assert_allclose(namespace, op.dense_matrix(like), identity, **tol)

    op.update(array(namespace, [1.0, 0.5, -0.5]), array(namespace, [2.0, 1.0, 0.5]))
    op.update(array(namespace, [0.5, -1.0, 1.0]), array(namespace, [1.0, 1.5, 0.5]))

    assert_allclose(namespace, op.dense_matrix(), op.matmat(identity), **tol)


def test_lbfgs_compact_form_unavailable_before_first_pair(namespace):
    op = LBFGSOperator(3, LBFGSOptions(memory=5))
    with pytest.raises(NotImplementedError):
        op.compact_form()


def test_lbfgs_run_matches_exact_optimum_with_bounded_overhead(namespace):
    """L-BFGS reaches the same optimum as the exact Hessian.

    ``BoundConstrainedQP`` supplies the true identity Hessian. The L-BFGS run
    must hit the same point without a blow-up in iteration count.
    """
    exact_problem = BoundConstrainedQP(namespace)
    lower, upper = exact_problem.bounds()

    def objective(x):
        diff = x - exact_problem.center
        return 0.5 * namespace.sum(diff * diff)

    def gradient(x):
        return x - exact_problem.center

    lbfgs_problem = FunctionProblem(
        2,
        objective,
        gradient=gradient,
        bounds=(lower, upper),
    )
    x0 = array(namespace, [0.25, 0.75])

    lbfgs = solve(lbfgs_problem, x0, options=Options(hessian="lbfgs", linsolve="dense"))
    exact = solve(exact_problem, x0, options=Options(hessian="exact", linsolve="dense"))

    assert lbfgs.status is Status.OPTIMAL
    assert lbfgs.derivative_sources.hessian == "lbfgs"
    assert_allclose(
        namespace, lbfgs.x, exact_problem.known_solution(), rtol=1e-6, atol=1e-6
    )
    assert lbfgs.n_iter <= 3 * exact.n_iter + 10
