"""Unit tests for KKT operator assembly."""

from __future__ import annotations

import pytest

from ipax.backend.operators import Dense, Diagonal
from ipax.ipm.hessian import LBFGSOperator
from ipax.ipm.kkt import build_condensed_operator, build_saddle_operator
from ipax.linalg.regularize import RegularizationState
from ipax.options import LBFGSOptions
from tests._helpers import array, assert_allclose, implemented, transpose


class _CountingDense(Dense):
    """Dense operator that records whether materialization is batched."""

    def __init__(self, A):
        super().__init__(A)
        self.matvec_calls = 0
        self.rmatvec_calls = 0
        self.matmat_calls = 0
        self.rmatmat_calls = 0

    def matvec(self, v):
        self.matvec_calls += 1
        return super().matvec(v)

    def rmatvec(self, v):
        self.rmatvec_calls += 1
        return super().rmatvec(v)

    def matmat(self, V):
        self.matmat_calls += 1
        return super().matmat(V)

    def rmatmat(self, V):
        self.rmatmat_calls += 1
        return super().rmatmat(V)


class _MatmatExplodesDense(Dense):
    def matmat(self, V):
        del V
        raise AssertionError("dense_matrix should avoid identity matmat")


def test_condensed_operator_matches_dense_formula(namespace, tol):
    W_dense = array(namespace, [[4.0, 0.5], [0.5, 3.0]])
    sigma_x_dense = array(namespace, [[0.25, 0.0], [0.0, 0.75]])
    J_dense = array(namespace, [[1.0, 2.0], [-1.0, 0.5]])
    sigma_s_dense = array(namespace, [[2.0, 0.0], [0.0, 0.5]])
    v = array(namespace, [0.25, -1.0])

    with implemented("KKT assembly"):
        op = build_condensed_operator(
            Dense(W_dense),
            Dense(sigma_x_dense),
            Diagonal(array(namespace, [2.0, 0.5])),
            Dense(J_dense),
            RegularizationState(delta_w=1e-6, delta_c=1e-8),
        )
        actual = op.matvec(v)

    expected_dense = (
        W_dense
        + sigma_x_dense
        + namespace.matmul(
            transpose(namespace, J_dense),
            namespace.matmul(sigma_s_dense, J_dense),
        )
        + 1e-6 * namespace.eye(2, dtype=v.dtype)
    )
    assert_allclose(namespace, actual, namespace.matmul(expected_dense, v), **tol)


def test_condensed_operator_matmat_batches_block_products(namespace, tol):
    W_dense = array(namespace, [[4.0, 0.5], [0.5, 3.0]])
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    W = _CountingDense(W_dense)
    J_dense = array(namespace, [[1.0, 2.0], [-1.0, 0.5]])
    J = _CountingDense(J_dense)
    V = array(namespace, [[0.25, 1.5, -0.5], [-1.0, 0.75, 2.0]])

    op = build_condensed_operator(
        W, sigma_x, sigma_s, J, RegularizationState(delta_w=1e-6)
    )
    actual = op.matmat(V)

    expected_dense = (
        W_dense
        + namespace.asarray([[0.25, 0.0], [0.0, 0.75]], dtype=V.dtype)
        + namespace.matmul(
            transpose(namespace, J_dense),
            namespace.matmul(
                namespace.asarray([[2.0, 0.0], [0.0, 0.5]], dtype=V.dtype),
                J_dense,
            ),
        )
        + 1e-6 * namespace.eye(2, dtype=V.dtype)
    )
    expected = namespace.matmul(expected_dense, V)
    assert_allclose(namespace, actual, expected, **tol)
    assert W.matmat_calls == 1
    assert J.matmat_calls == 1
    assert J.rmatmat_calls == 1
    assert W.matvec_calls == 0
    assert J.matvec_calls == 0
    assert J.rmatvec_calls == 0


def test_condensed_dense_matrix_matches_materialized(namespace, tol):
    W_dense = array(namespace, [[4.0, 0.5], [0.5, 3.0]])
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    J = Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]]))
    op = build_condensed_operator(
        Dense(W_dense), sigma_x, sigma_s, J, RegularizationState(delta_w=1e-6)
    )

    actual = op.dense_matrix()
    expected = op.matmat(namespace.eye(2, dtype=W_dense.dtype))

    assert_allclose(namespace, actual, expected, **tol)


def test_condensed_dense_matrix_uses_direct_dense_hooks(namespace, tol):
    W_dense = array(namespace, [[4.0, 0.5], [0.5, 3.0]])
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    empty_sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, 2), dtype=W_dense.dtype))
    op = build_condensed_operator(
        _MatmatExplodesDense(W_dense),
        sigma_x,
        empty_sigma_s,
        empty_jac,
        RegularizationState(delta_w=1e-6),
    )

    actual = op.dense_matrix()
    expected = W_dense + namespace.asarray(
        [[0.250001, 0.0], [0.0, 0.750001]], dtype=W_dense.dtype
    )

    assert_allclose(namespace, actual, expected, **tol)


def test_condensed_operator_diagonal_matches_materialized(namespace, tol):
    """The cheap Jacobi diagonal equals the materialized operator's diagonal."""
    W = Dense(array(namespace, [[4.0, 0.5], [0.5, 3.0]]))
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    J = Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]]))

    op = build_condensed_operator(
        W, sigma_x, sigma_s, J, RegularizationState(delta_w=1e-6)
    )
    dense = op.matmat(namespace.eye(2, dtype=array(namespace, [0.0]).dtype))
    assert_allclose(namespace, op.diagonal(), namespace.linalg.diagonal(dense), **tol)


def test_condensed_diagonal_without_inequalities(namespace, tol):
    """With no inequalities the Gram term drops out cleanly."""
    W = Dense(array(namespace, [[4.0, 0.5], [0.5, 3.0]]))
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    empty_sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, 2), dtype=array(namespace, [0.0]).dtype))

    op = build_condensed_operator(
        W, sigma_x, empty_sigma_s, empty_jac, RegularizationState()
    )
    expected = array(namespace, [4.0 + 0.25, 3.0 + 0.75])
    assert_allclose(namespace, op.diagonal(), expected, **tol)


def _lbfgs_operator(namespace):
    op = LBFGSOperator(3, LBFGSOptions(memory=5))
    op.update(array(namespace, [1.0, 0.5, -0.5]), array(namespace, [2.0, 1.0, 0.5]))
    op.update(array(namespace, [0.5, -1.0, 1.0]), array(namespace, [1.0, 1.5, 0.5]))
    return op


def test_condensed_dense_structured_solve_matches_materialized(namespace, tol):
    W = _lbfgs_operator(namespace)
    sigma_x = Diagonal(array(namespace, [0.25, 0.75, 1.25]))
    empty_sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, 3), dtype=array(namespace, [0.0]).dtype))
    rhs = array(namespace, [1.0, -2.0, 0.5])
    op = build_condensed_operator(
        W, sigma_x, empty_sigma_s, empty_jac, RegularizationState(delta_w=1e-6)
    )

    dense = op.matmat(namespace.eye(3, dtype=rhs.dtype))
    actual = op.dense_structured_solve(rhs)
    expected = namespace.linalg.solve(dense, rhs)

    assert_allclose(namespace, actual, expected, **tol)


def test_condensed_dense_structured_solve_handles_diagonal_hessian(namespace, tol):
    W = Diagonal(array(namespace, [2.0, 3.0, 4.0]))
    sigma_x = Diagonal(array(namespace, [0.25, 0.75, 1.25]))
    empty_sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, 3), dtype=array(namespace, [0.0]).dtype))
    rhs = array(namespace, [[1.0, -2.0], [0.5, 2.0], [3.0, -4.0]])
    op = build_condensed_operator(
        W, sigma_x, empty_sigma_s, empty_jac, RegularizationState(delta_w=1e-6)
    )

    dense = op.matmat(namespace.eye(3, dtype=rhs.dtype))
    actual = op.dense_structured_solve(rhs)
    expected = namespace.linalg.solve(dense, rhs)

    assert_allclose(namespace, actual, expected, **tol)


def test_condensed_dense_structured_solve_requires_diagonal_sigma_x(namespace):
    W = _lbfgs_operator(namespace)
    sigma_x = Dense(
        array(
            namespace,
            [[0.25, 0.1, 0.0], [0.1, 0.75, 0.2], [0.0, 0.2, 1.25]],
        )
    )
    empty_sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, 3), dtype=array(namespace, [0.0]).dtype))
    op = build_condensed_operator(
        W, sigma_x, empty_sigma_s, empty_jac, RegularizationState(delta_w=1e-6)
    )

    with pytest.raises(NotImplementedError, match="diagonal Sigma_x"):
        op.dense_structured_solve(array(namespace, [1.0, -2.0, 0.5]))


def test_saddle_operator_matches_dense_bordered_matrix(namespace, tol):
    n_dense = array(namespace, [[4.0, 0.5], [0.5, 3.0]])
    eq_jac = array(namespace, [[1.0, 1.0]])
    delta_c = 1e-8
    v = array(namespace, [0.25, -1.0, 0.5])

    op = build_saddle_operator(Dense(n_dense), Dense(eq_jac), delta_c)
    actual = op.matvec(v)

    top = namespace.concat((n_dense, transpose(namespace, eq_jac)), axis=1)
    bottom = namespace.concat(
        (eq_jac, -delta_c * namespace.eye(1, dtype=v.dtype)), axis=1
    )
    expected_dense = namespace.concat((top, bottom), axis=0)
    assert_allclose(namespace, actual, namespace.matmul(expected_dense, v), **tol)
    assert op.shape == (3, 3)


def test_saddle_operator_matmat_batches_block_products(namespace, tol):
    N_dense = array(namespace, [[4.0, 0.5], [0.5, 3.0]])
    J_dense = array(namespace, [[1.0, 1.0]])
    N = _CountingDense(N_dense)
    J = _CountingDense(J_dense)
    op = build_saddle_operator(N, J, delta_c=1e-8)
    V = array(
        namespace,
        [[0.25, 1.5, -0.5], [-1.0, 0.75, 2.0], [0.5, -0.25, 1.25]],
    )

    actual = op.matmat(V)

    top = namespace.concat((N_dense, transpose(namespace, J_dense)), axis=1)
    bottom = namespace.concat(
        (J_dense, -1e-8 * namespace.eye(1, dtype=V.dtype)), axis=1
    )
    expected = namespace.matmul(namespace.concat((top, bottom), axis=0), V)
    assert_allclose(namespace, actual, expected, **tol)
    assert N.matmat_calls == 1
    assert J.matmat_calls == 1
    assert J.rmatmat_calls == 1
    assert N.matvec_calls == 0
    assert J.matvec_calls == 0
    assert J.rmatvec_calls == 0


def test_saddle_dense_matrix_matches_materialized(namespace, tol):
    condensed = build_condensed_operator(
        Dense(array(namespace, [[4.0, 0.5], [0.5, 3.0]])),
        Diagonal(array(namespace, [0.25, 0.75])),
        Diagonal(array(namespace, [])),
        Dense(namespace.zeros((0, 2), dtype=array(namespace, [0.0]).dtype)),
        RegularizationState(delta_w=1e-6),
    )
    saddle = build_saddle_operator(
        condensed, Dense(array(namespace, [[1.0, 1.0]])), delta_c=1e-4
    )

    actual = saddle.dense_matrix()
    expected = saddle.matmat(namespace.eye(3, dtype=array(namespace, [0.0]).dtype))

    assert_allclose(namespace, actual, expected, **tol)


def test_saddle_dense_structured_solve_matches_materialized(namespace, tol):
    W = _lbfgs_operator(namespace)
    sigma_x = Diagonal(array(namespace, [0.25, 0.75, 1.25]))
    empty_sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, 3), dtype=array(namespace, [0.0]).dtype))
    condensed = build_condensed_operator(
        W, sigma_x, empty_sigma_s, empty_jac, RegularizationState(delta_w=1e-6)
    )
    saddle = build_saddle_operator(
        condensed, Dense(array(namespace, [[1.0, -1.0, 0.5]])), 1e-4
    )
    rhs = array(namespace, [1.0, -2.0, 0.5, 0.25])

    dense = saddle.matmat(namespace.eye(4, dtype=rhs.dtype))
    actual = saddle.dense_structured_solve(rhs)
    expected = namespace.linalg.solve(dense, rhs)

    assert_allclose(namespace, actual, expected, **tol)


def test_saddle_dense_structured_solve_handles_matrix_rhs(namespace, tol):
    W = _lbfgs_operator(namespace)
    sigma_x = Diagonal(array(namespace, [0.25, 0.75, 1.25]))
    empty_sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, 3), dtype=array(namespace, [0.0]).dtype))
    condensed = build_condensed_operator(
        W, sigma_x, empty_sigma_s, empty_jac, RegularizationState(delta_w=1e-6)
    )
    saddle = build_saddle_operator(
        condensed, Dense(array(namespace, [[1.0, -1.0, 0.5]])), 1e-4
    )
    rhs = array(namespace, [[1.0, -2.0], [-2.0, 0.5], [0.5, 1.5], [0.25, -0.75]])

    dense = saddle.matmat(namespace.eye(4, dtype=rhs.dtype))
    actual = saddle.dense_structured_solve(rhs)
    expected = namespace.linalg.solve(dense, rhs)

    assert_allclose(namespace, actual, expected, **tol)


def _empty_ineq(namespace):
    dtype = array(namespace, [0.0]).dtype
    return (
        Diagonal(array(namespace, [])),
        Dense(namespace.zeros((0, 2), dtype=dtype)),
    )


def test_condensed_expected_inertia_assemblable(namespace):
    """Assemblable Hessian: (n positive, m_I negative) for the bordered system."""
    W = Dense(array(namespace, [[4.0, 0.5], [0.5, 3.0]]))
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))

    empty_sigma_s, empty_jac = _empty_ineq(namespace)
    no_ineq = build_condensed_operator(
        W, sigma_x, empty_sigma_s, empty_jac, RegularizationState()
    )
    assert no_ineq.expected_inertia() == (2, 0, 0)

    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    jac = Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]]))
    with_ineq = build_condensed_operator(
        W, sigma_x, sigma_s, jac, RegularizationState()
    )
    assert with_ineq.expected_inertia() == (2, 2, 0)


def test_condensed_expected_inertia_none_for_low_rank_hessian(namespace):
    """A diagonal-plus-low-rank (L-BFGS-style) Hessian disables the inertia check."""

    class _LowRankW(Dense):
        def diagonal_low_rank_form(self):  # presence alone is the signal
            raise NotImplementedError

    W = _LowRankW(array(namespace, [[4.0, 0.5], [0.5, 3.0]]))
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    empty_sigma_s, empty_jac = _empty_ineq(namespace)
    op = build_condensed_operator(
        W, sigma_x, empty_sigma_s, empty_jac, RegularizationState()
    )
    assert op.expected_inertia() is None


def test_saddle_expected_inertia_adds_equalities(namespace):
    """The equality border adds m_E negatives: (n, m_E + m_I, 0)."""
    W = Dense(array(namespace, [[4.0, 0.5], [0.5, 3.0]]))
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    jac = Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]]))
    condensed = build_condensed_operator(
        W, sigma_x, sigma_s, jac, RegularizationState()
    )
    saddle = build_saddle_operator(
        condensed, Dense(array(namespace, [[1.0, 1.0]])), 1e-8
    )
    assert saddle.expected_inertia() == (2, 3, 0)  # n=2, m_E=1, m_I=2


def test_saddle_expected_inertia_none_propagates(namespace):
    """A low-rank condensed block propagates ``None`` through the saddle."""

    class _LowRankW(Dense):
        def diagonal_low_rank_form(self):
            raise NotImplementedError

    empty_sigma_s, empty_jac = _empty_ineq(namespace)
    condensed = build_condensed_operator(
        _LowRankW(array(namespace, [[4.0, 0.5], [0.5, 3.0]])),
        Diagonal(array(namespace, [0.25, 0.75])),
        empty_sigma_s,
        empty_jac,
        RegularizationState(),
    )
    saddle = build_saddle_operator(
        condensed, Dense(array(namespace, [[1.0, 1.0]])), 1e-8
    )
    assert saddle.expected_inertia() is None
