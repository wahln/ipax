"""Unit tests for KKT operator assembly."""

from __future__ import annotations

import pytest

from ipax.backend.namespace import array_namespace
from ipax.backend.operators import Dense, Diagonal, LinearOperator
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


# --- solve-kernel helpers (module-private, tested directly) -----------------


def test_diagonal_solve_vector_and_bad_rank(namespace, tol):
    from ipax.ipm.kkt import _diagonal_solve

    d = array(namespace, [2.0, 4.0])
    actual = _diagonal_solve(d, array(namespace, [1.0, 2.0]))
    assert_allclose(namespace, actual, array(namespace, [0.5, 0.5]), **tol)
    with pytest.raises(ValueError, match="vector or matrix"):
        _diagonal_solve(d, namespace.zeros((2, 1, 1), dtype=d.dtype))


def test_woodbury_solve_rejects_bad_rank(namespace):
    from ipax.ipm.kkt import _woodbury_factors, _woodbury_solve

    d = array(namespace, [2.0, 2.0])
    u = array(namespace, [[1.0], [0.5]])
    m = array(namespace, [[3.0]])
    factors = _woodbury_factors(d, u, m)
    with pytest.raises(ValueError, match="vector or matrix"):
        _woodbury_solve(factors, namespace.zeros((2, 1, 1), dtype=d.dtype))


# --- batched transposes (symmetric blocks: A.T @ V == A @ V) ----------------


def test_condensed_and_saddle_rmatmat_match_matmat(namespace, tol):
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    jac = Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]]))
    condensed = build_condensed_operator(
        Dense(array(namespace, [[4.0, 0.5], [0.5, 3.0]])),
        Diagonal(array(namespace, [0.25, 0.75])),
        sigma_s,
        jac,
        RegularizationState(delta_w=1e-6),
    )
    V = array(namespace, [[1.0, 2.0], [3.0, 4.0]])
    assert_allclose(namespace, condensed.rmatmat(V), condensed.matmat(V), **tol)

    saddle = build_saddle_operator(
        condensed, Dense(array(namespace, [[1.0, 1.0]])), 1e-8
    )
    Vs = array(namespace, [[1.0, 2.0], [3.0, 4.0], [0.5, -1.0]])
    assert_allclose(namespace, saddle.rmatmat(Vs), saddle.matmat(Vs), **tol)


# --- dense materialization of non-diagonal Sigma blocks ---------------------


def test_condensed_dense_matrix_general_sigma_blocks(namespace, tol):
    # Sigma_x carries no cheap diagonal and Sigma_s is a full matrix, so the
    # materialization must take the general dense_matrix branches (and derive
    # its dtype template from W instead of Sigma_x's diagonal).
    class _NoDiagonalDense(Dense):
        def diagonal(self, like=None):
            raise NotImplementedError("no cheap diagonal")

    W_dense = array(namespace, [[4.0, 0.5], [0.5, 3.0]])
    sigma_x_dense = array(namespace, [[0.25, 0.1], [0.1, 0.75]])
    sigma_s_dense = array(namespace, [[2.0, 0.25], [0.25, 0.5]])
    J_dense = array(namespace, [[1.0, 2.0], [-1.0, 0.5]])
    op = build_condensed_operator(
        Dense(W_dense),
        _NoDiagonalDense(sigma_x_dense),
        Dense(sigma_s_dense),
        Dense(J_dense),
        RegularizationState(delta_w=1e-6),
    )

    expected = (
        W_dense
        + sigma_x_dense
        + namespace.matmul(
            transpose(namespace, J_dense),
            namespace.matmul(sigma_s_dense, J_dense),
        )
        + 1e-6 * namespace.eye(2, dtype=W_dense.dtype)
    )
    assert_allclose(namespace, op.dense_matrix(), expected, **tol)


# --- diagonal-Hessian structured solve ---------------------------------------


def test_condensed_diagonal_hessian_structured_solve(namespace, tol):
    # W, Sigma_x diagonal and no inequalities: N = diag(W + Sigma_x + delta_w),
    # solved directly without materialization (incl. the delta_w shift).
    empty_sigma_s, empty_jac = _empty_ineq(namespace)
    op = build_condensed_operator(
        Diagonal(array(namespace, [1.0, 3.0])),
        Diagonal(array(namespace, [0.5, 1.0])),
        empty_sigma_s,
        empty_jac,
        RegularizationState(delta_w=0.5),
    )
    actual = op.dense_structured_solve(array(namespace, [2.0, 4.5]))
    assert_allclose(namespace, actual, array(namespace, [1.0, 1.0]), **tol)

    # Unregularized: N = diag(W + Sigma_x) with no delta_w shift.
    unregularized = build_condensed_operator(
        Diagonal(array(namespace, [1.0, 3.0])),
        Diagonal(array(namespace, [0.5, 1.0])),
        *_empty_ineq(namespace),
        RegularizationState(delta_w=0.0),
    )
    actual = unregularized.dense_structured_solve(array(namespace, [3.0, 8.0]))
    assert_allclose(namespace, actual, array(namespace, [2.0, 2.0]), **tol)


# --- saddle with an empty equality block (m == 0) ----------------------------


def _diagonal_condensed(namespace, *, delta_w=0.5):
    empty_sigma_s, empty_jac = _empty_ineq(namespace)
    return build_condensed_operator(
        Diagonal(array(namespace, [1.0, 3.0])),
        Diagonal(array(namespace, [0.5, 1.0])),
        empty_sigma_s,
        empty_jac,
        RegularizationState(delta_w=delta_w),
    )


def _empty_eq(namespace):
    dtype = array(namespace, [0.0]).dtype
    return Dense(namespace.zeros((0, 2), dtype=dtype))


def test_saddle_empty_equality_block_delegates_to_condensed(namespace, tol):
    condensed = _diagonal_condensed(namespace)
    saddle = build_saddle_operator(condensed, _empty_eq(namespace), 1e-8)

    assert_allclose(namespace, saddle.dense_matrix(), condensed.dense_matrix(), **tol)
    rhs = array(namespace, [2.0, 4.5])
    assert_allclose(
        namespace,
        saddle.dense_structured_solve(rhs),
        condensed.dense_structured_solve(rhs),
        **tol,
    )
    # COO assembly must skip the (2,2) block entirely, and the pattern
    # signature must not consult the (absent) equality Jacobian.
    _rows, _cols, values, shape = saddle.to_coo()
    assert shape == (2, 2)
    assert_allclose(namespace, saddle.coo_values(), values, **tol)
    assert saddle.coo_pattern_signature() is not None


def test_saddle_structured_solve_rejects_bad_rank(namespace):
    saddle = build_saddle_operator(
        _diagonal_condensed(namespace),
        Dense(array(namespace, [[1.0, 1.0]])),
        1e-8,
    )
    rhs3 = namespace.zeros((3, 1, 1), dtype=array(namespace, [0.0]).dtype)
    with pytest.raises(ValueError, match="vector/matrix"):
        saddle.dense_structured_solve(rhs3)


def test_saddle_structured_solve_without_dual_regularization(namespace, tol):
    # delta_c = 0: the Schur complement carries no shift.
    saddle = build_saddle_operator(
        _diagonal_condensed(namespace),
        Dense(array(namespace, [[1.0, 1.0]])),
        0.0,
    )
    rhs = array(namespace, [1.0, -2.0, 0.5])
    dense = saddle.matmat(namespace.eye(3, dtype=rhs.dtype))
    assert_allclose(
        namespace,
        saddle.dense_structured_solve(rhs),
        namespace.linalg.solve(dense, rhs),
        **tol,
    )


# --- values-only assembly over a generic condensed block ---------------------


def test_saddle_coo_values_generic_condensed_block(namespace, tol):
    # A condensed block without the structure/value split (here a plain
    # Diagonal) must still produce values in to_coo order.
    saddle = build_saddle_operator(
        Diagonal(array(namespace, [2.0, 3.0])),
        Dense(array(namespace, [[1.0, 4.0]])),
        0.25,
    )
    assert_allclose(namespace, saddle.coo_values(), saddle.to_coo()[2], **tol)


# --- pattern-signature propagation -------------------------------------------


def test_pattern_signature_none_propagates_from_unstructured_blocks(namespace):
    from ipax.backend.operators import MatrixFreeJacobian

    # Condensed: a matrix-free inequality Jacobian has no signature.
    sigma_s = Diagonal(array(namespace, [2.0]))
    mf_ineq = MatrixFreeJacobian((1, 2), matvec=lambda v: v[:1])
    condensed = build_condensed_operator(
        Diagonal(array(namespace, [1.0, 2.0])),
        Diagonal(array(namespace, [0.5, 0.5])),
        sigma_s,
        mf_ineq,
        RegularizationState(),
    )
    assert condensed.coo_pattern_signature() is None

    # Saddle: an unstructured condensed block propagates None ...
    mf_condensed = MatrixFreeJacobian((2, 2), matvec=lambda v: v)
    eq = Dense(array(namespace, [[1.0, 1.0]]))
    assert build_saddle_operator(mf_condensed, eq, 0.0).coo_pattern_signature() is None

    # ... and so does a matrix-free equality Jacobian.
    mf_eq = MatrixFreeJacobian((1, 2), matvec=lambda v: v[:1])
    structured = Diagonal(array(namespace, [1.0, 2.0]))
    assert build_saddle_operator(structured, mf_eq, 0.0).coo_pattern_signature() is None


# --- SPD preconditioner fallbacks --------------------------------------------


def test_saddle_approximate_schur_falls_back_to_identity_dual(namespace, tol):
    # A matrix-free equality Jacobian exposes no cheap row-Gram diagonal, so the
    # approximate-Schur dual block degrades to the SPD identity.
    from ipax.backend.operators import MatrixFreeJacobian

    condensed = _diagonal_condensed(namespace)
    mf_eq = MatrixFreeJacobian((1, 2), matvec=lambda v: v[:1])
    saddle = build_saddle_operator(condensed, mf_eq, 0.1)
    dual = saddle._approximate_schur_diagonal(condensed.diagonal())
    assert_allclose(namespace, dual, array(namespace, [1.0]), **tol)


# --- augmented dense route (Friedlander & Orban 2012 bordered system) -------


def test_condensed_logical_dense_block_excludes_inequality_gram(namespace, tol):
    """``logical_dense_block`` is W + Sigma_x + delta_w*I, with no Gram term."""
    W_dense = array(namespace, [[4.0, 0.5], [0.5, 3.0]])
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    jac = Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]]))
    op = build_condensed_operator(
        Dense(W_dense), sigma_x, sigma_s, jac, RegularizationState(delta_w=1e-6)
    )

    actual = op.logical_dense_block()
    expected = (
        W_dense
        + namespace.asarray([[0.25, 0.0], [0.0, 0.75]], dtype=W_dense.dtype)
        + 1e-6 * namespace.eye(2, dtype=W_dense.dtype)
    )
    assert_allclose(namespace, actual, expected, **tol)


def test_inequality_border_dense_matches_jacobian_and_sigma_s(namespace, tol):
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    jac_dense = array(namespace, [[1.0, 2.0], [-1.0, 0.5]])
    op = build_condensed_operator(
        Dense(array(namespace, [[4.0, 0.5], [0.5, 3.0]])),
        Diagonal(array(namespace, [0.25, 0.75])),
        sigma_s,
        Dense(jac_dense),
        RegularizationState(),
    )

    border = op.inequality_border_dense()
    assert border is not None
    jac, neg_inv_sigma_s = border
    assert_allclose(namespace, jac, jac_dense, **tol)
    assert_allclose(namespace, neg_inv_sigma_s, array(namespace, [-0.5, -2.0]), **tol)


def test_inequality_border_dense_none_without_inequalities(namespace):
    empty_sigma_s, empty_jac = _empty_ineq(namespace)
    op = build_condensed_operator(
        Dense(array(namespace, [[4.0, 0.5], [0.5, 3.0]])),
        Diagonal(array(namespace, [0.25, 0.75])),
        empty_sigma_s,
        empty_jac,
        RegularizationState(),
    )
    assert op.inequality_border_dense() is None


def test_condensed_augmented_dense_matrix_matches_bordered_formula(namespace, tol):
    W_dense = array(namespace, [[4.0, 0.5], [0.5, 3.0]])
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    jac_dense = array(namespace, [[1.0, 2.0], [-1.0, 0.5]])
    op = build_condensed_operator(
        Dense(W_dense),
        sigma_x,
        sigma_s,
        Dense(jac_dense),
        RegularizationState(delta_w=1e-6),
    )

    actual = op.augmented_dense_matrix()

    primal = (
        W_dense
        + namespace.asarray([[0.25, 0.0], [0.0, 0.75]], dtype=W_dense.dtype)
        + 1e-6 * namespace.eye(2, dtype=W_dense.dtype)
    )
    e_block = namespace.asarray([[-0.5, 0.0], [0.0, -2.0]], dtype=W_dense.dtype)
    top = namespace.concat((primal, transpose(namespace, jac_dense)), axis=1)
    bottom = namespace.concat((jac_dense, e_block), axis=1)
    expected = namespace.concat((top, bottom), axis=0)

    assert actual.shape == (4, 4)
    assert_allclose(namespace, actual, expected, **tol)


def test_condensed_augmented_dense_matrix_without_inequalities(namespace, tol):
    """No inequalities: the augmented route has no border to add."""
    empty_sigma_s, empty_jac = _empty_ineq(namespace)
    op = build_condensed_operator(
        Dense(array(namespace, [[4.0, 0.5], [0.5, 3.0]])),
        Diagonal(array(namespace, [0.25, 0.75])),
        empty_sigma_s,
        empty_jac,
        RegularizationState(delta_w=1e-6),
    )
    assert_allclose(namespace, op.augmented_dense_matrix(), op.dense_matrix(), **tol)


class _MatmatOnlyJacobian(LinearOperator):
    """A Jacobian exposing only matvec/matmat, no ``dense_matrix`` override.

    Mirrors ``ipax.problem.scaling._RowScaled`` (the operator a scaled
    problem's inequality Jacobian actually is by default, since scaling is
    on by default) — the augmented route must still materialize it.
    """

    def __init__(self, matrix):
        self._matrix = matrix

    @property
    def shape(self):
        return int(self._matrix.shape[0]), int(self._matrix.shape[1])

    def matvec(self, v):
        xp = array_namespace(self._matrix, v)
        return xp.matmul(self._matrix, v)

    def rmatvec(self, v):
        xp = array_namespace(self._matrix, v)
        return xp.matmul(transpose(xp, self._matrix), v)

    def matmat(self, V):
        xp = array_namespace(self._matrix, V)
        return xp.matmul(self._matrix, V)


def test_inequality_border_dense_falls_back_to_matmat_probe(namespace, tol):
    """A Jacobian without ``dense_matrix`` (e.g. scaling's ``_RowScaled``) must
    still materialize via a matmat identity probe, not silently fail."""
    jac_dense = array(namespace, [[1.0, 2.0], [-1.0, 0.5]])
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    op = build_condensed_operator(
        Dense(array(namespace, [[4.0, 0.5], [0.5, 3.0]])),
        Diagonal(array(namespace, [0.25, 0.75])),
        sigma_s,
        _MatmatOnlyJacobian(jac_dense),
        RegularizationState(),
    )

    border = op.inequality_border_dense()
    assert border is not None
    jac, neg_inv_sigma_s = border
    assert_allclose(namespace, jac, jac_dense, **tol)
    assert_allclose(namespace, neg_inv_sigma_s, array(namespace, [-0.5, -2.0]), **tol)


def test_condensed_augmented_dense_matrix_raises_for_low_rank_hessian(namespace):
    """L-BFGS (PD by Powell damping already) has nothing to gain: falls back."""
    W = _lbfgs_operator(namespace)
    sigma_x = Diagonal(array(namespace, [0.25, 0.75, 1.25]))
    empty_sigma_s, empty_jac = (
        Diagonal(array(namespace, [])),
        Dense(namespace.zeros((0, 3), dtype=array(namespace, [0.0]).dtype)),
    )
    op = build_condensed_operator(
        W, sigma_x, empty_sigma_s, empty_jac, RegularizationState(delta_w=1e-6)
    )
    with pytest.raises(NotImplementedError):
        op.augmented_dense_matrix()


def test_saddle_augmented_dense_matrix_matches_bordered_formula(namespace, tol):
    W_dense = array(namespace, [[4.0, 0.5], [0.5, 3.0]])
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    jac_dense = array(namespace, [[1.0, 2.0], [-1.0, 0.5]])
    condensed = build_condensed_operator(
        Dense(W_dense),
        sigma_x,
        sigma_s,
        Dense(jac_dense),
        RegularizationState(delta_w=1e-6),
    )
    eq_dense = array(namespace, [[1.0, 1.0]])
    saddle = build_saddle_operator(condensed, Dense(eq_dense), delta_c=1e-4)

    actual = saddle.augmented_dense_matrix()

    primal = (
        W_dense
        + namespace.asarray([[0.25, 0.0], [0.0, 0.75]], dtype=W_dense.dtype)
        + 1e-6 * namespace.eye(2, dtype=W_dense.dtype)
    )
    # Layout: [primal (n=2) | equality dual (m_eq=1) | inequality border (m_ineq=2)].
    dtype = W_dense.dtype
    core_top = namespace.concat((primal, transpose(namespace, eq_dense)), axis=1)
    core_bottom = namespace.concat(
        (eq_dense, -1e-4 * namespace.eye(1, dtype=dtype)), axis=1
    )
    core = namespace.concat((core_top, core_bottom), axis=0)  # (3, 3)

    conn = namespace.concat(
        (jac_dense, namespace.zeros((2, 1), dtype=dtype)), axis=1
    )  # (2, 3): touches only the primal columns
    e_block = namespace.asarray([[-0.5, 0.0], [0.0, -2.0]], dtype=dtype)
    top = namespace.concat((core, transpose(namespace, conn)), axis=1)
    bottom = namespace.concat((conn, e_block), axis=1)
    expected = namespace.concat((top, bottom), axis=0)

    assert actual.shape == (5, 5)
    assert_allclose(namespace, actual, expected, **tol)


def test_saddle_augmented_dense_matrix_without_inequalities_matches_dense_matrix(
    namespace, tol
):
    condensed = _diagonal_condensed(namespace)
    saddle = build_saddle_operator(
        condensed, Dense(array(namespace, [[1.0, 1.0]])), 1e-4
    )
    assert_allclose(
        namespace, saddle.augmented_dense_matrix(), saddle.dense_matrix(), **tol
    )


def test_saddle_augmented_dense_matrix_raises_when_condensed_lacks_support(namespace):
    """A condensed block without ``logical_dense_block`` (e.g. plain Dense) falls back."""
    saddle = build_saddle_operator(
        Dense(array(namespace, [[4.0, 0.5], [0.5, 3.0]])),
        Dense(array(namespace, [[1.0, 1.0]])),
        1e-8,
    )
    with pytest.raises(NotImplementedError):
        saddle.augmented_dense_matrix()


def test_saddle_block_preconditioner_without_equalities(namespace, tol):
    # m == 0: the block preconditioner reduces to the condensed Woodbury inverse.
    W = LBFGSOperator(2, LBFGSOptions(memory=3))
    W.update(array(namespace, [1.0, 0.0]), array(namespace, [2.0, 0.5]))
    dtype = array(namespace, [0.0]).dtype
    condensed = build_condensed_operator(
        W,
        Diagonal(array(namespace, [0.5, 0.5])),
        Diagonal(array(namespace, [])),
        Dense(namespace.zeros((0, 2), dtype=dtype)),
        RegularizationState(delta_w=1e-6),
    )
    saddle = build_saddle_operator(condensed, _empty_eq(namespace), 0.0)

    apply = saddle.lbfgs_block_preconditioner_apply()
    r = array(namespace, [1.0, -2.0])
    assert_allclose(namespace, apply(r), condensed.lbfgs_inverse_apply()(r), **tol)
