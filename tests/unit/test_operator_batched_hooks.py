"""Batched/transpose algebra hooks on the built-in operators.

``matmat``/``rmatmat``/``dense_matrix`` are the batched fast paths behind the
dense KKT route's single-materialization solve (each block is pushed through
once instead of column-by-column). Every hook must agree with the operator's
dense matrix so batching never changes semantics.
"""

from __future__ import annotations

from ipax.backend.operators import (
    Composite,
    Dense,
    Diagonal,
    Identity,
    LowRank,
    MatrixFreeJacobian,
    VStack,
)
from tests._helpers import array, assert_allclose, transpose


def test_diagonal_rmatmat_matches_transposed_dense(namespace, tol):
    op = Diagonal(array(namespace, [2.0, -0.5]))
    V = array(namespace, [[1.0, 2.0], [3.0, 4.0]])
    expected = namespace.matmul(transpose(namespace, op.dense_matrix()), V)
    assert_allclose(namespace, op.rmatmat(V), expected, **tol)


def test_identity_rmatmat_is_identity(namespace, tol):
    op = Identity(2)
    V = array(namespace, [[1.0, 2.0], [3.0, 4.0]])
    assert_allclose(namespace, op.rmatmat(V), V, **tol)


def test_identity_pattern_signature_is_structural():
    # Pure Python-tuple metadata: stable across instances of the same size,
    # distinct across sizes.
    assert Identity(3).coo_pattern_signature() == Identity(3).coo_pattern_signature()
    assert Identity(3).coo_pattern_signature() != Identity(4).coo_pattern_signature()


def test_lowrank_rmatmat_matches_transposed_dense(namespace, tol):
    U = array(namespace, [[1.0, 0.5], [2.0, -1.0], [0.0, 3.0]])
    W = array(namespace, [[2.0, 1.0], [0.5, 0.25], [1.0, -1.0]])
    op = LowRank(U, W)  # 3×3 = U @ W.T
    V = array(namespace, [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    expected = namespace.matmul(transpose(namespace, op.dense_matrix()), V)
    assert_allclose(namespace, op.rmatmat(V), expected, **tol)


def _composite(namespace):
    A = array(namespace, [[2.0, 1.0], [0.0, 3.0]])
    B = array(namespace, [[1.0, -1.0], [0.5, 2.0]])
    return Composite(Dense(A), Dense(B)), namespace.matmul(A, B)


def test_composite_matmat_matches_dense_product(namespace, tol):
    op, product = _composite(namespace)
    V = array(namespace, [[1.0, 2.0], [3.0, 4.0]])
    assert_allclose(namespace, op.matmat(V), namespace.matmul(product, V), **tol)


def test_composite_rmatmat_matches_transposed_product(namespace, tol):
    op, product = _composite(namespace)
    V = array(namespace, [[1.0, 2.0], [3.0, 4.0]])
    expected = namespace.matmul(transpose(namespace, product), V)
    assert_allclose(namespace, op.rmatmat(V), expected, **tol)


def test_composite_dense_matrix_multiplies_term_matrices(namespace, tol):
    op, product = _composite(namespace)
    assert_allclose(namespace, op.dense_matrix(), product, **tol)


def test_vstack_pattern_signature_none_when_any_block_unstructured(namespace):
    top = Dense(array(namespace, [[1.0, 2.0]]))
    bottom = MatrixFreeJacobian((1, 2), matvec=lambda v: v[:1])
    assert VStack((top, bottom)).coo_pattern_signature() is None


def test_vstack_gram_sums_block_grams(namespace, tol):
    # For J = [J1; J2], Jᵀ diag(w) J = J1ᵀ diag(w1) J1 + J2ᵀ diag(w2) J2 — the
    # block-wise Gram used by the condensed dense route on a stacked Jacobian.
    import pytest

    from ipax.backend.operators import COOOperator
    from ipax.backend.sparse import get_sparse_adapter

    if get_sparse_adapter(namespace) is None:
        pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")

    top = COOOperator(
        namespace.asarray([0, 0, 1]),
        namespace.asarray([0, 1, 2]),
        array(namespace, [2.0, -1.0, 3.0]),
        (2, 3),
    )
    bottom = COOOperator(
        namespace.asarray([0, 1]),
        namespace.asarray([1, 2]),
        array(namespace, [4.0, 5.0]),
        (2, 3),
    )
    stacked = VStack((top, bottom))
    weights = array(namespace, [2.0, 0.5, 1.0, 1.5])
    dense = stacked.dense_matrix()
    scaled = namespace.expand_dims(weights, axis=1) * dense
    expected = namespace.matmul(transpose(namespace, dense), scaled)
    assert_allclose(namespace, stacked.gram(weights), expected, **tol)
