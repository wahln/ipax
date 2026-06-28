"""Unit tests for the optional ``diagonal`` / ``gram_diagonal`` capabilities.

These feed the matrix-free Jacobi preconditioner: the diagonal of the
condensed Newton matrix is assembled from per-operator diagonals without ever
forming an ``n×n`` matrix. Each assertion is cross-checked against the densely
materialized operator.
"""

from __future__ import annotations

import pytest

from ipax.backend.operators import (
    Dense,
    Diagonal,
    Identity,
    LowRank,
    MatrixFreeJacobian,
    VStack,
)
from tests._helpers import array, assert_allclose, transpose


def test_dense_diagonal_matches_materialized(namespace, tol):
    A = array(namespace, [[2.0, -1.0, 0.5], [0.0, 3.0, 4.0], [1.0, 2.0, -5.0]])
    op = Dense(A)
    assert_allclose(namespace, op.diagonal(), namespace.linalg.diagonal(A), **tol)


def test_diagonal_operator_diagonal(namespace, tol):
    d = array(namespace, [2.0, -3.0, 4.0])
    assert_allclose(namespace, Diagonal(d).diagonal(), d, **tol)


def test_dense_gram_diagonal_matches_formula(namespace, tol):
    A = array(namespace, [[1.0, 2.0, 0.0], [-1.0, 0.5, 3.0]])
    w = array(namespace, [2.0, 0.5])
    op = Dense(A)

    # Independent check via the full matrix product Aᵀ diag(w) A.
    wa = namespace.expand_dims(w, axis=1) * A
    gram = namespace.matmul(transpose(namespace, A), wa)
    expected = namespace.linalg.diagonal(gram)
    assert_allclose(namespace, op.gram_diagonal(w), expected, **tol)


def test_diagonal_gram_diagonal(namespace, tol):
    d = array(namespace, [2.0, -3.0, 4.0])
    w = array(namespace, [0.5, 1.0, 2.0])
    # diag(diag(d)ᵀ diag(w) diag(d)) = d² · w.
    expected = d * d * w
    assert_allclose(namespace, Diagonal(d).gram_diagonal(w), expected, **tol)


def test_dense_row_gram_diagonal_matches_formula(namespace, tol):
    A = array(namespace, [[1.0, 2.0, 0.0], [-1.0, 0.5, 3.0]])
    w = array(namespace, [2.0, 0.5, 1.0])  # length = n columns
    op = Dense(A)

    # Independent check via the full matrix product A diag(w) Aᵀ (row energies).
    aw = A * namespace.expand_dims(w, axis=0)
    gram = namespace.matmul(aw, transpose(namespace, A))
    expected = namespace.linalg.diagonal(gram)
    assert_allclose(namespace, op.row_gram_diagonal(w), expected, **tol)


def test_identity_gram_diagonal_returns_weights(namespace, tol):
    w = array(namespace, [0.5, 1.0, 2.0])
    assert_allclose(namespace, Identity(3).gram_diagonal(w), w, **tol)


def test_vstack_gram_diagonal_slices_row_weights(namespace, tol):
    top = Dense(array(namespace, [[1.0, 2.0, 0.0], [-1.0, 0.5, 3.0]]))
    bottom = Dense(array(namespace, [[0.0, -2.0, 1.0]]))
    op = VStack((top, bottom))
    weights = array(namespace, [2.0, 0.5, 3.0])
    dense = op.dense_matrix()

    weighted = namespace.expand_dims(weights, axis=1) * dense
    gram = namespace.matmul(transpose(namespace, dense), weighted)
    expected = namespace.linalg.diagonal(gram)

    assert_allclose(namespace, op.gram_diagonal(weights), expected, **tol)


def test_identity_diagonal_uses_template_array(namespace, tol):
    like = array(namespace, [0.0, 0.0, 0.0])
    assert_allclose(
        namespace, Identity(3).diagonal(like), namespace.ones_like(like), **tol
    )


def test_missing_diagonal_raises_not_implemented(namespace):
    op = MatrixFreeJacobian((2, 2), matvec=lambda v: v)
    with pytest.raises(NotImplementedError):
        op.diagonal()
    with pytest.raises(NotImplementedError):
        op.gram_diagonal(array(namespace, [1.0, 1.0]))


def test_lowrank_has_no_cheap_diagonal(namespace):
    U = array(namespace, [[1.0, 2.0], [0.0, -1.0]])
    V = array(namespace, [[-1.0, 0.0], [2.0, 1.0]])
    with pytest.raises(NotImplementedError):
        LowRank(U, V).diagonal()
