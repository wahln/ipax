"""Public ``COOOperator`` / ``CSROperator`` / ``CSCOperator`` (backend-agnostic).

These carry Array-API triplets and emit structure for the sparse-direct route
(``to_coo``/``coo_values``/``coo_pattern_signature``) in pure Array API — so the
structure-side tests run on every backend including the ``array-api-strict``
purity gate. The matvec-family delegates to the per-backend sparse adapter and is
covered by the contract battery (``tests/contracts``), skipped where no adapter.
"""

from __future__ import annotations

import pytest

import ipax
from ipax.backend.operators import COOOperator, CSCOperator, CSROperator
from ipax.backend.sparse import get_sparse_adapter
from tests._helpers import array, assert_allclose, transpose


def _require_adapter(namespace):
    if get_sparse_adapter(namespace) is None:
        pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")


def _coo_to_dense(namespace, rows, cols, values, shape):
    m, n = shape
    grid = [[0.0] * n for _ in range(m)]
    for k in range(int(rows.shape[0])):
        grid[int(rows[k])][int(cols[k])] += float(values[k])
    return array(namespace, grid)


def test_operators_are_public_api():
    assert ipax.COOOperator is COOOperator
    assert ipax.CSROperator is CSROperator
    assert ipax.CSCOperator is CSCOperator


def test_coo_emits_structure_and_values(namespace, tol):
    rows = namespace.asarray([0, 0, 1, 1])
    cols = namespace.asarray([0, 1, 0, 1])
    values = array(namespace, [2.0, 1.0, 1.0, 3.0])
    op = COOOperator(rows, cols, values, (2, 2))

    _, _, v, shape = op.to_coo()
    assert shape == (2, 2)
    assert_allclose(namespace, v, values, **tol)
    assert_allclose(namespace, op.coo_values(), values, **tol)


def test_coo_pattern_signature_is_opt_in():
    # Signature is pure Python-tuple metadata, so a single backend suffices.
    import array_api_compat.numpy as xp

    rows = xp.asarray([0, 1])
    cols = xp.asarray([0, 1])
    values = xp.asarray([1.0, 2.0])

    # Default: conservative (no reuse).
    assert COOOperator(rows, cols, values, (2, 2)).coo_pattern_signature() is None
    # Opt-in: stable, and equal across instances sharing key + shape (+ symmetry).
    a = COOOperator(rows, cols, values, (2, 2), pattern_key="J")
    b = COOOperator(rows, cols, xp.asarray([3.0, 4.0]), (2, 2), pattern_key="J")
    assert a.coo_pattern_signature() is not None
    assert a.coo_pattern_signature() == b.coo_pattern_signature()
    # A different key (or shape) must not collide.
    other = COOOperator(rows, cols, values, (2, 2), pattern_key="K")
    assert other.coo_pattern_signature() != a.coo_pattern_signature()


def test_csr_expands_to_correct_coo_structure(namespace, tol):
    # CSR for [[2, -1, 0.5], [0, 3, 4]].
    indptr = namespace.asarray([0, 3, 5])
    indices = namespace.asarray([0, 1, 2, 1, 2])
    data = array(namespace, [2.0, -1.0, 0.5, 3.0, 4.0])
    op = CSROperator(indptr, indices, data, (2, 3))

    r, c, v, shape = op.to_coo()
    dense = _coo_to_dense(namespace, r, c, v, shape)
    assert_allclose(
        namespace, dense, array(namespace, [[2.0, -1.0, 0.5], [0.0, 3.0, 4.0]]), **tol
    )


def test_csc_expands_to_correct_coo_structure(namespace, tol):
    # CSC for [[2, -1, 0.5], [0, 3, 4]].
    indptr = namespace.asarray([0, 1, 3, 5])
    indices = namespace.asarray([0, 0, 1, 0, 1])
    data = array(namespace, [2.0, -1.0, 3.0, 0.5, 4.0])
    op = CSCOperator(indptr, indices, data, (2, 3))

    r, c, v, shape = op.to_coo()
    dense = _coo_to_dense(namespace, r, c, v, shape)
    assert_allclose(
        namespace, dense, array(namespace, [[2.0, -1.0, 0.5], [0.0, 3.0, 4.0]]), **tol
    )


def test_coo_values_matches_to_coo_values(namespace, tol):
    indptr = namespace.asarray([0, 1, 3, 5])
    indices = namespace.asarray([0, 0, 1, 0, 1])
    data = array(namespace, [2.0, -1.0, 3.0, 0.5, 4.0])
    op = CSCOperator(indptr, indices, data, (2, 3))
    assert_allclose(namespace, op.coo_values(), op.to_coo()[2], **tol)


def test_symmetry_hint_passthrough(namespace):
    rows = namespace.asarray([0, 1])
    cols = namespace.asarray([0, 1])
    values = array(namespace, [1.0, 2.0])
    assert COOOperator(rows, cols, values, (2, 2)).symmetry_hint() is None
    assert (
        COOOperator(rows, cols, values, (2, 2), symmetric=True).symmetry_hint() is True
    )


def test_coo_rejects_mismatched_lengths(namespace):
    rows = namespace.asarray([0, 1])
    cols = namespace.asarray([0])
    values = array(namespace, [1.0, 2.0])
    with pytest.raises(ValueError, match="equal length"):
        COOOperator(rows, cols, values, (2, 2))


def test_csr_rejects_wrong_indptr_length(namespace):
    indptr = namespace.asarray([0, 3])  # length 2, needs shape[0] + 1 = 3
    indices = namespace.asarray([0, 1, 2, 1, 2])
    data = array(namespace, [2.0, -1.0, 0.5, 3.0, 4.0])
    with pytest.raises(ValueError, match="indptr"):
        CSROperator(indptr, indices, data, (2, 3))


def test_csc_rejects_wrong_indptr_length(namespace):
    indptr = namespace.asarray([0, 1, 3])  # length 3, needs shape[1] + 1 = 4
    indices = namespace.asarray([0, 0, 1, 0, 1])
    data = array(namespace, [2.0, -1.0, 3.0, 0.5, 4.0])
    with pytest.raises(ValueError, match="indptr"):
        CSCOperator(indptr, indices, data, (2, 3))


# A = [[2, -1, 0.5], [0, 3, 4]] as COO triplets (delegated algebra via adapter).
def _coo_AB(namespace):
    return COOOperator(
        namespace.asarray([0, 0, 0, 1, 1]),
        namespace.asarray([0, 1, 2, 1, 2]),
        array(namespace, [2.0, -1.0, 0.5, 3.0, 4.0]),
        (2, 3),
    )


def test_gram_diagonal_matches_dense(namespace, tol):
    # diag(Aᵀ diag(w) A)_k = Σ_i w_i A_ik²  (column energies, length n).
    _require_adapter(namespace)
    op = _coo_AB(namespace)
    weights = array(namespace, [1.0, 1.0])
    expected = array(namespace, [4.0, 10.0, 16.25])  # [2²; 1²+3²; 0.5²+4²]
    assert_allclose(namespace, op.gram_diagonal(weights), expected, **tol)


def test_row_gram_diagonal_matches_dense(namespace, tol):
    # diag(A diag(w) Aᵀ)_j = Σ_k w_k A_jk²  (row energies, length m).
    _require_adapter(namespace)
    op = _coo_AB(namespace)
    weights = array(namespace, [1.0, 1.0, 1.0])
    expected = array(namespace, [5.25, 25.0])  # [2²+1²+0.5²; 3²+4²]
    assert_allclose(namespace, op.row_gram_diagonal(weights), expected, **tol)


def test_rmatmat_matches_dense(namespace, tol):
    _require_adapter(namespace)
    op = _coo_AB(namespace)
    dense = array(namespace, [[2.0, -1.0, 0.5], [0.0, 3.0, 4.0]])
    V = array(namespace, [[1.0, 0.0], [0.0, 1.0]])
    expected = namespace.matmul(transpose(namespace, dense), V)
    assert_allclose(namespace, op.rmatmat(V), expected, **tol)
