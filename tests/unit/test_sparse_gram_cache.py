"""Per-operator Gram caching in the SciPy sparse adapter (n ≪ m fast path).

The condensed dense route calls ``ineq_jac.gram(Σ_s)`` on every ``factor()`` —
once per IPM iteration *and* once per δ_w retry / SOC / Mehrotra re-solve within
an iteration, where the weights are bit-identical. The adapter therefore:

- memoizes the last ``(weights, result)`` pair (an O(m) value compare vs the
  Σ nnz_row² SpGEMM floor),
- caches the ``Aᵀ`` CSR transpose and a same-pattern scaled buffer so cache
  misses skip the per-call CSC→CSR conversion and ``multiply`` allocation.

These tests reach into adapter internals on purpose (compute counters); the
backend-agnostic correctness of ``gram`` itself stays covered by
``test_public_sparse_operators.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

scipy_sparse = pytest.importorskip("scipy.sparse")

import array_api_compat.numpy as xp_numpy  # noqa: E402

from ipax.backend.sparse.numpy_scipy import SparseOperator  # noqa: E402


def _operator(matrix=None) -> SparseOperator:
    if matrix is None:
        # 6×3 tall matrix with an empty row and a duplicate-free CSR pattern.
        matrix = scipy_sparse.csr_matrix(
            np.asarray(
                [
                    [2.0, 0.0, 1.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, -1.0, 0.5],
                    [0.0, 0.0, 4.0],
                    [-2.0, 1.0, 0.0],
                ]
            )
        )
    return SparseOperator(matrix, xp_numpy)


def _reference_gram(matrix, weights: np.ndarray) -> np.ndarray:
    dense = matrix.toarray()
    return dense.T @ (weights[:, None] * dense)


def test_gram_matches_reference_through_cached_path():
    op = _operator()
    weights = np.asarray([0.5, 2.0, 1.0, 0.0, 3.0, 0.25])

    actual = np.asarray(op.gram(xp_numpy.asarray(weights)))

    np.testing.assert_allclose(
        actual, _reference_gram(op.scipy_matrix, weights), rtol=1e-14, atol=1e-14
    )


def test_gram_memo_hits_for_equal_weights():
    op = _operator()
    w1 = xp_numpy.asarray([0.5, 2.0, 1.0, 0.25, 3.0, 0.75])
    w2 = xp_numpy.asarray(np.array(w1, copy=True))  # equal values, distinct object

    first = np.asarray(op.gram(w1))
    second = np.asarray(op.gram(w2))

    assert op._gram_compute_count == 1
    np.testing.assert_array_equal(first, second)


def test_gram_recomputes_when_weights_change():
    op = _operator()
    w1 = np.asarray([0.5, 2.0, 1.0, 0.25, 3.0, 0.75])
    w2 = 2.0 * w1

    first = np.asarray(op.gram(xp_numpy.asarray(w1)))
    second = np.asarray(op.gram(xp_numpy.asarray(w2)))

    assert op._gram_compute_count == 2
    np.testing.assert_allclose(
        first, _reference_gram(op.scipy_matrix, w1), rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        second, _reference_gram(op.scipy_matrix, w2), rtol=1e-14, atol=1e-14
    )


def test_gram_memo_survives_interleaved_calls():
    # retry → SOC → Mehrotra pattern: same weights re-requested after a change.
    op = _operator()
    w1 = np.asarray([0.5, 2.0, 1.0, 0.25, 3.0, 0.75])
    w2 = 3.0 * w1

    op.gram(xp_numpy.asarray(w1))
    op.gram(xp_numpy.asarray(w1))
    op.gram(xp_numpy.asarray(w2))
    result = np.asarray(op.gram(xp_numpy.asarray(w2)))

    assert op._gram_compute_count == 2
    np.testing.assert_allclose(
        result, _reference_gram(op.scipy_matrix, w2), rtol=1e-14, atol=1e-14
    )


def test_gram_diagonal_reuses_squared_matrix():
    op = _operator()
    weights = xp_numpy.asarray([0.5, 2.0, 1.0, 0.25, 3.0, 0.75])
    row_weights = xp_numpy.asarray([1.0, 2.0, 3.0])

    diag_first = np.asarray(op.gram_diagonal(weights))
    squared = op._squared
    assert squared is not None
    diag_second = np.asarray(op.gram_diagonal(weights))
    row_diag = np.asarray(op.row_gram_diagonal(row_weights))

    assert op._squared is squared  # built once, shared by both diagonals
    np.testing.assert_array_equal(diag_first, diag_second)
    dense = op.scipy_matrix.toarray()
    np.testing.assert_allclose(
        diag_first,
        ((dense**2) * np.asarray(weights)[:, None]).sum(axis=0),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        row_diag, (dense**2) @ np.asarray(row_weights), rtol=1e-14
    )


def test_gram_handles_empty_matrix():
    matrix = scipy_sparse.csr_matrix((0, 3))
    op = SparseOperator(matrix, xp_numpy)
    weights = xp_numpy.asarray(np.zeros((0,)))

    result = np.asarray(op.gram(weights))

    np.testing.assert_array_equal(result, np.zeros((3, 3)))


# --- gram_capable(): structural capability probe (solver auto-selection) ----


def test_gram_capable_reflects_override():
    from ipax.backend.operators import CSROperator, Dense, VStack

    indptr = xp_numpy.asarray([0, 2, 3])
    indices = xp_numpy.asarray([0, 2, 1])
    data = xp_numpy.asarray([2.0, 1.0, 3.0])
    csr = CSROperator(indptr, indices, data, (2, 3))
    dense = Dense(xp_numpy.zeros((2, 3)))

    assert csr.gram_capable()  # scipy adapter implements gram
    assert not dense.gram_capable()  # base fallback would densify
    assert VStack((csr, csr)).gram_capable()
    assert not VStack((csr, dense)).gram_capable()


def test_gram_capable_forwards_through_row_scaling():
    from ipax.backend.operators import CSROperator, Dense
    from ipax.problem.scaling import _RowScaled

    indptr = xp_numpy.asarray([0, 2, 3])
    indices = xp_numpy.asarray([0, 2, 1])
    data = xp_numpy.asarray([2.0, 1.0, 3.0])
    csr = CSROperator(indptr, indices, data, (2, 3))
    d = xp_numpy.asarray([2.0, 0.5])

    assert _RowScaled(csr, d).gram_capable()
    assert not _RowScaled(Dense(xp_numpy.zeros((2, 3))), d).gram_capable()


# --- dense-accumulation strategy for dense-ish matrices ----------------------


def test_gram_dense_strategy_matches_reference(monkeypatch):
    # Above the density threshold the Gram is accumulated by chunked dense
    # GEMM (the SpGEMM Σ nnz_row² floor is the wrong algorithm for dense-ish
    # rows, e.g. TROTS dose matrices at ~30%); force multiple chunks to
    # exercise the accumulation loop.
    import ipax.backend.sparse.numpy_scipy as adapter

    monkeypatch.setattr(adapter, "_GRAM_DENSE_MIN_DENSITY", 0.0)
    monkeypatch.setattr(adapter, "_GRAM_DENSE_CHUNK_ELEMENTS", 6)  # 2 rows/chunk

    op = _operator()
    weights = np.asarray([0.5, 2.0, 1.0, 0.25, 3.0, 0.75])
    actual = np.asarray(op.gram(xp_numpy.asarray(weights)))

    np.testing.assert_allclose(
        actual, _reference_gram(op.scipy_matrix, weights), rtol=1e-13, atol=1e-13
    )
    assert op._gram_compute_count == 1
    # The memo works across strategies too.
    op.gram(xp_numpy.asarray(np.array(weights, copy=True)))
    assert op._gram_compute_count == 1


def test_gram_sparse_strategy_below_density_threshold(monkeypatch):
    import ipax.backend.sparse.numpy_scipy as adapter

    monkeypatch.setattr(adapter, "_GRAM_DENSE_MIN_DENSITY", 1.1)  # never dense
    op = _operator()
    weights = np.asarray([0.5, 2.0, 1.0, 0.25, 3.0, 0.75])
    actual = np.asarray(op.gram(xp_numpy.asarray(weights)))
    np.testing.assert_allclose(
        actual, _reference_gram(op.scipy_matrix, weights), rtol=1e-13, atol=1e-13
    )
    assert op._gram_transpose is not None  # sparse path caches the transpose
