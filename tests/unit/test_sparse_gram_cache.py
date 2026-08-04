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


def test_gram_dense_strategy_exactly_symmetric(monkeypatch):
    # The condensed matrix N = Aᵀ diag(w) A is symmetric by construction and
    # the downstream Cholesky assumes it. The syrk accumulation (nonnegative
    # weights) computes one triangle and mirrors it, so the result is
    # *bitwise* symmetric — a general GEMM accumulation only gets there to
    # rounding error.
    import ipax.backend.sparse.numpy_scipy as adapter

    monkeypatch.setattr(adapter, "_GRAM_DENSE_MIN_DENSITY", 0.0)
    monkeypatch.setattr(adapter, "_GRAM_DENSE_CHUNK_ELEMENTS", 64)

    rng = np.random.default_rng(7)
    matrix = scipy_sparse.csr_matrix(rng.standard_normal((41, 7)))
    weights = rng.uniform(1e-8, 1e3, size=41)

    op = _operator(matrix)
    actual = np.asarray(op.gram(xp_numpy.asarray(weights)))

    np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_allclose(
        actual, _reference_gram(op.scipy_matrix, weights), rtol=1e-12, atol=1e-12
    )


def _spy_blas_dispatch(monkeypatch):
    """Count ``get_blas_funcs`` calls to pin which accumulation branch ran."""
    import ipax.backend.sparse.numpy_scipy as adapter

    calls: list[str] = []
    real = adapter.get_blas_funcs

    def spy(name, *args, **kwargs):
        calls.append(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(adapter, "get_blas_funcs", spy)
    return calls


def test_gram_dense_strategy_zero_weights_on_syrk_path(monkeypatch):
    # w == 0 entries sit on the boundary of the ``w >= 0`` syrk predicate;
    # they must stay on the syrk branch and contribute exactly nothing.
    import ipax.backend.sparse.numpy_scipy as adapter

    monkeypatch.setattr(adapter, "_GRAM_DENSE_MIN_DENSITY", 0.0)
    monkeypatch.setattr(adapter, "_GRAM_DENSE_CHUNK_ELEMENTS", 6)
    calls = _spy_blas_dispatch(monkeypatch)

    op = _operator()
    weights = np.asarray([0.5, 0.0, 1.0, 0.0, 3.0, 0.75])
    actual = np.asarray(op.gram(xp_numpy.asarray(weights)))

    assert calls == ["syrk"]
    np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_allclose(
        actual, _reference_gram(op.scipy_matrix, weights), rtol=1e-13, atol=1e-13
    )


def test_gram_dense_strategy_nonfinite_weights_fall_back(monkeypatch):
    # An inf weight passes ``w >= 0`` but must not take the √w/syrk path
    # (different overflow envelope); the finiteness gate routes it to GEMM.
    import ipax.backend.sparse.numpy_scipy as adapter

    monkeypatch.setattr(adapter, "_GRAM_DENSE_MIN_DENSITY", 0.0)
    monkeypatch.setattr(adapter, "_GRAM_DENSE_CHUNK_ELEMENTS", 6)
    calls = _spy_blas_dispatch(monkeypatch)

    op = _operator()
    weights = np.asarray([0.5, np.inf, 1.0, 0.25, 3.0, 0.75])
    actual = np.asarray(op.gram(xp_numpy.asarray(weights)))

    assert calls == []  # GEMM fallback — syrk never requested
    expected = _reference_gram(op.scipy_matrix, weights)
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-13)


def test_gram_dense_strategy_mixed_dtype_promotes(monkeypatch):
    # float32 matrix × float64 weights: result_type promotes to float64 and
    # the chunk densification upcasts before the in-place √w scaling.
    import ipax.backend.sparse.numpy_scipy as adapter

    monkeypatch.setattr(adapter, "_GRAM_DENSE_MIN_DENSITY", 0.0)
    monkeypatch.setattr(adapter, "_GRAM_DENSE_CHUNK_ELEMENTS", 64)
    calls = _spy_blas_dispatch(monkeypatch)

    rng = np.random.default_rng(3)
    matrix = scipy_sparse.csr_matrix(rng.standard_normal((17, 4)).astype(np.float32))
    weights = rng.uniform(0.1, 10.0, size=17)  # float64

    op = _operator(matrix)
    actual = np.asarray(op.gram(xp_numpy.asarray(weights)))

    assert calls == ["syrk"]
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_allclose(
        actual,
        _reference_gram(op.scipy_matrix, weights),
        rtol=1e-6,  # float32 data limits the achievable agreement
        atol=1e-6,
    )


def test_gram_accumulate_dtype_float32_reduced_but_close(monkeypatch):
    # ``accumulate_dtype="float32"`` runs the dense accumulation in float32
    # (cached reduced-data CSR, fp32 syrk) and upcasts the n×n result: the
    # values must be float64-typed, close to the exact Gram at fp32 rounding,
    # and *different* from it (proof the reduced path really ran).
    import ipax.backend.sparse.numpy_scipy as adapter

    monkeypatch.setattr(adapter, "_GRAM_DENSE_MIN_DENSITY", 0.0)
    monkeypatch.setattr(adapter, "_GRAM_DENSE_CHUNK_ELEMENTS", 64)
    calls = _spy_blas_dispatch(monkeypatch)

    rng = np.random.default_rng(21)
    matrix = scipy_sparse.csr_matrix(rng.standard_normal((37, 6)) * 1e3)
    weights = rng.uniform(1e-3, 1e2, size=37)

    op = _operator(matrix)
    exact = np.asarray(op.gram(xp_numpy.asarray(weights)))
    reduced = np.asarray(op.gram(xp_numpy.asarray(weights), accumulate_dtype="float32"))

    assert calls == ["syrk", "syrk"]  # both strategies stayed on the syrk path
    assert reduced.dtype == np.float64
    rel = np.max(np.abs(reduced - exact)) / np.max(np.abs(exact))
    assert 1e-12 < rel < 1e-4
    np.testing.assert_array_equal(reduced, reduced.T)


def test_gram_memo_is_keyed_by_accumulate_dtype(monkeypatch):
    # A reduced-precision request must not serve (or poison) the native memo —
    # and the two dtypes get *separate* slots, so the mixed route's legitimate
    # exact/reduced alternation with identical weights (a mixed PD failure
    # re-materializing exactly inside a δ_w retry) never thrashes the memo.
    import ipax.backend.sparse.numpy_scipy as adapter

    monkeypatch.setattr(adapter, "_GRAM_DENSE_MIN_DENSITY", 0.0)

    op = _operator()
    weights = np.asarray([0.5, 2.0, 1.0, 0.25, 3.0, 0.75])
    op.gram(xp_numpy.asarray(weights))
    assert op._gram_compute_count == 1
    op.gram(xp_numpy.asarray(weights), accumulate_dtype="float32")
    assert op._gram_compute_count == 2  # recomputed, not served from memo
    op.gram(xp_numpy.asarray(weights), accumulate_dtype="float32")
    assert op._gram_compute_count == 2  # reduced request memo-hits now
    op.gram(xp_numpy.asarray(weights))
    assert op._gram_compute_count == 2  # native slot survived the alternation


def test_reduced_data_csr_is_released_on_native_fallback(monkeypatch):
    # After the mixed route permanently falls back, requests are all-native:
    # the fp32 data copy (4 bytes/nnz — multi-GB at RT scale) must be freed on
    # the first native request rather than pinned for the rest of the run.
    import ipax.backend.sparse.numpy_scipy as adapter

    monkeypatch.setattr(adapter, "_GRAM_DENSE_MIN_DENSITY", 0.0)

    op = _operator()
    weights = np.asarray([0.5, 2.0, 1.0, 0.25, 3.0, 0.75])
    op.gram(xp_numpy.asarray(weights), accumulate_dtype="float32")
    assert op._reduced_csr is not None
    op.gram(xp_numpy.asarray(weights))
    assert op._reduced_csr is None


def test_gram_dense_strategy_negative_weights_match_reference(monkeypatch):
    # Negative weights make √w-scaling impossible, so the accumulation must
    # fall back to the general GEMM form — values still match the reference.
    import ipax.backend.sparse.numpy_scipy as adapter

    monkeypatch.setattr(adapter, "_GRAM_DENSE_MIN_DENSITY", 0.0)
    monkeypatch.setattr(adapter, "_GRAM_DENSE_CHUNK_ELEMENTS", 6)

    op = _operator()
    weights = np.asarray([0.5, -2.0, 1.0, 0.25, -3.0, 0.75])
    actual = np.asarray(op.gram(xp_numpy.asarray(weights)))

    np.testing.assert_allclose(
        actual, _reference_gram(op.scipy_matrix, weights), rtol=1e-13, atol=1e-13
    )


def test_gram_dense_strategy_float32(monkeypatch):
    # float32 data (TROTS dose matrices are float32 in the files) routes to
    # the matching-precision BLAS and keeps the dtype.
    import ipax.backend.sparse.numpy_scipy as adapter

    monkeypatch.setattr(adapter, "_GRAM_DENSE_MIN_DENSITY", 0.0)
    monkeypatch.setattr(adapter, "_GRAM_DENSE_CHUNK_ELEMENTS", 64)

    rng = np.random.default_rng(11)
    matrix = scipy_sparse.csr_matrix(rng.standard_normal((23, 4)).astype(np.float32))
    weights = rng.uniform(0.1, 10.0, size=23).astype(np.float32)

    op = _operator(matrix)
    actual = np.asarray(op.gram(xp_numpy.asarray(weights)))

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_allclose(
        actual,
        _reference_gram(op.scipy_matrix, weights.astype(np.float64)),
        rtol=5e-6,
        atol=5e-6,
    )


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
