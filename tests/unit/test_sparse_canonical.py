"""Unit coverage for the compiled COO→canonical compressed-sparse map.

Driven on NumPy (the helper is parameterized by the array module; the CuPy
adapter reuses the same code path on device). Validates that the compiled map
reproduces a from-scratch ``coo_matrix`` canonicalization exactly while only
recomputing values, and that the int32/int64 index-width selection (optimization
#3) is correct.
"""

from __future__ import annotations

import numpy as np
import pytest

scipy_sparse = pytest.importorskip("scipy.sparse")

from ipax.backend.sparse._canonical import (  # noqa: E402
    compile_compressed,
    compile_lower_triangle,
    index_dtype,
)


def _scatter_add(out: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    np.add.at(out, idx, vals)


def _csc(rows, cols, values, shape):
    """Compile a canonical CSC map (major=col, minor=row) and apply ``values``."""
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    compiled = compile_compressed(
        np,
        _scatter_add,
        major=cols,
        minor=rows,
        n_major=shape[1],
        n_minor=shape[0],
    )
    data = compiled.data(np.asarray(values, dtype=np.float64))
    return scipy_sparse.csc_matrix(
        (data, compiled.indices, compiled.indptr), shape=shape
    )


def test_compiled_csc_matches_coo_canonicalization():
    rows = [0, 1, 1, 2, 0]
    cols = [0, 0, 1, 2, 0]  # note: (0,0) appears twice ⇒ duplicate to be summed
    values = [2.0, 1.0, 3.0, 4.0, 5.0]
    shape = (3, 3)

    reference = scipy_sparse.coo_matrix((values, (rows, cols)), shape=shape).tocsc()
    reference.sum_duplicates()

    compiled = _csc(rows, cols, values, shape)

    np.testing.assert_allclose(compiled.toarray(), reference.toarray())


def test_compiled_map_recomputes_values_only_for_fixed_pattern():
    # The pattern (rows/cols) is fixed; a second value vector must reuse the same
    # compiled structure and still match a from-scratch canonicalization.
    rows = np.asarray([0, 1, 1, 2, 0])
    cols = np.asarray([0, 0, 1, 2, 0])
    shape = (3, 3)
    compiled = compile_compressed(
        np, _scatter_add, major=cols, minor=rows, n_major=3, n_minor=3
    )

    for values in ([2.0, 1.0, 3.0, 4.0, 5.0], [-1.0, 7.0, 0.5, 9.0, 2.0]):
        data = compiled.data(np.asarray(values, dtype=np.float64))
        built = scipy_sparse.csc_matrix(
            (data, compiled.indices, compiled.indptr), shape=shape
        )
        reference = scipy_sparse.coo_matrix((values, (rows, cols)), shape=shape).tocsc()
        reference.sum_duplicates()
        np.testing.assert_allclose(built.toarray(), reference.toarray())


def test_compiled_lower_triangle_keep_filters_upper_entries():
    # The cuDSS symmetric path factors only the lower triangle; the ``keep`` mask
    # must drop row<col entries before compiling.
    rows = np.asarray([0, 1, 0, 1])
    cols = np.asarray([0, 0, 1, 1])  # (0,1) is strictly upper
    values = np.asarray([2.0, 1.0, 1.0, 3.0])
    keep = rows >= cols

    compiled = compile_compressed(
        np, _scatter_add, major=rows, minor=cols, n_major=2, n_minor=2, keep=keep
    )
    data = compiled.data(values)
    built = scipy_sparse.csr_matrix(
        (data, compiled.indices, compiled.indptr), shape=(2, 2)
    )

    reference = scipy_sparse.tril(
        scipy_sparse.coo_matrix((values, (rows, cols)), shape=(2, 2))
    ).tocsr()
    reference.sum_duplicates()
    np.testing.assert_allclose(built.toarray(), reference.toarray())


def test_compiled_lower_triangle_matches_scipy_tril():
    # Build a canonical symmetric full CSR, then check the compiled lower-triangle
    # gather reproduces scipy's tril(...).tocsr() for two value vectors.
    rows = np.asarray([0, 0, 1, 1, 2, 2])
    cols = np.asarray([0, 1, 0, 1, 1, 2])
    shape = (3, 3)
    full = scipy_sparse.coo_matrix(
        (np.ones(rows.shape[0]), (rows, cols)), shape=shape
    ).tocsr()
    full.sum_duplicates()
    full.sort_indices()

    lower = compile_lower_triangle(np, full.indptr, full.indices, 3)

    for data in (np.arange(1.0, full.nnz + 1.0), np.linspace(-2.0, 2.0, full.nnz)):
        built = scipy_sparse.csr_matrix(
            (lower.data(data), lower.indices, lower.indptr), shape=shape
        )
        reference = scipy_sparse.tril(
            scipy_sparse.csr_matrix((data, full.indices, full.indptr), shape=shape)
        ).tocsr()
        reference.sum_duplicates()
        np.testing.assert_allclose(built.toarray(), reference.toarray())


def test_index_dtype_prefers_int32_when_dimensions_fit():
    assert index_dtype(np, 10, 10, 4) == np.int32
    assert index_dtype(np, 2**31, 3, 3) == np.int64
    assert index_dtype(np, 3, 3, 2**31) == np.int64


def test_compiled_index_dtype_is_int32_for_small_systems():
    rows = np.asarray([0, 1, 2])
    cols = np.asarray([0, 1, 2])
    compiled = compile_compressed(
        np, _scatter_add, major=rows, minor=cols, n_major=3, n_minor=3
    )
    assert compiled.indptr.dtype == np.int32
    assert compiled.indices.dtype == np.int32


def test_compiled_empty_pattern_is_well_formed():
    empty = np.asarray([], dtype=np.int64)
    compiled = compile_compressed(
        np, _scatter_add, major=empty, minor=empty, n_major=0, n_minor=0
    )
    assert compiled.nnz == 0
    assert compiled.indptr.shape == (1,)
    assert compiled.data(np.asarray([], dtype=np.float64)).shape == (0,)
