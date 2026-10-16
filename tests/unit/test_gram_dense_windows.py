"""Bit-identity of the SciPy adapter's chunked dense Gram (``_gram_dense_accumulate``).

Row windows are densified into a reused chunk buffer (zeroed, then SciPy's own
``csr_todense`` kernel — exactly what ``toarray`` does) instead of a fresh
``csr_matrix(...).toarray()`` per chunk. The result must be *bitwise* identical
to the previous implementation, which is kept verbatim below as the oracle:
duplicates summed in the same order, ``-0.0`` data normalized the same way, the
same dtype promotion, the same syrk/GEMM split.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

scipy_sparse = pytest.importorskip("scipy.sparse")

from scipy.linalg.blas import get_blas_funcs  # noqa: E402

import ipax.backend.sparse.numpy_scipy as adapter  # noqa: E402
from ipax.backend.sparse.numpy_scipy import SparseOperator  # noqa: E402


def _old_gram_dense_accumulate(
    a: scipy_sparse.csr_matrix, w: np.ndarray, chunk_elements: int
) -> np.ndarray:
    """The pre-2026-09-28 ``_gram_dense_accumulate``, verbatim (the oracle)."""
    m, n = a.shape
    dtype = np.result_type(a.dtype, w.dtype)
    chunk = max(1, chunk_elements // max(int(n), 1))
    use_syrk = dtype in (np.float32, np.float64) and bool(
        np.all((w >= 0.0) & np.isfinite(w))
    )
    if use_syrk:
        syrk = get_blas_funcs("syrk", dtype=dtype)
        sqrt_w = np.sqrt(w.astype(dtype, copy=False))
        out = np.zeros((n, n), dtype=dtype, order="F")
    else:
        out = np.zeros((n, n), dtype=dtype)
    for start in range(0, m, chunk):
        end = min(m, start + chunk)
        lo, hi = int(a.indptr[start]), int(a.indptr[end])
        window = scipy_sparse.csr_matrix(
            (a.data[lo:hi], a.indices[lo:hi], a.indptr[start : end + 1] - lo),
            shape=(end - start, n),
        )
        block = np.asarray(window.toarray(), dtype=dtype)
        if use_syrk:
            block *= sqrt_w[start:end, None]
            out = syrk(1.0, block.T, beta=1.0, c=out, trans=0, lower=1, overwrite_c=1)
        else:
            out += block.T @ (w[start:end, None] * block)
    if use_syrk:
        out = np.ascontiguousarray(out)
        for j in range(n):
            out[j, j + 1 :] = out[j + 1 :, j]
    return out


def _random_csr(
    rng: np.random.Generator,
    m: int,
    n: int,
    density: float,
    dtype: type,
    *,
    duplicates: bool = False,
    empty_rows: bool = True,
    negative_zeros: bool = False,
) -> scipy_sparse.csr_matrix:
    mask = rng.random((m, n)) < density
    if empty_rows and m > 2:
        mask[rng.random(m) < 0.25] = False
        mask[0] = False
        mask[-1] = False
    rows, cols = np.nonzero(mask)
    vals = rng.standard_normal(rows.shape[0])
    if negative_zeros and vals.shape[0]:
        vals[:: max(1, vals.shape[0] // 7)] = -0.0
    if duplicates and rows.shape[0]:
        # Repeat a subset of coordinates and shuffle within rows: a
        # non-canonical CSR (duplicates, unsorted indices) that ``toarray`` sums.
        pick = rng.random(rows.shape[0]) < 0.3
        rows = np.concatenate([rows, rows[pick]])
        cols = np.concatenate([cols, cols[pick]])
        vals = np.concatenate([vals, rng.standard_normal(int(pick.sum()))])
        order = np.lexsort((rng.random(rows.shape[0]), rows))
        rows, cols, vals = rows[order], cols[order], vals[order]
    counts = np.bincount(rows, minlength=m)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    a = scipy_sparse.csr_matrix(
        (vals.astype(dtype), cols.astype(np.int32), indptr), shape=(m, n)
    )
    if duplicates and rows.shape[0]:
        assert not a.has_canonical_format
    return a


def _weights(rng: np.random.Generator, m: int, kind: str) -> np.ndarray:
    if kind == "positive":
        return rng.random(m) + 0.1
    if kind == "zeros":
        w = rng.random(m)
        w[rng.random(m) < 0.3] = 0.0
        return w
    if kind == "negative":  # mixed sign ⇒ GEMM path
        return rng.standard_normal(m)
    if kind == "nonfinite":  # ⇒ GEMM path
        w = rng.random(m)
        w[m // 2] = np.inf
        return w
    if kind == "float32":
        return (rng.random(m) + 0.1).astype(np.float32)
    raise AssertionError(kind)


def _assert_bitwise(new: np.ndarray, old: np.ndarray) -> None:
    assert new.dtype == old.dtype
    assert new.shape == old.shape
    assert new.flags.c_contiguous == old.flags.c_contiguous
    # Byte equality: stricter than array_equal (distinguishes ±0.0, NaN payloads).
    assert new.tobytes() == old.tobytes()


_CASES = [
    # (m, n, density, data dtype, weights kind, duplicates, negative zeros)
    (40, 7, 0.4, np.float64, "positive", False, False),
    (40, 7, 0.4, np.float64, "zeros", False, True),
    (40, 7, 0.4, np.float64, "negative", False, False),
    (40, 7, 0.4, np.float64, "nonfinite", False, False),
    (40, 7, 0.4, np.float64, "positive", True, False),
    (40, 7, 0.4, np.float64, "negative", True, True),
    (33, 5, 0.5, np.float32, "positive", False, False),  # f32 data, f64 w: cast
    (33, 5, 0.5, np.float32, "negative", True, False),
    (33, 5, 0.5, np.float32, "float32", True, True),  # f32 data, f32 w
    (1, 6, 0.9, np.float64, "positive", False, False),  # single row
    (25, 4, 0.0, np.float64, "positive", False, False),  # all rows empty
    (0, 4, 0.5, np.float64, "positive", False, False),  # no rows
]


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("chunk_elements", [1, 12, 30, 50_000_000])
@pytest.mark.parametrize(
    ("m", "n", "density", "dtype", "wkind", "dups", "negzero"), _CASES
)
def test_gram_dense_accumulate_bitwise_equal_to_previous(
    monkeypatch, chunk_elements, m, n, density, dtype, wkind, dups, negzero
):
    rng = np.random.default_rng(zlib.crc32(repr((m, n, wkind, dups, negzero)).encode()))
    a = _random_csr(rng, m, n, density, dtype, duplicates=dups, negative_zeros=negzero)
    w = _weights(rng, m, wkind)
    monkeypatch.setattr(adapter, "_GRAM_DENSE_CHUNK_ELEMENTS", chunk_elements)
    new = SparseOperator._gram_dense_accumulate(a, w)
    _assert_bitwise(new, _old_gram_dense_accumulate(a, w, chunk_elements))


@pytest.mark.parametrize("chunk_elements", [12, 50_000_000])
def test_gram_dense_accumulate_public_fallback_is_bitwise_equal(
    monkeypatch, chunk_elements
):
    # Without SciPy's private ``csr_todense`` the adapter densifies through the
    # public ``toarray(out=...)`` into the same reused buffer.
    rng = np.random.default_rng(7)
    a = _random_csr(rng, 30, 6, 0.4, np.float32, duplicates=True, negative_zeros=True)
    monkeypatch.setattr(adapter, "_GRAM_DENSE_CHUNK_ELEMENTS", chunk_elements)
    monkeypatch.setattr(adapter, "_csr_todense", None)
    for wkind in ("positive", "negative", "float32"):
        w = _weights(rng, 30, wkind)
        new = SparseOperator._gram_dense_accumulate(a, w)
        _assert_bitwise(new, _old_gram_dense_accumulate(a, w, chunk_elements))


def test_gram_dense_accumulate_int64_indices_bitwise_equal(monkeypatch):
    # Index arrays SciPy keeps as int64 (e.g. nnz ≥ 2³¹ in production; forced
    # here by assignment) take the same kernel.
    rng = np.random.default_rng(11)
    a = _random_csr(rng, 30, 6, 0.4, np.float64, duplicates=True)
    a.indptr = a.indptr.astype(np.int64)
    a.indices = a.indices.astype(np.int64)
    monkeypatch.setattr(adapter, "_GRAM_DENSE_CHUNK_ELEMENTS", 12)
    for wkind in ("positive", "negative"):
        w = _weights(rng, 30, wkind)
        new = SparseOperator._gram_dense_accumulate(a, w)
        _assert_bitwise(new, _old_gram_dense_accumulate(a, w, 12))


def test_public_gram_bitwise_equal_through_dense_strategy(monkeypatch):
    # End to end through ``SparseOperator.gram`` (native and mixed accumulate).
    import array_api_compat.numpy as xp_numpy

    rng = np.random.default_rng(3)
    a = _random_csr(rng, 50, 8, 0.4, np.float32, duplicates=True)
    monkeypatch.setattr(adapter, "_GRAM_DENSE_CHUNK_ELEMENTS", 40)
    op = SparseOperator(a, xp_numpy)
    csr = op._csr_matrix
    w = _weights(rng, 50, "positive")
    _assert_bitwise(np.asarray(op.gram(w)), _old_gram_dense_accumulate(csr, w, 40))
    w2 = _weights(rng, 50, "positive")
    got = np.asarray(op.gram(w2, accumulate_dtype="float32"))
    ref = _old_gram_dense_accumulate(
        op._reduced_data_csr(np.dtype(np.float32)), w2.astype(np.float32), 40
    ).astype(np.float64)
    _assert_bitwise(got, ref)
