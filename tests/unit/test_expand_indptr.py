"""``_expand_indptr``: compressed ``indptr`` → per-entry major index.

The ``repeat``-based expansion must reproduce the previous ``searchsorted`` form
exactly (values, shape and dtype) on every backend, including empty rows, empty
matrices and a single row; namespaces without a usable array-valued ``repeat``
fall back to ``searchsorted``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ipax.backend.namespace import device
from ipax.backend.operators import CSCOperator, CSROperator, _expand_indptr


def _reference(xp: Any, indptr: Any, nnz: int) -> Any:
    """The pre-2026-09-28 implementation, kept verbatim as the oracle."""
    arange = xp.arange(nnz, device=device(indptr))
    return xp.searchsorted(indptr, arange, side="right") - 1


def _patterns() -> list[tuple[str, list[int]]]:
    rng = np.random.default_rng(20260928)
    cases: list[tuple[str, list[int]]] = [
        ("empty_matrix", []),
        ("rows_all_empty", [0, 0, 0, 0]),
        ("single_row", [7]),
        ("single_empty_row", [0]),
        ("leading_trailing_empty", [0, 0, 3, 1, 0, 0, 2, 0]),
    ]
    for k in range(4):
        counts = rng.integers(0, 6, size=int(rng.integers(1, 60)))
        counts[rng.random(counts.shape[0]) < 0.3] = 0  # plenty of empty rows
        cases.append((f"random_{k}", [int(c) for c in counts]))
    return cases


def _indptr(xp: Any, counts: list[int], dtype: Any) -> Any:
    return xp.asarray([0, *np.cumsum(counts, dtype=np.int64).tolist()], dtype=dtype)


def _assert_identical(xp: Any, new: Any, old: Any) -> None:
    assert tuple(new.shape) == tuple(old.shape)
    assert new.dtype == old.dtype
    assert bool(xp.all(new == old))


@pytest.mark.parametrize(("name", "counts"), _patterns(), ids=lambda v: str(v))
@pytest.mark.parametrize("index_dtype", ["int64", "int32"])
def test_expand_indptr_matches_searchsorted(namespace, name, counts, index_dtype):
    del name
    xp = namespace
    indptr = _indptr(xp, counts, getattr(xp, index_dtype))
    nnz = sum(counts)
    _assert_identical(xp, _expand_indptr(xp, indptr, nnz), _reference(xp, indptr, nnz))


class _NoRepeat:
    """Namespace proxy without ``repeat`` (e.g. a pre-2023.12 namespace)."""

    def __init__(self, xp: Any) -> None:
        self._xp = xp

    def __getattr__(self, name: str) -> Any:
        if name == "repeat":
            raise AttributeError(name)
        return getattr(self._xp, name)


class _RejectingRepeat(_NoRepeat):
    """Proxy whose ``repeat`` rejects array counts (CuPy < 14, strict without
    data-dependent shapes)."""

    def __getattr__(self, name: str) -> Any:
        if name == "repeat":

            def repeat(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("array-valued repeats unsupported")

            return repeat
        return getattr(self._xp, name)


@pytest.mark.parametrize("proxy", [_NoRepeat, _RejectingRepeat])
def test_expand_indptr_falls_back_without_array_repeat(namespace, proxy):
    xp = namespace
    counts = [0, 2, 0, 3, 1, 0]
    indptr = _indptr(xp, counts, xp.int64)
    got = _expand_indptr(proxy(xp), indptr, sum(counts))
    _assert_identical(xp, got, _reference(xp, indptr, sum(counts)))


def test_expand_indptr_rejects_indptr_nnz_mismatch(namespace):
    xp = namespace
    indptr = xp.asarray([0, 2, 3])
    with pytest.raises(ValueError, match="indptr"):
        _expand_indptr(xp, indptr, 5)


def test_csr_csc_constructors_use_identical_expansion(namespace):
    xp = namespace
    counts = [2, 0, 3, 0]
    indptr = _indptr(xp, counts, xp.int64)
    indices = xp.asarray([0, 2, 1, 2, 3])
    data = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=xp.float64)
    rows_ref = _reference(xp, indptr, 5)
    r, c, _, _ = CSROperator(indptr, indices, data, (4, 4)).to_coo()
    _assert_identical(xp, r, rows_ref)
    r, c, _, _ = CSCOperator(indptr, indices, data, (4, 4)).to_coo()
    _assert_identical(xp, c, rows_ref)
