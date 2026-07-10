# Copyright 2026 Niklas Wahl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compiled COO→canonical compressed-sparse maps (ADAPTER — concrete xp allowed).

Shared by the SciPy (CPU) and CuPy (CUDA) sparse adapters. In an interior-point
solve the KKT sparsity pattern is **fixed across iterations** and only the
numeric values move (Σ_x, Σ_s, δ_w escalation, Jacobian entries). The
COO→CSR/CSC canonicalization (sort column indices + sum duplicates) is therefore
purely *structural* and identical every iteration — yet a fresh
``coo_matrix(...).tocsc()`` re-sorts ``O(nnz log nnz)`` every time, which on a
GPU is a wasted sort plus a segment scan that the factorization's symbolic-reuse
fast path never amortizes.

This module compiles that transform once per fixed pattern into a gather /
segment-sum map: the sorted-unique ``(major, minor)`` structure
(``indptr``/``indices``) plus the ``inverse`` index that scatters each emitted
COO triplet into its canonical slot. Re-applying the map to a new value vector is
a single ``O(nnz)`` scatter-add with **no sort** — the IPOPT structure/values
split, lifted up from the factorization into the assembly that feeds it.

The helper is parameterized by the concrete array module ``xp`` (NumPy or CuPy —
this is adapter territory, where invariants #1/#4 allow a concrete import) and a
``scatter_add`` callable, so one implementation serves both adapters. ``major``
is the compressed axis: rows for CSR, columns for CSC.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ipax.typing import Array

# int32 indices are valid (and halve index bandwidth, speeding up GPU analysis
# and triangular solves — optimization #3) whenever every dimension and the
# triplet count fit a signed 32-bit integer; otherwise fall back to int64.
_INT32_MAX = 2**31 - 1


def index_dtype(xp: Any, *dims: int) -> Any:
    """Return the narrowest safe index dtype (int32 when all ``dims`` fit)."""
    return xp.int32 if all(d <= _INT32_MAX for d in dims) else xp.int64


class CompiledCompressed:
    """A fixed COO→compressed-sparse map; recompute *values* only, never resort.

    Holds the canonical structure (``indptr``, ``indices``) for one sparsity
    pattern and the ``inverse`` scatter index from emitted COO order into the
    canonical slots. :meth:`data` turns a new value vector into the canonical
    ``data`` array with a single scatter-add.
    """

    __slots__ = ("_inverse", "_keep", "_scatter_add", "_xp", "indices", "indptr", "nnz")

    def __init__(
        self,
        xp: Any,
        scatter_add: Callable[[Any, Any, Any], None],
        *,
        indptr: Array,
        indices: Array,
        inverse: Array,
        nnz: int,
        keep: Array | None,
    ) -> None:
        self._xp = xp
        self._scatter_add = scatter_add
        self.indptr = indptr
        self.indices = indices
        self.nnz = nnz
        self._inverse = inverse
        # Boolean mask of which emitted triplets survive a triangle filter (the
        # cuDSS symmetric lower-triangle path); ``None`` keeps every triplet.
        self._keep = keep

    def data(self, values: Array) -> Array:
        """Canonical ``data`` for ``values`` via one scatter-add (no sort)."""
        if self._keep is not None:
            values = values[self._keep]
        out = self._xp.zeros((self.nnz,), dtype=values.dtype)
        self._scatter_add(out, self._inverse, values)
        return out


class CompiledLowerTriangle:
    """A fixed full-CSR → lower-triangle-CSR map; recompute *values* only.

    The cuDSS symmetric route factors the lower triangle. Given an
    already-canonical full CSR pattern, the lower triangle is a *subset* of its
    entries in the same sorted order (no duplicates to resum), so extracting it
    is a single boolean gather — far cheaper than ``tril(...)`` plus a
    re-canonicalization every iteration.
    """

    __slots__ = ("_mask", "indices", "indptr", "nnz")

    def __init__(self, *, indptr: Array, indices: Array, mask: Array, nnz: int) -> None:
        self.indptr = indptr
        self.indices = indices
        self.nnz = nnz
        self._mask = mask

    def data(self, full_data: Array) -> Array:
        """Lower-triangle ``data`` gathered from the full canonical CSR data."""
        return full_data[self._mask]


def compile_lower_triangle(
    xp: Any, indptr: Array, indices: Array, n: int
) -> CompiledLowerTriangle:
    """Compile the full-CSR → lower-triangle map for a canonical CSR pattern."""
    # Row owning each stored entry: the indptr bucket containing it. CuPy only
    # accepts array-valued ``repeat`` counts from v14, so the ownership map is
    # built with searchsorted (as in the adapters' Gram nnz→row maps) instead of
    # ``repeat(arange(n), diff(indptr))``.
    nnz_full = int(indices.shape[0])
    rows = xp.searchsorted(indptr, xp.arange(nnz_full), side="right") - 1
    mask = indices <= rows  # col ≤ row ⇒ lower triangle (incl. diagonal)
    lower_indices = indices[mask]
    lower_rows = rows[mask]
    nnz = int(lower_indices.shape[0])

    idx_dtype = index_dtype(xp, n, nnz)
    lower_counts = xp.bincount(lower_rows, minlength=n)
    lower_indptr = xp.zeros((n + 1,), dtype=idx_dtype)
    if nnz:
        lower_indptr[1:] = xp.cumsum(lower_counts).astype(idx_dtype, copy=False)
    return CompiledLowerTriangle(
        indptr=lower_indptr,
        indices=lower_indices.astype(idx_dtype, copy=False),
        mask=mask,
        nnz=nnz,
    )


def compile_compressed(
    xp: Any,
    scatter_add: Callable[[Any, Any, Any], None],
    *,
    major: Array,
    minor: Array,
    n_major: int,
    n_minor: int,
    keep: Array | None = None,
) -> CompiledCompressed:
    """Compile the canonical compressed-sparse map for a fixed COO pattern.

    ``major``/``minor`` are the emitted COO coordinate vectors of the compressed
    and the secondary axis: ``(row, col)`` for CSR, ``(col, row)`` for CSC.
    ``keep`` optionally pre-filters the triplets (e.g. the lower triangle of a
    symmetric matrix), so the same machinery serves cuDSS's lower-triangle solve.
    The sort/unique runs **once** here; per-iteration cost moves to :meth:`data`.
    """
    if keep is not None:
        major = major[keep]
        minor = minor[keep]
    # Lexicographic (major, minor) key in int64 — n_major*n_minor overflows int32
    # well within RT scale (1e5·1e5), so the key must be 64-bit regardless of the
    # index dtype the structure is finally stored in.
    major64 = major.astype(xp.int64, copy=False)
    minor64 = minor.astype(xp.int64, copy=False)
    keys = major64 * n_minor + minor64

    # ``unique`` returns the *sorted* distinct keys and an inverse index mapping
    # each input key to its position — exactly canonical CSR/CSC order, with the
    # inverse doubling as the duplicate-summing scatter map.
    uniq, inverse = xp.unique(keys, return_inverse=True)
    inverse = xp.reshape(inverse, (-1,))
    nnz = int(uniq.shape[0])

    idx_dtype = index_dtype(xp, n_major, n_minor, nnz)
    out_major = (uniq // n_minor).astype(idx_dtype, copy=False)
    indices = (uniq % n_minor).astype(idx_dtype, copy=False)

    counts = xp.bincount(out_major, minlength=n_major)
    indptr = xp.zeros((n_major + 1,), dtype=idx_dtype)
    if nnz:
        indptr[1:] = xp.cumsum(counts).astype(idx_dtype, copy=False)

    return CompiledCompressed(
        xp,
        scatter_add,
        indptr=indptr,
        indices=indices,
        inverse=inverse,
        nnz=nnz,
        keep=keep,
    )


__all__ = [
    "CompiledCompressed",
    "CompiledLowerTriangle",
    "compile_compressed",
    "compile_lower_triangle",
    "index_dtype",
]
