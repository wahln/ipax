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

"""Unit tests for the TROTS HDF5 matrix reader (no dataset required).

Regression (Head-and-Neck_05, 2026-07-19): MATLAB v7.3 stores an **all-zero
sparse matrix** as a group holding only ``jc`` — ``data`` and ``ir`` are omitted
entirely when nnz = 0 (HN_05's 'Brainstem' matrix, 2101×n, is one). The reader
assumed all three CSC arrays exist and crashed with a KeyError, taking down
every load of that case. These tests build synthetic v7.3-shaped files, so they
run wherever ``h5py`` is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
trots = pytest.importorskip("benchmarks.corpus.trots")


def _write_case(path, matrix_writer) -> None:
    """A minimal ``data.matrix`` table with one entry, A written by callback."""
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        a_obj = matrix_writer(refs)
        t = refs.create_dataset("t", data=np.array([[1.0]]))
        # 'Zero' as MATLAB uint16 char codes.
        nm = refs.create_dataset(
            "nm", data=np.array([[ord(ch)] for ch in "Zero"], dtype=np.uint16)
        )
        b = refs.create_dataset("b", data=np.array([[1.0], [2.0], [3.0]]))
        c = refs.create_dataset("c", data=np.zeros((0,)))
        mats = f.create_group("data").create_group("matrix")
        for field, obj in [("A", a_obj), ("Type", t), ("Name", nm), ("b", b), ("c", c)]:
            arr = np.array([[obj.ref]], dtype=h5py.ref_dtype)
            mats.create_dataset(field, data=arr)


def test_all_zero_sparse_matrix_loads_as_empty(tmp_path):
    # nnz = 0: MATLAB writes only ``jc`` (+ the MATLAB_sparse row count); the
    # reader must synthesize empty data/ir instead of KeyError-ing.
    path = str(tmp_path / "empty_sparse.mat")

    def zero_sparse(refs):
        g = refs.create_group("a")
        g.create_dataset("jc", data=np.zeros(3, dtype=np.uint64))  # 2 columns
        g.attrs["MATLAB_sparse"] = np.uint64(2101)
        return g

    _write_case(path, zero_sparse)
    mat = trots._load_matrix(path, 0)

    assert mat.name == "Zero"
    assert mat.mtype == 1
    assert mat.matrix.shape == (2101, 2)
    assert mat.matrix.nnz == 0
    np.testing.assert_array_equal(mat.b, [1.0, 2.0, 3.0])
    assert mat.c == 0.0


def test_ordinary_sparse_matrix_still_loads(tmp_path):
    # The nnz > 0 path must be untouched: a 2x2 CSC with entries (0,0)=5, (1,1)=7.
    path = str(tmp_path / "sparse.mat")

    def csc(refs):
        g = refs.create_group("a")
        g.create_dataset("data", data=np.array([5.0, 7.0]))
        g.create_dataset("ir", data=np.array([0, 1], dtype=np.uint64))
        g.create_dataset("jc", data=np.array([0, 1, 2], dtype=np.uint64))
        g.attrs["MATLAB_sparse"] = np.uint64(2)
        return g

    _write_case(path, csc)
    mat = trots._load_matrix(path, 0)

    assert mat.matrix.shape == (2, 2)
    assert mat.matrix.nnz == 2
    assert mat.matrix[0, 0] == 5.0 and mat.matrix[1, 1] == 7.0
    assert mat.source_dtype == "float64"


def _sparse_writer(data, ir, jc, nrows):
    def write(refs):
        g = refs.create_group("a")
        g.create_dataset("data", data=np.asarray(data))
        g.create_dataset("ir", data=np.asarray(ir, dtype=np.uint64))
        g.create_dataset("jc", data=np.asarray(jc, dtype=np.uint64))
        g.attrs["MATLAB_sparse"] = np.uint64(nrows)
        return g

    return write


def _assert_same_matrix(a, b):
    assert a.name == b.name and a.mtype == b.mtype and a.c == b.c
    assert a.source_dtype == b.source_dtype
    assert a.matrix.shape == b.matrix.shape
    assert a.matrix.dtype == b.matrix.dtype  # dtype must round-trip exactly
    np.testing.assert_array_equal(a.matrix.toarray(), b.matrix.toarray())
    np.testing.assert_array_equal(a.b, b.b)


def test_matrix_cache_round_trips_and_avoids_rereading(tmp_path, monkeypatch):
    # The .mat parse is the dominant cost of a TROTS run (tens of seconds for
    # the larger cases). The second load must come from the on-disk cache and
    # reproduce the parsed matrix exactly — including the float32 storage
    # dtype and source_dtype, which gram_dtype="auto" depends on.
    path = str(tmp_path / "cached.mat")
    _write_case(
        path,
        _sparse_writer(np.array([5.0, 7.0], dtype=np.float32), [0, 1], [0, 1, 2], 2),
    )
    cache = tmp_path / "cache"
    monkeypatch.setenv("IPAX_TROTS_CACHE", str(cache))

    first = trots._load_matrix(path, 0)
    assert list(cache.rglob("*.npz")), "first load should populate the cache"

    # A hit must not re-parse the HDF5 at all (that parse is the whole cost).
    def no_parse(*args, **kwargs):
        raise AssertionError("cache hit must not re-parse the .mat file")

    monkeypatch.setattr(trots, "_parse_matrix", no_parse)
    second = trots._load_matrix(path, 0)

    _assert_same_matrix(first, second)
    assert second.source_dtype == "float32"


def test_cache_narrows_losslessly_widened_values(tmp_path, monkeypatch):
    # Dense-stored matrices are widened to float64 at parse time even though
    # the file holds float32, which would double the cache. The cache stores
    # them narrow and widens on read — the rebuilt matrix must be bit-identical
    # to the parsed one, and still report float64.
    path = str(tmp_path / "dense32.mat")
    values = np.array([[1.5, 0.0], [0.0, -2.25]], dtype=np.float32)

    def dense32(refs):
        # Dense matrices are stored transposed (vars x voxels).
        return refs.create_dataset("a", data=values.T)

    _write_case(path, dense32)
    cache = tmp_path / "cache"
    monkeypatch.setenv("IPAX_TROTS_CACHE", str(cache))

    parsed = trots._load_matrix(path, 0)
    assert parsed.source_dtype == "float32"
    assert parsed.matrix.dtype == np.float64  # widened at parse

    npz = next(cache.rglob("*.npz"))
    with np.load(npz) as z:
        assert z["data"].dtype == np.float32  # stored narrow
        assert str(z["value_dtype"]) == "float64"

    monkeypatch.setattr(
        trots, "_parse_matrix", lambda *a, **k: pytest.fail("should hit cache")
    )
    cached = trots._load_matrix(path, 0)
    assert cached.matrix.dtype == np.float64  # widened back on read
    np.testing.assert_array_equal(cached.matrix.toarray(), parsed.matrix.toarray())


def test_cache_keeps_full_precision_when_narrowing_would_lose(tmp_path, monkeypatch):
    # A genuinely float64 matrix must round-trip exactly, never narrowed.
    path = str(tmp_path / "dense64.mat")
    values = np.array([[1.0 + 2.0**-40, 0.0], [0.0, 3.0]], dtype=np.float64)

    def dense64(refs):
        return refs.create_dataset("a", data=values.T)

    _write_case(path, dense64)
    monkeypatch.setenv("IPAX_TROTS_CACHE", str(tmp_path / "cache"))

    parsed = trots._load_matrix(path, 0)
    monkeypatch.setattr(
        trots, "_parse_matrix", lambda *a, **k: pytest.fail("should hit cache")
    )
    cached = trots._load_matrix(path, 0)

    np.testing.assert_array_equal(cached.matrix.toarray(), parsed.matrix.toarray())
    assert cached.matrix.toarray()[0, 0] == 1.0 + 2.0**-40


def test_matrix_cache_is_invalidated_when_the_file_changes(tmp_path, monkeypatch):
    path = str(tmp_path / "changing.mat")
    _write_case(path, _sparse_writer(np.array([5.0, 7.0]), [0, 1], [0, 1, 2], 2))
    monkeypatch.setenv("IPAX_TROTS_CACHE", str(tmp_path / "cache"))

    first = trots._load_matrix(path, 0)
    assert first.matrix[0, 0] == 5.0

    # Rewrite with different values (and a different size/mtime): the cache key
    # must miss rather than serve the stale matrix.
    _write_case(
        path, _sparse_writer(np.array([9.0, 9.0, 9.0]), [0, 1, 0], [0, 2, 3], 2)
    )
    second = trots._load_matrix(path, 0)

    assert second.matrix[0, 0] == 9.0


def test_matrix_cache_can_be_disabled(tmp_path, monkeypatch):
    path = str(tmp_path / "nocache.mat")
    _write_case(path, _sparse_writer(np.array([5.0, 7.0]), [0, 1], [0, 1, 2], 2))
    cache = tmp_path / "cache"
    monkeypatch.setenv("IPAX_TROTS_CACHE", "off")

    mat = trots._load_matrix(path, 0)

    assert mat.matrix[0, 0] == 5.0
    assert not cache.exists()


def test_matrix_cache_failure_is_not_fatal(tmp_path, monkeypatch):
    # An unwritable cache location must degrade to plain parsing, never break
    # the load (the cache is an optimization, not a dependency).
    path = str(tmp_path / "readonly.mat")
    _write_case(path, _sparse_writer(np.array([5.0, 7.0]), [0, 1], [0, 1, 2], 2))
    monkeypatch.setenv("IPAX_TROTS_CACHE", str(tmp_path / "cache"))

    def boom(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(trots, "_write_cached_matrix", boom)

    mat = trots._load_matrix(path, 0)
    assert mat.matrix[0, 0] == 5.0


def test_float32_stored_matrix_records_source_dtype(tmp_path):
    # TROTS dose matrices are float32 in the files; the loader records that
    # source precision so the assembled (float64-promoted) constraint block
    # can declare it — the metadata behind DenseOptions(gram_dtype="auto").
    path = str(tmp_path / "sparse32.mat")

    def csc32(refs):
        g = refs.create_group("a")
        g.create_dataset("data", data=np.array([5.0, 7.0], dtype=np.float32))
        g.create_dataset("ir", data=np.array([0, 1], dtype=np.uint64))
        g.create_dataset("jc", data=np.array([0, 1, 2], dtype=np.uint64))
        g.attrs["MATLAB_sparse"] = np.uint64(2)
        return g

    _write_case(path, csc32)
    mat = trots._load_matrix(path, 0)

    assert mat.source_dtype == "float32"
