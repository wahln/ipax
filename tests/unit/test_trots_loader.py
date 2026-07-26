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
