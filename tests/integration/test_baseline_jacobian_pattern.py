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

"""Declaring a fixed Jacobian pattern for a solver that demands one.

IPOPT wants the constraint Jacobian's sparsity declared **once**, up front. Many
CUTEst problems do not oblige: an entry whose value happens to be exactly zero at
a given point is simply absent from the operator's COO triplets there, so the
structural pattern the solver sees depends on where it looks.

The fix is a *union* pattern — sample several points, declare the superset, and
let the callback place an explicit stored zero wherever an entry is missing at
the current point. These tests pin that behaviour, including the two ways it
used to break: concluding a pattern was stable from too few samples (which
surfaced mid-solve as an opaque NumPy broadcast error), and reporting an entry
outside the declared pattern as anything other than a clear rejection.

No ``ipyopt`` needed — the translation is pure Python over ipax operators.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.baselines import BaselineUnsupported, _ipyopt_blocks
from ipax.backend.operators import COOOperator
from ipax.problem.base import Problem


class _VanishingEntry(Problem):
    """Two constraints in two variables; the (1, 1) entry vanishes when x1 == 0.

    ``c(x) = [x0, x1**2]`` — the second row's derivative ``2*x1`` is structurally
    present but *numerically* zero at the origin, and the operator omits it
    there. A pattern read only at the origin therefore misses it.
    """

    n_vars = 2
    x0 = np.zeros(2)

    def objective(self, x):
        return float(np.sum(x))

    def gradient(self, x):
        return np.ones(2)

    def eq_constraints(self, x):
        return np.asarray([x[0], x[1] ** 2])

    def eq_jacobian(self, x):
        rows = [0]
        cols = [0]
        values = [1.0]
        if x[1] != 0.0:  # the operator omits an exactly-zero entry
            rows.append(1)
            cols.append(1)
            values.append(2.0 * float(x[1]))
        return COOOperator(
            np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64),
            np.asarray(values, dtype=float),
            (2, 2),
        )


class _StablePattern(Problem):
    """``c(x) = [x0 + x1]`` — one row, both entries always present."""

    n_vars = 2
    x0 = np.zeros(2)

    def objective(self, x):
        return float(np.sum(x))

    def gradient(self, x):
        return np.ones(2)

    def eq_constraints(self, x):
        return np.asarray([x[0] + x[1]])

    def eq_jacobian(self, x):
        return COOOperator(
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([1.0, 1.0], dtype=float),
            (1, 2),
        )


def _block(problem):
    blocks, _m = _ipyopt_blocks(problem, np.asarray(problem.x0, dtype=float))
    assert len(blocks) == 1
    return blocks[0]


def test_union_pattern_declares_an_entry_absent_at_the_start_point():
    # The whole point: x0 is the origin, where the (1, 1) entry does not exist.
    block = _block(_VanishingEntry())
    declared = set(zip(block.rows.tolist(), block.cols.tolist(), strict=True))
    assert declared == {(0, 0), (1, 1)}


def test_a_missing_entry_becomes_an_explicit_stored_zero():
    block = _block(_VanishingEntry())
    values = block.values_fn(np.zeros(2))
    assert values.shape == block.rows.shape
    position = list(zip(block.rows.tolist(), block.cols.tolist(), strict=True)).index(
        (1, 1)
    )
    assert values[position] == 0.0


def test_values_land_in_their_declared_slots_when_the_entry_is_present():
    block = _block(_VanishingEntry())
    values = block.values_fn(np.asarray([7.0, 3.0]))
    pairs = list(zip(block.rows.tolist(), block.cols.tolist(), strict=True))
    assert values[pairs.index((0, 0))] == 1.0
    assert values[pairs.index((1, 1))] == 6.0  # 2 * 3


class _RemotePattern(Problem):
    """A third entry that only exists far from anywhere the sampling looks.

    This is the failure that produced the opaque mid-solve broadcast errors
    (S2MPJ ``EIGMINA``: 5 nonzeros at the start point and at the probe, 6 once
    the iterates move). No finite sample can guarantee coverage, so the values
    callback has to *notice* rather than assume.
    """

    n_vars = 3
    x0 = np.zeros(3)

    def objective(self, x):
        return float(np.sum(x))

    def gradient(self, x):
        return np.ones(3)

    def eq_constraints(self, x):
        return np.asarray([x[0] + x[1]])

    def eq_jacobian(self, x):
        rows, cols, values = [0, 0], [0, 1], [1.0, 1.0]
        if x[0] > 100.0:  # a column no sampled point ever exercises
            rows, cols, values = [0, 0, 0], [0, 1, 2], [1.0, 1.0, 1.0]
        return COOOperator(
            np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64),
            np.asarray(values, dtype=float),
            (1, 3),
        )


def test_an_unsampled_pattern_is_rejected_clearly_not_as_a_broadcast_error():
    # A pattern nothing in the sample set predicted must become an explicit
    # rejection here, rather than a NumPy shape error thrown from inside the
    # solve, where it cannot be attributed to anything.
    block = _block(_RemotePattern())
    np.testing.assert_allclose(block.values_fn(np.asarray([1.0, 1.0, 1.0])), [1.0, 1.0])
    with pytest.raises(BaselineUnsupported, match="pattern"):
        block.values_fn(np.asarray([1000.0, 1.0, 1.0]))


def test_a_stable_pattern_declares_no_spurious_entries():
    block = _block(_StablePattern())
    assert sorted(zip(block.rows.tolist(), block.cols.tolist(), strict=True)) == [
        (0, 0),
        (0, 1),
    ]
    np.testing.assert_allclose(block.values_fn(np.asarray([2.0, 5.0])), [1.0, 1.0])
