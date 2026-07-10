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

"""NumPy/SciPy Bunch-Kaufman LDLT dense adapter (LAPACK ``?sytrf``)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import ldl as _scipy_ldl
from scipy.linalg import solve_triangular

from ipax.linalg.solver import LinearSolveError


def _ldl_blocks(d: np.ndarray) -> list[tuple[int, int]]:
    """Identify the ``(start, size)`` of each 1x1/2x2 block along ``d``.

    ``scipy.linalg.ldl``'s block-diagonal ``d`` signals a 2x2 Bunch-Kaufman
    pivot with a nonzero first off-diagonal entry ``d[i, i+1]``.
    """
    n = d.shape[0]
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if i + 1 < n and d[i, i + 1] != 0.0:
            blocks.append((i, 2))
            i += 2
        else:
            blocks.append((i, 1))
            i += 1
    return blocks


def _block_diagonal_solve(
    d: np.ndarray, blocks: list[tuple[int, int]], rhs: np.ndarray
) -> np.ndarray:
    """Solve ``d @ x = rhs`` in ``O(n)`` by walking the 1x1/2x2 blocks."""
    out = np.empty_like(rhs)
    for start, size in blocks:
        if size == 1:
            out[start] = rhs[start] / d[start, start]
        else:
            block = d[start : start + 2, start : start + 2]
            out[start : start + 2] = np.linalg.solve(block, rhs[start : start + 2])
    return out


def _sign_count_blocks(
    d: np.ndarray, blocks: list[tuple[int, int]]
) -> tuple[int, int, int]:
    """Sign-count the block-diagonal ``d`` into ``(n_pos, n_neg, n_zero)``.

    A 1x1 block's sign is its (tolerance-gated) diagonal entry. A genuine 2x2
    Bunch-Kaufman pivot always contributes exactly one positive and one
    negative eigenvalue (its determinant is negative by construction — that
    is precisely why the algorithm chose a 2x2 pivot over a 1x1 one); this is
    verified via the block's determinant rather than assumed.
    """
    n = d.shape[0]
    scale = float(np.max(np.abs(d))) if n else 0.0
    tol = max(1.0, scale) * n * float(np.finfo(d.dtype).eps)
    pos = neg = zero = 0
    for start, size in blocks:
        if size == 1:
            value = d[start, start]
            if value > tol:
                pos += 1
            elif value < -tol:
                neg += 1
            else:
                zero += 1
        else:
            a, b, c = d[start, start], d[start, start + 1], d[start + 1, start + 1]
            det = a * c - b * b
            if det < -tol:
                pos += 1
                neg += 1
            else:
                # Should never happen for a valid Bunch-Kaufman block; treat
                # defensively as numerically singular rather than guessing.
                zero += 2
    return pos, neg, zero


class ScipyLDLFactorization:
    """Bunch-Kaufman LDLᵀ factorization of a dense symmetric matrix.

    Wraps ``scipy.linalg.ldl`` (LAPACK ``?sytrf``): a genuinely pivoted
    symmetric-indefinite factorization, unlike a plain ``eigh``, which does
    not reorder around ill-scaled entries. Solves via the permuted
    triangular factors directly (``O(n^2)`` per solve) instead of a generic
    ``O(n^3)`` solve on the unpermuted factor, and exposes the factorization's
    inertia — the dense analogue of ``FeralSparseSolver``.
    """

    def __init__(self) -> None:
        self._l: np.ndarray | None = None
        self._d: np.ndarray | None = None
        self._perm: np.ndarray | None = None
        self._blocks: list[tuple[int, int]] | None = None
        self._inertia: tuple[int, int, int] | None = None

    def factor(self, matrix: Any) -> None:
        a = np.asarray(matrix)
        try:
            lu, d, perm = _scipy_ldl(a)
        except Exception as exc:
            raise LinearSolveError("dense LDL^T factorization failed") from exc
        self._l = lu[perm, :]
        self._d = d
        self._perm = perm
        self._blocks = _ldl_blocks(d)
        self._inertia = _sign_count_blocks(d, self._blocks)

    def solve(self, rhs: Any) -> np.ndarray:
        if self._l is None or self._d is None or self._perm is None:
            raise RuntimeError("factor() must be called before solve()")
        assert self._blocks is not None
        b = np.asarray(rhs)
        perm = self._perm
        try:
            w = solve_triangular(self._l, b[perm], lower=True, unit_diagonal=True)
            z = _block_diagonal_solve(self._d, self._blocks, w)
            y = solve_triangular(self._l.T, z, lower=False, unit_diagonal=True)
        except Exception as exc:
            raise LinearSolveError("dense LDL^T solve failed") from exc
        x = np.empty_like(b)
        x[perm] = y
        return x

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """Inertia ``(n_pos, n_neg, n_zero)`` of the factored matrix, or ``None``."""
        return self._inertia


__all__ = ["ScipyLDLFactorization"]
