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

"""Torch Bunch-Kaufman LDLT dense adapter (CPU via LAPACK, CUDA via cuSOLVER/MAGMA)."""

from __future__ import annotations

from typing import Any

import torch

from ipax.linalg.solver import LinearSolveError


def cholesky_solve(factor: Any, rhs: Any) -> torch.Tensor:
    """Solve ``(L Lᵀ) x = rhs`` given the lower Cholesky factor ``L``.

    Gap-filler for the missing Array-API primitive: the ``linalg`` extension
    has no triangular solve (BLAS ``trsm`` / LAPACK ``potrs``), so a Cholesky
    factor cannot be back-substituted portably. Wraps ``torch.cholesky_solve``
    (LAPACK ``potrs`` on CPU, cuSOLVER on CUDA), O(n²) per right-hand side.
    """
    ell = torch.as_tensor(factor)
    b = torch.as_tensor(rhs)
    vector_rhs = b.ndim == 1
    b2d = b.unsqueeze(-1) if vector_rhs else b
    x = torch.cholesky_solve(b2d, ell, upper=False)
    return x.squeeze(-1) if vector_rhs else x


def _ldl_blocks_from_pivots(pivots: torch.Tensor) -> list[tuple[int, int]]:
    """Identify the ``(start, size)`` of each 1x1/2x2 block from LAPACK ``ipiv``.

    A 2x2 Bunch-Kaufman pivot is signalled by two consecutive equal *negative*
    entries in the pivot array (the standard ``?sytrf`` ``ipiv`` convention).
    """
    piv = [int(v) for v in pivots.tolist()]
    n = len(piv)
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if piv[i] < 0:
            blocks.append((i, 2))
            i += 2
        else:
            blocks.append((i, 1))
            i += 1
    return blocks


def _sign_count_blocks(
    ld: torch.Tensor, blocks: list[tuple[int, int]]
) -> tuple[int, int, int]:
    """Sign-count the packed ``LD`` diagonal/subdiagonal blocks (see numpy_scipy).

    A 1x1 block's sign is its diagonal entry; a genuine 2x2 Bunch-Kaufman
    pivot always contributes exactly one positive and one negative
    eigenvalue (verified via its determinant rather than assumed).
    """
    n = ld.shape[0]
    scale = float(torch.max(torch.abs(ld))) if n else 0.0
    eps = float(torch.finfo(ld.dtype).eps)
    tol = max(1.0, scale) * n * eps
    pos = neg = zero = 0
    for start, size in blocks:
        if size == 1:
            value = float(ld[start, start])
            if value > tol:
                pos += 1
            elif value < -tol:
                neg += 1
            else:
                zero += 1
        else:
            a = float(ld[start, start])
            b = float(ld[start + 1, start])
            c = float(ld[start + 1, start + 1])
            det = a * c - b * b
            if det < -tol:
                pos += 1
                neg += 1
            else:
                # Should never happen for a valid Bunch-Kaufman block; treat
                # defensively as numerically singular rather than guessing.
                zero += 2
    return pos, neg, zero


class TorchLDLFactorization:
    """Bunch-Kaufman LDLᵀ factorization of a dense symmetric matrix.

    Wraps ``torch.linalg.ldl_factor_ex``/``ldl_solve``: a genuinely pivoted
    symmetric-indefinite factorization (LAPACK ``?sytrf`` on CPU, cuSOLVER/
    MAGMA on CUDA — works unchanged on either device), unlike a plain
    ``eigh`` which does not reorder around ill-scaled entries. Exposes the
    factorization's inertia via the packed pivot structure — the dense
    analogue of ``FeralSparseSolver``.
    """

    def __init__(self) -> None:
        self._ld: torch.Tensor | None = None
        self._pivots: torch.Tensor | None = None
        self._inertia: tuple[int, int, int] | None = None

    def factor(self, matrix: Any) -> None:
        a = torch.as_tensor(matrix)
        try:
            ld, pivots, info = torch.linalg.ldl_factor_ex(a)
        except Exception as exc:
            raise LinearSolveError("dense LDL^T factorization failed") from exc
        if int(info) != 0:
            raise LinearSolveError(
                f"dense LDL^T factorization found a zero pivot (info={int(info)})"
            )
        self._ld = ld
        self._pivots = pivots
        blocks = _ldl_blocks_from_pivots(pivots)
        self._inertia = _sign_count_blocks(ld, blocks)

    def solve(self, rhs: Any) -> torch.Tensor:
        if self._ld is None or self._pivots is None:
            raise RuntimeError("factor() must be called before solve()")
        b = torch.as_tensor(rhs)
        vector_rhs = b.ndim == 1
        b2d = b.unsqueeze(-1) if vector_rhs else b
        try:
            x: torch.Tensor = torch.linalg.ldl_solve(self._ld, self._pivots, b2d)
        except Exception as exc:
            raise LinearSolveError("dense LDL^T solve failed") from exc
        return x.squeeze(-1) if vector_rhs else x

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """Inertia ``(n_pos, n_neg, n_zero)`` of the factored matrix, or ``None``."""
        return self._inertia


__all__ = ["TorchLDLFactorization", "cholesky_solve"]
