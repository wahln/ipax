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

"""``DenseSolver`` — Array-API reference solver.

Materializes the condensed/saddle KKT matrix and solves it with
``xp.linalg.solve``. Because ``xp.linalg.solve`` is an LU factorization it
*succeeds* on an indefinite system, which would let a non-descent Newton step
through on a nonconvex problem; so before solving it probes the condensed block
``N`` (the part that must be positive definite) with ``xp.linalg.cholesky`` and
raises :class:`LinearSolveError` when ``N`` is not PD, driving the IPM's δ_w
escalation — the dense analog of the sparse route's inertia check. Both
``cholesky`` and ``solve`` are in the Array-API ``linalg`` extension, so this
stays pure Array API. Target ≤ ~1e4 variables; the reference implementation every
other solver is checked against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ipax.backend.namespace import array_namespace
from ipax.linalg.solver import LinearSolveError

if TYPE_CHECKING:
    from ipax.backend.operators import LinearOperator
    from ipax.typing import Array


class DenseSolver:
    """Cholesky/solve on the materialized condensed KKT block."""

    def __init__(self) -> None:
        self._operator: LinearOperator | None = None

    def describe(self) -> str:
        """Human-readable label for diagnostics."""
        return "dense"

    def factor(self, K: LinearOperator) -> None:
        if K.shape[0] != K.shape[1]:
            raise ValueError("DenseSolver requires a square operator")
        self._operator = K

    def solve(self, rhs: Array) -> Array:
        if self._operator is None:
            raise RuntimeError("factor() must be called before solve()")

        n = self._operator.shape[0]
        if int(rhs.shape[0]) != n:
            raise ValueError("right-hand side dimension does not match operator")

        xp = array_namespace(rhs)
        identity = xp.eye(n, dtype=rhs.dtype)
        matrix = self._operator.matmat(identity)
        self._guard_positive_definite(matrix, xp, rhs.dtype)
        try:
            return xp.linalg.solve(matrix, rhs)
        except Exception as exc:
            raise LinearSolveError("dense linear solve failed") from exc

    def _guard_positive_definite(self, matrix: Array, xp: Any, dtype: Any) -> None:
        """Reject a non-PD condensed block so the IPM escalates δ_w.

        ``xp.linalg.solve`` (LU) would silently accept an indefinite ``N`` and
        return a non-descent step; a Cholesky on the condensed block fails exactly
        when ``N`` is not positive definite. Skips the probe when the operator
        exposes no condensed block (e.g. an L-BFGS low-rank Hessian, PD by
        damping) or when the backend lacks ``cholesky``.
        """
        primal_block = getattr(self._operator, "primal_block", None)
        block = primal_block() if primal_block is not None else None
        if block is None:
            return
        cholesky = getattr(xp.linalg, "cholesky", None)
        if cholesky is None:
            return
        n = block.shape[0]
        # For the condensed (no-equality) operator the materialized matrix *is*
        # ``N``; the saddle materializes ``N`` from its primal sub-block instead.
        primal = (
            matrix if n == matrix.shape[0] else block.matmat(xp.eye(n, dtype=dtype))
        )
        try:
            cholesky(primal)
        except Exception as exc:
            raise LinearSolveError("condensed block is not positive definite") from exc


__all__ = ["DenseSolver"]
