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

"""The ``LinearSolver`` protocol and strategy selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ipax.backend.namespace import Capabilities
    from ipax.backend.operators import LinearOperator
    from ipax.options import Options
    from ipax.typing import Array


class LinearSolveError(RuntimeError):
    """Backend-neutral numerical failure from a linear solve.

    The IPM regularization loop catches this type and retries with a larger
    primal regularization. Configuration errors, unsupported features, shape
    errors, and user/operator callback bugs should propagate as their original
    exceptions instead of being reclassified as numerical trouble.
    """


@runtime_checkable
class LinearSolver(Protocol):
    """Solve ``K x = rhs`` for a KKT or condensed operator ``K``."""

    def factor(self, K: LinearOperator) -> None:
        """Prepare/factor the operator."""
        ...

    def solve(self, rhs: Array) -> Array:
        """Return ``x`` such that ``K x = rhs`` to the configured tolerance."""
        ...


def select_solver(
    *,
    n_vars: int,
    has_equalities: bool,
    capabilities: Capabilities,
    options: Options,
) -> LinearSolver:
    """Pick a concrete linear solver from user preference and backend features."""
    from ipax.linalg.dense import DenseSolver
    from ipax.linalg.krylov import KrylovSolver

    def has_dense_solve() -> bool:
        return capabilities.has_linalg and "solve" in capabilities.linalg_functions

    mode = options.linsolve
    if mode == "dense":
        if not has_dense_solve():
            raise RuntimeError("dense linear solving requires xp.linalg.solve")
        return DenseSolver()

    if mode == "krylov":
        return KrylovSolver(options.krylov)

    if mode == "sparse":
        if not capabilities.has_sparse_adapter:
            raise RuntimeError("sparse linear solving is unavailable for this backend")
        from ipax.linalg.sparse import SparseDirectSolver

        return SparseDirectSolver()

    if (
        n_vars < 10_000
        and has_dense_solve()
        and (not has_equalities or "cholesky" in capabilities.linalg_functions)
    ):
        return DenseSolver()

    return KrylovSolver(options.krylov)


__all__ = ["LinearSolveError", "LinearSolver", "select_solver"]
