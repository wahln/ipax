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
    from collections.abc import Callable

    from ipax.backend.namespace import Capabilities
    from ipax.backend.operators import LinearOperator
    from ipax.options import Options
    from ipax.typing import Array

# Auto-selection thresholds. Below ``_DENSE_AUTO_MAX_VARS`` the dense route is
# the reference choice regardless of constraint shape. Between that and
# ``_TALL_DENSE_MAX_VARS`` the dense condensed (normal-equations) route is still
# preferred *when the problem is tall*: with ``m ≥ _TALL_ROW_EXCESS · n``
# inequality rows and a Gram-capable Jacobian, one n×n Cholesky per iteration
# beats Krylov's repeated O(nnz(∇g)) matvecs through the huge Jacobian
# (Breedveld 2017 §2: the condensed system is n×n however large m grows).
_DENSE_AUTO_MAX_VARS = 10_000
_TALL_DENSE_MAX_VARS = 20_000
_TALL_ROW_EXCESS = 10


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

    def set_outer_residual(self, residual: float) -> None:
        """Hint the current outer KKT residual for an adaptive inner tolerance.

        Iterative solvers use it to drive an inexact-Newton forcing sequence
        (solve loosely while the IPM is far from optimal, tighten as it converges);
        direct solvers ignore it. Optional — the driver calls it once per iteration
        and a solver may no-op.
        """
        ...


def select_solver(
    *,
    n_vars: int,
    has_equalities: bool,
    capabilities: Capabilities,
    options: Options,
    m_ineq: int = 0,
    ineq_gram_capable: Callable[[], bool] | None = None,
) -> LinearSolver:
    """Pick a concrete linear solver from user preference and backend features.

    ``m_ineq`` (total inequality rows) and ``ineq_gram_capable`` (a *lazy*
    structural probe of the inequality Jacobian — it may evaluate the Jacobian
    at ``x0``, so it is only called when the tall-problem heuristic actually
    needs it) extend auto-selection to tall ``n ≪ m`` problems.
    """
    from ipax.linalg.dense import DenseSolver
    from ipax.linalg.krylov import KrylovSolver

    def has_dense_solve() -> bool:
        return capabilities.has_linalg and "solve" in capabilities.linalg_functions

    mode = options.linsolve
    if mode == "dense":
        if not has_dense_solve():
            raise RuntimeError("dense linear solving requires xp.linalg.solve")
        return DenseSolver(options.dense)

    if mode == "krylov":
        return KrylovSolver(options.krylov)

    if mode == "sparse":
        if not capabilities.has_sparse_adapter:
            raise RuntimeError("sparse linear solving is unavailable for this backend")
        from ipax.linalg.sparse import SparseDirectSolver

        return SparseDirectSolver()

    dense_viable = has_dense_solve() and (
        not has_equalities or "cholesky" in capabilities.linalg_functions
    )
    if n_vars < _DENSE_AUTO_MAX_VARS and dense_viable:
        return DenseSolver(options.dense)

    # Tall n ≪ m: the condensed block stays n×n however many inequality rows
    # exist, and a Gram-capable Jacobian forms it without densifying m×n —
    # prefer the direct dense route over Krylov through the huge Jacobian.
    if (
        n_vars < _TALL_DENSE_MAX_VARS
        and dense_viable
        and m_ineq >= _TALL_ROW_EXCESS * n_vars
        and ineq_gram_capable is not None
        and ineq_gram_capable()
    ):
        return DenseSolver(options.dense)

    return KrylovSolver(options.krylov)


__all__ = ["LinearSolveError", "LinearSolver", "select_solver"]
