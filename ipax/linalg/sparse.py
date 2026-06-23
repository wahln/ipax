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

"""Backend-neutral sparse-direct ``LinearSolver`` facade (§5.2, route 3).

This is the core-side half of the sparse-direct route. It owns no concrete
sparse library (invariant #1): it emits the KKT operator's COO structure as
Array-API ``(rows, cols, values)`` vectors (invariant #4), resolves the sparse
adapter registered for that namespace, and delegates assembly + factorization to
it. Adding this route therefore never touches ``ipm/driver.py`` (invariant #3) —
the driver hands the same condensed/saddle :class:`LinearOperator` to whichever
solver was injected, and this one knows how to turn it into a sparse factor.

Only operators that expose :meth:`~ipax.backend.operators.LinearOperator.to_coo`
can be factored here. The condensed/saddle operators do so when their blocks are
assemblable (analytic/sparse Hessian, no condensed inequality Gram term);
otherwise ``to_coo`` raises and the caller must pick the dense or matrix-free
route instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ipax.backend.namespace import array_namespace

if TYPE_CHECKING:
    from ipax.backend.operators import LinearOperator
    from ipax.typing import Array


class SparseDirectSolver:
    """``LinearSolver`` that factors via the per-backend sparse adapter."""

    def __init__(self, *, require_inertia: bool = False) -> None:
        self._require_inertia = require_inertia
        self._inner: Any = None
        # When the operator carries an L-BFGS low-rank tail, ``to_coo`` returns a
        # larger *bordered* matrix than the logical system the driver solves; we
        # pad/truncate the RHS and solution across that gap (see ``solve``).
        self._logical_size = 0
        self._assembled_size = 0

    def describe(self) -> str:
        """Human-readable label, delegating to the dispatched backend solver."""
        if self._inner is None:
            return "sparse (unfactored)"
        inner_describe = getattr(self._inner, "describe", None)
        detail = (
            inner_describe() if callable(inner_describe) else type(self._inner).__name__
        )
        return f"sparse [{detail}]"

    def factor(self, K: LinearOperator) -> None:
        # The core emits structure; the adapter builds and factors the matrix.
        rows, cols, values, shape = K.to_coo()
        xp = array_namespace(values)
        # Forward the operator's structural symmetry hint (the condensed/saddle
        # blocks are symmetric by construction) so the adapter can skip its
        # per-iteration numerical A − Aᵀ test.
        symmetric = K.symmetry_hint()

        from ipax.backend.sparse import get_sparse_adapter

        adapter: Any = get_sparse_adapter(xp)
        if adapter is None:
            raise RuntimeError(
                "no sparse-direct adapter is available for this backend; "
                "install SciPy for the NumPy backend or choose another "
                "linsolve mode"
            )
        operator = adapter.from_coo(
            rows, cols, values, shape=shape, symmetric=symmetric
        )
        # The facade is created once per solve and reused every iteration, so the
        # inner solver persists — letting the backend cache its symbolic analysis
        # across the fixed-pattern KKT factorizations of an interior-point solve.
        if self._inner is None:
            self._inner = adapter.solver(require_inertia=self._require_inertia)
        self._inner.factor(operator)
        self._logical_size = int(K.shape[0])
        self._assembled_size = int(shape[0])

    def solve(self, rhs: Array) -> Array:
        if self._inner is None:
            raise RuntimeError("factor() must be called before solve()")
        # The auxiliary (low-rank border) variables have a zero right-hand side
        # and are discarded from the solution: solving the bordered system
        # ``[A U; Uᵀ M][Δx; p] = [r; 0]`` yields the logical Newton step ``Δx``.
        pad = self._assembled_size - self._logical_size
        if pad:
            xp = array_namespace(rhs)
            rhs = xp.concat((rhs, xp.zeros((pad,), dtype=rhs.dtype)))
        solution = self._inner.solve(rhs)
        if pad:
            solution = solution[: self._logical_size]
        return solution

    @property
    def inertia(self) -> tuple[int, int, int]:
        """Inertia ``(n₊, n₋, n₀)`` of the factored operator (if requested)."""
        if self._inner is None:
            raise RuntimeError("factor() must be called before reading inertia")
        return self._inner.inertia  # type: ignore[no-any-return]

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """Best-effort inertia of the factored operator; ``None`` if unavailable.

        Unlike :attr:`inertia` this never raises: it returns the LDLᵀ inertia
        when the dispatched backend solver computed it for free (e.g. Feral /
        cuDSS), and ``None`` otherwise (a non-inertia-revealing factorization
        such as SuperLU). The IPM uses it for inertia-guided δ_w correction and
        falls back to factorization-failure escalation when it is ``None``.
        """
        if self._inner is None:
            return None
        fn = getattr(self._inner, "inertia_or_none", None)
        return fn() if fn is not None else None


__all__ = ["SparseDirectSolver"]
