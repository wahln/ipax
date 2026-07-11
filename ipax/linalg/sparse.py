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

from ipax.backend.namespace import _namespace_name, array_namespace

if TYPE_CHECKING:
    from collections.abc import Callable

    from ipax.backend.operators import LinearOperator
    from ipax.typing import Array


def _dlpack_device_kind(arr: Array) -> int | None:
    """DLPack device kind for adapter-cache guarding, or ``None``."""
    dlpack_device = getattr(arr, "__dlpack_device__", None)
    if dlpack_device is None:
        return None
    try:
        kind, _ = dlpack_device()
    except Exception:
        return None
    return int(kind)


class SparseDirectSolver:
    """``LinearSolver`` that factors via the per-backend sparse adapter.

    ``form`` selects what is assembled: ``"augmented"`` (default) factors the
    operator's bordered COO emission (inequalities as an indefinite ``∇g`` /
    ``−Σ_s⁻¹`` border — the factor stays as sparse as ``∇g``), while
    ``"normal_equations"`` condenses the inequality Gram term sparsely into the
    logical ``n×n`` block (``normal_equations_coo``), for tall problems whose
    Jacobian rows are localized enough that ``∇gᵀ Σ_s ∇g`` stays sparse.
    On non-localized sparsity the Gram fills in catastrophically, so the form
    is selectable explicitly (``SparseOptions(kkt_route=...)``) and picked by
    ``select_solver`` only when the ``gram_fill_estimate`` probe (sampled
    column overlap of ``∇g``) certifies a sparse Gram pattern.
    """

    def __init__(
        self, *, require_inertia: bool = False, form: str = "augmented"
    ) -> None:
        if form not in ("augmented", "normal_equations"):
            raise ValueError(
                "SparseDirectSolver form must be 'augmented' or 'normal_equations'"
            )
        self._require_inertia = require_inertia
        self._form = form
        self._adapter: Any = None
        self._adapter_key: tuple[str, int | None] | None = None
        self._inner: Any = None
        # When the operator carries an L-BFGS low-rank tail, ``to_coo`` returns a
        # larger *bordered* matrix than the logical system the driver solves; we
        # pad/truncate the RHS and solution across that gap (see ``solve``).
        self._logical_size = 0
        self._assembled_size = 0
        # Bordered-equivalent inertia delta for the normal-equations form
        # (Haynsworth: the eliminated −Σ_s⁻¹ block's m_I negatives).
        self._inertia_offset: tuple[int, int, int] = (0, 0, 0)
        # Cached COO row/column structure for the current pattern signature: the
        # KKT pattern is fixed across IPM iterations, so on a cache hit only the
        # value vector is recomputed (``coo_values``) instead of the full triplet.
        self._struct_signature: object | None = None
        self._struct: tuple[Array, Array, tuple[int, int]] | None = None

    @property
    def form(self) -> str:
        """The assembled KKT form: ``"augmented"`` or ``"normal_equations"``."""
        return self._form

    def describe(self) -> str:
        """Human-readable label, delegating to the dispatched backend solver."""
        if self._inner is None:
            return "sparse (unfactored)"
        inner_describe = getattr(self._inner, "describe", None)
        detail = (
            inner_describe() if callable(inner_describe) else type(self._inner).__name__
        )
        prefix = "sparse-NE" if self._form == "normal_equations" else "sparse"
        return f"{prefix} [{detail}]"

    def set_outer_residual(self, residual: float) -> None:
        """No-op: a direct factorization has no inner tolerance to adapt."""
        del residual

    def _emission(
        self, K: LinearOperator
    ) -> tuple[Callable[[], object | None], Callable[..., Any], Callable[..., Any]]:
        """The (signature, to_coo, values) emission triple for the active form."""
        if self._form == "normal_equations":
            to_coo = getattr(K, "normal_equations_coo", None)
            values = getattr(K, "normal_equations_values", None)
            signature_fn = getattr(K, "normal_equations_pattern_signature", None)
            if to_coo is None or values is None or signature_fn is None:
                raise RuntimeError(
                    "the sparse normal-equations form requires a condensed KKT "
                    "operator that can fold the inequality Gram sparsely "
                    "(no equality constraints; a gram_coo-capable sparse "
                    "inequality Jacobian); use the default 'augmented' form "
                    "or another linsolve mode instead"
                )
            return signature_fn, to_coo, values
        return K.coo_pattern_signature, K.to_coo, K.coo_values

    def factor(self, K: LinearOperator) -> None:
        # The core emits structure; the adapter builds and factors the matrix.
        # On a fixed pattern (stable signature) reuse the cached row/column
        # vectors and recompute only the values — the index arrays (and the
        # low-rank border's index grids) are identical every iteration.
        signature_fn, to_coo_fn, values_fn = self._emission(K)
        signature = signature_fn()
        try:
            if (
                signature is not None
                and self._struct is not None
                and self._struct_signature == signature
            ):
                rows, cols, shape = self._struct
                values = values_fn()
            else:
                rows, cols, values, shape = to_coo_fn()
                if signature is not None:
                    self._struct = (rows, cols, shape)
                    self._struct_signature = signature
                else:
                    self._struct = None
                    self._struct_signature = None
        except NotImplementedError as exc:
            if self._form == "normal_equations":
                raise RuntimeError(
                    "the sparse normal-equations form is unavailable for this "
                    f"problem: {exc}"
                ) from exc
            raise
        xp = array_namespace(values)
        adapter_key = (_namespace_name(xp), _dlpack_device_kind(values))
        # Forward the operator's structural symmetry hint (the condensed/saddle
        # blocks are symmetric by construction) so the adapter can skip its
        # per-iteration numerical A − Aᵀ test.
        symmetric = K.symmetry_hint()

        from ipax.backend.sparse import get_sparse_adapter

        if self._adapter is None:
            self._adapter = get_sparse_adapter(xp)
            if self._adapter is None:
                raise RuntimeError(
                    "no sparse-direct adapter is available for this backend; "
                    "install SciPy for the NumPy backend or choose another "
                    "linsolve mode"
                )
            self._adapter_key = adapter_key
        elif adapter_key != self._adapter_key:
            raise RuntimeError(
                "SparseDirectSolver cannot reuse a cached sparse adapter across "
                f"array backends/devices: cached {self._adapter_key}, got "
                f"{adapter_key}"
            )
        adapter = self._adapter
        operator = adapter.from_coo(
            rows,
            cols,
            values,
            shape=shape,
            symmetric=symmetric,
            pattern_signature=signature,
        )
        # The facade is created once per solve and reused every iteration, so the
        # inner solver persists — letting the backend cache its symbolic analysis
        # across the fixed-pattern KKT factorizations of an interior-point solve.
        if self._inner is None:
            self._inner = adapter.solver(require_inertia=self._require_inertia)
        self._inner.factor(operator)
        self._logical_size = int(K.shape[0])
        self._assembled_size = int(shape[0])
        offset_fn = (
            getattr(K, "normal_equations_inertia_offset", None)
            if self._form == "normal_equations"
            else None
        )
        self._inertia_offset = offset_fn() if offset_fn is not None else (0, 0, 0)

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

    @staticmethod
    def _offset_inertia(
        inertia: tuple[int, int, int] | None, offset: tuple[int, int, int]
    ) -> tuple[int, int, int] | None:
        if inertia is None:
            return None
        return (
            inertia[0] + offset[0],
            inertia[1] + offset[1],
            inertia[2] + offset[2],
        )

    @property
    def inertia(self) -> tuple[int, int, int]:
        """Inertia ``(n₊, n₋, n₀)`` of the factored system, in *bordered* terms.

        For the normal-equations form the eliminated slack block's negatives
        are added back (see ``normal_equations_inertia_offset``), so the value
        is comparable with the operator's ``expected_inertia`` target either way.
        """
        if self._inner is None:
            raise RuntimeError("factor() must be called before reading inertia")
        result = self._offset_inertia(self._inner.inertia, self._inertia_offset)
        assert result is not None
        return result

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
        inner = fn() if fn is not None else None
        return self._offset_inertia(inner, self._inertia_offset)


__all__ = ["SparseDirectSolver"]
