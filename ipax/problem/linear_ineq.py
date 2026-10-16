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

"""Lower two-sided linear inequalities into one-sided constraints.

The interior-point loop handles inequalities in the one-sided standard form
``g(x) ≤ 0`` (slack ``s`` with ``g + s = 0``). A user may instead declare
constant-data two-sided linear inequalities ``l ≤ A x ≤ u`` through
:meth:`~ipax.problem.base.Problem.linear_ineq`. This module rewrites that block
into the equivalent one-sided rows and appends them to the problem's nonlinear
inequalities, so the IPM, gradient scaling, and every solver route consume them
with no special-casing — the same strategy by which linear *equalities* fold into
the equality system (driver ``_eq``/``_eq_jac``).

Each finite lower row ``l_i ≤ A_i x`` becomes ``l_i − A_i x ≤ 0`` (Jacobian row
``−A_i``); each finite upper row ``A_i x ≤ u_i`` becomes ``A_i x − u_i ≤ 0``
(Jacobian row ``+A_i``). A row finite on both sides yields both (a range
constraint); a row infinite on both sides is dropped. The lowered Jacobian is
constant, so the block contributes no Lagrangian-Hessian term — the lowered
multipliers are sliced off before the inner problem's ``lagrangian_hessian`` is
called, exactly as the driver slices linear-equality multipliers.

``A`` may be an explicit array or any :class:`~ipax.backend.operators.
LinearOperator`, and an operator is never densified: the public sparse
operators are rewritten at the triplet level (Array-API index/value arithmetic
only, invariant #4), a ``VStack`` is lowered block by block so per-block
capabilities such as the mixed-precision Gram hint survive, and any other
operator is wrapped in a signed row selection that folds the selection into
the Gram weights (see :func:`_lower_two_sided`). That keeps large sparse
inequality sets (``m ≫ n``, e.g. radiotherapy dose constraints) on the
condensed routes' sparse-Gram fast path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipax.backend.namespace import array_namespace, device
from ipax.backend.operators import (
    COOOperator,
    CSCOperator,
    CSROperator,
    Dense,
    LinearOperator,
    VStack,
    _signed_row_select_triplets,
    _SignedRowSelection,
    as_operator,
)
from ipax.problem.base import Problem

if TYPE_CHECKING:
    from ipax.typing import Array, Namespace, Scalar

# The public sparse operators whose triplets the lowering may rewrite directly.
# Exact types only: a user subclass may override the algebra (e.g. a custom
# ``gram``), so it is wrapped in a signed row selection instead, which forwards.
_TRIPLET_FAMILIES: tuple[type[LinearOperator], ...] = (
    COOOperator,
    CSROperator,
    CSCOperator,
)


def _row_masks(
    lower: Array, upper: Array, m_rows: int, xp: Namespace
) -> tuple[Array, Array]:
    """Validate the bounds of an ``m_rows``-row block; return the finite masks."""
    if int(lower.shape[0]) != m_rows or int(upper.shape[0]) != m_rows:
        raise ValueError("linear_ineq bounds must match the matrix row count")
    finite_l = xp.isfinite(lower)
    finite_u = xp.isfinite(upper)
    both = xp.logical_and(finite_l, finite_u)
    if bool(xp.any(xp.logical_and(both, lower > upper))):
        raise ValueError("linear_ineq has a row whose lower bound exceeds its upper")
    return finite_l, finite_u


def _lowered_offset(
    lower: Array, upper: Array, finite_l: Array, finite_u: Array, xp: Namespace
) -> Array:
    """``h`` of the lowered rows ``J x + h ≤ 0``: ``l_i`` on top of ``−u_i``."""
    return xp.concat(
        (
            xp.take(lower, xp.nonzero(finite_l)[0]),
            -xp.take(upper, xp.nonzero(finite_u)[0]),
        )
    )


def _empty_block(lower: Array, n_cols: int, xp: Namespace) -> tuple[Dense, Array]:
    """A zero-row lowered block (every row infinite on both sides)."""
    empty = xp.zeros_like(lower[:0])  # dtype and device of the bounds
    return Dense(xp.reshape(empty, (0, n_cols))), empty


def _lower_dense(
    a: Array, lower: Array, upper: Array, xp: Namespace
) -> tuple[Dense, Array]:
    """Explicit rank-2 ``A``: select and sign-flip rows by index into a ``Dense``."""
    if len(a.shape) != 2:
        raise ValueError("linear_ineq matrix must be rank-2")
    m_rows = int(a.shape[0])
    finite_l, finite_u = _row_masks(lower, upper, m_rows, xp)

    lower_idx = xp.nonzero(finite_l)[0]
    upper_idx = xp.nonzero(finite_u)[0]
    blocks: list[Array] = []
    offsets: list[Array] = []
    if int(lower_idx.shape[0]) > 0:
        blocks.append(-xp.take(a, lower_idx, axis=0))
        offsets.append(xp.take(lower, lower_idx))
    if int(upper_idx.shape[0]) > 0:
        blocks.append(xp.take(a, upper_idx, axis=0))
        offsets.append(-xp.take(upper, upper_idx))

    if not blocks:
        jac = xp.zeros((0, int(a.shape[1])), dtype=a.dtype)
        offset = xp.zeros((0,), dtype=a.dtype)
    else:
        jac = xp.concat(tuple(blocks), axis=0)
        offset = xp.concat(tuple(offsets))
    return Dense(jac), offset


def _lower_triplets(
    matrix: LinearOperator, lower: Array, upper: Array, xp: Namespace
) -> tuple[LinearOperator, Array]:
    """A public sparse operator: rewrite its COO triplets, same family out."""
    m_rows, n_cols = matrix.shape
    finite_l, finite_u = _row_masks(lower, upper, m_rows, xp)
    rows, cols, values, _ = matrix.to_coo()
    rows, cols, values, k = _signed_row_select_triplets(
        rows, cols, values, finite_l, finite_u
    )
    if k == 0:
        return _empty_block(lower, n_cols, xp)
    offset = _lowered_offset(lower, upper, finite_l, finite_u, xp)

    # The lowered pattern is fixed by the source pattern and the bound
    # finiteness, both constant for this operator's lifetime; derive a key only
    # when the source promised one (conservative otherwise). The trailing
    # identity token keeps two lowerings of the same source from aliasing
    # (the key is stable within a solve, where the one lowered operator is
    # reused; a re-solve lowers afresh and re-analyzes once).
    source_signature = matrix.coo_pattern_signature()
    pattern_key = (
        None
        if source_signature is None
        else ("linear_ineq_lowered", source_signature, object())
    )
    # ±1 row selection is exact, so the source's data-precision hint carries.
    hint = matrix.gram_accumulate_dtype_hint()
    shape = (k, n_cols)
    lowered: LinearOperator
    sxp = array_namespace(values)
    if isinstance(matrix, CSROperator):
        # Row-sorted in ⇒ row-sorted out (see _signed_row_select_triplets), so
        # the row pointer is a searchsorted of the row boundaries.
        boundaries = sxp.arange(k + 1, dtype=rows.dtype, device=device(rows))
        indptr = sxp.searchsorted(rows, boundaries)
        lowered = CSROperator(
            indptr, cols, values, shape, pattern_key=pattern_key, values_dtype_hint=hint
        )
    elif isinstance(matrix, CSCOperator):
        order = sxp.argsort(cols, stable=True)
        cols_sorted = sxp.take(cols, order)
        boundaries = sxp.arange(
            n_cols + 1, dtype=cols_sorted.dtype, device=device(cols_sorted)
        )
        indptr = sxp.searchsorted(cols_sorted, boundaries)
        lowered = CSCOperator(
            indptr,
            sxp.take(rows, order),
            sxp.take(values, order),
            shape,
            pattern_key=pattern_key,
            values_dtype_hint=hint,
        )
    else:
        lowered = COOOperator(
            rows, cols, values, shape, pattern_key=pattern_key, values_dtype_hint=hint
        )
    return lowered, offset


def _lower_wrapped(
    matrix: LinearOperator, lower: Array, upper: Array, xp: Namespace
) -> tuple[LinearOperator, Array]:
    """Any other operator: a signed row selection around it (no densify)."""
    m_rows, n_cols = matrix.shape
    finite_l, finite_u = _row_masks(lower, upper, m_rows, xp)
    op = _SignedRowSelection(matrix, finite_l, finite_u)
    if op.shape[0] == 0:
        return _empty_block(lower, n_cols, xp)
    return op, _lowered_offset(lower, upper, finite_l, finite_u, xp)


def _lower_stacked(
    matrix: VStack, lower: Array, upper: Array, xp: Namespace
) -> tuple[LinearOperator, Array]:
    """A ``VStack``: lower each block against its own bound slice, re-stack.

    Keeping the blocks separate preserves each block's own capabilities —
    notably its ``gram_accumulate_dtype_hint``, which ``VStack.gram`` honors
    block by block under ``gram_dtype="auto"`` — so a float32-sourced block
    still reduces its Gram accumulate after lowering. Lowered rows are ordered
    block by block (each block's lower rows, then its upper rows).
    """
    m_rows, n_cols = matrix.shape
    _row_masks(lower, upper, m_rows, xp)  # validate the whole block up front
    ops: list[LinearOperator] = []
    offsets: list[Array] = []
    start = 0
    for block in matrix.blocks:
        stop = start + block.shape[0]
        op, offset = _lower_two_sided(block, lower[start:stop], upper[start:stop], xp)
        start = stop
        if op.shape[0] == 0:
            continue
        ops.extend(op.blocks if isinstance(op, VStack) else (op,))
        offsets.append(offset)
    if not ops:
        return _empty_block(lower, n_cols, xp)
    if len(ops) == 1:
        return ops[0], offsets[0]
    return VStack(tuple(ops)), xp.concat(tuple(offsets))


def _lower_two_sided(
    matrix: Array | LinearOperator,
    lower: Array,
    upper: Array,
    xp: Namespace,
) -> tuple[LinearOperator, Array]:
    """Return ``(J, offset)`` so the lowered block is ``g(x) = J x + offset ≤ 0``.

    ``J`` stacks ``−A_i`` for every finite lower row over ``+A_i`` for every
    finite upper row (a range row appears in both halves; a row infinite on both
    sides is dropped), with ``offset`` holding ``l_i`` and ``−u_i`` to match.
    How ``J`` is built depends on the kind of ``A`` — none of them densifies an
    operator:

    * a rank-2 array (or a :class:`~ipax.backend.operators.Dense`, unwrapped)
      is sliced by index into a ``Dense`` block;
    * a :class:`~ipax.backend.operators.COOOperator`/``CSROperator``/
      ``CSCOperator`` is rewritten at the triplet level into a new operator of
      the same family (Array-API index/value arithmetic only — invariant #4),
      keeping its data-precision hint and deriving a pattern key from its own;
    * a :class:`~ipax.backend.operators.VStack` is lowered block by block and
      re-stacked, so per-block capabilities (precision hints) survive;
    * any other operator (matrix-free, or a user subclass) is wrapped in a
      signed row selection whose Gram capabilities fold the selection into the
      weights.
    """
    if type(matrix) is Dense:  # exact type: a subclass's overrides must survive
        matrix = matrix.dense_matrix()
    if not isinstance(matrix, LinearOperator):
        return _lower_dense(matrix, lower, upper, xp)
    if type(matrix) is VStack:  # exact type, as for Dense above
        return _lower_stacked(matrix, lower, upper, xp)
    if type(matrix) in _TRIPLET_FAMILIES:
        return _lower_triplets(matrix, lower, upper, xp)
    return _lower_wrapped(matrix, lower, upper, xp)


class LoweredLinearIneqProblem(Problem):
    """A :class:`Problem` with its two-sided ``linear_ineq`` block lowered.

    Wraps the resolved problem and appends the lowered, one-sided linear
    inequality rows after any nonlinear inequalities. The lowered block is any
    :class:`~ipax.backend.operators.LinearOperator` (``Dense``, a sparse
    operator, a ``VStack`` of them, or a signed row selection) and is evaluated
    through its ``matvec``, so it is never densified here. Provenance (``sources``,
    ``has_analytic_hessian``) and every non-inequality method delegate to the
    inner problem; :meth:`linear_ineq` now returns ``None`` so downstream layers
    (scaling, the driver) see a plain inequality problem.
    """

    def __init__(
        self,
        inner: Problem,
        lowered_jac: LinearOperator,
        offset: Array,
        n_nonlinear_ineq: int,
    ) -> None:
        self._inner = inner
        self._jac = lowered_jac
        self._offset = offset
        self._m_nonlinear = n_nonlinear_ineq
        # Forward the provenance the driver reads off the problem.
        self.sources = getattr(inner, "sources", None)
        self.has_analytic_hessian = getattr(inner, "has_analytic_hessian", True)

    @property
    def n_vars(self) -> int:
        return self._inner.n_vars

    def bounds(self) -> tuple[Array | None, Array | None]:
        return self._inner.bounds()

    def objective(self, x: Array) -> Scalar:
        return self._inner.objective(x)

    def gradient(self, x: Array) -> Array:
        return self._inner.gradient(x)

    def eq_constraints(self, x: Array) -> Array:
        return self._inner.eq_constraints(x)

    def eq_jacobian(self, x: Array) -> Array | LinearOperator:
        return self._inner.eq_jacobian(x)

    def linear_eq(self) -> tuple[Array | LinearOperator, Array] | None:
        return self._inner.linear_eq()

    def linear_ineq(self) -> None:
        return None  # already lowered into the inequality block

    def ineq_constraints(self, x: Array) -> Array:
        lin = self._jac.matvec(x) + self._offset
        if self._m_nonlinear == 0:
            return lin
        xp = array_namespace(x)
        return xp.concat((self._inner.ineq_constraints(x), lin))

    def ineq_jacobian(self, x: Array) -> LinearOperator:
        if self._m_nonlinear == 0:
            return self._jac
        # Flatten a stacked lowered block into the stack so each of its blocks
        # (and its own precision hint) sits at the top level beside the
        # nonlinear rows, which lead so their multipliers come first.
        lowered = self._jac.blocks if isinstance(self._jac, VStack) else (self._jac,)
        return VStack((as_operator(self._inner.ineq_jacobian(x)), *lowered))

    def lagrangian_hessian(
        self,
        x: Array,
        y_eq: Array,
        y_ineq: Array,
        sigma: Scalar = 1.0,
    ) -> Array | LinearOperator:
        # The lowered block is affine ⇒ no Hessian term; pass the inner problem
        # only its own (nonlinear) inequality multipliers.
        return self._inner.lagrangian_hessian(
            x, y_eq, y_ineq[: self._m_nonlinear], sigma
        )


def lower_linear_inequalities(problem: Problem, x0: Array, xp: Namespace) -> Problem:
    """Return a problem with any two-sided ``linear_ineq`` block lowered.

    A no-op (returns ``problem`` unchanged) when the problem declares no
    ``linear_ineq``. The starting point ``x0`` is used only to count existing
    nonlinear inequalities so the lowered multipliers can be sliced off before
    the inner Lagrangian Hessian.
    """
    data = problem.linear_ineq()
    if data is None:
        return problem
    matrix, lower, upper = data
    lowered_jac, offset = _lower_two_sided(matrix, lower, upper, xp)
    try:
        n_nonlinear = int(problem.ineq_constraints(x0).shape[0])
    except NotImplementedError:
        n_nonlinear = 0
    return LoweredLinearIneqProblem(problem, lowered_jac, offset, n_nonlinear)


__all__ = ["LoweredLinearIneqProblem", "lower_linear_inequalities"]
