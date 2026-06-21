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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipax.backend.namespace import array_namespace
from ipax.backend.operators import Dense, LinearOperator, VStack, as_operator
from ipax.problem.base import Problem

if TYPE_CHECKING:
    from ipax.typing import Array, Namespace, Scalar


def _lower_two_sided(
    matrix: Array | LinearOperator,
    lower: Array,
    upper: Array,
    xp: Namespace,
) -> tuple[Dense, Array]:
    """Return ``(J, offset)`` so the lowered block is ``g(x) = J x + offset ≤ 0``.

    Only an explicit (dense/array) ``A`` is supported: the lowered rows are
    selected and sign-flipped by index, which a generic matrix-free operator
    cannot express. Matrix-free two-sided inequalities should be declared through
    :meth:`~ipax.problem.base.Problem.ineq_constraints` directly.
    """
    if isinstance(matrix, LinearOperator):
        raise NotImplementedError(
            "matrix-free two-sided linear inequalities are not supported; declare "
            "them through Problem.ineq_constraints/ineq_jacobian instead"
        )
    a = matrix
    if len(a.shape) != 2:
        raise ValueError("linear_ineq matrix must be rank-2")
    m_rows = int(a.shape[0])
    if int(lower.shape[0]) != m_rows or int(upper.shape[0]) != m_rows:
        raise ValueError("linear_ineq bounds must match the matrix row count")

    finite_l = xp.isfinite(lower)
    finite_u = xp.isfinite(upper)
    both = xp.logical_and(finite_l, finite_u)
    if bool(xp.any(xp.logical_and(both, lower > upper))):
        raise ValueError("linear_ineq has a row whose lower bound exceeds its upper")

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


class LoweredLinearIneqProblem(Problem):
    """A :class:`Problem` with its two-sided ``linear_ineq`` block lowered.

    Wraps the resolved problem and appends the lowered, one-sided linear
    inequality rows after any nonlinear inequalities. Provenance (``sources``,
    ``has_analytic_hessian``) and every non-inequality method delegate to the
    inner problem; :meth:`linear_ineq` now returns ``None`` so downstream layers
    (scaling, the driver) see a plain inequality problem.
    """

    def __init__(
        self,
        inner: Problem,
        lowered_jac: Dense,
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
        return VStack((as_operator(self._inner.ineq_jacobian(x)), self._jac))

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
