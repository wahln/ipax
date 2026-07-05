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

"""Gradient-based NLP auto-scaling (IPOPT §3.8).

A :class:`ScaledProblem` is a thin :class:`~ipax.problem.base.Problem` adapter
that rescales the objective and each constraint by a constant computed **once**
at the starting point, so their gradients have an ∞-norm of at most
``max_gradient`` (Wächter & Biegler 2006, §3.8: factor ``min(1, g_max/‖∇·‖∞)``).
Variables and bounds are left unscaled, matching IPOPT's ``gradient-based``
method. The IPM driver runs unchanged on the scaled problem (invariant #3); the
returned solution is unscaled back to the original problem in
:func:`ipax.solve.solve`.

Explicit dense and sparse Jacobians expose row norms in one operation. A
matrix-free Jacobian may provide the optional ``row_inf_norms`` capability;
without it, constraint rows are left unscaled rather than paying for one full
adjoint application per row.

Algebra (so the unscaling is exact). With objective factor ``s_f`` and
constraint factors ``D_c = diag(d_eq)``, ``D_g = diag(d_ineq)``, the scaled
problem is ``min s_f·f`` s.t. ``D_c·c = 0``, ``D_g·g ≤ 0`` (bounds unchanged).
Its stationarity ``s_f·∇f + ∇cᵀ D_c ỹ + ∇gᵀ D_g λ̃ − z̃_L + z̃_U = 0`` divided by
``s_f`` recovers the original multipliers:

    y_eq   = d_eq   · ỹ / s_f
    y_ineq = d_ineq · λ̃ / s_f
    z      =          z̃ / s_f
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ipax.backend.namespace import array_namespace
from ipax.backend.operators import LinearOperator, as_operator
from ipax.problem.base import Problem

if TYPE_CHECKING:
    from ipax.typing import Array, Namespace, Scalar


class _RowScaled(LinearOperator):
    """``diag(d) @ J`` — scales the rows of a Jacobian operator by ``d``.

    Forwards the optional structure-exposing capabilities (``to_coo`` for the
    sparse-direct route, ``gram_diagonal``/``row_gram_diagonal`` for the Krylov
    preconditioners) so row-scaling never degrades the solver routes.
    """

    def __init__(self, jac: LinearOperator, d: Array) -> None:
        self._jac = jac
        self._d = d

    @property
    def shape(self) -> tuple[int, int]:
        return self._jac.shape

    def matvec(self, v: Array) -> Array:
        return self._d * self._jac.matvec(v)

    def rmatvec(self, v: Array) -> Array:
        return self._jac.rmatvec(self._d * v)

    def matmat(self, V: Array) -> Array:
        xp = array_namespace(self._d, V)
        return xp.expand_dims(self._d, axis=1) * self._jac.matmat(V)

    def gram_diagonal(self, weights: Array) -> Array:
        # diag((DJ)ᵀ W (DJ))_k = Σ_i W_i d_i² J_ik² = J.gram_diagonal(d²·W).
        return self._jac.gram_diagonal(self._d * self._d * weights)

    def row_gram_diagonal(self, weights: Array) -> Array:
        # diag((DJ) W (DJ)ᵀ)_j = d_j² Σ_k W_k J_jk² = d²·J.row_gram_diagonal(W).
        return self._d * self._d * self._jac.row_gram_diagonal(weights)

    def row_inf_norms(self, like: Array | None = None) -> Array:
        xp = array_namespace(self._d)
        return xp.abs(self._d) * self._jac.row_inf_norms(like)

    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        rows, cols, values, shape = self._jac.to_coo(like)
        xp = array_namespace(values)
        return rows, cols, values * xp.take(self._d, rows), shape

    def coo_pattern_signature(self) -> object | None:
        signature = self._jac.coo_pattern_signature()
        if signature is None:
            return None
        return ("row_scaled", self.shape, signature)


@dataclass(frozen=True, slots=True)
class ProblemScaling:
    """Constant scale factors computed once at the starting point.

    ``obj`` is the scalar objective factor ``s_f``. ``eq``/``ineq``/``linear_eq``
    are per-row constraint factor vectors (``None`` when that block is absent).
    ``combined_eq`` concatenates ``eq`` then ``linear_eq`` in the driver's
    equality ordering (nonlinear stacked before linear) for unscaling ``y_eq``.
    """

    obj: float
    eq: Array | None
    ineq: Array | None
    linear_eq: Array | None
    combined_eq: Array | None


def _scalar_factor(xp: Namespace, grad: Array, max_gradient: float) -> float:
    """``min(1, g_max/‖grad‖∞)``; identity when the gradient is tiny."""
    if int(grad.shape[0]) == 0:
        return 1.0
    norm = float(xp.max(xp.abs(grad)))
    if norm <= max_gradient:
        return 1.0
    return max_gradient / norm


def _vector_factors(
    xp: Namespace,
    jac: LinearOperator | None,
    max_gradient: float,
    dtype: object,
    like: Array,
) -> Array | None:
    """Per-row factors, or identity when cheap row norms are unavailable."""
    if jac is None:
        return None
    m = jac.shape[0]
    if m == 0:
        return xp.zeros((0,), dtype=dtype)
    try:
        row_norms = jac.row_inf_norms(like)
    except NotImplementedError:
        # Avoid m full adjoint applications for a matrix-free m-by-n Jacobian.
        return xp.ones((m,), dtype=dtype)
    safe = xp.where(row_norms > 0.0, row_norms, xp.ones_like(row_norms))
    ones = xp.ones_like(safe)
    return xp.minimum(ones, max_gradient / safe)


def compute_scaling(
    problem: Problem,
    x0: Array,
    xp: Namespace,
    *,
    has_eq: bool,
    has_ineq: bool,
    max_gradient: float,
) -> ProblemScaling:
    """Gradient-based scale factors at ``x0`` (Wächter & Biegler 2006, §3.8)."""
    dtype = x0.dtype
    obj = _scalar_factor(xp, problem.gradient(x0), max_gradient)

    eq = (
        _vector_factors(
            xp, as_operator(problem.eq_jacobian(x0)), max_gradient, dtype, x0
        )
        if has_eq
        else None
    )
    ineq = (
        _vector_factors(
            xp, as_operator(problem.ineq_jacobian(x0)), max_gradient, dtype, x0
        )
        if has_ineq
        else None
    )
    linear = problem.linear_eq()
    linear_eq = (
        _vector_factors(xp, as_operator(linear[0]), max_gradient, dtype, x0)
        if linear is not None
        else None
    )

    parts = [p for p in (eq, linear_eq) if p is not None]
    if not parts:
        combined_eq = None
    elif len(parts) == 1:
        combined_eq = parts[0]
    else:
        combined_eq = xp.concat(tuple(parts))

    return ProblemScaling(
        obj=obj, eq=eq, ineq=ineq, linear_eq=linear_eq, combined_eq=combined_eq
    )


class ScaledProblem(Problem):
    """A :class:`Problem` rescaled by constant :class:`ProblemScaling` factors.

    Wraps the *resolved* problem so every derivative is already bound; the
    driver consumes this exactly like a plain resolved problem. Provenance
    (``sources``, ``has_analytic_hessian``) is forwarded unchanged.
    """

    def __init__(self, inner: Problem, scaling: ProblemScaling) -> None:
        self._inner = inner
        self._scaling = scaling
        # Forward derivative provenance the driver reads off the problem.
        self.sources = getattr(inner, "sources", None)
        self.has_analytic_hessian = getattr(inner, "has_analytic_hessian", True)

    @property
    def scaling(self) -> ProblemScaling:
        return self._scaling

    @property
    def n_vars(self) -> int:
        return self._inner.n_vars

    def bounds(self) -> tuple[Array | None, Array | None]:
        return self._inner.bounds()  # variables/bounds are not scaled

    def objective(self, x: Array) -> Scalar:
        return self._scaling.obj * self._inner.objective(x)

    def gradient(self, x: Array) -> Array:
        return self._scaling.obj * self._inner.gradient(x)

    def eq_constraints(self, x: Array) -> Array:
        return self._scaling.eq * self._inner.eq_constraints(x)

    def eq_jacobian(self, x: Array) -> LinearOperator:
        return _RowScaled(as_operator(self._inner.eq_jacobian(x)), self._scaling.eq)

    def ineq_constraints(self, x: Array) -> Array:
        return self._scaling.ineq * self._inner.ineq_constraints(x)

    def ineq_jacobian(self, x: Array) -> LinearOperator:
        return _RowScaled(as_operator(self._inner.ineq_jacobian(x)), self._scaling.ineq)

    def linear_eq(self) -> tuple[LinearOperator, Array] | None:
        data = self._inner.linear_eq()
        if data is None:
            return None
        matrix, rhs = data
        d = self._scaling.linear_eq
        assert d is not None
        return _RowScaled(as_operator(matrix), d), d * rhs

    def linear_ineq(self) -> tuple[Array | LinearOperator, Array, Array] | None:
        return self._inner.linear_ineq()

    def lagrangian_hessian(
        self,
        x: Array,
        y_eq: Array,
        y_ineq: Array,
        sigma: Scalar = 1.0,
    ) -> Array | LinearOperator:
        # Scaled Lagrangian Hessian: the objective term carries s_f·σ and each
        # constraint curvature term carries its row factor, so feed the inner
        # Hessian the scaled multipliers. ``y_eq`` is the nonlinear block only
        # (the driver slices linear equalities away — they have no Hessian term).
        scaling = self._scaling
        if scaling.eq is not None:
            y_eq = scaling.eq * y_eq
        if scaling.ineq is not None:
            y_ineq = scaling.ineq * y_ineq
        return self._inner.lagrangian_hessian(
            x, y_eq, y_ineq, sigma=scaling.obj * float(sigma)
        )


__all__ = ["ProblemScaling", "ScaledProblem", "compute_scaling"]
