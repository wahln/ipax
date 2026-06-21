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

"""Convenience ``Problem`` implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipax.backend.namespace import array_namespace
from ipax.backend.operators import LinearOperator, as_operator
from ipax.problem.base import Problem

if TYPE_CHECKING:
    from collections.abc import Callable

    from ipax.typing import Array, Scalar


class FunctionProblem(Problem):
    """Assemble a :class:`Problem` from callables without subclassing.

    A thin adapter for the common case where the model is already expressed as
    functions of ``x``. Pass at least ``n_vars`` and ``objective``; every other
    derivative is optional and, when omitted, supplied by the resolver's
    autodiff → finite-difference fallback (the Hessian falls back to L-BFGS).
    Whether a callable was supplied is recorded explicitly, so the resolver
    treats a missing callable exactly as a hand-written subclass that leaves the
    corresponding method unimplemented.

    Examples
    --------
    >>> import numpy as np
    >>> from ipax import FunctionProblem, solve
    >>> problem = FunctionProblem(
    ...     n_vars=2,
    ...     objective=lambda x: np.sum(x**2),
    ...     gradient=lambda x: 2 * x,
    ... )
    >>> result = solve(problem, np.ones(2))
    """

    def __init__(
        self,
        n_vars: int,
        objective: Callable[[Array], Scalar],
        *,
        gradient: Callable[[Array], Array] | None = None,
        bounds: tuple[Array | None, Array | None] = (None, None),
        eq_constraints: Callable[[Array], Array] | None = None,
        eq_jacobian: Callable[[Array], Array | LinearOperator] | None = None,
        ineq_constraints: Callable[[Array], Array] | None = None,
        ineq_jacobian: Callable[[Array], Array | LinearOperator] | None = None,
        linear_eq: tuple[Array | LinearOperator, Array] | None = None,
        linear_ineq: tuple[Array | LinearOperator, Array, Array] | None = None,
        lagrangian_hessian: (
            Callable[[Array, Array, Array, Scalar], Array | LinearOperator] | None
        ) = None,
    ) -> None:
        """Build a problem from objective and optional derivative callables.

        Parameters
        ----------
        n_vars
            Number of optimization variables (must be non-negative).
        objective
            Scalar objective ``f(x)``.
        gradient
            ``∇f(x)``. Derived by autodiff/finite-difference if omitted.
        bounds
            ``(x_L, x_U)`` variable bounds; ``None`` on either side means that
            side is unbounded. Defaults to fully unbounded.
        eq_constraints, eq_jacobian
            Nonlinear equalities ``c(x) = 0`` and their Jacobian ``∇c(x)``. The
            Jacobian may be a dense array or a
            :class:`~ipax.backend.operators.LinearOperator`; it is derived if
            omitted.
        ineq_constraints, ineq_jacobian
            Nonlinear inequalities ``g(x) ≤ 0`` and their Jacobian ``∇g(x)``,
            same conventions as the equality pair.
        linear_eq
            ``(A_eq, b_eq)`` declaring the constant-data equalities
            ``A_eq x = b_eq``. Assembled once; carries no Hessian term.
        linear_ineq
            ``(A_ineq, l, u)`` declaring ``l ≤ A_ineq x ≤ u`` (``l``/``u`` may
            be ``∓inf``).
        lagrangian_hessian
            ``W(x, y_eq, y_ineq, σ)`` following the IPOPT ``eval_h`` convention.
            Returns a dense array or matvec operator; falls back to autodiff-HVP
            then L-BFGS when omitted.

        Raises
        ------
        ValueError
            If ``n_vars`` is negative.
        """
        if n_vars < 0:
            raise ValueError("n_vars must be non-negative")
        self._n_vars = n_vars
        self._objective = objective
        self._gradient = gradient
        # FunctionProblem overrides every optional method (delegating to a
        # callable that may be None), so the derivative resolver cannot detect
        # what is "provided" by class-method comparison. Record it explicitly.
        provided = {"objective"}
        if gradient is not None:
            provided.add("gradient")
        if eq_constraints is not None:
            provided.add("eq_constraints")
        if eq_jacobian is not None:
            provided.add("eq_jacobian")
        if ineq_constraints is not None:
            provided.add("ineq_constraints")
        if ineq_jacobian is not None:
            provided.add("ineq_jacobian")
        if lagrangian_hessian is not None:
            provided.add("lagrangian_hessian")
        self._provided: frozenset[str] = frozenset(provided)
        self._bounds = bounds
        self._eq_constraints = eq_constraints
        self._eq_jacobian = eq_jacobian
        self._ineq_constraints = ineq_constraints
        self._ineq_jacobian = ineq_jacobian
        self._linear_eq = linear_eq
        self._linear_ineq = linear_ineq
        self._lagrangian_hessian = lagrangian_hessian

    @property
    def n_vars(self) -> int:
        return self._n_vars

    def bounds(self) -> tuple[Array | None, Array | None]:
        return self._bounds

    def objective(self, x: Array) -> Scalar:
        return self._objective(x)

    def gradient(self, x: Array) -> Array:
        if self._gradient is None:
            raise NotImplementedError
        return self._gradient(x)

    def eq_constraints(self, x: Array) -> Array:
        if self._eq_constraints is None:
            raise NotImplementedError
        return self._eq_constraints(x)

    def eq_jacobian(self, x: Array) -> Array | LinearOperator:
        if self._eq_jacobian is None:
            raise NotImplementedError
        return self._eq_jacobian(x)

    def ineq_constraints(self, x: Array) -> Array:
        if self._ineq_constraints is None:
            raise NotImplementedError
        return self._ineq_constraints(x)

    def ineq_jacobian(self, x: Array) -> Array | LinearOperator:
        if self._ineq_jacobian is None:
            raise NotImplementedError
        return self._ineq_jacobian(x)

    def linear_eq(self) -> tuple[Array | LinearOperator, Array] | None:
        return self._linear_eq

    def linear_ineq(self) -> tuple[Array | LinearOperator, Array, Array] | None:
        return self._linear_ineq

    def lagrangian_hessian(
        self,
        x: Array,
        y_eq: Array,
        y_ineq: Array,
        sigma: Scalar = 1.0,
    ) -> Array | LinearOperator:
        if self._lagrangian_hessian is None:
            raise NotImplementedError
        return self._lagrangian_hessian(x, y_eq, y_ineq, sigma)


class QuadraticProblem(Problem):
    """``min 0.5 xᵀ Q x + cᵀ x`` with a constant Hessian ``Q``.

    The objective gradient (``Q x + c``) and Lagrangian Hessian (``Q``) are
    exact, so no derivative fallback is engaged. The problem is unconstrained on
    its own; compose it with bounds or constraints through a subclass, or use
    :class:`FunctionProblem` when constraints are also needed.

    Parameters
    ----------
    Q
        Symmetric ``(n_vars, n_vars)`` Hessian, as a dense array or a
        :class:`~ipax.backend.operators.LinearOperator` (for a matrix-free
        quadratic). A non-unit ``σ`` is only supported for the dense form.
    c
        Rank-1 linear term of length ``n_vars``.

    Raises
    ------
    ValueError
        If ``c`` is not rank-1 or ``Q`` is not square of shape
        ``(n_vars, n_vars)``.
    """

    def __init__(self, Q: Array | LinearOperator, c: Array) -> None:
        if len(c.shape) != 1:
            raise ValueError("linear term c must be a rank-1 array")
        self._Q = Q
        self._Q_op = as_operator(Q)
        self._c = c
        self._n_vars = int(c.shape[0])
        if self._Q_op.shape != (self._n_vars, self._n_vars):
            raise ValueError("Q must be square with shape (n_vars, n_vars)")

    @property
    def n_vars(self) -> int:
        return self._n_vars

    def objective(self, x: Array) -> Scalar:
        xp = array_namespace(x, self._c)
        Qx = self._Q_op.matvec(x)
        return 0.5 * xp.sum(x * Qx) + xp.sum(self._c * x)

    def gradient(self, x: Array) -> Array:
        return self._Q_op.matvec(x) + self._c

    def lagrangian_hessian(
        self,
        x: Array,
        y_eq: Array,
        y_ineq: Array,
        sigma: Scalar = 1.0,
    ) -> Array | LinearOperator:
        del x, y_eq, y_ineq
        if isinstance(self._Q, LinearOperator):
            if float(sigma) != 1.0:
                raise NotImplementedError("scaled operator Hessians are not supported")
            return self._Q
        return sigma * self._Q


class LinearProblem(Problem):
    """``min cᵀ x`` (a linear program) with a zero Hessian.

    Bounds and linear constraints are the only way to make the problem
    well-posed, so they are accepted directly in the constructor. The gradient
    is the constant ``c`` and the Hessian is identically zero.

    Parameters
    ----------
    c
        Rank-1 cost vector of length ``n_vars``.
    bounds
        ``(x_L, x_U)`` variable bounds; ``None`` on either side means unbounded.
    linear_eq
        ``(A_eq, b_eq)`` for ``A_eq x = b_eq``.
    linear_ineq
        ``(A_ineq, l, u)`` for ``l ≤ A_ineq x ≤ u`` (``l``/``u`` may be ``∓inf``).

    Raises
    ------
    ValueError
        If ``c`` is not rank-1.
    """

    def __init__(
        self,
        c: Array,
        *,
        bounds: tuple[Array | None, Array | None] = (None, None),
        linear_eq: tuple[Array | LinearOperator, Array] | None = None,
        linear_ineq: tuple[Array | LinearOperator, Array, Array] | None = None,
    ) -> None:
        if len(c.shape) != 1:
            raise ValueError("linear term c must be a rank-1 array")
        self._c = c
        self._bounds = bounds
        self._linear_eq = linear_eq
        self._linear_ineq = linear_ineq
        self._n_vars = int(c.shape[0])

    @property
    def n_vars(self) -> int:
        return self._n_vars

    def bounds(self) -> tuple[Array | None, Array | None]:
        return self._bounds

    def objective(self, x: Array) -> Scalar:
        xp = array_namespace(x, self._c)
        return xp.sum(self._c * x)

    def gradient(self, x: Array) -> Array:
        del x
        return self._c

    def linear_eq(self) -> tuple[Array | LinearOperator, Array] | None:
        return self._linear_eq

    def linear_ineq(self) -> tuple[Array | LinearOperator, Array, Array] | None:
        return self._linear_ineq

    def lagrangian_hessian(
        self,
        x: Array,
        y_eq: Array,
        y_ineq: Array,
        sigma: Scalar = 1.0,
    ) -> Array:
        del y_eq, y_ineq, sigma
        xp = array_namespace(x, self._c)
        return xp.zeros((self._n_vars, self._n_vars), dtype=x.dtype)


__all__ = ["FunctionProblem", "LinearProblem", "QuadraticProblem"]
