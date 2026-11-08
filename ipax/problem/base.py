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

"""The user-facing ``Problem`` abstract base class.

Capability-graded: only ``n_vars`` and ``objective`` are required. Everything
else is optional and resolved by autodiff or finite differences when absent.
Linear constraints are declared separately from nonlinear ones (constant
Jacobian, no Hessian term) — a real performance lever at RT scale.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipax.backend.operators import LinearOperator
    from ipax.typing import Array, Scalar


class Problem(ABC):
    """User-facing NLP definition.

    Subclass and implement :attr:`n_vars` and :meth:`objective`; everything else
    is optional. **Overriding an optional method declares that the problem
    provides it**; a method left as inherited is filled by the solver: the
    gradient and nonlinear Jacobians by *analytic → autodiff →
    finite-difference*, the Lagrangian Hessian by the analytic one when
    supplied, otherwise L-BFGS (autodiff Hessian-vector products only with
    ``Options(hessian="autodiff-hvp")``). A class that overrides methods it
    cannot always serve (a delegating wrapper, say) declares what it really
    provides through :meth:`provides`. Constraint *values*
    (:meth:`eq_constraints`, :meth:`ineq_constraints`) have no fallback: the
    constraint exists only if its value method is implemented.

    New optional capabilities arrive as new methods with defaults, never as
    new parameters of existing methods, so a subclass written against one
    release keeps working.

    Nonlinear and linear constraints are declared through separate methods so
    the constant-data linear blocks (:meth:`linear_eq`, :meth:`linear_ineq`) can
    be assembled once with no Hessian contribution — a real performance lever at
    RT scale. Jacobians and the Hessian may be returned either as dense
    Array-API arrays or as :class:`~ipax.backend.operators.LinearOperator`
    instances, so a structured or matrix-free model never needs to materialize a
    matrix.

    All arrays are read in whatever Array-API backend ``x`` carries; the class
    never imports a concrete array library. See
    :class:`~ipax.problem.function.FunctionProblem`,
    :class:`~ipax.problem.function.QuadraticProblem`, and
    :class:`~ipax.problem.function.LinearProblem` for ready-made implementations.
    """

    # ---- dimensions & bounds (required: n_vars; bounds default to ±inf) ----
    @property
    @abstractmethod
    def n_vars(self) -> int:
        """Number of optimization variables."""

    def bounds(self) -> tuple[Array | None, Array | None]:
        """``(x_L, x_U)``; ``None`` entries → ∓∞. Default: unbounded."""
        return (None, None)

    # ---- objective (required) ----
    @abstractmethod
    def objective(self, x: Array) -> Scalar:
        """Scalar objective ``f(x)``."""

    # ---- which optional methods are implemented ----
    def provides(self) -> frozenset[str] | None:
        """Names of the optional methods this problem implements, or ``None``.

        ``None`` (the default) infers them from which methods the subclass
        overrides. Return a set to declare them explicitly, e.g.
        ``frozenset({"gradient", "ineq_constraints"})`` for a wrapper that
        overrides every method but only delegates some; a method not listed is
        derived (or, for constraint values, absent) even if overridden. Valid
        names: ``gradient``, ``eq_constraints``, ``eq_jacobian``,
        ``ineq_constraints``, ``ineq_jacobian``, ``lagrangian_hessian``.
        """
        return None

    # ---- objective gradient (optional → autodiff → finite-diff) ----
    def gradient(self, x: Array) -> Array:
        """``∇f(x)``. Derived by autodiff or finite differences when not provided."""
        raise NotImplementedError

    # ---- NONLINEAR constraints (optional; present only if defined) ----
    def eq_constraints(self, x: Array) -> Array:
        """``c(x) = 0``."""
        raise NotImplementedError

    def eq_jacobian(self, x: Array) -> Array | LinearOperator:
        """``∇c(x)``. Optional → autodiff → finite-diff."""
        raise NotImplementedError

    def ineq_constraints(self, x: Array) -> Array:
        """``g(x) ≤ 0``."""
        raise NotImplementedError

    def ineq_jacobian(self, x: Array) -> Array | LinearOperator:
        """``∇g(x)``. Optional → autodiff → finite-diff."""
        raise NotImplementedError

    # ---- LINEAR constraints (optional; constant data, declared once) ----
    def linear_eq(self) -> tuple[Array | LinearOperator, Array] | None:
        """``(A_eq, b_eq)`` for ``A_eq x = b_eq``."""
        return None

    def linear_ineq(
        self,
    ) -> tuple[Array | LinearOperator, Array, Array] | None:
        """``(A_ineq, l, u)`` for ``l ≤ A_ineq x ≤ u`` (l/u may be ∓inf).

        ``A_ineq`` may be a rank-2 array or any
        :class:`~ipax.backend.operators.LinearOperator`. The solver lowers the
        block once into one-sided rows (a row finite on both sides becomes two
        rows; a row infinite on both sides is dropped) without densifying an
        operator: ``COOOperator``/``CSROperator``/``CSCOperator`` are rewritten
        at the triplet level (same family, precision hint kept, pattern key
        derived from the source's), a ``VStack`` block by block (so each block
        keeps its own ``gram_accumulate_dtype_hint``), and any other operator
        through a signed row-selection wrapper whose Gram capabilities fold
        into the weights. A matrix-free ``A_ineq`` needs ``rmatvec``. The
        lowered rows follow any nonlinear inequalities, and their multipliers
        are sliced off before :meth:`lagrangian_hessian` is called.

        Layout of the lowered rows (and so of their entries in
        ``Result.y_ineq``, after the nonlinear ones): all finite-lower rows in
        source order, then all finite-upper rows in source order — except for
        a ``VStack``, which is laid out block by block (each block's lower
        rows, then its upper rows).
        """
        return None

    # ---- Hessian of the Lagrangian (optional → autodiff-HVP → L-BFGS) ----
    def lagrangian_hessian(
        self,
        x: Array,
        y_eq: Array,
        y_ineq: Array,
        sigma: Scalar = 1.0,
    ) -> Array | LinearOperator:
        """``W = σ∇²f + Σ y_eq·∇²c + Σ y_ineq·∇²g``, as matrix or matvec operator.

        Follows the IPOPT ``eval_h`` convention so a structured/analytic Hessian
        can be supplied as a cheap matvec without ever materializing ``W``.

        The solver always calls it with four positional arguments, so the
        parameter names are free. ``sigma`` is a Python float that may differ
        from 1 — it does whenever the default gradient-based scaling rescales
        the objective (‖∇f(x0)‖∞ > ``ScalingOptions.max_gradient``) — so an
        implementation must always scale the objective term by it. ``y_eq`` and ``y_ineq`` hold the multipliers
        of the *nonlinear* constraints only (linear blocks have no curvature)
        and are empty arrays, not ``None``, when there are none.
        """
        raise NotImplementedError


__all__ = ["Problem"]
