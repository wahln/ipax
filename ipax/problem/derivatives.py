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

"""Derivative-precedence resolver.

Inspects which ``Problem`` methods are provided (``Problem.provides()``, else
which are overridden) and what the backend supports,
then binds each derivative to a concrete source:

- gradient / nonlinear Jacobians: analytic → autodiff → finite-difference
- Lagrangian Hessian: analytic operator → L-BFGS (autodiff-HVP on request)

The chosen sources are recorded in :class:`~ipax.result.DerivativeSources`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipax._deprecation import warn_deprecated
from ipax.errors import (
    DerivativeUnavailableError,
    FiniteDifferenceWarning,
    ProblemDefinitionError,
    warn_external,
)
from ipax.problem.autodiff import get_autodiff_adapter
from ipax.problem.base import Problem
from ipax.problem.finitediff import gradient_fd, jacobian_fd
from ipax.result import DerivativeSources

if TYPE_CHECKING:
    from collections.abc import Callable

    from ipax.backend.operators import LinearOperator
    from ipax.options import Options
    from ipax.typing import Array, Namespace, Scalar


_OPTIONAL_METHODS = frozenset(
    {
        "gradient",
        "eq_constraints",
        "eq_jacobian",
        "ineq_constraints",
        "ineq_jacobian",
        "lagrangian_hessian",
    }
)


def _declared(problem: Problem) -> frozenset[str] | None:
    """The method names ``problem`` declares through :meth:`Problem.provides`."""
    declared = problem.provides()
    if declared is None:
        legacy = getattr(problem, "_provided", None)
        if legacy is None:
            return None
        # Before 0.14 the resolver read this private attribute; honour it for
        # one more release so delegating wrappers that set it keep working.
        warn_deprecated(
            f"{type(problem).__name__}._provided",
            "Problem.provides()",
            since="0.14",
            remove_in="0.15",
            category=DeprecationWarning,
        )
        declared = frozenset(legacy) - {"objective"}
    unknown = set(declared) - _OPTIONAL_METHODS - {"objective"}
    if unknown:
        raise ProblemDefinitionError(
            f"{type(problem).__name__}.provides() names unknown methods "
            f"{sorted(unknown)}; valid names are {sorted(_OPTIONAL_METHODS)}"
        )
    return frozenset(declared)


def _provides(problem: Problem, declared: frozenset[str] | None, name: str) -> bool:
    """Whether ``problem`` supplies ``name`` rather than leaving it to the solver.

    An explicit :meth:`~ipax.problem.base.Problem.provides` set decides;
    otherwise overriding the method counts as providing it.
    """
    if declared is not None:
        return name in declared
    return getattr(type(problem), name) is not getattr(Problem, name)


class ResolvedProblem(Problem):
    """A ``Problem`` with all derivatives bound to concrete callables/operators.

    The IPM driver consumes this, never the raw ``Problem``: it always has a
    gradient, the Jacobians it needs, and a record of how each was obtained.
    Constraint *values* and bounds delegate straight to the underlying problem;
    only the derivatives are rebound.
    """

    def __init__(
        self,
        problem: Problem,
        *,
        gradient: Callable[[Array], Array],
        eq_jacobian: Callable[[Array], Array | LinearOperator] | None,
        ineq_jacobian: Callable[[Array], Array | LinearOperator] | None,
        sources: DerivativeSources,
        has_analytic_hessian: bool,
    ) -> None:
        self._problem = problem
        self._gradient_fn = gradient
        self._eq_jacobian_fn = eq_jacobian
        self._ineq_jacobian_fn = ineq_jacobian
        self.sources = sources
        self.has_analytic_hessian = has_analytic_hessian

    @property
    def n_vars(self) -> int:
        return self._problem.n_vars

    def bounds(self) -> tuple[Array | None, Array | None]:
        return self._problem.bounds()

    def objective(self, x: Array) -> Scalar:
        return self._problem.objective(x)

    def gradient(self, x: Array) -> Array:
        return self._gradient_fn(x)

    def eq_constraints(self, x: Array) -> Array:
        return self._problem.eq_constraints(x)

    def eq_jacobian(self, x: Array) -> Array | LinearOperator:
        if self._eq_jacobian_fn is None:
            raise NotImplementedError
        return self._eq_jacobian_fn(x)

    def ineq_constraints(self, x: Array) -> Array:
        return self._problem.ineq_constraints(x)

    def ineq_jacobian(self, x: Array) -> Array | LinearOperator:
        if self._ineq_jacobian_fn is None:
            raise NotImplementedError
        return self._ineq_jacobian_fn(x)

    def linear_eq(self) -> tuple[Array | LinearOperator, Array] | None:
        return self._problem.linear_eq()

    def linear_ineq(self) -> tuple[Array | LinearOperator, Array, Array] | None:
        return self._problem.linear_ineq()

    def lagrangian_hessian(
        self,
        x: Array,
        y_eq: Array,
        y_ineq: Array,
        sigma: Scalar = 1.0,
    ) -> Array | LinearOperator:
        # Analytic passthrough only; quasi-Newton / autodiff-HVP Hessians are
        # assembled by the driver, which owns the persistent L-BFGS history.
        return self._problem.lagrangian_hessian(x, y_eq, y_ineq, sigma)


def _resolve_gradient(
    problem: Problem,
    declared: frozenset[str] | None,
    adapter: object | None,
    options: Options,
) -> tuple[Callable[[Array], Array], str]:
    if _provides(problem, declared, "gradient"):
        return problem.gradient, "analytic"
    if options.enable_autodiff and adapter is not None:
        grad = adapter.grad  # type: ignore[attr-defined]
        return (lambda x: grad(problem.objective, x)), "autodiff"
    if options.enable_finite_diff:
        warn_external(
            "objective gradient resolved by finite differences; supply an "
            "analytic gradient or an autodiff backend for accuracy and speed",
            FiniteDifferenceWarning,
        )
        return (lambda x: gradient_fd(problem.objective, x)), "finite-diff"
    raise DerivativeUnavailableError(
        "no gradient source available: the problem defines no analytic "
        "gradient and both autodiff and finite differences are disabled"
    )


def _resolve_jacobian(
    problem: Problem,
    declared: frozenset[str] | None,
    constraints_name: str,
    jacobian_name: str,
    constraints_fn: Callable[[Array], Array],
    adapter: object | None,
    options: Options,
) -> tuple[Callable[[Array], Array | LinearOperator] | None, str]:
    if not _provides(problem, declared, constraints_name):
        return None, "n/a"
    if _provides(problem, declared, jacobian_name):
        return getattr(problem, jacobian_name), "analytic"
    if options.enable_autodiff and adapter is not None:
        jac = adapter.jacobian  # type: ignore[attr-defined]
        return (lambda x: jac(constraints_fn, x)), "autodiff"
    if options.enable_finite_diff:
        warn_external(
            f"{constraints_name} Jacobian resolved by finite differences; this "
            "is dense and only appropriate for small problems",
            FiniteDifferenceWarning,
        )
        return (lambda x: jacobian_fd(constraints_fn, x)), "finite-diff"
    raise DerivativeUnavailableError(
        f"no Jacobian source available for {constraints_name}: define "
        f"{jacobian_name}, or enable autodiff / finite differences"
    )


def _resolve_hessian_source(
    problem: Problem,
    declared: frozenset[str] | None,
    adapter: object | None,
    options: Options,
) -> tuple[bool, str]:
    """Return ``(has_analytic, label)`` for the Lagrangian Hessian (§3.2).

    Precedence depends on ``options.hessian``. The explicit quasi-Newton /
    matrix-free modes are honored **literally**, even when the problem supplies
    an analytic ``lagrangian_hessian``: ``"lbfgs"`` always uses the limited-memory
    approximation and ``"autodiff-hvp"`` always uses autodiff HVPs. ``"auto"``
    (default) and ``"exact"`` prefer a supplied analytic operator (``"exact"``
    requires one). With no analytic Hessian, ``"auto"`` falls back to L-BFGS.
    """
    mode = options.hessian
    if mode == "lbfgs":
        # Honor the explicit request even if an analytic Hessian is available.
        return False, "lbfgs"
    if mode == "autodiff-hvp":
        if adapter is None:
            raise DerivativeUnavailableError(
                "hessian='autodiff-hvp' requires an autodiff-capable backend "
                "(PyTorch or JAX); none is available for this namespace"
            )
        return False, "autodiff-hvp"
    if _provides(problem, declared, "lagrangian_hessian"):
        return True, "exact"
    if mode == "exact":
        raise DerivativeUnavailableError(
            "hessian='exact' requires problem.lagrangian_hessian; use "
            "hessian='auto' (or omit it) to fall back to L-BFGS, or "
            "hessian='autodiff-hvp' on an autodiff-capable backend"
        )
    return False, "lbfgs"


def resolve(
    problem: Problem,
    xp: Namespace,
    options: Options,
) -> ResolvedProblem:
    """Bind every derivative the solver needs, honoring precedence.

    Raises :class:`~ipax.errors.DerivativeUnavailableError` (a
    ``RuntimeError``) if a required derivative cannot be produced (e.g.
    autodiff disabled and finite-diff disabled with no analytic gradient).
    """
    adapter = get_autodiff_adapter(xp) if options.enable_autodiff else None

    declared = _declared(problem)
    gradient_fn, gradient_src = _resolve_gradient(problem, declared, adapter, options)
    eq_jacobian_fn, eq_jac_src = _resolve_jacobian(
        problem,
        declared,
        "eq_constraints",
        "eq_jacobian",
        problem.eq_constraints,
        adapter,
        options,
    )
    ineq_jacobian_fn, ineq_jac_src = _resolve_jacobian(
        problem,
        declared,
        "ineq_constraints",
        "ineq_jacobian",
        problem.ineq_constraints,
        adapter,
        options,
    )
    has_analytic_hessian, hessian_src = _resolve_hessian_source(
        problem, declared, adapter, options
    )

    sources = DerivativeSources(
        gradient=gradient_src,
        eq_jacobian=eq_jac_src,
        ineq_jacobian=ineq_jac_src,
        hessian=hessian_src,
    )
    return ResolvedProblem(
        problem,
        gradient=gradient_fn,
        eq_jacobian=eq_jacobian_fn,
        ineq_jacobian=ineq_jacobian_fn,
        sources=sources,
        has_analytic_hessian=has_analytic_hessian,
    )


__all__ = ["ResolvedProblem", "resolve"]
