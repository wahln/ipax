"""Reference-solver baselines for accuracy cross-checks.

A pluggable interface so ipax results can be validated against established
solvers. Only baselines whose optional dependency is installed are returned by
:func:`available_baselines`:

* :class:`ScipyBaseline` — ``scipy.optimize`` (general NLP).
* :class:`CyipoptBaseline` — IPOPT via ``cyipopt`` (general NLP); shares the
  SciPy-style problem translation since ``cyipopt.minimize_ipopt`` mirrors
  ``scipy.optimize.minimize``.
* :class:`OsqpBaseline` — OSQP (convex QP with linear constraints only); raises
  :class:`BaselineUnsupported` for nonlinear/non-quadratic problems.

A baseline that cannot express a given problem (matrix-free Jacobian, nonlinear
constraint for OSQP, …) raises :class:`BaselineUnsupported`, which the
cross-check records as "skipped". Cross-checks are NumPy-only.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from ipax.problem.base import Problem


class BaselineUnsupported(RuntimeError):
    """The baseline cannot express this problem (e.g. a matrix-free Jacobian)."""


@dataclass(frozen=True)
class ReferenceResult:
    """A reference solver's outcome, normalized across baselines."""

    name: str
    x: np.ndarray
    objective: float
    success: bool
    n_iter: int
    solve_time: float


@runtime_checkable
class Baseline(Protocol):
    """A reference solver that can solve an ipax :class:`Problem` from ``x0``."""

    name: str

    def solve(self, problem: Problem, x0: np.ndarray) -> ReferenceResult: ...


def _provides(problem: Problem, name: str, x0: np.ndarray) -> bool:
    """Whether ``problem`` actually implements the (optional) method ``name``."""
    try:
        getattr(problem, name)(x0)
    except NotImplementedError:
        return False
    return True


def _dense(matrix: object) -> np.ndarray:
    """Require a dense 2-D Jacobian/Hessian; reject matrix-free operators."""
    import numpy as np

    if getattr(matrix, "ndim", None) == 2:
        return np.asarray(matrix, dtype=float)
    raise BaselineUnsupported("baseline requires a dense matrix")


# -- shared SciPy-style translation (SciPy + cyipopt) ------------------------


@dataclass(frozen=True)
class _ScipyProblem:
    objective: Any
    gradient: Any
    bounds: Any
    constraints: list[Any]


def _scipy_problem(problem: Problem, x0: np.ndarray) -> _ScipyProblem:
    """Translate ipax conventions into SciPy ``minimize`` arguments.

    ``c(x)=0`` and ``g(x)≤0`` become ``NonlinearConstraint``s, ``A x = b`` a
    ``LinearConstraint``, and ``bounds()`` a ``Bounds``. Dense Jacobians are
    required (validated up front), so matrix-free problems raise
    :class:`BaselineUnsupported`.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

    n = int(problem.n_vars)

    def objective(x: np.ndarray) -> float:
        return float(problem.objective(np.asarray(x)))

    def gradient(x: np.ndarray) -> np.ndarray:
        return np.asarray(problem.gradient(np.asarray(x)), dtype=float)

    constraints: list[Any] = []
    if _provides(problem, "eq_constraints", x0):
        _dense(problem.eq_jacobian(x0))
        constraints.append(
            NonlinearConstraint(
                lambda x: np.asarray(problem.eq_constraints(np.asarray(x))),
                0.0,
                0.0,
                jac=lambda x: _dense(problem.eq_jacobian(np.asarray(x))),
            )
        )
    if _provides(problem, "ineq_constraints", x0):
        _dense(problem.ineq_jacobian(x0))
        constraints.append(
            NonlinearConstraint(
                lambda x: np.asarray(problem.ineq_constraints(np.asarray(x))),
                -np.inf,
                0.0,  # ipax convention g(x) <= 0
                jac=lambda x: _dense(problem.ineq_jacobian(np.asarray(x))),
            )
        )
    linear_eq = problem.linear_eq()
    if linear_eq is not None:
        matrix, rhs = linear_eq
        b = np.asarray(rhs, dtype=float)
        constraints.append(LinearConstraint(_dense(matrix), b, b))

    lower, upper = problem.bounds()
    bounds = None
    if lower is not None or upper is not None:
        lb = np.full(n, -np.inf) if lower is None else np.asarray(lower, dtype=float)
        ub = np.full(n, np.inf) if upper is None else np.asarray(upper, dtype=float)
        bounds = Bounds(lb, ub)

    return _ScipyProblem(objective, gradient, bounds, constraints)


class ScipyBaseline:
    """``scipy.optimize.minimize(method="trust-constr")`` reference."""

    name = "scipy-trust-constr"

    def solve(self, problem: Problem, x0: np.ndarray) -> ReferenceResult:
        import numpy as np
        from scipy.optimize import minimize

        x0 = np.asarray(x0, dtype=float)
        model = _scipy_problem(problem, x0)
        start = perf_counter()
        res = minimize(
            model.objective,
            x0,
            method="trust-constr",
            jac=model.gradient,
            bounds=model.bounds,
            constraints=model.constraints,
            options={"gtol": 1e-9, "xtol": 1e-10, "maxiter": 3000},
        )
        elapsed = perf_counter() - start
        return ReferenceResult(
            name=self.name,
            x=np.asarray(res.x, dtype=float),
            objective=float(res.fun),
            success=bool(res.success),
            n_iter=int(getattr(res, "niter", 0)),
            solve_time=elapsed,
        )


class CyipoptBaseline:
    """IPOPT via ``cyipopt.minimize_ipopt`` (SciPy-compatible signature)."""

    name = "cyipopt"

    def solve(self, problem: Problem, x0: np.ndarray) -> ReferenceResult:
        import numpy as np
        from cyipopt import minimize_ipopt

        x0 = np.asarray(x0, dtype=float)
        model = _scipy_problem(problem, x0)
        start = perf_counter()
        res = minimize_ipopt(
            model.objective,
            x0,
            jac=model.gradient,
            bounds=model.bounds,
            constraints=model.constraints,
            options={"tol": 1e-9, "max_iter": 3000, "print_level": 0},
        )
        elapsed = perf_counter() - start
        x = np.asarray(res.x, dtype=float)
        return ReferenceResult(
            name=self.name,
            x=x,
            objective=float(problem.objective(x)),
            success=bool(res.success),
            n_iter=int(getattr(res, "nit", getattr(res, "niter", 0))),
            solve_time=elapsed,
        )


def _probe(n: int) -> np.ndarray:
    """A deterministic *nonzero* test point (so an origin start can't hide curvature)."""
    import numpy as np

    return np.arange(1, n + 1, dtype=float)


def _affine_part(fn: Any, jac: Any, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(J, f0)`` for an affine map ``f(x)=J x + f0``; raise if nonlinear.

    Affinity is checked at a nonzero probe point, not at ``x0`` — evaluating it at
    the origin (a common start) is degenerate (``J·0 + f0 = f(0)`` always holds).
    """
    import numpy as np

    matrix = _dense(jac(x0))
    f0 = np.asarray(fn(np.zeros_like(x0)), dtype=float)
    probe = _probe(int(x0.shape[0]))
    actual = np.asarray(fn(probe), dtype=float)
    if not np.allclose(matrix @ probe + f0, actual, atol=1e-7, rtol=1e-6):
        raise BaselineUnsupported("OSQP requires affine constraints")
    return matrix, f0


class OsqpBaseline:
    """OSQP convex-QP reference (objective quadratic, constraints linear).

    Extracts ``P`` from the (constant) Lagrangian Hessian and ``q`` from the
    affine gradient, then stacks bounds + linear/affine constraints into the
    ``l ≤ A x ≤ u`` box OSQP solves. Anything nonlinear/non-quadratic raises
    :class:`BaselineUnsupported`.
    """

    name = "osqp"

    def solve(self, problem: Problem, x0: np.ndarray) -> ReferenceResult:
        import numpy as np
        import osqp
        import scipy.sparse as sp

        x0 = np.asarray(x0, dtype=float)
        n = int(problem.n_vars)
        zero = np.zeros(n)

        # Objective Hessian P = ∇²f: feed zero multipliers of the *correct length*
        # (empty arrays would break problems that index y_eq/y_ineq).
        m_eq = (
            int(np.asarray(problem.eq_constraints(x0)).shape[0])
            if _provides(problem, "eq_constraints", x0)
            else 0
        )
        m_ineq = (
            int(np.asarray(problem.ineq_constraints(x0)).shape[0])
            if _provides(problem, "ineq_constraints", x0)
            else 0
        )
        try:
            p_matrix = _dense(
                problem.lagrangian_hessian(x0, np.zeros(m_eq), np.zeros(m_ineq), 1.0)
            )
        except NotImplementedError as exc:
            raise BaselineUnsupported("OSQP requires an explicit Hessian") from exc
        # Quadratic objective ⇒ affine gradient ∇f(x) = P x + q (checked at a
        # nonzero probe so an origin start cannot hide a higher-order objective).
        q = np.asarray(problem.gradient(zero), dtype=float)
        probe = _probe(n)
        if not np.allclose(
            np.asarray(problem.gradient(probe), dtype=float),
            p_matrix @ probe + q,
            atol=1e-7,
            rtol=1e-6,
        ):
            raise BaselineUnsupported("OSQP requires a quadratic objective")

        rows: list[np.ndarray] = []
        lo: list[np.ndarray] = []
        hi: list[np.ndarray] = []

        lower, upper = problem.bounds()
        if lower is not None or upper is not None:
            rows.append(np.eye(n))
            lo.append(
                np.full(n, -np.inf) if lower is None else np.asarray(lower, float)
            )
            hi.append(np.full(n, np.inf) if upper is None else np.asarray(upper, float))

        linear_eq = problem.linear_eq()
        if linear_eq is not None:
            matrix, rhs = linear_eq
            b = np.asarray(rhs, dtype=float)
            rows.append(_dense(matrix))
            lo.append(b)
            hi.append(b)

        if _provides(problem, "eq_constraints", x0):
            j, c0 = _affine_part(problem.eq_constraints, problem.eq_jacobian, x0)
            rows.append(j)
            lo.append(-c0)
            hi.append(-c0)
        if _provides(problem, "ineq_constraints", x0):
            j, g0 = _affine_part(problem.ineq_constraints, problem.ineq_jacobian, x0)
            rows.append(j)
            lo.append(np.full(g0.shape[0], -np.inf))
            hi.append(-g0)  # J x + g0 <= 0

        a_matrix = sp.csc_matrix(np.vstack(rows)) if rows else sp.csc_matrix((0, n))
        solver = osqp.OSQP()
        solver.setup(
            P=sp.csc_matrix(p_matrix),
            q=q,
            A=a_matrix,
            l=np.concatenate(lo) if lo else np.zeros(0),
            u=np.concatenate(hi) if hi else np.zeros(0),
            verbose=False,
            eps_abs=1e-9,
            eps_rel=1e-9,
            max_iter=20000,
        )
        start = perf_counter()
        res = solver.solve()
        elapsed = perf_counter() - start
        x = np.asarray(res.x, dtype=float)
        success = "solved" in str(res.info.status)
        return ReferenceResult(
            name=self.name,
            x=x,
            objective=float(problem.objective(x)) if success else float("inf"),
            success=success,
            n_iter=int(res.info.iter),
            solve_time=elapsed,
        )


def available_baselines() -> list[Baseline]:
    """Reference solvers whose optional dependency is installed."""
    baselines: list[Baseline] = []
    if importlib.util.find_spec("scipy") is not None:
        baselines.append(ScipyBaseline())
    if importlib.util.find_spec("cyipopt") is not None:
        baselines.append(CyipoptBaseline())
    if importlib.util.find_spec("osqp") is not None:
        baselines.append(OsqpBaseline())
    return baselines


__all__ = [
    "Baseline",
    "BaselineUnsupported",
    "CyipoptBaseline",
    "OsqpBaseline",
    "ReferenceResult",
    "ScipyBaseline",
    "available_baselines",
]
