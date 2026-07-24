"""Reference-solver baselines for accuracy cross-checks.

A pluggable interface so ipax results can be validated against established
solvers. Only baselines whose optional dependency is installed are returned by
:func:`available_baselines`:

* :class:`ScipyBaseline` — ``scipy.optimize`` (general NLP).
* :class:`CyipoptBaseline` — IPOPT via ``cyipopt`` (general NLP); shares the
  SciPy-style problem translation since ``cyipopt.minimize_ipopt`` mirrors
  ``scipy.optimize.minimize``.
* :class:`IpyoptBaseline` — IPOPT via ``ipyopt`` (general NLP); the *sparse-native*
  IPOPT binding — it consumes the constraint Jacobian as a COO pattern + values
  rather than a dense matrix, so unlike the SciPy-style path it scales to the
  tall, sparse RT-sized systems (``m ≫ n``) that would densify to gigabytes.
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


# -- sparse translation (ipyopt) --------------------------------------------
#
# IPOPT treats every constraint as two-sided ``g_l ≤ g(x) ≤ g_u``; ipax's
# equality (``c(x)=0``), inequality (``g(x)≤0``) and linear (``l ≤ Ax ≤ u``)
# blocks each collapse onto that with the right bounds. The Jacobian is handed
# over as a fixed COO *pattern* (constructor) plus a values callback in that
# exact order — which is precisely the ``to_coo()`` / ``coo_values()`` contract
# ipax operators already satisfy for the sparse-direct route, so no densification.
_IPOPT_INF = 2.0e19  # IPOPT's "infinite" bound sentinel


@dataclass(frozen=True)
class _ConstraintBlock:
    """One ipax constraint block, translated to IPOPT's two-sided form."""

    g_fn: Any  # x -> constraint values (m_block,)
    g_l: np.ndarray
    g_u: np.ndarray
    rows: np.ndarray  # COO row indices, already offset into the stacked Jacobian
    cols: np.ndarray
    values_fn: Any  # x -> Jacobian nonzeros in (rows, cols) order


def _union_pattern(
    jac_fn: Any, points: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int] | None]:
    """The union of an operator's COO pattern over several evaluation points.

    IPOPT wants a *fixed* Jacobian sparsity declared once; a problem whose
    structural nonzeros differ between points (e.g. a value that is exactly zero
    at ``x0``, so the operator omits it there, but nonzero elsewhere — S2MPJ
    ``OET7``) would otherwise overflow the declared pattern. Taking the union
    over ``x0`` plus a generic probe point captures the superset, and the
    returned ``(row, col) -> position`` map lets the values callback scatter each
    point's nonzeros into that fixed layout.

    The common case — a *stable* pattern (every point emits the same COO in the
    same order, e.g. a linear RT dose-constraint Jacobian) — returns ``None`` for
    the map, signalling the fast path: the values callback can use ``to_coo()``'s
    values verbatim, no per-nonzero scatter (which at RT scale, millions of
    nonzeros × hundreds of iterations, is the difference between usable and not).
    """
    import numpy as np

    from ipax.backend.operators import as_operator

    def _coo(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            r, c, v, _shape = as_operator(jac_fn(np.asarray(x))).to_coo()
        except NotImplementedError as exc:
            raise BaselineUnsupported("ipyopt needs a COO-structured Jacobian") from exc
        return np.asarray(r, np.int64), np.asarray(c, np.int64), np.asarray(v, float)

    patterns = [(_coo(x)[0], _coo(x)[1]) for x in points]
    r0, c0 = patterns[0]
    if all(
        r.shape == r0.shape and np.array_equal(r, r0) and np.array_equal(c, c0)
        for r, c in patterns[1:]
    ):
        return r0, c0, None  # stable pattern → fast path (no scatter map)

    # Variable structure: build the union pattern + a (row, col) -> position map.
    seen: dict[tuple[int, int], int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for r, c in patterns:
        for ri, ci in zip(r.tolist(), c.tolist(), strict=True):
            key = (int(ri), int(ci))
            if key not in seen:
                seen[key] = len(rows)
                rows.append(int(ri))
                cols.append(int(ci))
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), seen


def _ipyopt_blocks(
    problem: Problem, x0: np.ndarray
) -> tuple[list[_ConstraintBlock], int]:
    """Translate every ipax constraint block into IPOPT two-sided form."""
    import numpy as np

    from ipax.backend.operators import as_operator

    blocks: list[_ConstraintBlock] = []
    offset = 0
    n = int(problem.n_vars)
    # A second, generic point so the declared Jacobian pattern is the union over
    # x0 and it, clamped into the bounds so it stays a valid evaluation point.
    lower, upper = problem.bounds()
    probe = _probe(n)
    if lower is not None:
        probe = np.maximum(probe, np.asarray(lower, dtype=float))
    if upper is not None:
        probe = np.minimum(probe, np.asarray(upper, dtype=float))
    points = [x0, probe]

    def _nonlinear(values_attr: str, jac_attr: str, lo: float, hi: float) -> None:
        nonlocal offset
        if not _provides(problem, values_attr, x0):
            return
        jfn = getattr(problem, jac_attr)
        rows_local, cols_local, index_of = _union_pattern(jfn, points)
        nnz = int(rows_local.shape[0])
        m = int(as_operator(jfn(x0)).to_coo()[3][0])
        vfn = getattr(problem, values_attr)

        def _values(
            x: np.ndarray, jfn: Any = jfn, index_of: Any = index_of, nnz: int = nnz
        ) -> np.ndarray:
            r, c, v, _shape = as_operator(jfn(np.asarray(x))).to_coo()
            v = np.asarray(v, dtype=float)
            if index_of is None:  # stable pattern: values are already aligned
                return v
            out = np.zeros(nnz, dtype=float)
            r = np.asarray(r).tolist()
            c = np.asarray(c).tolist()
            for k in range(len(r)):
                pos = index_of.get((int(r[k]), int(c[k])))
                if pos is None:  # an entry outside the union pattern
                    raise BaselineUnsupported("ipyopt: Jacobian pattern is not fixed")
                out[pos] += v[k]
            return out

        blocks.append(
            _ConstraintBlock(
                g_fn=lambda x, vfn=vfn: np.asarray(vfn(np.asarray(x)), dtype=float),
                g_l=np.full(m, lo),
                g_u=np.full(m, hi),
                rows=rows_local + offset,
                cols=cols_local,
                values_fn=_values,
            )
        )
        offset += m

    _nonlinear("eq_constraints", "eq_jacobian", 0.0, 0.0)
    _nonlinear("ineq_constraints", "ineq_jacobian", -_IPOPT_INF, 0.0)

    def _linear(data: Any, lo: np.ndarray, hi: np.ndarray) -> None:
        nonlocal offset
        op = as_operator(data)
        try:
            rows, cols, values, shape = op.to_coo()
        except NotImplementedError as exc:
            raise BaselineUnsupported("ipyopt needs a COO-structured Jacobian") from exc
        m = int(shape[0])
        rows = np.asarray(rows, dtype=np.int64) + offset
        cols = np.asarray(cols, dtype=np.int64)
        values = np.asarray(values, dtype=float)  # constant across x
        blocks.append(
            _ConstraintBlock(
                g_fn=lambda x, op=op: np.asarray(
                    op.matvec(np.asarray(x, dtype=float)), dtype=float
                ),
                g_l=np.asarray(lo, dtype=float),
                g_u=np.asarray(hi, dtype=float),
                rows=rows,
                cols=cols,
                values_fn=lambda x, values=values: values,
            )
        )
        offset += m

    linear_eq = problem.linear_eq()
    if linear_eq is not None:
        a, b = linear_eq
        b = np.asarray(b, dtype=float)
        _linear(a, b, b)  # A x = b  ⇒  b ≤ A x ≤ b
    linear_ineq = problem.linear_ineq()
    if linear_ineq is not None:
        a, lo, hi = linear_ineq
        lo = np.where(np.isfinite(lo), lo, -_IPOPT_INF)
        hi = np.where(np.isfinite(hi), hi, _IPOPT_INF)
        _linear(a, lo, hi)

    return blocks, offset


class IpyoptBaseline:
    """IPOPT via the sparse-native ``ipyopt`` binding.

    Uses a limited-memory (L-BFGS) Hessian to mirror ipax's default Hessian mode
    and to sidestep supplying a Hessian sparsity pattern. Reports IPOPT's own
    iteration count, so the comparison is on the language-neutral axis (the
    algorithm), not wall-clock across a compiled solver and a pure-Python one.
    """

    name = "ipyopt"

    def solve(self, problem: Problem, x0: np.ndarray) -> ReferenceResult:
        import ipyopt
        import numpy as np

        x0 = np.asarray(x0, dtype=float)
        n = int(problem.n_vars)

        lower, upper = problem.bounds()
        x_l = (
            np.full(n, -_IPOPT_INF) if lower is None else np.asarray(lower, dtype=float)
        )
        x_u = (
            np.full(n, _IPOPT_INF) if upper is None else np.asarray(upper, dtype=float)
        )
        x_l = np.where(np.isfinite(x_l), x_l, -_IPOPT_INF)
        x_u = np.where(np.isfinite(x_u), x_u, _IPOPT_INF)

        blocks, m = _ipyopt_blocks(problem, x0)
        if blocks:
            g_l = np.concatenate([b.g_l for b in blocks])
            g_u = np.concatenate([b.g_u for b in blocks])
            jac_rows = np.concatenate([b.rows for b in blocks])
            jac_cols = np.concatenate([b.cols for b in blocks])
        else:
            g_l = g_u = np.zeros(0)
            jac_rows = jac_cols = np.zeros(0, dtype=np.int64)

        def eval_f(x: np.ndarray) -> float:
            return float(problem.objective(np.asarray(x)))

        def eval_grad_f(x: np.ndarray, out: np.ndarray) -> Any:
            out[:] = np.asarray(problem.gradient(np.asarray(x)), dtype=float)
            return out

        def eval_g(x: np.ndarray, out: np.ndarray) -> Any:
            if blocks:
                out[:] = np.concatenate([b.g_fn(x) for b in blocks])
            return out

        def eval_jac_g(x: np.ndarray, out: np.ndarray) -> Any:
            if blocks:
                out[:] = np.concatenate([b.values_fn(x) for b in blocks])
            return out

        last_iter = {"k": 0}

        def intermediate(*args: Any) -> Any:
            # IPOPT's intermediate callback: first arg is alg_mod, second iter.
            if len(args) >= 2:
                last_iter["k"] = int(args[1])
            return True

        nlp = ipyopt.Problem(
            n,
            x_l,
            x_u,
            m,
            g_l,
            g_u,
            (jac_rows, jac_cols),
            (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)),
            eval_f,
            eval_grad_f,
            eval_g,
            eval_jac_g,
            intermediate_callback=intermediate,
            ipopt_options={
                "hessian_approximation": "limited-memory",
                "tol": 1e-8,
                "max_iter": 3000,
                "print_level": 0,
                "sb": "yes",  # suppress the startup banner
            },
        )

        x = x0.copy()
        start = perf_counter()
        x_opt, _obj, status = nlp.solve(x)
        elapsed = perf_counter() - start

        x_opt = np.asarray(x_opt, dtype=float)
        n_iter = last_iter["k"]
        try:
            n_iter = int(nlp.stats().get("iter_count", n_iter))
        except Exception:  # stats key differs across builds; fall back to callback
            pass
        return ReferenceResult(
            name=self.name,
            x=x_opt,
            objective=float(problem.objective(x_opt)),
            success=status in (0, 1),  # Solve_Succeeded / Solved_To_Acceptable_Level
            n_iter=n_iter,
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
    if importlib.util.find_spec("ipyopt") is not None:
        baselines.append(IpyoptBaseline())
    if importlib.util.find_spec("osqp") is not None:
        baselines.append(OsqpBaseline())
    return baselines


__all__ = [
    "Baseline",
    "BaselineUnsupported",
    "CyipoptBaseline",
    "IpyoptBaseline",
    "OsqpBaseline",
    "ReferenceResult",
    "ScipyBaseline",
    "available_baselines",
]
