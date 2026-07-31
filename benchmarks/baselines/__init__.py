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
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

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


@dataclass(frozen=True)
class _Pattern:
    """A fixed COO sparsity declaration plus the means to scatter into it."""

    rows: np.ndarray
    cols: np.ndarray
    keys: np.ndarray  # the declared (row, col) pairs, encoded and sorted
    positions: np.ndarray  # keys[i] belongs at declared position positions[i]
    n_cols: int  # the width the keys were encoded with

    @property
    def nnz(self) -> int:
        return int(self.rows.shape[0])


def _encode(rows: np.ndarray, cols: np.ndarray, n_cols: int) -> np.ndarray:
    """``(row, col)`` pairs as single sortable integers."""
    import numpy as np

    return np.asarray(rows, dtype=np.int64) * np.int64(n_cols) + np.asarray(
        cols, dtype=np.int64
    )


def _union_pattern(jac_fn: Any, points: list[np.ndarray], n_cols: int) -> _Pattern:
    """The union of an operator's COO pattern over several evaluation points.

    IPOPT wants a *fixed* Jacobian sparsity declared once, and many CUTEst
    problems do not oblige: an entry whose value is exactly zero at a given point
    is absent from the operator's triplets there (S2MPJ ``OET7``, ``EIGMINA``),
    so the structural pattern depends on where you look. Sampling several points
    and declaring the union gives a superset the callback can scatter into,
    storing an explicit zero wherever an entry is missing at the current point.

    Sampling cannot *prove* coverage, though — ``EIGMINA`` emits five nonzeros at
    both the start point and the generic probe and six once the iterates move,
    which is precisely how a two-point sample used to conclude "stable" and hand
    the solver values of the wrong length. So the returned scatter map is always
    built, and the values callback verifies rather than assumes.
    """
    import numpy as np

    from ipax.backend.operators import as_operator

    def _coo(x: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            r, c, _v, _shape = as_operator(jac_fn(np.asarray(x))).to_coo()
        except NotImplementedError as exc:
            raise BaselineUnsupported("ipyopt needs a COO-structured Jacobian") from exc
        return np.asarray(r, np.int64), np.asarray(c, np.int64)

    patterns = [_coo(points[0])]  # the start point must evaluate
    for x in points[1:]:
        try:
            patterns.append(_coo(x))
        except BaselineUnsupported:
            raise
        except Exception:
            # A sampled point may leave the problem's domain (a log of a
            # negative, a division by zero). It contributes nothing; the others
            # still widen the pattern.
            continue

    keys = np.unique(
        np.concatenate([_encode(r, c, n_cols) for r, c in patterns if r is not None])
    )
    rows, cols = np.divmod(keys, np.int64(n_cols))
    # `keys` is sorted, so searchsorted finds a point's entries in it directly;
    # `positions` is the identity because the declaration is laid out in key
    # order. Kept explicit so the layout can change without touching the scatter.
    positions = np.arange(keys.shape[0], dtype=np.int64)
    return _Pattern(rows=rows, cols=cols, keys=keys, positions=positions, n_cols=n_cols)


def _ipyopt_blocks(
    problem: Problem, x0: np.ndarray
) -> tuple[list[_ConstraintBlock], int]:
    """Translate every ipax constraint block into IPOPT two-sided form."""
    import numpy as np

    from ipax.backend.operators import as_operator

    blocks: list[_ConstraintBlock] = []
    offset = 0
    n = int(problem.n_vars)
    # Sample points whose union is the declared Jacobian pattern, each clamped
    # into the bounds so it stays a valid evaluation point. Beyond x0 and a
    # generic probe, perturb x0: structural entries usually go missing because a
    # variable sits at exactly zero, and a start point of all zeros (common in
    # CUTEst) hides them all. Perturbing stays near the problem's own domain,
    # where a fresh random point would often hit a log of a negative.
    lower, upper = problem.bounds()

    def _clamp(x: np.ndarray) -> np.ndarray:
        """Bring a sample point *strictly* inside the box.

        Clamping onto a bound is the wrong place to look: entries routinely
        vanish exactly there (S2MPJ ``EIGMINA`` loses a Jacobian entry when its
        first variable sits at its upper bound of 1), and an interior-point
        solver's iterates never reach a bound anyway. Sampling the interior
        samples where the pattern actually has to hold.
        """
        lo = None if lower is None else np.asarray(lower, dtype=float)
        hi = None if upper is None else np.asarray(upper, dtype=float)
        if lo is not None and hi is not None:
            span = hi - lo
            inset = np.where(np.isfinite(span), 0.01 * span, 0.0)
            # A fixed variable (lo == hi) has no interior; leave it alone.
            return np.clip(x, lo + inset, hi - inset)
        if lo is not None:
            return np.maximum(x, lo + 0.01 * np.maximum(np.abs(lo), 1.0))
        if hi is not None:
            return np.minimum(x, hi - 0.01 * np.maximum(np.abs(hi), 1.0))
        return x

    # Seeded: a report has to reproduce, so the declared pattern cannot drift
    # between runs.
    rng = np.random.default_rng(0)
    points = [x0, _clamp(_probe(n))]
    for scale in (0.1, 1.0, 10.0):
        points.append(_clamp(x0 + rng.normal(scale=scale, size=n)))

    def _nonlinear(values_attr: str, jac_attr: str, lo: float, hi: float) -> None:
        nonlocal offset
        if not _provides(problem, values_attr, x0):
            return
        jfn = getattr(problem, jac_attr)
        pattern = _union_pattern(jfn, points, n)
        m = int(as_operator(jfn(x0)).to_coo()[3][0])
        vfn = getattr(problem, values_attr)

        def _values(
            x: np.ndarray, jfn: Any = jfn, pattern: _Pattern = pattern
        ) -> np.ndarray:
            r, c, v, _shape = as_operator(jfn(np.asarray(x))).to_coo()
            r = np.asarray(r, dtype=np.int64)
            c = np.asarray(c, dtype=np.int64)
            v = np.asarray(v, dtype=float)
            if r.shape[0] == pattern.nnz and np.array_equal(r, pattern.rows):
                if np.array_equal(c, pattern.cols):
                    return v  # already in the declared layout

            # Scatter into the declared layout, leaving stored zeros where this
            # point has no entry. Vectorized: at RT scale this runs on millions
            # of nonzeros every iteration.
            key = _encode(r, c, pattern.n_cols)
            index = np.searchsorted(pattern.keys, key)
            index = np.clip(index, 0, pattern.keys.shape[0] - 1)
            if not np.array_equal(pattern.keys[index], key):
                # No finite sample can guarantee coverage, so say so plainly
                # rather than hand IPOPT an array of the wrong length and let it
                # surface as an unattributable shape error mid-solve.
                raise BaselineUnsupported(
                    "ipyopt: the Jacobian has a nonzero outside the sampled "
                    "sparsity pattern"
                )
            return np.bincount(
                pattern.positions[index], weights=v, minlength=pattern.nnz
            )

        blocks.append(
            _ConstraintBlock(
                g_fn=lambda x, vfn=vfn: np.asarray(vfn(np.asarray(x)), dtype=float),
                g_l=np.full(m, lo),
                g_u=np.full(m, hi),
                rows=pattern.rows + offset,
                cols=pattern.cols,
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


@dataclass(frozen=True)
class IpyoptBaseline:
    """IPOPT via the sparse-native ``ipyopt`` binding.

    Uses a limited-memory (L-BFGS) Hessian to mirror ipax's default Hessian mode
    and to sidestep supplying a Hessian sparsity pattern. Reports IPOPT's own
    iteration count, so the comparison is on the language-neutral axis (the
    algorithm), not wall-clock across a compiled solver and a pure-Python one.

    ``max_iter``/``max_time`` bound the reference the way :class:`ipax.Options`
    bounds ipax, so an unattended full-corpus comparison terminates on both
    sides under budgets that can be stated as equal. ``max_time=None`` leaves
    IPOPT's own (unbounded) default in place.

    ``options`` passes extra IPOPT options straight through (as ``(key, value)``
    pairs, so the dataclass stays hashable). It is what makes a *parameter-
    matched* arm possible: IPOPT's defaults are not ipax's — notably
    ``mu_strategy`` (IPOPT's build default is adaptive; ipax defaults to
    monotone) and ``limited_memory_max_history`` (6 vs ipax's 10) — so a
    default-vs-default verdict and a matched-parameter verdict answer different
    questions, and both are worth running.
    """

    name: ClassVar[str] = "ipyopt"
    max_iter: int = 3000
    max_time: float | None = None
    options: tuple[tuple[str, Any], ...] = ()

    def solve(self, problem: Problem, x0: np.ndarray) -> ReferenceResult:
        import ipyopt
        import numpy as np

        x0 = np.asarray(x0, dtype=float)
        n = int(problem.n_vars)

        # IPOPT needs an explicit objective gradient (`eval_grad_f`). A Problem
        # that leaves `gradient` to ipax's derivative resolution (autodiff /
        # finite-diff) raises NotImplementedError here — fail fast with
        # BaselineUnsupported (recorded as "skipped") instead of crashing
        # mid-solve, matching the matrix-free Jacobian rejection above.
        try:
            problem.gradient(x0)
        except NotImplementedError as exc:
            raise BaselineUnsupported(
                "ipyopt needs an explicit problem.gradient"
            ) from exc

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

        ipopt_options: dict[str, Any] = {
            "hessian_approximation": "limited-memory",
            "tol": 1e-8,
            "max_iter": self.max_iter,
            "print_level": 0,
            "sb": "yes",  # suppress the startup banner
        }
        if self.max_time is not None:
            # Wall time is the axis ipax's ``max_time`` caps, so budgets stated
            # for the two solvers mean the same thing.
            ipopt_options["max_wall_time"] = float(self.max_time)
        # Caller overrides last, so a parameter-matched arm can replace any of
        # the defaults chosen above.
        ipopt_options.update(dict(self.options))

        def _build(options: dict[str, Any]) -> Any:
            return ipyopt.Problem(
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
                ipopt_options=options,
            )

        try:
            nlp = _build(ipopt_options)
        except Exception:
            # ``max_wall_time`` needs IPOPT >= 3.14; older builds reject the
            # option at construction and only expose the CPU-time cap.
            if "max_wall_time" not in ipopt_options:
                raise
            ipopt_options["max_cpu_time"] = ipopt_options.pop("max_wall_time")
            nlp = _build(ipopt_options)

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
