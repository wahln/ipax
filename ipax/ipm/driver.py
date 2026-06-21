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

"""The main IPM loop: convergence test, logging, callbacks.

``IPMDriver`` ties the components together but owns no linear-algebra strategy:
the ``LinearSolver`` is injected (invariant #3), so new solve routes never touch
this file. Strict convergence is the scaled KKT ∞-norm ≤ ``tol`` (IPOPT
scaling ``s_d, s_c``; Wächter & Biegler 2006, eqs. 4–6); an explicit checker
handles optional guarded acceptable-stagnation and wall-time exits.

Handles bounds + (nonlinear) inequalities + equalities, with the Lagrangian
Hessian resolved by precedence — analytic operator, autodiff HVP, or the
persistent Powell-damped L-BFGS approximation whose curvature pairs this loop
updates from each accepted step. Inequalities/bounds use the condensed
normal-equations route (Breedveld 2017, eq. 18); equalities border it into the
Friedlander–Orban quasidefinite saddle. Globalization is the filter line-search
(default) or the Breedveld step controller, with a Gauss-Newton feasibility
restoration phase. The injected ``LinearSolver`` is the dense reference solver,
the matrix-free Krylov solver, or the sparse-direct route.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeVar

from ipax._logging import (
    HEADER_REPEAT_INTERVAL,
    ITERATION,
    format_header,
    format_record,
    logger,
)
from ipax.backend.namespace import array_namespace
from ipax.backend.operators import (
    Dense,
    Diagonal,
    LinearOperator,
    MatrixFreeJacobian,
    as_operator,
)
from ipax.ipm.barrier import fraction_to_boundary, update_mu
from ipax.ipm.breedveld_ls import BreedveldController
from ipax.ipm.corrections import CorrectionContext, select_corrector
from ipax.ipm.filter_ls import Filter, FilterLineSearch
from ipax.ipm.hessian import LBFGSOperator
from ipax.ipm.init import apply_warm_start, initialize
from ipax.ipm.kkt import build_condensed_operator, build_saddle_operator
from ipax.ipm.restoration import restore
from ipax.ipm.step import NewtonStep, recover_eliminated
from ipax.ipm.termination import ConditionChecker
from ipax.linalg.regularize import RegularizationState, escalate_delta_w
from ipax.linalg.solver import LinearSolveError
from ipax.problem.autodiff import get_autodiff_adapter
from ipax.result import (
    DerivativeSources,
    IterationInfo,
    IterationRecord,
    KKTResiduals,
    Result,
    Status,
)

if TYPE_CHECKING:
    from ipax.linalg.solver import LinearSolver
    from ipax.options import Options
    from ipax.problem.base import Problem
    from ipax.result import IterationCallback, WarmStart
    from ipax.typing import Array, Namespace


T = TypeVar("T")


class _VStack(LinearOperator):
    """Vertical stack of operators sharing the variable dimension."""

    def __init__(self, ops: tuple[LinearOperator, ...]) -> None:
        self._ops = ops
        self._n = ops[0].shape[1]
        self._rows = tuple(op.shape[0] for op in ops)

    @property
    def shape(self) -> tuple[int, int]:
        return sum(self._rows), self._n

    def matvec(self, v: Array) -> Array:
        xp = array_namespace(v)
        return xp.concat(tuple(op.matvec(v) for op in self._ops))

    def rmatvec(self, v: Array) -> Array:
        xp = array_namespace(v)
        result = None
        offset = 0
        for op, rows in zip(self._ops, self._rows, strict=True):
            piece = op.rmatvec(v[offset : offset + rows])
            result = piece if result is None else result + piece
            offset += rows
        assert result is not None
        return xp.asarray(result)

    def row_gram_diagonal(self, weights: Array) -> Array:
        # Rows are stacked, so the weighted row energies concatenate. Propagates
        # NotImplementedError if any block cannot supply them cheaply.
        xp = array_namespace(weights)
        return xp.concat(tuple(op.row_gram_diagonal(weights) for op in self._ops))

    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        # Vertically stacked blocks ⇒ concatenate triplets with row offsets.
        del like
        rows_parts: list[Array] = []
        cols_parts: list[Array] = []
        vals_parts: list[Array] = []
        offset = 0
        xp = None
        for op, n_rows in zip(self._ops, self._rows, strict=True):
            r, c, v, _ = op.to_coo()
            if xp is None:
                xp = array_namespace(v)
            rows_parts.append(r + offset)
            cols_parts.append(c)
            vals_parts.append(v)
            offset += n_rows
        assert xp is not None
        return (
            xp.concat(tuple(rows_parts)),
            xp.concat(tuple(cols_parts)),
            xp.concat(tuple(vals_parts)),
            (offset, self._n),
        )


# IPOPT (Wächter & Biegler 2006) constants kept out of the loop body.
_S_MAX = 100.0  # eq. (5): cap on the dual/complementarity scaling factors
_KAPPA_EPSILON = 10.0  # eq. (7): barrier sub-problem tolerance factor κ_ε
_MAX_REG_ATTEMPTS = 40
_MAX_MU_REDUCTIONS = 64


def _norm_inf(xp: Namespace, v: Array) -> float:
    if int(v.shape[0]) == 0:
        return 0.0
    return float(xp.max(xp.abs(v)))


def _norm1(xp: Namespace, v: Array) -> float:
    if int(v.shape[0]) == 0:
        return 0.0
    return float(xp.sum(xp.abs(v)))


class IPMDriver:
    """Primal–dual interior-point iteration (condensed normal-equations route)."""

    def __init__(
        self,
        problem: Problem,
        *,
        xp: Namespace,
        solver: LinearSolver,
        options: Options,
        lower: Array | None,
        upper: Array | None,
        has_ineq: bool,
        has_eq: bool = False,
        callback: IterationCallback | None = None,
        record_transform: Callable[[IterationRecord], IterationRecord] | None = None,
        warm_start: WarmStart | None = None,
    ) -> None:
        self._problem = problem
        self._xp = xp
        self._solver = solver
        self._options = options
        self._lower = lower
        self._upper = upper
        self._has_ineq = has_ineq
        self._has_eq = has_eq
        self._callback = callback
        self._record_transform = record_transform
        self._warm_start = warm_start
        # Higher-order step correction strategy. The default is a no-op
        # (``active`` is False), so the predictor direction is used unchanged.
        corrections = options.corrections
        assert not isinstance(corrections, str)  # normalized by Options.__post_init__
        self._corrector = select_corrector(corrections)
        self._problem_time_total = 0.0
        self._linear_eq_data = self._normalize_linear_eq(problem.linear_eq())
        self._n = int(problem.n_vars)
        # Derivative provenance from the resolver (§3.3); surfaced in Result.
        self._sources: DerivativeSources = getattr(
            problem, "sources", DerivativeSources(gradient="analytic", hessian="exact")
        )
        self._has_analytic_hessian: bool = getattr(
            problem, "has_analytic_hessian", True
        )

    @staticmethod
    def _normalize_linear_eq(
        data: tuple[Array | LinearOperator, Array] | None,
    ) -> tuple[LinearOperator, Array] | None:
        if data is None:
            return None
        matrix, rhs = data
        return as_operator(matrix), rhs

    # -- model evaluation -------------------------------------------------

    def _time_problem_call(self, callback: Callable[[], T]) -> T:
        """Run a user problem callback and accumulate its wall-clock seconds."""
        start = perf_counter()
        try:
            return callback()
        finally:
            self._problem_time_total += perf_counter() - start

    def _gradient(self, x: Array) -> Array:
        return self._time_problem_call(lambda: self._problem.gradient(x))

    def _objective(self, x: Array) -> float:
        return float(self._time_problem_call(lambda: self._problem.objective(x)))

    def _ineq(self, x: Array) -> Array:
        if not self._has_ineq:
            return self._xp.zeros((0,), dtype=x.dtype)
        return self._time_problem_call(lambda: self._problem.ineq_constraints(x))

    def _ineq_jac(self, x: Array) -> LinearOperator:
        if not self._has_ineq:
            return Dense(self._xp.zeros((0, self._n), dtype=x.dtype))
        jacobian = self._time_problem_call(lambda: self._problem.ineq_jacobian(x))
        return as_operator(jacobian)

    def _eq(self, x: Array) -> Array:
        """Combined equalities ``c(x)``: nonlinear stacked with ``A x − b``."""
        xp = self._xp
        parts: list[Array] = []
        if self._has_eq:
            parts.append(
                self._time_problem_call(lambda: self._problem.eq_constraints(x))
            )
        if self._linear_eq_data is not None:
            a_op, b = self._linear_eq_data
            parts.append(a_op.matvec(x) - b)
        if not parts:
            return xp.zeros((0,), dtype=x.dtype)
        if len(parts) == 1:
            return parts[0]
        return xp.concat(tuple(parts))

    def _eq_jac(self, x: Array) -> LinearOperator:
        """Combined equality Jacobian ``∇c(x)`` as a ``LinearOperator``."""
        ops: list[LinearOperator] = []
        if self._has_eq:
            jacobian = self._time_problem_call(lambda: self._problem.eq_jacobian(x))
            ops.append(as_operator(jacobian))
        if self._linear_eq_data is not None:
            ops.append(self._linear_eq_data[0])
        if not ops:
            return Dense(self._xp.zeros((0, self._n), dtype=x.dtype))
        if len(ops) == 1:
            return ops[0]
        return _VStack(tuple(ops))

    def _lagrangian_gradient(
        self,
        grad: Array,
        ineq_jac: LinearOperator,
        eq_jac: LinearOperator,
        y_eq: Array,
        y_ineq: Array,
        m: int,
        m_eq: int,
    ) -> Array:
        """``∇_x L = ∇f + ∇cᵀ y_eq + ∇gᵀ y_ineq`` — the L-BFGS curvature source."""
        lag = grad
        if m_eq > 0:
            lag = lag + eq_jac.rmatvec(y_eq)
        if m > 0:
            lag = lag + ineq_jac.rmatvec(y_ineq)
        return lag

    def _hessian(
        self,
        x: Array,
        y_eq_nonlinear: Array,
        y_ineq: Array,
        lbfgs: LBFGSOperator,
    ) -> LinearOperator:
        """Lagrangian Hessian operator, by precedence (§3.2, §4.3).

        - An analytic ``lagrangian_hessian`` is used whenever supplied, with the
          current nonlinear equality/inequality multipliers (so ``W`` carries
          the constraint curvature ``Σ y·∇²c`` for nonconvex problems). Linear
          equalities are intentionally sliced away because their Hessian term is
          zero by contract.
        - ``hessian="autodiff-hvp"``: exact Hessian-vector products of the
          Lagrangian via the backend autodiff adapter (no matrix formed).
        - ``hessian="lbfgs"`` (default): the persistent Powell-damped L-BFGS
          approximation, whose curvature pairs the driver updates each step.
        """
        if self._has_analytic_hessian:
            hessian = self._time_problem_call(
                lambda: self._problem.lagrangian_hessian(
                    x, y_eq_nonlinear, y_ineq, sigma=1.0
                )
            )
            return as_operator(hessian)
        if self._options.hessian == "autodiff-hvp":
            return self._autodiff_hvp_operator(x, y_eq_nonlinear, y_ineq)
        if self._options.hessian == "lbfgs":
            return lbfgs
        raise RuntimeError("no Hessian operator is available for this solve")

    def _autodiff_hvp_operator(
        self, x: Array, y_eq_nonlinear: Array, y_ineq: Array
    ) -> LinearOperator:
        """Matrix-free Hessian of the Lagrangian via autodiff HVPs (§4.3)."""
        adapter = get_autodiff_adapter(self._xp)
        if adapter is None:
            raise NotImplementedError(
                "autodiff-hvp Hessian requires a PyTorch/JAX backend"
            )
        xp = self._xp
        problem = self._problem
        has_eq = self._has_eq
        has_ineq = self._has_ineq

        def lagrangian(z: Array) -> Array:
            value = self._time_problem_call(lambda: problem.objective(z))
            if has_eq:
                constraints = self._time_problem_call(lambda: problem.eq_constraints(z))
                value = value + xp.sum(y_eq_nonlinear * constraints)
            if has_ineq:
                constraints = self._time_problem_call(
                    lambda: problem.ineq_constraints(z)
                )
                value = value + xp.sum(y_ineq * constraints)
            return value

        def matvec(v: Array) -> Array:
            return adapter.hvp(lagrangian, x, v)  # type: ignore[attr-defined]

        return MatrixFreeJacobian((self._n, self._n), matvec, matvec)

    # -- masks / bound bookkeeping ---------------------------------------

    def _masks(self, dtype: object) -> tuple[Array, Array, Array, Array]:
        xp = self._xp
        n = self._n
        if self._lower is None:
            mask_l = xp.zeros((n,), dtype=xp.bool)
            lower_safe = xp.zeros((n,), dtype=dtype)
        else:
            mask_l = xp.isfinite(self._lower)
            lower_safe = xp.where(mask_l, self._lower, xp.zeros((n,), dtype=dtype))
        if self._upper is None:
            mask_u = xp.zeros((n,), dtype=xp.bool)
            upper_safe = xp.zeros((n,), dtype=dtype)
        else:
            mask_u = xp.isfinite(self._upper)
            upper_safe = xp.where(mask_u, self._upper, xp.zeros((n,), dtype=dtype))
        return mask_l, mask_u, lower_safe, upper_safe

    # -- KKT error -------------------------------------------------------

    def kkt_error(
        self,
        *,
        grad: Array,
        ineq_jac: LinearOperator,
        m: int,
        g: Array,
        s: Array,
        y_ineq: Array,
        z_lower: Array,
        z_upper: Array,
        x_minus_l: Array,
        u_minus_x: Array,
        mask_l: Array,
        mask_u: Array,
        n_bounds: int,
        mu: float,
        c: Array,
        eq_jac: LinearOperator,
        m_eq: int,
        y_eq: Array,
    ) -> KKTResiduals:
        """Scaled KKT ∞-norm with IPOPT ``s_d, s_c`` scaling (§4.5)."""
        xp = self._xp
        r_d = grad
        if m > 0:
            r_d = r_d + ineq_jac.rmatvec(y_ineq)
        if m_eq > 0:
            r_d = r_d + eq_jac.rmatvec(y_eq)
        r_d = r_d - z_lower + z_upper
        dual_inf = _norm_inf(xp, r_d)

        prim = _norm_inf(xp, g + s) if m > 0 else 0.0
        if m_eq > 0:
            prim = max(prim, _norm_inf(xp, c))

        compl = 0.0
        if m > 0:
            compl = max(compl, _norm_inf(xp, s * y_ineq - mu))
        zero = xp.zeros_like(x_minus_l)
        compl = max(
            compl, _norm_inf(xp, xp.where(mask_l, x_minus_l * z_lower - mu, zero))
        )
        compl = max(
            compl, _norm_inf(xp, xp.where(mask_u, u_minus_x * z_upper - mu, zero))
        )

        n_dual = m + n_bounds + m_eq
        if n_dual > 0:
            sum_dual = (
                _norm1(xp, y_ineq)
                + _norm1(xp, z_lower)
                + _norm1(xp, z_upper)
                + _norm1(xp, y_eq)
            )
            s_d = max(_S_MAX, sum_dual / n_dual) / _S_MAX
            s_c = max(_S_MAX, sum_dual / n_dual) / _S_MAX
        else:
            s_d = 1.0
            s_c = 1.0
        return KKTResiduals(
            dual_infeasibility=dual_inf / s_d,
            primal_infeasibility=prim,
            complementarity=compl / s_c,
        )

    # -- main loop -------------------------------------------------------

    def run(self, x0: Array) -> Result:
        """Iterate to convergence and return the :class:`Result`."""
        run_start = perf_counter()
        xp = self._xp
        opts = self._options
        dtype = x0.dtype
        n = self._n

        mask_l, mask_u, lower_safe, upper_safe = self._masks(dtype)
        n_bounds = int(xp.sum(xp.astype(mask_l, xp.int64))) + int(
            xp.sum(xp.astype(mask_u, xp.int64))
        )

        m = int(self._ineq(x0).shape[0]) if self._has_ineq else 0
        if self._has_eq:
            eq0 = self._time_problem_call(lambda: self._problem.eq_constraints(x0))
            m_nonlinear_eq = int(eq0.shape[0])
        else:
            m_nonlinear_eq = 0
        m_eq = int(self._eq(x0).shape[0])
        delta_c = opts.regularization.delta_c if m_eq > 0 else 0.0
        ineq_fn: Callable[[Array], Array] | None = self._ineq if m > 0 else None

        start = initialize(
            xp=xp,
            x0=x0,
            lower_safe=lower_safe,
            upper_safe=upper_safe,
            mask_l=mask_l,
            mask_u=mask_u,
            ineq_fn=ineq_fn,
            mu_init=opts.barrier.mu_init,
            m=m,
        )
        x, s = start.x, start.s
        y_ineq, z_lower, z_upper = start.y_ineq, start.z_lower, start.z_upper
        y_eq = xp.zeros((m_eq,), dtype=dtype)

        if self._warm_start is not None:
            s, y_eq, y_ineq, z_lower, z_upper = apply_warm_start(
                xp=xp,
                warm=self._warm_start,
                s=s,
                y_eq=y_eq,
                y_ineq=y_ineq,
                z_lower=z_lower,
                z_upper=z_upper,
                m=m,
                m_eq=m_eq,
                n=n,
                mask_l=mask_l,
                mask_u=mask_u,
            )

        mu = opts.barrier.mu_init
        history: list[IterationRecord] = []
        alpha_p = 1.0
        alpha_d = 1.0
        reg_applied = 0.0
        status = Status.MAX_ITER
        message = "maximum iterations reached"
        optimality = ConditionChecker.for_optimality(opts.optimality)
        acceptable = ConditionChecker.for_acceptable(opts.acceptable)

        filt = Filter()
        line_search = FilterLineSearch(opts.line_search)
        breedveld = BreedveldController(opts.breedveld)
        use_breedveld = opts.globalization == "breedveld"

        ones = xp.ones((n,), dtype=dtype)

        # Persistent L-BFGS history (§4.3). Curvature pairs are pushed at the
        # top of each iteration from the just-completed step, so the Hessian
        # used this iteration reflects every accepted step so far.
        lbfgs = LBFGSOperator(n, opts.lbfgs)
        use_lbfgs = (not self._has_analytic_hessian) and opts.hessian == "lbfgs"
        prev_x: Array | None = None
        prev_grad: Array | None = None
        prev_ineq_jac: LinearOperator | None = None
        prev_eq_jac: LinearOperator | None = None
        problem_time_mark = self._problem_time_total
        last_step_solve_time = 0.0

        def bound_gaps(x: Array) -> tuple[Array, Array]:
            x_minus_l = xp.where(mask_l, x - lower_safe, ones)
            u_minus_x = xp.where(mask_u, upper_safe - x, ones)
            return x_minus_l, u_minus_x

        # Count of logged rows, used to reprint the header periodically.
        rows_logged = 0

        for it in range(opts.max_iter + 1):
            self._step_solve_seconds = 0.0
            x_minus_l, u_minus_x = bound_gaps(x)
            g = self._ineq(x)
            c = self._eq(x)
            grad = self._gradient(x)
            ineq_jac = self._ineq_jac(x)
            eq_jac = self._eq_jac(x)

            if use_lbfgs:
                if prev_x is not None:
                    # prev_* are set together with prev_x at the loop tail.
                    assert prev_grad is not None
                    assert prev_ineq_jac is not None and prev_eq_jac is not None
                    delta = x - prev_x
                    gamma = self._lagrangian_gradient(
                        grad, ineq_jac, eq_jac, y_eq, y_ineq, m, m_eq
                    ) - self._lagrangian_gradient(
                        prev_grad, prev_ineq_jac, prev_eq_jac, y_eq, y_ineq, m, m_eq
                    )
                    lbfgs.update(delta, gamma)
                prev_x, prev_grad = x, grad
                prev_ineq_jac, prev_eq_jac = ineq_jac, eq_jac

            err_kwargs = {
                "grad": grad,
                "ineq_jac": ineq_jac,
                "m": m,
                "g": g,
                "s": s,
                "y_ineq": y_ineq,
                "z_lower": z_lower,
                "z_upper": z_upper,
                "x_minus_l": x_minus_l,
                "u_minus_x": u_minus_x,
                "mask_l": mask_l,
                "mask_u": mask_u,
                "n_bounds": n_bounds,
                "c": c,
                "eq_jac": eq_jac,
                "m_eq": m_eq,
                "y_eq": y_eq,
            }
            residuals = self.kkt_error(mu=0.0, **err_kwargs)
            e0 = residuals.error
            theta = self._theta(
                x, g, s, c, m, m_eq, mask_l, mask_u, lower_safe, upper_safe
            )
            objective = self._objective(x)
            record_problem_time = self._problem_time_total - problem_time_mark
            record = IterationRecord(
                iteration=it,
                objective=objective,
                mu=mu,
                theta=theta,
                kkt_error=e0,
                alpha_primal=alpha_p,
                alpha_dual=alpha_d,
                regularization=reg_applied,
                dual_infeasibility=residuals.dual_infeasibility,
                primal_infeasibility=residuals.primal_infeasibility,
                complementarity=residuals.complementarity,
                problem_time=record_problem_time,
                step_solve_time=last_step_solve_time,
            )
            if self._record_transform is not None:
                record = self._record_transform(record)
            problem_time_mark = self._problem_time_total
            history.append(record)
            if logger.isEnabledFor(ITERATION):
                if rows_logged % HEADER_REPEAT_INTERVAL == 0:
                    logger.log(ITERATION, format_header())
                logger.log(
                    ITERATION,
                    format_record(
                        record, acceptable=acceptable.conditions_hold(record)
                    ),
                )
                rows_logged += 1

            stop_requested = self._invoke_callback(
                record, x, s, y_eq, y_ineq, z_lower, z_upper, m, m_eq
            )

            decision = optimality.observe(record) or acceptable.observe(record)
            if decision is not None:
                status = decision.status
                message = decision.message
                break
            if (
                opts.max_time is not None
                and perf_counter() - run_start >= opts.max_time
            ):
                status = Status.MAX_TIME
                message = "stopped: maximum wall time reached"
                break
            if stop_requested:
                status = Status.STOPPED
                message = "stopped: iteration callback requested termination"
                break
            if it == opts.max_iter:
                break

            sigma_s = y_ineq / s if m > 0 else xp.zeros((0,), dtype=dtype)
            sigma_l = xp.where(mask_l, z_lower / x_minus_l, xp.zeros_like(x))
            sigma_u = xp.where(mask_u, z_upper / u_minus_x, xp.zeros_like(x))
            sigma_x = sigma_l + sigma_u

            r_pi = g + s if m > 0 else xp.zeros((0,), dtype=dtype)
            w = self._hessian(x, y_eq[:m_nonlinear_eq], y_ineq, lbfgs)

            def solve_step_timed(
                w: LinearOperator,
                sigma_x: Array,
                sigma_s: Array,
                ineq_jac: LinearOperator,
                rhs_x: Array,
                eq_jac: LinearOperator,
                m_eq: int,
                r_y: Array,
                delta_c: float,
            ) -> tuple[Array, Array, float, bool]:
                problem_before = self._problem_time_total
                start = perf_counter()
                result = self._solve_step(
                    w, sigma_x, sigma_s, ineq_jac, rhs_x, eq_jac, m_eq, r_y, delta_c
                )
                elapsed = perf_counter() - start
                problem_elapsed = self._problem_time_total - problem_before
                self._step_solve_seconds += max(0.0, elapsed - problem_elapsed)
                return result

            rhs_kwargs: dict[str, Any] = {
                "grad": grad,
                "s": s,
                "ineq_jac": ineq_jac,
                "eq_jac": eq_jac,
                "y_eq": y_eq,
                "sigma_s": sigma_s,
                "r_pi": r_pi,
                "mask_l": mask_l,
                "mask_u": mask_u,
                "x_minus_l": x_minus_l,
                "u_minus_x": u_minus_x,
                "m": m,
                "m_eq": m_eq,
            }
            recover_kwargs: dict[str, Any] = {
                "xp": xp,
                "ineq_jac": ineq_jac,
                "m": m,
                "s": s,
                "y_ineq": y_ineq,
                "r_pi": r_pi,
                "sigma_s": sigma_s,
                "z_lower": z_lower,
                "z_upper": z_upper,
                "sigma_l": sigma_l,
                "sigma_u": sigma_u,
                "x_minus_l": x_minus_l,
                "u_minus_x": u_minus_x,
                "mask_l": mask_l,
                "mask_u": mask_u,
            }

            if self._corrector.active:
                # Mehrotra/Gondzio: the affine and correction directions
                # share one KKT operator. Sparse-direct reuses its factorization;
                # dense/Krylov perform another solve against that operator.
                corrected = self._corrected_step(
                    w=w,
                    sigma_x=sigma_x,
                    sigma_s=sigma_s,
                    ineq_jac=ineq_jac,
                    eq_jac=eq_jac,
                    c=c,
                    delta_c=delta_c,
                    m_eq=m_eq,
                    solve_step_timed=solve_step_timed,
                    rhs_kwargs=rhs_kwargs,
                    recover_kwargs=recover_kwargs,
                )
                if corrected is None:
                    status = Status.NUMERICAL_ERROR
                    message = "condensed factorization failed despite regularization"
                    break
                mu, step, reg_applied = corrected
            else:
                mu = self._reduce_mu(mu, **err_kwargs)
                rhs = self._condensed_rhs(mu, mu, mu, **rhs_kwargs)
                dx, dy_eq, reg_applied, ok = solve_step_timed(
                    w, sigma_x, sigma_s, ineq_jac, rhs, eq_jac, m_eq, -c, delta_c
                )
                if not ok:
                    status = Status.NUMERICAL_ERROR
                    message = "condensed factorization failed despite regularization"
                    break
                step = recover_eliminated(dx, mu=mu, dy_eq=dy_eq, **recover_kwargs)

            if use_breedveld:
                tau = opts.breedveld.tau
            else:
                # Predictor-corrector targets can be far below machine epsilon
                # (or zero after an affine fallback). Use the configured barrier
                # floor for the boundary fraction so ``1 - μ`` never rounds to
                # exactly one and slacks/bound gaps remain strictly positive.
                mu_fraction = max(
                    mu, opts.barrier.mu_min, opts.optimality.kkt_tol / 10.0
                )
                tau = max(opts.barrier.tau_min, 1.0 - mu_fraction)
            alpha_p_max = self._alpha_primal(
                step, s, m, x_minus_l, u_minus_x, mask_l, mask_u, tau
            )
            alpha_d = self._alpha_dual(
                step, y_ineq, z_lower, z_upper, m, mask_l, mask_u, tau
            )

            # Bind the loop-varying state as defaults (the closure is consumed
            # within this iteration, but this keeps it explicit and lint-clean).
            def eval_point(
                alpha: float,
                x: Array = x,
                s: Array = s,
                step: NewtonStep = step,
                mu: float = mu,
            ) -> tuple[float, float]:
                x_t = x + alpha * step.dx
                s_t = s + alpha * step.ds if m > 0 else s
                return (
                    self._theta_l1(x_t, s_t, m, m_eq),
                    self._phi(x_t, s_t, mu, m, mask_l, mask_u, lower_safe, upper_safe),
                )

            soc_primal: tuple[Array, Array] | None = None

            def is_strictly_interior(x_t: Array, s_t: Array) -> bool:
                if m > 0 and not bool(xp.all(s_t > 0.0)):
                    return False
                inside_l = xp.logical_or(xp.logical_not(mask_l), x_t > lower_safe)
                inside_u = xp.logical_or(xp.logical_not(mask_u), x_t < upper_safe)
                return bool(xp.all(xp.logical_and(inside_l, inside_u)))

            def soc(
                alpha: float,
                x: Array = x,
                s: Array = s,
                step: NewtonStep = step,
                ineq_jac: LinearOperator = ineq_jac,
                sigma_s: Array = sigma_s,
                sigma_x: Array = sigma_x,
                w: LinearOperator = w,
                eq_jac: LinearOperator = eq_jac,
                mu: float = mu,
            ) -> tuple[float, float] | None:
                """Second-order correction for nonlinear constraint residuals."""
                nonlocal soc_primal
                if opts.line_search.max_soc <= 0:
                    return None

                base_x = x + alpha * step.dx
                base_s = s + alpha * step.ds if m > 0 else s
                corr_x = xp.zeros_like(x)
                corr_s = xp.zeros_like(s)
                empty_ineq = xp.zeros((0,), dtype=dtype)

                for _ in range(opts.line_search.max_soc):
                    x_c = base_x + corr_x
                    s_c = base_s + corr_s if m > 0 else s
                    if not is_strictly_interior(x_c, s_c):
                        return None

                    c_c = self._eq(x_c)
                    g_c = self._ineq(x_c)
                    r_pi_c = g_c + s_c if m > 0 else empty_ineq
                    rhs_soc = xp.zeros_like(step.dx)
                    if m > 0:
                        rhs_soc = rhs_soc - ineq_jac.rmatvec(sigma_s * r_pi_c)

                    dx_c, _, _, ok = solve_step_timed(
                        w,
                        sigma_x,
                        sigma_s,
                        ineq_jac,
                        rhs_soc,
                        eq_jac,
                        m_eq,
                        -c_c,
                        delta_c,
                    )
                    if not ok:
                        return None

                    corr_x = corr_x + dx_c
                    if m > 0:
                        corr_s = corr_s - r_pi_c - ineq_jac.matvec(dx_c)

                x_soc = base_x + corr_x
                s_soc = base_s + corr_s if m > 0 else s
                if not is_strictly_interior(x_soc, s_soc):
                    return None

                soc_primal = (
                    alpha * step.dx + corr_x,
                    alpha * step.ds + corr_s if m > 0 else step.ds,
                )
                return (
                    self._theta_l1(x_soc, s_soc, m, m_eq),
                    self._phi(
                        x_soc,
                        s_soc,
                        mu,
                        m,
                        mask_l,
                        mask_u,
                        lower_safe,
                        upper_safe,
                    ),
                )

            theta0 = self._theta_l1(x, s, m, m_eq)
            phi0 = self._phi(x, s, mu, m, mask_l, mask_u, lower_safe, upper_safe)
            dphi = self._dphi(
                x, s, step, grad, mu, m, mask_l, mask_u, x_minus_l, u_minus_x
            )

            used_soc = False
            if use_breedveld:
                alpha_p, restoration = breedveld.search(
                    alpha_max=alpha_p_max,
                    theta0=theta0,
                    phi0=phi0,
                    dphi=dphi,
                    eval_point=eval_point,
                    iteration=it,
                )
            else:
                result = line_search.search(
                    alpha_max=alpha_p_max,
                    theta0=theta0,
                    phi0=phi0,
                    dphi=dphi,
                    eval_point=eval_point,
                    entries=filt.entries,
                    soc=soc,
                )
                alpha_p = result.alpha
                restoration = result.restoration
                used_soc = result.used_soc
                if result.accepted and result.augment:
                    filt.augment(theta0, phi0)

            if restoration:
                logger.debug("iter %d: entering feasibility restoration", it)
                filt.augment(theta0, phi0)
                x, s, infeasible = self._restore(
                    x, s, m, m_eq, mask_l, mask_u, lower_safe, upper_safe
                )
                if infeasible:
                    status = Status.INFEASIBLE
                    message = (
                        "locally infeasible: restoration could not reduce the "
                        "constraint violation"
                    )
                    break
                alpha_p = 0.0
                # The restoration jump is not a Newton step, so the next
                # curvature pair would be meaningless — drop the history anchor.
                prev_x = None
                last_step_solve_time = self._step_solve_seconds
                continue

            if used_soc and soc_primal is not None:
                dx_primal, ds_primal = soc_primal
                x = x + dx_primal
            else:
                x = x + alpha_p * step.dx
            if m > 0:
                if used_soc and soc_primal is not None:
                    s = s + ds_primal
                else:
                    s = s + alpha_p * step.ds
                y_ineq = y_ineq + alpha_d * step.dy_ineq
            if m_eq > 0:
                y_eq = y_eq + alpha_p * step.dy_eq
            z_lower = xp.where(
                mask_l, z_lower + alpha_d * step.dz_lower, xp.zeros_like(x)
            )
            z_upper = xp.where(
                mask_u, z_upper + alpha_d * step.dz_upper, xp.zeros_like(x)
            )
            last_step_solve_time = self._step_solve_seconds

        x_minus_l, u_minus_x = bound_gaps(x)
        g = self._ineq(x)
        c = self._eq(x)
        final_theta = self._theta(
            x, g, s, c, m, m_eq, mask_l, mask_u, lower_safe, upper_safe
        )
        final_kkt = history[-1].kkt_error
        final_record = history[-1]

        # The human-readable result/timing summary is emitted by ``solve`` once
        # the solution has been unscaled; the driver only assembles the Result.
        logger.debug(
            "driver finished: status=%s after %d iteration(s)",
            status.value,
            len(history),
        )

        return Result(
            status=status,
            x=x,
            objective=self._objective(x),
            y_eq=y_eq if m_eq > 0 else None,
            y_ineq=y_ineq if m > 0 else None,
            z_lower=z_lower if self._lower is not None else None,
            z_upper=z_upper if self._upper is not None else None,
            n_iter=len(history),
            kkt_error=final_kkt,
            dual_infeasibility=final_record.dual_infeasibility,
            primal_infeasibility=final_record.primal_infeasibility,
            complementarity=final_record.complementarity,
            constraint_violation=final_theta,
            derivative_sources=self._sources,
            history=tuple(history),
            message=message,
        )

    # -- helpers ---------------------------------------------------------

    def _invoke_callback(
        self,
        record: IterationRecord,
        x: Array,
        s: Array,
        y_eq: Array,
        y_ineq: Array,
        z_lower: Array,
        z_upper: Array,
        m: int,
        m_eq: int,
    ) -> bool:
        """Call the user iteration hook; ``True`` means a stop was requested.

        Optional blocks collapse to ``None`` (matching :class:`Result`) so the
        callback only sees state that actually exists for this problem.
        """
        if self._callback is None:
            return False
        info = IterationInfo(
            record=record,
            x=x,
            s=s if m > 0 else None,
            y_eq=y_eq if m_eq > 0 else None,
            y_ineq=y_ineq if m > 0 else None,
            z_lower=z_lower if self._lower is not None else None,
            z_upper=z_upper if self._upper is not None else None,
        )
        return bool(self._callback(info))

    def _theta(
        self,
        x: Array,
        g: Array,
        s: Array,
        c: Array,
        m: int,
        m_eq: int,
        mask_l: Array,
        mask_u: Array,
        lower_safe: Array,
        upper_safe: Array,
    ) -> float:
        """Original-problem violation: ``c``, ``max(g, 0)`` and bound overshoot."""
        xp = self._xp
        zero = xp.zeros_like(x)
        viol = 0.0
        if m > 0:
            viol = max(viol, _norm_inf(xp, xp.maximum(g, xp.zeros_like(g))))
        if m_eq > 0:
            viol = max(viol, _norm_inf(xp, c))
        viol = max(
            viol,
            _norm_inf(xp, xp.maximum(xp.where(mask_l, lower_safe - x, zero), zero)),
        )
        viol = max(
            viol,
            _norm_inf(xp, xp.maximum(xp.where(mask_u, x - upper_safe, zero), zero)),
        )
        return viol

    def _reduce_mu(self, mu: float, **err_kwargs: object) -> float:
        opts = self._options
        for _ in range(_MAX_MU_REDUCTIONS):
            e_mu = self.kkt_error(mu=mu, **err_kwargs).error  # type: ignore[arg-type]
            if mu <= opts.barrier.mu_min or e_mu > _KAPPA_EPSILON * mu:
                return mu
            new_mu = update_mu(mu, opts.barrier, opts.optimality.kkt_tol)
            if new_mu >= mu:
                return mu
            mu = new_mu
        return mu

    # -- condensed RHS / corrector primitives ----------------------------

    def _condensed_rhs(
        self,
        target_s: Array | float,
        target_l: Array | float,
        target_u: Array | float,
        *,
        grad: Array,
        s: Array,
        ineq_jac: LinearOperator,
        eq_jac: LinearOperator,
        y_eq: Array,
        sigma_s: Array,
        r_pi: Array,
        mask_l: Array,
        mask_u: Array,
        x_minus_l: Array,
        u_minus_x: Array,
        m: int,
        m_eq: int,
    ) -> Array:
        """Right-hand side of the condensed x-system for complementarity targets.

        ``target_*`` is the per-component complementarity target ``τ`` (scalar
        ``μ`` for the standard step; vectors for the corrector). The terms
        ``τ/s``, ``τ_L/(x−x_L)``, ``τ_U/(x_U−x)`` are the only places ``τ``
        enters the condensed reduction (§2.3).
        """
        xp = self._xp
        zeros = xp.zeros_like(grad)
        rhs = -grad
        if m > 0:
            rhs = rhs - ineq_jac.rmatvec(sigma_s * r_pi + target_s / s)
        if m_eq > 0:
            rhs = rhs - eq_jac.rmatvec(y_eq)
        rhs = rhs + xp.where(mask_l, target_l / x_minus_l, zeros)
        rhs = rhs - xp.where(mask_u, target_u / u_minus_x, zeros)
        return rhs

    def _solve_targets_reused_operator(
        self,
        comp_s: Array,
        comp_l: Array,
        comp_u: Array,
        *,
        c: Array,
        m_eq: int,
        rhs_kwargs: dict[str, Any],
        recover_kwargs: dict[str, Any],
    ) -> NewtonStep | None:
        """Solve the current condensed system for new complementarity targets.

        The sparse-direct route reuses the factorization established by the
        affine solve. Dense and Krylov solvers reuse the operator but perform a
        fresh direct/iterative solve. Returns ``None`` on a failed/non-finite
        solve so the corrector can fall back to the affine step.
        """
        xp = self._xp
        rhs_x = self._condensed_rhs(comp_s, comp_l, comp_u, **rhs_kwargs)
        rhs = xp.concat((rhs_x, -c)) if m_eq > 0 else rhs_x
        start = perf_counter()
        try:
            sol = self._solver.solve(rhs)
        except LinearSolveError:
            self._step_solve_seconds += perf_counter() - start
            return None
        self._step_solve_seconds += perf_counter() - start
        if not bool(xp.all(xp.isfinite(sol))):
            return None
        dx_t = sol[: self._n]
        dy_t = sol[self._n :] if m_eq > 0 else xp.zeros((0,), dtype=rhs.dtype)
        return recover_eliminated(
            dx_t,
            mu=0.0,
            comp_s=comp_s,
            comp_l=comp_l,
            comp_u=comp_u,
            dy_eq=dy_t,
            **recover_kwargs,
        )

    def _corrected_step(
        self,
        *,
        w: LinearOperator,
        sigma_x: Array,
        sigma_s: Array,
        ineq_jac: LinearOperator,
        eq_jac: LinearOperator,
        c: Array,
        delta_c: float,
        m_eq: int,
        solve_step_timed: Callable[..., tuple[Array, Array, float, bool]],
        rhs_kwargs: dict[str, Any],
        recover_kwargs: dict[str, Any],
    ) -> tuple[float, NewtonStep, float] | None:
        """Affine predictor + higher-order corrector.

        Returns ``(μ, step, δ_w)`` where ``μ`` is the barrier target the
        corrected ``step`` aims at (adopted by the line search), or ``None`` if
        the affine operator setup/solve failed despite regularization.
        """
        affine_rhs = self._condensed_rhs(0.0, 0.0, 0.0, **rhs_kwargs)
        dx_aff, dy_aff, reg_applied, ok = solve_step_timed(
            w, sigma_x, sigma_s, ineq_jac, affine_rhs, eq_jac, m_eq, -c, delta_c
        )
        if not ok:
            return None
        affine_step = recover_eliminated(dx_aff, mu=0.0, dy_eq=dy_aff, **recover_kwargs)
        solve_targets = functools.partial(
            self._solve_targets_reused_operator,
            c=c,
            m_eq=m_eq,
            rhs_kwargs=rhs_kwargs,
            recover_kwargs=recover_kwargs,
        )
        alpha_primal = functools.partial(
            self._alpha_primal,
            s=recover_kwargs["s"],
            m=recover_kwargs["m"],
            x_minus_l=recover_kwargs["x_minus_l"],
            u_minus_x=recover_kwargs["u_minus_x"],
            mask_l=recover_kwargs["mask_l"],
            mask_u=recover_kwargs["mask_u"],
        )
        alpha_dual = functools.partial(
            self._alpha_dual,
            y_ineq=recover_kwargs["y_ineq"],
            z_lower=recover_kwargs["z_lower"],
            z_upper=recover_kwargs["z_upper"],
            m=recover_kwargs["m"],
            mask_l=recover_kwargs["mask_l"],
            mask_u=recover_kwargs["mask_u"],
        )
        context = CorrectionContext(
            affine=affine_step,
            s=recover_kwargs["s"],
            y_ineq=recover_kwargs["y_ineq"],
            x_minus_l=recover_kwargs["x_minus_l"],
            u_minus_x=recover_kwargs["u_minus_x"],
            z_lower=recover_kwargs["z_lower"],
            z_upper=recover_kwargs["z_upper"],
            mask_l=recover_kwargs["mask_l"],
            mask_u=recover_kwargs["mask_u"],
            solve=solve_targets,
            alpha_primal=alpha_primal,
            alpha_dual=alpha_dual,
            mu_min=max(
                self._options.barrier.mu_min, self._options.optimality.kkt_tol / 10.0
            ),
        )
        result = self._corrector.correct(context)
        return result.mu, result.step, reg_applied

    def _inertia_acceptable(self, operator: LinearOperator) -> bool:
        """IPOPT inertia check on a successful sparse LDLᵀ factorization.

        Best-effort and solver-agnostic (invariant #3): it only engages when the
        injected solver reports the factor's inertia (a sparse LDLᵀ backend such
        as Feral / cuDSS) *and* the KKT operator knows its target inertia. In
        every other case — dense Cholesky / Krylov (no inertia, but Cholesky
        already fails on indefiniteness), a non-inertia-revealing factorization,
        or an L-BFGS low-rank Hessian (PD by Powell damping) — it returns ``True``
        and leaves correction to the factorization-failure escalation.
        """
        target_fn = getattr(operator, "expected_inertia", None)
        target = target_fn() if target_fn is not None else None
        if target is None:
            return True
        inertia_fn = getattr(self._solver, "inertia_or_none", None)
        actual = inertia_fn() if inertia_fn is not None else None
        if actual is None:
            return True
        return bool(actual == target)

    def _solve_step(
        self,
        w: LinearOperator,
        sigma_x: Array,
        sigma_s: Array,
        ineq_jac: LinearOperator,
        rhs_x: Array,
        eq_jac: LinearOperator,
        m_eq: int,
        r_y: Array,
        delta_c: float,
    ) -> tuple[Array, Array, float, bool]:
        """Factor/solve the condensed (or bordered saddle) step (§2.3, §4.4).

        Returns ``(Δx, Δy_eq, δ_w, ok)``. With equalities present the condensed
        block is bordered into the quasidefinite saddle (Friedlander–Orban) and
        solved through the same injected ``LinearSolver``; Breedveld δ_w
        escalation handles a failed factorization.
        """
        xp = self._xp
        reg = RegularizationState()
        sigma_x_op = Diagonal(sigma_x)
        sigma_s_op = Diagonal(sigma_s)
        empty = xp.zeros((0,), dtype=rhs_x.dtype)
        for _ in range(_MAX_REG_ATTEMPTS):
            condensed = build_condensed_operator(
                w, sigma_x_op, sigma_s_op, ineq_jac, reg
            )
            if m_eq > 0:
                operator = build_saddle_operator(condensed, eq_jac, delta_c)
                rhs = xp.concat((rhs_x, r_y))
            else:
                operator = condensed
                rhs = rhs_x
            try:
                self._solver.factor(operator)
                sol = self._solver.solve(rhs)
            except LinearSolveError:
                escalate_delta_w(reg, self._options.regularization)
                logger.debug(
                    "factorization failed; escalating delta_w to %.2e", reg.delta_w
                )
                continue
            if not self._inertia_acceptable(operator):
                # A symmetric-indefinite LDLᵀ can succeed with the *wrong* inertia
                # (a non-descent step the failure path never sees); IPOPT bumps
                # δ_w until the (1,1) block is PD (Wächter & Biegler 2006, §3.1).
                escalate_delta_w(reg, self._options.regularization)
                logger.debug(
                    "KKT inertia mismatch; escalating delta_w to %.2e", reg.delta_w
                )
                continue
            if bool(xp.all(xp.isfinite(sol))):
                dx = sol[: self._n]
                dy = sol[self._n :] if m_eq > 0 else empty
                return dx, dy, reg.delta_w, True
            escalate_delta_w(reg, self._options.regularization)
            logger.debug("non-finite step; escalating delta_w to %.2e", reg.delta_w)
        return rhs_x, empty, reg.delta_w, False

    def _alpha_primal(
        self,
        step: NewtonStep,
        s: Array,
        m: int,
        x_minus_l: Array,
        u_minus_x: Array,
        mask_l: Array,
        mask_u: Array,
        tau: float,
    ) -> float:
        xp = self._xp
        alpha = 1.0
        if m > 0:
            alpha = min(alpha, fraction_to_boundary(s, step.ds, tau))
        v_l = xp.where(mask_l, x_minus_l, xp.ones_like(x_minus_l))
        dv_l = xp.where(mask_l, step.dx, xp.zeros_like(x_minus_l))
        alpha = min(alpha, fraction_to_boundary(v_l, dv_l, tau))
        v_u = xp.where(mask_u, u_minus_x, xp.ones_like(u_minus_x))
        dv_u = xp.where(mask_u, -step.dx, xp.zeros_like(u_minus_x))
        alpha = min(alpha, fraction_to_boundary(v_u, dv_u, tau))
        return alpha

    def _alpha_dual(
        self,
        step: NewtonStep,
        y_ineq: Array,
        z_lower: Array,
        z_upper: Array,
        m: int,
        mask_l: Array,
        mask_u: Array,
        tau: float,
    ) -> float:
        xp = self._xp
        alpha = 1.0
        if m > 0:
            alpha = min(alpha, fraction_to_boundary(y_ineq, step.dy_ineq, tau))
        v_l = xp.where(mask_l, z_lower, xp.ones_like(z_lower))
        dv_l = xp.where(mask_l, step.dz_lower, xp.zeros_like(z_lower))
        alpha = min(alpha, fraction_to_boundary(v_l, dv_l, tau))
        v_u = xp.where(mask_u, z_upper, xp.ones_like(z_upper))
        dv_u = xp.where(mask_u, step.dz_upper, xp.zeros_like(z_upper))
        alpha = min(alpha, fraction_to_boundary(v_u, dv_u, tau))
        return alpha

    def _phi(
        self,
        x: Array,
        s: Array,
        mu: float,
        m: int,
        mask_l: Array,
        mask_u: Array,
        lower_safe: Array,
        upper_safe: Array,
    ) -> float:
        """Barrier objective ``φ_μ`` (Wächter & Biegler 2006, §2)."""
        xp = self._xp
        val = self._objective(x)
        x_minus_l = xp.where(mask_l, x - lower_safe, xp.ones_like(x))
        u_minus_x = xp.where(mask_u, upper_safe - x, xp.ones_like(x))
        if m > 0:
            val = val - mu * float(xp.sum(xp.log(s)))
        val = val - mu * float(
            xp.sum(xp.where(mask_l, xp.log(x_minus_l), xp.zeros_like(x)))
        )
        val = val - mu * float(
            xp.sum(xp.where(mask_u, xp.log(u_minus_x), xp.zeros_like(x)))
        )
        return val

    def _theta_l1(self, x: Array, s: Array, m: int, m_eq: int) -> float:
        """Constraint violation ``θ = ‖(c, g+s)‖₁`` used by the filter."""
        xp = self._xp
        val = 0.0
        if m > 0:
            val += _norm1(xp, self._ineq(x) + s)
        if m_eq > 0:
            val += _norm1(xp, self._eq(x))
        return val

    def _dphi(
        self,
        x: Array,
        s: Array,
        step: NewtonStep,
        grad: Array,
        mu: float,
        m: int,
        mask_l: Array,
        mask_u: Array,
        x_minus_l: Array,
        u_minus_x: Array,
    ) -> float:
        """Directional derivative ``∇φ_μᵀ d`` along the step."""
        xp = self._xp
        dd = float(xp.sum(grad * step.dx))
        if m > 0:
            dd -= mu * float(xp.sum(step.ds / s))
        dd -= mu * float(
            xp.sum(xp.where(mask_l, step.dx / x_minus_l, xp.zeros_like(x)))
        )
        dd += mu * float(
            xp.sum(xp.where(mask_u, step.dx / u_minus_x, xp.zeros_like(x)))
        )
        return dd

    def _restore(
        self,
        x: Array,
        s: Array,
        m: int,
        m_eq: int,
        mask_l: Array,
        mask_u: Array,
        lower_safe: Array,
        upper_safe: Array,
    ) -> tuple[Array, Array, bool]:
        """Run the Gauss-Newton restoration phase (delegates to ``restore``)."""
        return restore(
            xp=self._xp,
            x=x,
            s=s,
            m=m,
            m_eq=m_eq,
            eq_fn=self._eq,
            eq_jac_fn=self._eq_jac,
            ineq_fn=self._ineq,
            ineq_jac_fn=self._ineq_jac,
            mask_l=mask_l,
            mask_u=mask_u,
            lower_safe=lower_safe,
            upper_safe=upper_safe,
            tol=self._options.optimality.kkt_tol,
        )


__all__ = ["IPMDriver"]
