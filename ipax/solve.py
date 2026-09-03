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

"""Top-level convenience entry point."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING

from ipax._logging import (
    ITERATION,
    OPTIONS,
    PROBLEM,
    RESULT,
    SOLVER,
    configure_verbosity,
    format_options,
    format_problem,
    format_result,
    format_setup,
    format_solver,
    format_timing,
    logger,
)
from ipax.backend.namespace import array_namespace, capabilities
from ipax.ipm.driver import IPMDriver
from ipax.ipm.init import relax_fixed_bounds
from ipax.linalg.solver import select_restoration_solver, select_solver
from ipax.options import Options, ScalingOptions
from ipax.problem.derivatives import resolve
from ipax.problem.linear_ineq import lower_linear_inequalities
from ipax.problem.scaling import ProblemScaling, ScaledProblem, compute_scaling
from ipax.result import (
    IterationInfo,
    IterationRecord,
    Result,
    Routes,
    Status,
    WarmStart,
)

if TYPE_CHECKING:
    from ipax.backend.operators import LinearOperator
    from ipax.problem.base import Problem
    from ipax.result import IterationCallback
    from ipax.typing import Array, Namespace


def _has_nonlinear_equalities(problem: Problem, x: Array) -> bool:
    try:
        values = problem.eq_constraints(x)
    except NotImplementedError:
        return False
    return int(values.shape[0]) > 0


def _count_inequalities(problem: Problem, x: Array) -> int:
    """Total inequality rows at ``x`` (nonlinear + lowered linear), 0 if none."""
    try:
        values = problem.ineq_constraints(x)
    except NotImplementedError:
        return 0
    return int(values.shape[0])


def _bound_violation(
    xp: Namespace,
    x: Array,
    lower: Array | None,
    upper: Array | None,
) -> float:
    zero = xp.zeros_like(x)
    violation = 0.0
    if lower is not None:
        violation = max(violation, float(xp.max(xp.maximum(lower - x, zero))))
    if upper is not None:
        violation = max(violation, float(xp.max(xp.maximum(x - upper, zero))))
    return violation


def solve(
    problem: Problem,
    x0: Array,
    *,
    options: Options | None = None,
    callback: IterationCallback | None = None,
    warm_start: WarmStart | None = None,
) -> Result:
    """Solve ``problem`` starting from ``x0``.

    The interior-point path handles bounds, (nonlinear) inequalities, and
    equalities through the condensed normal-equations / regularized-saddle route
    (Breedveld 2017, eq. 18) with an injected :class:`LinearSolver` — the dense
    reference solver, the matrix-free Krylov solver (``linsolve="krylov"``,
    auto-selected at scale), or the sparse-direct solver (``linsolve="sparse"``,
    which assembles and factors the saddle for bound/equality problems with an
    assemblable Hessian) — monotone μ, fraction-to-boundary, the filter
    line-search, and feasibility restoration.

    With ``options.scaling="gradient-based"`` the objective and each
    constraint are rescaled once at ``x0`` so their gradients are ``O(1)``
    (IPOPT ``nlp_scaling_method``); the returned ``x``, objective, and
    multipliers are reported in the original problem's scale, while
    ``kkt_error``/``constraint_violation`` remain the scaled-space metrics that
    drove convergence.

    A full :class:`~ipax.options.ScalingOptions` object can be supplied for a
    custom ``max_gradient``.

    ``warm_start``, if given, seeds the interior-point iterate with supplied
    slacks/multipliers instead of the default μ-complementarity start — pass a
    :class:`~ipax.result.WarmStart` (e.g. ``WarmStart.from_result(prev)``) and
    the corresponding ``x0`` when re-solving a perturbed problem. The values are
    in the original problem's units; with scaling enabled they are rescaled
    internally to match the scaled subproblem.

    ``callback``, if given, is invoked once per iteration with an
    :class:`~ipax.result.IterationInfo` snapshot (the iteration record plus the
    current primal/dual iterate). Returning a truthy value stops the solve
    early with :attr:`~ipax.result.Status.STOPPED`. Progress logging is emitted
    through the ``"ipax"`` logger; ``options.verbose`` (1=info, 2=debug) opts in
    to a console handler, while applications may instead configure that logger
    themselves.

    Termination is configured by ``options``: the single-iteration
    :class:`~ipax.options.OptimalityConditionOptions` (reports
    ``Status.OPTIMAL``), the multi-iteration
    :class:`~ipax.options.AcceptableStoppingOptions` (reports
    ``Status.ACCEPTABLE``), and the top-level ``max_iter`` / ``max_time`` limits.
    """
    start_time = perf_counter()
    opts = Options() if options is None else options
    scaling_options = opts.scaling
    if not isinstance(scaling_options, ScalingOptions):  # normalized by Options
        raise RuntimeError("scaling options were not normalized")
    configure_verbosity(opts.verbose)
    xp = array_namespace(x0)
    lower, upper = problem.bounds()

    if lower is not None and upper is not None and bool(xp.any(lower > upper)):
        return Result(
            status=Status.INFEASIBLE,
            x=x0,
            objective=float(problem.objective(x0)),
            n_iter=0,
            kkt_error=float("inf"),
            constraint_violation=_bound_violation(xp, x0, lower, upper),
            solve_time=perf_counter() - start_time,
            device=_describe_device(x0),
            message="infeasible bounds: x_L > x_U",
        )

    # Relax fixed / near-degenerate bound pairs (x_L == x_U) so the barrier has a
    # strict interior; without this the first Newton step is non-finite (§3.6).
    lower, upper = relax_fixed_bounds(xp, lower, upper)

    # Bind gradient/Jacobian/Hessian sources by precedence (§3.2). The driver
    # consumes the resolved problem, so it always has the derivatives it needs.
    resolved: Problem = resolve(problem, xp, opts)
    # Lower any two-sided ``l ≤ A x ≤ u`` block into the one-sided inequality
    # machinery so the driver and every solver route handle it unchanged. A
    # no-op when the problem declares no ``linear_ineq``.
    resolved = lower_linear_inequalities(resolved, x0, xp)

    m_ineq = _count_inequalities(resolved, x0)
    has_ineq = m_ineq > 0
    has_eq = _has_nonlinear_equalities(resolved, x0)
    has_equalities = has_eq or resolved.linear_eq() is not None

    # NLP auto-scaling: rescale once at x0 so gradients are O(1); the
    # driver runs on the scaled problem and the result is unscaled below.
    problem_scaling: ProblemScaling | None = None
    model: Problem = resolved
    if scaling_options.method == "gradient-based":
        problem_scaling = compute_scaling(
            resolved,
            x0,
            xp,
            has_eq=has_eq,
            has_ineq=has_ineq,
            max_gradient=scaling_options.max_gradient,
        )
        model = ScaledProblem(resolved, problem_scaling)

    effective_warm = warm_start
    if warm_start is not None and problem_scaling is not None:
        effective_warm = _scale_warm_start(warm_start, problem_scaling)

    # Lazy structural probes for the tall n ≪ m selection heuristic: they
    # evaluate the (scaled) inequality Jacobian once at ``x0`` — only when
    # ``select_solver`` actually reaches the tall-problem branch — and share
    # the evaluated operator so the Jacobian is built at most once.
    _probe_cache: list[LinearOperator | None] = []

    def _ineq_jac_probe() -> LinearOperator | None:
        if not _probe_cache:
            from ipax.backend.operators import as_operator

            try:
                _probe_cache.append(as_operator(model.ineq_jacobian(x0)))
            except NotImplementedError:
                _probe_cache.append(None)
        return _probe_cache[0]

    def _ineq_gram_capable() -> bool:
        """Whether ∇g can form ``∇gᵀ diag(w) ∇g`` without densifying to m×n."""
        op = _ineq_jac_probe()
        return op is not None and op.gram_capable()

    def _ineq_density() -> float | None:
        """``nnz/(m·n)`` of ∇g, or ``None`` without explicit COO structure."""
        op = _ineq_jac_probe()
        if op is None:
            return None
        rows, cols = op.shape
        if rows == 0 or cols == 0:
            return None
        try:
            values = op.to_coo()[2]
        except NotImplementedError:
            return None
        return int(values.shape[0]) / (int(rows) * int(cols))

    def _eq_jac_coo_capable() -> bool:
        """Whether the combined equality Jacobian can emit COO triplets.

        The sparse normal-equations saddle keeps ``∇c`` as an explicit border
        in the factored matrix, so a matrix-free equality Jacobian must veto
        the route here rather than crash at the first factorization.
        """
        from ipax.backend.operators import as_operator

        ops: list[LinearOperator] = []
        if has_eq:
            try:
                ops.append(as_operator(model.eq_jacobian(x0)))
            except NotImplementedError:
                return False
        linear = model.linear_eq()
        if linear is not None:
            ops.append(as_operator(linear[0]))
        for op in ops:
            try:
                op.to_coo()
            except NotImplementedError:
                return False
        return True

    def _ineq_gram_fill() -> float | None:
        """Estimated Gram-pattern density of ∇g (sampled column overlap)."""
        op = _ineq_jac_probe()
        if op is None or not op.gram_coo_capable():
            return None
        if has_equalities and not _eq_jac_coo_capable():
            return None
        return op.gram_fill_estimate()

    # The sparse normal-equations form folds the Hessian into its n×n block
    # only for a diagonal+low-rank W (the L-BFGS route); with an analytic or
    # HVP Hessian the auto heuristic must not gamble on the operator being
    # COO-emittable, so the fill probe is withheld and the form stays
    # explicitly selectable via SparseOptions(kkt_route="normal_equations").
    ne_foldable = has_ineq and _hessian_source(resolved) == "lbfgs"

    solver = select_solver(
        n_vars=int(resolved.n_vars),
        has_equalities=has_equalities,
        capabilities=capabilities(xp),
        options=opts,
        m_ineq=m_ineq,
        ineq_gram_capable=_ineq_gram_capable if has_ineq else None,
        ineq_density=_ineq_density if has_ineq else None,
        ineq_gram_fill=_ineq_gram_fill if ne_foldable else None,
    )
    restoration_solver_factory = select_restoration_solver(opts)

    # Pre-solve diagnostics (verbosity tiers 3–5; gated by the logger threshold
    # so an application's own handlers still receive every record).
    _log_setup(resolved, x0, xp, opts, solver, lower, upper, has_ineq, has_eq)

    reported_callback: IterationCallback | None = callback
    record_transform: Callable[[IterationRecord], IterationRecord] | None = None
    if problem_scaling is not None:
        reported_callback = _unscale_callback(callback, problem_scaling)

        def transform_record(record: IterationRecord) -> IterationRecord:
            return _unscale_record(record, problem_scaling)

        record_transform = transform_record

    driver = IPMDriver(
        model,
        xp=xp,
        solver=solver,
        restoration_solver_factory=restoration_solver_factory,
        options=opts,
        lower=lower,
        upper=upper,
        has_ineq=has_ineq,
        has_eq=has_eq,
        callback=reported_callback,
        record_transform=record_transform,
        warm_start=effective_warm,
    )
    result = driver.run(x0)
    if problem_scaling is not None:
        result = _unscale_result(result, problem_scaling)
    result = replace(
        result,
        solve_time=perf_counter() - start_time,
        linear_solver=_describe_solver(solver),
        device=_describe_device(result.x),
        routes=_build_routes(solver, resolved, opts),
    )

    # Post-solve summary (tier 1) and the timing split (tier 2).
    if logger.isEnabledFor(RESULT):
        logger.log(RESULT, format_result(result))
    if logger.isEnabledFor(ITERATION):
        logger.log(ITERATION, format_timing(result.history))
    return result


def _log_setup(
    resolved: Problem,
    x0: Array,
    xp: Namespace,
    opts: Options,
    solver: object,
    lower: Array | None,
    upper: Array | None,
    has_ineq: bool,
    has_eq: bool,
) -> None:
    """Emit the problem/solver/options diagnostics (verbosity tiers 1, 3–5).

    ``RESULT`` (tier 1, ``verbose >= 1``) is the least restrictive tier that
    needs the problem counts, so they are only evaluated (an extra constraint
    callback each) when at least that tier is enabled — a silent/``verbose=0``
    solve pays nothing extra.
    """
    if logger.isEnabledFor(RESULT):
        n_ineq = int(resolved.ineq_constraints(x0).shape[0]) if has_ineq else 0
        n_eq_nl = int(resolved.eq_constraints(x0).shape[0]) if has_eq else 0
        linear = resolved.linear_eq()
        n_eq_lin = 0 if linear is None else int(linear[0].shape[0])
        n_lower = _count_finite(xp, lower)
        n_upper = _count_finite(xp, upper)
        logger.log(
            RESULT,
            format_setup(
                n_vars=int(resolved.n_vars),
                n_lower=n_lower,
                n_upper=n_upper,
                n_eq=n_eq_nl + n_eq_lin,
                n_ineq=n_ineq,
                linear_solver=_describe_solver(solver),
                hessian=_hessian_source(resolved),
            ),
        )
        if logger.isEnabledFor(PROBLEM):
            logger.log(
                PROBLEM,
                format_problem(
                    n_vars=int(resolved.n_vars),
                    n_ineq=n_ineq,
                    n_eq_nonlinear=n_eq_nl,
                    n_eq_linear=n_eq_lin,
                    n_lower=n_lower,
                    n_upper=n_upper,
                ),
            )
    if logger.isEnabledFor(SOLVER):
        logger.log(SOLVER, format_solver(opts, type(solver).__name__))
    if logger.isEnabledFor(OPTIONS):
        logger.log(OPTIONS, format_options(opts))


def _hessian_source(resolved: Problem) -> str:
    """The resolved Hessian source (``"analytic"``/``"autodiff-hvp"``/``"lbfgs"``).

    ``resolve()`` always attaches ``sources`` to the returned
    :class:`~ipax.problem.derivatives.ResolvedProblem`, but the static type here
    is the ``Problem`` ABC (which doesn't declare it) — mirrors the ``getattr``
    fallback the driver uses for the same attribute.
    """
    sources = getattr(resolved, "sources", None)
    return "unknown" if sources is None else str(sources.hessian)


def _describe_solver(solver: object) -> str:
    """A human-readable label for the linear solver actually used.

    The concrete solvers expose ``describe()`` (the sparse facade delegates to
    the backend adapter that was dispatched at factor time, surfacing the engine
    and device, e.g. ``"sparse [Feral LDL^T (CPU)]"`` or ``"cuDSS (GPU)"``); any
    solver lacking it falls back to its class name.
    """
    describe = getattr(solver, "describe", None)
    return describe() if callable(describe) else type(solver).__name__


def _build_routes(solver: object, resolved: Problem, opts: Options) -> Routes:
    """Assemble the requested → resolved route record for ``Result.routes``.

    Read *after* the run so runtime resolutions are reflected: the sparse
    facade's dispatched backend, the dense augmented route's fallback, an
    ``"auto"`` Krylov preconditioner's promotion — all through the same
    duck-typed ``describe()``/``kkt_form()`` hooks the solvers expose
    (invariant #3: no solver-specific knowledge here).
    """
    kkt_form_fn = getattr(solver, "kkt_form", None)
    scaling = opts.scaling
    corrections = opts.corrections
    return Routes(
        linear_solver=_describe_solver(solver),
        linsolve_requested=str(opts.linsolve),
        kkt_form=kkt_form_fn() if callable(kkt_form_fn) else "unknown",
        hessian=_hessian_source(resolved),
        hessian_requested=str(opts.hessian),
        globalization=str(opts.globalization),
        mu_schedule=str(opts.mu_schedule),
        scaling=str(getattr(scaling, "method", scaling)),
        corrections=str(getattr(corrections, "method", corrections)),
    )


def _describe_device(x: Array) -> str:
    """Stringified device of the solution array (e.g. ``"cpu"``, CUDA device).

    Read from the array's standard ``.device`` attribute so it reflects where the
    solve actually ran without importing any concrete backend.
    """
    device = getattr(x, "device", None)
    return "" if device is None else str(device)


def _count_finite(xp: Namespace, bound: Array | None) -> int:
    """Number of finite entries in a bound vector (``None`` → 0)."""
    if bound is None:
        return 0
    return int(xp.sum(xp.astype(xp.isfinite(bound), xp.int64)))


def _unscale_record(
    record: IterationRecord, scaling: ProblemScaling
) -> IterationRecord:
    """Map the reported objective to original units; residuals stay scaled."""
    return replace(record, objective=record.objective / scaling.obj)


def _unscale_callback(
    callback: IterationCallback | None, scaling: ProblemScaling
) -> IterationCallback | None:
    """Present callback primal/dual state in the original problem's units."""
    if callback is None:
        return None

    def wrapped(info: IterationInfo) -> bool | None:
        s = info.s
        if s is not None and scaling.ineq is not None:
            s = s / scaling.ineq
        y_eq = info.y_eq
        if y_eq is not None and scaling.combined_eq is not None:
            y_eq = scaling.combined_eq * y_eq / scaling.obj
        y_ineq = info.y_ineq
        if y_ineq is not None and scaling.ineq is not None:
            y_ineq = scaling.ineq * y_ineq / scaling.obj
        z_lower = None if info.z_lower is None else info.z_lower / scaling.obj
        z_upper = None if info.z_upper is None else info.z_upper / scaling.obj
        return callback(
            replace(
                info,
                s=s,
                y_eq=y_eq,
                y_ineq=y_ineq,
                z_lower=z_lower,
                z_upper=z_upper,
            )
        )

    return wrapped


def _scale_warm_start(warm: WarmStart, scaling: ProblemScaling) -> WarmStart:
    """Map an original-units warm start into the scaled subproblem's space.

    The exact inverse of :func:`_unscale_result` (and the callback unscaling):
    scaled slacks ``s̃ = d_ineq · s`` (since ``g̃ = d_ineq·g`` and ``g̃ + s̃ = 0``);
    scaled multipliers ``ỹ = y · s_f / d`` and ``z̃ = z · s_f``. Factors are in
    ``(0, 1]`` so the divisions are safe.
    """
    s_f = scaling.obj
    s = warm.s
    if s is not None and scaling.ineq is not None:
        s = scaling.ineq * s
    y_eq = warm.y_eq
    if y_eq is not None and scaling.combined_eq is not None:
        y_eq = y_eq * s_f / scaling.combined_eq
    y_ineq = warm.y_ineq
    if y_ineq is not None and scaling.ineq is not None:
        y_ineq = y_ineq * s_f / scaling.ineq
    z_lower = None if warm.z_lower is None else warm.z_lower * s_f
    z_upper = None if warm.z_upper is None else warm.z_upper * s_f
    return WarmStart(s=s, y_eq=y_eq, y_ineq=y_ineq, z_lower=z_lower, z_upper=z_upper)


def _unscale_result(result: Result, scaling: ProblemScaling) -> Result:
    """Map a scaled-space solution back to the original problem.

    ``x`` and the constraint *feasibility* are scale-invariant, but the
    objective and every multiplier carry the objective factor ``s_f`` (and each
    constraint multiplier its row factor), so they are divided/multiplied back
    here. ``kkt_error``/``constraint_violation`` remain the scaled-space metrics
    that drove convergence.
    """
    s_f = scaling.obj
    y_eq = result.y_eq
    if y_eq is not None and scaling.combined_eq is not None:
        y_eq = scaling.combined_eq * y_eq / s_f
    y_ineq = result.y_ineq
    if y_ineq is not None and scaling.ineq is not None:
        y_ineq = scaling.ineq * y_ineq / s_f
    z_lower = None if result.z_lower is None else result.z_lower / s_f
    z_upper = None if result.z_upper is None else result.z_upper / s_f
    return replace(
        result,
        objective=result.objective / s_f,
        y_eq=y_eq,
        y_ineq=y_ineq,
        z_lower=z_lower,
        z_upper=z_upper,
    )


__all__ = ["solve"]
