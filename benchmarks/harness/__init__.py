"""Benchmark harness: metrics, environment capture, result IO (§9.2, §9.4).

The quality-control core: run one ``(problem, backend, config)`` case, capture
correctness/robustness/accuracy metrics into a :class:`CaseResult`, and render
the collection as machine-readable JSON and a human-readable Markdown report.
Reporting is deliberately dependency-free (tables, not plots) so the QC sweep
runs anywhere; perf/scaling plots are a later, optional layer.
"""

from __future__ import annotations

import gc
import math
import platform
import sys
import traceback
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import TYPE_CHECKING

import ipax

if TYPE_CHECKING:
    from benchmarks.corpus import BenchmarkProblem
    from ipax.typing import Namespace

# Accuracy gates for scoring a case "correct" (float64; relaxed from solver tol
# to absorb backend arithmetic differences). A case is correct when it stops at
# a success status, its scaled KKT residual is small, and — when the optimum is
# known — the iterate matches it. A case is the weaker "converged" (reached a
# valid KKT point) when it stops at a success status with a small scaled KKT
# residual, *regardless* of whether it matched the documented objective — so
# ``correct`` implies ``converged``. The gap between the two is problems solved
# to a different local minimum on a nonconvex objective, which the strict
# documented-optimum gate would otherwise count as a failure.
_KKT_GATE = 1e-6
_X_GATE = 1e-5
# Tolerance for matching the dataset-documented objective (SIF ``LO SOLTN``):
# relative + absolute, since documented values vary in precision.
_OBJ_GATE = 1e-4


@dataclass(frozen=True)
class CaseResult:
    """All QC metrics for a single solved (or failed) benchmark case."""

    problem: str
    kind: str
    backend: str
    config: str
    status: str
    success: bool
    correct: bool  # matched the dataset-documented outcome (objective/infeasibility)
    converged: bool  # reached a valid KKT point (may be a different local optimum)
    n_iter: int
    kkt_error: float
    dual_infeasibility: float
    primal_infeasibility: float
    complementarity: float
    constraint_violation: float
    error_vs_optimum: float | None
    objective: float
    expected_objective: float | None  # dataset-documented optimum (SIF LO SOLTN)
    expected_infeasible: bool  # dataset documents the problem as infeasible
    pbclass: str | None  # CUTEst classification string
    solve_time: float
    linear_solver: str
    gradient_source: str
    hessian_source: str
    error: str | None  # exception text when the solve raised, else None


def capture_environment() -> dict[str, object]:
    """Backend/toolchain versions captured with every report (§9.4)."""
    env: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "ipax": getattr(ipax, "__version__", "unknown"),
    }
    for pkg in ("numpy", "scipy", "torch", "cupy", "jax"):
        try:
            env[pkg] = __import__(pkg).__version__
        except Exception:  # optional backend, record absence
            env[pkg] = None
    env["gpu"] = _detect_gpu_name()
    return env


def _detect_gpu_name() -> str | None:
    """Best-effort CUDA device name for the report header (None if no GPU)."""
    try:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:  # no CuPy / no CUDA device
        return None


def _inf_norm(xp: Namespace, a: object, b: object) -> float:
    return float(xp.max(xp.abs(a - b)))


def run_case(
    case: BenchmarkProblem,
    *,
    config: str,
    options: ipax.Options,
    xp: Namespace,
    backend: str,
) -> CaseResult:
    """Solve one case and score it; never raises (failures become a result row).

    A raised solve (e.g. an unsupported solver/backend combination) is recorded
    with ``error`` set and ``success``/``correct`` false, so one bad cell never
    aborts the sweep.
    """

    def _fail(status: str, error: str) -> CaseResult:
        return CaseResult(
            problem=case.name,
            kind=case.kind,
            backend=backend,
            config=config,
            status=status,
            success=False,
            correct=False,
            converged=False,
            n_iter=0,
            kkt_error=float("inf"),
            dual_infeasibility=float("inf"),
            primal_infeasibility=float("inf"),
            complementarity=float("inf"),
            constraint_violation=float("inf"),
            error_vs_optimum=None,
            objective=float("inf"),
            expected_objective=None,
            expected_infeasible=False,
            pbclass=None,
            solve_time=0.0,
            linear_solver="",
            gradient_source="n/a",
            hessian_source="n/a",
            error=error,
        )

    try:
        problem, x0 = case.build(xp)
    except Exception:  # report build failures as a row
        return _fail("build_error", traceback.format_exc(limit=2).strip())

    try:
        result = ipax.solve(problem, x0, options=options)
    except Exception:  # report solve failures as a row
        return _fail("solve_error", traceback.format_exc(limit=2).strip())

    optimum = case.optimum(problem)
    error_vs_optimum = None if optimum is None else _inf_norm(xp, result.x, optimum)

    # Dataset-sourced expected outcome (S2MPJ problems carry these; others default).
    expected_objective = getattr(problem, "expected_objective", None)
    expected_infeasible = bool(getattr(problem, "expected_infeasible", False))
    pbclass = getattr(problem, "pbclass", None)
    objective = float(result.objective)

    if expected_infeasible:
        # The dataset documents this problem as infeasible, so *detecting*
        # infeasibility is the correct outcome — not a failure to optimize.
        # There is no KKT point to reach, so "converged" mirrors "correct" here.
        correct = result.status is ipax.Status.INFEASIBLE
        converged = correct
    else:
        # A valid KKT point: success status with a small scaled KKT residual
        # (which already bounds primal infeasibility). This is the weaker tier
        # that credits a solve to a *different* local optimum than the documented
        # one — a genuine convergence, not a bug.
        converged = result.success and result.kkt_error <= _KKT_GATE
        objective_ok = True
        if expected_objective is not None and math.isfinite(expected_objective):
            objective_ok = abs(objective - expected_objective) <= _OBJ_GATE * (
                1.0 + abs(expected_objective)
            )
        # ``correct`` additionally requires matching the documented optimum.
        correct = (
            converged
            and objective_ok
            and (error_vs_optimum is None or error_vs_optimum <= _X_GATE)
        )
    sources = result.derivative_sources
    return CaseResult(
        problem=case.name,
        kind=case.kind,
        backend=backend,
        config=config,
        status=result.status.value,
        success=result.success,
        correct=correct,
        converged=converged,
        n_iter=result.n_iter,
        kkt_error=result.kkt_error,
        dual_infeasibility=result.dual_infeasibility,
        primal_infeasibility=result.primal_infeasibility,
        complementarity=result.complementarity,
        constraint_violation=result.constraint_violation,
        error_vs_optimum=error_vs_optimum,
        objective=objective,
        expected_objective=expected_objective,
        expected_infeasible=expected_infeasible,
        pbclass=pbclass,
        solve_time=result.solve_time,
        linear_solver=result.linear_solver,
        gradient_source=sources.gradient,
        hessian_source=sources.hessian,
        error=None,
    )


def to_payload(
    results: list[CaseResult], environment: dict[str, object]
) -> dict[str, object]:
    """JSON-ready payload bundling the environment with every case row."""
    return {"environment": environment, "results": [asdict(r) for r in results]}


def _fmt(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):  # NaN/inf
        return "—"
    return f"{value:.2e}"


def format_markdown(results: list[CaseResult], environment: dict[str, object]) -> str:
    """Render the QC sweep as a Markdown report (summary + per-case detail)."""
    total = len(results)
    n_correct = sum(1 for r in results if r.correct)
    n_converged = sum(1 for r in results if r.converged)
    lines = [
        "# ipax quality-control benchmark",
        "",
        f"- generated: `{environment.get('timestamp')}`",
        f"- python `{environment.get('python')}` on `{environment.get('platform')}`",
        f"- numpy `{environment.get('numpy')}` · scipy `{environment.get('scipy')}`"
        f" · torch `{environment.get('torch')}`",
        f"- **correct: {n_correct}/{total}** · "
        f"**converged (KKT): {n_converged}/{total}** cases",
        "",
        "## Success rate by configuration",
        "",
        "`correct` matched the documented outcome; `converged` reached a valid KKT "
        "point (possibly a different local optimum — a superset of `correct`).",
        "",
        "| config | correct | converged | solved | mean iters |",
        "| --- | --- | --- | --- | --- |",
    ]
    configs: dict[str, list[CaseResult]] = {}
    for r in results:
        configs.setdefault(r.config, []).append(r)
    for config, rows in configs.items():
        solved = [r for r in rows if r.success]
        correct = sum(1 for r in rows if r.correct)
        converged = sum(1 for r in rows if r.converged)
        mean_iter = (
            sum(r.n_iter for r in solved) / len(solved) if solved else float("nan")
        )
        iters = "—" if mean_iter != mean_iter else f"{mean_iter:.1f}"
        lines.append(
            f"| `{config}` | {correct}/{len(rows)} | {converged}/{len(rows)} "
            f"| {len(solved)}/{len(rows)} | {iters} |"
        )

    lines += [
        "",
        "## Per-case detail",
        "",
        "Status `infeasible (exp)` marks problems the dataset documents as "
        "infeasible (detecting infeasibility is the correct outcome). `Δf*` is the "
        "gap to the documented `LO SOLTN` objective when one is recorded. A `≈` flag "
        "marks a case that converged to a valid KKT point at a *different* objective "
        "than the documented one; `⚠️` marks a case that did not converge.",
        "",
        "| problem | backend | config | status | iters | kkt | infeas "
        "| Δf* | err vs x* | time (s) | solver |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(results, key=lambda r: (r.problem, r.backend, r.config)):
        flag = "" if r.correct else (" ≈" if r.converged else " ⚠️")
        status = r.status + (" (exp)" if r.expected_infeasible else "")
        err = "—" if r.error_vs_optimum is None else _fmt(r.error_vs_optimum)
        obj_gap = (
            "—"
            if r.expected_objective is None or not math.isfinite(r.expected_objective)
            else _fmt(abs(r.objective - r.expected_objective))
        )
        lines.append(
            f"| {r.problem} | {r.backend} | `{r.config}` | {status}{flag} "
            f"| {r.n_iter} | {_fmt(r.kkt_error)} | {_fmt(r.constraint_violation)} "
            f"| {obj_gap} | {err} | {r.solve_time:.3f} | {r.linear_solver or '—'} |"
        )
    lines += _routing_hint_lines(results)
    return "\n".join(lines) + "\n"


def _routing_hint_lines(results: list[CaseResult]) -> list[str]:
    """The *Routing hints* section: measured wins next to rows that missed.

    A row qualifies when the curated registry (:mod:`benchmarks.routing_hints`)
    records a lever win for its problem, this run's default configuration did
    not score ``correct`` on it (a correct row needs no hint), **and** the
    lever is observable in that row's configuration. The last condition
    matters: a hint recorded on ``lbfgs/dense`` for an ``LBFGSOptions`` knob is
    a no-op under ``hessian="exact"``, so printing it next to an ``exact/*``
    row would recommend a setting that cannot act and quote a win measured
    elsewhere. Returns no lines at all when nothing qualifies, so the section
    simply does not exist in a clean report.
    """
    from benchmarks.routing_hints import hints_for

    def _applies(hint: object, config: str) -> bool:
        # Config labels are "<hessian>/<linsolve>", e.g. "exact/sparse".
        wanted = getattr(hint, "hessian", None)
        return wanted is None or config.split("/", 1)[0] == wanted

    hinted = [
        (r, hint)
        for r in sorted(results, key=lambda r: (r.problem, r.backend, r.config))
        if not r.correct
        for hint in hints_for(r.problem)
        if _applies(hint, r.config)
    ]
    if not hinted:
        return []
    lines = [
        "",
        "## Routing hints",
        "",
        "Problems above that the default configuration missed and for which a "
        "**measured** opt-in win is on record (`benchmarks/routing_hints.py`). "
        "The *this run* column is the default result the lever beats.",
        "",
        "| problem | config | this run | lever | measured win |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r, hint in hinted:
        this_run = f"{r.status}, {r.n_iter} it, obj {r.objective:.4g}"
        lines.append(
            f"| {r.problem} | `{r.config}` | {this_run} "
            f"| `{hint.options}` | {hint.win} |"
        )
    return lines


# -- reference cross-check (§9.3) --------------------------------------------

# Agreement gates between ipax and a reference solver. Looser than the QC
# accuracy gate: the reference is a *different* algorithm/tolerance, so we check
# the solutions agree, not that they are bitwise identical.
_X_AGREE = 1e-4
_OBJ_AGREE = 1e-5


@dataclass(frozen=True)
class CrossCheckResult:
    """Agreement between an ipax solve and a reference baseline on one problem."""

    problem: str
    baseline: str
    status: str  # ok | skipped | ipax_failed | baseline_failed
    agree: bool
    x_gap: float | None  # ||x_ipax - x_ref||_inf
    objective_gap: float | None
    ipax_error_vs_optimum: float | None
    baseline_error_vs_optimum: float | None
    ipax_time: float
    baseline_time: float
    note: str


def cross_check(
    case: BenchmarkProblem,
    *,
    ipax_options: ipax.Options,
    baseline: object,
    xp: Namespace,
) -> CrossCheckResult:
    """Solve one case with ipax and a reference baseline and compare (NumPy).

    ``skipped`` when the baseline cannot express the problem;
    ``baseline_failed``/``ipax_failed`` when a solver raises. ``agree`` requires
    both to succeed and their solutions/objectives to match within the gates.
    """
    from benchmarks.baselines import BaselineUnsupported

    def _result(
        status: str,
        *,
        agree: bool = False,
        x_gap: float | None = None,
        objective_gap: float | None = None,
        ipax_error_vs_optimum: float | None = None,
        baseline_error_vs_optimum: float | None = None,
        ipax_time: float = 0.0,
        baseline_time: float = 0.0,
        note: str = "",
    ) -> CrossCheckResult:
        return CrossCheckResult(
            problem=case.name,
            baseline=getattr(baseline, "name", "baseline"),
            status=status,
            agree=agree,
            x_gap=x_gap,
            objective_gap=objective_gap,
            ipax_error_vs_optimum=ipax_error_vs_optimum,
            baseline_error_vs_optimum=baseline_error_vs_optimum,
            ipax_time=ipax_time,
            baseline_time=baseline_time,
            note=note,
        )

    problem, x0 = case.build(xp)
    try:
        ipax_result = ipax.solve(problem, x0, options=ipax_options)
    except Exception as exc:  # report, do not raise
        return _result("ipax_failed", note=f"{type(exc).__name__}: {exc}")

    try:
        reference = baseline.solve(problem, x0)
    except BaselineUnsupported as exc:
        return _result("skipped", note=str(exc))
    except Exception as exc:  # record baseline trouble, keep going
        return _result("baseline_failed", note=f"{type(exc).__name__}: {exc}")

    x_gap = _inf_norm(xp, ipax_result.x, xp.asarray(reference.x))
    objective_gap = abs(ipax_result.objective - reference.objective)
    optimum = case.optimum(problem)
    ipax_err = None if optimum is None else _inf_norm(xp, ipax_result.x, optimum)
    base_err = (
        None if optimum is None else _inf_norm(xp, xp.asarray(reference.x), optimum)
    )
    agree = (
        ipax_result.success
        and reference.success
        and x_gap <= _X_AGREE
        and objective_gap <= _OBJ_AGREE * (1.0 + abs(reference.objective))
    )
    return _result(
        "ok",
        agree=agree,
        x_gap=x_gap,
        objective_gap=objective_gap,
        ipax_error_vs_optimum=ipax_err,
        baseline_error_vs_optimum=base_err,
        ipax_time=ipax_result.solve_time,
        baseline_time=reference.solve_time,
    )


def format_crosscheck(
    results: list[CrossCheckResult], environment: dict[str, object]
) -> str:
    """Render the cross-check as a Markdown report (agreement + accuracy)."""
    checked = [r for r in results if r.status == "ok"]
    agreed = sum(1 for r in checked if r.agree)
    lines = [
        "# ipax reference cross-check",
        "",
        f"- generated: `{environment.get('timestamp')}`",
        f"- scipy `{environment.get('scipy')}`",
        f"- **agree: {agreed}/{len(checked)}** comparable cases "
        f"({len(results) - len(checked)} skipped/failed)",
        "",
        "| problem | baseline | status | agree | x gap | obj gap "
        "| ipax err | ref err | ipax t | ref t |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(results, key=lambda r: (r.problem, r.baseline)):
        agree = "—" if r.status != "ok" else ("yes" if r.agree else "**NO**")
        lines.append(
            f"| {r.problem} | {r.baseline} | {r.status} | {agree} "
            f"| {_fmt_opt(r.x_gap)} | {_fmt_opt(r.objective_gap)} "
            f"| {_fmt_opt(r.ipax_error_vs_optimum)} "
            f"| {_fmt_opt(r.baseline_error_vs_optimum)} "
            f"| {r.ipax_time:.3f} | {r.baseline_time:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _fmt_opt(value: float | None) -> str:
    return "—" if value is None else _fmt(value)


# -- scaling & memory study (§9.2) -------------------------------------------


@dataclass(frozen=True)
class ScalingPoint:
    """One ``(route, n)`` measurement: wall-clock and peak memory of a solve."""

    route: str
    n_vars: int
    n_iter: int
    success: bool
    solve_time: float
    peak_memory_mb: float
    kkt_error: float


def measure_solve(
    problem: object, x0: object, options: ipax.Options
) -> tuple[float, float, ipax.Result]:
    """Solve and return ``(wall_seconds, peak_bytes, result)``.

    Peak memory is the ``tracemalloc`` high-water mark *during the solve only*
    (the problem is built before tracing starts), which — since NumPy registers
    its allocations with tracemalloc — captures the dense ``n×n`` KKT matrix and
    so separates the dense route from the matrix-free/sparse ones.
    """
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    start = perf_counter()
    result = ipax.solve(problem, x0, options=options)  # type: ignore[arg-type]
    elapsed = perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, float(peak), result


def fit_exponent(sizes: list[int], values: list[float]) -> float:
    """Empirical scaling exponent ``p`` from a log–log least-squares fit.

    Returns the slope of ``log(value)`` vs ``log(n)`` (so ``value ∝ n^p``), or
    ``nan`` when fewer than two positive, finite points are available.
    """
    points = [
        (n, v)
        for n, v in zip(sizes, values, strict=True)
        if n > 0 and v > 0 and math.isfinite(v)
    ]
    if len(points) < 2:
        return float("nan")
    import numpy as np

    log_n = np.log(np.asarray([p[0] for p in points], dtype=float))
    log_v = np.log(np.asarray([p[1] for p in points], dtype=float))
    return float(np.polyfit(log_n, log_v, 1)[0])


def format_scaling(points: list[ScalingPoint], environment: dict[str, object]) -> str:
    """Render the scaling study as Markdown (per-route exponents + detail)."""
    by_route: dict[str, list[ScalingPoint]] = {}
    for p in points:
        by_route.setdefault(p.route, []).append(p)

    lines = [
        "# ipax scaling & memory study",
        "",
        f"- generated: `{environment.get('timestamp')}`",
        f"- numpy `{environment.get('numpy')}` · scipy `{environment.get('scipy')}`",
        "- synthetic RT-like generator; empirical exponent `p` in `cost ∝ nᵖ`.",
        "",
        "## Scaling exponents",
        "",
        "| route | sizes | time exponent | memory exponent |",
        "| --- | --- | --- | --- |",
    ]
    for route, rows in by_route.items():
        rows = sorted(rows, key=lambda r: r.n_vars)
        ok = [r for r in rows if r.success]
        sizes = [r.n_vars for r in ok]
        time_p = fit_exponent(sizes, [r.solve_time for r in ok])
        mem_p = fit_exponent(sizes, [r.peak_memory_mb for r in ok])
        span = f"{sizes[0]}–{sizes[-1]}" if sizes else "—"
        lines.append(f"| {route} | {span} | {_exp(time_p)} | {_exp(mem_p)} |")

    lines += [
        "",
        "## Per-point detail",
        "",
        "| route | n | iters | time (s) | peak mem (MB) | kkt | ok |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in sorted(points, key=lambda p: (p.route, p.n_vars)):
        ok = "yes" if p.success else "**NO**"
        lines.append(
            f"| {p.route} | {p.n_vars} | {p.n_iter} | {p.solve_time:.3f} "
            f"| {p.peak_memory_mb:.1f} | {_fmt(p.kkt_error)} | {ok} |"
        )
    return "\n".join(lines) + "\n"


def _exp(value: float) -> str:
    return "—" if not math.isfinite(value) else f"{value:.2f}"


# -- device-efficiency study (GPU profiling) ---------------------------------
#
# The iterative IPM loop is where Array-API GPU performance is won or lost: every
# ``float()``/``bool()`` on a 0-d device array forces a host<->device sync that
# serializes the GPU. This study measures, per solve, the **host-sync count** and
# **wall time per iteration** (plus the GPU-vs-CPU time split on CuPy) so the
# no-cost optimization — consolidating per-iteration scalar reads to one sync —
# can be quantified before and after. Backend specifics (CuPy/Torch timers) live
# here in ``benchmarks/``, never in the ``ipax/`` core (invariant #1).


@dataclass(frozen=True)
class DeviceMetrics:
    """Host-sync / timing profile for one ``(backend, route, n)`` solve."""

    backend: str
    device: str
    route: str
    n_vars: int
    n_iter: int
    success: bool
    solve_time: float  # total wall (s), device-synchronized at both ends
    time_per_iter: float  # solve_time / n_iter
    gpu_time: float | None  # measured GPU compute time (s); CuPy only
    host_syncs: int | None  # device->host scalar materializations during solve
    syncs_per_iter: float | None  # host_syncs / n_iter (the headline metric)
    peak_device_mb: float | None  # device-memory high-water (CuPy/Torch-CUDA)
    kkt_error: float


class _ScalarSyncCounter:
    """Count device->host scalar materializations during a solve.

    Patches the array type's scalar dunders (``__float__``/``__int__``/
    ``__bool__``/``__index__``/``item``) for the measurement window. Each forces a
    host sync on a GPU backend, so the tally is the per-solve host-sync total the
    driver-loop optimization targets. Built-in array types that forbid attribute
    assignment (NumPy's ``ndarray``) make counting unavailable — :attr:`result`
    is then ``None`` (on CPU a host scalar read is free anyway).
    """

    _NAMES = ("__float__", "__int__", "__bool__", "__index__", "item")

    def __init__(self, array_type: type) -> None:
        self._type = array_type
        self._orig: dict[str, object] = {}
        self._orig_read_scalars: object | None = None
        self.count = 0
        self.available = True

    def __enter__(self) -> _ScalarSyncCounter:
        counter = self
        for name in self._NAMES:
            orig = getattr(self._type, name, None)
            if orig is None:
                continue

            def make(orig: object):
                def wrapped(self, *args, **kwargs):
                    counter.count += 1
                    return orig(self, *args, **kwargs)  # type: ignore[operator]

                return wrapped

            try:
                setattr(self._type, name, make(orig))
            except TypeError:  # built-in/extension type (NumPy) — cannot patch
                self.available = False
                self._restore()
                break
            self._orig[name] = orig
        if self.available:
            # A fused multi-part ``read_scalars`` batch is one *extra*
            # device→host transfer on top of whatever scalar dunders it
            # triggers: the bulk path triggers none (→ counts 1), a single
            # part goes through ``float`` (→ its dunder counts the 1), and
            # the per-element *fallback* keeps its k dunder counts (+1 for
            # the failed bulk attempt) — so a backend without a bulk
            # transfer is honestly reported as unfused rather than
            # pretending the batching happened.
            from ipax.backend import scalars as _scalars_module

            orig_read = _scalars_module.read_scalars

            def counting_read(xp, parts):  # type: ignore[no-untyped-def]
                if len(parts) > 1:
                    counter.count += 1
                return orig_read(xp, parts)

            self._orig_read_scalars = orig_read
            _scalars_module.read_scalars = counting_read
        return self

    def _restore(self) -> None:
        for name, orig in self._orig.items():
            setattr(self._type, name, orig)
        self._orig.clear()
        if self._orig_read_scalars is not None:
            from ipax.backend import scalars as _scalars_module

            _scalars_module.read_scalars = self._orig_read_scalars  # type: ignore[assignment]
            self._orig_read_scalars = None

    def __exit__(self, *exc: object) -> None:
        self._restore()

    @property
    def result(self) -> int | None:
        return self.count if self.available else None


def _sync_device(backend: str) -> None:
    """Block until the device finishes queued work (no-op off GPU)."""
    if backend == "cupy":
        import cupy as cp

        cp.cuda.Device().synchronize()
    elif backend == "torch":
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()


def _reset_device_memory(backend: str) -> None:
    """Reset the device-memory high-water mark before a measured solve."""
    if backend == "cupy":
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
    elif backend == "torch":
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def _peak_device_mb(backend: str) -> float | None:
    """Device-memory high-water mark in MB (None off GPU)."""
    if backend == "cupy":
        import cupy as cp

        return cp.get_default_memory_pool().total_bytes() / 1e6
    if backend == "torch":
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
    return None


def _gpu_time(backend: str, fn: object) -> float | None:
    """Measured GPU compute time of ``fn`` in seconds (CuPy only, else None).

    Uses ``cupyx.profiler.benchmark`` (one extra run), whose ``gpu_times`` is the
    on-stream device time. A large gap between wall time and this value is the
    signature of a host-sync-bound loop.
    """
    if backend != "cupy":
        return None
    try:
        from cupyx.profiler import benchmark

        measured = benchmark(fn, n_repeat=1, n_warmup=0)  # type: ignore[arg-type]
        return float(measured.gpu_times.mean())
    except Exception:  # profiler unavailable / non-idempotent call
        return None


def measure_device_solve(
    problem: object,
    x0: object,
    options: ipax.Options,
    *,
    backend: str,
    route: str,
    warmup: bool = True,
) -> DeviceMetrics:
    """Profile one solve for host-sync count and per-iteration timing.

    ``warmup`` runs (and discards) one solve first so device handle/allocator
    init and any JIT cost stay out of the measured run. The measured solve is
    device-synchronized at both ends so ``solve_time`` is true end-to-end wall.
    """
    if warmup:
        ipax.solve(problem, x0, options=options)  # type: ignore[arg-type]

    _reset_device_memory(backend)
    _sync_device(backend)
    with _ScalarSyncCounter(type(x0)) as counter:
        start = perf_counter()
        result = ipax.solve(problem, x0, options=options)  # type: ignore[arg-type]
        _sync_device(backend)
        wall = perf_counter() - start
    host_syncs = counter.result

    n_iter = result.n_iter
    gpu_time = _gpu_time(backend, lambda: ipax.solve(problem, x0, options=options))
    peak = _peak_device_mb(backend)
    return DeviceMetrics(
        backend=backend,
        device=result.device or "cpu",
        route=route,
        n_vars=int(problem.n_vars),  # type: ignore[attr-defined]
        n_iter=n_iter,
        success=result.success,
        solve_time=wall,
        time_per_iter=wall / n_iter if n_iter else float("nan"),
        gpu_time=gpu_time,
        host_syncs=host_syncs,
        syncs_per_iter=(
            host_syncs / n_iter if (host_syncs is not None and n_iter) else None
        ),
        peak_device_mb=peak,
        kkt_error=result.kkt_error,
    )


def format_device(metrics: list[DeviceMetrics], environment: dict[str, object]) -> str:
    """Render the device-efficiency study as Markdown (syncs/iter + timing)."""
    lines = [
        "# ipax device-efficiency study",
        "",
        f"- generated: `{environment.get('timestamp')}`",
        f"- gpu: `{environment.get('gpu')}`",
        f"- cupy `{environment.get('cupy')}` · torch `{environment.get('torch')}`",
        "- headline metric: **host syncs / iter** "
        "(device->host scalar reads in the loop).",
        "",
        "| backend | device | route | n | iters | wall (s) | s/iter "
        "| syncs | syncs/iter | gpu (s) | peak MB | kkt |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in sorted(metrics, key=lambda m: (m.backend, m.route, m.n_vars)):
        lines.append(
            f"| {m.backend} | {m.device} | {m.route} | {m.n_vars} | {m.n_iter} "
            f"| {m.solve_time:.3f} | {_fmt(m.time_per_iter)} "
            f"| {_fmt_count(m.host_syncs)} | {_fmt_opt(m.syncs_per_iter)} "
            f"| {_fmt_opt(m.gpu_time)} | {_fmt_opt(m.peak_device_mb)} "
            f"| {_fmt(m.kkt_error)} |"
        )
    return "\n".join(lines) + "\n"


def _fmt_count(value: int | None) -> str:
    return "—" if value is None else str(value)


__all__ = [
    "CaseResult",
    "CrossCheckResult",
    "DeviceMetrics",
    "ScalingPoint",
    "capture_environment",
    "cross_check",
    "fit_exponent",
    "format_crosscheck",
    "format_device",
    "format_markdown",
    "format_scaling",
    "measure_device_solve",
    "measure_solve",
    "run_case",
    "to_payload",
]
