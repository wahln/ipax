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
# known — the iterate matches it.
_KKT_GATE = 1e-6
_X_GATE = 1e-5


@dataclass(frozen=True)
class CaseResult:
    """All QC metrics for a single solved (or failed) benchmark case."""

    problem: str
    kind: str
    backend: str
    config: str
    status: str
    success: bool
    correct: bool
    n_iter: int
    kkt_error: float
    dual_infeasibility: float
    primal_infeasibility: float
    complementarity: float
    constraint_violation: float
    error_vs_optimum: float | None
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
    for pkg in ("numpy", "scipy", "torch"):
        try:
            env[pkg] = __import__(pkg).__version__
        except Exception:  # optional backend, record absence
            env[pkg] = None
    return env


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
            n_iter=0,
            kkt_error=float("inf"),
            dual_infeasibility=float("inf"),
            primal_infeasibility=float("inf"),
            complementarity=float("inf"),
            constraint_violation=float("inf"),
            error_vs_optimum=None,
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

    correct = (
        result.success
        and result.kkt_error <= _KKT_GATE
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
        n_iter=result.n_iter,
        kkt_error=result.kkt_error,
        dual_infeasibility=result.dual_infeasibility,
        primal_infeasibility=result.primal_infeasibility,
        complementarity=result.complementarity,
        constraint_violation=result.constraint_violation,
        error_vs_optimum=error_vs_optimum,
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
    lines = [
        "# ipax quality-control benchmark",
        "",
        f"- generated: `{environment.get('timestamp')}`",
        f"- python `{environment.get('python')}` on `{environment.get('platform')}`",
        f"- numpy `{environment.get('numpy')}` · scipy `{environment.get('scipy')}`"
        f" · torch `{environment.get('torch')}`",
        f"- **correct: {n_correct}/{total}** cases",
        "",
        "## Success rate by configuration",
        "",
        "| config | correct | solved | mean iters |",
        "| --- | --- | --- | --- |",
    ]
    configs: dict[str, list[CaseResult]] = {}
    for r in results:
        configs.setdefault(r.config, []).append(r)
    for config, rows in configs.items():
        solved = [r for r in rows if r.success]
        correct = sum(1 for r in rows if r.correct)
        mean_iter = (
            sum(r.n_iter for r in solved) / len(solved) if solved else float("nan")
        )
        iters = "—" if mean_iter != mean_iter else f"{mean_iter:.1f}"
        lines.append(
            f"| `{config}` | {correct}/{len(rows)} | {len(solved)}/{len(rows)} "
            f"| {iters} |"
        )

    lines += [
        "",
        "## Per-case detail",
        "",
        "| problem | backend | config | status | iters | kkt | infeas "
        "| err vs x* | time (s) | solver |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(results, key=lambda r: (r.problem, r.backend, r.config)):
        flag = "" if r.correct else " ⚠️"
        err = "—" if r.error_vs_optimum is None else _fmt(r.error_vs_optimum)
        lines.append(
            f"| {r.problem} | {r.backend} | `{r.config}` | {r.status}{flag} "
            f"| {r.n_iter} | {_fmt(r.kkt_error)} | {_fmt(r.constraint_violation)} "
            f"| {err} | {r.solve_time:.3f} | {r.linear_solver or '—'} |"
        )
    return "\n".join(lines) + "\n"


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


__all__ = [
    "CaseResult",
    "CrossCheckResult",
    "ScalingPoint",
    "capture_environment",
    "cross_check",
    "fit_exponent",
    "format_crosscheck",
    "format_markdown",
    "format_scaling",
    "measure_solve",
    "run_case",
    "to_payload",
]
