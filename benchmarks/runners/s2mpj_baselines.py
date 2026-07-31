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

"""ipax vs a reference solver (IPOPT via ``ipyopt``) on S2MPJ problems.

The accuracy sweep says *whether* ipax reaches each problem's optimum; this says
whether an established solver reaches the **same** answer from the same start —
which turns every ipax failure into a diagnosable question: is it a hard problem
(the reference fails too), or an ipax gap (the reference succeeds)?

Reporting is deliberately on the **language-neutral axis**: correctness and
IPOPT's own **iteration count**, not wall-clock. ipax is pure-Python over an
Array-API abstraction; IPOPT is compiled — a raw time comparison measures the
implementation language, not the algorithm, so it is recorded but never the
headline.

The four verdicts:

* ``agree``     — both correct (validates ipax on that problem)
* ``ipax-gap``  — ipax wrong, reference correct (the actionable ones)
* ``ipax-wins`` — ipax correct, reference wrong/struggled
* ``both-hard`` — neither reaches the documented optimum

A starred variant (``agree*``, ``ipax-gap*``, …) marks a problem the dataset
gives no reference objective for, scored on solver *success* and objective
agreement instead.

Usage::

    python -m benchmarks.runners.s2mpj_baselines AGG DEGENLPA OET7
    IPAX_S2MPJ_DIR=/path/to/S2MPJ python -m benchmarks.runners.s2mpj_baselines \\
        --all --jobs 12 --out benchmarks/reports/s2mpj_baselines

Like the accuracy sweep this is download-gated (no license to vendor S2MPJ) and
needs the optional ``ipyopt`` binding, so it is dev tooling, never part of the
per-PR pipeline. It is **advisory**: an ``ipax-gap`` is a finding to triage, not
a failure, so the exit status is 0 unless the run itself broke.

``--all --jobs N`` is the full-corpus mode. It borrows the accuracy sweep's
unattended-run machinery, because a ~1100-problem run over two solvers hits all
of the same hazards: rows are flushed after every problem (so a native crash in
a factorization keeps the partial report), ``--resume`` continues from that
report, each worker marks the problem it is running so a native crash leaves at
most ``--jobs`` named candidates (``--resume --exclude`` steps past the culprit),
and both solvers carry an explicit iteration/time budget. Pin BLAS threads (``OMP_NUM_THREADS=1``) for
comparable timings.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import numpy as np

import ipax
from benchmarks.baselines import BaselineUnsupported, IpyoptBaseline
from benchmarks.corpus.s2mpj import list_s2mpj_problems, s2mpj_dir, s2mpj_problems
from benchmarks.harness import capture_environment

# The accuracy sweep's dense-route cap and build guard, reused so this report
# covers the same problems as the sweep's `lbfgs/dense` config (the comparison
# is only useful next to the rows it explains) and so a pathological O(n²)
# pure-Python build cannot stall an unattended run.
from benchmarks.runners.s2mpj import _DENSE_MAX_VARS, _build_within

# Same objective tolerance the accuracy sweep scores ipax with, so ipax and the
# reference are judged identically (harness ``_OBJ_GATE``).
_OBJ_GATE = 1e-4

# Verdict glossary for the report — kept next to ``_verdict`` so the two cannot
# drift apart.
_VERDICT_MEANING = {
    "agree": "both correct",
    "ipax-gap": "reference correct, ipax not — **actionable**",
    "ipax-wins": "ipax correct, reference not",
    "both-hard": "neither reached the documented optimum",
    "agree*": "unscored: both succeeded at the same objective",
    "differ*": "unscored: both succeeded at different objectives",
    "ipax-gap*": "unscored: reference succeeded, ipax failed — **actionable**",
    "ipax-wins*": "unscored: ipax succeeded, reference failed",
    "both-hard*": "unscored: neither solver succeeded",
    "ref-error": "the reference could not run this problem",
}


def _objective_ok(objective: float, expected: float | None) -> bool | None:
    """Whether ``objective`` matches the documented optimum (``None`` if none)."""
    if expected is None or not math.isfinite(expected):
        return None
    return abs(objective - expected) <= _OBJ_GATE * (1.0 + abs(expected))


def _correct(success: bool, objective: float, expected: float | None) -> bool | None:
    """Whether a solver *solved* the problem: converged **and** at the optimum.

    Success is part of the criterion, matching the accuracy sweep's ``correct``
    (harness ``run_case``). Objective agreement alone would credit a stall that
    happens to sit at the documented optimum — and, worse, would let two
    *failed* solvers score as agreeing with each other.
    """
    ok = _objective_ok(objective, expected)
    if ok is None:
        return None
    return bool(success and ok)


@dataclass
class Row:
    """One problem's ipax-vs-reference comparison.

    Every field carries a default so ``--resume`` can read a report written by
    an older schema (mirrors the accuracy sweep's ``_row_to_case_result``): a
    long full-corpus run must survive the runner gaining a column.
    """

    problem: str = ""
    n_vars: int = 0
    expected_objective: float | None = None
    ipax_status: str = ""
    ipax_iters: int = 0
    ipax_objective: float = float("nan")
    ipax_infeasibility: float = float("nan")
    ipax_correct: bool | None = None
    ref_name: str = "ipyopt"
    ref_success: bool = False
    ref_iters: int = 0
    ref_objective: float | None = None
    ref_infeasibility: float = float("nan")
    ref_correct: bool | None = None
    ref_error: str | None = None
    verdict: str = ""
    ipax_time: float = 0.0
    ref_time: float = 0.0


def _verdict(ipax_ok: bool | None, ref_ok: bool | None) -> str:
    # When a problem has no documented optimum, fall back to objective agreement
    # between the two solvers instead of an absolute correctness verdict.
    if ipax_ok is None or ref_ok is None:
        return "unscored"
    if ipax_ok and ref_ok:
        return "agree"
    if not ipax_ok and ref_ok:
        return "ipax-gap"
    if ipax_ok and not ref_ok:
        return "ipax-wins"
    return "both-hard"


def _infeasibility(problem: Any, x: Any) -> float:
    """∞-norm constraint violation of ``x`` — every block, bounds included.

    Both solvers' points are measured with the *same* ruler, because an
    objective comparison alone is not a verdict: a lower objective at a less
    feasible point is not a better answer, and that is exactly how a barrier
    method fails. Returns ``inf`` if the problem cannot be evaluated there.
    """
    try:
        x = np.asarray(x, dtype=float)
        worst = 0.0

        eq = problem.eq_constraints(x)
        worst = max(worst, float(np.max(np.abs(np.asarray(eq)))) if len(eq) else 0.0)
    except NotImplementedError:
        pass
    except Exception:
        return float("inf")
    try:
        ineq = np.asarray(problem.ineq_constraints(x))
        if ineq.size:
            worst = max(worst, float(np.max(np.maximum(ineq, 0.0))))
    except NotImplementedError:
        pass
    except Exception:
        return float("inf")

    try:
        linear_eq = problem.linear_eq()
        if linear_eq is not None:
            A, b = linear_eq
            residual = _matvec(A, x) - np.asarray(b, dtype=float)
            if residual.size:
                worst = max(worst, float(np.max(np.abs(residual))))
        linear_ineq = problem.linear_ineq()
        if linear_ineq is not None:
            A, lo, hi = linear_ineq
            value = _matvec(A, x)
            lo = np.asarray(lo, dtype=float)
            hi = np.asarray(hi, dtype=float)
            if value.size:
                below = np.max(np.maximum(lo - value, 0.0), initial=0.0)
                above = np.max(np.maximum(value - hi, 0.0), initial=0.0)
                worst = max(worst, float(below), float(above))
        lower, upper = problem.bounds()
        if lower is not None:
            worst = max(
                worst,
                float(np.max(np.maximum(np.asarray(lower, dtype=float) - x, 0.0))),
            )
        if upper is not None:
            worst = max(
                worst,
                float(np.max(np.maximum(x - np.asarray(upper, dtype=float), 0.0))),
            )
    except Exception:
        return float("inf")
    return worst


def _matvec(operator: Any, x: np.ndarray) -> np.ndarray:
    """``A @ x`` for either a dense array or a :class:`LinearOperator`."""
    if hasattr(operator, "matvec"):
        return np.asarray(operator.matvec(x), dtype=float)
    return np.asarray(np.asarray(operator) @ x, dtype=float)


def _build_case(root: str | None, name: str) -> Any:
    """The S2MPJ benchmark case for ``name`` (L-BFGS mode, sparse operators).

    ``feasibility=True`` admits the objective-free problems (the CUTEst
    nonlinear-equation systems) as ``min 0`` subject to the constraints — they
    are a large share of ipax's unexplained failures, so the triage wants them.
    """
    built = s2mpj_problems(
        [name], directory=root, hessian="lbfgs", sparse=True, feasibility=True
    )
    if not built:
        raise RuntimeError(f"{name}: not available in this S2MPJ checkout")
    return built[0]


def _compare_built(
    name: str,
    case: Any,
    problem: Any,
    x0: Any,
    options: ipax.Options,
    ref_max_iter: int,
    ref_max_time: float | None,
    ref_options: tuple[tuple[str, Any], ...] = (),
) -> Row:
    """Solve one built problem with both solvers and score them."""
    expected = getattr(problem, "expected_objective", None)

    t0 = perf_counter()
    try:
        r = ipax.solve(problem, x0, options=options)
        ipax_status = r.status.value
        ipax_iters = r.n_iter
        ipax_obj = float(r.objective)
        ipax_infeas = _infeasibility(problem, r.x)
    except Exception as exc:
        # An ipax exception is a finding, not a lost row: the reference still
        # runs, so "ipax raised where IPOPT solved" is recorded as a gap.
        ipax_status = f"error: {type(exc).__name__}"
        ipax_iters = 0
        ipax_obj = float("nan")
        ipax_infeas = float("nan")
    ipax_time = perf_counter() - t0
    ipax_success = ipax_status in ("optimal", "acceptable")
    ipax_ok = _correct(ipax_success, ipax_obj, expected)

    ref_name = IpyoptBaseline.name
    ref_success = False
    ref_iters = 0
    ref_obj: float | None = None
    ref_infeas = float("nan")
    ref_ok: bool | None = None
    ref_err: str | None = None
    ref_time = 0.0
    try:
        # A fresh build so the reference sees the untouched starting point.
        ref_problem, ref_x0 = case.build(np)
        ref = IpyoptBaseline(
            max_iter=ref_max_iter, max_time=ref_max_time, options=ref_options
        ).solve(ref_problem, ref_x0)
        ref_name = ref.name
        ref_success = ref.success
        ref_iters = ref.n_iter
        ref_obj = float(ref.objective)
        ref_infeas = _infeasibility(ref_problem, ref.x)
        ref_ok = _correct(ref_success, ref_obj, expected)
        ref_time = ref.solve_time
    except BaselineUnsupported as exc:
        ref_err = f"unsupported: {exc}"
    except Exception as exc:
        ref_err = f"{type(exc).__name__}: {exc}"

    # Fallback verdict for unscored problems (no documented optimum): compare
    # the two solvers' *success*, then their objectives. Success first, because
    # "ipax failed, IPOPT solved" (AGG) is an ipax-gap regardless of the
    # objective the failed run happened to report.
    verdict = _verdict(ipax_ok, ref_ok)
    if verdict == "unscored":
        if ref_err is not None:
            verdict = "ref-error"
        elif ipax_success and not ref_success:
            verdict = "ipax-wins*"
        elif ref_success and not ipax_success:
            verdict = "ipax-gap*"
        elif not ipax_success and not ref_success:
            # Neither solved it. Two failures at the same objective agreed
            # about nothing — that is a hard problem, not an agreement.
            verdict = "both-hard*"
        elif ref_obj is not None:
            rel = abs(ipax_obj - ref_obj) / max(1.0, abs(ref_obj))
            verdict = "agree*" if rel <= 1e-4 else "differ*"

    return Row(
        # The bare S2MPJ name (the case carries the "s2mpj/"-prefixed one), so
        # rows key on the same string --resume/--exclude are given.
        problem=name,
        n_vars=int(problem.n_vars),
        expected_objective=expected,
        ipax_status=ipax_status,
        ipax_iters=ipax_iters,
        ipax_objective=ipax_obj,
        ipax_infeasibility=ipax_infeas,
        ipax_correct=ipax_ok,
        ref_name=ref_name,
        ref_success=ref_success,
        ref_iters=ref_iters,
        ref_objective=ref_obj,
        ref_infeasibility=ref_infeas,
        ref_correct=ref_ok,
        ref_error=ref_err,
        verdict=verdict,
        ipax_time=ipax_time,
        ref_time=ref_time,
    )


def _compare_problem(
    root: str | None,
    name: str,
    options: ipax.Options,
    max_vars: int,
    ref_max_iter: int,
    ref_max_time: float | None,
    max_build_seconds: float = 0.0,
    ref_options: tuple[tuple[str, Any], ...] = (),
    inflight_prefix: str | None = None,
) -> tuple[Row | None, str | None]:
    """Compare one problem: ``(row, skip_reason)``, exactly one of them set.

    Top-level (picklable) so ``--jobs`` worker processes can run it; the serial
    path calls it inline. It never raises — an unattended full-corpus run must
    turn a broken problem into a counted skip rather than losing the sweep.
    ``skip_reason`` is ``"no_objective"`` (not a minimization problem),
    ``"too_large"`` (over the route's variable cap), ``"slow_build"``, or
    ``"build_error"``.

    ``inflight_prefix`` makes the worker name the problem it is running, in its
    own pid-suffixed marker file. A native crash takes the worker down mid-solve
    with no chance to report, and the parent's pending-futures list cannot
    identify the culprit — with the whole corpus submitted up front, every
    unstarted problem is in it too. At most ``--jobs`` markers survive a crash,
    and one of them is the culprit.
    """
    marker = None
    if inflight_prefix is not None:
        marker = Path(f"{inflight_prefix}.{os.getpid()}")
        marker.write_text(name, encoding="utf-8")
    try:
        return _compare_problem_inner(
            root,
            name,
            options,
            max_vars,
            ref_max_iter,
            ref_max_time,
            max_build_seconds,
            ref_options,
        )
    finally:
        if marker is not None:
            marker.unlink(missing_ok=True)


def _compare_problem_inner(
    root: str | None,
    name: str,
    options: ipax.Options,
    max_vars: int,
    ref_max_iter: int,
    ref_max_time: float | None,
    max_build_seconds: float,
    ref_options: tuple[tuple[str, Any], ...],
) -> tuple[Row | None, str | None]:
    """The comparison proper; see :func:`_compare_problem`."""
    if max_build_seconds > 0 and not _build_within(
        str(root), name, None, max_build_seconds
    ):
        return None, "slow_build"
    try:
        case = _build_case(root, name)
        problem, x0 = case.build(np)
    except NotImplementedError:
        return None, "no_objective"
    except Exception:
        return None, "build_error"

    if max_vars and int(problem.n_vars) > max_vars:
        return None, "too_large"

    row = _compare_built(
        name, case, problem, x0, options, ref_max_iter, ref_max_time, ref_options
    )
    return row, None


def compare_one(name: str, options: ipax.Options, *, root: str | None = None) -> Row:
    """Solve one S2MPJ problem with ipax and the reference; score both.

    The unguarded single-problem entry point (raises on a build failure); the
    sweep goes through :func:`_compare_problem`.
    """
    case = _build_case(root, name)
    problem, x0 = case.build(np)
    return _compare_built(name, case, problem, x0, options, 3000, None)


# -- report -------------------------------------------------------------------


def _atomic_write(
    path: Path, text: str, *, attempts: int = 5, retry_delay: float = 0.2
) -> None:
    """Write ``text`` to ``path`` without ever leaving it truncated.

    The report is rewritten after every problem so an unattended run survives a
    crash — which makes the rewrite itself the danger: a plain write truncates
    first, so a process killed in that window replaces hours of rows with an
    empty file. Write beside the target, then rename over it.

    The rename is retried: on Windows it raises ``PermissionError`` (WinError 5)
    whenever the target is momentarily open — an antivirus scan, the search
    indexer, or just someone reading the report while the sweep runs. That is
    transient, and treating it as fatal cost a 964-row run.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(attempts):
        try:
            tmp.replace(path)
            return
        except OSError:
            if attempt == attempts - 1:
                tmp.unlink(missing_ok=True)
                raise
            sleep(retry_delay)


def _write_reports(
    json_path: Path, md_path: Path, payload: str, report: str, retry_delay: float = 0.2
) -> None:
    """Flush both reports, degrading a write failure to a warning.

    Flushing exists so an unattended run survives a crash, so it must never be
    the thing that ends one: a report that cannot be written right now is
    rewritten in full after the next problem anyway.
    """
    try:
        _atomic_write(json_path, payload, retry_delay=retry_delay)
        _atomic_write(md_path, report, retry_delay=retry_delay)
    except OSError as exc:
        print(f"warning: could not write the report ({exc}); continuing", flush=True)


def _row_from_dict(row: dict[str, object]) -> Row:
    """Rebuild a :class:`Row` from a persisted JSON row, ignoring unknown keys."""
    fields = {f.name for f in dataclasses.fields(Row)}
    return Row(**{k: v for k, v in row.items() if k in fields})  # type: ignore[arg-type]


def to_payload(
    rows: list[Row],
    environment: dict[str, object],
    config: str,
    skipped: dict[str, int],
    parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    """JSON-ready payload bundling the environment with every comparison row."""
    return {
        "environment": environment,
        "config": config,
        "parameters": parameters or {},
        "skipped": dict(skipped),
        "rows": [asdict(r) for r in rows],
    }


def _fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.6g}"


def _gaps(rows: list[Row]) -> list[Row]:
    """The actionable rows: the reference solved it and ipax did not."""
    return [r for r in rows if r.verdict in ("ipax-gap", "ipax-gap*")]


def _parameter_section(parameters: dict[str, object]) -> list[str]:
    """The two solvers' settings, with the knobs that do *not* match called out.

    Without this a reader takes the verdicts for a like-for-like comparison.
    They are not: ipax and IPOPT are compared **as each ships**, and their
    defaults differ in ways that move verdicts.
    """
    if not parameters:
        return []
    lines = ["## Parameters", ""]
    for side in ("ipax", "reference"):
        settings = parameters.get(side)
        if not isinstance(settings, dict):
            continue
        rendered = ", ".join(f"`{k}={v}`" for k, v in settings.items() if v != {})
        lines.append(f"- **{side}**: {rendered}")
    lines += [
        "",
        "Aligned on purpose: starting point, gradient-based NLP scaling "
        "(`max_gradient=100`), a `1e-8` convergence tolerance with a `1e-6` "
        "acceptable level, a limited-memory Hessian, and a filter line search.",
        "",
        "**Not aligned** — these are defaults-as-shipped on both sides, so read "
        "a verdict as *ipax-as-shipped vs IPOPT-as-shipped*, not as an "
        "algorithm-for-algorithm result:",
        "",
        "- **μ strategy**: ipax defaults to `monotone`; IPOPT's build default "
        "is `adaptive` (verified: on `AGG`, unset and `adaptive` both take 185 "
        "iterations, `monotone` takes 352). Re-run a gap with "
        "`--ref-option mu_strategy=monotone` to separate a μ-schedule "
        "difference from a structural gap.",
        "- **L-BFGS history**: ipax keeps 10 pairs, IPOPT's default is 6 "
        "(verified: `limited_memory_max_history=10` moves `AGG` from 185 to "
        "100 iterations).",
        "- **Linear algebra**: this ipax config solves the condensed normal "
        "equations densely; IPOPT factors the sparse augmented system with "
        "MUMPS and an inertia correction.",
        "- **Wall-clock budget**: nominally equal, but a second buys compiled "
        "IPOPT far more iterations than pure-Python ipax, so an ipax "
        "`max_time` row is not evidence of an algorithmic gap.",
        "",
    ]
    return lines


def format_report(
    rows: list[Row],
    environment: dict[str, object],
    config: str,
    skipped: dict[str, int],
    parameters: dict[str, object] | None = None,
) -> str:
    """Render the comparison as a Markdown report (verdicts, gaps, detail)."""
    tally = Counter(r.verdict for r in rows)
    skip_total = sum(skipped.values())
    lines = [
        "# ipax vs IPOPT on S2MPJ",
        "",
        f"- generated: `{environment.get('timestamp')}`",
        f"- ipax `{environment.get('ipax')}` · python `{environment.get('python')}`"
        f" on `{environment.get('platform')}`",
        f"- ipax config: `{config}`",
        f"- **{len(rows)} problems compared**"
        + (
            f" · {skip_total} skipped ("
            + ", ".join(f"{k}: {v}" for k, v in sorted(skipped.items()) if v)
            + ")"
            if skip_total
            else ""
        ),
        "",
        "Iteration counts are the comparable axis; wall-clock is recorded but "
        "compares a compiled solver against a pure-Python one, so it is never "
        "the headline. A `*` verdict marks a problem the dataset documents no "
        "objective for; those are scored on solver *success* first, so an "
        "`ipax-wins*` can still hide a reference that reported a better "
        "objective before running out of iterations (S2MPJ `OET7`) — read the "
        "objective columns before believing a starred verdict.",
        "",
        *_parameter_section(parameters or {}),
        "## Verdicts",
        "",
        "| verdict | count | meaning |",
        "| --- | --- | --- |",
    ]
    for verdict, count in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])):
        meaning = _VERDICT_MEANING.get(verdict, "")
        lines.append(f"| `{verdict}` | {count} | {meaning} |")

    gaps = _gaps(rows)
    lines += [
        "",
        "## ipax-gap — the reference solved it, ipax did not",
        "",
    ]
    if gaps:
        lines += [
            "The triage backlog: each row is a problem an established solver "
            "reaches from the same starting point.",
            "",
            "| problem | n | ipax status | IPOPT iters "
            "| objective (ipax → IPOPT) | violation (ipax → IPOPT) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in sorted(gaps, key=lambda r: r.problem):
            lines.append(
                f"| {r.problem} | {r.n_vars} | {r.ipax_status} | {r.ref_iters} "
                f"| {_fmt(r.ipax_objective)} → {_fmt(r.ref_objective)} "
                f"| {_fmt(r.ipax_infeasibility)} → {_fmt(r.ref_infeasibility)} |"
            )
    else:
        lines.append("None — the reference solved nothing ipax missed.")

    lines += [
        "",
        "## Per-problem detail",
        "",
        "| problem | n | ipax status | ipax it | ipax f | ipax viol | IPOPT "
        "| IPOPT it | IPOPT f | IPOPT viol | f* | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(rows, key=lambda r: (r.verdict, r.problem)):
        ref = "ok" if r.ref_success else ("—" if r.ref_error else "fail")
        lines.append(
            f"| {r.problem} | {r.n_vars} | {r.ipax_status} | {r.ipax_iters} "
            f"| {_fmt(r.ipax_objective)} | {_fmt(r.ipax_infeasibility)} "
            f"| {ref} | {r.ref_iters} | {_fmt(r.ref_objective)} "
            f"| {_fmt(r.ref_infeasibility)} | {_fmt(r.expected_objective)} "
            f"| `{r.verdict}` |"
        )
    return "\n".join(lines) + "\n"


def _format(rows: list[Row]) -> str:
    """The console table (small runs); the full report goes to ``--out``."""
    out: list[str] = []
    head = (
        f"{'problem':16s} {'n':>5s} {'ipax status':16s} {'ip.it':>5s} "
        f"{'IPOPT':>7s} {'ref.it':>6s} {'verdict':>9s}"
    )
    out.append(head)
    out.append("-" * len(head))
    for r in sorted(rows, key=lambda x: (x.verdict, x.problem)):
        ref = "ok" if r.ref_success else ("—" if r.ref_error else "fail")
        out.append(
            f"{r.problem:16s} {r.n_vars:5d} {r.ipax_status:16s} {r.ipax_iters:5d} "
            f"{ref:>7s} {r.ref_iters:6d} {r.verdict:>9s}"
        )
    return "\n".join(out)


def _summary(rows: list[Row], skipped: Counter[str]) -> str:
    tally = Counter(r.verdict for r in rows)
    out = [
        "verdicts: " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())),
    ]
    if skipped:
        out.append(
            "skipped: " + "  ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
        )
    gaps = [r.problem for r in _gaps(rows)]
    if gaps:
        out.append(f"ipax-gap ({len(gaps)}): " + " ".join(sorted(gaps)))
    return "\n".join(out)


# -- driver -------------------------------------------------------------------


def _parse_ref_option(text: str) -> tuple[str, Any]:
    """``"mu_strategy=monotone"`` → ``("mu_strategy", "monotone")``, typed.

    IPOPT dispatches on the Python type, so numeric-looking values are handed
    over as numbers rather than strings.
    """
    key, sep, raw = text.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"reference option must be key=value, got {text!r}")
    value: Any = raw.strip()
    for cast in (int, float):
        try:
            value = cast(value)
            break
        except ValueError:
            continue
    return key.strip(), value


def _ref_settings(
    args: argparse.Namespace,
) -> tuple[int, float | None, tuple[tuple[str, Any], ...]]:
    """The reference solver's budget and option overrides.

    Both caps default to ipax's. A reference handed a larger budget than ipax
    manufactures ``ipax-gap`` rows — the verdict would report the budget, not
    the algorithm — so symmetry is the default and an asymmetry has to be asked
    for explicitly.
    """
    max_iter = args.max_iter if args.ref_max_iter is None else args.ref_max_iter
    max_time = args.max_time if args.ref_max_time is None else args.ref_max_time
    overrides = tuple(_parse_ref_option(o) for o in (args.ref_option or []))
    return max_iter, max_time, overrides


def _parameters(
    options: ipax.Options,
    ref_max_iter: int,
    ref_max_time: float | None,
    ref_options: tuple[tuple[str, Any], ...],
) -> dict[str, object]:
    """Both solvers' settings, recorded so a report is self-describing."""
    return {
        "ipax": {
            "hessian": options.hessian,
            "linsolve": options.linsolve,
            "mu_schedule": options.mu_schedule,
            "globalization": options.globalization,
            "lbfgs_memory": options.lbfgs.memory,
            "scaling": options.scaling.method,
            "dual_inf_tol": options.optimality.dual_inf_tol,
            "max_iter": options.max_iter,
            "max_time": options.max_time,
        },
        "reference": {
            "solver": IpyoptBaseline.name,
            "hessian_approximation": "limited-memory",
            "tol": 1e-8,
            "max_iter": ref_max_iter,
            "max_time": ref_max_time,
            "overrides": dict(ref_options),
        },
    }


def _select_names(args: argparse.Namespace, root: str | None) -> tuple[str, ...]:
    """Resolve the problem selection: a file, explicit names, the corpus, or none."""
    if args.names_file:
        text = Path(args.names_file).read_text()
        lines = (line.split("#", 1)[0].strip() for line in text.splitlines())
        return tuple(line for line in lines if line)
    explicit = list(args.problems)
    if args.names:
        explicit += [n.strip() for n in args.names.split(",") if n.strip()]
    if explicit:
        return tuple(explicit)
    if args.all:
        names = list_s2mpj_problems(root)
        return tuple(names[: args.limit] if args.limit else names)
    if args.sample:
        return tuple(list_s2mpj_problems(root)[: args.sample])
    return ()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare ipax against IPOPT (ipyopt) on S2MPJ problems."
    )
    parser.add_argument("problems", nargs="*", help="explicit S2MPJ problem names")
    parser.add_argument(
        "--names", default=None, help="comma-separated S2MPJ problem names"
    )
    parser.add_argument(
        "--names-file",
        default=None,
        help="file with one problem name per line (# comments allowed); takes "
        "precedence over the other selectors — handy for re-running a subset "
        "such as the previous run's ipax-gap list",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="compare every problem in the checkout (the full CUTEst set)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="cap the number of problems (0 = all)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="instead, take the first N corpus problems",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="comma-separated problem names to skip (e.g. one that natively "
        "crashes a worker); combine with --resume to step past it",
    )
    parser.add_argument(
        "--dir", default=None, help="S2MPJ checkout (else $IPAX_S2MPJ_DIR)"
    )
    parser.add_argument("--config", default="lbfgs/dense", help="ipax hessian/linsolve")
    parser.add_argument(
        "--max-vars",
        type=int,
        default=_DENSE_MAX_VARS,
        help=f"variable cap (default {_DENSE_MAX_VARS}, the accuracy sweep's dense "
        "cap, so the two reports cover the same problems; 0 = no cap)",
    )
    parser.add_argument("--max-iter", type=int, default=1000, help="ipax iteration cap")
    parser.add_argument(
        "--max-time", type=float, default=60.0, help="per-solve wall-time cap (seconds)"
    )
    parser.add_argument(
        "--ref-max-iter",
        type=int,
        default=None,
        help="reference iteration cap (default: --max-iter). Giving the "
        "reference a larger budget manufactures ipax-gap rows, so ask for an "
        "asymmetry only on purpose.",
    )
    parser.add_argument(
        "--ref-max-time",
        type=float,
        default=None,
        help="reference wall-time cap in seconds (default: --max-time)",
    )
    parser.add_argument(
        "--ref-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra IPOPT option, repeatable — the parameter-matched-arm lever. "
        "IPOPT's defaults are not ipax's: 'mu_strategy=monotone' matches ipax's "
        "default barrier schedule (IPOPT's build default is adaptive) and "
        "'limited_memory_max_history=10' matches ipax's L-BFGS memory (IPOPT "
        "defaults to 6).",
    )
    parser.add_argument(
        "--max-build-seconds",
        type=float,
        default=0.0,
        help="skip a problem whose build exceeds this wall-time, probed in a "
        "subprocess (0 = no build guard)",
    )
    parser.add_argument(
        "--out",
        default="benchmarks/reports/s2mpj_baselines",
        help="report path (a .json and a .md are written)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep rows from an existing --out report and skip problems already "
        "in it, so a run that died continues instead of starting over",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="compare this many problems concurrently in worker processes "
        "(default 1 = serial); pin BLAS threads for comparable timings",
    )
    args = parser.parse_args(argv)

    root = s2mpj_dir(args.dir)
    if root is None:
        print(
            "S2MPJ comparison: no checkout found (set IPAX_S2MPJ_DIR or pass "
            "--dir); nothing to do"
        )
        return 0

    names = tuple(dict.fromkeys(_select_names(args, root)))
    if not names:
        parser.error("give problem names, --names/--names-file, --all, or --sample N")

    hessian, linsolve = args.config.split("/")
    options = ipax.Options(
        hessian=hessian,
        linsolve=linsolve,
        max_iter=args.max_iter,
        max_time=args.max_time,
    )
    ref_max_iter, ref_max_time, ref_options = _ref_settings(args)
    parameters = _parameters(options, ref_max_iter, ref_max_time, ref_options)
    environment = capture_environment()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    md_path = out.with_suffix(".md")
    inflight_path = out.with_suffix(".inflight")

    rows: list[Row] = []
    skipped: Counter[str] = Counter()
    if args.resume and json_path.exists():
        prior = json.loads(json_path.read_text())
        environment = prior.get("environment", environment)
        rows = [_row_from_dict(r) for r in prior.get("rows", [])]
        skipped.update(prior.get("skipped", {}))
        print(f"resuming from {json_path}: {len(rows)} rows kept")

    done = {r.problem for r in rows}
    exclude = {n.strip() for n in (args.exclude or "").split(",") if n.strip()}
    work = [n for n in names if n not in done and n not in exclude]

    def _flush() -> None:
        # Persist after every problem: an unattended run over this corpus can hit
        # a native crash (a backend factorization on an overflowed model), and the
        # report must survive it. Rows are sorted so reports are deterministic
        # regardless of --jobs completion order, and diff cleanly across runs.
        ordered = sorted(rows, key=lambda r: r.problem)
        _write_reports(
            json_path,
            md_path,
            json.dumps(
                to_payload(ordered, environment, args.config, skipped, parameters),
                indent=2,
            ),
            format_report(ordered, environment, args.config, skipped, parameters),
        )

    def _record(name: str, row: Row | None, skip: str | None) -> None:
        if skip is not None:
            skipped[skip] += 1
        if row is not None:
            rows.append(row)
        done.add(name)
        _flush()

    def _worker_args(name: str) -> tuple[object, ...]:
        return (
            root,
            name,
            options,
            args.max_vars,
            ref_max_iter,
            ref_max_time,
            args.max_build_seconds,
            ref_options,
            str(inflight_path),
        )

    if args.jobs <= 1:
        for name in work:
            # Record the in-flight problem so a wrapper can identify (and then
            # --exclude) one that natively crashes the process. Cleared in
            # ``finally``, so the file only survives a hard crash — naming
            # exactly the culprit.
            inflight_path.write_text(name)
            try:
                row, skip = _compare_problem(*_worker_args(name))  # type: ignore[arg-type]
                _record(name, row, skip)
            finally:
                inflight_path.unlink(missing_ok=True)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
            futures = {
                pool.submit(_compare_problem, *_worker_args(name)): name  # type: ignore[arg-type]
                for name in work
            }

            # No parent-side in-flight list here: every problem is submitted up
            # front, so the pending futures are the whole queue and name nothing.
            # Each worker marks its own current problem instead (see
            # ``_compare_problem``), leaving at most ``--jobs`` markers on a crash.
            try:
                for future in as_completed(dict(futures)):
                    name = futures[future]
                    row, skip = future.result()
                    del futures[future]
                    _record(name, row, skip)
            except BrokenProcessPool:
                # The workers' own markers, not the pending-futures list: with
                # the corpus submitted up front the latter is the whole queue.
                markers = sorted(inflight_path.parent.glob(inflight_path.name + ".*"))
                candidates = sorted(m.read_text(encoding="utf-8") for m in markers)
                _flush()
                print(
                    "S2MPJ comparison: a worker process died (native crash?); the "
                    f"culprit is one of the {len(candidates)} problem(s) that were "
                    f"running: {', '.join(candidates) or '(no marker survived)'} — "
                    "then --resume --exclude it"
                )
                # Exit 2 so a wrapper can tell an infrastructure failure apart
                # from a completed (advisory) run.
                return 2

    _flush()
    # The full table is the report's job; keep the console readable on a
    # corpus-scale run.
    if len(rows) <= 50:
        print(_format(rows))
    print(_summary(rows, skipped))
    print(f"-> {json_path}, {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
