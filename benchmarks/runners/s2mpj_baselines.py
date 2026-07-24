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

Usage::

    python -m benchmarks.runners.s2mpj_baselines AGG DEGENLPA OET7
    python -m benchmarks.runners.s2mpj_baselines --sample 40 --out cmp.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

import ipax
from benchmarks.baselines import BaselineUnsupported, IpyoptBaseline
from benchmarks.corpus.s2mpj import list_s2mpj_problems, s2mpj_problems

# Same objective tolerance the accuracy sweep scores ipax with, so ipax and the
# reference are judged identically (harness ``_OBJ_GATE``).
_OBJ_GATE = 1e-4


def _objective_ok(objective: float, expected: float | None) -> bool | None:
    """Whether ``objective`` matches the documented optimum (``None`` if none)."""
    if expected is None or not math.isfinite(expected):
        return None
    return abs(objective - expected) <= _OBJ_GATE * (1.0 + abs(expected))


@dataclass
class Row:
    problem: str
    n_vars: int
    expected_objective: float | None
    ipax_status: str
    ipax_iters: int
    ipax_objective: float
    ipax_correct: bool | None
    ref_name: str
    ref_success: bool
    ref_iters: int
    ref_objective: float | None
    ref_correct: bool | None
    ref_error: str | None
    verdict: str
    ipax_time: float
    ref_time: float


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


def compare_one(name: str, options: ipax.Options) -> Row:
    """Solve one S2MPJ problem with ipax and the reference; score both."""
    built = s2mpj_problems([name], hessian="lbfgs", sparse=True, feasibility=True)
    if not built:
        raise RuntimeError(f"{name}: not available in this S2MPJ checkout")
    problem, x0 = built[0].build(np)
    expected = getattr(problem, "expected_objective", None)

    t0 = perf_counter()
    r = ipax.solve(problem, x0, options=options)
    ipax_time = perf_counter() - t0
    ipax_obj = float(r.objective)
    ipax_ok = _objective_ok(ipax_obj, expected)

    ref_name = "ipyopt"
    ref_success = False
    ref_iters = 0
    ref_obj: float | None = None
    ref_ok: bool | None = None
    ref_err: str | None = None
    ref_time = 0.0
    try:
        # A fresh build so the reference sees the untouched starting point.
        ref_problem, ref_x0 = built[0].build(np)
        ref = IpyoptBaseline().solve(ref_problem, ref_x0)
        ref_name = ref.name
        ref_success = ref.success
        ref_iters = ref.n_iter
        ref_obj = float(ref.objective)
        ref_ok = _objective_ok(ref_obj, expected)
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
        ipax_success = r.status.value in ("optimal", "acceptable")
        if ref_err is not None:
            verdict = "ref-error"
        elif ipax_success and not ref_success:
            verdict = "ipax-wins*"
        elif ref_success and not ipax_success:
            verdict = "ipax-gap*"
        elif ref_obj is not None:
            rel = abs(ipax_obj - ref_obj) / max(1.0, abs(ref_obj))
            verdict = "agree*" if rel <= 1e-4 else "differ*"

    return Row(
        problem=name,
        n_vars=int(problem.n_vars),
        expected_objective=expected,
        ipax_status=r.status.value,
        ipax_iters=r.n_iter,
        ipax_objective=ipax_obj,
        ipax_correct=ipax_ok,
        ref_name=ref_name,
        ref_success=ref_success,
        ref_iters=ref_iters,
        ref_objective=ref_obj,
        ref_correct=ref_ok,
        ref_error=ref_err,
        verdict=verdict,
        ipax_time=ipax_time,
        ref_time=ref_time,
    )


def _format(rows: list[Row]) -> str:
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
    tally: dict[str, int] = {}
    for r in rows:
        tally[r.verdict] = tally.get(r.verdict, 0) + 1
    out.append("")
    out.append("verdicts: " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    gaps = [r.problem for r in rows if r.verdict in ("ipax-gap", "ipax-gap*")]
    if gaps:
        out.append("ipax-gap (reference solved, ipax did not): " + " ".join(gaps))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare ipax against IPOPT (ipyopt) on S2MPJ problems."
    )
    parser.add_argument("problems", nargs="*", help="explicit S2MPJ problem names")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="instead, take the first N corpus problems",
    )
    parser.add_argument("--config", default="lbfgs/dense", help="ipax hessian/linsolve")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--out", default=None, help="write the rows to this JSON path")
    args = parser.parse_args(argv)

    names = list(args.problems)
    if args.sample:
        names = list_s2mpj_problems()[: args.sample]
    if not names:
        parser.error("give problem names or --sample N")

    hessian, linsolve = args.config.split("/")
    options = ipax.Options(
        hessian=hessian, linsolve=linsolve, max_iter=args.max_iter, max_time=300.0
    )

    rows: list[Row] = []
    for name in names:
        try:
            rows.append(compare_one(name, options))
        except Exception as exc:
            print(f"{name}: SKIP ({type(exc).__name__}: {exc})", flush=True)
    print(_format(rows))

    if args.out:
        payload: dict[str, Any] = {
            "config": args.config,
            "rows": [asdict(r) for r in rows],
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
