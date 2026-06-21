"""Reference cross-check runner.

Solves the curated corpus with ipax and every available reference baseline
(SciPy now; cyipopt/OSQP when installed) and reports whether the solutions
agree. Cross-checks are NumPy-only — the reference solvers consume NumPy arrays.

Run from the repository root::

    python -m benchmarks.runners.crosscheck --out benchmarks/reports/crosscheck

This is **advisory**: it exits non-zero only if ipax itself raises on a case
(a real bug). Disagreement with a reference is flagged in the report but does
not fail, since the QC sweep already gates accuracy against known optima.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import ipax
from benchmarks.baselines import available_baselines
from benchmarks.corpus import default_corpus
from benchmarks.harness import (
    CrossCheckResult,
    capture_environment,
    cross_check,
    format_crosscheck,
)
from ipax.testing.backends import import_namespace

# The cross-check holds the solver strategy fixed and varies only the reference.
_REFERENCE_CONFIG = ipax.Options(hessian="exact", linsolve="dense")


def run_crosscheck() -> tuple[list[CrossCheckResult], dict[str, object]]:
    environment = capture_environment()
    baselines = available_baselines()
    results: list[CrossCheckResult] = []
    if not baselines:
        return results, environment

    xp = import_namespace("numpy")
    for case in default_corpus():
        if case.backends is not None and "numpy" not in case.backends:
            continue
        for baseline in baselines:
            results.append(
                cross_check(
                    case,
                    ipax_options=_REFERENCE_CONFIG,
                    baseline=baseline,
                    xp=xp,
                )
            )
    return results, environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ipax reference cross-check")
    parser.add_argument("--out", default="benchmarks/reports/crosscheck")
    args = parser.parse_args(argv)

    results, environment = run_crosscheck()
    if not results:
        print("cross-check: no reference baselines available (install scipy)")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"environment": environment, "results": [asdict(r) for r in results]}
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    out.with_suffix(".md").write_text(
        format_crosscheck(results, environment), encoding="utf-8"
    )

    checked = [r for r in results if r.status == "ok"]
    agreed = sum(1 for r in checked if r.agree)
    ipax_failures = [r for r in results if r.status == "ipax_failed"]
    print(
        f"cross-check: {agreed}/{len(checked)} agree "
        f"-> {out.with_suffix('.json')}, {out.with_suffix('.md')}"
    )
    for r in results:
        if r.status == "ok" and not r.agree:
            print(f"  DISAGREE {r.problem} vs {r.baseline}: {r.note or 'see report'}")
        elif r.status in ("ipax_failed", "baseline_failed"):
            print(f"  {r.status.upper()} {r.problem} vs {r.baseline}: {r.note}")
    return 1 if ipax_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
