"""Quality-control sweep runner.

Runs the curated corpus across a configuration matrix (Hessian × linear-solver ×
corrections × scaling) and every available backend, scoring each case for
correctness/robustness/accuracy, and writes a JSON + Markdown report. Intended
as a regression gate for solver quality — *not* a per-PR test (it is heavier and
lives outside ``tests/``).

Run from the repository root::

    python -m benchmarks.runners.qc --out benchmarks/reports/qc

Exits non-zero if any case is not "correct", so it can gate a nightly job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ipax
from benchmarks.corpus import BenchmarkProblem, default_corpus
from benchmarks.harness import (
    CaseResult,
    capture_environment,
    format_markdown,
    run_case,
    to_payload,
)
from ipax.backend.namespace import capabilities
from ipax.testing.backends import import_namespace


def default_configs() -> list[tuple[str, ipax.Options]]:
    """Curated solver configurations spanning the main quality axes."""
    options = ipax.Options
    return [
        ("exact/dense", options(hessian="exact", linsolve="dense")),
        ("lbfgs/dense", options(hessian="lbfgs", linsolve="dense")),
        (
            "exact/dense+mehrotra",
            options(hessian="exact", linsolve="dense", corrections="mehrotra"),
        ),
        (
            "exact/dense+gondzio",
            options(hessian="exact", linsolve="dense", corrections="gondzio"),
        ),
        (
            "exact/dense+scaled",
            options(hessian="exact", linsolve="dense", scaling="gradient-based"),
        ),
        ("exact/krylov", options(hessian="exact", linsolve="krylov")),
        ("lbfgs/krylov", options(hessian="lbfgs", linsolve="krylov")),
        ("exact/sparse", options(hessian="exact", linsolve="sparse")),
    ]


def run_sweep(
    corpus: list[BenchmarkProblem],
    configs: list[tuple[str, ipax.Options]],
    backends: list[str],
) -> tuple[list[CaseResult], dict[str, object]]:
    """Solve every (case, backend, config) cell; collect :class:`CaseResult`s."""
    environment = capture_environment()
    results: list[CaseResult] = []
    for backend in backends:
        try:
            xp = import_namespace(backend)
        except ImportError:
            continue
        caps = capabilities(xp)
        for case in corpus:
            if case.backends is not None and backend not in case.backends:
                continue
            for label, options in configs:
                if options.linsolve == "sparse" and not caps.has_sparse_adapter:
                    continue
                results.append(
                    run_case(
                        case,
                        config=label,
                        options=options,
                        xp=xp,
                        backend=backend,
                    )
                )
    return results, environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ipax quality-control benchmark")
    parser.add_argument(
        "--out",
        default="benchmarks/reports/qc",
        help="output path stem (writes <out>.json and <out>.md)",
    )
    parser.add_argument(
        "--backends",
        default="numpy,torch",
        help="comma-separated backends to sweep (unavailable ones are skipped)",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="skip the matplotlib plot"
    )
    args = parser.parse_args(argv)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    results, environment = run_sweep(default_corpus(), default_configs(), backends)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    md_path = out.with_suffix(".md")
    json_path.write_text(json.dumps(to_payload(results, environment), indent=2))
    md_path.write_text(format_markdown(results, environment), encoding="utf-8")
    if not args.no_plot:
        from benchmarks.harness.plots import PlottingUnavailable, plot_qc_iterations

        try:
            plot_qc_iterations(results, out.with_name(out.stem + "_iters.png"))
        except PlottingUnavailable:
            pass

    n_correct = sum(1 for r in results if r.correct)
    flagged = [r for r in results if not r.correct]
    print(f"QC sweep: {n_correct}/{len(results)} correct -> {json_path}, {md_path}")
    for r in flagged:
        reason = r.error.splitlines()[-1] if r.error else f"status={r.status}"
        print(f"  FAIL {r.problem} [{r.backend}] {r.config}: {reason}")
    return 0 if not flagged else 1


if __name__ == "__main__":
    raise SystemExit(main())
