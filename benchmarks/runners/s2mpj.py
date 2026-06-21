"""S2MPJ accuracy sweep runner (download-gated, opt-in).

Solves the S2MPJ-translated CUTEst problems with ipax's default L-BFGS Hessian
across the host-bridgeable backends, scoring each case and writing a JSON +
Markdown report — the broad-coverage complement to the curated QC corpus.

This is **not** part of the per-PR pipeline: it needs a local S2MPJ checkout (no
license to vendor), so it returns early with nothing to do when ``IPAX_S2MPJ_DIR``
(or ``--dir``) is unset. Run from the repository root::

    IPAX_S2MPJ_DIR=/path/to/S2MPJ python -m benchmarks.runners.s2mpj

S2MPJ problems carry no analytic Lagrangian Hessian through the bridge, so only
L-BFGS configurations are swept. Exits non-zero if any case is not "correct".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ipax
from benchmarks.corpus.s2mpj import s2mpj_dir, s2mpj_problems
from benchmarks.harness import (
    CaseResult,
    capture_environment,
    format_markdown,
    run_case,
    to_payload,
)
from ipax.testing.backends import import_namespace


def default_configs() -> list[tuple[str, ipax.Options]]:
    """L-BFGS configurations (no analytic Hessian crosses the NumPy bridge)."""
    options = ipax.Options
    return [
        ("lbfgs/dense", options(hessian="lbfgs", linsolve="dense")),
        ("lbfgs/krylov", options(hessian="lbfgs", linsolve="krylov")),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ipax S2MPJ accuracy sweep")
    parser.add_argument("--out", default="benchmarks/reports/s2mpj")
    parser.add_argument(
        "--dir", default=None, help="S2MPJ checkout (else $IPAX_S2MPJ_DIR)"
    )
    parser.add_argument(
        "--backends",
        default="numpy",
        help="comma-separated host backends to bridge (numpy,torch)",
    )
    parser.add_argument(
        "--names",
        default=None,
        help="comma-separated S2MPJ problem names (else a curated default set)",
    )
    args = parser.parse_args(argv)

    root = s2mpj_dir(args.dir)
    if root is None:
        print(
            "S2MPJ sweep: no checkout found (set IPAX_S2MPJ_DIR or pass --dir); "
            "nothing to do"
        )
        return 0

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    names = (
        tuple(n.strip() for n in args.names.split(",") if n.strip())
        if args.names
        else None
    )
    configs = default_configs()
    environment = capture_environment()

    results: list[CaseResult] = []
    for backend in backends:
        try:
            xp = import_namespace(backend)
        except ImportError:
            continue
        corpus = s2mpj_problems(names, directory=root, backends=(backend,))
        for case in corpus:
            for label, options in configs:
                results.append(
                    run_case(
                        case, config=label, options=options, xp=xp, backend=backend
                    )
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps(to_payload(results, environment), indent=2)
    )
    out.with_suffix(".md").write_text(
        format_markdown(results, environment), encoding="utf-8"
    )

    n_correct = sum(1 for r in results if r.correct)
    flagged = [r for r in results if not r.correct]
    print(
        f"S2MPJ sweep: {n_correct}/{len(results)} correct -> {out.with_suffix('.json')}"
    )
    for r in flagged:
        reason = r.error.splitlines()[-1] if r.error else f"status={r.status}"
        print(f"  FAIL {r.problem} [{r.backend}] {r.config}: {reason}")
    return 0 if not flagged else 1


if __name__ == "__main__":
    raise SystemExit(main())
