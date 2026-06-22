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
from collections import Counter
from pathlib import Path

import ipax
from benchmarks.corpus.s2mpj import (
    list_s2mpj_problems,
    s2mpj_dir,
    s2mpj_problems,
)
from benchmarks.harness import (
    CaseResult,
    capture_environment,
    format_markdown,
    run_case,
    to_payload,
)
from ipax.testing.backends import import_namespace


def default_configs(
    max_iter: int, max_time: float | None, scaling: str = "none"
) -> list[tuple[str, ipax.Options]]:
    """L-BFGS configurations (no analytic Hessian crosses the NumPy bridge)."""
    options = ipax.Options
    common = {
        "hessian": "lbfgs",
        "max_iter": max_iter,
        "max_time": max_time,
        "scaling": scaling,
    }
    return [
        ("lbfgs/dense", options(linsolve="dense", **common)),
        ("lbfgs/krylov", options(linsolve="krylov", **common)),
    ]


def _select_names(args: argparse.Namespace, root: str) -> tuple[str, ...] | None:
    """Resolve the problem selection: explicit names, the whole set, or curated."""
    if args.names:
        return tuple(n.strip() for n in args.names.split(",") if n.strip())
    if args.all:
        names = list_s2mpj_problems(root)
        if args.limit:
            names = names[: args.limit]
        return tuple(names)
    return None  # curated default in s2mpj_problems


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
    parser.add_argument(
        "--all",
        action="store_true",
        help="sweep every problem in the checkout (the full CUTEst set)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="cap the number of problems (0 = all)"
    )
    parser.add_argument(
        "--max-vars",
        type=int,
        default=1000,
        help="skip problems with more than this many variables (0 = no cap)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000, help="solver iteration cap"
    )
    parser.add_argument(
        "--max-time", type=float, default=60.0, help="per-solve wall-time cap (seconds)"
    )
    parser.add_argument(
        "--scaling",
        default="none",
        help="problem scaling: 'none' or 'gradient-based'",
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
    names = _select_names(args, root)
    configs = default_configs(args.max_iter, args.max_time, args.scaling)
    environment = capture_environment()

    results: list[CaseResult] = []
    skipped_no_objective = 0
    skipped_too_large = 0
    for backend in backends:
        try:
            xp = import_namespace(backend)
        except ImportError:
            continue
        for case in s2mpj_problems(names, directory=root, backends=(backend,)):
            # One guarded build to gate applicability/size before running configs:
            # objective-free problems are not minimization problems, and the size
            # cap keeps the full sweep tractable. Genuine build errors fall through
            # to run_case so their traceback is recorded as a row.
            try:
                problem, _x0 = case.build(xp)
            except NotImplementedError:
                skipped_no_objective += 1
                continue
            except Exception:
                results.append(
                    run_case(
                        case,
                        config=configs[0][0],
                        options=configs[0][1],
                        xp=xp,
                        backend=backend,
                    )
                )
                continue
            if args.max_vars and int(problem.n_vars) > args.max_vars:
                skipped_too_large += 1
                continue
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

    by_status: Counter[str] = Counter(r.status for r in results)
    n_correct = sum(1 for r in results if r.correct)
    print(
        f"S2MPJ sweep: {n_correct}/{len(results)} correct "
        f"(skipped {skipped_no_objective} objective-free, {skipped_too_large} oversized)"
        f" -> {out.with_suffix('.json')}, {out.with_suffix('.md')}"
    )
    for status, count in sorted(by_status.items()):
        print(f"  {status:16s} {count}")
    return 0 if n_correct == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
