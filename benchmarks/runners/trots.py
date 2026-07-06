"""TROTS corpus runner (download-gated, opt-in).

Two jobs, both gated on a local TROTS dataset (``IPAX_TROTS_DIR`` or ``--dir``):

1. **Verification** (default, fast): for every case, evaluate the scalarised
   objective at the dataset's reference plan ``solutionX`` and compare it to the
   ``Objective Function Value`` in the matching ``Results/*.txt`` file. This is the
   loader's ground-truth oracle — it exercises every cost-function type
   (linear/quadratic/gEUD/LTCP/DVH) and the weighting/scalarisation without
   running the solver, so it is cheap and deterministic.

2. **Solve** (``--solve``, slow): additionally run ipax on each case and score the
   result with the shared harness. These radiotherapy problems have far more
   constraints than variables (``n ≈ 1e3``, ``m`` up to several ``1e5``), so the
   **condensed** routes (``lbfgs/dense``, ``exact/dense``, and the matrix-free
   ``*/krylov``) — which consume the sparse dose matrices through matvecs and form
   only the ``n×n`` system — are the appropriate default; the sparse-direct route
   (``--config lbfgs/sparse,exact/sparse``) factors the full ``(n+m)`` saddle and
   is impractical at these row counts. The generic routes do not replace the
   specialised dose-matrix condensation (out of scope, see ``AGENTS.md``), so the
   large cases may hit ``--max-time`` before converging.

Run from the repository root::

    IPAX_TROTS_DIR=/path/to/TROTS python -m benchmarks.runners.trots
    IPAX_TROTS_DIR=/path/to/TROTS python -m benchmarks.runners.trots --solve \
        --cases Prostate_BT_01 --config lbfgs/sparse,exact/sparse

Exits non-zero if any verification case's objective does not match its reference
(within the significant figures that reference reports).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import ipax
from benchmarks.corpus.trots import (
    TROTSExactProblem,
    TROTSProblem,
    list_trots_cases,
    load_trots_file,
    objective_at,
    reference_for,
    trots_dir,
)
from benchmarks.harness import capture_environment, run_case
from ipax.testing.backends import import_namespace

# The first patient of each data group (a broad-coverage default selection).
_DEFAULT_CASES = (
    "Protons_01",
    "Prostate_BT_01",
    "Prostate_CK_01",
    "Prostate_VMAT_101",
    "Liver_01",
    "Head-and-Neck_01",
)

# (label, options). Condensed routes lead — they fit the n ≪ m regime.
_CONFIGS: dict[str, dict[str, object]] = {
    "lbfgs/dense": {"hessian": "lbfgs", "linsolve": "dense"},
    "exact/dense": {"hessian": "exact", "linsolve": "dense"},
    "lbfgs/krylov": {"hessian": "lbfgs", "linsolve": "krylov"},
    "exact/krylov": {"hessian": "exact", "linsolve": "krylov"},
    "lbfgs/sparse": {"hessian": "lbfgs", "linsolve": "sparse"},
    "exact/sparse": {"hessian": "exact", "linsolve": "sparse"},
}
_DEFAULT_SOLVE_CONFIGS = ("lbfgs/dense", "exact/dense")


@dataclass
class VerificationRow:
    """One case's objective-reproduction check against its reference result."""

    case: str
    n_vars: int
    n_entries: int
    objective_at_solution: float
    reference_objective: float | None
    rel_error: float | None
    significant_figures: int | None
    verified: bool


def _verify_case(root: str, case: str) -> VerificationRow:
    instance = load_trots_file(f"{root}/{case}.mat")
    obj = objective_at(instance, instance.solution)
    ref = reference_for(case, root)
    if ref is None:
        return VerificationRow(
            case=case,
            n_vars=instance.n,
            n_entries=len(instance.entries),
            objective_at_solution=obj,
            reference_objective=None,
            rel_error=None,
            significant_figures=None,
            verified=True,  # nothing to contradict (no reference file)
        )
    rel = abs(obj - ref.objective) / max(1.0, abs(ref.objective))
    tol = 10.0 ** (-(max(ref.significant_figures, 1) - 1))
    return VerificationRow(
        case=case,
        n_vars=instance.n,
        n_entries=len(instance.entries),
        objective_at_solution=obj,
        reference_objective=ref.objective,
        rel_error=rel,
        significant_figures=ref.significant_figures,
        verified=rel <= tol,
    )


def _solve_case(root: str, case: str, label: str, opts: ipax.Options, xp: object):
    """Score one solve via the shared harness (wrapping the case as a corpus problem)."""
    from benchmarks.corpus import BenchmarkProblem

    hessian = str(opts.hessian)

    def build(namespace):
        instance = load_trots_file(f"{root}/{case}.mat")
        cls = TROTSExactProblem if hessian == "exact" else TROTSProblem
        problem = cls(instance, namespace, sparse=True)
        ref = reference_for(case, root)
        problem.expected_objective = None if ref is None else ref.objective
        return problem, problem.initial_point()

    bench = BenchmarkProblem(
        name=f"trots/{case}", kind="RT", tags=("trots",), build=build
    )
    return run_case(bench, config=label, options=opts, xp=xp, backend="numpy")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ipax TROTS corpus runner")
    parser.add_argument("--out", default="benchmarks/reports/trots")
    parser.add_argument(
        "--dir", default=None, help="TROTS data dir (else $IPAX_TROTS_DIR)"
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="comma-separated case names (else first per group)",
    )
    parser.add_argument(
        "--all", action="store_true", help="verify every case in the dir"
    )
    parser.add_argument(
        "--solve", action="store_true", help="also attempt a solve per case (slow)"
    )
    parser.add_argument(
        "--config",
        default=",".join(_DEFAULT_SOLVE_CONFIGS),
        help="comma-separated solve configs: " + ", ".join(_CONFIGS),
    )
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-time", type=float, default=120.0)
    args = parser.parse_args(argv)

    root = trots_dir(args.dir)
    if root is None:
        print("TROTS runner: no dataset found (set IPAX_TROTS_DIR or pass --dir)")
        return 0

    if args.cases:
        cases = tuple(c.strip() for c in args.cases.split(",") if c.strip())
    elif args.all:
        cases = tuple(list_trots_cases(root))
    else:
        available = set(list_trots_cases(root))
        cases = tuple(c for c in _DEFAULT_CASES if c in available)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    environment = capture_environment()

    # 1) Verification (always).
    verification = [_verify_case(root, case) for case in cases]
    for row in verification:
        ref = (
            "n/a"
            if row.reference_objective is None
            else f"{row.reference_objective:.8g}"
        )
        rel = "n/a" if row.rel_error is None else f"{row.rel_error:.2e}"
        flag = "ok" if row.verified else "MISMATCH"
        print(
            f"  {row.case:20s} n={row.n_vars:6d}  obj={row.objective_at_solution:.8g}  "
            f"ref={ref:>14s}  rel_err={rel}  [{flag}]"
        )

    # 2) Solves (opt-in).
    solve_rows = []
    if args.solve:
        xp = import_namespace("numpy")
        labels = [c.strip() for c in args.config.split(",") if c.strip()]
        unknown = [label for label in labels if label not in _CONFIGS]
        if unknown:
            parser.error(f"unknown --config {unknown}; choose from {list(_CONFIGS)}")
        for case in cases:
            for label in labels:
                opts = ipax.Options(
                    max_iter=args.max_iter, max_time=args.max_time, **_CONFIGS[label]
                )
                result = _solve_case(root, case, label, opts, xp)
                solve_rows.append(result)
                print(
                    f"  {case:20s} [{label:12s}] {result.status:12s} "
                    f"iters={result.n_iter:4d} kkt={result.kkt_error:.2e} "
                    f"cviol={result.constraint_violation:.2e} t={result.solve_time:.1f}s"
                )

    payload = {
        "environment": environment,
        "verification": [asdict(r) for r in verification],
        "solves": [asdict(r) for r in solve_rows],
    }
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2))
    _write_markdown(out.with_suffix(".md"), verification, solve_rows, environment)

    n_ver = sum(1 for r in verification if r.verified)
    n_checked = sum(1 for r in verification if r.reference_objective is not None)
    print(
        f"TROTS runner: {n_ver}/{len(verification)} cases consistent "
        f"({n_checked} with references) -> {json_path}"
    )
    return 0 if all(r.verified for r in verification) else 1


def _write_markdown(path: Path, verification, solves, environment) -> None:
    lines = ["# TROTS corpus report", ""]
    lines.append("## Objective reproduction (at reference `solutionX`)")
    lines.append("")
    lines.append("| case | n | objective_at | reference | rel err | sig figs | ok |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | :---: |")
    for r in verification:
        ref = "" if r.reference_objective is None else f"{r.reference_objective:.8g}"
        rel = "" if r.rel_error is None else f"{r.rel_error:.2e}"
        sig = "" if r.significant_figures is None else str(r.significant_figures)
        lines.append(
            f"| {r.case} | {r.n_vars} | {r.objective_at_solution:.8g} | {ref} | "
            f"{rel} | {sig} | {'✓' if r.verified else '✗'} |"
        )
    if solves:
        lines += ["", "## Solves", ""]
        lines.append("| case | config | status | iters | kkt | cviol | time (s) |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        for s in solves:
            lines.append(
                f"| {s.problem} | {s.config} | {s.status} | {s.n_iter} | "
                f"{s.kkt_error:.2e} | {s.constraint_violation:.2e} | {s.solve_time:.1f} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
