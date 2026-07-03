"""S2MPJ accuracy sweep runner (download-gated, opt-in).

Solves the S2MPJ-translated CUTEst problems across several solver configurations
and host-bridgeable backends, scoring each case and writing a JSON + Markdown
report — the broad-coverage complement to the curated QC corpus.

This is **not** part of the per-PR pipeline: it needs a local S2MPJ checkout (no
license to vendor), so it returns early with nothing to do when ``IPAX_S2MPJ_DIR``
(or ``--dir``) is unset. Run from the repository root::

    IPAX_S2MPJ_DIR=/path/to/S2MPJ python -m benchmarks.runners.s2mpj

The default sweep exercises both Hessian routes (default L-BFGS *and* the exact
Lagrangian Hessian S2MPJ supplies) over the dense, matrix-free Krylov, and
sparse-direct linear solvers. Scaling defaults to ``gradient-based`` to match the
solver default (the configuration users actually get).

Because the routes have very different size ceilings, each config carries its own
**per-route variable cap** rather than one global ``--max-vars``: a problem runs a
config only when it fits that route's cap. Small problems are therefore
cross-validated across every route, mid-size ones fall through to Krylov + sparse,
and the largest run on the sparse-direct route alone — so a single full-corpus run
(`--all`) gives each route the size range it can actually carry. ``--max-vars``
remains as a global ceiling for quick restricted runs; the per-route caps are
``--dense-max-vars`` / ``--krylov-max-vars`` / ``--sparse-max-vars``. Exits
non-zero if any case is not "correct".
"""

from __future__ import annotations

import argparse
import dataclasses
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
from ipax.options import KrylovOptions
from ipax.testing.backends import import_namespace

# Per-route variable caps. The linear-solver routes have very different size
# ceilings, so one global cap is the wrong knob: it either starves the sparse route
# of the large models it exists for, or it lets the dense route attempt sizes it
# cannot afford. The dense route forms and factors an ``n×n`` matrix (O(n²) memory,
# O(n³) factorization); the matrix-free Krylov route is O(n) memory but
# iteration-capped; the sparse-direct route factors only the nonzeros. Each config
# carries its route's cap, and a problem runs a config only when ``n ≤`` that cap —
# so a small problem is cross-validated on every route while progressively larger
# ones fall through to just the routes that can carry them.
_DENSE_MAX_VARS = 2000
_KRYLOV_MAX_VARS = 10000
_SPARSE_MAX_VARS = 25000

# (label, options, per-route variable cap). A cap of 0 means "no cap".
ConfigSpec = tuple[str, ipax.Options, int]


def default_configs(
    max_iter: int,
    max_time: float | None,
    scaling: str = "gradient-based",
    *,
    dense_max_vars: int = _DENSE_MAX_VARS,
    krylov_max_vars: int = _KRYLOV_MAX_VARS,
    sparse_max_vars: int = _SPARSE_MAX_VARS,
    krylov_preconditioner: str | None = None,
) -> list[ConfigSpec]:
    """The regular sweep matrix: both Hessian routes over the solver routes.

    ``lbfgs/dense`` is the default a user gets below ~1e4 vars; the others add the
    matrix-free Krylov route, the exact-Hessian accuracy ceiling, and the
    sparse-direct route (exact + sparse, which factors true COO sparsity). Each
    config is tagged with its route's variable cap so a single full-corpus run
    stays tractable: dense problems stay small, while the sparse route reaches the
    large models it is meant for.

    ``krylov_preconditioner`` overrides the matrix-free preconditioner on the two
    Krylov configs (default: leave :class:`KrylovOptions`'s ``jacobi``) — the lever
    for an ``auto`` vs ``jacobi`` A/B, which only affects the Krylov route.
    """
    options = ipax.Options
    common: dict[str, object] = {
        "max_iter": max_iter,
        "max_time": max_time,
        "scaling": scaling,
    }
    krylov_common = dict(common)
    if krylov_preconditioner is not None:
        krylov_common["krylov"] = KrylovOptions(preconditioner=krylov_preconditioner)  # type: ignore[arg-type]
    return [
        (
            "lbfgs/dense",
            options(hessian="lbfgs", linsolve="dense", **common),
            dense_max_vars,
        ),
        (
            "lbfgs/krylov",
            options(hessian="lbfgs", linsolve="krylov", **krylov_common),
            krylov_max_vars,
        ),
        # L-BFGS + sparse-direct is the typical radiotherapy setup.
        (
            "lbfgs/sparse",
            options(hessian="lbfgs", linsolve="sparse", **common),
            sparse_max_vars,
        ),
        (
            "exact/dense",
            options(hessian="exact", linsolve="dense", **common),
            dense_max_vars,
        ),
        # Exact Hessian + Krylov: helps attribute numerical errors to the
        # matrix-free subspace solver rather than the Hessian approximation.
        (
            "exact/krylov",
            options(hessian="exact", linsolve="krylov", **krylov_common),
            krylov_max_vars,
        ),
        (
            "exact/sparse",
            options(hessian="exact", linsolve="sparse", **common),
            sparse_max_vars,
        ),
    ]


def _problem_mode(options: ipax.Options) -> tuple[str, bool]:
    """``(hessian, sparse)`` the S2MPJ build needs for a given solver config."""
    return options.hessian, options.linsolve == "sparse"


def _effective_cap(route_cap: int, global_cap: int) -> int:
    """Tighter of the route cap and the global ceiling (0 on either = no cap)."""
    caps = [c for c in (route_cap, global_cap) if c > 0]
    return min(caps) if caps else 0


def _probe_build_n(root: str, name: str, size: int | None, queue: object) -> None:
    """Worker: instantiate one (sized) problem and report its variable count.

    Runs in a spawned subprocess so a pathological O(n²) pure-Python build can be
    abandoned by killing the process — there is no cross-platform way to interrupt
    a GIL-bound construction loop in-process.
    """
    from benchmarks.corpus.s2mpj import _instantiate

    try:
        queue.put(int(_instantiate(root, name, size).n))  # type: ignore[attr-defined]
    except Exception:
        queue.put(None)


def _build_within(root: str, name: str, size: int | None, timeout: float) -> bool:
    """Whether the (sized) build of ``name`` completes within ``timeout`` seconds."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_probe_build_n, args=(root, name, size, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False
    try:
        return queue.get_nowait() is not None
    except Exception:
        return False


def _select_names(args: argparse.Namespace, root: str) -> tuple[str, ...] | None:
    """Resolve the problem selection: a file, explicit names, the whole set, or curated."""
    if args.names_file:
        text = Path(args.names_file).read_text()
        lines = (line.split("#", 1)[0].strip() for line in text.splitlines())
        return tuple(line for line in lines if line)
    if args.names:
        return tuple(n.strip() for n in args.names.split(",") if n.strip())
    if args.all:
        names = list_s2mpj_problems(root)
        if args.limit:
            names = names[: args.limit]
        return tuple(names)
    return None  # curated default in s2mpj_problems


def _row_to_case_result(row: dict[str, object]) -> CaseResult:
    """Rebuild a :class:`CaseResult` from a persisted JSON row, tolerating an
    older schema on ``--resume``: back-fill ``converged`` from ``correct`` (a
    correct row is by definition converged) and ignore unknown keys, so a sweep
    started before the ``converged`` tier can still be resumed."""
    fields = {f.name for f in dataclasses.fields(CaseResult)}
    data = {k: v for k, v in row.items() if k in fields}
    data.setdefault("converged", bool(data.get("correct", False)))
    return CaseResult(**data)  # type: ignore[arg-type]


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
        "--names-file",
        default=None,
        help="file with one problem name per line (# comments allowed); takes "
        "precedence over --names/--all — handy for a focused re-run of a subset",
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
        default=0,
        help="global variable ceiling over every route (0 = use per-route caps)",
    )
    parser.add_argument(
        "--dense-max-vars",
        type=int,
        default=_DENSE_MAX_VARS,
        help=f"variable cap for the dense routes (default {_DENSE_MAX_VARS}; 0 = none)",
    )
    parser.add_argument(
        "--krylov-max-vars",
        type=int,
        default=_KRYLOV_MAX_VARS,
        help=f"variable cap for the Krylov route (default {_KRYLOV_MAX_VARS}; 0 = none)",
    )
    parser.add_argument(
        "--sparse-max-vars",
        type=int,
        default=_SPARSE_MAX_VARS,
        help=f"variable cap for the sparse route (default {_SPARSE_MAX_VARS}; 0 = none)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=0,
        help="target variable count for scalable problems (0 = SIF defaults); the "
        "lever for a scaling sweep that reaches the sparse route's regime",
    )
    parser.add_argument(
        "--max-build-seconds",
        type=float,
        default=0.0,
        help="skip a problem whose (sized) build exceeds this wall-time, probed in "
        "a subprocess (0 = no build guard); use it for large --size sweeps where "
        "some problems build in O(n²) pure Python",
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000, help="solver iteration cap"
    )
    parser.add_argument(
        "--max-time", type=float, default=60.0, help="per-solve wall-time cap (seconds)"
    )
    parser.add_argument(
        "--scaling",
        default="gradient-based",
        help="problem scaling: 'gradient-based' (default, matches solver) or 'none'",
    )
    parser.add_argument(
        "--preconditioner",
        default=None,
        choices=["none", "jacobi", "lbfgs", "auto"],
        help="override the matrix-free Krylov preconditioner (default: jacobi). "
        "Only affects the Krylov configs — the lever for an auto-vs-jacobi A/B.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep rows from an existing --out report and skip problems already in "
        "it — so a sweep that died (e.g. a native crash in a backend) continues "
        "instead of starting over",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="comma-separated problem names to skip (e.g. a problem that natively "
        "crashes a backend); combine with --resume to step past it",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="comma-separated config labels to run (e.g. 'exact/sparse'); default "
        "runs the whole matrix. Use it to run one configuration per process.",
    )
    parser.add_argument(
        "--include-objective-free",
        action="store_true",
        help="also run the objective-free problems (CUTEst feasibility / nonlinear "
        "equation systems) as 'min 0' subject to the constraints, instead of "
        "skipping them",
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
    size = args.size or None
    configs = default_configs(
        args.max_iter,
        args.max_time,
        args.scaling,
        dense_max_vars=args.dense_max_vars,
        krylov_max_vars=args.krylov_max_vars,
        sparse_max_vars=args.sparse_max_vars,
        krylov_preconditioner=args.preconditioner,
    )
    if args.config:
        wanted = {c.strip() for c in args.config.split(",") if c.strip()}
        configs = [spec for spec in configs if spec[0] in wanted]
        if not configs:
            parser.error(f"--config {args.config!r} matched no known config labels")
    environment = capture_environment()

    # Each config may need a differently-built problem (exact vs L-BFGS Hessian,
    # dense vs sparse operators), so build one case set per distinct mode and look
    # the problem up by name. The gate mode (default L-BFGS/dense) is always built.
    gate_mode = ("lbfgs", False)
    modes = {_problem_mode(options) for _, options, _ in configs} | {gate_mode}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    md_path = out.with_suffix(".md")
    inflight_path = out.with_suffix(".inflight")

    results: list[CaseResult] = []
    if args.resume and json_path.exists():
        prior = json.loads(json_path.read_text())
        environment = prior.get("environment", environment)
        results = [_row_to_case_result(row) for row in prior.get("results", [])]
        print(f"resuming from {json_path}: {len(results)} rows kept")
    # ``done`` is keyed by (backend, name) so a multi-backend resume re-runs the
    # problem on backends it has not yet covered.
    done = {(r.backend, r.problem.split("/", 1)[-1]) for r in results}
    exclude = {n.strip() for n in (args.exclude or "").split(",") if n.strip()}

    def _flush() -> None:
        # Persist after every problem: an unattended sweep over this corpus can hit
        # a native crash (e.g. a backend factorization on an overflowed model), and
        # the report must survive it rather than losing the whole run.
        json_path.write_text(json.dumps(to_payload(results, environment), indent=2))
        md_path.write_text(format_markdown(results, environment), encoding="utf-8")

    skipped_no_objective = 0
    skipped_too_large = 0
    skipped_slow_build = 0
    for backend in backends:
        try:
            xp = import_namespace(backend)
        except ImportError:
            continue
        cases_by_mode = {
            mode: {
                case.name: case
                for case in s2mpj_problems(
                    names,
                    directory=root,
                    backends=(backend,),
                    hessian=mode[0],
                    sparse=mode[1],
                    size=size,
                    feasibility=args.include_objective_free,
                )
            }
            for mode in modes
        }
        for case_name, gate_case in cases_by_mode[gate_mode].items():
            bare = case_name.split("/", 1)[-1]
            if (backend, bare) in done or bare in exclude:
                continue
            # Record the in-flight problem so a wrapper can identify (and then
            # --exclude) a problem that natively crashes the process. The file is
            # cleared in ``finally`` once the problem is fully handled, so it only
            # survives a hard crash — naming exactly the culprit.
            inflight_path.write_text(f"{backend} {bare}")
            try:
                # Optional build guard: abandon a problem whose sized build is too
                # slow (pure-Python O(n²)) before it stalls an unattended sweep.
                if args.max_build_seconds > 0 and not _build_within(
                    root, bare, size, args.max_build_seconds
                ):
                    skipped_slow_build += 1
                    continue
                # One guarded build to gate applicability/size before the configs:
                # objective-free problems are not minimization problems, and the
                # size cap keeps the full sweep tractable. Genuine build errors fall
                # through to run_case so their traceback is recorded as a row.
                try:
                    problem, _x0 = gate_case.build(xp)
                except NotImplementedError:
                    skipped_no_objective += 1
                    continue
                except Exception:
                    results.append(
                        run_case(
                            gate_case,
                            config=configs[0][0],
                            options=configs[0][1],
                            xp=xp,
                            backend=backend,
                        )
                    )
                    done.add((backend, bare))
                    _flush()
                    continue
                # Run each config only when the problem fits that route's variable
                # cap (tightened by --max-vars). A problem too large for every route
                # is counted oversized and contributes no rows.
                n_vars = int(problem.n_vars)
                ran_any = False
                for label, options, route_cap in configs:
                    cap = _effective_cap(route_cap, args.max_vars)
                    if cap and n_vars > cap:
                        continue
                    case = cases_by_mode[_problem_mode(options)][case_name]
                    results.append(
                        run_case(
                            case, config=label, options=options, xp=xp, backend=backend
                        )
                    )
                    ran_any = True
                if not ran_any:
                    skipped_too_large += 1
                done.add((backend, bare))
                _flush()
            finally:
                inflight_path.unlink(missing_ok=True)

    _flush()
    by_status: Counter[str] = Counter(r.status for r in results)
    n_correct = sum(1 for r in results if r.correct)
    n_converged = sum(1 for r in results if r.converged)
    print(
        f"S2MPJ sweep: {n_correct}/{len(results)} correct, "
        f"{n_converged}/{len(results)} converged (KKT) "
        f"(skipped {skipped_no_objective} objective-free, {skipped_too_large} oversized,"
        f" {skipped_slow_build} slow-build)"
        f" -> {json_path}, {md_path}"
    )
    for status, count in sorted(by_status.items()):
        print(f"  {status:16s} {count}")
    # Per-config coverage: each route ran a different problem count (its cap).
    per_config_total: Counter[str] = Counter(r.config for r in results)
    per_config_correct: Counter[str] = Counter(r.config for r in results if r.correct)
    per_config_converged: Counter[str] = Counter(
        r.config for r in results if r.converged
    )
    for label, _options, _cap in configs:
        total = per_config_total.get(label, 0)
        print(
            f"  {label:16s} {per_config_correct.get(label, 0)}/{total} correct, "
            f"{per_config_converged.get(label, 0)}/{total} converged"
        )
    return 0 if n_correct == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
