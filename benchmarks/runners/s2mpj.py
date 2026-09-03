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

``--jobs N`` runs N problems concurrently in worker processes (problems are
independent; each worker runs one problem's whole config matrix so the shared
instance cache still amortizes construction). Reports stay deterministic — rows
are sorted before every flush — and the flush-per-problem crash survival is
kept. Two caveats: per-solve wall times (and therefore ``--max-time`` hits) can
inflate under CPU oversubscription, so pin BLAS threads (e.g.
``OMP_NUM_THREADS=1``) or keep N modest for timing-comparable sweeps; and a
native crash in a worker breaks the whole pool — each worker records what it is
running in a pid-suffixed ``.inflight`` marker, so at most N candidates are named
and ``--resume --exclude <name>`` steps past the culprit.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import ipax
from benchmarks.corpus.s2mpj import (
    _DEFAULT_NAMES,
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
from ipax.options import (
    BarrierOptions,
    KrylovOptions,
    LBFGSOptions,
    LineSearchOptions,
    RegularizationOptions,
    RestorationOptions,
)
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

# Sentinel distinguishing "keep the solver default" from an explicit value for
# option-override levers whose overrides can legitimately be None.
_KEEP_DEFAULT: float | None = float("nan")


def default_configs(
    max_iter: int,
    max_time: float | None,
    scaling: str = "gradient-based",
    *,
    dense_max_vars: int = _DENSE_MAX_VARS,
    krylov_max_vars: int = _KRYLOV_MAX_VARS,
    sparse_max_vars: int = _SPARSE_MAX_VARS,
    krylov_preconditioner: str | None = None,
    mu_schedule: str | None = None,
    feasible_kkt_progress: float | None = _KEEP_DEFAULT,
    free_mode_acceptance: str | None = None,
    slack_init_scale: float | None = None,
    equality_dual_repair: float | None = None,
    powell_damping: bool | None = None,
    damping_skip_ratio: float | None = None,
    lbfgs_seed: str | None = None,
    backtrack_interpolation: bool | None = None,
    exact_lbfgs_inverse: bool | None = None,
    restoration_linear_solver: str | None = None,
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
    if restoration_linear_solver is not None:
        common["restoration"] = RestorationOptions(
            linear_solver=restoration_linear_solver  # type: ignore[arg-type]
        )
    if mu_schedule is not None:
        # μ-oracle A/B lever (e.g. probing-default vs monotone); None keeps
        # the solver default so ordinary sweeps track it automatically.
        common["mu_schedule"] = mu_schedule
    # Line-search override levers; unset levers keep the solver defaults so
    # ordinary sweeps track them automatically.
    ls_overrides: dict[str, Any] = {}
    if feasible_kkt_progress is not _KEEP_DEFAULT:
        # Tier-3 rescue A/B lever (None disables the feasible-point
        # KKT-progress acceptance).
        ls_overrides["feasible_kkt_progress"] = feasible_kkt_progress
    if free_mode_acceptance is not None:
        # NWW §5 free-mode acceptance A/B lever ("rigorous" keeps the W&B gate
        # in both regimes); only observable under a non-monotone --mu-schedule.
        ls_overrides["free_mode_acceptance"] = free_mode_acceptance
    if backtrack_interpolation is not None:
        # Interpolated-backtrack A/B lever (2026-08 RT perf study): the
        # solver default (False) is W&B's plain halving; True selects
        # safeguarded quadratic interpolation of the rejected trial's merit
        # (N&W eq. 3.58) instead — opt-in per a non-unanimous full-corpus
        # sweep (see docs/benchmarks/s2mpj.md). Affects every config (the
        # filter line search runs on both Hessian routes), unlike the
        # L-BFGS-only levers above.
        ls_overrides["backtrack_interpolation"] = backtrack_interpolation
    if ls_overrides:
        common["line_search"] = LineSearchOptions(**ls_overrides)
    if slack_init_scale is not None:
        # Scale-aware slack-init A/B lever (BarrierOptions.slack_init_scale;
        # 0.0 = the flat-floor default). None keeps the solver default so
        # ordinary sweeps track it automatically.
        common["barrier"] = BarrierOptions(slack_init_scale=slack_init_scale)
    if equality_dual_repair is not None:
        # Divergence-gated equality-multiplier repair A/B lever
        # (RegularizationOptions.equality_dual_repair; None = the solver default,
        # which is off). Only observable on equality-constrained problems.
        common["regularization"] = RegularizationOptions(
            equality_dual_repair=equality_dual_repair
        )
    if (
        powell_damping is not None
        or damping_skip_ratio is not None
        or lbfgs_seed is not None
    ):
        # L-BFGS A/B levers (LBFGSOptions; the solver defaults damp every
        # non-PD pair with the "direct" ξ seed): `--powell-damping off` skips
        # all non-PD pairs like IPOPT's limited-memory update,
        # `--damping-skip-ratio C` skips only pairs with s·y < −C·s·Bs, and
        # `--lbfgs-seed scalar1` switches the ξ estimate to IPOPT's
        # δᵀγ/δᵀδ. Only observable on the L-BFGS configs.
        common["lbfgs"] = LBFGSOptions(
            powell_damping=True if powell_damping is None else powell_damping,
            damping_skip_ratio=damping_skip_ratio,
            seed_formula="direct" if lbfgs_seed is None else lbfgs_seed,  # type: ignore[arg-type]
        )
    krylov_common = dict(common)
    if krylov_preconditioner is not None or exact_lbfgs_inverse is not None:
        # exact_lbfgs_inverse A/B lever (2026-08 RT perf study): the solver
        # default (True) applies the condensed Woodbury inverse outright on
        # bound-only L-BFGS systems instead of the plain Jacobi diagonal.
        # Only observable on the two Krylov configs.
        krylov_common["krylov"] = KrylovOptions(
            preconditioner=(  # type: ignore[arg-type]
                "jacobi" if krylov_preconditioner is None else krylov_preconditioner
            ),
            exact_lbfgs_inverse=(
                True if exact_lbfgs_inverse is None else exact_lbfgs_inverse
            ),
        )
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


def _select_names(args: argparse.Namespace, root: str) -> tuple[str, ...]:
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
    return _DEFAULT_NAMES  # the curated default set


def _run_problem_cases(
    root: str,
    bare: str,
    backend: str,
    size: int | None,
    feasibility: bool,
    configs: list[ConfigSpec],
    global_max_vars: int,
    max_build_seconds: float,
    inflight_prefix: str | None = None,
) -> tuple[list[CaseResult], str | None]:
    """Run one problem across the config matrix; ``(rows, skip_reason)``.

    Top-level (picklable) so ``--jobs`` worker processes can run it; the serial
    path calls it inline. One problem stays wholly inside one call so the
    lru-cached S2MPJ instance (and its verified fast evaluator) is shared across
    the per-config rebuilds. ``skip_reason`` is ``"no_objective"``,
    ``"too_large"``, ``"slow_build"``, or ``None`` when rows were produced (a
    genuine build error is recorded as a ``build_error`` row, not a skip).

    ``inflight_prefix`` makes the worker record what it is running in its own
    pid-suffixed marker file, cleared on the way out. A native crash takes the
    worker down with no chance to report, and the parent cannot identify the
    culprit from its pending futures: with every problem submitted up front,
    that list is the whole queue. At most ``--jobs`` markers survive a crash.
    """
    marker = None
    if inflight_prefix is not None:
        marker = Path(f"{inflight_prefix}.{os.getpid()}")
        marker.write_text(f"{backend} {bare}", encoding="utf-8")
    try:
        return _run_problem_cases_inner(
            root,
            bare,
            backend,
            size,
            feasibility,
            configs,
            global_max_vars,
            max_build_seconds,
        )
    finally:
        if marker is not None:
            marker.unlink(missing_ok=True)


def _run_problem_cases_inner(
    root: str,
    bare: str,
    backend: str,
    size: int | None,
    feasibility: bool,
    configs: list[ConfigSpec],
    global_max_vars: int,
    max_build_seconds: float,
) -> tuple[list[CaseResult], str | None]:
    """The sweep proper; see :func:`_run_problem_cases`."""
    xp = import_namespace(backend)

    # Optional build guard: abandon a problem whose sized build is too slow
    # (pure-Python O(n²)) before it stalls an unattended sweep.
    if max_build_seconds > 0 and not _build_within(root, bare, size, max_build_seconds):
        return [], "slow_build"

    gate_mode = ("lbfgs", False)
    modes = {_problem_mode(options) for _, options, _ in configs} | {gate_mode}
    cases_by_mode = {
        mode: s2mpj_problems(
            (bare,),
            directory=root,
            backends=(backend,),
            hessian=mode[0],
            sparse=mode[1],
            size=size,
            feasibility=feasibility,
        )[0]
        for mode in modes
    }
    gate_case = cases_by_mode[gate_mode]

    # One guarded build to gate applicability/size before the configs:
    # objective-free problems are not minimization problems, and the size cap
    # keeps the full sweep tractable. Genuine build errors fall through to
    # run_case so their traceback is recorded as a row.
    try:
        problem, _x0 = gate_case.build(xp)
    except NotImplementedError:
        return [], "no_objective"
    except Exception:
        row = run_case(
            gate_case,
            config=configs[0][0],
            options=configs[0][1],
            xp=xp,
            backend=backend,
        )
        return [row], None

    # Run each config only when the problem fits that route's variable cap
    # (tightened by --max-vars). A problem too large for every route is counted
    # oversized and contributes no rows.
    n_vars = int(problem.n_vars)
    rows: list[CaseResult] = []
    for label, options, route_cap in configs:
        cap = _effective_cap(route_cap, global_max_vars)
        if cap and n_vars > cap:
            continue
        rows.append(
            run_case(
                cases_by_mode[_problem_mode(options)],
                config=label,
                options=options,
                xp=xp,
                backend=backend,
            )
        )
    return rows, None if rows else "too_large"


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
        "--mu-schedule",
        default=None,
        choices=["monotone", "adaptive", "breedveld", "probing", "quality"],
        help="override the barrier μ oracle on every config (default: the solver "
        "default) — the lever for a schedule A/B such as probing vs monotone.",
    )
    parser.add_argument(
        "--free-mode-acceptance",
        default=None,
        choices=["obj-constr-filter", "rigorous"],
        help="override LineSearchOptions.free_mode_acceptance on every config "
        "(default: the solver default) — the lever for the NWW §5 free-mode "
        "acceptance A/B; only observable under a non-monotone --mu-schedule.",
    )
    parser.add_argument(
        "--feasible-kkt-progress",
        default=None,
        help="override LineSearchOptions.feasible_kkt_progress on every config: "
        "'none' disables the feasible-point KKT-progress rescue, a float sets "
        "the required decrease fraction (default: the solver default) — the "
        "lever for a Tier-3 rescue A/B.",
    )
    parser.add_argument(
        "--slack-init-scale",
        type=float,
        default=None,
        help="override BarrierOptions.slack_init_scale on every config (default: "
        "the solver default 0.0 = flat slack floor) — the lever for the "
        "scale-aware slack-init A/B; e.g. 0.1 scales the floor to "
        "0.1·max|g(x0)|.",
    )
    parser.add_argument(
        "--equality-dual-repair",
        type=float,
        default=None,
        help="override RegularizationOptions.equality_dual_repair on every config "
        "(default: the solver default None = off) — the lever for the "
        "divergence-gated equality-multiplier repair A/B; e.g. 1e10 repairs "
        "multipliers whose stationarity residual exceeds 1e10x the "
        "least-squares estimate's.",
    )
    parser.add_argument(
        "--powell-damping",
        choices=("on", "off"),
        default=None,
        help="override LBFGSOptions.powell_damping on every config (default: the "
        "solver default, which damps non-PD curvature pairs) — the lever for "
        "the damp-vs-skip A/B ('off' skips non-PD pairs like IPOPT's "
        "limited-memory update); only observable on the L-BFGS configs.",
    )
    parser.add_argument(
        "--damping-skip-ratio",
        type=float,
        default=None,
        help="override LBFGSOptions.damping_skip_ratio on every config (default: "
        "the solver default None = damp every non-PD pair) — the hybrid lever: "
        "a curvature pair with s·y < −RATIO·s·Bs is skipped instead of "
        "Powell-damped; only observable on the L-BFGS configs.",
    )
    parser.add_argument(
        "--lbfgs-seed",
        choices=("direct", "scalar1"),
        default=None,
        help="override LBFGSOptions.seed_formula on every config (default: the "
        "solver default 'direct' = γᵀγ/δᵀγ) — the lever for the ξ-seed A/B; "
        "'scalar1' is IPOPT's δᵀγ/δᵀδ; only observable on the L-BFGS configs.",
    )
    parser.add_argument(
        "--backtrack-interpolation",
        choices=("on", "off"),
        default=None,
        help="override LineSearchOptions.backtrack_interpolation on every config "
        "(default: the solver default 'off' = W&B's plain halving) — the lever "
        "for the interpolation-vs-halving A/B; 'on' selects the safeguarded "
        "N&W eq. 3.58 quadratic-interpolation backtrack instead. Affects every "
        "config.",
    )
    parser.add_argument(
        "--exact-lbfgs-inverse",
        choices=("on", "off"),
        default=None,
        help="override KrylovOptions.exact_lbfgs_inverse on every config "
        "(default: the solver default 'on' = apply the condensed Woodbury "
        "inverse outright on bound-only L-BFGS systems) — the lever for the "
        "exact-inverse-vs-Jacobi A/B; only observable on the Krylov configs.",
    )
    parser.add_argument(
        "--restoration-linear-solver",
        choices=("dense", "krylov"),
        default=None,
        help="override the feasibility-restoration linear solver on every config "
        "(default: keep the solver default 'dense'); use 'krylov' for the "
        "paired matrix-free restoration sweep.",
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
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="run this many problems concurrently in worker processes (default 1 = "
        "serial). Problems are independent, so wall time divides ~linearly; pin "
        "BLAS threads (e.g. OMP_NUM_THREADS=1) or keep it modest when per-solve "
        "timings must stay comparable to a serial run",
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
    names = tuple(dict.fromkeys(_select_names(args, root)))  # dedupe, keep order
    size = args.size or None
    configs = default_configs(
        args.max_iter,
        args.max_time,
        args.scaling,
        dense_max_vars=args.dense_max_vars,
        krylov_max_vars=args.krylov_max_vars,
        sparse_max_vars=args.sparse_max_vars,
        krylov_preconditioner=args.preconditioner,
        mu_schedule=args.mu_schedule,
        feasible_kkt_progress=(
            _KEEP_DEFAULT
            if args.feasible_kkt_progress is None
            else (
                None
                if args.feasible_kkt_progress.lower() == "none"
                else float(args.feasible_kkt_progress)
            )
        ),
        free_mode_acceptance=args.free_mode_acceptance,
        slack_init_scale=args.slack_init_scale,
        equality_dual_repair=args.equality_dual_repair,
        powell_damping=(
            None if args.powell_damping is None else args.powell_damping == "on"
        ),
        damping_skip_ratio=args.damping_skip_ratio,
        lbfgs_seed=args.lbfgs_seed,
        backtrack_interpolation=(
            None
            if args.backtrack_interpolation is None
            else args.backtrack_interpolation == "on"
        ),
        exact_lbfgs_inverse=(
            None
            if args.exact_lbfgs_inverse is None
            else args.exact_lbfgs_inverse == "on"
        ),
        restoration_linear_solver=args.restoration_linear_solver,
    )
    if args.config:
        wanted = {c.strip() for c in args.config.split(",") if c.strip()}
        configs = [spec for spec in configs if spec[0] in wanted]
        if not configs:
            parser.error(f"--config {args.config!r} matched no known config labels")
    environment = capture_environment()

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
        # the report must survive it rather than losing the whole run. Rows are
        # sorted so the reports are deterministic regardless of --jobs completion
        # order (and diff cleanly across runs).
        rows = sorted(results, key=lambda r: (r.backend, r.problem, r.config))
        json_path.write_text(json.dumps(to_payload(rows, environment), indent=2))
        md_path.write_text(format_markdown(rows, environment), encoding="utf-8")

    skipped: Counter[str] = Counter()
    work: list[tuple[str, str]] = []
    for backend in backends:
        try:
            import_namespace(backend)
        except ImportError:
            continue
        for bare in names:
            if (backend, bare) in done or bare in exclude:
                continue
            work.append((backend, bare))

    def _worker_args(backend: str, bare: str) -> tuple[object, ...]:
        return (
            root,
            bare,
            backend,
            size,
            args.include_objective_free,
            configs,
            args.max_vars,
            args.max_build_seconds,
            str(inflight_path),
        )

    def _record(
        backend: str, bare: str, rows: list[CaseResult], skip: str | None
    ) -> None:
        if skip is not None:
            skipped[skip] += 1
        results.extend(rows)
        done.add((backend, bare))
        _flush()

    if args.jobs <= 1:
        for backend, bare in work:
            # Record the in-flight problem so a wrapper can identify (and then
            # --exclude) a problem that natively crashes the process. The file is
            # cleared in ``finally`` once the problem is fully handled, so it only
            # survives a hard crash — naming exactly the culprit.
            inflight_path.write_text(f"{backend} {bare}")
            try:
                rows, skip = _run_problem_cases(*_worker_args(backend, bare))
                _record(backend, bare, rows, skip)
            finally:
                inflight_path.unlink(missing_ok=True)
    else:
        # Problems are independent, so fan them out over worker processes. Each
        # worker runs one problem's whole config matrix (sharing the lru-cached
        # instance + verified fast evaluator), and the parent flushes after every
        # completion, preserving the crash-surviving report. Each worker also
        # marks what it is running, so a native crash that breaks the pool leaves
        # at most --jobs named candidates for --exclude.
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
            futures = {
                pool.submit(_run_problem_cases, *_worker_args(backend, bare)): (
                    backend,
                    bare,
                )
                for backend, bare in work
            }

            try:
                for future in as_completed(dict(futures)):
                    backend, bare = futures[future]
                    rows, skip = future.result()
                    del futures[future]
                    _record(backend, bare, rows, skip)
            except BrokenProcessPool:
                # The workers' own markers, not the pending futures: with every
                # problem submitted up front the latter names the whole queue.
                markers = sorted(inflight_path.parent.glob(inflight_path.name + ".*"))
                candidates = sorted(
                    m.read_text(encoding="utf-8").replace(" ", "/") for m in markers
                )
                _flush()
                named = ", ".join(candidates) or "(no marker survived)"
                print(
                    "S2MPJ sweep: a worker process died (native crash?); the "
                    f"culprit is one of the {len(candidates)} problem(s) that "
                    f"were running: {named} — then --resume --exclude it"
                )
                # Exit 2, not 1: a wrapper must be able to tell "pool broke,
                # resume after identifying the crasher" apart from the normal
                # "completed with some incorrect cases" exit 1.
                return 2

    _flush()
    by_status: Counter[str] = Counter(r.status for r in results)
    n_correct = sum(1 for r in results if r.correct)
    n_converged = sum(1 for r in results if r.converged)
    print(
        f"S2MPJ sweep: {n_correct}/{len(results)} correct, "
        f"{n_converged}/{len(results)} converged (KKT) "
        f"(skipped {skipped['no_objective']} objective-free, "
        f"{skipped['too_large']} oversized, {skipped['slow_build']} slow-build)"
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
