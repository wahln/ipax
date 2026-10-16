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
   only the ``n×n`` system — are the appropriate choice; the sparse-direct route
   (``--config lbfgs/sparse,exact/sparse``) factors the full ``(n+m)`` saddle and
   is impractical at these row counts. The loader's per-structure tiled Gram and
   grouped exact Hessian carry the dose-matrix structure through the public
   ``Problem`` / ``LinearOperator`` protocols (see ``AGENTS.md``); large cases
   may still hit ``--max-time`` before converging.

   By default a solve reproduces **Breedveld et al. 2017's solver setup** except
   for the μ update: the exact Hessian on the condensed dense route
   (``--config exact/dense``), Mehrotra predictor–corrector
   (``--corrections mehrotra``), the ratio-control step globalization
   (``--globalization breedveld``), a ``1e-8`` KKT tolerance (``--tol``), and
   the paper's §3.1 initialisation (``--init breedveld``): the least-squares
   fluence rescaled towards its mean and floored at 5 (eqs. (23)–(24)), LSQR
   multipliers with floored bound multipliers and slacks (eqs. (26)–(28)), and
   ``μ0`` their average complementarity (eq. (10)); ``--lsqr-iter``/
   ``--lsqr-tol`` override the eq. (26) LSQR's iteration cap/tolerance
   (``--init breedveld`` only), opt-in because a lower cap is only known to
   reproduce the default bit-for-bit on the measured corpus, not in general
   (docs/benchmarks/trots.md's "LSQR dual start"). The initialisation's time
   — the starting point included, under either ``--init`` — counts toward the
   reported solve time. μ comes from the Mehrotra probing oracle
   (``--mu-schedule probing``, the closest transcription of Breedveld's
   Mehrotra μ) with the El-Bakry centrality floor off
   (``--kappa-centrality 0``): the 2026-09-27 sweep over all seven TROTS sites
   (docs/benchmarks/trots.md) found the floor, not the oracle, behind the
   barrier-dominated excursions on the nonconvex cases, and probing without
   it beats the monotone reduction on every measured case (Protons_01 93 vs
   214 iterations, Liver_01 57 vs 74, Prostate_CK_01 39 vs 87, VMAT_101 404
   vs 635, the BT pair certifying the −0.78/−0.75 optima where monotone stops
   at 0). ``--mu-schedule monotone`` is the previous runner default. The
   pre-2026-09-26 default stays one flag set away: ``--config
   lbfgs/dense,exact/dense --init ipax --mu-schedule monotone --globalization
   filter --corrections none --kappa-centrality 1e-2`` (the RT recipes of
   ``docs/benchmarks/trots.md`` add levers on top of it).

   Each solve row also carries the paper's own result for the case, the
   Mehrotra-method iterations and seconds of Breedveld et al. 2017's timing
   supplement (``ref_iters`` / ``ref_time_s`` in the JSON report, "paper" on
   the console; :mod:`benchmarks.corpus.trots_reference`, CC BY 4.0). Those
   seconds were measured on the paper's 2017 16-core machine.

Run from the repository root::

    IPAX_TROTS_DIR=/path/to/TROTS python -m benchmarks.runners.trots
    IPAX_TROTS_DIR=/path/to/TROTS python -m benchmarks.runners.trots --solve \
        --cases Protons_01
    IPAX_TROTS_DIR=/path/to/TROTS python -m benchmarks.runners.trots --solve \
        --cases Prostate_BT_01 --config lbfgs/sparse,exact/sparse

Exits non-zero if any verification case's objective does not match its reference
(within the significant figures that reference reports).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import ipax
from benchmarks.corpus.trots import (
    _BREEDVELD_START_FLOOR,
    TROTSExactProblem,
    TROTSProblem,
    list_trots_cases,
    load_trots_file,
    objective_at,
    reference_for,
    trots_dir,
)
from benchmarks.corpus.trots_reference import breedveld2017_reference
from benchmarks.harness import capture_environment, run_case
from ipax.options import BarrierOptions, BreedveldOptions, OptimalityConditionOptions
from ipax.testing.backends import import_namespace

if TYPE_CHECKING:
    from benchmarks.harness import CaseResult
    from ipax.problem.base import Problem
    from ipax.result import WarmStart
    from ipax.typing import Array

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
# Breedveld et al. 2017's setup: exact Hessian on the condensed dense route.
_DEFAULT_SOLVE_CONFIGS = ("exact/dense",)
_INIT_STRATEGIES = ("breedveld", "ipax")


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


def _solve_case(
    root: str,
    case: str,
    label: str,
    opts: ipax.Options,
    xp: object,
    *,
    backend: str = "numpy",
    plain_gram: bool = False,
    per_term_hessian: bool = False,
    init: str = "breedveld",
    lsqr_iter: int | None = None,
    lsqr_tol: float | None = None,
    dual_floor: float = _BREEDVELD_START_FLOOR,
):
    """Score one solve via the shared harness (wrapping the case as a corpus problem).

    ``dual_floor`` is the Breedveld start's multiplier floor
    (:meth:`TROTSProblem.breedveld_warm_start`; default the paper's 5).

    ``plain_gram`` builds the problem with ``tiled_gram=False`` (the core's
    plain Gram for the linear block, dense row chunks for the exact Hessian);
    ``per_term_hessian`` keeps the exact Hessian's original per-term SpGEMM
    path (``grouped_hessian=False``). Both are A/B levers; unset, the loader
    defaults apply. ``lsqr_iter``/``lsqr_tol`` override the Breedveld dual
    start's LSQR iteration cap / tolerance (``init="breedveld"`` only; see
    :meth:`TROTSProblem.breedveld_warm_start`).

    The starting point is produced by the case's ``prepare_solve`` hook, which
    the harness times as initialisation and counts in the solve time (the
    paper's accounting); ``build`` only loads the case and returns the
    loader's uniform placeholder, so the least-squares fluence solve of
    eq. (23) never runs outside the timers. ``init="breedveld"`` starts from
    Breedveld et al. 2017 §3.1: the primal point of
    :meth:`TROTSProblem.initial_point` with ``strategy="breedveld"``
    (eqs. (23)–(24)) and the dual start of
    :meth:`TROTSProblem.breedveld_warm_start` (eqs. (26)–(28)) with
    ``BarrierOptions.mu_init`` set to its ``μ0`` (eq. (10)). ``init="ipax"``
    is the loader's default start (the least-squares fluence with a tiny
    positive floor), with no warm start.
    """
    from benchmarks.corpus import BenchmarkProblem

    if init not in _INIT_STRATEGIES:
        raise ValueError(f"unknown init {init!r}; expected one of {_INIT_STRATEGIES}")
    hessian = str(opts.hessian)

    def build(namespace):
        instance = load_trots_file(f"{root}/{case}.mat")
        kwargs: dict[str, object] = {"sparse": True}
        if plain_gram:
            kwargs["tiled_gram"] = False
        problem: TROTSProblem
        if hessian == "exact":
            problem = TROTSExactProblem(
                instance, namespace, grouped_hessian=not per_term_hessian, **kwargs
            )
        else:
            problem = TROTSProblem(instance, namespace, **kwargs)
        ref = reference_for(case, root)
        problem.expected_objective = None if ref is None else ref.objective
        # A cheap placeholder: the start itself (the least-squares fluence of
        # eq. (23) and what follows) is initialisation and belongs to the
        # timed ``prepare_solve`` hook below, not to the untimed build.
        return problem, problem.initial_point(warm_start=False)

    def prepare_solve(
        problem: Problem, x0: Array, options: ipax.Options
    ) -> tuple[Array, WarmStart | None, ipax.Options]:
        del x0  # the placeholder from ``build``
        if not isinstance(problem, TROTSProblem):
            raise TypeError(f"expected a TROTSProblem, got {type(problem).__name__}")
        if init == "ipax":
            return problem.initial_point(), None, options
        # Breedveld et al. 2017 §3.1: primal start (eqs. (23)–(24)), dual start
        # (eqs. (26)–(28)) and μ0 (eq. (10)).
        start = problem.initial_point(strategy="breedveld")
        # μ0 in solver space: the solve's scaling multiplies it by s_f.
        warm, mu0 = problem.breedveld_warm_start(
            start,
            scaling=options.scaling,
            lsqr_iter=lsqr_iter,
            lsqr_tol=lsqr_tol,
            dual_floor=dual_floor,
        )
        barrier = dataclasses.replace(options.barrier, mu_init=mu0)
        return start, warm, dataclasses.replace(options, barrier=barrier)

    bench = BenchmarkProblem(
        name=f"trots/{case}",
        kind="RT",
        tags=("trots",),
        build=build,
        prepare_solve=prepare_solve,
    )
    return run_case(bench, config=label, options=opts, xp=xp, backend=backend)


def _solve_options(args: argparse.Namespace, label: str) -> ipax.Options:
    """The solve :class:`ipax.Options` for one ``--config`` label and the levers.

    Every lever is passed explicitly (its default is Breedveld et al. 2017's
    setup, with the monotone μ update); passing a value equal to the ipax default is a no-op, so e.g. the
    previous default's flags rebuild its options exactly. ``--slack-init-scale``
    is only accepted with ``--init ipax`` (see :func:`_parse_args`): the
    Breedveld start supplies its own slacks (eq. (28)).
    """
    barrier_kwargs: dict[str, float] = {}
    if args.slack_init_scale is not None:
        barrier_kwargs["slack_init_scale"] = args.slack_init_scale
    if args.kappa_centrality is not None:
        barrier_kwargs["kappa_centrality"] = args.kappa_centrality
    return ipax.Options(
        max_iter=args.max_iter,
        max_time=args.max_time,
        **_CONFIGS[label],
        mu_schedule=args.mu_schedule,
        globalization=args.globalization,
        corrections=args.corrections,
        barrier=BarrierOptions(**barrier_kwargs),
        breedveld=BreedveldOptions(common_step=args.common_step),
        optimality=OptimalityConditionOptions(
            dual_inf_tol=args.tol,
            constr_viol_tol=args.tol,
            compl_inf_tol=args.tol,
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    """The runner's CLI; solve defaults are Breedveld et al. 2017's setup
    with the monotone μ update (see the module docstring)."""
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
        help="comma-separated solve configs (default: exact/dense, Breedveld "
        "et al. 2017's exact Hessian): " + ", ".join(_CONFIGS),
    )
    parser.add_argument(
        "--backend",
        default="numpy",
        help="array namespace for the solves (numpy, cupy, torch); the loader"
        " mirrors the case to the device and the report row records it",
    )
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-time", type=float, default=120.0)
    # Solver levers (docs/benchmarks/trots.md). The defaults reproduce Breedveld
    # et al. 2017's setup; each is passed explicitly, so a value equal to the
    # ipax default is a no-op.
    parser.add_argument(
        "--mu-schedule",
        default="probing",
        help='Options.mu_schedule (default "probing": Mehrotra\'s predictor-based '
        "mu as in Breedveld et al. 2017, paired with --kappa-centrality 0; "
        '"monotone" is the Fiacco-McCormick reduction and the previous runner '
        'default; the old proton recipe used "quality")',
    )
    parser.add_argument(
        "--globalization",
        default="breedveld",
        help='Options.globalization (default "breedveld": the paper\'s ratio '
        'control; "filter" is the IPOPT filter line search)',
    )
    parser.add_argument(
        "--corrections",
        default="mehrotra",
        help='Options.corrections (default "mehrotra": predictor-corrector; '
        '"none", "gondzio")',
    )
    parser.add_argument(
        "--init",
        choices=_INIT_STRATEGIES,
        default="breedveld",
        help="starting point (default breedveld: Breedveld et al. 2017 §3.1 — "
        "rescaled least-squares fluence floored at 5, LSQR multipliers, floored "
        "bound multipliers/slacks and mu_init = their average complementarity; "
        "ipax: the loader's least-squares fluence with the solver's own dual "
        "start; either start is timed as part of the solve)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="KKT tolerance: OptimalityConditionOptions dual_inf_tol, "
        "constr_viol_tol and compl_inf_tol (default 1e-8)",
    )
    parser.add_argument(
        "--slack-init-scale",
        type=float,
        default=None,
        help="BarrierOptions.slack_init_scale (RT recipe: 0.1); only with "
        "--init ipax — an error with --init breedveld, whose slacks come from "
        "eq. (28)",
    )
    parser.add_argument(
        "--kappa-centrality",
        type=float,
        default=0.0,
        help="BarrierOptions.kappa_centrality (default 0: the El-Bakry floor off, "
        "which is what makes probing mu converge the nonconvex TROTS cases; "
        "ipax's own default is 1e-2; the old proton stack used 1e-4)",
    )
    parser.add_argument(
        "--lsqr-iter",
        type=int,
        default=None,
        help="LSQR iteration cap for the Breedveld dual start's eq. (26) solve "
        "(default 200 via TROTSProblem.breedveld_warm_start); only with "
        "--init breedveld -- an error with --init ipax, which does not run it",
    )
    parser.add_argument(
        "--lsqr-tol",
        type=float,
        default=None,
        help="LSQR atol = btol for the Breedveld dual start's eq. (26) solve "
        "(default 1e-6 via TROTSProblem.breedveld_warm_start); only with "
        "--init breedveld -- an error with --init ipax, which does not run it",
    )
    parser.add_argument(
        "--plain-gram",
        action="store_true",
        help="disable the loader's tiled per-structure Gram (tiled_gram=False)",
    )
    parser.add_argument(
        "--per-term-hessian",
        action="store_true",
        help="exact configs: the per-term SpGEMM Hessian (grouped_hessian=False)",
    )
    # Head-and-Neck_01 diagnosis levers (2026-09-28, docs/benchmarks/trots.md).
    parser.add_argument(
        "--common-step",
        action="store_true",
        help="BreedveldOptions(common_step=True): one step length "
        "min(alpha_primal, alpha_dual) for the primal and dual updates, as "
        "Breedveld et al. 2017 §2.1 (only with --globalization breedveld)",
    )
    parser.add_argument(
        "--dual-floor",
        type=float,
        default=_BREEDVELD_START_FLOOR,
        help="floor of the Breedveld start's multipliers y0, z0 (eqs. (26)-(27); "
        "default 5 as in the paper; the slacks keep 5); only with --init "
        "breedveld",
    )
    return parser


def _parse_args(
    parser: argparse.ArgumentParser, argv: list[str] | None
) -> argparse.Namespace:
    """Parse ``argv`` and reject flag combinations that would be ignored."""
    args = parser.parse_args(argv)
    if args.slack_init_scale is not None and args.init == "breedveld":
        parser.error(
            "--slack-init-scale applies only with --init ipax; the Breedveld "
            "start (--init breedveld) sets the slacks itself (eq. (28))"
        )
    if args.lsqr_iter is not None and args.init != "breedveld":
        parser.error(
            "--lsqr-iter applies only with --init breedveld; --init ipax does "
            "not run the Breedveld dual start's eq. (26) LSQR solve"
        )
    if args.lsqr_tol is not None and args.init != "breedveld":
        parser.error(
            "--lsqr-tol applies only with --init breedveld; --init ipax does "
            "not run the Breedveld dual start's eq. (26) LSQR solve"
        )
    if args.lsqr_tol is not None and not (
        math.isfinite(args.lsqr_tol) and args.lsqr_tol > 0.0
    ):
        parser.error("--lsqr-tol must be positive and finite")
    if args.dual_floor != _BREEDVELD_START_FLOOR and args.init != "breedveld":
        parser.error("--dual-floor applies only with --init breedveld")
    if not (math.isfinite(args.dual_floor) and args.dual_floor > 0.0):
        parser.error("--dual-floor must be positive and finite")
    if args.common_step and args.globalization != "breedveld":
        parser.error(
            "--common-step applies only with --globalization breedveld (the "
            "filter line search keeps separate primal and dual step lengths)"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = _parse_args(parser, argv)

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
        xp = import_namespace(args.backend)
        labels = [c.strip() for c in args.config.split(",") if c.strip()]
        unknown = [label for label in labels if label not in _CONFIGS]
        if unknown:
            parser.error(f"unknown --config {unknown}; choose from {list(_CONFIGS)}")
        for case in cases:
            for label in labels:
                opts = _solve_options(args, label)
                result = _solve_case(
                    root,
                    case,
                    label,
                    opts,
                    xp,
                    backend=args.backend,
                    plain_gram=args.plain_gram,
                    per_term_hessian=args.per_term_hessian,
                    init=args.init,
                    lsqr_iter=args.lsqr_iter,
                    lsqr_tol=args.lsqr_tol,
                    dual_floor=args.dual_floor,
                )
                row = _solve_row(result)
                solve_rows.append(row)
                print(_solve_line(case, label, row))

    payload = {
        "environment": environment,
        "verification": [asdict(r) for r in verification],
        "solves": solve_rows,
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


def _solve_row(result: CaseResult) -> dict[str, object]:
    """One solve's JSON row: the harness result plus the paper's reference.

    ``ref_iters`` / ``ref_time_s`` are Breedveld et al. 2017's Mehrotra-method
    iterations and wall-clock seconds for the case (the run behind the
    dataset's ``Results/*.txt``; supplement ESM 2, see
    :mod:`benchmarks.corpus.trots_reference`), ``None`` for cases the paper
    does not solve (``Prostate_BT_*``). The paper's times are on its 2017
    16-core machine: context, not a like-for-like comparison.
    """
    ref = breedveld2017_reference(result.problem)
    return {
        **asdict(result),
        "ref_iters": None if ref is None else ref.mehrotra.iterations,
        "ref_time_s": None if ref is None else ref.mehrotra.seconds,
    }


def _solve_line(case: str, label: str, row: dict[str, object]) -> str:
    """The console line for one solve row (see :func:`_solve_row`)."""
    ref_iters, ref_time = row["ref_iters"], row["ref_time_s"]
    paper = (
        "paper n/a"
        if ref_iters is None or ref_time is None
        else f"paper iters={ref_iters} t={ref_time:.1f}s"
    )
    return (
        f"  {case:20s} [{label:12s}] {row['status']!s:12s} "
        f"iters={row['n_iter']:4d} kkt={row['kkt_error']:.2e} "
        f"cviol={row['constraint_violation']:.2e} t={row['solve_time']:.1f}s  "
        f"({paper})"
    )


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
        lines.append(
            "| case | config | status | iters | kkt | cviol | time (s) "
            "| paper iters | paper time (s) |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s in solves:
            ref_iters = "" if s["ref_iters"] is None else str(s["ref_iters"])
            ref_time = "" if s["ref_time_s"] is None else f"{s['ref_time_s']:.1f}"
            lines.append(
                f"| {s['problem']} | {s['config']} | {s['status']} | {s['n_iter']} | "
                f"{s['kkt_error']:.2e} | {s['constraint_violation']:.2e} | "
                f"{s['solve_time']:.1f} | {ref_iters} | {ref_time} |"
            )
        lines += [
            "",
            "Paper columns: Breedveld et al. 2017 (doi:10.1007/s10589-017-9919-4), "
            "supplement ESM 2, Mehrotra method, on the paper's 2017 hardware; "
            "CC BY 4.0.",
        ]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
