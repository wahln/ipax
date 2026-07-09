"""Smoke test for the QC benchmark harness (keeps benchmarks/ from bit-rotting).

Not a performance benchmark — it runs a single fast case so the harness, corpus,
and report formatting stay importable and correct as the solver evolves.
"""

from __future__ import annotations

import json

import ipax
from benchmarks.corpus import default_corpus
from benchmarks.harness import (
    capture_environment,
    format_markdown,
    run_case,
    to_payload,
)
from benchmarks.runners.qc import default_configs, run_sweep


def _bound_qp_case():
    return next(c for c in default_corpus() if c.name == "bound_qp")


def test_run_sweep_honors_per_problem_excluded_configs(backend_name):
    # A per-problem exclude_configs entry must suppress exactly those rows.
    # (Tested on a synthetic exclusion: the corpus no longer ships one since
    # the HS71 corrector-fragility workaround was replaced by the corrector's
    # step-length acceptance fallback.)
    from dataclasses import replace

    base = _bound_qp_case()
    case = replace(base, exclude_configs=("exact/dense+mehrotra",))
    results, _env = run_sweep([case], default_configs(), [backend_name])
    emitted = {r.config for r in results}
    assert emitted.isdisjoint(case.exclude_configs)
    assert "exact/dense" in emitted  # stable configs still run


def test_run_case_scores_a_known_optimum(namespace):
    case = _bound_qp_case()
    result = run_case(
        case,
        config="exact/dense",
        options=ipax.Options(hessian="exact", linsolve="dense"),
        xp=namespace,
        backend="test",
    )
    assert result.correct
    assert result.converged  # correct is a subset of converged
    assert result.success
    assert result.error is None
    assert result.error_vs_optimum is not None and result.error_vs_optimum <= 1e-6
    assert result.linear_solver == "dense"


def test_run_case_records_failures_without_raising(namespace):
    # hessian="exact" with no analytic Hessian source must be reported, not raised.
    class _NoHessian(ipax.Problem):
        @property
        def n_vars(self) -> int:
            return 1

        def objective(self, x):
            return namespace.sum(x * x)

        def gradient(self, x):
            return 2.0 * x

    from benchmarks.corpus import BenchmarkProblem

    case = BenchmarkProblem(
        name="no_hessian",
        kind="NLP",
        tags=(),
        build=lambda xp: (_NoHessian(), namespace.asarray([1.0])),
    )
    result = run_case(
        case,
        config="exact/dense",
        options=ipax.Options(hessian="exact", linsolve="dense"),
        xp=namespace,
        backend="test",
    )
    assert not result.correct
    assert not result.converged  # a failed solve reached no KKT point
    assert result.error is not None  # captured, not raised


def test_run_case_converged_but_not_correct_when_objective_differs(namespace):
    # A solve that reaches a valid KKT point but a *different* objective than the
    # dataset-documented one is `converged` but not `correct` — the tier that
    # credits convergence to a different local optimum on a nonconvex problem.
    from benchmarks.corpus import BenchmarkProblem

    class _QP(ipax.Problem):
        expected_objective = -12345.0  # deliberately wrong "documented" optimum

        @property
        def n_vars(self) -> int:
            return 2

        def objective(self, x):
            return namespace.sum(x * x)

        def gradient(self, x):
            return 2.0 * x

    case = BenchmarkProblem(
        name="wrong_opt",
        kind="NLP",
        tags=(),
        build=lambda xp: (_QP(), namespace.asarray([1.0, 1.0])),
    )
    result = run_case(
        case,
        config="lbfgs/dense",
        options=ipax.Options(hessian="lbfgs", linsolve="dense"),
        xp=namespace,
        backend="test",
    )
    assert result.converged  # reached a KKT point (x -> 0, kkt small)
    assert not result.correct  # but the objective misses the documented value


def test_report_round_trips_json_and_markdown(namespace):
    case = _bound_qp_case()
    label, options = default_configs()[0]
    results = [
        run_case(case, config=label, options=options, xp=namespace, backend="test")
    ]
    env = capture_environment()

    payload = to_payload(results, env)
    restored = json.loads(json.dumps(payload))  # must be JSON-serializable
    assert restored["results"][0]["problem"] == "bound_qp"

    markdown = format_markdown(results, env)
    assert "# ipax quality-control benchmark" in markdown
    assert "bound_qp" in markdown
    assert "converged" in markdown  # the KKT-convergence tier is reported
