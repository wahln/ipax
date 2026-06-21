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
from benchmarks.runners.qc import default_configs


def _bound_qp_case():
    return next(c for c in default_corpus() if c.name == "bound_qp")


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
    assert result.error is not None  # captured, not raised


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
