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


def test_run_case_reports_soc_attempted_and_accepted(namespace):
    # ``CaseResult.soc_attempted``/``.soc_accepted`` are sums of the public
    # ``IterationRecord.soc_attempted``/``.used_soc`` off ``Result.history``
    # (see AGENTS.md; ipax/result.py) — not a re-implementation of the
    # controller's bookkeeping. Reuses the textbook Maratos-effect setup
    # (tests/integration/test_breedveld_soc.py) whose far-start Newton steps
    # trigger a Breedveld second-order correction.
    import math

    from benchmarks.corpus import BenchmarkProblem
    from ipax.options import BreedveldOptions
    from ipax.testing.problems import Maratos
    from tests._helpers import array

    start = [0.5 * math.cos(2.5), 0.5 * math.sin(2.5)]
    case = BenchmarkProblem(
        name="maratos",
        kind="NLP",
        tags=(),
        build=lambda xp: (Maratos(namespace), array(namespace, start)),
    )

    def _run(max_soc: int):
        return run_case(
            case,
            config="exact/dense",
            options=ipax.Options(
                hessian="exact",
                linsolve="dense",
                globalization="breedveld",
                breedveld=BreedveldOptions(max_soc=max_soc),
            ),
            xp=namespace,
            backend="test",
        )

    off = _run(0)
    assert off.soc_attempted == 0
    assert off.soc_accepted == 0

    on = _run(1)
    assert on.soc_accepted >= 1
    assert on.soc_attempted >= on.soc_accepted


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


def test_run_case_applies_prepare_solve(namespace, monkeypatch):
    # A case's ``prepare_solve`` hook runs after ``build`` and its start,
    # warm start and options are what ``ipax.solve`` receives; its wall time
    # counts toward the row's solve_time (the TROTS Breedveld initialisation).
    import time
    from dataclasses import replace

    from ipax.result import WarmStart

    seen: dict[str, object] = {}
    warm = WarmStart()
    tuned = ipax.Options(hessian="exact", linsolve="dense", max_iter=50)

    def prepare(problem, x0, options):
        seen["prepare"] = (problem, x0, options)
        time.sleep(0.05)
        started = x0 * 1.0  # a fresh array: the hook's start, not build's
        seen["start"] = started
        return started, warm, tuned

    real_solve = ipax.solve

    def spy_solve(problem, x0, **kwargs):
        seen["solve"] = (problem, x0, kwargs)
        return real_solve(problem, x0, **kwargs)

    monkeypatch.setattr(ipax, "solve", spy_solve)
    base = ipax.Options(hessian="exact", linsolve="dense")
    result = run_case(
        replace(_bound_qp_case(), prepare_solve=prepare),
        config="exact/dense",
        options=base,
        xp=namespace,
        backend="test",
    )

    problem, _x0, given = seen["prepare"]
    assert given is base
    solved_problem, solved_x0, kwargs = seen["solve"]
    assert solved_problem is problem
    assert solved_x0 is seen["start"]  # the solve starts from the hook's x0
    assert kwargs["options"] is tuned
    assert kwargs["warm_start"] is warm
    assert result.error is None and result.correct
    # prepare_time attributes the hook's share; solve_time is the combined
    # (paper-accounting) figure.
    assert result.prepare_time >= 0.04  # sleep(0.05), Windows timer jitter
    assert result.solve_time >= result.prepare_time
    row = json.loads(json.dumps(to_payload([result], capture_environment())))
    assert row["results"][0]["prepare_time"] == result.prepare_time


def test_run_case_without_prepare_solve_passes_no_warm_start(namespace, monkeypatch):
    seen: dict[str, object] = {}
    real_solve = ipax.solve

    def spy_solve(problem, x0, **kwargs):
        seen.update(kwargs)
        return real_solve(problem, x0, **kwargs)

    monkeypatch.setattr(ipax, "solve", spy_solve)
    options = ipax.Options(hessian="exact", linsolve="dense")
    result = run_case(
        _bound_qp_case(),
        config="exact/dense",
        options=options,
        xp=namespace,
        backend="test",
    )
    assert seen == {"options": options}
    assert result.prepare_time == 0.0


def test_run_case_records_a_failing_prepare_solve(namespace):
    from dataclasses import replace

    def boom(problem, x0, options):
        raise RuntimeError("init failed")

    result = run_case(
        replace(_bound_qp_case(), prepare_solve=boom),
        config="exact/dense",
        options=ipax.Options(hessian="exact", linsolve="dense"),
        xp=namespace,
        backend="test",
    )
    assert result.status == "prepare_error"
    assert "init failed" in (result.error or "")


def test_run_case_keeps_prepare_time_when_the_solve_raises(namespace, monkeypatch):
    # Regression: a failed solve row dropped the (possibly expensive) prepare_solve
    # time, so failures looked cheaper than they were under paper accounting.
    import time
    from dataclasses import replace

    from ipax.result import WarmStart

    def prepare(problem, x0, options):
        time.sleep(0.05)
        return x0, WarmStart(), options

    def failing_solve(problem, x0, **kwargs):
        raise RuntimeError("solve failed")

    monkeypatch.setattr(ipax, "solve", failing_solve)
    result = run_case(
        replace(_bound_qp_case(), prepare_solve=prepare),
        config="exact/dense",
        options=ipax.Options(hessian="exact", linsolve="dense"),
        xp=namespace,
        backend="test",
    )
    assert result.status == "solve_error"
    assert result.prepare_time >= 0.04  # sleep(0.05), Windows timer jitter
    assert result.solve_time == result.prepare_time


def test_run_case_keeps_elapsed_time_when_prepare_solve_raises(namespace):
    import time
    from dataclasses import replace

    def boom(problem, x0, options):
        time.sleep(0.05)
        raise RuntimeError("init failed")

    result = run_case(
        replace(_bound_qp_case(), prepare_solve=boom),
        config="exact/dense",
        options=ipax.Options(hessian="exact", linsolve="dense"),
        xp=namespace,
        backend="test",
    )
    assert result.status == "prepare_error"
    assert result.prepare_time >= 0.04  # sleep(0.05), Windows timer jitter
