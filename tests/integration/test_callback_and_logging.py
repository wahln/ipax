"""Iteration callback + logging hooks (intermediate step, not in the plan).

The callback receives an :class:`~ipax.result.IterationInfo` snapshot each
iteration and may request an early stop; logging is emitted through the
``"ipax"`` logger and opted into by ``Options.verbose``.
"""

from __future__ import annotations

import logging

from ipax import (
    AcceptableStoppingOptions,
    FunctionProblem,
    IterationInfo,
    OptimalityConditionOptions,
    Options,
    Status,
    solve,
)
from ipax.testing.problems import BoundConstrainedQP, UnconstrainedQuadratic
from tests._helpers import array, implemented


def test_callback_is_invoked_once_per_recorded_iteration(namespace):
    problem = BoundConstrainedQP(namespace)
    x0 = array(namespace, [0.25, 0.75])

    seen: list[IterationInfo] = []

    with implemented("bound handling"):
        result = solve(
            problem,
            x0,
            options=Options(hessian="exact", linsolve="dense"),
            callback=seen.append,
        )

    # One call per appended history row, in order, carrying the same records.
    assert len(seen) == result.n_iter
    assert [info.record for info in seen] == list(result.history)
    assert [info.record.iteration for info in seen] == list(range(result.n_iter))
    assert all(record.problem_time >= 0.0 for record in result.history)
    assert all(record.step_solve_time >= 0.0 for record in result.history)
    assert result.history[0].problem_time > 0.0
    if result.n_iter > 1:
        assert any(record.step_solve_time > 0.0 for record in result.history[1:])
    # Bounded problem: the snapshot exposes the live bound multipliers.
    assert seen[0].z_lower is not None
    assert seen[0].z_upper is not None
    assert seen[0].y_eq is None


def test_callback_truthy_return_requests_stop(namespace):
    problem = BoundConstrainedQP(namespace)
    x0 = array(namespace, [0.25, 0.75])

    calls = 0

    def stop_after_two(info: IterationInfo) -> bool:
        nonlocal calls
        calls += 1
        return info.record.iteration >= 1

    with implemented("bound handling"):
        result = solve(
            problem,
            x0,
            options=Options(hessian="exact", linsolve="dense"),
            callback=stop_after_two,
        )

    assert result.status is Status.STOPPED
    assert not result.success
    assert "callback" in result.message
    assert calls == 2
    assert result.n_iter == 2


def test_acceptable_stopping_is_integrated_with_the_driver(namespace):
    problem = BoundConstrainedQP(namespace)
    x0 = array(namespace, [0.25, 0.75])

    with implemented("bound handling"):
        result = solve(
            problem,
            x0,
            options=Options(
                hessian="exact",
                linsolve="dense",
                acceptable=AcceptableStoppingOptions(dual_inf_tol=1e6, n_iter=1),
            ),
        )

    assert result.status is Status.ACCEPTABLE
    assert result.success
    assert result.n_iter == 1
    assert "acceptable" in result.message
    assert result.kkt_error == max(
        result.dual_infeasibility,
        result.primal_infeasibility,
        result.complementarity,
    )


def test_component_optimality_threshold_is_integrated_with_the_driver(namespace):
    problem = BoundConstrainedQP(namespace)
    x0 = array(namespace, [0.25, 0.75])

    with implemented("bound handling"):
        result = solve(
            problem,
            x0,
            options=Options(
                hessian="exact",
                linsolve="dense",
                # Loose optimality on dual/complementarity, tight on feasibility.
                optimality=OptimalityConditionOptions(
                    dual_inf_tol=1e-4, compl_inf_tol=1e-4, constr_viol_tol=1e-8
                ),
            ),
        )

    assert result.status is Status.OPTIMAL
    assert result.success
    assert result.kkt_error <= 1e-4
    assert result.primal_infeasibility <= 1e-8
    assert "constraint violation" in result.message


def test_wall_time_stopping_is_integrated_with_the_driver(namespace):
    problem = BoundConstrainedQP(namespace)
    x0 = array(namespace, [0.25, 0.75])

    with implemented("bound handling"):
        result = solve(
            problem,
            x0,
            options=Options(hessian="exact", linsolve="dense", max_time=1e-12),
        )

    assert result.status is Status.MAX_TIME
    assert not result.success
    assert result.n_iter == 1


def test_objective_stagnation_is_guarded_and_skips_the_initial_record(namespace):
    lower = array(namespace, [0.0])
    upper = array(namespace, [2.0])
    problem = FunctionProblem(
        1,
        lambda x: namespace.sum(x * 0.0),
        gradient=lambda x: namespace.zeros_like(x),
        bounds=(lower, upper),
    )

    with implemented("bound handling"):
        result = solve(
            problem,
            array(namespace, [1.0]),
            options=Options(
                linsolve="dense",
                acceptable=AcceptableStoppingOptions(
                    f_rel_change_tol=0.0,
                    dual_inf_tol=1.0,
                    constr_viol_tol=1.0,
                    compl_inf_tol=1.0,
                    n_iter=1,
                ),
            ),
        )

    assert result.status is Status.ACCEPTABLE
    assert result.n_iter == 2
    assert result.kkt_error <= 1.0


def test_convergence_takes_priority_over_stop_request(namespace):
    """A converged final iterate reports OPTIMAL even if the callback stops."""
    Q = array(namespace, [[4.0, 1.0], [1.0, 3.0]])
    b = array(namespace, [1.0, 2.0])
    problem = UnconstrainedQuadratic(Q, b, namespace)
    x0 = problem.known_solution()  # unconstrained minimum: KKT error ~0 at iter 0

    with implemented("dense solver"):
        result = solve(
            problem,
            x0,
            options=Options(hessian="exact", linsolve="dense"),
            callback=lambda info: True,
        )

    assert result.status is Status.OPTIMAL


def test_verbose_emits_iteration_log_to_ipax_logger(namespace, caplog):
    from ipax._logging import ITERATION, RESULT

    problem = BoundConstrainedQP(namespace)
    x0 = array(namespace, [0.25, 0.75])

    # caplog at the ITERATION level captures the result summary (tier 1) and the
    # per-iteration table (tier 2) but not the lower setup tiers.
    with caplog.at_level(ITERATION, logger="ipax"):
        with implemented("bound handling"):
            solve(
                problem,
                x0,
                options=Options(hessian="exact", linsolve="dense", verbose=2),
            )

    records = [rec for rec in caplog.records if rec.name == "ipax"]
    headers = [
        rec.getMessage()
        for rec in records
        if rec.levelno == ITERATION and rec.getMessage().lstrip().startswith("iter")
    ]
    assert headers
    assert "prob_s" in headers[0] and "step_s" in headers[0]
    # final result summary (tier 1) and the timing split (tier 2)
    assert any(
        rec.levelno == RESULT and rec.getMessage().startswith("result:")
        for rec in records
    )
    assert any("timing:" in rec.getMessage() for rec in records)


def test_verbosity_tiers_are_emitted_at_their_levels(namespace, caplog):
    from ipax._logging import ITERATION, OPTIONS, PROBLEM, RESULT, SOLVER

    problem = BoundConstrainedQP(namespace)
    x0 = array(namespace, [0.25, 0.75])

    # Capturing at DEBUG records every tier regardless of verbose; the tiers are
    # distinguished by their numeric level (what the console threshold gates on).
    with caplog.at_level(logging.DEBUG, logger="ipax"):
        with implemented("bound handling"):
            solve(problem, x0, options=Options(hessian="exact", linsolve="dense"))

    by_level = {rec.levelno for rec in caplog.records if rec.name == "ipax"}
    assert {RESULT, ITERATION, PROBLEM, SOLVER, OPTIONS} <= by_level

    def message_at(level: int) -> str:
        return next(
            rec.getMessage()
            for rec in caplog.records
            if rec.name == "ipax" and rec.levelno == level
        )

    assert message_at(PROBLEM).startswith("problem structure:")
    assert message_at(SOLVER).startswith("solver setup:")
    assert message_at(OPTIONS).startswith("options:")


def test_silent_by_default(namespace, caplog):
    problem = BoundConstrainedQP(namespace)
    x0 = array(namespace, [0.25, 0.75])

    with caplog.at_level(logging.DEBUG, logger="ipax"):
        with implemented("bound handling"):
            solve(problem, x0, options=Options(hessian="exact", linsolve="dense"))

    # verbose=0 still routes records through the logger (caplog captures them),
    # but no console handler is attached; assert nothing crashes and the table
    # rows are present for downstream handlers.
    iter_rows = [
        rec for rec in caplog.records if rec.getMessage().lstrip()[:1].isdigit()
    ]
    assert len(iter_rows) >= 1
