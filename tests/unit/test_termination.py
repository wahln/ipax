"""Unit tests for the shared optimality / acceptable termination checker."""

from __future__ import annotations

import math

from ipax.ipm.termination import ConditionChecker
from ipax.options import AcceptableStoppingOptions, OptimalityConditionOptions
from ipax.result import IterationRecord, Status


def _record(
    iteration: int,
    objective: float,
    *,
    dual_inf: float = 0.0,
    primal_inf: float = 0.0,
    compl: float = 0.0,
) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        objective=objective,
        mu=0.1,
        theta=0.0,
        kkt_error=max(dual_inf, primal_inf, compl),
        alpha_primal=1.0,
        alpha_dual=1.0,
        regularization=0.0,
        dual_infeasibility=dual_inf,
        primal_infeasibility=primal_inf,
        complementarity=compl,
    )


def test_optimality_fires_in_a_single_iteration():
    checker = ConditionChecker.for_optimality(
        OptimalityConditionOptions(
            dual_inf_tol=1e-8, constr_viol_tol=1e-8, compl_inf_tol=1e-8
        )
    )

    decision = checker.observe(
        _record(0, 1.0, dual_inf=1e-9, primal_inf=1e-9, compl=1e-9)
    )

    assert decision is not None
    assert decision.status is Status.OPTIMAL
    assert "dual infeasibility" in decision.message


def test_optimality_withheld_until_every_component_holds():
    checker = ConditionChecker.for_optimality(
        OptimalityConditionOptions(
            dual_inf_tol=1e-8, constr_viol_tol=1e-8, compl_inf_tol=1e-8
        )
    )

    # Complementarity is still too large on the first iterate.
    assert checker.observe(_record(0, 1.0, dual_inf=1e-9, compl=1.0)) is None
    decision = checker.observe(_record(1, 1.0, dual_inf=1e-9, compl=1e-9))
    assert decision is not None
    assert decision.status is Status.OPTIMAL


def test_optimality_never_fires_on_a_nonfinite_residual():
    checker = ConditionChecker.for_optimality(
        OptimalityConditionOptions(dual_inf_tol=1e8)
    )

    assert checker.observe(_record(0, 1.0, dual_inf=math.nan)) is None


def test_optimality_f_tol_checks_absolute_objective_magnitude():
    checker = ConditionChecker.for_optimality(
        OptimalityConditionOptions(
            f_tol=1e-3, dual_inf_tol=1e-6, constr_viol_tol=None, compl_inf_tol=None
        )
    )

    # |f| above the gate keeps the iterate non-optimal even with tiny residuals.
    assert checker.observe(_record(0, 1.0, dual_inf=1e-9)) is None
    # A level test, so it can fire immediately once |f| is small enough.
    decision = checker.observe(_record(1, 1e-4, dual_inf=1e-9))
    assert decision is not None
    assert decision.status is Status.OPTIMAL
    assert "objective value" in decision.message


def test_optimality_rel_change_needs_a_previous_iterate():
    checker = ConditionChecker.for_optimality(
        OptimalityConditionOptions(
            f_rel_change_tol=0.0,
            dual_inf_tol=1e-6,
            constr_viol_tol=None,
            compl_inf_tol=None,
        )
    )

    # No previous objective on iterate 0, so f_rel_change_tol cannot hold yet.
    assert checker.observe(_record(0, 1.0, dual_inf=1e-9)) is None
    decision = checker.observe(_record(1, 1.0, dual_inf=1e-9))
    assert decision is not None
    assert decision.status is Status.OPTIMAL
    assert "objective change" in decision.message


def test_acceptable_requires_consecutive_iterations():
    checker = ConditionChecker.for_acceptable(
        AcceptableStoppingOptions(dual_inf_tol=1e-3, n_iter=2)
    )

    assert checker.observe(_record(0, 10.0, dual_inf=1e-4)) is None
    decision = checker.observe(_record(1, 9.0, dual_inf=1e-4))

    assert decision is not None
    assert decision.status is Status.ACCEPTABLE
    assert "consecutive iterations" in decision.message


def test_acceptable_count_resets_when_a_condition_fails():
    checker = ConditionChecker.for_acceptable(
        AcceptableStoppingOptions(dual_inf_tol=1e-3, n_iter=2)
    )

    assert checker.observe(_record(0, 10.0, dual_inf=1e-4)) is None
    assert checker.observe(_record(1, 9.0, dual_inf=1.0)) is None  # resets
    assert checker.observe(_record(2, 8.0, dual_inf=1e-4)) is None
    decision = checker.observe(_record(3, 7.0, dual_inf=1e-4))

    assert decision is not None
    assert decision.status is Status.ACCEPTABLE


def test_acceptable_objective_change_handles_a_stuck_dual_infeasibility():
    # The motivating case: dual infeasibility plateaus above any sane tolerance,
    # but the objective and primal feasibility have settled.
    checker = ConditionChecker.for_acceptable(
        AcceptableStoppingOptions(
            f_rel_change_tol=1e-4,
            constr_viol_tol=1e-6,
            n_iter=2,
        )
    )
    stuck = {"dual_inf": 0.5, "primal_inf": 1e-9}

    assert checker.observe(_record(0, 5.0, **stuck)) is None
    assert checker.observe(_record(1, 5.0001, **stuck)) is None
    decision = checker.observe(_record(2, 5.0002, **stuck))

    assert decision is not None
    assert decision.status is Status.ACCEPTABLE
    assert "objective change" in decision.message


def test_disabled_acceptable_checker_never_fires():
    checker = ConditionChecker.for_acceptable(AcceptableStoppingOptions())

    assert checker.observe(_record(0, 1.0)) is None
    assert checker.observe(_record(1, 1.0)) is None
