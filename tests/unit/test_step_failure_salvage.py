"""Unit tests for step-failure classification (ACCEPTABLE vs NUMERICAL_ERROR)."""

from __future__ import annotations

from types import SimpleNamespace

from ipax.ipm.driver import (
    _RESTORATION_INFEASIBLE_FACTOR,
    _STEP_FAILURE_ACCEPT_FACTOR,
    _classify_step_failure,
    _restoration_reports_infeasible,
    _within_relaxed_tol,
)
from ipax.options import OptimalityConditionOptions
from ipax.result import IterationRecord, Status


def _record(dual: float, primal: float, compl: float) -> IterationRecord:
    return IterationRecord(
        iteration=10,
        objective=1.0,
        mu=1e-9,
        theta=primal,
        kkt_error=max(dual, primal, compl),
        alpha_primal=1.0,
        alpha_dual=1.0,
        regularization=0.0,
        dual_infeasibility=dual,
        primal_infeasibility=primal,
        complementarity=compl,
    )


_OPT = OptimalityConditionOptions(
    dual_inf_tol=1e-8, constr_viol_tol=1e-8, compl_inf_tol=1e-8
)


def test_near_optimal_step_failure_is_salvaged_as_acceptable():
    # Within 1e2 × tol (e.g. EIGMINA reached ~2.6e-7): a usable solution.
    status, message = _classify_step_failure(_OPT, _record(2.6e-7, 1e-10, 5e-8))
    assert status is Status.ACCEPTABLE
    assert "acceptable" in message


def test_far_from_optimal_step_failure_is_a_numerical_error():
    # KKT residual far above the relaxed tolerance (e.g. DIAGIQB ~1e2).
    status, _ = _classify_step_failure(_OPT, _record(9.8e2, 0.0, 1.0))
    assert status is Status.NUMERICAL_ERROR


def test_nonfinite_residual_is_a_numerical_error():
    status, _ = _classify_step_failure(_OPT, _record(float("inf"), 0.0, 0.0))
    assert status is Status.NUMERICAL_ERROR


def test_salvage_boundary_tracks_the_accept_factor():
    tol = 1e-8
    at = _STEP_FAILURE_ACCEPT_FACTOR * tol
    assert _classify_step_failure(_OPT, _record(at, 0.0, 0.0))[0] is Status.ACCEPTABLE
    just_over = at * 1.0001
    assert (
        _classify_step_failure(_OPT, _record(just_over, 0.0, 0.0))[0]
        is Status.NUMERICAL_ERROR
    )


def test_within_relaxed_tol_gates_both_salvage_paths():
    # The shared near-optimal check used by both the step-failure and the
    # restoration-handoff salvage paths.
    assert _within_relaxed_tol(_OPT, _record(2.6e-7, 1e-10, 5e-8))
    assert not _within_relaxed_tol(_OPT, _record(9.8e2, 0.0, 1.0))
    # Feasibility matters too: a near-zero dual residual but a violated
    # constraint (large primal infeasibility) is not near-optimal.
    assert not _within_relaxed_tol(_OPT, _record(1e-10, 5.0, 1e-10))


def test_restoration_verdict_ignores_a_feasible_restored_point():
    # S2MPJ Task 1: restoration can stall a hair above its own tolerance at a
    # point that is feasible by the driver's θ (a degenerate optimum where CQ
    # fails, or a limit cycle that keeps re-reaching feasibility). Such a point
    # must NOT be believed infeasible — declaring a feasible point "infeasible"
    # is a contradiction. A genuinely infeasible stationary point (θ bounded away
    # from zero) still is.
    tol = _OPT.constr_viol_tol
    feasible_tol = _RESTORATION_INFEASIBLE_FACTOR * tol
    assert not _restoration_reports_infeasible(tol, _OPT)  # at the raw tolerance
    assert not _restoration_reports_infeasible(feasible_tol, _OPT)  # at the boundary
    assert not _restoration_reports_infeasible(1e-6, _OPT)  # HS72-scale violation
    assert _restoration_reports_infeasible(feasible_tol * 1.0001, _OPT)  # just over
    assert _restoration_reports_infeasible(1e-2, _OPT)  # genuinely infeasible


def test_restoration_verdict_falls_back_to_kkt_tol_without_constr_viol_tol():
    # With the constraint-violation condition disabled, the feasibility band is
    # keyed off the representative kkt_tol instead of raising.
    opt = OptimalityConditionOptions(constr_viol_tol=None)
    assert opt.constr_viol_tol is None
    feasible_tol = _RESTORATION_INFEASIBLE_FACTOR * opt.kkt_tol
    assert not _restoration_reports_infeasible(feasible_tol, opt)
    assert _restoration_reports_infeasible(feasible_tol * 10.0, opt)


def test_no_enabled_kkt_conditions_do_not_salvage():
    # Defensive: with no KKT-component tolerances there is nothing to certify, so
    # a step failure stays a numerical error. (The real options object always has
    # at least one enabled, so this uses a duck-typed stand-in.)
    opt = SimpleNamespace(dual_inf_tol=None, constr_viol_tol=None, compl_inf_tol=None)
    status, _ = _classify_step_failure(opt, _record(1e-12, 0.0, 0.0))
    assert status is Status.NUMERICAL_ERROR
