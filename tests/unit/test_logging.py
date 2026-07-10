"""Unit tests for the layered verbosity diagnostics."""

from __future__ import annotations

import logging

from ipax._logging import (
    ITERATION,
    OPTIONS,
    PROBLEM,
    RESULT,
    SOLVER,
    configure_verbosity,
    format_options,
    format_problem,
    format_record,
    format_result,
    format_setup,
    format_solver,
    format_timing,
    logger,
    verbosity_threshold,
)
from ipax.options import Options, ScalingOptions
from ipax.result import DerivativeSources, IterationRecord, Result, Status


def test_threshold_ladder_matches_tiers():
    # 0 silences ipax output (above every tier); 1..5 map to the content tiers.
    assert verbosity_threshold(0) > RESULT
    assert verbosity_threshold(1) == RESULT
    assert verbosity_threshold(2) == ITERATION
    assert verbosity_threshold(3) == PROBLEM
    assert verbosity_threshold(4) == SOLVER
    assert verbosity_threshold(5) == OPTIONS
    assert verbosity_threshold(6) == logging.DEBUG
    assert verbosity_threshold(99) == logging.DEBUG


def test_tier_levels_are_strictly_ordered():
    # Lower verbose shows only headline tiers ⇒ higher numeric level.
    assert RESULT > ITERATION > PROBLEM > SOLVER > OPTIONS > logging.DEBUG


def test_configure_verbosity_is_idempotent_and_sets_handler_level():
    def _ipax_handlers():
        return [
            h for h in logger.handlers if getattr(h, "_ipax_verbose_handler", False)
        ]

    configure_verbosity(2)
    configure_verbosity(4)  # repeat call must not stack handlers
    handlers = _ipax_handlers()
    assert len(handlers) == 1
    assert handlers[0].level == verbosity_threshold(4)


def test_app_handler_on_ipax_logger_prevents_duplicate_output():
    # An application that attaches its own handler to the "ipax" logger keeps
    # full control: configure_verbosity must not add a second console handler
    # (that duplicate is what prints each iteration record twice).
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    try:
        logger.handlers[:] = [logging.NullHandler()]
        app_handler = logging.StreamHandler()
        logger.addHandler(app_handler)
        configure_verbosity(2)
        tagged = [
            h for h in logger.handlers if getattr(h, "_ipax_verbose_handler", False)
        ]
        assert tagged == []  # deferred to the application's handler
        # The threshold is still lowered so the requested tiers reach that handler.
        assert logger.level <= verbosity_threshold(2)
    finally:
        logger.handlers[:] = saved_handlers
        logger.setLevel(saved_level)


def test_owned_handler_dropped_when_app_attaches_later():
    # If ipax created its console handler first (verbose call) and the application
    # later attaches its own, a subsequent configure_verbosity must drop ipax's
    # handler and defer to the app — otherwise both emit (duplicate output).
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    try:
        logger.handlers[:] = [logging.NullHandler()]
        configure_verbosity(2)  # ipax creates its owned handler
        assert any(getattr(h, "_ipax_verbose_handler", False) for h in logger.handlers)
        logger.addHandler(logging.StreamHandler())  # app attaches its own, later
        configure_verbosity(2)  # must now defer and drop ipax's owned handler
        tagged = [
            h for h in logger.handlers if getattr(h, "_ipax_verbose_handler", False)
        ]
        assert tagged == []
    finally:
        logger.handlers[:] = saved_handlers
        logger.setLevel(saved_level)


def test_format_record_marks_acceptable_iterates():
    record = IterationRecord(3, 1.0, 1e-9, 1e-9, 1e-9, 1.0, 1.0, 0.0, 0.0, 0.0)
    plain = format_record(record)
    marked = format_record(record, acceptable=True)
    assert not plain.endswith("*")
    assert marked == f"{plain} *"


def test_format_record_reports_line_search_trials():
    record = IterationRecord(
        3, 1.0, 1e-9, 1e-9, 1e-9, 1.0, 1.0, 0.0, 0.0, 0.0, line_search_iters=4
    )
    row = format_record(record)
    assert "   4 " in row  # right-justified "ls" column


def test_format_record_marks_restored_iterates():
    record = IterationRecord(
        3, 1.0, 1e-9, 1e-9, 1e-9, 1.0, 1.0, 0.0, 0.0, 0.0, restored=True
    )
    row = format_record(record)
    assert row.rstrip().endswith("R")


def test_format_record_combines_restored_and_acceptable_tags():
    record = IterationRecord(
        3, 1.0, 1e-9, 1e-9, 1e-9, 1.0, 1.0, 0.0, 0.0, 0.0, restored=True
    )
    row = format_record(record, acceptable=True)
    assert row.rstrip().endswith("R *")


def test_format_setup_reports_condensed_headline():
    line = format_setup(
        n_vars=120,
        n_lower=45,
        n_upper=30,
        n_eq=12,
        n_ineq=8,
        linear_solver="dense (DenseSolver)",
        hessian="lbfgs",
    )
    assert "120 variables" in line
    assert "45 lower, 30 upper" in line
    assert "12 equalities" in line
    assert "8 inequalities" in line
    assert "linear solver = dense (DenseSolver)" in line
    assert "hessian = lbfgs" in line


def test_format_result_reports_status_and_sources():
    result = Result(
        status=Status.OPTIMAL,
        x=None,
        objective=1.25,
        n_iter=7,
        kkt_error=3e-9,
        constraint_violation=0.0,
        solve_time=0.042,
        linear_solver="sparse [Feral LDL^T (CPU)]",
        device="cpu",
        derivative_sources=DerivativeSources(gradient="analytic", hessian="lbfgs"),
        message="converged",
    )
    text = format_result(result)
    assert text.startswith("result: optimal - converged")
    assert "iterations    = 7" in text
    assert "solve time    = 4.200e-02s" in text
    assert "linear solver = sparse [Feral LDL^T (CPU)]" in text
    assert "device        = cpu" in text
    assert "grad:analytic" in text and "hess:lbfgs" in text


def test_format_timing_sums_history():
    history = (
        IterationRecord(0, 1.0, 0.1, 0.0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.25),
        IterationRecord(1, 1.0, 0.1, 0.0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.75),
    )
    text = format_timing(history)
    assert "1.000e+00s" in text  # problem-callbacks total
    assert "inner-solve = 1.000e+00s" in text


def test_format_problem_and_solver_and_options():
    problem = format_problem(
        n_vars=10, n_ineq=2, n_eq_nonlinear=1, n_eq_linear=3, n_lower=4, n_upper=5
    )
    assert "variables    = 10" in problem
    assert "1 nonlinear + 3 linear" in problem

    opts = Options(
        hessian="exact",
        linsolve="dense",
        scaling=ScalingOptions(method="gradient-based"),
    )
    solver = format_solver(opts, "DenseSolver")
    assert "linear solver = dense (DenseSolver)" in solver
    assert "scaling       = gradient-based" in solver

    dump = format_options(opts)
    assert dump.startswith("options:")
    assert "barrier:" in dump  # nested sub-options expanded
    assert "mu_init" in dump
