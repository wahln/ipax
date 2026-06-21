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
    format_result,
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
        derivative_sources=DerivativeSources(gradient="analytic", hessian="lbfgs"),
        message="converged",
    )
    text = format_result(result)
    assert text.startswith("result: optimal - converged")
    assert "iterations    = 7" in text
    assert "solve time    = 4.200e-02s" in text
    assert "linear solver = sparse [Feral LDL^T (CPU)]" in text
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
