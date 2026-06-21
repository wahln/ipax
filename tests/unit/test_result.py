"""Unit tests for solver result diagnostics."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from ipax.result import (
    DerivativeSources,
    IterationRecord,
    KKTResiduals,
    Result,
    Status,
)
from tests._helpers import array


@pytest.mark.parametrize(
    "status",
    [
        Status.OPTIMAL,
        Status.ACCEPTABLE,
    ],
)
def test_success_is_true_for_strict_and_acceptable_convergence(namespace, status):
    converged = Result(
        status=status,
        x=array(namespace, [1.0]),
        objective=0.0,
    )
    failed = Result(status=Status.MAX_ITER, x=array(namespace, [1.0]), objective=0.0)

    assert converged.success
    assert not failed.success


def test_kkt_residuals_preserve_components_and_maximum():
    residuals = KKTResiduals(
        dual_infeasibility=2e-4,
        primal_infeasibility=3e-5,
        complementarity=4e-6,
    )

    assert residuals.error == 2e-4


def test_nonfinite_kkt_component_makes_the_aggregate_error_infinite():
    residuals = KKTResiduals(
        dual_infeasibility=1e-4,
        primal_infeasibility=math.nan,
        complementarity=1e-5,
    )

    assert residuals.error == math.inf


def test_result_is_immutable(namespace):
    result = Result(status=Status.OPTIMAL, x=array(namespace, [1.0]), objective=0.0)

    with pytest.raises(FrozenInstanceError):
        result.objective = 1.0  # type: ignore[misc]


def test_iteration_history_and_derivative_sources_are_explicit(namespace):
    history = (
        IterationRecord(
            iteration=0,
            objective=1.0,
            mu=0.1,
            theta=0.0,
            kkt_error=1e-2,
            alpha_primal=1.0,
            alpha_dual=1.0,
            regularization=0.0,
            dual_infeasibility=8e-3,
            primal_infeasibility=9e-4,
            complementarity=1e-5,
        ),
    )
    result = Result(
        status=Status.OPTIMAL,
        x=array(namespace, [1.0]),
        objective=0.0,
        derivative_sources=DerivativeSources(gradient="analytic", hessian="exact"),
        history=history,
    )

    assert result.history == history
    assert result.derivative_sources.gradient == "analytic"
    assert result.derivative_sources.hessian == "exact"
    assert history[0].problem_time == 0.0
    assert history[0].step_solve_time == 0.0
    assert history[0].dual_infeasibility == 8e-3
    assert history[0].primal_infeasibility == 9e-4
    assert history[0].complementarity == 1e-5
