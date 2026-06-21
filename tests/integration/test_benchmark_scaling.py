"""Smoke test for the scaling & memory study harness (fast, tiny sizes)."""

from __future__ import annotations

import math

import ipax
from benchmarks.generators import initial_point, make_rt_like_problem
from benchmarks.harness import (
    capture_environment,
    fit_exponent,
    format_scaling,
    measure_solve,
)
from benchmarks.runners.scaling import run_scaling


def test_fit_exponent_recovers_power_law():
    sizes = [10, 100, 1000]
    quadratic = [float(n) ** 2 for n in sizes]
    assert math.isclose(fit_exponent(sizes, quadratic), 2.0, abs_tol=1e-9)
    # Degenerate input -> nan, never raises.
    assert math.isnan(fit_exponent([100], [1.0]))


def test_measure_solve_reports_time_and_memory():
    problem = make_rt_like_problem(80, n_structures=4, density=0.2, seed=0)
    x0 = initial_point(80)
    elapsed, peak, result = measure_solve(
        problem, x0, ipax.Options(hessian="exact", linsolve="krylov")
    )
    assert elapsed > 0.0
    assert peak > 0.0  # tracemalloc records allocations during the solve
    assert result.success


def test_run_scaling_produces_points_and_report():
    points = run_scaling(["krylov"], [60, 120])
    assert [p.n_vars for p in points] == [60, 120]
    assert all(p.success for p in points)
    assert all(p.peak_memory_mb > 0.0 for p in points)

    markdown = format_scaling(points, capture_environment())
    assert "scaling & memory study" in markdown
    assert "memory exponent" in markdown
