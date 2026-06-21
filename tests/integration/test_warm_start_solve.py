"""Integration: warm-starting a solve from a prior solution."""

from __future__ import annotations

from ipax import Options, Status, WarmStart, solve
from ipax.options import ScalingOptions
from ipax.problem.function import FunctionProblem
from ipax.testing.problems import BoundConstrainedQP
from tests._helpers import array, assert_allclose, implemented

_DENSE = Options(hessian="exact", linsolve="dense")


def test_warm_start_from_optimum_converges_immediately(namespace):
    xp = namespace
    problem = BoundConstrainedQP(xp)
    x0 = array(xp, [0.25, 0.75])

    with implemented("dense solver"):
        cold = solve(problem, x0, options=_DENSE)
    assert cold.status is Status.OPTIMAL

    # Re-solve from the optimum with the prior multipliers: the iterate is
    # already KKT-optimal, so it should not need more work than the cold solve.
    with implemented("dense solver"):
        warm = solve(
            problem, cold.x, options=_DENSE, warm_start=WarmStart.from_result(cold)
        )
    assert warm.status is Status.OPTIMAL
    assert warm.n_iter <= cold.n_iter
    assert_allclose(xp, warm.x, cold.x, rtol=1e-7, atol=1e-7)
    assert_allclose(xp, warm.z_lower, cold.z_lower, rtol=1e-6, atol=1e-6)
    assert_allclose(xp, warm.z_upper, cold.z_upper, rtol=1e-6, atol=1e-6)


def _badly_scaled_qp(xp):
    """``min 0.5*K*‖x‖²`` s.t. ``x0 + x1 = 1`` — gradient ∞-norm ≫ max_gradient."""
    k = 1.0e6
    a = array(xp, [[1.0, 1.0]])
    b = array(xp, [1.0])
    return FunctionProblem(
        2,
        lambda x: 0.5 * k * xp.sum(x * x),
        gradient=lambda x: k * x,
        linear_eq=(a, b),
        lagrangian_hessian=lambda x, ye, yi, sigma: sigma * k * xp.eye(2),
    )


def test_warm_start_with_scaling_round_trips(namespace):
    # The original-units warm start must be rescaled into the scaled subproblem,
    # so a warm re-solve still lands on the correct unscaled multiplier.
    xp = namespace
    problem = _badly_scaled_qp(xp)
    opts = Options(
        hessian="exact",
        linsolve="dense",
        scaling=ScalingOptions(method="gradient-based"),
    )
    x0 = array(xp, [1.0, 1.0])

    with implemented("dense solver"):
        cold = solve(problem, x0, options=opts)
    assert cold.status is Status.OPTIMAL

    with implemented("dense solver"):
        warm = solve(
            problem, cold.x, options=opts, warm_start=WarmStart.from_result(cold)
        )
    assert warm.status is Status.OPTIMAL
    assert warm.n_iter <= cold.n_iter
    assert_allclose(xp, warm.x, array(xp, [0.5, 0.5]), rtol=1e-6, atol=1e-6)
    assert_allclose(xp, warm.y_eq, array(xp, [-0.5e6]), rtol=1e-6, atol=1e-3)


def test_warm_start_dimension_mismatch_raises(namespace):
    import pytest

    xp = namespace
    problem = BoundConstrainedQP(xp)
    x0 = array(xp, [0.25, 0.75])
    bad = WarmStart(z_lower=array(xp, [1.0, 2.0, 3.0]))  # n is 2, not 3
    with pytest.raises(ValueError, match="z_lower has length 3, expected 2"):
        solve(problem, x0, options=_DENSE, warm_start=bad)
