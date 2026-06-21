"""Integration: a badly-scaled NLP is solved correctly with auto-scaling on.

The objective is inflated by a large factor ``K`` so its gradient ∞-norm is far
above ``max_gradient``; gradient-based scaling must recover the true (unscaled)
optimum, objective, and multipliers.
"""

from __future__ import annotations

import ipax
from ipax import Options, Status, solve
from ipax.options import ScalingOptions
from ipax.problem.function import FunctionProblem
from tests._helpers import array, assert_allclose, assert_scalar_close, implemented

K = 1.0e6  # objective inflation: ∇f(x0) ∞-norm = K ≫ max_gradient


def _badly_scaled_qp(xp):
    """``min 0.5*K*‖x‖²`` s.t. ``x0 + x1 = 1``. Optimum [0.5, 0.5]."""
    a = array(xp, [[1.0, 1.0]])
    b = array(xp, [1.0])
    return FunctionProblem(
        2,
        lambda x: 0.5 * K * xp.sum(x * x),
        gradient=lambda x: K * x,
        linear_eq=(a, b),
        lagrangian_hessian=lambda x, ye, yi, sigma: sigma * K * xp.eye(2),
    )


def _scaled_options(linsolve: str) -> Options:
    return Options(
        hessian="exact",
        linsolve=linsolve,
        scaling=ScalingOptions(method="gradient-based"),
    )


def test_badly_scaled_qp_solves_with_scaling(namespace):
    xp = namespace
    problem = _badly_scaled_qp(xp)
    x0 = array(xp, [1.0, 1.0])

    with implemented("dense solver"):
        result = solve(problem, x0, options=_scaled_options("dense"))

    assert result.status is Status.OPTIMAL
    assert result.kkt_error <= 1e-6
    # x, objective and multiplier are reported in the *original* problem scale.
    assert_allclose(xp, result.x, array(xp, [0.5, 0.5]), rtol=1e-6, atol=1e-6)
    assert_scalar_close(result.objective, 0.25 * K, rtol=1e-6, atol=1e-3)
    assert_allclose(xp, result.y_eq, array(xp, [-0.5 * K]), rtol=1e-6, atol=1e-3)
    # original-problem stationarity: ∇f(x*) + Aᵀ y* ≈ 0.
    stationarity = problem.gradient(result.x) + array(xp, [1.0, 1.0]) * result.y_eq[0]
    assert_allclose(xp, stationarity, array(xp, [0.0, 0.0]), rtol=1e-5, atol=1e-3)


def test_options_accepts_simple_scaling_shorthand(namespace):
    xp = namespace
    problem = _badly_scaled_qp(xp)
    x0 = array(xp, [1.0, 1.0])
    options = Options(
        hessian="exact",
        linsolve="dense",
        scaling="gradient-based",
    )

    result = solve(problem, x0, options=options)

    assert result.status is Status.OPTIMAL
    assert_allclose(xp, result.x, array(xp, [0.5, 0.5]), rtol=1e-6, atol=1e-6)
    assert isinstance(options.scaling, ScalingOptions)
    assert options.scaling.method == "gradient-based"
    assert ipax.ScalingOptions is ScalingOptions


def test_history_and_callback_use_original_problem_units(namespace):
    xp = namespace
    problem = _badly_scaled_qp(xp)
    x0 = array(xp, [1.0, 1.0])
    seen = []

    result = solve(
        problem,
        x0,
        options=_scaled_options("dense"),
        callback=lambda info: seen.append(info),
    )

    assert result.status is Status.OPTIMAL
    assert_scalar_close(result.history[-1].objective, result.objective, rtol=1e-12)
    assert_scalar_close(seen[-1].record.objective, result.objective, rtol=1e-12)
    assert_allclose(xp, seen[-1].y_eq, result.y_eq, rtol=1e-12, atol=1e-6)


def test_callback_slacks_and_inequality_duals_use_original_units(namespace):
    xp = namespace
    jac = array(xp, [[1000.0]])
    problem = FunctionProblem(
        1,
        lambda x: 0.5 * (x[0] - 2.0) ** 2,
        gradient=lambda x: xp.stack((x[0] - 2.0,)),
        ineq_constraints=lambda x: xp.stack((1000.0 * (x[0] - 1.0),)),
        ineq_jacobian=lambda x: jac,
        lagrangian_hessian=lambda x, ye, yi, sigma: sigma * xp.eye(1),
    )
    seen = []

    result = solve(
        problem,
        array(xp, [0.5]),
        options=_scaled_options("dense"),
        callback=lambda info: seen.append(info) or True,
    )

    assert result.status is Status.STOPPED
    info = seen[0]
    assert_allclose(
        xp,
        problem.ineq_constraints(info.x) + info.s,
        array(xp, [0.0]),
        atol=1e-12,
    )
    assert_allclose(xp, info.s * info.y_ineq, array(xp, [0.1]), atol=1e-12)


def test_scaling_off_by_default_leaves_problem_unscaled(namespace):
    # The default Options does not scale; the same problem still solves (it is
    # convex), confirming scaling is opt-in and the result space is identical.
    xp = namespace
    problem = _badly_scaled_qp(xp)
    x0 = array(xp, [1.0, 1.0])

    with implemented("dense solver"):
        result = solve(problem, x0, options=Options(hessian="exact", linsolve="dense"))

    assert result.status is Status.OPTIMAL
    assert_allclose(xp, result.x, array(xp, [0.5, 0.5]), rtol=1e-6, atol=1e-6)


def test_scaling_matches_across_solver_routes(namespace):
    # The scaled problem must agree through the matrix-free Krylov route, which
    # exercises _RowScaled.matvec/rmatvec/gram on the linear-equality Jacobian.
    xp = namespace
    problem = _badly_scaled_qp(xp)
    x0 = array(xp, [1.0, 1.0])

    with implemented("krylov solver"):
        result = solve(problem, x0, options=_scaled_options("krylov"))

    assert result.status is Status.OPTIMAL
    assert_allclose(xp, result.x, array(xp, [0.5, 0.5]), rtol=1e-5, atol=1e-5)
    assert_allclose(xp, result.y_eq, array(xp, [-0.5 * K]), rtol=1e-5, atol=1e-2)
