"""Regression: feasible-point ascent directions must not freeze the iteration.

Found via the S2MPJ v9 sweep (2026-07): on θ ≡ 0 problems (unconstrained /
bounds-only — 85 of the 126 exact/krylov regressions) an iterative KKT solve
can return a *non-descent* direction without failing — CG on an indefinite
condensed operator "succeeds" with garbage, and success is not a descent
certificate. The filter line search then correctly rejects every trial
(at a feasible iterate only Armijo-type acceptance exists, W&B 2006 §2.3),
restoration is a no-op, and nothing changes state: the identical ascent
direction is recomputed forever (POWELLBSLS burned 10 001 iterations with
dφ = +7.7e-3 frozen to every digit).

Two safeguards, both regression-tested here:

1. **Descent enforcement** — at a feasible iterate (θ = 0), a step with
   dφ ≥ 0 is treated like a failed factorization: δ_w is escalated and the
   KKT system re-solved until the direction has the descent property the
   line search needs (the Krylov analogue of the dense route's Cholesky
   PD-probe; W&B 2006 §3.1 inertia-correction semantics).
2. **Stall detector** (`Options.max_stall_iter`) — consecutive iterations
   with a zero accepted steplength and a frozen KKT error terminate honestly
   (:attr:`Status.STALLED`, or ACCEPTABLE within the relaxed tolerance)
   instead of burning the whole iteration budget.
"""

from __future__ import annotations

import pytest

from ipax import FunctionProblem, Options, Status, solve
from ipax.ipm.driver import IPMDriver
from ipax.ipm.filter_ls import FilterLineSearch, LineSearchResult
from ipax.testing.problems import BoundConstrainedQP
from tests._helpers import array


def _powell_badly_scaled(xp):
    """POWELLBSLS: ``(1e4·x1x2 − 1)² + (e^{−x1} + e^{−x2} − 1.0001)²``.

    Unconstrained (θ ≡ 0) with a wildly scaled, indefinite Hessian around the
    SIF start ``(0, 1)`` — the exact configuration that dead-looped on the
    Krylov route. Optimum f* = 0 at ``(1.098…e-5, 9.106…)``.
    """

    def residuals(x):
        r1 = 1e4 * x[0] * x[1] - 1.0
        r2 = xp.exp(-x[0]) + xp.exp(-x[1]) - 1.0001
        return r1, r2

    def objective(x):
        r1, r2 = residuals(x)
        return r1 * r1 + r2 * r2

    def gradient(x):
        r1, r2 = residuals(x)
        e1, e2 = xp.exp(-x[0]), xp.exp(-x[1])
        g1 = 2.0 * r1 * 1e4 * x[1] + 2.0 * r2 * (-e1)
        g2 = 2.0 * r1 * 1e4 * x[0] + 2.0 * r2 * (-e2)
        return xp.stack((g1, g2))

    def lagrangian_hessian(x, y_eq, y_ineq, sigma=1.0):
        r1, r2 = residuals(x)
        e1, e2 = xp.exp(-x[0]), xp.exp(-x[1])
        a1, b1 = 1e4 * x[1], 1e4 * x[0]  # ∇r1
        h11 = 2.0 * (a1 * a1 + e1 * e1) + 2.0 * r2 * e1
        h22 = 2.0 * (b1 * b1 + e2 * e2) + 2.0 * r2 * e2
        h12 = 2.0 * (a1 * b1 + e1 * e2) + 2.0 * r1 * 1e4
        hess = xp.stack((xp.stack((h11, h12)), xp.stack((h12, h22))))
        return sigma * hess

    problem = FunctionProblem(
        2, objective, gradient=gradient, lagrangian_hessian=lagrangian_hessian
    )
    return problem, array(xp, [0.0, 1.0])


def test_krylov_escapes_feasible_ascent_direction(namespace):
    # Pre-fix: CG's garbage direction on the indefinite Hessian is rejected
    # forever (max_iter at f = 1.135, e0 = 100 frozen); the dense route already
    # solved this via its Cholesky PD-probe + delta_w escalation.
    problem, x0 = _powell_badly_scaled(namespace)
    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="krylov", max_iter=300),
    )
    assert result.status is Status.OPTIMAL, (
        f"{result.status} (f={float(result.objective):.3e}, "
        f"kkt={result.kkt_error:.3e}, n_iter={result.n_iter})"
    )
    assert float(result.objective) <= 1e-10


def test_stall_detector_terminates_honestly(namespace, monkeypatch):
    # Force every line search to fail: restoration at a feasible point is a
    # no-op, so without the detector the run burns the whole budget with
    # alpha = 0 and a frozen KKT error.
    def always_fail(self, **kwargs):
        return LineSearchResult(self._o.alpha_min_frac, False, False, True)

    monkeypatch.setattr(FilterLineSearch, "search", always_fail)
    result = solve(
        BoundConstrainedQP(namespace),
        array(namespace, [0.25, 0.75]),
        options=Options(
            hessian="exact", linsolve="dense", max_iter=500, max_stall_iter=10
        ),
    )
    assert result.status is Status.STALLED
    assert result.n_iter < 50  # detector, not the iteration budget
    assert "stalled" in result.message


def test_stall_detector_disabled_runs_to_budget(namespace, monkeypatch):
    def always_fail(self, **kwargs):
        return LineSearchResult(self._o.alpha_min_frac, False, False, True)

    monkeypatch.setattr(FilterLineSearch, "search", always_fail)
    result = solve(
        BoundConstrainedQP(namespace),
        array(namespace, [0.25, 0.75]),
        options=Options(
            hessian="exact", linsolve="dense", max_iter=40, max_stall_iter=None
        ),
    )
    assert result.status is Status.MAX_ITER


def test_max_stall_iter_validation():
    with pytest.raises(ValueError, match="max_stall_iter"):
        Options(max_stall_iter=0)


def test_descent_enforcement_reuses_step_when_already_descent(namespace, monkeypatch):
    # The enforcement must be a no-op on a healthy solve: a PD problem's step
    # is descent, so no extra KKT solves happen (count them).
    problem = BoundConstrainedQP(namespace)
    calls = {"n": 0}
    original = IPMDriver._solve_step

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(IPMDriver, "_solve_step", counting)
    result = solve(
        problem,
        array(namespace, [0.25, 0.75]),
        options=Options(hessian="exact", linsolve="dense", mu_schedule="monotone"),
    )

    assert result.status is Status.OPTIMAL
    # One solve per non-terminal iteration on the plain monotone path (the
    # final iteration converges before solving): no corrector and no descent
    # re-solves on a convex QP.
    assert calls["n"] == result.n_iter - 1
