"""Regression: a step-solve failure enters restoration, not numerical_error.

S2MPJ v11 mining (v0.7.0 item 2): the exact/dense route reported
``numerical_error`` on ~44 objective-free nonlinear-equation / NLS systems.
Their common failure is that the condensed factorization exhausts the δ_w
escalation ladder — for GROWTH the equality multipliers diverge (|y| 8e5 →
3.5e7), the exact Lagrangian Hessian ``Σ yᵢ ∇²cᵢ`` grows with them, and no δ_w
regularizes a runaway. Reporting that as ``numerical_error`` reads as a solver
bug; the L-BFGS routes report the honest ``infeasible`` / ``restoration_failed``
for the same situation.

Wächter & Biegler 2006, §3.3: a Newton step the inertia correction (δ_w
ladder) cannot complete is a trigger for the *feasibility restoration* phase,
not an outright failure. The driver now routes such a step-solve failure into
the same restoration handler a filter line-search failure takes, so it either
makes feasibility progress or terminates with an honest infeasibility verdict.
"""

from __future__ import annotations

import gc
import weakref

import pytest

from ipax import Options, Status, solve
from ipax.ipm.driver import IPMDriver
from ipax.linalg.krylov import KrylovSolver
from ipax.options import RestorationOptions
from ipax.testing.problems import HS35, InfeasibleEqualities
from tests._helpers import array


def _fail_step_solve_once(monkeypatch):
    """Force the first condensed step solve to report a factorization failure."""
    calls = {"n": 0}
    original = IPMDriver._solve_step

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        dx, dy_eq, dw, ok = original(self, *args, **kwargs)
        if calls["n"] == 1:
            return dx, dy_eq, dw, False  # signal a failed factorization
        return dx, dy_eq, dw, ok

    monkeypatch.setattr(IPMDriver, "_solve_step", flaky)
    return calls


def _count_krylov_solves(monkeypatch):
    """Count ``KrylovSolver.solve`` calls; on a dense main route only restoration
    can make them, so the count tells whether the opt-in route was wired in."""
    calls = {"n": 0}
    original = KrylovSolver.solve

    def counting(self, rhs):
        calls["n"] += 1
        return original(self, rhs)

    monkeypatch.setattr(KrylovSolver, "solve", counting)
    return calls


_RESTORATION_ROUTES = [RestorationOptions(), RestorationOptions(linear_solver="krylov")]


@pytest.mark.parametrize("restoration", _RESTORATION_ROUTES, ids=["dense", "krylov"])
def test_step_solve_failure_enters_restoration_not_numerical_error(
    namespace, monkeypatch, restoration
):
    xp = namespace
    problem = HS35(xp)
    x0 = array(xp, [2.0, 2.0, 2.0])
    calls = _fail_step_solve_once(monkeypatch)
    krylov_solves = _count_krylov_solves(monkeypatch)

    result = solve(
        problem,
        x0,
        options=Options(
            hessian="exact", linsolve="dense", max_iter=100, restoration=restoration
        ),
    )

    assert calls["n"] >= 1
    assert result.status is not Status.NUMERICAL_ERROR
    assert result.status is Status.OPTIMAL
    assert abs(float(result.objective) - 1.0 / 9.0) <= 1e-6
    assert (krylov_solves["n"] > 0) == (restoration.linear_solver == "krylov")


def test_matrix_free_restoration_does_not_pin_its_operator_after_returning(
    namespace, monkeypatch
):
    """The restoration solver must not keep the last restoration point's
    Jacobians (an m×n operator at scale) alive for the rest of the run."""
    xp = namespace
    problem = HS35(xp)
    x0 = array(xp, [2.0, 2.0, 2.0])
    _fail_step_solve_once(monkeypatch)
    factored: list[weakref.ref] = []
    original_factor = KrylovSolver.factor

    def recording_factor(self, operator):
        factored.append(weakref.ref(operator))
        return original_factor(self, operator)

    monkeypatch.setattr(KrylovSolver, "factor", recording_factor)
    alive_after_restoration: list[int] = []

    def callback(record):
        del record
        if factored:
            gc.collect()
            alive_after_restoration.append(sum(ref() is not None for ref in factored))

    result = solve(
        problem,
        x0,
        options=Options(
            hessian="exact",
            linsolve="dense",
            max_iter=100,
            restoration=RestorationOptions(linear_solver="krylov"),
        ),
        callback=callback,
    )

    assert result.status is Status.OPTIMAL
    assert alive_after_restoration, "the run never iterated past restoration"
    assert alive_after_restoration[-1] == 0


def test_step_solve_failure_on_infeasible_problem_reports_honestly(
    namespace, monkeypatch
):
    # A genuinely infeasible problem (x = 0 and x = 1). A step-solve failure
    # must resolve to an honest infeasibility verdict rather than the crash-like
    # numerical_error — exactly the diagnostic upgrade the mining targeted.
    xp = namespace
    problem = InfeasibleEqualities(xp)
    x0 = array(xp, [0.4])
    _fail_step_solve_once(monkeypatch)

    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", max_iter=100),
    )

    assert result.status is not Status.NUMERICAL_ERROR
    assert result.status in (Status.INFEASIBLE, Status.RESTORATION_FAILED)


def test_step_failure_near_optimum_still_converges(namespace, monkeypatch):
    # A step-solve failure late in a converging run must not derail it: the
    # ACCEPTABLE salvage or a re-centering resume still reaches the optimum
    # rather than reporting numerical_error. (The near-optimal ACCEPTABLE
    # classification itself is unit-tested in test_step_failure_salvage.py.)
    xp = namespace
    problem = HS35(xp)
    x0 = array(xp, [0.5, 0.5, 0.5])  # strictly feasible interior

    calls = {"n": 0}
    original = IPMDriver._solve_step

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        dx, dy_eq, dw, ok = original(self, *args, **kwargs)
        if calls["n"] == 6:  # a single failure once the run is well underway
            return dx, dy_eq, dw, False
        return dx, dy_eq, dw, ok

    monkeypatch.setattr(IPMDriver, "_solve_step", flaky)
    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", max_iter=100),
    )
    assert result.status is not Status.NUMERICAL_ERROR
    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    assert abs(float(result.objective) - 1.0 / 9.0) <= 1e-6
