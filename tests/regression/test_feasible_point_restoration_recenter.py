"""Regression: line-search failure at a feasible point re-centers, not restores.

S2MPJ v11 audit (HS101, all exact routes): the main loop limit-cycled at a
restored feasible point. Restoration returns slacks on its 1e-12 boundary
floor and the driver kept the stale multipliers (λ ~ 1e6 on the active set),
so Σ_s = λ/s reached ~1e18 — the KKT system needed δ_w ~ 1e5 to pass the
inertia check and the fraction-to-boundary rule capped every step at ~1e-11.
The line search then failed forever, re-entering restoration at the same
feasible point ~25× until the stall detector returned an *infeasible* iterate.

Restoration cannot move an already-feasible point, so entering it there is
useless by construction. The driver instead repairs the barrier state: slacks
re-floored on the current μ and inequality multipliers clipped into the
central band (Wächter & Biegler 2006, §3.3 / eq. (16)).
"""

from __future__ import annotations

from ipax import Options, Status, solve
from ipax.ipm.driver import IPMDriver
from ipax.ipm.filter_ls import FilterLineSearch, LineSearchResult
from ipax.result import WarmStart
from ipax.testing.problems import HS35
from tests._helpers import array


def _fail_first_search(monkeypatch):
    """Force the first line search into restoration; run the rest normally."""
    calls = {"n": 0}
    original = FilterLineSearch.search

    def flaky(self, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return LineSearchResult(self._o.alpha_min_frac, False, False, True)
        return original(self, **kwargs)

    monkeypatch.setattr(FilterLineSearch, "search", flaky)


def test_feasible_entry_recenters_instead_of_restoring(namespace, monkeypatch):
    # HS35 from a strictly feasible start: theta0 = 0, so a (forced) line-search
    # failure must NOT call restore() — the repair path re-centers the barrier
    # state at the same point and the solve still reaches the optimum.
    xp = namespace
    problem = HS35(xp)
    x0 = array(xp, [0.5, 0.5, 0.5])  # g = -1: strictly feasible interior
    _fail_first_search(monkeypatch)

    def forbidden_restore(self, *args, **kwargs):
        raise AssertionError("restore() must not run at a feasible point")

    monkeypatch.setattr(IPMDriver, "_restore", forbidden_restore)
    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", max_iter=100),
    )

    assert result.status is Status.OPTIMAL
    assert abs(float(result.objective) - 1.0 / 9.0) <= 1e-6
    # The repair surfaces as a restoration-tagged iteration row.
    assert any(rec.restored for rec in result.history)


def test_infeasible_entry_still_restores(namespace, monkeypatch):
    # Gate boundary: from an infeasible point (theta0 >> tol) the classic
    # restoration phase must still run — the repair only replaces it where
    # restoration provably cannot move the point.
    xp = namespace
    problem = HS35(xp)
    x0 = array(xp, [2.0, 2.0, 2.0])  # g = +3: violated inequality
    _fail_first_search(monkeypatch)

    restore_calls = {"n": 0}
    original_restore = IPMDriver._restore

    def counting_restore(self, *args, **kwargs):
        restore_calls["n"] += 1
        return original_restore(self, *args, **kwargs)

    monkeypatch.setattr(IPMDriver, "_restore", counting_restore)
    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", max_iter=100),
    )

    assert restore_calls["n"] >= 1
    assert result.status is Status.OPTIMAL
    assert abs(float(result.objective) - 1.0 / 9.0) <= 1e-6


def test_recenter_breaks_a_poisoned_dual_deadlock(namespace, monkeypatch):
    # The HS101 mechanism in miniature: start feasible with a wildly poisoned
    # warm-started multiplier (λ = 1e6 on a constraint that is active at the
    # optimum). Without the repair, a line-search failure at a feasible point
    # resumes with that multiplier untouched. With it, the solve must still
    # converge to the optimum.
    xp = namespace
    problem = HS35(xp)
    x0 = array(xp, [0.5, 0.5, 0.5])
    _fail_first_search(monkeypatch)

    def forbidden_restore(self, *args, **kwargs):
        raise AssertionError("restore() must not run at a feasible point")

    monkeypatch.setattr(IPMDriver, "_restore", forbidden_restore)
    warm = WarmStart(y_ineq=array(xp, [1e6]))
    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", max_iter=200),
        warm_start=warm,
    )

    assert result.status is Status.OPTIMAL
    assert abs(float(result.objective) - 1.0 / 9.0) <= 1e-6
