"""Regression: uncertified restoration stalls must not become INFEASIBLE.

S2MPJ restfix subset (2026-07-08): the trailing-window stall guard in
restoration made LAKES/NASH/SWOPF on exact/krylov exit restoration *early*
(before the clock), and the driver converted that mere stall into a "locally
infeasible" verdict — a semantic upgrade the evidence does not support. Only
a stationarity-type exit of the infeasibility minimization (projected
gradient ~ 0, or no descent at the Levenberg-Marquardt damping ceiling) is a
certificate of local infeasibility (Wächter & Biegler 2006, §3.3).

Behavior under test:

1. A window/budget exit with the violation still large **resumes** the main
   loop as long as restoration keeps reducing θ between stalls.
2. When an uncertified stall stops making θ progress, the run ends as
   :attr:`Status.RESTORATION_FAILED` — an honest failure label — never as
   :attr:`Status.INFEASIBLE`.
"""

from __future__ import annotations

import math

from ipax import Options, Status, solve
from ipax.ipm.driver import IPMDriver
from ipax.ipm.filter_ls import FilterLineSearch, LineSearchResult
from ipax.ipm.restoration import RestorationExit
from ipax.testing.problems import HS7
from tests._helpers import array


def _fail_search_after(monkeypatch, n_good: int):
    """Let the first ``n_good`` line searches succeed, then fail every one."""
    calls = {"n": 0}
    original = FilterLineSearch.search

    def flaky(self, **kwargs):
        calls["n"] += 1
        if calls["n"] <= n_good:
            return original(self, **kwargs)
        return LineSearchResult(self._o.alpha_min_frac, False, False, True)

    monkeypatch.setattr(FilterLineSearch, "search", flaky)


def _hs7_point_with_violation(xp, theta: float):
    # HS7 equality: (1 + x1^2)^2 + x2^2 - 4 = 0, so x = (0, sqrt(3 + theta))
    # violates it by exactly theta.
    return array(xp, [0.0, math.sqrt(3.0 + theta)])


def test_non_progressing_uncertified_stall_is_restoration_failed(
    namespace, monkeypatch
):
    # Every restoration returns the same wildly violated point via the stall
    # window (no certificate). The driver may resume once on the first stall,
    # but a repeat with zero θ progress must end the run as RESTORATION_FAILED,
    # not as INFEASIBLE (no stationarity certificate was ever produced).
    xp = namespace
    problem = HS7(xp)
    x0 = array(xp, [2.0, 2.0])
    _fail_search_after(monkeypatch, 1)

    restore_calls = {"n": 0}

    def stub_restore(self, x, s, m, m_eq, mask_l, mask_u, lower_safe, upper_safe):
        restore_calls["n"] += 1
        return _hs7_point_with_violation(xp, 8.0), s, RestorationExit.STALL_WINDOW

    monkeypatch.setattr(IPMDriver, "_restore", stub_restore)
    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", max_iter=40),
    )

    assert result.status is Status.RESTORATION_FAILED
    assert result.status is not Status.INFEASIBLE
    # First stall resumes (progress against the infinite baseline); the second,
    # non-progressing stall spends the one-shot x0-anchored probe (call 3,
    # itself an uncertified stall, so it resumes once more); the fourth stall
    # has no progress and no probe left: exactly four restoration entries.
    assert restore_calls["n"] == 4


def test_progressing_uncertified_stalls_resume_the_main_loop(namespace, monkeypatch):
    # Restoration keeps stalling without a certificate but reduces θ each time
    # (8 -> 4 -> 2 -> 1); the driver must resume through the whole progressing
    # streak and only fail on the first non-progressing repeat.
    xp = namespace
    problem = HS7(xp)
    x0 = array(xp, [2.0, 2.0])
    _fail_search_after(monkeypatch, 1)

    thetas = [8.0, 4.0, 2.0, 1.0, 1.0]
    restore_calls = {"n": 0}

    def stub_restore(self, x, s, m, m_eq, mask_l, mask_u, lower_safe, upper_safe):
        theta = thetas[min(restore_calls["n"], len(thetas) - 1)]
        restore_calls["n"] += 1
        return _hs7_point_with_violation(xp, theta), s, RestorationExit.BUDGET

    monkeypatch.setattr(IPMDriver, "_restore", stub_restore)
    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", max_iter=40),
    )

    assert result.status is Status.RESTORATION_FAILED
    # The four progressing stalls resume; the first non-progressing repeat
    # spends the x0 probe (one extra call, resumed once more), and the next
    # repeat terminates: len(thetas) + 2 restoration entries in total.
    assert restore_calls["n"] == len(thetas) + 2


def test_uncertified_stall_still_gets_the_x0_anchored_rescue(namespace, monkeypatch):
    # The first restexit A/B (2026-07-08) regression: HS90/HS91/SNAKE/BT9/HS39
    # were second-chance rescues (feasibility IS reachable from x0), but the
    # uncertified-stall path terminated as RESTORATION_FAILED without ever
    # probing from x0. A non-progressing uncertified stall must spend the same
    # one-shot probe before giving up.
    xp = namespace
    problem = HS7(xp)
    x0 = array(xp, [2.0, 2.0])
    # Searches 2 and 3 fail (driving the two uncertified stalls that spend the
    # probe); after the rescue jumps to the feasible point the search works
    # again — the real HS90/SNAKE anatomy, where the solve converges from the
    # rescued basin.
    calls = {"n": 0}
    original = FilterLineSearch.search

    def flaky(self, **kwargs):
        calls["n"] += 1
        if calls["n"] in (2, 3):
            return LineSearchResult(self._o.alpha_min_frac, False, False, True)
        return original(self, **kwargs)

    monkeypatch.setattr(FilterLineSearch, "search", flaky)

    probe_seen = {"n": 0}

    def stub_restore(self, x, s, m, m_eq, mask_l, mask_u, lower_safe, upper_safe):
        if float(xp.max(xp.abs(x - x0))) < 1e-9:
            probe_seen["n"] += 1
            feasible = array(xp, [0.0, 1.7320508075688772])  # sqrt(3)
            return feasible, s, RestorationExit.FEASIBLE
        return _hs7_point_with_violation(xp, 8.0), s, RestorationExit.STALL_WINDOW

    monkeypatch.setattr(IPMDriver, "_restore", stub_restore)
    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", max_iter=40),
    )

    assert probe_seen["n"] == 1
    assert result.status is not Status.INFEASIBLE
    assert result.status is not Status.RESTORATION_FAILED
    assert result.constraint_violation <= 1e-6


def test_certified_stationary_claim_still_reports_infeasible(namespace, monkeypatch):
    # Control: a stationarity-certified claim (surviving its x0-anchored second
    # chance) must keep reporting INFEASIBLE exactly as before.
    xp = namespace
    problem = HS7(xp)
    x0 = array(xp, [2.0, 2.0])
    _fail_search_after(monkeypatch, 1)

    def stub_restore(self, x, s, m, m_eq, mask_l, mask_u, lower_safe, upper_safe):
        return array(xp, [5.0, -5.0]), s, RestorationExit.STATIONARY

    monkeypatch.setattr(IPMDriver, "_restore", stub_restore)
    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", max_iter=40),
    )

    assert result.status is Status.INFEASIBLE
