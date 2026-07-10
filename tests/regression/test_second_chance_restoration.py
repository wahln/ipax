"""Regression: second-chance restoration anchored at the starting point.

S2MPJ v10 audit (2026-07): of the 52 feasible problems reported INFEASIBLE,
the restored point was a genuine *local* minimizer of the constraint
infeasibility — no local restoration can escape it — yet for 28 of them a
bound-constrained Gauss-Newton started from the **user's x0** reaches
feasibility directly (BT9, CATENARY, CRESC4/50, HS39/87/90/91/101/102/111,
ELATTAR, GASOIL, MESH, ROBOT, SWOPF, TRAINH, UBH5, ...). The driver therefore
gives the local-infeasibility claim one extra, well-anchored probe before
believing it: rerun restoration once from the initial point, and resume the
main loop if that reaches believable feasibility.
"""

from __future__ import annotations

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


def test_second_chance_restoration_rescues_a_bad_basin(namespace, monkeypatch):
    # HS7 from an infeasible start: after one accepted step the line search is
    # forced to fail, and restoration from the *current* iterate claims local
    # infeasibility at a wildly violated point. Restoration from the original
    # x0 (which this stub recognizes by its argument) finds the feasible point
    # (0, sqrt(3)). The driver must use that second chance instead of reporting
    # INFEASIBLE on a feasible problem.
    xp = namespace
    problem = HS7(xp)
    x0 = array(xp, [2.0, 2.0])  # theta0 = |(1+4)^2 + 4 - 4| = 25, infeasible
    _fail_search_after(monkeypatch, 1)

    restore_calls: list[float] = []

    def stub_restore(self, x, s, m, m_eq, mask_l, mask_u, lower_safe, upper_safe):
        dist_to_x0 = float(xp.max(xp.abs(x - x0)))
        restore_calls.append(dist_to_x0)
        if dist_to_x0 < 1e-9:
            # anchored at the starting point: feasibility is reachable
            feasible = array(xp, [0.0, 1.7320508075688772])  # sqrt(3)
            return feasible, s, RestorationExit.FEASIBLE
        # from the wandered-off iterate: a genuinely stuck local minimizer
        # (theta = |26^2 + 25 - 4| = 697, with a stationarity certificate)
        return array(xp, [5.0, -5.0]), s, RestorationExit.STATIONARY

    monkeypatch.setattr(IPMDriver, "_restore", stub_restore)
    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", max_iter=40),
    )

    # The infeasibility claim was probed from x0 before being believed ...
    assert len(restore_calls) >= 2
    assert restore_calls[1] < 1e-9
    # ... and the feasible rescue point ended the run non-INFEASIBLE.
    assert result.status is not Status.INFEASIBLE
    assert result.constraint_violation <= 1e-6


def test_second_chance_fires_at_most_once(namespace, monkeypatch):
    # Every restoration claims infeasibility, including from x0: the x0-anchored
    # retry must run exactly once (no restart loop), and the final verdict stays
    # INFEASIBLE (the claim survived its probe).
    xp = namespace
    problem = HS7(xp)
    x0 = array(xp, [2.0, 2.0])
    _fail_search_after(monkeypatch, 1)

    restore_calls: list[float] = []

    def stub_restore(self, x, s, m, m_eq, mask_l, mask_u, lower_safe, upper_safe):
        restore_calls.append(float(xp.max(xp.abs(x - x0))))
        return array(xp, [5.0, -5.0]), s, RestorationExit.STATIONARY

    monkeypatch.setattr(IPMDriver, "_restore", stub_restore)
    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", max_iter=40),
    )

    assert result.status is Status.INFEASIBLE
    anchored = [d for d in restore_calls if d < 1e-9]
    assert len(anchored) == 1
