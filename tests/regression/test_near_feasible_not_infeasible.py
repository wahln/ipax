"""Regression: a near-feasible restoration stall is not declared INFEASIBLE.

S2MPJ v11 mining (v0.7.0 item 3): LEWISPOL (9 nonlinear eqs / 6 vars with a
multiplicity-3 degenerate root) and a small NLS/NE cluster (ARGAUSS, LANCZOS2/3,
MISRA1B) floor at a constraint violation of ~1e-5 — the float64 precision limit
near a degenerate root — yet ipax declared them *locally infeasible*. That
verdict is wrong: the problem is feasible (it has solutions), the solver merely
stalled at a near-feasible point it cannot improve.

IPOPT uses ``constr_viol_tol = 1e-4`` as its feasibility threshold; ipax's
"believe restoration's infeasibility verdict" threshold was ``1e3 ×
constr_viol_tol`` (= 1e-5 at the default), tighter than the achievable
precision on these degenerate problems. Aligning it to ~1e-4 means a point
feasible to that band is never declared infeasible — it resumes and, if it
cannot improve, ends as STALLED (honest) rather than INFEASIBLE (wrong).
Genuinely infeasible problems (violation bounded well away from zero) are
unaffected.
"""

from __future__ import annotations

import math

from ipax import Options, Status, solve
from ipax.ipm.driver import IPMDriver
from ipax.ipm.filter_ls import FilterLineSearch, LineSearchResult
from ipax.ipm.restoration import RestorationExit
from ipax.testing.problems import HS7
from tests._helpers import array


def _fail_every_search(monkeypatch):
    def always_restore(self, **kwargs):
        return LineSearchResult(self._o.alpha_min_frac, False, False, True)

    monkeypatch.setattr(FilterLineSearch, "search", always_restore)


def test_near_feasible_restoration_stall_is_not_infeasible(namespace, monkeypatch):
    # Restoration floors at a NEAR-feasible point: HS7 constraint
    # (1+x1²)² + x2² - 4 at [1, √(5e-5)] is exactly 5e-5 — below IPOPT's 1e-4
    # feasibility band but above the old 1e-5 threshold. A stationarity
    # certificate there must NOT be read as local infeasibility (the objective
    # is far from optimal, so the run honestly stalls instead).
    xp = namespace
    problem = HS7(xp)
    x0 = array(xp, [2.0, 2.0])
    x_near = array(xp, [1.0, math.sqrt(5e-5)])  # theta_l1 = 5e-5
    _fail_every_search(monkeypatch)

    def stub_restore(self, x, s, m, m_eq, mask_l, mask_u, lower_safe, upper_safe):
        return x_near, s, RestorationExit.STATIONARY

    monkeypatch.setattr(IPMDriver, "_restore", stub_restore)
    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", max_iter=80),
    )

    assert result.status is not Status.INFEASIBLE
    assert result.status is Status.STALLED


def test_genuinely_infeasible_restoration_still_reports_infeasible(
    namespace, monkeypatch
):
    # Control: restoration flooring at a violation bounded well away from zero
    # (theta = |26² + 25 - 4| = 697) is genuine local infeasibility and must
    # still be reported — the relaxed band only reprieves near-feasible points.
    xp = namespace
    problem = HS7(xp)
    x0 = array(xp, [2.0, 2.0])
    x_far = array(xp, [5.0, -5.0])  # theta_l1 = 697, far above 1e-4
    _fail_every_search(monkeypatch)

    def stub_restore(self, x, s, m, m_eq, mask_l, mask_u, lower_safe, upper_safe):
        return x_far, s, RestorationExit.STATIONARY

    monkeypatch.setattr(IPMDriver, "_restore", stub_restore)
    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", max_iter=80),
    )

    assert result.status is Status.INFEASIBLE
