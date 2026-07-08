"""Regression: false "locally infeasible" verdicts and lost best iterates.

S2MPJ v10 audit (2026-07): 52 feasible problems were reported INFEASIBLE
(160 config-rows; 159 already wrong in v8). The DEGENLPA anatomy: the solver
reaches an essentially optimal iterate (θ = 1.7e-7, KKT = 1.7e-6), the
endgame on the degenerate LP then diverges, restoration gives up at
θ ≈ 1e-4 — and the driver declared "locally infeasible" a problem whose own
run history had certified near-feasibility moments earlier, discarding the
good iterate along the way.

Two behaviors under test:

1. **Infeasibility veto** — a local-infeasibility claim is contradicted by
   evidence whenever an accepted iterate already reached the same feasibility
   threshold the verdict uses; such runs report :attr:`Status.STALLED`
   instead.
2. **Best-iterate return** — on failure statuses the driver returns the
   accepted iterate with the lowest scaled KKT error rather than whatever
   wreckage the final iteration left behind.
"""

from __future__ import annotations

from ipax import Options, Status, solve
from ipax.ipm.driver import IPMDriver
from ipax.ipm.filter_ls import FilterLineSearch, LineSearchResult
from ipax.testing.problems import EqualityConstrainedQP
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


def test_infeasible_verdict_vetoed_by_feasible_history(namespace, monkeypatch):
    # After a few good iterations (θ ~ 1e-12 on this QP) the line search is
    # forced to fail and restoration "returns" a wildly infeasible point while
    # claiming local infeasibility. The veto must reject that claim — the run
    # has already visited feasibility — and report STALLED with a usable point.
    xp = namespace
    problem = EqualityConstrainedQP(xp)
    # x0 already satisfies the (linear) constraint, so the very first record
    # certifies θ = 0: feasibility evidence exists before any step. Every
    # line search fails (the QP would otherwise converge in one step).
    _fail_search_after(monkeypatch, 0)

    def bogus_restore(self, x, s, m, m_eq, mask_l, mask_u, lower_safe, upper_safe):
        return array(xp, [10.0, 10.0]), s, True  # θ = 19, claims "infeasible"

    monkeypatch.setattr(IPMDriver, "_restore", bogus_restore)
    result = solve(
        problem,
        array(xp, [2.0, -1.0]),
        options=Options(hessian="exact", linsolve="dense", max_iter=60),
    )

    assert result.status is not Status.INFEASIBLE
    assert result.status is Status.STALLED
    # The returned point is the best accepted iterate, not the bogus jump.
    assert result.constraint_violation <= 1e-6
    assert float(xp.max(xp.abs(result.x))) < 5.0


def test_failure_status_returns_best_iterate(namespace, monkeypatch):
    # After a good start (3 accepted steps toward the optimum), restoration
    # "recovers" to a terrible (but interior, feasible-flagged) point and the
    # run stalls there. The result must carry the lowest-KKT accepted iterate
    # from the good phase, not the wreckage.
    from ipax.testing.problems import BoundConstrainedQP

    xp = namespace
    problem = BoundConstrainedQP(xp)
    _fail_search_after(monkeypatch, 3)

    def bogus_restore(self, x, s, m, m_eq, mask_l, mask_u, lower_safe, upper_safe):
        return array(xp, [10.0, 0.5]), s, False  # bad interior jump, no claim

    monkeypatch.setattr(IPMDriver, "_restore", bogus_restore)
    result = solve(
        problem,
        array(xp, [0.25, 0.75]),
        options=Options(hessian="exact", linsolve="dense", max_iter=100),
    )

    assert not result.status.is_success  # stalled or similar, honestly reported
    best = min(r.kkt_error for r in result.history)
    assert result.kkt_error == best
    # The wreckage point [10, 0.5] is not what comes back.
    assert float(xp.max(xp.abs(result.x))) < 5.0


def test_genuine_infeasibility_still_reported(namespace):
    from ipax.testing.problems import InfeasibleEqualities

    problem = InfeasibleEqualities(namespace)
    result = solve(
        problem,
        array(namespace, [0.5]),
        options=Options(hessian="exact", linsolve="dense", max_iter=200),
    )
    assert result.status is Status.INFEASIBLE
