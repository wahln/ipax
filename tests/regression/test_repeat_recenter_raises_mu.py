"""Regression: a repeated feasible-point re-center raises μ instead of
treadmilling at the same barrier.

Observed on an RT fluence case (4882 lower-bounded variables, 2 inequalities,
L-BFGS/sparse): the probing μ oracle pinned μ at its ε/10 floor (lg(mu) = −9.0)
while the iterate sat at exact feasibility (θ = 0). At θ0 = 0 the W&B switching
condition (eq. 19) holds for every descent direction, so each trial must pass
the full Armijo test against a barrier made violently stiff by thousands of
near-active bounds — 11–27 backtracking trials per iteration, a line-search
failure every ~10–15 iterations, and each failure re-centered slacks/duals *at
the same pinned μ* (the 0.6.1 feasible re-center), reproducing the identical
stall for 480 iterations. IPOPT's adaptive oracle escapes this by raising μ.

The fix: the first feasible re-center keeps the current μ (the HS101 repair,
unchanged); a repeat raises it. This test forces two consecutive line-search
failures at a feasible iterate mid-solve and asserts the loop's μ actually
increased after the second re-center — under the monotone schedule μ can
otherwise only decrease, so the raise is unambiguous in the iteration history.
"""

from __future__ import annotations

from ipax import Options, Status, solve
from ipax.ipm.filter_ls import FilterLineSearch, LineSearchResult
from ipax.testing.problems import HS35
from tests._helpers import array


def _fail_searches(monkeypatch, fail_calls):
    """Force the line searches numbered in ``fail_calls`` into restoration."""
    calls = {"n": 0}
    original = FilterLineSearch.search

    def flaky(self, **kwargs):
        calls["n"] += 1
        if calls["n"] in fail_calls:
            return LineSearchResult(self._o.alpha_min_frac, False, False, True)
        return original(self, **kwargs)

    monkeypatch.setattr(FilterLineSearch, "search", flaky)


def test_repeat_recenter_raise_sticks_in_the_loop(namespace, monkeypatch):
    # HS35 from a strictly feasible interior point: the (linear) inequality
    # keeps θ = 0 along the whole trajectory, so both forced failures hit the
    # feasible re-center path. Failures late enough that monotone μ has
    # dropped below mu_init, leaving headroom for the raise.
    problem = HS35(namespace)
    x0 = array(namespace, [0.5, 0.5, 0.5])
    _fail_searches(monkeypatch, fail_calls={5, 6})

    result = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", max_iter=200),
    )

    # The solve must still reach the optimum — the raise is a detour, not a
    # derailment (monotone re-descends μ once the barrier problem is solved).
    assert result.status is Status.OPTIMAL
    assert abs(float(result.objective) - 1.0 / 9.0) <= 1e-6

    restored = [i for i, rec in enumerate(result.history) if rec.restored]
    assert len(restored) >= 2, "expected two re-centered iterations"
    i1, i2 = restored[0], restored[1]
    assert i2 == i1 + 1, "forced failures should re-center on consecutive rows"

    mu_before_first = result.history[i1 - 1].mu
    mu_after_first = result.history[i1].mu
    mu_at_stall = result.history[i2 - 1].mu
    mu_after_second = result.history[i2].mu

    # Guard: the stall must have headroom below the mu_init cap, or the raise
    # is unobservable — if this trips, move fail_calls later in the solve.
    assert mu_at_stall < Options().barrier.mu_init

    # First re-center: no escalation (μ unchanged or normally decreased).
    assert mu_after_first <= mu_before_first
    # Second re-center: the raise reached the loop. Monotone μ never
    # increases on its own, so strict growth here proves the escalation.
    assert mu_after_second > mu_at_stall
