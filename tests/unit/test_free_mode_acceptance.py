"""Free-mode line-search acceptance (NWW 2009 §5 globalization framework).

Under a free-mode μ oracle the barrier problem changes every iteration, so the
W&B filter/Armijo machinery is not a consistent merit gate — NWW §5 carries
global convergence entirely in the iterate-level KKT-error monitor
(``FreeModeMonitor``) and lets the free-mode line search "interfere as little
as possible". The weak per-trial test is the §5 obj-constr variant: a trial is
acceptable when ``(θ_t + margin, f_t + margin)`` — the RAW objective, not
φ_μ — is acceptable to the filter of previous free iterates, with IPOPT's
margins (``filter_margin_fact · min(filter_max_margin, kkt_error)``). The
rigorous W&B search governs whenever the monitor is in monotone mode, and the
default (monotone) μ schedule never engages the weak path at all.
"""

from __future__ import annotations

import logging

from ipax import Options, Status, solve
from ipax._logging import LOGGER_NAME
from ipax.ipm.filter_ls import FilterLineSearch
from ipax.options import BarrierOptions, LineSearchOptions
from ipax.testing.problems import HS35
from tests._helpers import array

# ---------------------------------------------------------------------------
# The weak per-trial test itself (pure-float unit tests).
# ---------------------------------------------------------------------------


def test_free_search_accepts_tiny_objective_decrease_rigorous_backtracks():
    # The RT signature: at a feasible iterate with a large |dφ|, Armijo demands
    # a proportional decrease (η·α·|dφ|) and grinds through backtracks, while
    # the free-mode filter needs only a margin-sized improvement over the
    # history — the first trial is accepted.
    ls = FilterLineSearch(LineSearchOptions())

    free = ls.search_free(
        alpha_max=1.0,
        theta_max=1e4,
        eval_point=lambda alpha: (0.0, 1.0 - 1e-6),
        entries=[(0.0, 1.0)],  # the current iterate, remembered by the driver
        margin=1e-8,
    )
    assert free.accepted
    assert free.alpha == 1.0
    assert free.n_trials == 1
    assert not free.augment  # the W&B filter is never augmented from here

    rigorous = ls.search(
        alpha_max=1.0,
        theta0=0.0,
        phi0=1.0,
        dphi=-1e2,  # switching holds at θ0 = 0 ⇒ full Armijo on every trial
        theta_max=1e4,
        eval_point=lambda alpha: (0.0, 1.0 - 1e-6),
        entries=[],
    )
    assert rigorous.n_trials > 1  # Armijo needs φ ≤ 1 − 1e-2·α: backtracks


def test_free_search_margin_semantics():
    # Sufficient progress means beating an entry by the margin in θ OR f
    # (IPOPT AdaptiveMuUpdate::CheckSufficientProgress: Acceptable(f + margin,
    # θ + margin)). An improvement smaller than the margin is not progress.
    ls = FilterLineSearch(LineSearchOptions())

    blocked = ls.search_free(
        alpha_max=1.0,
        theta_max=1e4,
        eval_point=lambda alpha: (0.0, 0.95),  # 0.95 + 0.1 ≥ 1.0: within margin
        entries=[(0.0, 1.0)],
        margin=0.1,
    )
    assert not blocked.accepted
    assert blocked.restoration

    accepted = ls.search_free(
        alpha_max=1.0,
        theta_max=1e4,
        eval_point=lambda alpha: (0.0, 0.85),  # 0.85 + 0.1 < 1.0: past margin
        entries=[(0.0, 1.0)],
        margin=0.1,
    )
    assert accepted.accepted
    assert accepted.n_trials == 1


def test_free_search_accepts_theta_progress_with_worse_objective():
    # Nonmonotone by design: at an infeasible history entry, sufficient θ
    # progress accepts even a (much) worse objective.
    ls = FilterLineSearch(LineSearchOptions())

    result = ls.search_free(
        alpha_max=1.0,
        theta_max=1e4,
        eval_point=lambda alpha: (0.5, 5.0),
        entries=[(1.0, 1.0)],
        margin=0.1,
    )

    assert result.accepted
    assert result.n_trials == 1


def test_free_search_emits_per_trial_debug_trace(caplog):
    # The free-mode search has its own trace labels (raw f, not φ_μ, and
    # free-accept / free-filter reasons), so a non-monotone run's line-search
    # behavior is readable from the log like the rigorous path's.
    ls = FilterLineSearch(LineSearchOptions())
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        result = ls.search_free(
            alpha_max=1.0,
            theta_max=1e4,
            # Rejected at α = 1 (within margin), accepted once α halves.
            eval_point=lambda alpha: (0.0, 1.0 if alpha > 0.5 else 0.5),
            entries=[(0.0, 1.0)],
            margin=0.1,
        )

    assert result.accepted
    msgs = [r.getMessage() for r in caplog.records if "ls trial" in r.getMessage()]
    assert any("free-filter" in m for m in msgs)  # the rejected trial
    assert any("free-accept" in m for m in msgs)  # the accepted one
    assert all("f=" in m for m in msgs)  # raw objective, not φ_μ


def test_free_search_keeps_the_safety_guards():
    # θ_max and non-finite rejections are safety invariants, not merit tests —
    # they survive the weak acceptance unchanged.
    ls = FilterLineSearch(LineSearchOptions())

    exploding = ls.search_free(
        alpha_max=1.0,
        theta_max=1e4,
        eval_point=lambda alpha: (1e30, -1e30),
        entries=[(1.0, 1.0)],
        margin=0.1,
    )
    assert not exploding.accepted
    assert exploding.restoration

    non_finite = ls.search_free(
        alpha_max=1.0,
        theta_max=1e4,
        eval_point=lambda alpha: (0.0, float("-inf")),
        entries=[(1.0, 1.0)],
        margin=0.1,
    )
    assert not non_finite.accepted
    assert non_finite.restoration


def test_free_search_backtracks_past_non_finite_gradient_region():
    # Same overshoot protection as the rigorous path (L-BFGS route): a trial
    # with finite θ/f but overflowing derivatives is rejected so the search
    # backtracks into the finite region.
    ls = FilterLineSearch(LineSearchOptions())

    result = ls.search_free(
        alpha_max=1.0,
        theta_max=1e4,
        eval_point=lambda alpha: (0.0, 0.5),
        entries=[(0.0, 1.0)],
        margin=1e-8,
        grad_finite=lambda alpha: alpha <= 0.3,
    )

    assert result.accepted
    assert result.alpha == 0.25
    assert result.n_trials == 3


# ---------------------------------------------------------------------------
# Driver wiring (recorder-monkeypatch pattern, multi-backend).
# ---------------------------------------------------------------------------


def _record_search_kinds(monkeypatch):
    kinds: list[str] = []
    original = FilterLineSearch.search
    original_free = FilterLineSearch.search_free

    def rigorous(self, **kwargs):
        kinds.append("rigorous")
        return original(self, **kwargs)

    def free(self, **kwargs):
        kinds.append("free")
        return original_free(self, **kwargs)

    monkeypatch.setattr(FilterLineSearch, "search", rigorous)
    monkeypatch.setattr(FilterLineSearch, "search_free", free)
    return kinds


def test_free_oracle_runs_the_weak_acceptance(namespace, monkeypatch):
    # With a free-mode μ oracle and the safeguard never tripping, every line
    # search goes through the weak path.
    kinds = _record_search_kinds(monkeypatch)
    problem = HS35(namespace)
    result = solve(
        problem,
        array(namespace, [0.5, 0.5, 0.5]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            mu_schedule="adaptive",
            barrier=BarrierOptions(fallback="never"),
        ),
    )

    assert result.status is Status.OPTIMAL
    assert len(kinds) > 0
    assert all(kind == "free" for kind in kinds)


def test_monitor_trip_reinstates_the_rigorous_search(namespace, monkeypatch):
    # A tripping KKT-error safeguard (tight κ) flips the loop to monotone mode
    # — from there the rigorous W&B filter search must govern again.
    kinds = _record_search_kinds(monkeypatch)
    problem = HS35(namespace)
    result = solve(
        problem,
        array(namespace, [0.5, 0.5, 0.5]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            mu_schedule="adaptive",
            barrier=BarrierOptions(fallback_kappa=0.01, fallback_window=0),
        ),
    )

    assert result.status is Status.OPTIMAL
    assert "free" in kinds  # free mode ran before the trip
    assert "rigorous" in kinds  # the trip reinstated the W&B gate


def test_rigorous_option_disables_the_weak_acceptance(namespace, monkeypatch):
    # The opt-out: free-mode oracles with free_mode_acceptance="rigorous"
    # keep the W&B gate in both regimes (the pre-feature behavior).
    kinds = _record_search_kinds(monkeypatch)
    problem = HS35(namespace)
    result = solve(
        problem,
        array(namespace, [0.5, 0.5, 0.5]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            mu_schedule="adaptive",
            line_search=LineSearchOptions(free_mode_acceptance="rigorous"),
        ),
    )

    assert result.status is Status.OPTIMAL
    assert len(kinds) > 0
    assert all(kind == "rigorous" for kind in kinds)


def test_free_search_failure_switches_to_monotone(namespace, monkeypatch):
    # A failed free-mode line search is the switch-to-monotone trigger
    # (NWW §5; IPOPT's skipped-line-search signal), not a restoration entry:
    # the very next search must run the rigorous W&B gate from the same
    # iterate, and the solve must still converge.
    from ipax.ipm.filter_ls import LineSearchResult

    kinds: list[str] = []
    original = FilterLineSearch.search
    original_free = FilterLineSearch.search_free
    forced_failure = LineSearchResult(1e-8, False, False, True, n_trials=3)
    failed_once: list[bool] = []

    def free(self, **kwargs):
        kinds.append("free")
        if not failed_once:
            failed_once.append(True)
            return forced_failure
        return original_free(self, **kwargs)

    def rigorous(self, **kwargs):
        kinds.append("rigorous")
        return original(self, **kwargs)

    monkeypatch.setattr(FilterLineSearch, "search", rigorous)
    monkeypatch.setattr(FilterLineSearch, "search_free", free)

    problem = HS35(namespace)
    result = solve(
        problem,
        array(namespace, [0.5, 0.5, 0.5]),
        options=Options(hessian="exact", linsolve="dense", mu_schedule="adaptive"),
    )

    assert result.status is Status.OPTIMAL
    assert kinds[0] == "free"  # the forced failure
    assert kinds[1] == "rigorous"  # the monitor is suspended: W&B governs


def test_monotone_default_never_uses_the_weak_acceptance(namespace, monkeypatch):
    # Containment: the default μ schedule is monotone — there is no free mode,
    # so the weak path must never engage and the default trajectory is the
    # rigorous one, unchanged. (μ-trace identity of the default vs an explicit
    # rigorous opt-out pins the same claim on the observable output.)
    kinds = _record_search_kinds(monkeypatch)
    problem = HS35(namespace)
    x0 = array(namespace, [0.5, 0.5, 0.5])
    default = solve(problem, x0, options=Options(hessian="exact", linsolve="dense"))

    assert default.status is Status.OPTIMAL
    assert len(kinds) > 0
    assert all(kind == "rigorous" for kind in kinds)

    rigorous = solve(
        problem,
        x0,
        options=Options(
            hessian="exact",
            linsolve="dense",
            line_search=LineSearchOptions(free_mode_acceptance="rigorous"),
        ),
    )
    assert [r.mu for r in default.history] == [r.mu for r in rigorous.history]
    assert [r.objective for r in default.history] == [
        r.objective for r in rigorous.history
    ]
