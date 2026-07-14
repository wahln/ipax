"""Unit tests for the barrier μ oracles (``Options.mu_schedule``).

Published rules under test:

* ``"monotone"`` — guarded Fiacco–McCormick reduction
  (Wächter & Biegler 2006, eq. (7)).
* ``"adaptive"`` — LOQO centrality rule, μ = σ·(average complementarity) with
  σ = 0.1·min(0.05·(1−ξ)/ξ, 2)³ and ξ = min complementarity / average
  (Nocedal, Wächter & Waltz 2009, eqs. (3.1) and (3.6)).
* ``"breedveld"`` — duality-gap update μ⁺ = σ(α)·(average complementarity)
  with σ = ((α−1)/(α+10))² from the last accepted steplength
  (Breedveld et al. 2017, eqs. (10)–(12)).
* ``"probing"`` (default) — Mehrotra σ-rule from an affine probe (NWW 2009,
  eqs. (3.2)–(3.5)), usable with or without a corrector.

The oracle is orthogonal to ``corrections``: correctors consume the oracle's
μ target rather than choosing their own. The free-mode oracles are safeguarded
twice: the KKT-error fallback of NWW 2009, §5.1 (Algorithm A) switches to the
monotone rule when progress stalls, and the centrality floor
μ ≥ κ_cent·(primal/dual infeasibility) (El-Bakry et al. 1996) keeps an
aggressive oracle from decentering an unsolved iterate.
"""

from __future__ import annotations

import pytest

from ipax import Options, Status, solve
from ipax.ipm.barrier import (
    FreeModeMonitor,
    adaptive_mu,
    breedveld_mu,
    complementarity_measures,
    fallback_mu,
)
from ipax.options import BarrierOptions, MuSchedule
from ipax.testing.problems import HS35, BoundConstrainedQP
from tests._helpers import array, assert_allclose, assert_scalar_close

_TOL = 1e-8  # kkt tolerance handed to the schedule; floor is tol/10 = 1e-9


# -- adaptive (LOQO) rule ---------------------------------------------------


def test_adaptive_mu_matches_loqo_formula():
    # ξ = 2.5e-3 / 1e-2 = 0.25 ⇒ σ = 0.1·(0.05·0.75/0.25)³ = 0.1·0.15³
    # (Nocedal, Wächter & Waltz 2009, eq. (3.6))
    mu = adaptive_mu(1e-2, 2.5e-3, BarrierOptions(), _TOL)
    assert_scalar_close(mu, 0.1 * 0.15**3 * 1e-2)


def test_adaptive_mu_perfectly_centered_hits_floor():
    # All products equal the average ⇒ ξ = 1 ⇒ σ = 0 ⇒ μ = max(μ_min, tol/10).
    mu = adaptive_mu(1e-2, 1e-2, BarrierOptions(), _TOL)
    assert_scalar_close(mu, 1e-9)


def test_adaptive_mu_sigma_capped_at_0_8_for_uncentered_iterates():
    # ξ → 0 clips the argument at 2 ⇒ σ = 0.1·2³ = 0.8; μ never exceeds
    # 0.8× the current average complementarity (NWW 2009, after eq. (3.6)).
    mu = adaptive_mu(1e-2, 0.0, BarrierOptions(), _TOL)
    assert_scalar_close(mu, 0.8 * 1e-2)
    mu = adaptive_mu(1.0, 1e-2, BarrierOptions(), _TOL)  # 0.05·99/1 > 2 ⇒ clip
    assert_scalar_close(mu, 0.8)


def test_adaptive_mu_zero_gap_hits_floor():
    mu = adaptive_mu(0.0, 0.0, BarrierOptions(), _TOL)
    assert_scalar_close(mu, 1e-9)


# -- breedveld rule ---------------------------------------------------------


def test_breedveld_mu_matches_eq_12():
    # σ = ((α−1)/(α+10))² (Breedveld 2017, eq. (12)); α = 0.5 ⇒ σ = (1/21)².
    mu = breedveld_mu(2.0, 0.5, BarrierOptions(), _TOL)
    assert_scalar_close(mu, ((0.5 - 1.0) / (0.5 + 10.0)) ** 2 * 2.0)


def test_breedveld_mu_full_step_hits_floor():
    # α = 1 ⇒ σ = 0: a full step drives μ straight to the floor.
    mu = breedveld_mu(1e-2, 1.0, BarrierOptions(), _TOL)
    assert_scalar_close(mu, 1e-9)


def test_breedveld_mu_rejected_step_recentered():
    # α = 0 ⇒ σ = (1/10)² = 0.01 of the current duality gap.
    mu = breedveld_mu(1e-2, 0.0, BarrierOptions(), _TOL)
    assert_scalar_close(mu, 1e-4)


# -- complementarity measures ----------------------------------------------


def test_complementarity_measures_average_and_min(namespace):
    xp = namespace
    # 2 slack pairs (0.5, 0.5), active-lower pairs (0.2, 0.2), active-upper
    # pair (0.4); masked-out entries carry junk that must not leak in.
    avg, mn = complementarity_measures(
        s=array(xp, [1.0, 2.0]),
        y_ineq=array(xp, [0.5, 0.25]),
        z_lower=array(xp, [2.0, 1.0, 5.0]),
        z_upper=array(xp, [9.0, 9.0, 0.8]),
        x_minus_l=array(xp, [0.1, 0.2, 7.0]),
        u_minus_x=array(xp, [3.0, 4.0, 0.5]),
        mask_l=xp.asarray([True, True, False]),
        mask_u=xp.asarray([False, False, True]),
        m=2,
        n_bounds=3,
    )
    assert_scalar_close(avg, (0.5 + 0.5 + 0.2 + 0.2 + 0.4) / 5.0)
    assert_scalar_close(mn, 0.2)


def test_complementarity_measures_bounds_only(namespace):
    xp = namespace
    avg, mn = complementarity_measures(
        s=array(xp, []),
        y_ineq=array(xp, []),
        z_lower=array(xp, [2.0, 1.0, 5.0]),
        z_upper=array(xp, [9.0, 9.0, 0.8]),
        x_minus_l=array(xp, [0.1, 0.2, 7.0]),
        u_minus_x=array(xp, [3.0, 4.0, 0.5]),
        mask_l=xp.asarray([True, True, False]),
        mask_u=xp.asarray([False, False, True]),
        m=0,
        n_bounds=3,
    )
    assert_scalar_close(avg, (0.2 + 0.2 + 0.4) / 3.0)
    assert_scalar_close(mn, 0.2)


def test_complementarity_measures_requires_pairs(namespace):
    xp = namespace
    with pytest.raises(ValueError):
        complementarity_measures(
            s=array(xp, []),
            y_ineq=array(xp, []),
            z_lower=array(xp, [0.0]),
            z_upper=array(xp, [0.0]),
            x_minus_l=array(xp, [1.0]),
            u_minus_x=array(xp, [1.0]),
            mask_l=xp.asarray([False]),
            mask_u=xp.asarray([False]),
            m=0,
            n_bounds=0,
        )


# -- driver wiring ----------------------------------------------------------


def _solve_hs35(namespace, schedule: MuSchedule, corrections: str = "none"):
    problem = HS35(namespace)
    return problem, solve(
        problem,
        array(namespace, [0.5, 0.5, 0.5]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            mu_schedule=schedule,
            corrections=corrections,
        ),
    )


def _mu_trace(result) -> tuple[float, ...]:
    return tuple(record.mu for record in result.history)


@pytest.mark.parametrize("schedule", ["adaptive", "breedveld", "quality"])
def test_schedule_solves_hs35_and_changes_mu_trace(namespace, schedule):
    problem, result = _solve_hs35(namespace, schedule)
    _, monotone = _solve_hs35(namespace, "monotone")

    assert result.status is Status.OPTIMAL
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    # The schedule must actually steer μ: a silently-ignored option reproduces
    # the monotone trace exactly.
    assert _mu_trace(result) != _mu_trace(monotone)


@pytest.mark.parametrize("schedule", ["adaptive", "breedveld", "quality"])
def test_schedule_solves_bound_qp(namespace, schedule):
    problem = BoundConstrainedQP(namespace)
    result = solve(
        problem,
        array(namespace, [0.25, 0.75]),
        options=Options(hessian="exact", linsolve="dense", mu_schedule=schedule),
    )
    assert result.status is Status.OPTIMAL
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)


def test_breedveld_schedule_keeps_mu_until_first_step(namespace):
    # σ(α) needs a previously *accepted* steplength (Breedveld 2017, eq. (12));
    # before the first step the schedule must leave μ at μ_init rather than
    # consume the α = 1 sentinel (which would crash μ to the floor at once).
    _, result = _solve_hs35(namespace, "breedveld")
    assert len(result.history) >= 3
    assert result.history[1].mu == pytest.approx(BarrierOptions().mu_init)
    assert result.history[2].mu != pytest.approx(BarrierOptions().mu_init)


# -- μ oracle × corrector orthogonality --------------------------------------


def test_probing_standalone_solves_and_changes_mu_trace(namespace):
    # NWW 2009 "Mehrotra probing" as a plain strategy: the affine solve is only
    # a σ probe; the step itself is the ordinary centered Newton direction.
    problem, result = _solve_hs35(namespace, "probing")
    _, monotone = _solve_hs35(namespace, "monotone")

    assert result.status is Status.OPTIMAL
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert _mu_trace(result) != _mu_trace(monotone)


def test_quality_standalone_solves_and_changes_mu_trace(namespace):
    # NWW 2009 §3.3 quality function as a plain strategy: like probing it needs
    # the affine direction, so the standalone mode runs the corrector path with
    # a plain centered re-solve; σ then minimizes the predicted full-KKT model.
    problem, result = _solve_hs35(namespace, "quality")
    _, monotone = _solve_hs35(namespace, "monotone")

    assert result.status is Status.OPTIMAL
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert _mu_trace(result) != _mu_trace(monotone)


def test_default_schedule_is_monotone(namespace):
    # Default decided by the S2MPJ v10 paired A/B (2026-07-08): monotone beat
    # probing on every config, 3837 vs 3770 correct in total — probing crashes
    # μ faster than the duals can follow on enough problems (tail stalls at
    # kkt ~1e-4). The non-monotone oracles remain explicit opt-ins.
    assert Options().mu_schedule == "monotone"

    problem = HS35(namespace)
    x0 = array(namespace, [0.5, 0.5, 0.5])
    default = solve(problem, x0, options=Options(hessian="exact", linsolve="dense"))
    _, monotone = _solve_hs35(namespace, "monotone")
    assert _mu_trace(default) == _mu_trace(monotone)

    default_corr = solve(
        problem,
        x0,
        options=Options(hessian="exact", linsolve="dense", corrections="mehrotra"),
    )
    _, monotone_corr = _solve_hs35(namespace, "monotone", corrections="mehrotra")
    assert _mu_trace(default_corr) == _mu_trace(monotone_corr)


@pytest.mark.parametrize("corrections", ["mehrotra", "gondzio"])
def test_adaptive_oracle_steers_the_corrector(namespace, corrections):
    # The corrector consumes the oracle's μ target (NWW 2009: "the corrector is
    # not part of the selection of the barrier parameter"): choosing the LOQO
    # oracle with corrections active must change the μ trace vs. probing.
    problem, result = _solve_hs35(namespace, "adaptive", corrections=corrections)
    _, probing = _solve_hs35(namespace, "probing", corrections=corrections)

    assert result.status is Status.OPTIMAL
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert _mu_trace(result) != _mu_trace(probing)


def test_breedveld_oracle_with_corrector_solves(namespace):
    problem, result = _solve_hs35(namespace, "breedveld", corrections="gondzio")
    assert result.status is Status.OPTIMAL
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)


# -- free-mode KKT-error fallback (NWW 2009, §5.1, Algorithm A) ---------------


def test_monitor_stays_free_on_decreasing_errors():
    monitor = FreeModeMonitor(BarrierOptions())
    for error in (1.0, 0.5, 0.25, 0.125):
        free, entered = monitor.observe(error)
        assert free
        assert not entered


def test_monitor_enters_monotone_on_stalled_error_and_resumes():
    # κ = 0.9999, l_max = 5 (paper defaults): an error above κ·M_k trips the
    # switch; the free mode resumes once the error drops back below κ·M_k.
    monitor = FreeModeMonitor(BarrierOptions())
    assert monitor.observe(1.0) == (True, False)
    free, entered = monitor.observe(1.5)  # 1.5 > 0.9999·1.0 ⇒ monotone
    assert not free
    assert entered
    assert monitor.observe(1.2) == (False, False)  # still above κ·M_switch
    assert monitor.observe(0.9) == (True, False)  # ≤ 0.9999·1.0 ⇒ free again


def test_monitor_measures_against_max_of_recent_errors():
    # M_k = max{Φ_{k−l}, …, Φ_k} with l = min(k, l_max): a rebound below the
    # window's max is still acceptable free-mode progress.
    monitor = FreeModeMonitor(BarrierOptions(fallback_window=1))
    assert monitor.observe(1.0) == (True, False)
    assert monitor.observe(0.5) == (True, False)
    # window (l_max = 1) still holds 1.0, so 0.9 ≤ κ·1.0 stays free ...
    assert monitor.observe(0.9) == (True, False)
    # ... but now the window is (0.5, 0.9), and 0.95 > κ·0.9 trips the switch.
    assert monitor.observe(0.95) == (False, True)


def test_monitor_disabled_never_switches():
    monitor = FreeModeMonitor(BarrierOptions(fallback="never"))
    for error in (1.0, 2.0, 4.0, 8.0):
        assert monitor.observe(error) == (True, False)


def test_fallback_mu_reinitializes_from_average_complementarity():
    # Monotone-mode re-entry μ = 0.8·(average complementarity) (NWW 2009 §5.1).
    assert_scalar_close(fallback_mu(2.0, BarrierOptions(), _TOL), 0.8 * 2.0)
    # ... floored like every schedule at max(μ_min, tol/10).
    assert_scalar_close(fallback_mu(0.0, BarrierOptions(), _TOL), 1e-9)


def test_fallback_mu_respects_centrality_floor():
    # The RT-fluence pin: a frozen primal with free-running duals collapses the
    # complementarity far below the true KKT error, so the fallback's
    # complementarity-based re-entry μ landed at the ε/10 floor and stayed
    # there for hundreds of iterations while the dual residual sat at ~1e-4.
    # The re-entry value must respect the same El-Bakry centrality floor the
    # free-mode oracles already have: μ ≥ κ_cent·(primal/dual infeasibility).
    opts = BarrierOptions()  # kappa_centrality = 1e-2
    mu = fallback_mu(1e-9, opts, _TOL, infeasibility=1e-4)
    assert_scalar_close(mu, 1e-2 * 1e-4)


def test_fallback_mu_floor_never_lowers_the_reentry_value():
    # A healthy complementarity dominates: the floor is a max, not a target.
    opts = BarrierOptions()
    mu = fallback_mu(2.0, opts, _TOL, infeasibility=1e-4)
    assert_scalar_close(mu, 0.8 * 2.0)


def test_fallback_mu_infeasibility_defaults_to_inactive():
    # Callers that pass no infeasibility keep the pure NWW re-entry value.
    assert_scalar_close(fallback_mu(2.0, BarrierOptions(), _TOL), 1.6)


def test_fallback_mu_floor_disabled_by_zero_kappa():
    # κ_cent = 0 disables the floor (same semantics as the free-mode paths).
    opts = BarrierOptions(kappa_centrality=0.0)
    mu = fallback_mu(0.0, opts, _TOL, infeasibility=1e-4)
    assert_scalar_close(mu, 1e-9)


def test_fallback_options_validation():
    with pytest.raises(ValueError, match="fallback_kappa"):
        BarrierOptions(fallback_kappa=1.0)
    with pytest.raises(ValueError, match="fallback_window"):
        BarrierOptions(fallback_window=-1)
    with pytest.raises(ValueError, match="fallback_mu_factor"):
        BarrierOptions(fallback_mu_factor=0.0)
    with pytest.raises(ValueError, match="kappa_centrality"):
        BarrierOptions(kappa_centrality=-1.0)


# -- centrality floor (El-Bakry et al. 1996 centrality condition) -------------


def test_centrality_floor_steers_free_mode_mu(namespace):
    # μ_free ≥ κ_cent·max(dual, primal infeasibility): a large κ visibly lifts
    # the μ trace on a problem the oracle would otherwise decenter, without
    # costing convergence.
    problem = HS35(namespace)
    x0 = array(namespace, [0.5, 0.5, 0.5])

    def run(barrier: BarrierOptions):
        return solve(
            problem,
            x0,
            options=Options(
                hessian="exact",
                linsolve="dense",
                mu_schedule="adaptive",
                barrier=barrier,
            ),
        )

    lifted = run(BarrierOptions(kappa_centrality=0.5))
    default = run(BarrierOptions())

    assert lifted.status is Status.OPTIMAL
    assert default.status is Status.OPTIMAL
    assert _mu_trace(lifted) != _mu_trace(default)


def test_forced_fallback_steers_mu_and_still_converges(namespace):
    # κ = 0.01 with a length-0 window demands a 100× KKT-error reduction every
    # free-mode iteration — practically guaranteed to trip, so the run must
    # fall back to monotone μ handling and still reach the optimum.
    problem = HS35(namespace)
    x0 = array(namespace, [0.5, 0.5, 0.5])

    def run(barrier: BarrierOptions):
        return solve(
            problem,
            x0,
            options=Options(
                hessian="exact",
                linsolve="dense",
                mu_schedule="adaptive",
                barrier=barrier,
            ),
        )

    forced = run(BarrierOptions(fallback_kappa=0.01, fallback_window=0))
    pure_free = run(BarrierOptions(fallback="never"))

    assert forced.status is Status.OPTIMAL
    assert pure_free.status is Status.OPTIMAL
    assert_allclose(namespace, forced.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert _mu_trace(forced) != _mu_trace(pure_free)


def test_driver_passes_infeasibility_to_fallback_mu(namespace, monkeypatch):
    # Wiring proof for the fallback-path centrality floor: when the safeguard
    # trips, the driver must hand the current primal/dual infeasibility to
    # ``fallback_mu`` so the re-entry μ respects the El-Bakry floor. (The
    # formula itself is pinned by the fallback_mu unit tests above; a trace
    # comparison cannot discriminate this path because the free-mode oracles
    # share the same floor.)
    import ipax.ipm.driver as driver_mod

    seen: dict[str, float] = {}
    original = driver_mod.fallback_mu

    def recording(avg_compl, options, tol, infeasibility=0.0):
        seen["infeasibility"] = float(infeasibility)
        return original(avg_compl, options, tol)

    monkeypatch.setattr(driver_mod, "fallback_mu", recording)

    problem = HS35(namespace)
    result = solve(
        problem,
        array(namespace, [0.5, 0.5, 0.5]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            mu_schedule="adaptive",
            # Forced trip: κ = 0.01 with a length-0 window demands a 100× KKT
            # error reduction every free-mode iteration (see above).
            barrier=BarrierOptions(fallback_kappa=0.01, fallback_window=0),
        ),
    )

    assert result.status is Status.OPTIMAL
    assert "infeasibility" in seen, "the forced fallback never tripped"
    assert seen["infeasibility"] > 0.0
