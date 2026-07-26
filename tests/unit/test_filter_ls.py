"""Unit tests for filter accept/reject behavior."""

from __future__ import annotations

import logging

from ipax._logging import LOGGER_NAME
from ipax.ipm.filter_ls import Filter, FilterLineSearch
from ipax.options import LineSearchOptions
from tests._helpers import implemented


def test_reject_reason_classifies_each_gate():
    # The rejection classifier names the first failing acceptance gate; ``None``
    # means the trial is acceptable. This is the label the per-trial debug trace
    # surfaces so a heavy-backtracking iteration can be diagnosed from the log.
    ls = FilterLineSearch(LineSearchOptions())
    # Args: (θ_t, φ_t, θ0, φ0, dphi, α, θ_max, θ_min, entries). Every case sits
    # at θ0 = 1.0 with θ_min = 1e10, i.e. below θ_min, so the f-type branch is
    # reachable and each gate is exercised as intended; the θ_min gate itself is
    # covered in tests/unit/test_theta_min_switching.py.

    # non-finite θ/φ (here φ = -inf on an f-type step).
    assert ls._reject_reason(
        0.5, float("-inf"), 1.0, 1.0, -1e6, 1.0, 1e10, 1e10, []
    ) == ("non-finite")
    # θ past the eq. (18) guard θ_max.
    assert ls._reject_reason(1e30, 1.0, 1.0, 1.0, -1e6, 1.0, 1e4, 1e10, []) == (
        "theta-max"
    )
    # dominated by a filter entry.
    assert ls._reject_reason(
        2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1e10, 1e10, [(1.0, 1.0)]
    ) == ("filter")
    # f-type step (switching holds, θ0 ≤ θ_min) failing the Armijo decrease.
    assert ls._reject_reason(0.5, 5.0, 1.0, 1.0, -1e6, 1.0, 1e10, 1e10, []) == "armijo"
    # θ-type step (switching fails) with neither θ- nor φ-progress.
    assert (
        ls._reject_reason(2.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1e10, 1e10, []) == "no-decrease"
    )
    # acceptable ⇒ None.
    assert ls._reject_reason(0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1e10, 1e10, []) is None


def test_search_emits_per_trial_debug_trace(caplog):
    # At DEBUG level the search logs each backtracking trial with its α, θ, φ and
    # the reason it was rejected (or "accept"), so 10+-trial iterations can be
    # read directly from the log without a rerun.
    line_search = FilterLineSearch(LineSearchOptions())
    # f-type ray: Armijo bound is 1 - 100α; φ = 5 fails until α shrinks enough
    # that φ drops to -200 and the step is accepted.
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        result = line_search.search(
            alpha_max=1.0,
            theta0=1.0,
            phi0=1.0,
            dphi=-1e6,
            theta_max=1e10,
            theta_min=1e10,
            eval_point=lambda alpha: (0.5, 5.0) if alpha > 0.02 else (0.5, -200.0),
            entries=[],
            soc=None,
        )

    assert result.accepted
    msgs = [r.getMessage() for r in caplog.records if "ls trial" in r.getMessage()]
    assert len(msgs) >= 2  # several backtracks were traced
    assert any("armijo" in m for m in msgs)  # rejected trials are labelled
    assert any("accept" in m for m in msgs)  # the accepted trial is labelled
    assert all(
        "alpha=" in m and "theta=" in m and "phi=" in m for m in msgs
    )  # each line carries the trial quantities


def test_search_traces_soc_trials(caplog):
    # A second-order-correction trial is labelled soc-* in the trace, so a
    # heavy-backtracking log distinguishes SOC attempts from plain trials.
    line_search = FilterLineSearch(LineSearchOptions())
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        result = line_search.search(
            alpha_max=1.0,
            theta0=1.0,
            phi0=1.0,
            dphi=1.0,
            theta_max=1e10,
            theta_min=1e10,
            eval_point=lambda alpha: (2.0, 2.0),  # θ rises ⇒ SOC is attempted
            entries=[],
            soc=lambda alpha: (0.1, 2.0),  # ... and the corrected trial is good
        )

    assert result.accepted and result.used_soc
    msgs = [r.getMessage() for r in caplog.records if "ls trial" in r.getMessage()]
    assert any("soc-accept" in m for m in msgs)


def test_search_traces_kkt_progress_rescue(caplog):
    # The rescue's acceptance is labelled distinctly from a plain accept — it
    # is the one path that overrides an Armijo rejection.
    line_search = FilterLineSearch(LineSearchOptions())
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        result = line_search.search(
            alpha_max=1.0,
            theta0=0.0,
            phi0=1.0,
            dphi=-1e-6,
            theta_max=1e4,
            theta_min=1e10,
            eval_point=lambda alpha: (0.0, 1.0 + 0.1 * alpha),  # Armijo fails
            entries=[],
            kkt_progress=lambda alpha: True,
        )

    assert result.accepted
    msgs = [r.getMessage() for r in caplog.records if "ls trial" in r.getMessage()]
    assert any("kkt-progress" in m for m in msgs)


def test_empty_filter_accepts_candidate():
    filt = Filter()
    with implemented("filter"):
        assert filt.is_acceptable(theta=1.0, phi=10.0)


def test_filter_rejects_candidate_dominated_in_both_measures():
    filt = Filter(entries=[(1.0, 10.0)])
    with implemented("filter"):
        assert not filt.is_acceptable(theta=1.1, phi=10.5)


def test_filter_accepts_candidate_that_improves_one_measure():
    filt = Filter(entries=[(1.0, 10.0)])
    with implemented("filter"):
        assert filt.is_acceptable(theta=0.5, phi=11.0)
        assert filt.is_acceptable(theta=1.1, phi=9.0)


def test_filter_augment_removes_entries_dominated_by_new_pair():
    filt = Filter(entries=[(1.0, 10.0), (0.25, 20.0)])
    with implemented("filter"):
        filt.augment(theta=0.5, phi=8.0)

    assert (1.0, 10.0) not in filt.entries
    assert (0.25, 20.0) in filt.entries
    assert (0.5, 8.0) in filt.entries


def test_filter_clear_empties_the_entries():
    # The filter's φ coordinates are φ_μ values — meaningful only for the μ they
    # were recorded at. The driver re-initializes the filter whenever μ changes
    # (W&B 2006 filter re-initialization on a barrier update; IPOPT's
    # ``FilterLSAcceptor::Reset`` via ``MonotoneMuUpdate``), so ``clear`` must
    # restore the empty (everything-acceptable) state.
    filt = Filter(entries=[(1.0, 10.0), (0.25, 20.0)])
    filt.clear()

    assert filt.entries == []
    assert filt.is_acceptable(theta=1e6, phi=1e6)


def test_ascent_step_rejected_at_feasible_point():
    # Regression: at a feasible iterate (θ0 = 0) the θ-progress branch
    # degenerates to "0 ≤ 0" and accepted *any* feasible trial — including a
    # corrector direction with dφ > 0 that inflated φ by orders of magnitude
    # (RT least-squares example). W&B §2.3: no θ-type step exists at θ = 0;
    # only Armijo/φ-decrease acceptance applies.
    line_search = FilterLineSearch(LineSearchOptions())

    result = line_search.search(
        alpha_max=1.0,
        theta0=0.0,
        phi0=1.0,
        dphi=5.0,  # ascent direction: switching can never hold
        theta_max=1e4,
        theta_min=1e10,
        eval_point=lambda alpha: (0.0, 1.0 + 10.0 * alpha),
        entries=[],
    )

    assert not result.accepted
    assert result.restoration


def test_phi_decrease_still_accepted_at_feasible_point():
    # The φ sub-branch (φ_t ≤ φ0 − γ_φ·θ0, i.e. non-increase at θ0 = 0) must
    # survive the θ-branch fix so re-centering steps remain acceptable.
    line_search = FilterLineSearch(LineSearchOptions())

    result = line_search.search(
        alpha_max=1.0,
        theta0=0.0,
        phi0=1.0,
        dphi=5.0,
        theta_max=1e4,
        theta_min=1e10,
        eval_point=lambda alpha: (0.0, 0.5),
        entries=[],
    )

    assert result.accepted
    assert result.alpha == 1.0
    assert result.n_trials == 1


def test_switching_condition_survives_overflowing_directional_derivative():
    # Regression (INDEF): a badly-scaled iterate yields an enormous |dphi|; the
    # switching condition's ``(-dphi) ** s_phi`` must not raise OverflowError.
    line_search = FilterLineSearch(LineSearchOptions())

    result = line_search.search(
        alpha_max=1.0,
        theta0=1.0,
        phi0=1.0,
        dphi=-1e308,  # would overflow float ** s_phi before the fix
        theta_max=1e10,
        theta_min=1e10,
        eval_point=lambda alpha: (0.5, 0.5),
        entries=[],
        soc=None,
    )

    # No crash; the f-type Armijo test cannot be met for such a step, so the
    # search exhausts α and hands off to restoration.
    assert result.restoration
    assert not result.accepted


def test_line_search_reports_accepted_soc_trial():
    line_search = FilterLineSearch(LineSearchOptions())

    result = line_search.search(
        alpha_max=1.0,
        theta0=1.0,
        phi0=1.0,
        dphi=1.0,
        theta_max=1e10,
        theta_min=1e10,
        eval_point=lambda alpha: (2.0, 2.0),
        entries=[],
        soc=lambda alpha: (0.1, 2.0),
    )

    assert result.accepted
    assert result.used_soc
    assert not result.restoration
    assert result.n_trials == 1


def test_theta_max_guard_rejects_exploding_infeasibility():
    # Regression (HS7, S2MPJ Task 1): an f-type step whose barrier objective φ
    # collapses toward -∞ must NOT be accepted while the constraint violation θ
    # explodes past the Wächter & Biegler 2006 eq. (18) guard θ_max. Before the
    # guard, the switching + Armijo branch let such a step through and the solver
    # diverged to a false "infeasible".
    line_search = FilterLineSearch(LineSearchOptions())

    result = line_search.search(
        alpha_max=1.0,
        theta0=10.0,
        phi0=1.0,
        dphi=-1e6,  # switching condition holds -> f-type branch
        theta_max=1e4,
        theta_min=1e10,
        eval_point=lambda alpha: (1e30, -1e30),  # huge θ, collapsing φ
        entries=[],
        soc=None,
    )

    assert not result.accepted
    assert result.restoration


def test_theta_max_guard_rejects_non_finite_theta():
    line_search = FilterLineSearch(LineSearchOptions())

    result = line_search.search(
        alpha_max=1.0,
        theta0=10.0,
        phi0=1.0,
        dphi=-1e6,
        theta_max=1e4,
        theta_min=1e10,
        eval_point=lambda alpha: (float("inf"), -1e30),
        entries=[],
        soc=None,
    )

    assert not result.accepted
    assert result.restoration


def test_accept_rejects_non_finite_phi():
    # Regression (S2MPJ Task 2, BRATU1D/LUKVLE8): a step that overshoots into a
    # region where the objective evaluates to -∞ (finite θ) must be rejected, not
    # accepted. ``φ_t = -∞`` is trivially below the Armijo bound, so before the
    # φ-finiteness guard the f-type branch let such a full step through instead of
    # backtracking to a finite, usable iterate.
    line_search = FilterLineSearch(LineSearchOptions())

    result = line_search.search(
        alpha_max=1.0,
        theta0=1.0,
        phi0=1.0,
        dphi=-1e6,  # switching condition holds -> f-type branch
        theta_max=1e10,  # θ is fine; only φ is non-finite
        theta_min=1e10,
        eval_point=lambda alpha: (0.5, float("-inf")),
        entries=[],
        soc=None,
    )

    assert not result.accepted
    assert result.restoration


def test_search_backtracks_past_non_finite_gradient_region():
    # Regression (S2MPJ Task 2, RAT42LS): a step whose θ/φ are finite but whose
    # gradient overflows must be rejected so the search backtracks to a damped
    # step in the finite region, instead of accepting it and poisoning the next
    # KKT solve. Here θ decreases (θ-type acceptable) at every α, but the gradient
    # is only finite for α ≤ 0.3, so acceptance lands on α = 0.25.
    line_search = FilterLineSearch(LineSearchOptions())

    result = line_search.search(
        alpha_max=1.0,
        theta0=1.0,
        phi0=1.0,
        dphi=-1.0,
        theta_max=1e10,
        theta_min=1e10,
        eval_point=lambda alpha: (0.5, 0.5),  # always filter-acceptable
        entries=[],
        soc=None,
        grad_finite=lambda alpha: alpha <= 0.3,
    )

    assert result.accepted
    assert result.alpha == 0.25
    # Trials at alpha = 1.0, 0.5, 0.25: the first two rejected on a non-finite
    # gradient, the third accepted.
    assert result.n_trials == 3


def test_kkt_progress_rescues_first_trial_at_feasible_point():
    # At θ0 = 0 the W&B eq. (19) switching condition holds for every descent
    # direction (RHS = 0), so every trial faces the full Armijo test — the gate
    # that ground the RT fluence case to 11–27 backtracks per iteration while
    # IPOPT (hovering at θ > 0) accepted first trials through the near-vacuous
    # θ-type branch. The rescue: a *first* trial that fails Armijo is still
    # accepted when the caller certifies sufficient scaled-KKT-error decrease —
    # at a feasible iterate, optimality progress IS progress.
    ls = FilterLineSearch(LineSearchOptions())
    seen: list[float] = []

    def kkt_progress(alpha):
        seen.append(alpha)
        return True

    result = ls.search(
        alpha_max=1.0,
        theta0=0.0,
        phi0=1.0,
        dphi=-1e-6,  # descent ⇒ switching holds ⇒ f-type Armijo
        theta_max=1e4,
        theta_min=1e10,
        eval_point=lambda alpha: (0.0, 1.0 + 0.1 * alpha),  # φ rises ⇒ Armijo fails
        entries=[],
        kkt_progress=kkt_progress,
    )

    assert result.accepted
    assert result.alpha == 1.0
    assert result.n_trials == 1
    assert not result.augment  # f-type-like: the filter is not augmented
    assert seen == [1.0]


def test_kkt_progress_consulted_only_on_the_first_trial():
    # The certificate costs a gradient/Jacobian evaluation, so it is a
    # first-trial rescue only (mirroring SOC); rejected ⇒ plain backtracking.
    ls = FilterLineSearch(LineSearchOptions())
    seen: list[float] = []

    def kkt_progress(alpha):
        seen.append(alpha)
        return False

    result = ls.search(
        alpha_max=1.0,
        theta0=0.0,
        phi0=1.0,
        dphi=-1e-6,
        theta_max=1e4,
        theta_min=1e10,
        # φ rises along the ray: Armijo keeps failing, α exhausts.
        eval_point=lambda alpha: (0.0, 1.0 + 0.1 * alpha),
        entries=[],
        kkt_progress=kkt_progress,
    )

    assert not result.accepted
    assert result.restoration
    assert seen == [1.0]  # not retried on the backtracked trials


def test_kkt_progress_never_rescues_ascent_directions():
    # The feasible ascent-stall guard must be untouched: an ascent direction
    # (dφ > 0) at a feasible point fails the switching condition and takes the
    # θ-branch — its rejection reason is not "armijo", so the certifier is
    # never consulted and the φ-inflating trial stays rejected.
    ls = FilterLineSearch(LineSearchOptions())
    seen: list[float] = []

    def kkt_progress(alpha):
        seen.append(alpha)
        return True

    result = ls.search(
        alpha_max=1.0,
        theta0=0.0,
        phi0=1.0,
        dphi=5.0,  # ascent: switching can never hold
        theta_max=1e4,
        theta_min=1e10,
        eval_point=lambda alpha: (0.0, 1.0 + 10.0 * alpha),  # φ inflates
        entries=[],
        kkt_progress=kkt_progress,
    )

    assert not result.accepted
    assert result.restoration
    assert seen == []


def test_search_hands_off_to_restoration_when_gradient_never_finite():
    # If no α on the ray yields a finite gradient, the search exhausts α and
    # hands off to restoration — the same fallback as an unacceptable θ/φ.
    line_search = FilterLineSearch(LineSearchOptions())

    result = line_search.search(
        alpha_max=1.0,
        theta0=1.0,
        phi0=1.0,
        dphi=-1.0,
        theta_max=1e10,
        theta_min=1e10,
        eval_point=lambda alpha: (0.5, 0.5),
        entries=[],
        soc=None,
        grad_finite=lambda alpha: False,
    )

    assert not result.accepted
    assert result.restoration
    # Every halving of alpha down to alpha_min_frac counts as one trial.
    assert result.n_trials > 1
