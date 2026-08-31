# Copyright 2026 Niklas Wahl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Filter line-search globalization — default (Wächter & Biegler §2–3, §4.1).

- Filter ``F`` of ``(θ, φ_μ)`` pairs, ``θ(x) = ‖(c, g+s)‖₁``.
- Acceptance by sufficient decrease in ``θ`` **or** barrier objective ``φ_μ``;
  switching condition + Armijo for f-type steps.
- Second-order correction (SOC) when a full step increases ``θ``.
- Hands off to the feasibility restoration phase when no ``α ≥ α_min`` is
  acceptable.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ipax._logging import logger

if TYPE_CHECKING:
    from ipax.options import LineSearchOptions

# Switching-condition constant δ (Wächter & Biegler 2006, eq. 19).
_SWITCH_DELTA = 1.0


def _safe_pow(base: float, exponent: float) -> float:
    """``base ** exponent`` with IEEE overflow semantics (→ ``inf``, never raise).

    Python's ``float.__pow__`` raises ``OverflowError`` instead of returning
    ``inf`` when the result exceeds the double range, which crashes the switching
    test on a badly-scaled iterate (e.g. an enormous directional derivative
    ``dphi``). Treat such an overflow as ``+inf`` so the comparison still has a
    meaningful, finite-safe answer.
    """
    try:
        return float(base**exponent)
    except OverflowError:
        return float("inf")


@dataclass(slots=True)
class Filter:
    """The ``(θ, φ)`` filter set."""

    entries: list[tuple[float, float]] = field(default_factory=list)

    def is_acceptable(self, theta: float, phi: float) -> bool:
        """Pareto non-domination: not dominated by any existing entry."""
        return all(theta < tj or phi < fj for tj, fj in self.entries)

    def augment(self, theta: float, phi: float) -> None:
        """Add ``(θ, φ)`` and drop entries it dominates."""
        self.entries = [
            (tj, fj) for tj, fj in self.entries if not (theta <= tj and phi <= fj)
        ]
        self.entries.append((theta, phi))

    def clear(self) -> None:
        """Re-initialize to the empty (everything-acceptable) filter.

        The φ coordinates are barrier objectives φ_μ — meaningful only for the
        μ they were recorded at — so the filter history must be discarded
        whenever the barrier parameter changes (W&B 2006: the filter is
        re-initialized to eq. (18) at every barrier update; IPOPT
        ``FilterLSAcceptor::Reset``). The eq. (18) guard region {θ ≥ θ_max}
        survives a reset by construction: ipax keeps it as the separate
        ``theta_max`` argument, not as entries.
        """
        self.entries.clear()


@dataclass(frozen=True, slots=True)
class LineSearchResult:
    """Outcome of one filter line search."""

    alpha: float
    accepted: bool
    augment: bool  # θ-type accepted steps augment the filter
    restoration: bool
    used_soc: bool = False
    n_trials: int = 1  # number of backtracking trial step sizes evaluated


class FilterLineSearch:
    """Backtracking filter line search with optional SOC."""

    def __init__(self, options: LineSearchOptions) -> None:
        self._o = options

    def _filter_acceptable(
        self,
        theta_t: float,
        phi_t: float,
        entries: list[tuple[float, float]],
    ) -> bool:
        """Sufficient decrease w.r.t. every filter entry (W&B eq. 18)."""
        o = self._o
        for tj, fj in entries:
            if not (
                theta_t <= (1.0 - o.gamma_theta) * tj or phi_t <= fj - o.gamma_phi * tj
            ):
                return False
        return True

    def search(
        self,
        *,
        alpha_max: float,
        theta0: float,
        phi0: float,
        dphi: float,
        theta_max: float,
        theta_min: float,
        eval_point: Callable[[float], tuple[float, float]],
        entries: list[tuple[float, float]],
        soc: Callable[[float], tuple[float, float] | None] | None = None,
        grad_finite: Callable[[float], bool] | None = None,
        kkt_progress: Callable[[float], bool] | None = None,
    ) -> LineSearchResult:
        """Return the accepted ``(α, …)`` or signal restoration.

        ``eval_point(α)`` returns ``(θ, φ)`` at ``x + α d`` (and ``s + α ds``).
        ``soc(α)`` optionally returns ``(θ, φ)`` for a second-order-corrected
        trial when the full step increases θ (W&B §2.3, eq. 27). ``theta_max`` is
        the W&B eq. (18) guard: trials with ``θ ≥ θ_max`` (or non-finite θ) are
        never acceptable. ``theta_min`` is the eq. (23) constraint-violation
        threshold, which decides how far the search backtracks before conceding
        (see ``_alpha_min``).

        ``grad_finite(α)``, when supplied, reports whether the Lagrangian
        gradient at the trial point is finite. A step whose ``θ``/``φ`` are finite
        can still overshoot into a region where the *derivatives* overflow to
        inf/NaN (e.g. an exp/rational element function); the line search only
        evaluates ``θ``/``φ``, so such a point would be accepted and then poison
        the next KKT solve. Treating a non-finite-gradient trial as unacceptable
        keeps backtracking to a damped step that stays in the finite region,
        reusing the existing α-reduction (and restoration hand-off if the whole
        ray is bad).

        ``kkt_progress(α)``, when supplied, certifies that the trial at ``α``
        makes sufficient scaled-KKT-error progress. At an exactly feasible
        iterate (θ0 = 0) the switching condition (W&B eq. 19) holds for every
        descent direction — its right side is ``δ·θ0^{s_θ} = 0`` — so every
        trial faces the full Armijo test and there is no θ-type escape. The
        certifier rescues a *first* trial that fails Armijo: at a feasible
        iterate, KKT-error decrease is optimality progress even when the
        barrier objective wiggles up (the KKT-error-globalization philosophy
        of Nocedal, Wächter & Waltz 2009, §5.1). The caller supplies the
        certifier only at (numerically) feasible iterates — round-off keeps
        θ0 at ~1e-16 rather than exactly 0, and the switching condition
        degenerates there all the same. Consulted on the first trial only
        (the certificate costs a gradient/Jacobian evaluation), and only
        against an ``armijo`` rejection — never for an ascent direction
        (those fail switching and take the θ-branch, so the feasible
        ascent-stall guard is untouched). Accepted rescues are f-type-like:
        the filter is not augmented.
        """
        alpha = alpha_max
        alpha_min = self._alpha_min(dphi, theta0, theta_min)  # W&B eq. (23)
        first = True
        trials = 0
        trace = logger.isEnabledFor(logging.DEBUG)
        while alpha >= alpha_min:
            trials += 1
            theta_t, phi_t = eval_point(alpha)

            # SOC on the first (full-ish) trial that worsens feasibility.
            if first and soc is not None and theta_t > theta0:
                corrected = soc(alpha)
                if corrected is not None:
                    theta_c, phi_c = corrected
                    soc_reason = self._reject_reason(
                        theta_c,
                        phi_c,
                        theta0,
                        phi0,
                        dphi,
                        alpha,
                        theta_max,
                        theta_min,
                        entries,
                    )
                    if trace:
                        logger.debug(
                            "  ls trial %d: alpha=%.3e theta=%.3e phi=%.3e -> soc-%s",
                            trials,
                            alpha,
                            theta_c,
                            phi_c,
                            "accept" if soc_reason is None else soc_reason,
                        )
                    if soc_reason is None:
                        # The SOC point differs from ``x + α d``; its own gradient
                        # finiteness is checked inside ``soc`` (which returns None
                        # to reject a non-finite-derivative corrected trial).
                        # Step 5 is judged on the point actually taken — here the
                        # corrected one (IPOPT tests ``trial_barrier_obj()``).
                        augment = self._augments_filter(
                            phi_c, phi0, dphi, alpha, theta0
                        )
                        return LineSearchResult(
                            alpha, True, augment, False, True, trials
                        )
            first = False

            reason = self._reject_reason(
                theta_t, phi_t, theta0, phi0, dphi, alpha, theta_max, theta_min, entries
            )
            if reason is None and grad_finite is not None and not grad_finite(alpha):
                reason = "non-finite-grad"
            if (
                reason == "armijo"
                and trials == 1
                and kkt_progress is not None
                and kkt_progress(alpha)
            ):
                # Feasible-point rescue (see docstring): certified KKT-error
                # decrease stands in for the unreachable θ-type acceptance.
                if trace:
                    logger.debug(
                        "  ls trial %d: alpha=%.3e theta=%.3e phi=%.3e -> kkt-progress",
                        trials,
                        alpha,
                        theta_t,
                        phi_t,
                    )
                return LineSearchResult(alpha, True, False, False, n_trials=trials)
            if trace:
                logger.debug(
                    "  ls trial %d: alpha=%.3e theta=%.3e phi=%.3e -> %s",
                    trials,
                    alpha,
                    theta_t,
                    phi_t,
                    "accept" if reason is None else reason,
                )
            if reason is None:
                augment = self._augments_filter(phi_t, phi0, dphi, alpha, theta0)
                return LineSearchResult(alpha, True, augment, False, n_trials=trials)
            alpha = self._next_alpha(alpha, phi_t, phi0, dphi)
        return LineSearchResult(alpha_min, False, False, True, n_trials=trials)

    def _next_alpha(
        self, alpha: float, phi_t: float, phi0: float, dphi: float
    ) -> float:
        """The next trial step size after a rejection.

        Safeguarded quadratic interpolation (Nocedal & Wright 2006, eq. 3.58):
        the minimizer ``−φ'(0)·α² / (2(φ(α) − φ(0) − φ'(0)·α))`` of the
        quadratic through ``φ(0)``, ``φ'(0)`` and the rejected trial, clipped
        into ``[0.1·α, 0.5·α]``. The upper clip keeps every trial at least as
        short as W&B's plain halving; the lower clip stops one wild trial value
        (e.g. a barrier blow-up just inside the domain) from collapsing the
        step past what the model supports. Halving is the fallback whenever
        the model is unusable: non-finite ``φ(α)`` (trial outside the barrier
        domain), non-descent ``dφ`` (θ-type steps), or non-positive curvature
        along the ray. Note the interpolant drops below ``0.5·α`` only when
        ``φ(α) > φ(0)``: filter-/θ-driven rejections of a φ-decreasing trial
        still shrink by exactly the halving factor, so the eq. (23) hand-off
        skips a wider window only on rays the merit model already condemns.
        """
        o = self._o
        if not o.backtrack_interpolation:
            return 0.5 * alpha
        denominator = 2.0 * (phi_t - phi0 - dphi * alpha)
        if not (math.isfinite(denominator) and denominator > 0.0 and dphi < 0.0):
            return 0.5 * alpha
        alpha_q = -dphi * alpha * alpha / denominator
        return min(
            max(alpha_q, o.backtrack_shrink_min * alpha),
            o.backtrack_shrink_max * alpha,
        )

    def search_free(
        self,
        *,
        alpha_max: float,
        theta_max: float,
        eval_point: Callable[[float], tuple[float, float]],
        entries: list[tuple[float, float]],
        margin: float,
        grad_finite: Callable[[float], bool] | None = None,
    ) -> LineSearchResult:
        """Free-mode acceptance (NWW 2009, §5 globalization framework).

        Under a free-mode μ oracle the barrier problem changes every iteration,
        so the W&B filter/Armijo machinery is not a consistent per-trial merit
        gate ("the history in the filter [is] reset at every free iteration
        because the barrier problem itself changes" — NWW §5); global
        convergence is carried by the iterate-level KKT-error monitor
        (``FreeModeMonitor``), and the free-mode search should "interfere with
        adaptive steps as little as possible". The per-trial test is the §5
        obj-constr variant with IPOPT's margins
        (``AdaptiveMuUpdate::CheckSufficientProgress``): the trial is accepted
        when ``(θ_t + margin, f_t + margin)`` is acceptable to the filter of
        previous free iterates — ``eval_point(α)`` returns ``(θ, f)`` with the
        **raw objective** ``f``, which unlike φ_μ is comparable across μ
        re-targets. No switching/Armijo test applies; the θ_max guard,
        non-finite rejections, and the ``grad_finite`` overshoot check are
        safety invariants and stay. SOC and the feasible-point KKT rescue are
        rigorous-path mechanisms and are not consulted here.

        The caller owns the free filter and must remember each free iterate
        (driver-side ``augment``) *before* searching, so the current point is
        an entry — the margins are what keep this from degenerating to a
        monotone requirement. An empty ``entries`` list vacuously accepts any
        finite trial, so skipping that pre-augment would disable the merit
        test entirely. Acceptance never augments the W&B filter
        (``LineSearchResult.augment`` is always ``False``); a fully rejected
        ray hands off to restoration exactly like the rigorous search.
        """
        alpha = alpha_max
        # The eq. (23) α_min is defined through ∇φᵀd and the switching/Armijo
        # tests, none of which free mode uses, so ``gamma_alpha`` does not reach
        # here: free mode always concedes at the flat floor.
        alpha_min = self._o.alpha_min_frac
        trials = 0
        trace = logger.isEnabledFor(logging.DEBUG)
        while alpha >= alpha_min:
            trials += 1
            theta_t, f_t = eval_point(alpha)
            reason = self._free_reject_reason(theta_t, f_t, theta_max, margin, entries)
            if reason is None and grad_finite is not None and not grad_finite(alpha):
                reason = "non-finite-grad"
            if trace:
                logger.debug(
                    "  ls trial %d: alpha=%.3e theta=%.3e f=%.3e -> %s",
                    trials,
                    alpha,
                    theta_t,
                    f_t,
                    "free-accept" if reason is None else reason,
                )
            if reason is None:
                return LineSearchResult(alpha, True, False, False, n_trials=trials)
            alpha *= 0.5
        return LineSearchResult(alpha_min, False, False, True, n_trials=trials)

    def _free_reject_reason(
        self,
        theta_t: float,
        f_t: float,
        theta_max: float,
        margin: float,
        entries: list[tuple[float, float]],
    ) -> str | None:
        """The failing free-mode gate, or ``None`` if the trial is accepted."""
        if not math.isfinite(theta_t) or not math.isfinite(f_t):
            return "non-finite"
        if theta_t >= theta_max:
            return "theta-max"
        # Sufficient progress vs every previous free iterate: beat some entry's
        # θ or f by the margin (IPOPT: Acceptable(f + margin, θ + margin)).
        if all(theta_t + margin < tj or f_t + margin < fj for tj, fj in entries):
            return None
        return "free-filter"

    def _alpha_min(self, dphi: float, theta0: float, theta_min: float) -> float:
        """The step size at which the search concedes to restoration.

        By default the flat ``alpha_min_frac``. When ``gamma_alpha`` (γ_α) is
        requested, the *adaptive* rule of Wächter & Biegler 2006, eq. (23),
        derived from the current iterate::

                        ⎧ min{γ_θ, γ_φ·θ/(−∇φᵀd), δ·θ^{s_θ}/(−∇φᵀd)^{s_φ}}
            α_min = γ_α·⎨                        if ∇φᵀd < 0 and θ ≤ θ_min
                        ⎪ min{γ_θ, γ_φ·θ/(−∇φᵀd)} if ∇φᵀd < 0 and θ > θ_min
                        ⎩ γ_θ                     otherwise

        The third term is the switching-condition (eq. 19) bound: below θ_min the
        f-type branch is reachable, so α_min must stay under the α at which
        switching stops holding.

        The result is floored at ``alpha_min_frac``. That is not decoration: at a
        feasible iterate with a descent direction every eq. (23) term carries a
        factor of θ0 and the rule returns *exactly* 0.0, whereupon
        ``while α >= α_min`` never exits — α halves into the denormals, reaches
        0.0, and still passes ``>= 0.0``. The floor also bounds the opt-in to one
        direction: α_min can only rise, so requesting eq. (23) can make the search
        concede sooner but never backtrack further. (IPOPT applies eq. 23 raw.)
        """
        o = self._o
        if o.gamma_alpha is None:
            return o.alpha_min_frac
        a = o.gamma_theta
        if dphi < 0.0:
            a = min(a, o.gamma_phi * theta0 / (-dphi))
            if theta0 <= theta_min:
                # ``_safe_pow`` maps an overflowing (−∇φᵀd)^{s_φ} to inf, driving
                # the term to 0 (the floor then carries the result). The mirror
                # hazard is an *underflow* to 0.0 on a tiny dphi, where dividing
                # would raise ZeroDivisionError: there the exact term is
                # δ·θ^{s_θ}/0⁺ = +∞, so dropping it from the min is not a
                # workaround but precisely right.
                denom = _safe_pow(-dphi, o.s_phi)
                if denom > 0.0:
                    a = min(a, _SWITCH_DELTA * _safe_pow(theta0, o.s_theta) / denom)
        return max(o.alpha_min_frac, o.gamma_alpha * a)

    def _switching(self, dphi: float, alpha: float, theta0: float) -> bool:
        o = self._o
        return dphi < 0.0 and alpha * _safe_pow(
            -dphi, o.s_phi
        ) > _SWITCH_DELTA * _safe_pow(theta0, o.s_theta)

    def _is_ftype(
        self, dphi: float, alpha: float, theta0: float, theta_min: float
    ) -> bool:
        """Is this trial an f-type step — i.e. governed by the Armijo condition?

        By default the eq. (19) switching condition alone, which is what ipax has
        always used. W&B Algorithm A (Step 4) additionally requires an already
        nearly-feasible iterate, ``θ ≤ θ_min``: above θ_min the barrier objective
        is not yet the quantity to make progress on, so the step is judged by the
        eq. (20) sufficient-decrease test in θ or φ instead. That conjunct is
        opt-in via ``ftype_requires_theta_min`` (see the option for why).
        """
        if self._o.ftype_requires_theta_min and theta0 > theta_min:
            return False
        return self._switching(dphi, alpha, theta0)

    def _armijo_holds(
        self, phi_t: float, phi0: float, dphi: float, alpha: float
    ) -> bool:
        """Armijo decrease on the barrier objective (W&B 2006, eq. 20a)."""
        return phi_t <= phi0 + self._o.eta_phi * alpha * dphi

    def _augments_filter(
        self, phi_t: float, phi0: float, dphi: float, alpha: float, theta0: float
    ) -> bool:
        """Must an *accepted* trial be recorded in the filter? (W&B, Step 5.)

        The filter is augmented unless the iteration is f-type in the sense of
        W&B Step 5 — the switching condition (eq. 19) **and** Armijo (eq. 20a)
        both holding — which is a *different* predicate from the acceptance
        branch in ``_is_ftype``: it deliberately does not consult θ_min (IPOPT
        ``UpdateForNextIteration``: ``!IsFtype(α) || !ArmijoHolds(α)``, over a
        switching-only ``IsFtype``).

        The distinction only bites above θ_min, where eq. (20) can accept a step
        that fails Armijo: that step *is* recorded, because nothing else bounds
        it away from the point it came from.
        """
        return not (
            self._switching(dphi, alpha, theta0)
            and self._armijo_holds(phi_t, phi0, dphi, alpha)
        )

    def _reject_reason(
        self,
        theta_t: float,
        phi_t: float,
        theta0: float,
        phi0: float,
        dphi: float,
        alpha: float,
        theta_max: float,
        theta_min: float,
        entries: list[tuple[float, float]],
    ) -> str | None:
        """The first failing acceptance gate, or ``None`` if the trial is accepted.

        Names the gate (``non-finite`` / ``theta-max`` / ``filter`` / ``armijo``
        / ``no-decrease``) so the per-trial debug trace can report *why* a step
        size was rejected — the signal for diagnosing heavy backtracking.

        ``theta_min`` selects the branch together with the switching condition:
        Armijo governs only f-type trials (see ``_is_ftype``).
        """
        o = self._o
        # W&B eq. (18): the filter is initialized to the guard region {θ ≥ θ_max}.
        # Reject wildly infeasible (or non-finite) trials outright, before the
        # f-type switching/Armijo test — otherwise a step whose barrier objective
        # φ collapses toward -∞ could be accepted while θ explodes. A non-finite
        # φ_t itself must also be rejected: a trial that overshoots into a region
        # where the objective evaluates to ±∞/NaN (e.g. an overflowing exp/rational
        # element function) would otherwise pass the Armijo test — ``φ_t = -∞`` is
        # trivially below any finite bound — instead of backtracking to a finite,
        # usable iterate.
        if not math.isfinite(theta_t) or not math.isfinite(phi_t):
            return "non-finite"
        if theta_t >= theta_max:
            return "theta-max"
        if not self._filter_acceptable(theta_t, phi_t, entries):
            return "filter"
        if self._is_ftype(dphi, alpha, theta0, theta_min):
            # f-type step: require Armijo decrease on the barrier objective.
            return None if self._armijo_holds(phi_t, phi0, dphi, alpha) else "armijo"
        # θ-type step: sufficient decrease in θ or φ vs the current point
        # (W&B eq. 20). At a feasible iterate (θ0 = 0) no θ-progress step
        # exists — the branch would degenerate to "0 ≤ 0" and accept an
        # arbitrary ascent direction (W&B §2.3: only f-type/φ-decrease
        # acceptance applies there), so it requires θ0 > 0.
        theta_progress = theta0 > 0.0 and theta_t <= (1.0 - o.gamma_theta) * theta0
        if theta_progress or phi_t <= phi0 - o.gamma_phi * theta0:
            return None
        return "no-decrease"


__all__ = ["Filter", "FilterLineSearch", "LineSearchResult"]
