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

"""Higher-order primal–dual step corrections (Mehrotra & Gondzio).

The default interior-point step is a single centered Newton direction. Two
classical enhancements compute corrector directions against the predictor's
unchanged KKT operator:

- **Mehrotra predictor–corrector** (Mehrotra 1992, *SIAM J. Optim.* 2(4)). An
  *affine* (predictor) solve with complementarity target ``0`` gives the maximal
  un-centered step and a second-order complementarity correction ``−ΔΔ``. A
  single corrector solve then targets ``μ − ΔΔ`` per component.
- **Gondzio multiple centrality corrections** (Gondzio 1996, *Comput. Optim.
  Appl.* 6(2)). Starting from the Mehrotra step, up to ``K`` further corrections
  project the *trial* complementarity products (taken at slightly enlarged step
  lengths) into a target box ``[γ μ, μ/γ]`` and re-solve; each correction
  is kept only while it enlarges the step lengths.

Correctors *consume* the barrier target ``μ`` — they never choose it (Nocedal,
Wächter & Waltz 2009: "the corrector is not part of the selection of the
barrier parameter"). The driver's μ oracle (``Options.mu_schedule``) picks the
target; :func:`probing_mu` is the Mehrotra σ-rule oracle
(``σ = (μ_aff/μ)³``, NWW 2009 eqs. (3.2)–(3.5)), which lives here because it
probes the affine direction. :class:`CenteringOnly` backs the standalone
``mu_schedule="probing"`` mode: the affine solve is only a probe and the
returned direction is the plain centered Newton step.

Both use :attr:`CorrectionContext.solve` so the injected-linear-algebra
invariant (#3) holds — a corrector never sees the solver, only a callback that
re-solves the same condensed system for new complementarity targets. The
sparse-direct route reuses its numeric factorization; dense and Krylov routes
perform another solve with the unchanged operator. The default
:class:`NoCorrection` is a transparent no-op (``active`` is ``False``) and is
never invoked by the driver, so the un-corrected path is unchanged.

This route is built around the slack/bound complementarity products; problems
with no inequalities *and* no finite bounds have nothing to centre, so a
corrector returns the affine step unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ipax.backend.namespace import array_namespace

if TYPE_CHECKING:
    from collections.abc import Callable

    from ipax.ipm.step import NewtonStep
    from ipax.options import CorrectionsOptions
    from ipax.typing import Array, Namespace

# Gondzio (1996): trial step lengths are probed after an ``_ENLARGE`` increase
# and a correction is accepted only if it improves the summed step length by
# ``_ACCEPT`` of that. The configurable centrality box is ``[γ μ, μ/γ]``.
_ENLARGE = 0.1
_ACCEPT = 0.1
# Fraction-to-boundary used for the predictor/trial *maximal* step lengths.
_TAU_BOUNDARY = 1.0


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    """A corrector's output: the search direction and the barrier target it uses.

    ``mu`` is the (centered) barrier parameter the returned ``step`` aims at; the
    driver adopts it for the fraction-to-boundary ``τ`` and the filter
    line-search barrier objective so the globalization stays consistent with the
    corrected direction.
    """

    step: NewtonStep
    mu: float


@dataclass(frozen=True, slots=True)
class _CorrectionState:
    """A full direction together with its complementarity-equation targets."""

    result: CorrectionResult
    comp_s: Array
    comp_l: Array
    comp_u: Array


@dataclass(frozen=True, slots=True)
class CorrectionContext:
    """Iterate state and solve primitives handed to a corrector each iteration.

    ``affine`` is the predictor direction (complementarity target ``0``), already
    solved by the driver against the iteration's KKT operator. The remaining
    arrays are the current positive iterate blocks whose complementarity
    products the corrections target. ``solve`` maps
    per-component complementarity targets ``(τ_s, τ_L, τ_U)`` to the recovered
    full :class:`~ipax.ipm.step.NewtonStep` (``None`` on a failed solve);
    ``alpha_primal``/``alpha_dual`` return fraction-to-boundary step lengths for
    a trial direction. ``mu_min`` is the driver-wide barrier/tolerance floor.
    """

    affine: NewtonStep
    s: Array
    y_ineq: Array
    x_minus_l: Array
    u_minus_x: Array
    z_lower: Array
    z_upper: Array
    mask_l: Array
    mask_u: Array
    solve: Callable[..., NewtonStep | None]
    alpha_primal: Callable[..., float]
    alpha_dual: Callable[..., float]
    mu_min: float = 0.0


@runtime_checkable
class HigherOrderCorrection(Protocol):
    """A predictor–corrector strategy producing the final search direction."""

    @property
    def active(self) -> bool:
        """Whether this corrector runs (``False`` skips all corrector work)."""
        ...

    def correct(self, context: CorrectionContext, mu_target: float) -> CorrectionResult:
        """Return the corrected direction aiming at the oracle's ``mu_target``."""
        ...


# -- shared complementarity helpers -----------------------------------------


def _counts(ctx: CorrectionContext, xp: Namespace) -> int:
    """Number of complementarity pairs (slacks + finite lower/upper bounds)."""
    m = int(ctx.s.shape[0])
    n_l = int(xp.sum(xp.astype(ctx.mask_l, xp.int64)))
    n_u = int(xp.sum(xp.astype(ctx.mask_u, xp.int64)))
    return m + n_l + n_u


def _products(
    ctx: CorrectionContext,
    xp: Namespace,
    step: NewtonStep | None,
    alpha_p: float,
    alpha_d: float,
) -> tuple[Array, Array, Array]:
    """Per-component complementarity products at the (optionally stepped) point.

    With ``step`` ``None`` these are the current products ``s∘λ`` etc.; otherwise
    they are evaluated at ``(primal + α_p Δ, dual + α_d Δ)``. Bound blocks are
    masked to zero off their active bounds.
    """
    zero_l = xp.zeros_like(ctx.x_minus_l)
    zero_u = xp.zeros_like(ctx.u_minus_x)
    if step is None:
        v_s = ctx.s * ctx.y_ineq
        v_l = xp.where(ctx.mask_l, ctx.x_minus_l * ctx.z_lower, zero_l)
        v_u = xp.where(ctx.mask_u, ctx.u_minus_x * ctx.z_upper, zero_u)
        return v_s, v_l, v_u
    s_t = ctx.s + alpha_p * step.ds
    lam_t = ctx.y_ineq + alpha_d * step.dy_ineq
    v_s = s_t * lam_t
    xl_t = ctx.x_minus_l + alpha_p * step.dx
    zl_t = ctx.z_lower + alpha_d * step.dz_lower
    v_l = xp.where(ctx.mask_l, xl_t * zl_t, zero_l)
    xu_t = ctx.u_minus_x - alpha_p * step.dx
    zu_t = ctx.z_upper + alpha_d * step.dz_upper
    v_u = xp.where(ctx.mask_u, xu_t * zu_t, zero_u)
    return v_s, v_l, v_u


def _average(v_s: Array, v_l: Array, v_u: Array, count: int, xp: Namespace) -> float:
    if count == 0:
        return 0.0
    total = float(xp.sum(v_s)) + float(xp.sum(v_l)) + float(xp.sum(v_u))
    return total / count


def _second_order_targets(
    ctx: CorrectionContext,
    xp: Namespace,
    step: NewtonStep,
    target_s: Array,
    target_l: Array,
    target_u: Array,
) -> tuple[Array, Array, Array]:
    """Add the second-order term ``−ΔΔ`` to per-component centering targets.

    The slack/lower products gain ``−Δ·Δ``; the upper block's slack is
    ``x_U − x`` with increment ``−Δx``, so its cross term flips sign (``+ΔxΔz_U``).
    """
    zero_l = xp.zeros_like(ctx.x_minus_l)
    zero_u = xp.zeros_like(ctx.u_minus_x)
    comp_s = target_s - step.ds * step.dy_ineq
    comp_l = xp.where(ctx.mask_l, target_l - step.dx * step.dz_lower, zero_l)
    comp_u = xp.where(ctx.mask_u, target_u + step.dx * step.dz_upper, zero_u)
    return comp_s, comp_l, comp_u


def probing_mu(ctx: CorrectionContext) -> float:
    """Mehrotra σ-rule μ oracle from an affine probe of the current iterate.

    ``σ = (μ_aff/μ)³`` where ``μ_aff`` is the average complementarity after the
    maximal (τ = 1) boundary step along the affine direction (Mehrotra 1992;
    Nocedal, Wächter & Waltz 2009, eqs. (3.2)–(3.5)). Floored at
    ``ctx.mu_min``; ``0.0`` when there are no complementarity pairs.
    """
    xp = array_namespace(ctx.affine.dx)
    count = _counts(ctx, xp)
    if count == 0:
        return 0.0
    mu = _average(*_products(ctx, xp, None, 0.0, 0.0), count, xp)
    alpha_p = ctx.alpha_primal(ctx.affine, tau=_TAU_BOUNDARY)
    alpha_d = ctx.alpha_dual(ctx.affine, tau=_TAU_BOUNDARY)
    mu_aff = _average(*_products(ctx, xp, ctx.affine, alpha_p, alpha_d), count, xp)
    # Adaptive centering (Mehrotra 1992, eq. for σ); guard μ > 0 (count > 0).
    # σ is clipped at 1: Mehrotra's rule presumes a true Newton predictor whose
    # full step *reduces* complementarity, but a quasi-Newton (L-BFGS) probe
    # can inflate the dual products instead (μ_aff ≫ μ), and an unguarded
    # σ = (μ_aff/μ)³ then explodes μ within a few iterations (regression:
    # tests/regression/test_mu_oracle_inflation.py). Probing may hold μ at the
    # current gap, never raise it.
    sigma = min((mu_aff / mu) ** 3, 1.0) if mu > 0.0 else 0.0
    return max(sigma * mu, ctx.mu_min)


def _mehrotra_step(
    ctx: CorrectionContext, xp: Namespace, mu_target: float, gamma: float
) -> _CorrectionState:
    """The Mehrotra corrector direction toward ``mu_target`` (shared by both).

    The second-order targets ``μ − ΔΔ`` are clipped into the symmetric
    neighbourhood ``[γμ, μ/γ]`` (Colombo & Gondzio 2008): with a *quasi-Newton*
    affine direction the ``−ΔΔ`` term is not a second-order complementarity
    residual and can be arbitrarily large, and unclipped targets inflated the
    dual state by orders of magnitude in one accepted step (regression:
    tests/regression/test_mu_oracle_inflation.py).
    """
    ones_s = xp.ones_like(ctx.s)
    ones_x = xp.ones_like(ctx.x_minus_l)
    comp_s, comp_l, comp_u = _second_order_targets(
        ctx, xp, ctx.affine, mu_target * ones_s, mu_target * ones_x, mu_target * ones_x
    )
    if mu_target > 0.0:
        lo, hi = gamma * mu_target, mu_target / gamma
        zero_l = xp.zeros_like(ctx.x_minus_l)
        zero_u = xp.zeros_like(ctx.u_minus_x)
        comp_s = xp.clip(comp_s, lo, hi)
        comp_l = xp.where(ctx.mask_l, xp.clip(comp_l, lo, hi), zero_l)
        comp_u = xp.where(ctx.mask_u, xp.clip(comp_u, lo, hi), zero_u)
    step = ctx.solve(comp_s, comp_l, comp_u)
    if step is None:
        # The affine direction was solved with zero complementarity targets;
        # report that same target to the driver's globalization machinery.
        zero_s = xp.zeros_like(ctx.s)
        zero_x = xp.zeros_like(ctx.x_minus_l)
        return _CorrectionState(
            CorrectionResult(ctx.affine, 0.0), zero_s, zero_x, zero_x
        )
    return _CorrectionState(CorrectionResult(step, mu_target), comp_s, comp_l, comp_u)


class NoCorrection:
    """Default: no corrections. Never invoked (``active`` is ``False``)."""

    active = False

    def correct(self, context: CorrectionContext, mu_target: float) -> CorrectionResult:
        return CorrectionResult(context.affine, mu_target)


class CenteringOnly:
    """Plain centered re-solve at the oracle's μ target — no higher-order terms.

    Backs the standalone ``mu_schedule="probing"`` strategy (Nocedal, Wächter &
    Waltz 2009, §3): the affine solve is only a σ probe, and the returned
    direction is the ordinary centered Newton step targeting ``μ e``.
    """

    active = True

    def correct(self, context: CorrectionContext, mu_target: float) -> CorrectionResult:
        xp = array_namespace(context.affine.dx)
        if _counts(context, xp) == 0:
            return CorrectionResult(context.affine, mu_target)
        step = context.solve(
            mu_target * xp.ones_like(context.s),
            mu_target * xp.ones_like(context.x_minus_l),
            mu_target * xp.ones_like(context.u_minus_x),
        )
        if step is None:
            # Fall back to the affine direction, whose targets were zero.
            return CorrectionResult(context.affine, 0.0)
        return CorrectionResult(step, mu_target)


class MehrotraCorrector:
    """Mehrotra (1992) predictor–corrector toward the oracle's μ target."""

    active = True

    def __init__(self, options: CorrectionsOptions) -> None:
        self._options = options

    def correct(self, context: CorrectionContext, mu_target: float) -> CorrectionResult:
        xp = array_namespace(context.affine.dx)
        if _counts(context, xp) == 0:
            # nothing to centre — affine is the full Newton step
            return CorrectionResult(context.affine, mu_target)
        return _mehrotra_step(
            context, xp, mu_target, self._options.gondzio_gamma
        ).result


class GondzioCorrector:
    """Gondzio (1996) multiple centrality corrections on the Mehrotra step."""

    active = True

    def __init__(self, options: CorrectionsOptions) -> None:
        self._options = options

    def correct(self, context: CorrectionContext, mu_target: float) -> CorrectionResult:
        xp = array_namespace(context.affine.dx)
        if _counts(context, xp) == 0:
            return CorrectionResult(context.affine, mu_target)

        state = _mehrotra_step(context, xp, mu_target, self._options.gondzio_gamma)
        base = state.result
        step, mu_target = base.step, base.mu
        if mu_target <= 0.0:
            return base

        comp_s, comp_l, comp_u = state.comp_s, state.comp_l, state.comp_u
        gamma = self._options.gondzio_gamma
        lo = gamma * mu_target
        hi = mu_target / gamma
        zero_l = xp.zeros_like(context.x_minus_l)
        zero_u = xp.zeros_like(context.u_minus_x)
        for _ in range(self._options.gondzio_max_corrections):
            alpha_p = context.alpha_primal(step, tau=_TAU_BOUNDARY)
            alpha_d = context.alpha_dual(step, tau=_TAU_BOUNDARY)
            trial_p = min(alpha_p + _ENLARGE, 1.0)
            trial_d = min(alpha_d + _ENLARGE, 1.0)

            # Gondzio's correction RHS is the centrality residual at the
            # enlarged trial point. Accumulate it onto the current full
            # direction's complementarity targets; replacing those targets by
            # ``clip(v) - ΔΔ`` is equivalent only for a full (α=1) trial.
            v_s, v_l, v_u = _products(context, xp, step, trial_p, trial_d)
            candidate_s = comp_s + xp.clip(v_s, lo, hi) - v_s
            candidate_l = comp_l + xp.where(
                context.mask_l, xp.clip(v_l, lo, hi) - v_l, zero_l
            )
            candidate_u = comp_u + xp.where(
                context.mask_u, xp.clip(v_u, lo, hi) - v_u, zero_u
            )
            candidate = context.solve(candidate_s, candidate_l, candidate_u)
            if candidate is None:
                break

            new_p = context.alpha_primal(candidate, tau=_TAU_BOUNDARY)
            new_d = context.alpha_dual(candidate, tau=_TAU_BOUNDARY)
            if new_p + new_d >= alpha_p + alpha_d + _ACCEPT * 2.0 * _ENLARGE:
                step = candidate
                comp_s, comp_l, comp_u = candidate_s, candidate_l, candidate_u
            else:
                break
        return CorrectionResult(step, mu_target)


def select_corrector(options: CorrectionsOptions) -> HigherOrderCorrection:
    """Construct the corrector named by ``options.method`` (default no-op)."""
    method = options.method
    if method == "none":
        return NoCorrection()
    if method == "mehrotra":
        return MehrotraCorrector(options)
    if method == "gondzio":
        return GondzioCorrector(options)
    raise ValueError(f"unknown corrections method: {method!r}")


__all__ = [
    "CenteringOnly",
    "CorrectionContext",
    "CorrectionResult",
    "GondzioCorrector",
    "HigherOrderCorrection",
    "MehrotraCorrector",
    "NoCorrection",
    "probing_mu",
    "select_corrector",
]
