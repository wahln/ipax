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

"""Solver configuration as frozen dataclasses (no magic numbers in the loop body).

Defaults are tuned for RT-scale problems. Every numerical constant the IPM loop
needs lives here so the algorithm code stays declarative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

HessianMode = Literal["auto", "lbfgs", "exact", "autodiff-hvp"]
LinSolveMode = Literal["auto", "dense", "krylov", "sparse"]
Globalization = Literal["filter", "breedveld"]
MuSchedule = Literal["monotone", "adaptive", "breedveld", "probing", "quality"]
MuFallback = Literal["kkt-error", "never"]
FreeModeAcceptance = Literal["obj-constr-filter", "rigorous"]
KrylovMethod = Literal["cg", "minres", "gmres"]
KrylovPreconditioner = Literal["none", "jacobi", "lbfgs", "auto"]
DenseKKTRoute = Literal["condensed", "augmented"]
SparseKKTRoute = Literal["auto", "augmented", "normal_equations"]
ScalingMethod = Literal["none", "gradient-based"]
CorrectionsMethod = Literal["none", "mehrotra", "gondzio"]


def _validate_optional_positive(
    name: str, value: float | None, *, allow_zero: bool
) -> None:
    """Reject a non-``None`` tolerance that is not finite and (non-)negative."""
    if value is None:
        return
    valid_sign = value >= 0.0 if allow_zero else value > 0.0
    if not math.isfinite(value) or not valid_sign:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")


@dataclass(frozen=True, slots=True)
class BarrierOptions:
    """μ schedule and complementarity targets (Wächter & Biegler §3.1)."""

    mu_init: float = 1e-1
    mu_min: float = 1e-11
    kappa_mu: float = 0.2  # μ ← max(ε/10, κ_μ μ^θ_μ)
    theta_mu: float = 1.5
    tau_min: float = 0.99  # fraction-to-boundary floor
    # Free-mode safeguard (Nocedal, Wächter & Waltz 2009, §5.1, Algorithm A):
    # under a non-monotone μ oracle the scaled KKT error must stay below
    # κ·max(last l_max+1 free-mode errors); otherwise the oracle is suspended
    # and μ is handled monotonically — re-initialized at
    # ``fallback_mu_factor``·(average complementarity) — until the error drops
    # back below that threshold. ``"never"`` disables the safeguard (pure free
    # mode). Inert for the monotone schedule itself.
    fallback: MuFallback = "kkt-error"
    fallback_kappa: float = 0.9999  # κ ∈ (0, 1)
    fallback_window: int = 5  # l_max ≥ 0
    fallback_mu_factor: float = 0.8  # monotone re-entry μ factor
    # Centrality floor for μ re-targeting: μ ≥ κ_cent·max(dual, primal
    # infeasibility). El-Bakry et al. (1996)'s convergence theory requires the
    # complementarity gap not to vanish faster than the KKT residual; without
    # this floor an aggressive oracle can crush μ near a saddle while the dual
    # infeasibility is still O(1), pinning the iterate to the boundary with no
    # barrier left to re-center. Applied by the free-mode oracles *and* by the
    # KKT-error fallback's monotone re-entry μ (whose complementarity-based
    # value is otherwise powerless exactly when complementarity has collapsed
    # below the true residual). The complementarity component is deliberately
    # excluded so superlinear μ decrease near a solution is unimpeded.
    # ``0.0`` disables the floor.
    kappa_centrality: float = 1e-2
    # Scale-aware slack-initialization floor. The default flat slack floor
    # (``ipax.ipm.init._SLACK_FLOOR`` = 1e-2) pins violated-constraint slacks near
    # zero on a deeply-infeasible start, so the Newton direction drives them toward
    # their infeasible target ``s = -g < 0`` and the fraction-to-boundary rule clips
    # the primal step to ~1e-3 (the radiotherapy Phase-1 feasibility stall — many
    # near-active/violated slacks jammed against the floor). With this ``> 0`` the
    # floor becomes ``max(_SLACK_FLOOR, slack_init_scale·max|g(x_0)|)`` (init.py), so
    # the slacks — and, via ``y = μ_init/s``, the initial multipliers — start scaled
    # to the constraint magnitude instead of a fixed constant. ``0.0`` (the default)
    # leaves the flat floor unchanged; a value in ``[0.05, 0.5]`` matches the
    # constraint scale on RT-scale problems (Protons_01: feasibility at iter ~17 vs
    # ~42 with the flat floor). OPT-IN pending a full-corpus sweep of the default.
    slack_init_scale: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.fallback_kappa < 1.0:
            raise ValueError("fallback_kappa must lie in (0, 1)")
        if self.fallback_window < 0:
            raise ValueError("fallback_window must be non-negative")
        if self.fallback_mu_factor <= 0.0:
            raise ValueError("fallback_mu_factor must be positive")
        if not math.isfinite(self.kappa_centrality) or self.kappa_centrality < 0.0:
            raise ValueError("kappa_centrality must be finite and non-negative")
        if not math.isfinite(self.slack_init_scale) or self.slack_init_scale < 0.0:
            raise ValueError("slack_init_scale must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LineSearchOptions:
    """Filter line-search constants (Wächter & Biegler 2006, §2.3/§4.1).

    ``feasible_kkt_progress`` enables the feasible-point rescue acceptance: at
    an exactly feasible iterate (θ0 = 0) the eq. (19) switching condition holds
    for every descent direction, so every trial faces the full Armijo test —
    there is no θ-type escape (the branch IPOPT effectively lives on at its
    θ ≈ 1e-6 iterates, where "sufficient φ decrease vs θ0" is near-vacuous).
    A *first* trial that fails Armijo is then still accepted when the scaled
    KKT error decreases by at least this fraction — at a feasible iterate,
    optimality progress *is* progress (the KKT-error-globalization philosophy
    of Nocedal, Wächter & Waltz 2009, §5.1, applied to step acceptance). The
    certificate costs one extra gradient/Jacobian evaluation and is consulted
    on the first trial only. Must lie in ``(0, 1)``; ``None`` (the default)
    disables the rescue.
    """

    max_soc: int = 4
    # The absolute step size below which the search concedes to restoration.
    # Also the floor under the opt-in eq. (23) rule (see ``gamma_alpha``), where
    # it is what keeps a feasible iterate's α_min off zero.
    alpha_min_frac: float = 1e-8
    # γ_α — the safety factor of the *adaptive* minimum step size (W&B 2006,
    # eq. 23), which derives α_min from the current θ and ∇φᵀd instead of using
    # the flat ``alpha_min_frac``:
    #
    #     α_min = max(alpha_min_frac,
    #                 γ_α·min{γ_θ, γ_φ·θ/(−∇φᵀd), δ·θ^{s_θ}/(−∇φᵀd)^{s_φ}})
    #
    # ``None`` (the default) keeps the flat floor. IPOPT applies eq. (23)
    # unconditionally and names this option ``alpha_min_frac`` (default 0.05);
    # here it must be requested, and 0.05 is the value to request.
    #
    # OPT-IN: on the full S2MPJ corpus (2026-07-17) eq. (23) scored −4 as a
    # default — it concedes to restoration *earlier* than the flat floor, which
    # is its whole purpose, but ipax's restoration phase is a weaker recovery
    # than IPOPT's, so trading line-search trials for restoration entries loses
    # here (7 problems went optimal/acceptable → stalled, against 3 recovered).
    # Worth requesting where the restoration phase is known to be cheap or the
    # backtracking cost dominates. Must lie in (0, 1).
    gamma_alpha: float | None = None
    gamma_theta: float = 1e-5
    gamma_phi: float = 1e-5
    s_theta: float = 1.1
    s_phi: float = 2.3
    eta_phi: float = 1e-4  # Armijo constant
    # Require θ ≤ θ_min for a trial to be f-type (Armijo-governed), as W&B
    # Algorithm A (Step 4) and IPOPT specify; ipax keys the branch on the
    # eq. (19) switching condition alone. With this set, an infeasible iterate
    # judges trials by the eq. (20) sufficient-decrease test in θ *or* φ rather
    # than demanding Armijo decrease on φ.
    #
    # OPT-IN: on the full S2MPJ corpus (2026-07-17) the gate scored −10 as a
    # default, in two failure modes — 10 problems converged to a *different,
    # worse* optimum (ELATTAR, HS97, HS98, LUKVLE3: the θ-branch admits steps
    # Armijo refused, which changes the basin) and 12 stalled, likely because
    # above θ_min almost every accepted step becomes θ-type and so augments the
    # filter, whose accumulated entries then choke later iterations. Note that
    # ipax's γ_φ/η_φ differ from IPOPT's shipped values, so this gate has never
    # been tested alongside the constants it was designed against.
    ftype_requires_theta_min: bool = False
    # Required scaled-KKT-error decrease fraction for the feasible-point rescue
    # (see class docstring); None (default) disables it. OPT-IN: the S2MPJ v14
    # corpus sweep (2026-07-14) attributed 48 correct→incorrect flips to the
    # rescue as a default — on unconstrained/bounds-only problems (θ ≡ 0, the
    # rescue's whole domain) accepting Armijo-failing, KKT-decreasing steps
    # walked nonconvex least-squares runs into different, worse stationary
    # points (ROSENBRTU, MEYER3, GULF) or diverged f (FMINSURF family), while
    # only 3 wins depended on it. The free-mode acceptance
    # (``free_mode_acceptance``) covers the RT endgame this rescue was built
    # for, and does so under the NWW §5 monitor instead of per-step
    # certificates.
    feasible_kkt_progress: float | None = None
    # Free-mode acceptance (NWW 2009, §5): while a non-monotone μ oracle
    # (``Options.mu_schedule`` other than "monotone") is in free mode, the
    # barrier problem changes every iteration, so the W&B filter/Armijo
    # machinery is arguably not a consistent per-trial merit gate — NWW §5
    # instead carries global convergence in the iterate-level KKT-error monitor
    # (``BarrierOptions.fallback``). "obj-constr-filter" leans on that: it
    # accepts a trial when ``(θ + margin, f + margin)`` — the raw objective,
    # comparable across μ re-targets, unlike φ_μ — is acceptable to the filter
    # of previous free iterates.
    #
    # OPT-IN. "rigorous" (default) keeps the W&B gate in both regimes, for two
    # reasons. (1) It is the IPOPT-parity setting: released IPOPT never weakens
    # its per-trial test — ``SetRigorousLineSearch(false)`` is commented out in
    # IpAdaptiveMuUpdate.cpp, and its ``rigorous_`` flag only skips restoration
    # anyway, never the acceptance test. IPOPT's own (f, θ) margin filter is an
    # *iterate*-level progress check (AdaptiveMuUpdate::CheckSufficientProgress),
    # not a per-trial one. The mechanisms that make IPOPT's free mode work — the
    # filter reset on every μ change and the KKT-error monitor — are
    # unconditional here. (2) The paired S2MPJ A/Bs (2026-07-16) put the weak
    # test at ≈ neutral (quality +6, probing ±0) but with ~55 flips each way per
    # arm, about half of them landing on a *different, worse* local optimum on
    # basin-sensitive nonconvex problems (WOMFLET, OET7, SPIRAL, READING5,
    # ELATTAR, DISCS — the same set in both arms, so this is characterizable,
    # not noise): dropping the merit guardrail is load-bearing there.
    # Enable it for central-path-following workloads (radiotherapy-style) where
    # the rigorous gate grinds at near-feasible iterates. Inert for the
    # (default) monotone schedule either way.
    free_mode_acceptance: FreeModeAcceptance = "rigorous"
    # Margin of the free-mode filter: ``margin = fact · min(max_margin,
    # scaled KKT error)`` (IPOPT ``filter_margin_fact`` / ``filter_max_margin``
    # defaults).
    free_filter_margin_fact: float = 1e-5
    free_filter_max_margin: float = 1.0

    def __post_init__(self) -> None:
        # γ_α ≤ 0 would collapse the eq. (23) α_min onto the bare floor,
        # silently disabling the rule the caller just asked for; γ_α ≥ 1 would
        # let α_min reach the switching bound it is meant to sit safely *under*,
        # refusing steps the acceptance tests would still take. IPOPT bounds its
        # equivalent option to the same open interval.
        if self.gamma_alpha is not None and not 0.0 < self.gamma_alpha < 1.0:
            raise ValueError("gamma_alpha must lie in (0, 1) or be None")
        if not math.isfinite(self.alpha_min_frac) or self.alpha_min_frac <= 0.0:
            raise ValueError("alpha_min_frac must be finite and positive")
        # The rescue accepts when ``e_t ≤ (1 − γ)·e0``, so only γ ∈ (0, 1) is a
        # meaningful decrease fraction: γ ≤ 0 makes the bound ≥ e0 (an *increase*
        # in the KKT error would certify "progress"), and γ ≥ 1 demands a
        # non-positive error, which a norm never delivers — the rescue would
        # silently never fire. Reject both rather than degenerate quietly.
        if self.feasible_kkt_progress is not None and not (
            0.0 < self.feasible_kkt_progress < 1.0
        ):
            raise ValueError("feasible_kkt_progress must lie in (0, 1) or be None")
        if (
            not math.isfinite(self.free_filter_margin_fact)
            or self.free_filter_margin_fact < 0.0
        ):
            raise ValueError("free_filter_margin_fact must be finite and non-negative")
        if (
            not math.isfinite(self.free_filter_max_margin)
            or self.free_filter_max_margin < 0.0
        ):
            raise ValueError("free_filter_max_margin must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RegularizationOptions:
    """Friedlander–Orban primal–dual regularization + Breedveld δ_w bumping (§4.4)."""

    delta_w_init: float = 1e-6
    delta_w_max: float = 1e40
    delta_w_factor: float = 8.0  # escalation on Cholesky failure
    delta_c: float = 1e-8  # (2,2) block, equality regularization
    delta_c_max: float = 1e-1  # cap on δ_c escalation (rank-deficient ∇c)
    # δ_c escalation is a *last resort* for a singular DUAL block (rank-deficient
    # ∇c): it starts only once δ_w escalation has grown past this without resolving
    # the KKT failure — at which point δ_w is *reset to its floor* and the search
    # continues with (small δ_w, growing δ_c). A rank-deficient ∇c leaves δ_w
    # running to `delta_w_max` uselessly (it cannot repair the (2,2) block); an
    # ordinary indefinite (1,1) block is fixed by δ_w alone, so the trigger must
    # sit above any δ_w a genuine primal repair can need. Exploded multipliers can
    # legitimately demand δ_w ~ 1e8–1e9 (S2MPJ HS61: the primal block carries
    # −3.2e8 curvature, cured by δ_w = 2.3e9 alone; a lower trigger let δ_c
    # contaminate that repair and the distorted dual step cycled forever), hence
    # the generous default.
    delta_c_trigger: float = 1e10
    # Divergence-gated least-squares repair of the equality multipliers, applied
    # on every accepted step. A rank-deficient ∇c leaves the multipliers
    # *under-determined*: the dual block is singular, so ‖Δy‖ is bounded only by
    # the Friedlander & Orban (2012) (2,2)-block regularization ``delta_c``, and
    # one step can inject ‖c‖/δ_c ≈ 1e7 of pure null-space noise.
    # The filter cannot see it — acceptance tests (θ, φ) only — so a step that
    # improves feasibility while destroying stationarity is taken at full length,
    # and the poisoned multipliers then feed the L-BFGS Lagrangian pairs and every
    # later step system. Measured on S2MPJ NONSCOMPNE (∇c rank 24 of 25, exactly
    # singular): ‖y‖∞ goes 0 → 7.9e7 between iterations 0 and 1 on a problem whose
    # zero objective forces y* = 0.
    #
    # When set, ``y_eq`` is replaced by the least-squares estimate
    # ``argmin_y ‖∇f + Aᵀy‖`` (:func:`ipax.ipm.init.least_squares_duals`, solved
    # matrix-free) whenever the current multipliers' stationarity residual exceeds
    # this factor times the estimate's. The gate must be *conservative*: the
    # threshold has to clear a genuine 8–10 order-of-magnitude divergence without
    # disturbing ordinary primal–dual coupling. Firing eagerly is harmful — an
    # ungated repair pins the dual residual near 1e-8 and degrades healthy
    # equality problems from ``optimal`` to ``acceptable``, and gentler thresholds
    # cost SPANHYD its acceptable exit and move ROBOT's basin.
    #
    # It stays OPT-IN because the full-corpus sweep says it is a *trade*, not a
    # win. At 1e10 over the whole S2MPJ corpus on the three L-BFGS routes (3300
    # solves, 2026-07-30): **2217 → 2216 correct, net −1** — 22 fixed, 23 broken
    # (dense +2, krylov −3, sparse ±0). The fixes are the targeted family and are
    # exactly reproducible (VANDERM1, VANDERM2, COOLHANS, ARTIF, LAKES, HYDCAR20
    # → optimal, five of them long-standing gaps against IPOPT). The breakages
    # split two ways: 11 are ``max_time`` from the repair's own cost — a
    # least-squares CG solve on *every* accepted step, and iteration counts on
    # rows correct in both arms move only −0.5%, so this is per-iteration
    # overhead, not slower convergence — and the rest are trajectory changes
    # concentrated in the EIGEN family and ``RES``, which converge to a different
    # eigenvalue (EIGMAXA −1 → −0.998283 on all three routes). 39 unscored
    # objective drifts move in both directions (LAKES 2.8e11 → 3.5e5 and UBH5
    # 43.5 → 2.3 better; GASOIL 1.08 → 12.0 and HS90/HS91 1.36 → 86.7 worse).
    #
    # Enable it for the diagnosed signature — a solve that reaches a feasible
    # point and reports ``stalled`` with a large dual infeasibility, typically
    # with redundant or rank-deficient constraints — not as a general setting.
    equality_dual_repair: float | None = None

    def __post_init__(self) -> None:
        repair = self.equality_dual_repair
        if repair is not None and not (math.isfinite(repair) and repair >= 1.0):
            raise ValueError(
                "equality_dual_repair must be finite and >= 1.0, or None to "
                "disable the accepted-step multiplier repair"
            )


@dataclass(frozen=True, slots=True)
class BreedveldOptions:
    """Alternative step controller (Breedveld 2017 §2, eqs. 34–36)."""

    tau: float = 0.995  # fraction-to-boundary scaling
    backtrack: float = 0.9  # multiplicative backtracking factor
    max_backtrack: int = 20
    armijo_c: float = 1e-4  # Markov-filter sufficient-decrease constant
    ratio_limit: float = 20.0  # eq. (36): f⁺γ_y⁺ / fγ_y ratio-control bound


@dataclass(frozen=True, slots=True)
class LBFGSOptions:
    """Limited-memory Hessian (compact, Powell-damped) — invariant of PD (§4.3).

    ``damping_skip_ratio`` bounds how much a curvature pair may *contradict*
    positive curvature before it is dropped instead of damped: a pair with
    ``δᵀγ < −ratio · δᵀBδ`` is skipped, anything milder gets the usual Powell
    blend. Powell damping keeps ``B`` PD by fabricating positive curvature out
    of the pair — on strongly indefinite stretches that fabricated evidence
    redirects the search (S2MPJ ``ORTHRGDS``: ratios down to −25 damped ⇒
    1000+ iterations at a worse optimum; skipped ⇒ ~20 to IPOPT's). Blanket
    skipping (``powell_damping=False``, IPOPT's limited-memory policy) lost
    the full-corpus A/B by −31 — mild indefiniteness is where damping genuinely
    helps — hence a threshold rather than a switch. ``None`` (default) keeps
    pure Powell damping bit-for-bit.
    """

    memory: int = 10  # m ∈ [5, 20]
    powell_damping: bool = True
    initial_scaling: bool = True  # ξ seed from the newest pair (see seed_formula)
    damping_skip_ratio: float | None = None
    # ξ estimate when ``initial_scaling`` is on. Both are standard Rayleigh
    # seeds, but they diverge by the δ–γ misalignment factor 1/cos²∠(δ,γ):
    # "direct" (N&W eq. 7.20 inverted, γᵀγ/δᵀγ) is always ≥ "scalar1"
    # (IPOPT ``limited_memory_initialization``, δᵀγ/δᵀδ) — catastrophically so
    # on badly-scaled least squares (S2MPJ NELSONLS: median ξ 1e20 vs 6e3,
    # freezing every step; GASOIL: 508 stalled iterations vs optimal in 25).
    seed_formula: Literal["direct", "scalar1"] = "direct"

    def __post_init__(self) -> None:
        ratio = self.damping_skip_ratio
        if ratio is not None and not (math.isfinite(ratio) and ratio > 0.0):
            raise ValueError("damping_skip_ratio must be a positive finite float")
        if self.seed_formula not in ("direct", "scalar1"):
            raise ValueError("seed_formula must be 'direct' or 'scalar1'")


@dataclass(frozen=True, slots=True)
class KrylovOptions:
    """Matrix-free solver tolerances (§5.2)."""

    method: KrylovMethod = "cg"
    # When ``adaptive_tol`` is on this is the *floor* of the inexact-Newton forcing
    # sequence (the tightest inner tolerance, reached near convergence); when off it
    # is the fixed inner relative tolerance for every solve.
    rtol: float = 1e-10
    max_iter: int | None = None  # default: 2 * dim + 100 at the call site
    preconditioner: KrylovPreconditioner = "jacobi"
    gmres_restart: int = 30  # GMRES(m) restart length
    # Inexact-Newton forcing sequence (Eisenstat–Walker 1996): the inner solve need
    # only be as accurate as the *current* outer KKT residual demands, so early
    # iterations solve loosely (fast, and robust on ill-conditioned initial systems
    # a tight 1e-10 cannot reach) and tighten toward ``rtol`` as the IPM converges.
    # ``inner_rtol = clip(adaptive_eta · ‖outer KKT residual‖, rtol, adaptive_rtol_max)``.
    # Set ``adaptive_tol=False`` to force the fixed ``rtol`` on every solve.
    adaptive_tol: bool = True
    adaptive_eta: float = 0.1  # forcing factor: inner tol ≈ η · outer residual
    # Loosest inner tolerance (cap). Calibrated to the default outer scaled-KKT tol
    # (1e-8): the inner solve is never looser than the accuracy the IPM targets, so
    # early steps stay accurate enough to keep the iterate feasible (a looser cap
    # drives step-sensitive IPM problems into infeasibility) while still relaxing the
    # unreachable 1e-10 that stalls ill-conditioned initial systems.
    adaptive_rtol_max: float = 1e-8
    # ``"auto"`` starts with the cheap Jacobi diagonal and self-promotes to the
    # L-BFGS Woodbury/block preconditioner the first time a solve struggles —
    # either it fails to converge (then it retries the same solve promoted) or it
    # burns more than this fraction of the iteration budget. Sticky thereafter.
    auto_switch_ratio: float = 0.5

    def __post_init__(self) -> None:
        """Validate runtime values as well as static Literal hints."""
        if self.method not in ("cg", "minres", "gmres"):
            raise ValueError("Krylov method must be 'cg', 'minres', or 'gmres'")
        if self.preconditioner not in ("none", "jacobi", "lbfgs", "auto"):
            raise ValueError(
                "Krylov preconditioner must be 'none', 'jacobi', 'lbfgs', or 'auto'"
            )
        if self.gmres_restart < 1:
            raise ValueError("gmres_restart must be a positive integer")
        if not 0.0 < self.auto_switch_ratio <= 1.0:
            raise ValueError("auto_switch_ratio must lie in (0, 1]")
        if self.adaptive_eta <= 0.0:
            raise ValueError("adaptive_eta must be positive")
        # Equal floor/cap is allowed (adaptive collapses to the fixed rtol); the cap
        # must not be below the floor or above 1.
        if not self.rtol <= self.adaptive_rtol_max <= 1.0:
            raise ValueError("adaptive_rtol_max must lie in [rtol, 1]")


@dataclass(frozen=True, slots=True)
class DenseOptions:
    """Dense-route KKT materialization: condensed vs. augmented.

    ``"condensed"`` (default) forms the normal-equations block ``N`` by
    condensing the inequality Gram term ``∇gᵀ Σ_s ∇g`` into an explicit
    matmul, then Cholesky-probes it for positive definiteness
    (``ipax.linalg.dense``). ``"augmented"`` instead keeps ``∇g``/``−Σ_s⁻¹``
    as an explicit symmetric border — the dense analogue of the sparse-direct
    indefinite-augmented route (Friedlander & Orban 2012) — and factors it
    with a pivoted Bunch-Kaufman LDLᵀ where a backend adapter is available
    (NumPy/SciPy, Torch), falling back to a symmetric eigendecomposition
    (``xp.linalg.eigh``) otherwise. Either way it exposes real inertia, so
    the IPOPT inertia-guided δ_w correction (Wächter & Biegler 2006 §3.1)
    engages for the dense route too — closing a gap the plain Cholesky
    pass/fail check cannot. Falls back to ``"condensed"`` behavior
    automatically whenever the operator can't expose the unformed bordered
    matrix (e.g. an L-BFGS Hessian, already PD by Powell damping — there is
    nothing to gain there).

    ``augmented_max_size`` guards the augmented route against tall problems:
    the bordered matrix is ``(n + m_eq + m_ineq)²`` dense, so with ``m ≫ n``
    (e.g. radiotherapy-scale inequality counts) materializing it would allocate
    gigabytes for a system whose condensed form is only ``n × n``. When the
    assembled size would exceed this bound the solver silently falls back to
    the condensed route (losing only the inertia diagnostic, not correctness).

    ``gram_dtype="float32"`` (opt-in; default ``"native"``) accumulates the
    FLOP-dominant condensed inequality Gram term ``∇gᵀ Σ_s ∇g`` in float32 —
    ~2× on CPUs and up to the fp32/fp64 rate ratio on consumer GPUs, and the
    natural fit when the constraint data is float32 at the source (e.g. the
    TROTS dose matrices) — then restores working accuracy with fixed-precision
    iterative refinement against the exact float64 operator matvec (Carson &
    Higham 2018, SIAM J. Sci. Comput. 40(2)): up to ``refine_max_iters``
    correction steps, targeting a relative residual of ``refine_tol``. When
    the budget runs out, the contraction plateaus, or the sequence diverges
    (fp32 rounding concentrated in the small-eigenvalue subspace makes the
    *first* iterate the best one), the best iterate seen is still accepted if
    its *measured exact residual* is within ``refine_accept_tol`` — an
    honest, looser certificate, and still far tighter than the inexact-Newton
    forcing the Krylov route solves the same systems with (Dembo, Eisenstat &
    Steihaug 1982); only a solve missing even that level fails.
    A failed solve — or a positive-definiteness failure the exact matrix does
    not reproduce — rebuilds the exact matrix for that factorization, and
    ``refine_failure_limit`` *consecutive* failures switch the instance back
    to native precision for good (conditioning along an IPM run is not
    monotone, so one hard stretch must not forfeit the savings everywhere
    else). The accuracy of every returned step is certified by a measured
    exact residual — only accumulation cost is traded. (The PD probe runs on
    the approximate matrix and relies on the refinement rejection to catch a
    masked-indefinite exact block — see
    ``DenseSolver._materialize_and_guard``.) Applies to the inequality/bound
    **condensed** assembly; equality-constrained saddle systems currently
    assemble exactly and ignore the request.
    """

    kkt_route: DenseKKTRoute = "condensed"
    augmented_max_size: int = 20_000
    gram_dtype: str = "native"
    refine_tol: float = 1e-10
    refine_accept_tol: float = 1e-6
    refine_max_iters: int = 15
    refine_stall_ratio: float = 0.9
    refine_failure_limit: int = 3

    def __post_init__(self) -> None:
        if self.kkt_route not in ("condensed", "augmented"):
            raise ValueError("dense kkt_route must be 'condensed' or 'augmented'")
        if self.augmented_max_size < 1:
            raise ValueError("augmented_max_size must be a positive integer")
        if self.gram_dtype not in ("native", "float32"):
            raise ValueError("gram_dtype must be 'native' or 'float32'")
        if self.refine_tol <= 0.0:
            raise ValueError("refine_tol must be positive")
        if not self.refine_tol <= self.refine_accept_tol < 1.0:
            raise ValueError("refine_accept_tol must be in [refine_tol, 1)")
        if self.refine_max_iters < 1:
            raise ValueError("refine_max_iters must be a positive integer")
        if not 0.0 < self.refine_stall_ratio <= 1.0:
            raise ValueError("refine_stall_ratio must be in (0, 1]")
        if self.refine_failure_limit < 1:
            raise ValueError("refine_failure_limit must be a positive integer")


@dataclass(frozen=True, slots=True)
class SparseOptions:
    """Sparse-direct KKT assembly: augmented vs. sparse normal equations.

    ``"augmented"`` factors the bordered indefinite system — the
    inequality Jacobian ``∇g`` stays an explicit border with the ``−Σ_s⁻¹``
    slack block, so the factor is as sparse as ``∇g`` regardless of its
    column overlap (Friedlander & Orban 2012; Wächter & Biegler 2006 §3.1).
    ``"normal_equations"`` instead condenses the Gram term ``∇gᵀ Σ_s ∇g``
    *sparsely* into the logical ``n×n`` block via the Jacobian's ``gram_coo``
    (Breedveld 2017 §2: the condensed system is ``n×n`` however large ``m``
    grows) — the right choice for tall (``m ≫ n``) problems where either the
    Gram stays sparse (localized/banded rows) or ``∇g`` is dense enough that
    the bordered factor has no sparsity to exploit anyway. Requires a
    ``gram_coo``-capable inequality Jacobian (a sparse-operator Jacobian on a
    sparse-adapter backend), an L-BFGS Hessian, and COO-emittable equality
    Jacobians (they border into the factored saddle).

    ``"auto"`` (default) picks between the two per problem, reusing the
    tall-problem gate and measured thresholds of the ``linsolve="auto"``
    heuristic (``ipax/linalg/solver.py``): for ``m ≥ 10·n`` (and ``n`` under
    the tall bound) it selects the normal-equations form when the sampled
    Gram-fill estimate stays under the NE threshold **or** the Jacobian
    density is past the dense crossover — the TROTS dose-matrix regime (n≈70,
    m≈5.7k, ∇g fully dense), where the bordered factor is effectively a dense
    ``(n+m)`` factorization done through sparse machinery, ~8× slower
    end-to-end. Whenever the NE prerequisites are unmet, or the problem is
    not tall, auto stays on ``"augmented"`` — so non-tall problems are
    untouched by construction.
    """

    kkt_route: SparseKKTRoute = "auto"

    def __post_init__(self) -> None:
        if self.kkt_route not in ("auto", "augmented", "normal_equations"):
            raise ValueError(
                "sparse kkt_route must be 'auto', 'augmented' or 'normal_equations'"
            )


@dataclass(frozen=True, slots=True)
class ScalingOptions:
    """NLP auto-scaling (IPOPT ``nlp_scaling_method``; Wächter & Biegler 2006 §3.8).

    ``"gradient-based"`` (default) rescales the objective and each constraint once
    at the starting point so their gradients have an ∞-norm of at most
    ``max_gradient`` (factor ``min(1, max_gradient / ‖∇·‖∞)``); variables/bounds
    are left unscaled. This matches IPOPT's default and markedly improves
    convergence on badly-scaled problems; the returned ``x``, objective, and
    multipliers are reported in the original problem's units. ``"none"`` disables
    scaling so the solver sees the problem verbatim.
    """

    method: ScalingMethod = "gradient-based"
    max_gradient: float = 100.0  # IPOPT nlp_scaling_max_gradient

    def __post_init__(self) -> None:
        if self.method not in ("none", "gradient-based"):
            raise ValueError("scaling method must be 'none' or 'gradient-based'")
        if not math.isfinite(self.max_gradient):
            raise ValueError("scaling max_gradient must be finite")
        if self.max_gradient <= 0.0:
            raise ValueError("scaling max_gradient must be positive")


@dataclass(frozen=True, slots=True)
class CorrectionsOptions:
    """Higher-order predictor–corrector step corrections.

    The default ``"none"`` uses the single Newton/centering direction. The two
    optional higher-order schemes reuse the iteration's KKT operator for extra
    complementarity-target solves:

    * ``"mehrotra"`` — Mehrotra (1992) predictor–corrector (adaptive centering +
      second-order complementarity correction, one extra solve).
    * ``"gondzio"`` — Gondzio (1996) multiple centrality corrections (up to
      ``gondzio_max_corrections`` extra solves pulling the complementarity
      products into the box ``[γ μ, μ/γ]`` with ``γ = gondzio_gamma``).

    ``gondzio_gamma`` sets the centrality neighborhood and
    ``gondzio_max_corrections`` caps its extra solves per outer iteration.
    """

    method: CorrectionsMethod = "none"
    gondzio_max_corrections: int = 2  # Gondzio K_max (extra corrector solves)
    gondzio_gamma: float = 0.1  # centrality-box fraction γ ∈ (0, 1)

    def __post_init__(self) -> None:
        if self.method not in ("none", "mehrotra", "gondzio"):
            raise ValueError(
                "corrections method must be 'none', 'mehrotra', or 'gondzio'"
            )
        if self.gondzio_max_corrections < 0:
            raise ValueError("gondzio_max_corrections must be non-negative")
        if not 0.0 < self.gondzio_gamma < 1.0:
            raise ValueError("gondzio_gamma must lie in (0, 1)")


def _validate_condition_tols(
    f_tol: float | None,
    f_rel_change_tol: float | None,
    dual_inf_tol: float | None,
    constr_viol_tol: float | None,
    compl_inf_tol: float | None,
) -> None:
    """Validate the five shared termination-condition tolerances."""
    for name, value in (
        ("dual_inf_tol", dual_inf_tol),
        ("constr_viol_tol", constr_viol_tol),
        ("compl_inf_tol", compl_inf_tol),
    ):
        _validate_optional_positive(name, value, allow_zero=False)
    # ``f_tol`` is an objective magnitude and ``f_rel_change_tol`` a relative
    # change, so an exact-zero gate is meaningful for both.
    _validate_optional_positive("f_tol", f_tol, allow_zero=True)
    _validate_optional_positive("f_rel_change_tol", f_rel_change_tol, allow_zero=True)


@dataclass(frozen=True, slots=True)
class OptimalityConditionOptions:
    """Single-iteration optimality test → :attr:`Status.OPTIMAL`.

    The solve is declared optimal as soon as **every enabled** condition below
    holds in a single iteration. ``None`` disables a condition; at least one must
    remain enabled. The defaults reproduce the classic scaled-KKT test
    ``max(dual_inf, constr_viol, compl) ≤ 1e-8``.

    * ``f_tol`` — absolute objective magnitude ``|f_k| ≤ f_tol`` (a level test,
      so it can hold on iteration 0). Off by default.
    * ``f_rel_change_tol`` — relative objective change
      ``|f_k-f_{k-1}| / max(1, |f_k|, |f_{k-1}|)`` (needs a previous iterate, so
      it never holds on iteration 0). Off by default.
    * ``dual_inf_tol`` — scaled dual-infeasibility component.
    * ``constr_viol_tol`` — scaled primal-infeasibility (constraint violation).
    * ``compl_inf_tol`` — scaled complementarity component.

    The minimum of the enabled KKT-residual tolerances (``dual_inf_tol``,
    ``constr_viol_tol``, ``compl_inf_tol``) also sets the barrier μ floor via
    :attr:`kkt_tol`.
    """

    f_tol: float | None = None
    f_rel_change_tol: float | None = None
    dual_inf_tol: float | None = 1e-8
    constr_viol_tol: float | None = 1e-8
    compl_inf_tol: float | None = 1e-8

    def __post_init__(self) -> None:
        _validate_condition_tols(
            self.f_tol,
            self.f_rel_change_tol,
            self.dual_inf_tol,
            self.constr_viol_tol,
            self.compl_inf_tol,
        )
        enabled = (
            self.f_tol,
            self.f_rel_change_tol,
            self.dual_inf_tol,
            self.constr_viol_tol,
            self.compl_inf_tol,
        )
        if all(value is None for value in enabled):
            raise ValueError(
                "at least one optimality condition must be enabled "
                "(f_tol, f_rel_change_tol, dual_inf_tol, constr_viol_tol or "
                "compl_inf_tol)"
            )

    @property
    def kkt_tol(self) -> float:
        """Representative KKT tolerance feeding the barrier μ floor.

        The smallest enabled KKT-residual tolerance, or ``1e-8`` when only
        ``f_tol`` is set.
        """
        residual_tols = [
            value
            for value in (self.dual_inf_tol, self.constr_viol_tol, self.compl_inf_tol)
            if value is not None
        ]
        return min(residual_tols) if residual_tols else 1e-8


@dataclass(frozen=True, slots=True)
class AcceptableStoppingOptions:
    """Multi-iteration acceptable-level termination → :attr:`Status.ACCEPTABLE`.

    Mirrors :class:`OptimalityConditionOptions` but the conditions must hold for
    ``n_iter`` *consecutive* iterations before the solve stops. "Acceptable"
    means *what the caller is willing to accept* — e.g. stopping once the
    objective and primal feasibility have settled even though a
    dual-infeasibility-dominated residual will not reduce further.

    The defaults follow the IPOPT convention (``acceptable_tol = 1e-6`` — 1e2 ×
    the optimality tolerance — held for ``acceptable_iter = 15`` iterations): a
    problem whose achievable KKT floor sits between the acceptable and optimal
    tolerances (degenerate optimum, ill-conditioned least squares) then stops as
    :attr:`Status.ACCEPTABLE` after 15 stagnant iterations instead of grinding
    to ``max_iter``/``max_time`` at an essentially optimal point. Set every
    tolerance to ``None`` to disable the mechanism entirely.

    The fields match :class:`OptimalityConditionOptions` (``f_tol``,
    ``f_rel_change_tol``, ``dual_inf_tol``, ``constr_viol_tol``,
    ``compl_inf_tol``) plus ``n_iter``.
    """

    f_tol: float | None = None
    f_rel_change_tol: float | None = None
    # IPOPT ``acceptable_tol``: 1e2 × the 1e-8 optimality default, consistent
    # with the driver's step-failure salvage factor.
    dual_inf_tol: float | None = 1e-6
    constr_viol_tol: float | None = 1e-6
    compl_inf_tol: float | None = 1e-6
    n_iter: int = 15

    def __post_init__(self) -> None:
        _validate_condition_tols(
            self.f_tol,
            self.f_rel_change_tol,
            self.dual_inf_tol,
            self.constr_viol_tol,
            self.compl_inf_tol,
        )
        if self.n_iter < 1:
            raise ValueError("n_iter must be a positive integer")


@dataclass(frozen=True, slots=True)
class Options:
    """Top-level solver options.

    ``scaling`` accepts either a full :class:`ScalingOptions` object or the
    shorthand strings ``"none"`` and ``"gradient-based"``; ``corrections``
    likewise accepts a :class:`CorrectionsOptions` or one of ``"none"``,
    ``"mehrotra"``, ``"gondzio"``.

    ``hessian`` selects the Lagrangian Hessian source. ``"auto"`` (default) uses
    a supplied analytic ``lagrangian_hessian`` when present, else L-BFGS. The
    explicit modes are honored literally even when an analytic Hessian exists:
    ``"lbfgs"`` always uses the limited-memory approximation, ``"exact"`` requires
    the analytic operator (errors otherwise), ``"autodiff-hvp"`` uses backend
    autodiff Hessian-vector products.

    Termination has six sources, checked in priority order:

    * ``optimality`` (:class:`OptimalityConditionOptions`) — single-iteration
      test reporting :attr:`Status.OPTIMAL`.
    * ``acceptable`` (:class:`AcceptableStoppingOptions`) — multi-iteration test
      reporting :attr:`Status.ACCEPTABLE`; enabled by default (IPOPT convention:
      tolerances of ``1e-6`` held for 15 consecutive iterations). Set all its
      tolerances to ``None`` to disable.
    * ``diverging_iterates_tol`` — ‖x‖_∞ exceeding the threshold *while the
      objective has diverged below its negation* reports
      :attr:`Status.UNBOUNDED` (``None`` disables it).
    * ``max_stall_iter`` — after this many *consecutive* frozen iterations (no
      accepted step, KKT error unchanged) the run stops honestly with
      :attr:`Status.STALLED` — or :attr:`Status.ACCEPTABLE` when the iterate is
      within the relaxed KKT tolerance — instead of burning the whole
      iteration budget re-deriving the same rejected direction
      (``None`` disables it).
    * ``max_iter`` — iteration cap, reports :attr:`Status.MAX_ITER`.
    * ``max_time`` — wall-clock cap in seconds, reports :attr:`Status.MAX_TIME`
      (``None`` disables it).
    """

    optimality: OptimalityConditionOptions = field(
        default_factory=OptimalityConditionOptions
    )
    acceptable: AcceptableStoppingOptions = field(
        default_factory=AcceptableStoppingOptions
    )
    max_iter: int = 3000
    max_stall_iter: int | None = 25
    max_time: float | None = None
    # IPOPT ``diverging_iterates_tol``: if ‖x‖_∞ exceeds this WHILE the
    # objective sits below its negation, the solve stops with
    # :attr:`Status.UNBOUNDED` (unbounded below means f → −∞; the iterate norm
    # alone false-positives on convergent problems whose iterates wander
    # astronomically first — S2MPJ KOEBHELB — or whose optimum is legitimately
    # that large). ``None`` disables the test; slowly-diverging objectives
    # (e.g. −log x) then end at the iteration/time budget instead.
    diverging_iterates_tol: float | None = 1e20
    hessian: HessianMode = "auto"
    linsolve: LinSolveMode = "auto"
    globalization: Globalization = "filter"
    # The μ oracle. ``"monotone"`` (default) holds μ until the barrier KKT test
    # passes, then reduces it (Wächter & Biegler 2006, eq. (7)) — the default
    # was decided by the S2MPJ v10 paired A/B (2026-07): monotone beat probing
    # on every solver/Hessian config, 3837 vs 3770 correct in total, because an
    # aggressive oracle can crash μ faster than the duals converge (tail stalls
    # just above tolerance). ``"probing"`` uses the Mehrotra σ-rule from an
    # affine probe (NWW 2009, eqs. (3.2)–(3.5); one extra KKT solve per
    # iteration without a corrector) and uniquely rescues 12–18 problems per
    # config; ``"adaptive"`` re-targets μ every iteration by the LOQO
    # centrality rule (NWW 2009, eq. (3.6)); ``"breedveld"`` scales the duality
    # gap by the last accepted steplength (Breedveld et al. 2017,
    # eqs. (10)–(12)); ``"quality"`` picks σ by minimizing a linear model of
    # the *full* predicted KKT residual along the affine/centering family
    # (NWW 2009, §3.3 — IPOPT's adaptive default; one extra KKT solve per
    # iteration like probing). Unlike the complementarity-tracking oracles it
    # is bidirectional: σ > 1 raises μ when the dual residual dominates a
    # collapsed complementarity (the decentered-iterate failure mode).
    # The oracle is orthogonal to ``corrections``: an active
    # corrector aims at the oracle's μ. The non-monotone oracles are
    # safeguarded by the KKT-error fallback (``BarrierOptions.fallback``;
    # NWW 2009, §5.1) and the centrality floor.
    mu_schedule: MuSchedule = "monotone"

    barrier: BarrierOptions = field(default_factory=BarrierOptions)
    line_search: LineSearchOptions = field(default_factory=LineSearchOptions)
    regularization: RegularizationOptions = field(default_factory=RegularizationOptions)
    breedveld: BreedveldOptions = field(default_factory=BreedveldOptions)
    lbfgs: LBFGSOptions = field(default_factory=LBFGSOptions)
    krylov: KrylovOptions = field(default_factory=KrylovOptions)
    dense: DenseOptions = field(default_factory=DenseOptions)
    sparse: SparseOptions = field(default_factory=SparseOptions)
    scaling: ScalingOptions | ScalingMethod = field(default_factory=ScalingOptions)
    corrections: CorrectionsOptions | CorrectionsMethod = field(
        default_factory=CorrectionsOptions
    )

    # Derivative resolution toggles (§3.2).
    enable_autodiff: bool = True
    enable_finite_diff: bool = True

    # Console-handler verbosity tier (see ipax._logging): 0 silent, 1 result
    # summary, 2 +iteration table, 3 +problem, 4 +solver, 5 +options, >=6 +debug.
    verbose: int = 0

    def __post_init__(self) -> None:
        """Validate limits and normalize shorthand option values."""
        _validate_optional_positive("max_time", self.max_time, allow_zero=False)
        _validate_optional_positive(
            "diverging_iterates_tol", self.diverging_iterates_tol, allow_zero=False
        )
        if self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if self.max_stall_iter is not None and self.max_stall_iter < 1:
            raise ValueError("max_stall_iter must be a positive integer or None")
        if isinstance(self.scaling, str):
            object.__setattr__(self, "scaling", ScalingOptions(method=self.scaling))
        if isinstance(self.corrections, str):
            object.__setattr__(
                self, "corrections", CorrectionsOptions(method=self.corrections)
            )


__all__ = [
    "AcceptableStoppingOptions",
    "BarrierOptions",
    "BreedveldOptions",
    "CorrectionsMethod",
    "CorrectionsOptions",
    "DenseKKTRoute",
    "DenseOptions",
    "FreeModeAcceptance",
    "Globalization",
    "HessianMode",
    "KrylovMethod",
    "KrylovOptions",
    "KrylovPreconditioner",
    "LBFGSOptions",
    "LinSolveMode",
    "LineSearchOptions",
    "MuFallback",
    "MuSchedule",
    "OptimalityConditionOptions",
    "Options",
    "RegularizationOptions",
    "ScalingMethod",
    "ScalingOptions",
    "SparseKKTRoute",
    "SparseOptions",
]
