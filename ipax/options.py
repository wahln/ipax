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
MuSchedule = Literal["monotone", "adaptive", "breedveld", "probing"]
MuFallback = Literal["kkt-error", "never"]
KrylovMethod = Literal["cg", "minres", "gmres"]
KrylovPreconditioner = Literal["none", "jacobi", "lbfgs", "auto"]
DenseKKTRoute = Literal["condensed", "augmented"]
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
    # Centrality floor for the free-mode oracles: μ ≥ κ_cent·max(dual, primal
    # infeasibility). El-Bakry et al. (1996)'s convergence theory requires the
    # complementarity gap not to vanish faster than the KKT residual; without
    # this floor an aggressive oracle can crush μ near a saddle while the dual
    # infeasibility is still O(1), pinning the iterate to the boundary with no
    # barrier left to re-center (and leaving the KKT-error fallback's
    # complementarity-based re-entry μ powerless). The complementarity
    # component is deliberately excluded so superlinear μ decrease near a
    # solution is unimpeded. ``0.0`` disables the floor.
    kappa_centrality: float = 1e-2

    def __post_init__(self) -> None:
        if not 0.0 < self.fallback_kappa < 1.0:
            raise ValueError("fallback_kappa must lie in (0, 1)")
        if self.fallback_window < 0:
            raise ValueError("fallback_window must be non-negative")
        if self.fallback_mu_factor <= 0.0:
            raise ValueError("fallback_mu_factor must be positive")
        if not math.isfinite(self.kappa_centrality) or self.kappa_centrality < 0.0:
            raise ValueError("kappa_centrality must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LineSearchOptions:
    """Filter line-search parameters (Wächter & Biegler §2–3)."""

    max_soc: int = 4
    alpha_min_frac: float = 1e-8
    gamma_theta: float = 1e-5
    gamma_phi: float = 1e-5
    s_theta: float = 1.1
    s_phi: float = 2.3
    eta_phi: float = 1e-4  # Armijo constant


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
    """Limited-memory Hessian (compact, Powell-damped) — invariant of PD (§4.3)."""

    memory: int = 10  # m ∈ [5, 20]
    powell_damping: bool = True
    initial_scaling: bool = True  # direct-Hessian seed ξ = γᵀγ / δᵀγ


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
    """

    kkt_route: DenseKKTRoute = "condensed"
    augmented_max_size: int = 20_000

    def __post_init__(self) -> None:
        if self.kkt_route not in ("condensed", "augmented"):
            raise ValueError("dense kkt_route must be 'condensed' or 'augmented'")
        if self.augmented_max_size < 1:
            raise ValueError("augmented_max_size must be a positive integer")


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

    Termination has five sources, checked in priority order:

    * ``optimality`` (:class:`OptimalityConditionOptions`) — single-iteration
      test reporting :attr:`Status.OPTIMAL`.
    * ``acceptable`` (:class:`AcceptableStoppingOptions`) — multi-iteration test
      reporting :attr:`Status.ACCEPTABLE`; enabled by default (IPOPT convention:
      tolerances of ``1e-6`` held for 15 consecutive iterations). Set all its
      tolerances to ``None`` to disable.
    * ``diverging_iterates_tol`` — ‖x‖_∞ exceeding the threshold reports
      :attr:`Status.UNBOUNDED` (``None`` disables it).
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
    max_time: float | None = None
    # IPOPT ``diverging_iterates_tol``: if ‖x‖_∞ exceeds this the iterates are
    # declared diverging and the solve stops with :attr:`Status.UNBOUNDED` (the
    # problem is likely unbounded below). ``None`` disables the test.
    diverging_iterates_tol: float | None = 1e20
    hessian: HessianMode = "auto"
    linsolve: LinSolveMode = "auto"
    globalization: Globalization = "filter"
    # The μ oracle. ``"probing"`` (default) uses the Mehrotra σ-rule from an
    # affine probe — the strongest strategy in the NWW 2009 comparison (their
    # eqs. (3.2)–(3.5)); it costs one extra KKT solve per iteration when no
    # corrector is active (with corrections the affine solve is shared).
    # ``"monotone"`` holds μ until the barrier KKT test passes, then reduces it
    # (Wächter & Biegler 2006, eq. (7)); ``"adaptive"`` re-targets μ every
    # iteration by the LOQO centrality rule (NWW 2009, eq. (3.6));
    # ``"breedveld"`` scales the duality gap by the last accepted steplength
    # (Breedveld et al. 2017, eqs. (10)–(12)). The oracle is orthogonal to
    # ``corrections``: an active corrector aims at the oracle's μ. The
    # non-monotone oracles are safeguarded by the KKT-error fallback
    # (``BarrierOptions.fallback``; NWW 2009, §5.1).
    mu_schedule: MuSchedule = "probing"

    barrier: BarrierOptions = field(default_factory=BarrierOptions)
    line_search: LineSearchOptions = field(default_factory=LineSearchOptions)
    regularization: RegularizationOptions = field(default_factory=RegularizationOptions)
    breedveld: BreedveldOptions = field(default_factory=BreedveldOptions)
    lbfgs: LBFGSOptions = field(default_factory=LBFGSOptions)
    krylov: KrylovOptions = field(default_factory=KrylovOptions)
    dense: DenseOptions = field(default_factory=DenseOptions)
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
]
