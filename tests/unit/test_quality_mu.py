"""Unit tests for the quality-function μ oracle (NWW 2009, §3.3/§4).

``quality_mu`` chooses the centering parameter σ by minimizing a linear model
of the *full* KKT residual after the step ``d(σ) = d_aff + σ·d_cen`` — dual
and primal infeasibility (reduced by the fraction-to-boundary steplengths the
direction actually attains) plus the predicted complementarity. Unlike the
Mehrotra σ-rule (``probing_mu``, clipped at σ ≤ 1) and the LOQO rule (σ ≤ 0.8),
the quality function is **closed-loop and bidirectional**: when the dual
residual dominates and only a strongly centered direction can move it, the
minimizer sits at σ > 1 — a genuine μ *raise* (cf. IPOPT's adaptive default
``mu_oracle=quality-function``). This is the capability the RT fluence stall
lacked: complementarity had collapsed below the true KKT error and every
complementarity-tracking oracle followed it into the floor.

The contexts here are synthetic: ``solve``/``alpha_primal``/``alpha_dual`` are
hand-built callables, so each test pins one property of the σ selection
without a full solver in the loop.
"""

from __future__ import annotations

from ipax.ipm.corrections import CorrectionContext, probing_mu, quality_mu
from ipax.ipm.step import NewtonStep
from tests._helpers import array


def _step(namespace, *, ds, dy_ineq, dx=0.0):
    empty = namespace.zeros((0,), dtype=array(namespace, [0.0]).dtype)
    return NewtonStep(
        dx=array(namespace, [dx]),
        ds=array(namespace, [ds]),
        dy_eq=empty,
        dy_ineq=array(namespace, [dy_ineq]),
        dz_lower=array(namespace, [0.0]),
        dz_upper=array(namespace, [0.0]),
    )


def _context(namespace, *, affine, solve, alpha_primal, alpha_dual, mu_min=1e-9):
    """One slack pair (s = λ = 1 ⇒ μ_avg = 1), no active bounds."""
    return CorrectionContext(
        affine=affine,
        s=array(namespace, [1.0]),
        y_ineq=array(namespace, [1.0]),
        x_minus_l=array(namespace, [1.0]),
        u_minus_x=array(namespace, [1.0]),
        z_lower=array(namespace, [0.0]),
        z_upper=array(namespace, [0.0]),
        mask_l=namespace.asarray([False]),
        mask_u=namespace.asarray([False]),
        solve=solve,
        alpha_primal=alpha_primal,
        alpha_dual=alpha_dual,
        mu_min=mu_min,
    )


def _full_alpha(step, tau=0.995):
    del step, tau
    return 1.0


def test_quality_mu_crushes_mu_when_the_affine_step_is_perfect(namespace):
    # A perfect affine predictor (full step drives every product to zero) with
    # no primal/dual residual: the quality function is minimized by the
    # smallest σ — Mehrotra-like superlinear μ decrease, floored at mu_min.
    affine = _step(namespace, ds=-1.0, dy_ineq=-1.0)

    def solve(t_s, t_l, t_u):  # centered-at-μ_avg direction
        del t_s, t_l, t_u
        return _step(namespace, ds=-0.5, dy_ineq=-0.5)

    ctx = _context(
        namespace,
        affine=affine,
        solve=solve,
        alpha_primal=_full_alpha,
        alpha_dual=_full_alpha,
    )
    mu = quality_mu(ctx, dual_infeasibility=0.0, primal_infeasibility=0.0)

    assert mu >= ctx.mu_min
    assert mu <= 1e-4  # σ driven into the small-σ end (μ_avg = 1)


def test_quality_mu_raises_mu_when_the_dual_residual_dominates(namespace):
    # THE bidirectionality contract. The dual residual is large and the linear
    # model says it only falls once the direction is strongly centered: the
    # affine dual step would exit the cone (α_D ≈ 0), while centering opens it
    # (α_D grows with σ, reaching 1 near σ ≈ 1.5). The predicted-complementarity
    # penalty grows with σ, so the minimizer is finite but *above* 1 — a μ
    # raise no complementarity-tracking oracle (probing σ ≤ 1, LOQO σ ≤ 0.8)
    # can produce.
    affine = _step(namespace, ds=0.0, dy_ineq=-1.0)

    def solve(t_s, t_l, t_u):
        del t_s, t_l, t_u
        return _step(namespace, ds=0.1, dy_ineq=1.0)  # d_cen: ds=0.1, dy=2.0

    def alpha_dual(step, tau=0.995):
        del tau
        dy = float(step.dy_ineq[0])
        return min(max((dy + 1.0) / 3.0, 0.0), 1.0)  # 0 at dy=-1, 1 at dy=2

    ctx = _context(
        namespace,
        affine=affine,
        solve=solve,
        alpha_primal=_full_alpha,
        alpha_dual=alpha_dual,
    )
    mu = quality_mu(ctx, dual_infeasibility=10.0, primal_infeasibility=0.0)

    assert mu > 1.0  # raised above μ_avg = 1: σ* > 1
    assert mu < 10.0  # ... but the complementarity penalty keeps it finite


def test_quality_mu_no_pairs_returns_zero(namespace):
    empty = namespace.zeros((0,), dtype=array(namespace, [0.0]).dtype)
    affine = NewtonStep(
        dx=array(namespace, [0.0]),
        ds=empty,
        dy_eq=empty,
        dy_ineq=empty,
        dz_lower=array(namespace, [0.0]),
        dz_upper=array(namespace, [0.0]),
    )
    ctx = CorrectionContext(
        affine=affine,
        s=empty,
        y_ineq=empty,
        x_minus_l=array(namespace, [1.0]),
        u_minus_x=array(namespace, [1.0]),
        z_lower=array(namespace, [0.0]),
        z_upper=array(namespace, [0.0]),
        mask_l=namespace.asarray([False]),
        mask_u=namespace.asarray([False]),
        solve=lambda *targets: None,
        alpha_primal=_full_alpha,
        alpha_dual=_full_alpha,
    )
    assert quality_mu(ctx, dual_infeasibility=1.0, primal_infeasibility=1.0) == 0.0


def test_quality_mu_returns_mu_min_on_collapsed_complementarity(namespace):
    # μ_avg ≤ 0 (here λ = 0, so every slack product vanishes): there is no
    # complementarity scale left to build the σ family's targets from, so the
    # oracle degrades to its floor rather than dividing by a zero average.
    affine = _step(namespace, ds=-0.5, dy_ineq=-0.5)
    ctx = CorrectionContext(
        affine=affine,
        s=array(namespace, [1.0]),
        y_ineq=array(namespace, [0.0]),  # ⇒ s·λ = 0 ⇒ μ_avg = 0
        x_minus_l=array(namespace, [1.0]),
        u_minus_x=array(namespace, [1.0]),
        z_lower=array(namespace, [0.0]),
        z_upper=array(namespace, [0.0]),
        mask_l=namespace.asarray([False]),
        mask_u=namespace.asarray([False]),
        solve=lambda *targets: None,
        alpha_primal=_full_alpha,
        alpha_dual=_full_alpha,
        mu_min=1e-9,
    )

    mu = quality_mu(ctx, dual_infeasibility=1.0, primal_infeasibility=1.0)

    assert mu == 1e-9


def test_quality_mu_falls_back_to_probing_on_solve_failure(namespace):
    # No centering direction ⇒ no σ family to score: degrade honestly to the
    # Mehrotra σ-rule, which needs only the affine probe already in hand.
    affine = _step(namespace, ds=-0.5, dy_ineq=-0.5)

    def failing_solve(t_s, t_l, t_u):
        del t_s, t_l, t_u
        return None

    ctx = _context(
        namespace,
        affine=affine,
        solve=failing_solve,
        alpha_primal=_full_alpha,
        alpha_dual=_full_alpha,
    )
    mu = quality_mu(ctx, dual_infeasibility=1.0, primal_infeasibility=1.0)
    assert mu == probing_mu(ctx)
