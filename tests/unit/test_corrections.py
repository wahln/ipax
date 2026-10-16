"""Unit tests for higher-order step corrections (Mehrotra/Gondzio)."""

from __future__ import annotations

import itertools

import pytest

from ipax.backend.operators import Dense
from ipax.ipm.corrections import (
    CorrectionContext,
    GondzioCorrector,
    MehrotraCorrector,
    NoCorrection,
    _breedveld_correction,
    mehrotra_probing_mu,
    probing_mu,
    select_corrector,
)
from ipax.ipm.step import NewtonStep, recover_eliminated
from ipax.options import CorrectionsOptions
from tests._helpers import array, assert_allclose, assert_scalar_close


def _step(xp, *, ds: float, dy: float) -> NewtonStep:
    zero = array(xp, [0.0])
    return NewtonStep(
        dx=zero,
        ds=array(xp, [ds]),
        dy_eq=array(xp, []),
        dy_ineq=array(xp, [dy]),
        dz_lower=zero,
        dz_upper=zero,
    )


def _context(xp, affine, *, solve, alpha_primal, alpha_dual, mu_min=0.0):
    zero = array(xp, [0.0])
    gap = array(xp, [1.0])
    no_bound = zero > 0.5
    return CorrectionContext(
        affine=affine,
        s=array(xp, [2.0]),
        y_ineq=array(xp, [3.0]),
        x_minus_l=gap,
        u_minus_x=gap,
        z_lower=zero,
        z_upper=zero,
        mask_l=no_bound,
        mask_u=no_bound,
        solve=solve,
        alpha_primal=alpha_primal,
        alpha_dual=alpha_dual,
        mu_min=mu_min,
    )


def test_select_corrector_returns_requested_strategy():
    assert isinstance(select_corrector(CorrectionsOptions()), NoCorrection)
    assert isinstance(
        select_corrector(CorrectionsOptions(method="mehrotra")), MehrotraCorrector
    )
    assert isinstance(
        select_corrector(CorrectionsOptions(method="gondzio")), GondzioCorrector
    )


def test_active_flags():
    assert NoCorrection().active is False
    assert MehrotraCorrector(CorrectionsOptions(method="mehrotra")).active is True
    assert GondzioCorrector(CorrectionsOptions(method="gondzio")).active is True


def test_corrections_options_validation():
    with pytest.raises(ValueError, match="corrections method"):
        CorrectionsOptions(method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gondzio_max_corrections"):
        CorrectionsOptions(gondzio_max_corrections=-1)
    with pytest.raises(ValueError, match="gondzio_gamma"):
        CorrectionsOptions(gondzio_gamma=1.0)


def test_options_normalizes_corrections_shorthand():
    from ipax.options import Options

    opts = Options(corrections="gondzio")
    assert isinstance(opts.corrections, CorrectionsOptions)
    assert opts.corrections.method == "gondzio"


def test_recover_eliminated_complementarity_target_override(namespace):
    # With a zero Jacobian and no bounds, Δλ = -λ + τ_s/s. The scalar μ path uses
    # τ_s = μ; passing comp_s overrides it per component.
    xp = namespace
    dx = array(xp, [0.0])
    jac = Dense(array(xp, [[0.0]]))
    no_bound = array(xp, [0.0]) > 0.5
    z = array(xp, [0.0])
    gap = array(xp, [1.0])
    common = {
        "xp": xp,
        "ineq_jac": jac,
        "m": 1,
        "s": array(xp, [2.0]),
        "y_ineq": array(xp, [3.0]),
        "r_pi": array(xp, [0.0]),
        "sigma_s": array(xp, [1.5]),
        "z_lower": z,
        "z_upper": z,
        "sigma_l": z,
        "sigma_u": z,
        "x_minus_l": gap,
        "u_minus_x": gap,
        "mask_l": no_bound,
        "mask_u": no_bound,
    }
    scalar = recover_eliminated(dx, mu=4.0, **common)
    assert_scalar_close(float(scalar.dy_ineq[0]), -3.0 + 4.0 / 2.0)  # -1

    override = recover_eliminated(dx, mu=0.0, comp_s=array(xp, [10.0]), **common)
    assert_scalar_close(float(override.dy_ineq[0]), -3.0 + 10.0 / 2.0)  # 2


def test_probing_mu_matches_mehrotra_sigma_rule(namespace):
    # μ = 2·3 = 6; α_p = α_d = 0.5 ⇒ μ_aff = 1.5·2.5 = 3.75;
    # σ = (μ_aff/μ)³ (Mehrotra 1992; NWW 2009, eqs. (3.2)–(3.5)).
    xp = namespace
    affine = _step(xp, ds=-1.0, dy=-1.0)
    context = _context(
        xp,
        affine,
        solve=lambda *args: None,
        alpha_primal=lambda step, *, tau: 0.5,
        alpha_dual=lambda step, *, tau: 0.5,
    )
    assert_scalar_close(probing_mu(context), (3.75 / 6.0) ** 3 * 6.0)


def test_probing_mu_never_exceeds_average_complementarity(namespace):
    # Regression: a quasi-Newton affine probe can *inflate* the products
    # (μ_aff > μ), making σ = (μ_aff/μ)³ arbitrarily large and exploding μ
    # within a few iterations (RT least-squares example, L-BFGS Hessian).
    # σ is therefore clipped at 1: probing may hold μ but never raise it.
    xp = namespace
    inflating = _step(xp, ds=1.0, dy=1.0)  # trial products (2+1)(3+1) = 12 > 6
    context = _context(
        xp,
        inflating,
        solve=lambda *args: None,
        alpha_primal=lambda step, *, tau: 1.0,
        alpha_dual=lambda step, *, tau: 1.0,
    )
    assert_scalar_close(probing_mu(context), 6.0)  # σ clipped to 1 ⇒ μ = μ_avg


def test_probing_mu_respects_floor(namespace):
    xp = namespace
    affine = _step(xp, ds=-1.0, dy=-1.0)
    context = _context(
        xp,
        affine,
        solve=lambda *args: None,
        alpha_primal=lambda step, *, tau: 0.5,
        alpha_dual=lambda step, *, tau: 0.5,
        mu_min=2.0,
    )
    assert probing_mu(context) == 2.0


# -- unclipped Mehrotra σ rule (probing_sigma="mehrotra") -------------------


def test_mehrotra_probing_mu_matches_sigma_rule_against_measured_gap(namespace):
    # Mehrotra 1992; Breedveld 2017 supplement, eqs. (4)–(7) with μ_prev read
    # as the eq. (10) gap: μ_gap = 2·3 = 6; α = 0.5 ⇒ μ_aff = 1.5·2.5 = 3.75;
    # σ = (μ_aff/μ_gap)³, target σ·μ_gap — equal to the NWW rule when σ < 1.
    xp = namespace
    context = _context(
        xp,
        _step(xp, ds=-1.0, dy=-1.0),
        solve=lambda *args: None,
        alpha_primal=lambda step, *, tau: 0.5,
        alpha_dual=lambda step, *, tau: 0.5,
    )
    expected = (3.75 / 6.0) ** 3 * 6.0
    assert_scalar_close(mehrotra_probing_mu(context, tau=0.995), expected)
    assert_scalar_close(probing_mu(context), expected)


def test_mehrotra_probing_mu_probes_with_the_given_tau(namespace):
    # The probe step is his §2.1 fraction-to-boundary step (τ = 0.995), not
    # the NWW rule's maximal τ = 1 step.
    xp = namespace
    seen: list[float] = []

    def alpha(step, *, tau):
        seen.append(tau)
        return 0.5

    context = _context(
        xp,
        _step(xp, ds=-1.0, dy=-1.0),
        solve=lambda *args: None,
        alpha_primal=alpha,
        alpha_dual=alpha,
    )
    mehrotra_probing_mu(context, tau=0.995)
    assert seen == [0.995, 0.995]


def test_mehrotra_probing_mu_is_unclipped(namespace):
    # Unlike the NWW rule (σ ≤ 1), σ is not clipped: trial products
    # (2+1)(3+1) = 12 against the gap 6 give σ = 8 and μ = 48 — above the
    # current gap.
    xp = namespace
    context = _context(
        xp,
        _step(xp, ds=1.0, dy=1.0),
        solve=lambda *args: None,
        alpha_primal=lambda step, *, tau: 1.0,
        alpha_dual=lambda step, *, tau: 1.0,
    )
    assert_scalar_close(mehrotra_probing_mu(context, tau=0.995), 48.0)
    assert_scalar_close(probing_mu(context), 6.0)


def test_probing_rules_survive_overflowing_sigma(namespace):
    # Regression: σ = (μ_aff/μ_gap)³ was a Python-float ``**``, which raises
    # OverflowError for ratios above ~5.6e102. Trial products (2+1e75)(3+1e75)
    # ≈ 1e150 against the gap 6: the cube saturates to inf. The unclipped
    # Mehrotra rule then holds μ at the measured gap (non-finite target
    # fallback); the NWW rule clips σ at 1 ⇒ the same μ = 6.
    xp = namespace
    context = _context(
        xp,
        _step(xp, ds=1e75, dy=1e75),
        solve=lambda *args: None,
        alpha_primal=lambda step, *, tau: 1.0,
        alpha_dual=lambda step, *, tau: 1.0,
    )
    assert mehrotra_probing_mu(context, tau=0.995) == 6.0
    assert probing_mu(context) == 6.0


def test_mehrotra_probing_mu_respects_floor(namespace):
    xp = namespace
    context = _context(
        xp,
        _step(xp, ds=-1.0, dy=-1.0),
        solve=lambda *args: None,
        alpha_primal=lambda step, *, tau: 0.5,
        alpha_dual=lambda step, *, tau: 0.5,
        mu_min=2.5,
    )
    assert mehrotra_probing_mu(context, tau=0.995) == 2.5


def test_breedveld_gondzio_variant_targets_and_acceptance(namespace):
    # Breedveld 2017 supplement A.2: after the full Mehrotra corrector, probe
    # the trial step α̌ = min(1.1·α, 1), target the trial products by the
    # two-tier rule of eqs. (13)–(14), and keep a correction only when the
    # step length grows to ≥ 1.07·α.
    xp = namespace
    affine = _step(xp, ds=-1.0, dy=-1.0)
    mehrotra = _step(xp, ds=10.0, dy=10.0)
    first = _step(xp, ds=0.1, dy=0.1)
    second = _step(xp, ds=0.2, dy=0.2)
    solves = []
    returns = [mehrotra, first, second]

    def solve(comp_s, comp_l, comp_u):
        solves.append(float(comp_s[0]))
        return returns[len(solves) - 1]

    alphas = {id(affine): 0.5, id(mehrotra): 0.5, id(first): 0.6, id(second): 0.61}

    def alpha(step, *, tau):
        return alphas[id(step)]

    context = _context(xp, affine, solve=solve, alpha_primal=alpha, alpha_dual=alpha)
    corrector = GondzioCorrector(
        CorrectionsOptions(
            method="gondzio", gondzio_max_corrections=3, gondzio_variant="breedveld"
        )
    )
    result = corrector.correct(context, 1.0)

    # Mehrotra target: μ − ΔsΔλ = 1 − 1 = 0, clipped into [0.1, 10] ⇒ 0.1.
    assert_scalar_close(solves[0], 0.1)
    # Trial α̌ = 0.55: product (2+5.5)(3+5.5) = 63.75; 8 − 63.75 < −5·8 ⇒ the
    # downward correction is capped at −40 ⇒ new target 0.1 − 40.
    assert_scalar_close(solves[1], 0.1 - 40.0)
    # first: 0.6 + 0.6 ≥ 1.07·(0.5 + 0.5) ⇒ kept; second: 1.22 < 1.07·1.2 ⇒
    # rejected, the loop stops with the first correction.
    assert len(solves) == 3
    assert result.step is first
    assert result.mu == 1.0


@pytest.mark.parametrize(
    ("ds", "dy", "expected_correction"),
    [
        (10.0, 10.0, -40.0),  # 8 − t < −5·β_max·μ ⇒ capped at −40
        (1.0, 1.0, 8.0 - 2.55 * 3.55),  # β_max·μ < t, uncapped ⇒ β_max·μ − t
        (-1.0, -1.0, 0.0),  # inside [β_min·μ, β_max·μ] ⇒ untouched
        (-3.5, -5.0, 0.2 - 0.075 * 0.25),  # t < β_min·μ ⇒ β_min·μ − t
    ],
)
def test_breedveld_gondzio_capped_corrections(namespace, ds, dy, expected_correction):
    # Breedveld 2017 supplement A.2, eqs. (13)–(14) in the capped reading
    # (μ = 1, β = 0.2 / 8, trial α̌ = 1.1·0.5 = 0.55; the Mehrotra target
    # clips to 0.1).
    xp = namespace
    affine = _step(xp, ds=-1.0, dy=-1.0)
    mehrotra = _step(xp, ds=ds, dy=dy)
    solves = []

    def solve(comp_s, comp_l, comp_u):
        solves.append(float(comp_s[0]))
        return mehrotra if len(solves) == 1 else None

    context = _context(
        xp,
        affine,
        solve=solve,
        alpha_primal=lambda step, *, tau: 0.5,
        alpha_dual=lambda step, *, tau: 0.5,
    )
    GondzioCorrector(
        CorrectionsOptions(
            method="gondzio", gondzio_max_corrections=1, gondzio_variant="breedveld"
        )
    ).correct(context, 1.0)

    assert len(solves) == 2
    assert_scalar_close(solves[1], 0.1 + expected_correction)


def test_breedveld_correction_is_monotone_in_the_product(namespace):
    # Breedveld 2017 supplement A.2, eqs. (13)–(14): the literal two-tier
    # reading is non-monotone (t = 39 → −31, t = 41 → −1, t = 1000 → −960);
    # the capped reading (Gondzio 1996's outlier cap) is non-increasing in t:
    # 0 inside [0.2, 8], 8 − t above it, never below −5·8 = −40 (μ = 1).
    xp = namespace
    products = [7.0, 8.0, 9.0, 39.0, 40.0, 41.0, 48.0, 49.0, 1000.0]
    got = _breedveld_correction(xp, array(xp, products), 1.0)
    values = [float(got[i]) for i in range(len(products))]
    assert values == [0.0, 0.0, -1.0, -31.0, -32.0, -33.0, -40.0, -40.0, -40.0]
    assert all(a >= b for a, b in itertools.pairwise(values))
    below = _breedveld_correction(xp, array(xp, [0.05, 0.2]), 1.0)
    assert_allclose(xp, below, array(xp, [0.15, 0.0]))


def test_breedveld_gondzio_corrects_bound_blocks(namespace):
    # The bound blocks get the same capped correction on their finite
    # entries and stay zero off the masks (Breedveld 2017 supplement A.2,
    # eq. (14) for the upper block). μ = 1; α = 0.5 ⇒ trial α̌ = 0.55.
    xp = namespace
    zero2 = array(xp, [0.0, 0.0])
    affine = NewtonStep(
        dx=zero2,
        ds=array(xp, [-1.0]),
        dy_eq=array(xp, []),
        dy_ineq=array(xp, [-1.0]),
        dz_lower=zero2,
        dz_upper=zero2,
    )
    mehrotra = NewtonStep(
        dx=array(xp, [1.0, 1.0]),
        ds=array(xp, [-1.0]),
        dy_eq=array(xp, []),
        dy_ineq=array(xp, [-1.0]),
        dz_lower=array(xp, [10.0, 5.0]),
        dz_upper=array(xp, [-0.5, 3.0]),
    )
    solves = []

    def solve(comp_s, comp_l, comp_u):
        solves.append((comp_s, comp_l, comp_u))
        return mehrotra if len(solves) == 1 else None

    mask = array(xp, [1.0, 0.0]) > 0.5
    context = CorrectionContext(
        affine=affine,
        s=array(xp, [2.0]),
        y_ineq=array(xp, [3.0]),
        x_minus_l=array(xp, [1.0, 1.0]),
        u_minus_x=array(xp, [1.0, 1.0]),
        z_lower=array(xp, [2.0, 0.0]),
        z_upper=array(xp, [0.5, 0.0]),
        mask_l=mask,
        mask_u=mask,
        solve=solve,
        alpha_primal=lambda step, *, tau: 0.5,
        alpha_dual=lambda step, *, tau: 0.5,
    )
    GondzioCorrector(
        CorrectionsOptions(
            method="gondzio", gondzio_max_corrections=1, gondzio_variant="breedveld"
        )
    ).correct(context, 1.0)

    assert len(solves) == 2
    _, mehrotra_l, mehrotra_u = solves[0]
    assert_allclose(xp, mehrotra_l, array(xp, [1.0, 0.0]))  # μ − 0·0, masked
    assert_allclose(xp, mehrotra_u, array(xp, [1.0, 0.0]))
    comp_s, comp_l, comp_u = solves[1]
    # slack: (2−0.55)(3−0.55) = 3.5525 inside the box ⇒ untouched 0.1.
    assert_allclose(xp, comp_s, array(xp, [0.1]))
    # lower: (1+0.55)(2+5.5) = 11.625 > 8 ⇒ 1 + (8 − 11.625).
    assert_allclose(xp, comp_l, array(xp, [1.0 + 8.0 - 11.625, 0.0]))
    # upper: (1−0.55)(0.5−0.275) = 0.10125 < 0.2 ⇒ 1 + (0.2 − 0.10125).
    assert_allclose(xp, comp_u, array(xp, [1.0 + 0.2 - 0.10125, 0.0]))


def test_breedveld_gondzio_variant_rejects_small_growth(namespace):
    # A correction that grows the step by less than 7 % is discarded.
    xp = namespace
    affine = _step(xp, ds=-1.0, dy=-1.0)
    mehrotra = _step(xp, ds=-0.5, dy=-0.5)
    small = _step(xp, ds=-0.4, dy=-0.4)
    returns = [mehrotra, small]
    solves = []

    def solve(comp_s, comp_l, comp_u):
        solves.append(comp_s)
        return returns[len(solves) - 1]

    alphas = {id(affine): 0.5, id(mehrotra): 0.5, id(small): 0.53}
    context = _context(
        xp,
        affine,
        solve=solve,
        alpha_primal=lambda step, *, tau: alphas[id(step)],
        alpha_dual=lambda step, *, tau: alphas[id(step)],
    )
    corrector = GondzioCorrector(
        CorrectionsOptions(method="gondzio", gondzio_variant="breedveld")
    )
    result = corrector.correct(context, 1.0)
    assert len(solves) == 2
    assert result.step is mehrotra


def test_gondzio_variant_validated():
    assert CorrectionsOptions().gondzio_variant == "ipax"
    with pytest.raises(ValueError, match="gondzio_variant"):
        CorrectionsOptions(gondzio_variant="colombo")  # type: ignore[arg-type]


def test_mehrotra_second_order_targets_clipped_to_symmetric_box(namespace):
    # Regression: the −ΔΔ term from a *quasi-Newton* affine direction can be
    # arbitrarily large (RT least-squares example: targets ~1e5 inflated the
    # duals by five orders of magnitude in one step). Targets are clipped into
    # the symmetric neighbourhood [γμ, μ/γ] (Colombo & Gondzio 2008).
    xp = namespace
    affine = _step(xp, ds=-100.0, dy=100.0)  # −ΔsΔλ = +1e4 ≫ μ
    captured = []

    def solve(comp_s, comp_l, comp_u):
        captured.append((comp_s, comp_l, comp_u))
        return None

    context = _context(
        xp,
        affine,
        solve=solve,
        alpha_primal=lambda step, *, tau: 0.5,
        alpha_dual=lambda step, *, tau: 0.5,
    )
    corrector = MehrotraCorrector(CorrectionsOptions(method="mehrotra"))
    corrector.correct(context, 1.0)

    # γ = 0.1 (gondzio_gamma default): box [0.1, 10]; 1.0 + 1e4 clips to 10.
    assert len(captured) == 1
    assert_scalar_close(float(captured[0][0][0]), 10.0)


def test_mehrotra_failed_corrector_matches_affine_zero_target(namespace):
    xp = namespace
    affine = _step(xp, ds=-1.0, dy=-1.0)
    context = _context(
        xp,
        affine,
        solve=lambda *args: None,
        alpha_primal=lambda step, *, tau: 0.5,
        alpha_dual=lambda step, *, tau: 0.5,
    )

    corrector = MehrotraCorrector(CorrectionsOptions(method="mehrotra"))
    result = corrector.correct(context, probing_mu(context))

    # The corrector solve failed, so the affine direction (which targeted zero
    # complementarity) is returned with that zero target, not the oracle's μ.
    assert result.step is affine
    assert result.mu == 0.0


def test_mehrotra_falls_back_to_centered_when_correction_collapses_the_step(
    namespace,
):
    # Task-6 fragility (S2MPJ HS71 × exact/dense+mehrotra): the −ΔΔ term
    # presumes a Newton predictor, and on nonconvex curvature the corrected
    # direction can collapse the maximal boundary step. When the corrected
    # step keeps less than half the predictor's combined step length, the
    # corrector must degrade to the plain centered Newton step at the same
    # μ target (what corrections="none" would compute) instead of handing the
    # driver a crippled direction.
    xp = namespace
    affine = _step(xp, ds=-1.0, dy=-1.0)
    corrected = _step(xp, ds=-0.9, dy=-0.9)
    centered = _step(xp, ds=-0.5, dy=-0.5)
    solves = []

    def solve(comp_s, comp_l, comp_u):
        solves.append(comp_s)
        return corrected if len(solves) == 1 else centered

    alphas = {id(affine): 0.45, id(corrected): 0.05, id(centered): 0.4}

    def alpha(step, *, tau):
        return alphas[id(step)]

    context = _context(xp, affine, solve=solve, alpha_primal=alpha, alpha_dual=alpha)
    corrector = MehrotraCorrector(CorrectionsOptions(method="mehrotra"))
    result = corrector.correct(context, 1.0)

    assert result.step is centered
    assert result.mu == 1.0
    # The fallback re-solve targets the plain μ·e (no second-order term).
    assert len(solves) == 2
    assert_scalar_close(float(solves[1][0]), 1.0)


def test_mehrotra_keeps_a_correction_with_a_comparable_step(namespace):
    xp = namespace
    affine = _step(xp, ds=-1.0, dy=-1.0)
    corrected = _step(xp, ds=-0.9, dy=-0.9)
    solves = []

    def solve(comp_s, comp_l, comp_u):
        solves.append(comp_s)
        return corrected

    alphas = {id(affine): 0.45, id(corrected): 0.40}

    def alpha(step, *, tau):
        return alphas[id(step)]

    context = _context(xp, affine, solve=solve, alpha_primal=alpha, alpha_dual=alpha)
    corrector = MehrotraCorrector(CorrectionsOptions(method="mehrotra"))
    result = corrector.correct(context, 1.0)

    assert result.step is corrected
    assert result.mu == 1.0
    assert len(solves) == 1  # no fallback re-solve on the accepted path


def test_mehrotra_fallback_returns_affine_when_the_centered_solve_fails(namespace):
    xp = namespace
    affine = _step(xp, ds=-1.0, dy=-1.0)
    corrected = _step(xp, ds=-0.9, dy=-0.9)
    solves = []

    def solve(comp_s, comp_l, comp_u):
        solves.append(comp_s)
        return corrected if len(solves) == 1 else None

    alphas = {id(affine): 0.45, id(corrected): 0.05}

    def alpha(step, *, tau):
        return alphas[id(step)]

    context = _context(xp, affine, solve=solve, alpha_primal=alpha, alpha_dual=alpha)
    corrector = MehrotraCorrector(CorrectionsOptions(method="mehrotra"))
    result = corrector.correct(context, 1.0)

    assert result.step is affine
    assert result.mu == 0.0


def test_gondzio_accumulates_partial_trial_residual_and_uses_gamma(namespace):
    xp = namespace
    affine = _step(xp, ds=-1.0, dy=-1.0)
    base = _step(xp, ds=-0.4, dy=-0.5)
    candidate = _step(xp, ds=-0.2, dy=-0.2)
    candidate_2 = _step(xp, ds=-0.1, dy=-0.1)
    targets = []

    def solve(comp_s, comp_l, comp_u):
        targets.append((comp_s, comp_l, comp_u))
        if len(targets) == 1:
            return base
        if len(targets) == 2:
            return candidate
        return candidate_2

    def alpha_primal(step, *, tau):
        del tau
        if step is affine:
            return 0.5
        if step is base:
            return 0.2
        if step is candidate:
            return 0.8
        return 1.0

    def alpha_dual(step, *, tau):
        del tau
        if step is affine:
            return 0.5
        if step is base:
            return 0.3
        if step is candidate:
            return 0.8
        return 1.0

    context = _context(
        xp,
        affine,
        solve=solve,
        alpha_primal=alpha_primal,
        alpha_dual=alpha_dual,
    )
    gamma = 0.5
    result = GondzioCorrector(
        CorrectionsOptions(
            method="gondzio", gondzio_max_corrections=2, gondzio_gamma=gamma
        )
    ).correct(context, probing_mu(context))

    mu = 2.0 * 3.0
    mu_aff = (2.0 - 0.5) * (3.0 - 0.5)
    mu_target = (mu_aff / mu) ** 3 * mu
    # The Mehrotra −ΔΔ target is clipped into the symmetric box [γμ, μ/γ]
    # (Colombo & Gondzio 2008) before the Gondzio corrections accumulate on it.
    base_target = min(
        max(mu_target - (-1.0) * (-1.0), gamma * mu_target), mu_target / gamma
    )
    trial_product = (2.0 + 0.3 * -0.4) * (3.0 + 0.4 * -0.5)
    projected = min(max(trial_product, gamma * mu_target), mu_target / gamma)
    expected = base_target + projected - trial_product
    trial_product_2 = (2.0 + 0.9 * -0.2) * (3.0 + 0.9 * -0.2)
    projected_2 = min(max(trial_product_2, gamma * mu_target), mu_target / gamma)
    expected_2 = expected + projected_2 - trial_product_2

    assert len(targets) == 3
    assert_allclose(xp, targets[0][0], array(xp, [base_target]))
    assert_allclose(xp, targets[1][0], array(xp, [expected]))
    assert_allclose(xp, targets[2][0], array(xp, [expected_2]))
    assert result.step is candidate_2


# -- degenerate complementarity (no pairs / zero gap) -----------------------


def test_probing_rules_without_complementarity_pairs_return_zero(namespace):
    import dataclasses

    xp = namespace
    context = dataclasses.replace(
        _context(
            xp,
            _step(xp, ds=-1.0, dy=-1.0),
            solve=lambda *args: None,
            alpha_primal=lambda step, *, tau: 0.5,
            alpha_dual=lambda step, *, tau: 0.5,
            mu_min=0.5,
        ),
        s=array(xp, []),
        y_ineq=array(xp, []),
    )
    assert probing_mu(context) == 0.0
    assert mehrotra_probing_mu(context, tau=0.995) == 0.0


def test_probing_rules_at_zero_gap_fall_to_the_floor(namespace):
    # s = 0 ⇒ μ_gap = 0: σ = (μ_aff/μ)³ is undefined, so both rules return the
    # floor (the NWW rule via σ = 0, the unclipped rule via its μ_gap ≤ 0 guard).
    import dataclasses

    xp = namespace
    context = dataclasses.replace(
        _context(
            xp,
            _step(xp, ds=1.0, dy=1.0),
            solve=lambda *args: None,
            alpha_primal=lambda step, *, tau: 1.0,
            alpha_dual=lambda step, *, tau: 1.0,
            mu_min=0.5,
        ),
        s=array(xp, [0.0]),
    )
    assert probing_mu(context) == 0.5
    assert mehrotra_probing_mu(context, tau=0.995) == 0.5
