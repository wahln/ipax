"""Unit tests for higher-order step corrections (Mehrotra/Gondzio)."""

from __future__ import annotations

import pytest

from ipax.backend.operators import Dense
from ipax.ipm.corrections import (
    CorrectionContext,
    GondzioCorrector,
    MehrotraCorrector,
    NoCorrection,
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


def _context(xp, affine, *, solve, alpha_primal, alpha_dual):
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

    result = MehrotraCorrector(CorrectionsOptions(method="mehrotra")).correct(context)

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
    ).correct(context)

    mu = 2.0 * 3.0
    mu_aff = (2.0 - 0.5) * (3.0 - 0.5)
    mu_target = (mu_aff / mu) ** 3 * mu
    base_target = mu_target - (-1.0) * (-1.0)
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
