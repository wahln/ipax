"""Unit tests for barrier schedules and fraction-to-boundary."""

from __future__ import annotations

from ipax.ipm.barrier import fraction_to_boundary, update_mu
from ipax.options import BarrierOptions
from tests._helpers import array, assert_scalar_close, implemented


def test_update_mu_uses_monotone_fiacco_mccormick_formula():
    # Aggressive variant of Wächter & Biegler 2006, eq. (7):
    # μ⁺ = max(ε/10, κ_μ·min(μ, μ^θ_μ)) — κ_μ multiplies *both* branches, so
    # for μ ≤ 1 this is ipax's historical κ_μ·μ^θ_μ pace (the paper's plain
    # min() reduced μ 3–5× slower there and cost HS268/S268 in the S2MPJ v9
    # sweep), while μ > 1 still decreases linearly (κ_μ·μ) from any magnitude.
    options = BarrierOptions(kappa_mu=0.2, theta_mu=1.5)

    with implemented("mu schedule"):
        mu_next = update_mu(1e-1, options, tol=1e-8)

    assert_scalar_close(mu_next, max(1e-9, 0.2 * (1e-1) ** 1.5))


def test_update_mu_decreases_from_any_mu():
    # Regression: a plain κ_μ·μ^θ_μ *increases* for μ ≥ 25 (κ_μ = 0.2,
    # θ_μ = 1.5), permanently locking μ once an adaptive oracle inflated it —
    # the min() branch decreases linearly from any magnitude.
    options = BarrierOptions(kappa_mu=0.2, theta_mu=1.5)
    mu_next = update_mu(1e4, options, tol=1e-8)
    assert mu_next < 1e4
    assert_scalar_close(mu_next, 0.2 * 1e4)


def test_fraction_to_boundary_limits_negative_components(namespace):
    v = array(namespace, [1.0, 2.0])
    dv = array(namespace, [-0.5, -4.0])

    with implemented("fraction-to-boundary"):
        alpha = fraction_to_boundary(v, dv, tau=0.99)

    assert_scalar_close(alpha, 0.495)
    stepped = v + alpha * dv
    assert bool(namespace.all(stepped >= 0.01 * v))


def test_fraction_to_boundary_returns_one_for_nonnegative_step(namespace):
    v = array(namespace, [1.0, 2.0])
    dv = array(namespace, [0.5, 4.0])

    with implemented("fraction-to-boundary"):
        alpha = fraction_to_boundary(v, dv, tau=0.99)

    assert alpha == 1.0


def test_fraction_to_boundary_empty_vector_is_unconstrained(namespace):
    v = array(namespace, [])
    dv = array(namespace, [])
    assert fraction_to_boundary(v, dv, tau=0.99) == 1.0


def test_fraction_to_boundary_matches_elementwise_reference(namespace):
    """Vectorized rule equals the scalar loop it replaced (regression).

    The element-wise loop was O(n) host syncs/call and dominated GPU iteration
    time; this pins the vectorized form to identical results on mixed signs.
    """
    raw_v = [1.0, 2.0, 0.5, 3.0, 0.25, 10.0]
    raw_dv = [-0.5, 4.0, -4.0, 0.0, -0.1, -2.0]
    tau = 0.95

    expected = 1.0
    for vi, dvi in zip(raw_v, raw_dv, strict=True):
        if dvi < 0.0:
            expected = min(expected, tau * vi / (-dvi))
    expected = max(0.0, min(1.0, expected))

    alpha = fraction_to_boundary(array(namespace, raw_v), array(namespace, raw_dv), tau)
    assert_scalar_close(alpha, expected)
