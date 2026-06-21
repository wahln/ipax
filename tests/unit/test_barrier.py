"""Unit tests for barrier schedules and fraction-to-boundary."""

from __future__ import annotations

from ipax.ipm.barrier import fraction_to_boundary, update_mu
from ipax.options import BarrierOptions
from tests._helpers import array, assert_scalar_close, implemented


def test_update_mu_uses_monotone_fiacco_mccormick_formula():
    options = BarrierOptions(kappa_mu=0.2, theta_mu=1.5)

    with implemented("mu schedule"):
        mu_next = update_mu(1e-1, options, tol=1e-8)

    assert_scalar_close(mu_next, max(1e-9, 0.2 * ((1e-1) ** 1.5)))


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
