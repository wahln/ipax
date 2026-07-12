"""Unit tests for the feasible-point barrier-state re-centering.

When the filter line search fails at an already-feasible iterate, restoration
cannot move the point (it exits immediately at the same ``x``), so the driver
repairs the barrier state instead: slacks are re-floored on the current ``μ``
and the inequality multipliers are clipped into a band around the central
value ``μ/s`` (Wächter & Biegler 2006, §3.3 / eq. (16)). These tests pin the
helper's arithmetic; the driver-level behavior lives in
``tests/regression/test_feasible_point_restoration_recenter.py``.
"""

from __future__ import annotations

import math

from ipax.ipm.init import (
    _RECENTER_KAPPA,
    _RECENTER_SLACK_FRACTION,
    recenter_slacks_duals,
)
from tests._helpers import array


def test_active_slack_is_refloored_on_mu_and_poisoned_dual_is_clipped(namespace):
    # HS101 limit-cycle state: an active constraint (g ~ 0) whose slack sits on
    # restoration's 1e-12 boundary floor while the stale multiplier has grown
    # to ~1e6 — together Sigma_s = lambda/s ~ 1e18 and no step survives the
    # fraction-to-boundary rule. Re-centering must give the slack mu-scale
    # interiority and squash the multiplier to the central band.
    xp = namespace
    mu = 0.1
    g = array(xp, [-1e-14])
    y = array(xp, [3.2e6])
    s_new, y_new = recenter_slacks_duals(xp, g, y, mu)

    assert float(s_new[0]) == _RECENTER_SLACK_FRACTION * mu
    assert math.isclose(
        float(y_new[0]), _RECENTER_KAPPA * mu / float(s_new[0]), rel_tol=1e-12
    )


def test_inactive_slack_and_consistent_dual_pass_through(namespace):
    # A strictly-satisfied constraint keeps its natural slack s = -g, and a
    # multiplier already consistent with the central path (s*lambda ~ mu) must
    # not be disturbed — the repair only touches what is actually poisoned.
    xp = namespace
    mu = 0.1
    g = array(xp, [-2.0])
    y = array(xp, [mu / 2.0])
    s_new, y_new = recenter_slacks_duals(xp, g, y, mu)

    assert float(s_new[0]) == 2.0
    assert float(y_new[0]) == mu / 2.0


def test_vanished_dual_is_raised_to_the_band_floor(namespace):
    # The opposite poisoning: a multiplier collapsed to ~0 on an active
    # constraint makes the primal-dual Sigma_s vanish; the clip raises it to
    # the lower edge of the central band.
    xp = namespace
    mu = 0.1
    g = array(xp, [-1e-14])
    y = array(xp, [1e-15])
    s_new, y_new = recenter_slacks_duals(xp, g, y, mu)

    expected_floor = mu / (_RECENTER_KAPPA * float(s_new[0]))
    assert math.isclose(float(y_new[0]), expected_floor, rel_tol=1e-12)


def test_violated_constraint_gets_the_mu_floor(namespace):
    # A (slightly) violated row has -g < 0: the slack lands on the mu-scale
    # floor, staying strictly interior.
    xp = namespace
    mu = 1e-6
    g = array(xp, [0.5])
    y = array(xp, [1.0])
    s_new, _ = recenter_slacks_duals(xp, g, y, mu)

    assert float(s_new[0]) == _RECENTER_SLACK_FRACTION * mu
