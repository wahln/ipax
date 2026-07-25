"""Opt-in scale-aware slack-initialization floor.

The default flat slack floor (:data:`ipax.ipm.init._SLACK_FLOOR` = 1e-2) pins
violated-constraint slacks near zero on deeply-infeasible starts, so the Newton
direction drives them toward their infeasible target ``s = -g < 0`` and the
fraction-to-boundary rule clips the primal step to ~1e-3 (the radiotherapy
Phase-1 stall). ``BarrierOptions.slack_init_scale`` > 0 raises the floor to
``max(_SLACK_FLOOR, slack_init_scale * max|g(x0)|)`` so the slacks start with
room proportional to the constraint magnitude; ``0.0`` (default) leaves the flat
floor unchanged.
"""

from __future__ import annotations

import pytest

from ipax.ipm.init import _SLACK_FLOOR, initialize
from ipax.options import BarrierOptions
from tests._helpers import array


def _bounds_free(xp, n):
    """Masks/bounds for a problem with no finite variable bounds."""
    false = xp.asarray([False] * n)
    zeros = array(xp, [0.0] * n)
    return zeros, zeros, false, false


def test_default_scale_is_the_flat_floor(namespace):
    # slack_init_scale defaults to 0.0 -> the current flat _SLACK_FLOOR.
    xp = namespace
    x0 = array(xp, [0.0, 0.0])
    lo, up, ml, mu = _bounds_free(xp, 2)
    g = array(xp, [5.0, -3.0])  # one violated (g>0), one satisfied (s = -g = 3)
    start = initialize(
        xp=xp,
        x0=x0,
        lower_safe=lo,
        upper_safe=up,
        mask_l=ml,
        mask_u=mu,
        ineq_fn=lambda _x: g,
        mu_init=0.1,
        m=2,
    )
    assert float(start.s[0]) == _SLACK_FLOOR
    assert float(start.s[1]) == 3.0


def test_large_constraints_scale_the_floor_up(namespace):
    xp = namespace
    x0 = array(xp, [0.0, 0.0])
    lo, up, ml, mu = _bounds_free(xp, 2)
    g = array(xp, [5.0, -3.0])  # max|g| = 5 -> floor = max(1e-2, 0.1*5) = 0.5
    start = initialize(
        xp=xp,
        x0=x0,
        lower_safe=lo,
        upper_safe=up,
        mask_l=ml,
        mask_u=mu,
        ineq_fn=lambda _x: g,
        mu_init=0.1,
        m=2,
        slack_init_scale=0.1,
    )
    assert abs(float(start.s[0]) - 0.5) < 1e-12  # violated -> floor 0.5
    assert abs(float(start.s[1]) - 3.0) < 1e-12  # satisfied -> max(3, 0.5) = 3


def test_small_constraints_collapse_to_the_flat_floor(namespace):
    xp = namespace
    x0 = array(xp, [0.0, 0.0])
    lo, up, ml, mu = _bounds_free(xp, 2)
    # max|g| = 0.05 -> 0.1*0.05 = 5e-3 < 1e-2, so the floor stays _SLACK_FLOOR.
    g = array(xp, [0.05, -0.02])
    start = initialize(
        xp=xp,
        x0=x0,
        lower_safe=lo,
        upper_safe=up,
        mask_l=ml,
        mask_u=mu,
        ineq_fn=lambda _x: g,
        mu_init=0.1,
        m=2,
        slack_init_scale=0.1,
    )
    assert abs(float(start.s[0]) - _SLACK_FLOOR) < 1e-12  # violated -> 1e-2
    assert abs(float(start.s[1]) - 0.02) < 1e-12  # near-active satisfied -> -g


def test_larger_slack_floor_gives_proportionally_smaller_initial_duals(namespace):
    # y_ineq = mu_init / s: the scale-aware floor's real lever is the *dual*
    # scale (a floored slack of 1e-2 gives y = 10; a floor of 0.5 gives y = 0.2).
    xp = namespace
    x0 = array(xp, [0.0])
    lo, up, ml, mu = _bounds_free(xp, 1)
    g = array(xp, [5.0])
    flat = initialize(
        xp=xp,
        x0=x0,
        lower_safe=lo,
        upper_safe=up,
        mask_l=ml,
        mask_u=mu,
        ineq_fn=lambda _x: g,
        mu_init=0.1,
        m=1,
    )
    scaled = initialize(
        xp=xp,
        x0=x0,
        lower_safe=lo,
        upper_safe=up,
        mask_l=ml,
        mask_u=mu,
        ineq_fn=lambda _x: g,
        mu_init=0.1,
        m=1,
        slack_init_scale=0.1,
    )
    assert float(flat.y_ineq[0]) > float(scaled.y_ineq[0])
    assert abs(float(scaled.y_ineq[0]) - 0.2) < 1e-12  # 0.1 / 0.5


def test_slack_init_scale_validation():
    BarrierOptions(slack_init_scale=0.0)  # off (default)
    BarrierOptions(slack_init_scale=0.1)  # on
    with pytest.raises(ValueError, match="slack_init_scale"):
        BarrierOptions(slack_init_scale=-1.0)
    with pytest.raises(ValueError, match="slack_init_scale"):
        BarrierOptions(slack_init_scale=float("inf"))
