"""Unit tests for warm-start seeding."""

from __future__ import annotations

import pytest

from ipax.ipm.init import apply_warm_start
from ipax.result import Result, Status, WarmStart
from tests._helpers import array, assert_allclose


def _apply(xp, warm, *, m, m_eq, n, mask_l, mask_u):
    dtype = array(xp, [0.0]).dtype
    s = xp.full((m,), 0.5, dtype=dtype)
    y_ineq = xp.full((m,), 0.5, dtype=dtype)
    y_eq = xp.zeros((m_eq,), dtype=dtype)
    z_lower = xp.zeros((n,), dtype=dtype)
    z_upper = xp.zeros((n,), dtype=dtype)
    return apply_warm_start(
        xp=xp,
        warm=warm,
        s=s,
        y_eq=y_eq,
        y_ineq=y_ineq,
        z_lower=z_lower,
        z_upper=z_upper,
        m=m,
        m_eq=m_eq,
        n=n,
        mask_l=mask_l,
        mask_u=mask_u,
    )


def test_warm_start_floors_slacks_and_bound_duals(namespace):
    xp = namespace
    mask_l = array(xp, [1.0, 0.0]) > 0.5
    mask_u = array(xp, [0.0, 0.0]) > 0.5
    warm = WarmStart(
        s=array(xp, [-5.0]),  # negative → floored strictly positive
        y_eq=array(xp, [2.5]),  # free sign, passes through
        z_lower=array(xp, [-3.0, 7.0]),  # idx 0 active → floored; idx 1 → masked to 0
    )
    s, y_eq, y_ineq, z_lower, z_upper = _apply(
        xp, warm, m=1, m_eq=1, n=2, mask_l=mask_l, mask_u=mask_u
    )
    assert float(s[0]) > 0.0
    assert_allclose(xp, y_eq, array(xp, [2.5]), rtol=0, atol=0)
    assert float(z_lower[0]) > 0.0
    assert float(z_lower[1]) == 0.0  # off its bound
    # untouched blocks keep the default start
    assert_allclose(xp, y_ineq, array(xp, [0.5]), rtol=0, atol=0)
    assert_allclose(xp, z_upper, array(xp, [0.0, 0.0]), rtol=0, atol=0)


def test_warm_start_dimension_mismatch_raises(namespace):
    xp = namespace
    mask = array(xp, [0.0, 0.0]) > 0.5
    warm = WarmStart(z_lower=array(xp, [1.0, 2.0, 3.0]))  # n is 2, not 3
    with pytest.raises(ValueError, match="z_lower has length 3, expected 2"):
        _apply(xp, warm, m=0, m_eq=0, n=2, mask_l=mask, mask_u=mask)


def test_from_result_copies_multipliers(namespace):
    xp = namespace
    result = Result(
        status=Status.OPTIMAL,
        x=array(xp, [1.0]),
        objective=0.0,
        y_eq=array(xp, [2.0]),
        y_ineq=array(xp, [3.0]),
        z_lower=array(xp, [4.0]),
        z_upper=array(xp, [5.0]),
    )
    warm = WarmStart.from_result(result)
    assert warm.s is None  # slacks recomputed from feasibility
    assert warm.y_eq is result.y_eq
    assert warm.y_ineq is result.y_ineq
    assert warm.z_lower is result.z_lower
    assert warm.z_upper is result.z_upper
