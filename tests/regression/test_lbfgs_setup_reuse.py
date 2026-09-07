"""Compact inverse setup must be shared before allocating n-sized work."""

from __future__ import annotations

import pytest

from ipax.backend.operators import Dense, Diagonal
from ipax.ipm.hessian import LBFGSOperator
from ipax.ipm.kkt import build_condensed_operator
from ipax.linalg.regularize import RegularizationState
from ipax.options import LBFGSOptions
from tests._helpers import array, assert_allclose


@pytest.mark.parametrize("unbounded", [False, True])
@pytest.mark.parametrize("inverse_first", [False, True])
def test_compact_inverse_reuses_all_setup(
    namespace, monkeypatch, unbounded, inverse_first
):
    xp = namespace
    w = LBFGSOperator(3, LBFGSOptions(memory=2))
    w.update(array(xp, [1.0, 0.5, -0.5]), array(xp, [2.0, 1.0, 0.5]))
    rhs = array(xp, [1.0, -2.0, 0.5])
    sigma = array(xp, [0.0] * 3 if unbounded else [0.25, 0.75, 1.25])
    op = build_condensed_operator(
        w,
        Diagonal(sigma),
        Diagonal(array(xp, [])),
        Dense(xp.zeros((0, 3), dtype=rhs.dtype)),
        RegularizationState(delta_w=1e-3),
        sigma_x_zero=unbounded,
    )
    expected = xp.linalg.solve(op.matmat(xp.eye(3, dtype=rhs.dtype)), rhs)
    first = (
        op.lbfgs_inverse_apply()(rhs)
        if inverse_first
        else op.dense_structured_solve(rhs)
    )
    assert_allclose(xp, first, expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("a cache hit must not rebuild the diagonal or Gram")

    monkeypatch.setattr(op, "_woodbury_diagonal", forbidden)
    monkeypatch.setattr(w, "gram_blocks", forbidden)
    assert_allclose(xp, op.dense_structured_solve(rhs), expected)
    assert_allclose(xp, op.lbfgs_inverse_apply()(rhs), expected)


def test_unbounded_krylov_inverse_uses_cached_gram(namespace, monkeypatch):
    from ipax.ipm import kkt

    xp = namespace
    w = LBFGSOperator(3, LBFGSOptions(memory=2))
    w.update(array(xp, [1.0, 0.5, -0.5]), array(xp, [2.0, 1.0, 0.5]))
    rhs = array(xp, [1.0, -2.0, 0.5])
    op = build_condensed_operator(
        w,
        Diagonal(xp.zeros_like(rhs)),
        Diagonal(array(xp, [])),
        Dense(xp.zeros((0, 3), dtype=rhs.dtype)),
        RegularizationState(delta_w=1e-3),
        sigma_x_zero=True,
    )
    expected = xp.linalg.solve(op.matmat(xp.eye(3, dtype=rhs.dtype)), rhs)
    original = kkt._woodbury_factors_blocks

    def checked(*args, **kwargs):
        assert kwargs.get("gram_u") is not None
        return original(*args, **kwargs)

    monkeypatch.setattr(kkt, "_woodbury_factors_blocks", checked)
    assert_allclose(xp, op.lbfgs_inverse_apply()(rhs), expected)
