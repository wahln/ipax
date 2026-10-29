"""End-to-end oracles for ``minimize_ipax`` through ``scipy.optimize.minimize``.

Hock–Schittkowski problem 71 (Hock & Schittkowski 1981) written the way a SciPy
user would write it — ``NonlinearConstraint`` with exact Jacobians/Hessians and
``Bounds`` — must reach the published optimum, report trust-constr-signed
multipliers that satisfy stationarity, and agree with SciPy's own
``trust-constr``. NumPy-only: SciPy's ``minimize`` accepts nothing else.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("scipy")

from scipy.optimize import Bounds, NonlinearConstraint, minimize

from ipax.interfaces.scipy import minimize_ipax

HS71_X = np.array([1.0, 4.74299963, 3.82114998, 1.37940829])
HS71_F = 17.0140173


def _f(x):
    return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]


def _grad(x):
    return np.array(
        [
            x[3] * (2 * x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
        ]
    )


def _hess(x):
    s = 2 * x[0] + x[1] + x[2]
    return np.array(
        [
            [2 * x[3], x[3], x[3], s],
            [x[3], 0.0, 0.0, x[0]],
            [x[3], 0.0, 0.0, x[0]],
            [s, x[0], x[0], 0.0],
        ]
    )


def _prod_hess(x):
    H = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i != j:
                H[i, j] = np.prod([x[k] for k in range(4) if k not in (i, j)])
    return H


def _constraints(with_hessians: bool):
    hess_prod = (lambda x, v: v[0] * _prod_hess(x)) if with_hessians else None
    hess_sq = (lambda x, v: v[0] * 2.0 * np.eye(4)) if with_hessians else None
    return [
        NonlinearConstraint(
            lambda x: np.array([np.prod(x)]),
            25.0,
            np.inf,
            jac=lambda x: (np.prod(x) / x)[None, :],
            **({"hess": hess_prod} if with_hessians else {}),
        ),
        NonlinearConstraint(
            lambda x: np.array([np.sum(x**2)]),
            40.0,
            40.0,
            jac=lambda x: (2.0 * x)[None, :],
            **({"hess": hess_sq} if with_hessians else {}),
        ),
    ]


@pytest.mark.parametrize("with_hessians", [True, False], ids=["exact", "lbfgs"])
def test_hs71_reaches_published_optimum(with_hessians):
    res = minimize(
        _f,
        np.array([1.0, 5.0, 5.0, 1.0]),
        jac=_grad,
        hess=_hess if with_hessians else None,
        bounds=Bounds(np.ones(4), 5.0 * np.ones(4)),
        constraints=_constraints(with_hessians),
        method=minimize_ipax,
    )

    assert res.success, res.message
    assert res.ipax_result.routes.hessian == ("exact" if with_hessians else "lbfgs")
    np.testing.assert_allclose(res.x, HS71_X, atol=1e-6)
    assert res.fun == pytest.approx(HS71_F, rel=1e-8)
    assert res.constr_violation <= 1e-8

    # IPOPT reports λ = (−0.5523, 0.1615) for g = (Πx, Σx²) with L = f + λᵀg —
    # the same sign convention as trust-constr's ``v``.
    assert res.v[0][0] == pytest.approx(-0.55229366, abs=1e-6)
    assert res.v[1][0] == pytest.approx(0.16146857, abs=1e-6)

    # Stationarity: ∇f + Σ J_kᵀ v_k + (z_U − z_L) = 0.
    x = res.x
    J_prod = np.prod(x) / x
    J_sq = 2.0 * x
    residual = (
        _grad(x) + J_prod * res.v[0][0] + J_sq * res.v[1][0] + res.bound_multipliers
    )
    np.testing.assert_allclose(residual, 0.0, atol=1e-6)


def test_hs71_agrees_with_trust_constr():
    common = {
        "jac": _grad,
        "hess": _hess,
        "bounds": Bounds(np.ones(4), 5.0 * np.ones(4)),
        "constraints": _constraints(True),
    }
    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    ours = minimize(_f, x0, method=minimize_ipax, **common)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = minimize(
            _f,
            x0,
            method="trust-constr",
            options={"gtol": 1e-10, "xtol": 1e-12, "maxiter": 5000},
            **common,
        )

    # trust-constr stops ~1e-6 relative above the published optimum (17.0140173)
    # even at these tolerances; ipax's own accuracy is pinned by the test above.
    np.testing.assert_allclose(ours.x, ref.x, atol=1e-4)
    assert ours.fun == pytest.approx(ref.fun, rel=1e-5)
    assert ours.fun <= ref.fun + 1e-9
    # trust-constr appends the bounds as a trailing ``v`` entry; minimize_ipax
    # reports them separately as ``bound_multipliers`` (z_U − z_L).
    assert len(ref.v) == len(ours.v) + 1
    for v_ours, v_ref in zip(ours.v, ref.v[: len(ours.v)], strict=True):
        np.testing.assert_allclose(v_ours, v_ref, atol=1e-4)
    np.testing.assert_allclose(ours.bound_multipliers, ref.v[-1], atol=1e-4)
