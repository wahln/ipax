"""Unit tests for the SciPy-compatible ``minimize_ipax`` adapter.

The adapter translates the ``scipy.optimize.minimize`` calling convention
(``fun``/``jac``/``hess``/``hessp``, ``Bounds`` or ``(min, max)`` pairs,
``LinearConstraint``/``NonlinearConstraint``/dict constraints, ``callback``,
``tol``/``options``) onto a :class:`~ipax.FunctionProblem` and maps the
:class:`~ipax.Result` back to an ``OptimizeResult``. These tests use NumPy,
which is what SciPy's ``minimize`` hands a custom method today; the
multi-backend behaviour and the ``Problem`` contract battery on the
translated problem live in ``test_scipy_interface_backends.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("scipy")  # the adapter is SciPy's calling convention

import scipy.sparse as sp
from scipy.optimize import (
    BFGS,
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    OptimizeResult,
    OptimizeWarning,
    minimize,
    rosen,
    rosen_der,
    rosen_hess,
    rosen_hess_prod,
)
from scipy.sparse.linalg import aslinearoperator

from ipax import Options
from ipax.interfaces.scipy import STATUS_CODES, _translate, minimize_ipax

# ---------------------------------------------------------------------------
# Shared fixtures: a small separable QP with a mix of constraint kinds
# ---------------------------------------------------------------------------

TARGET = np.array([1.0, -2.0, 0.5, 3.0])


def _qp(x):
    return float(np.sum((x - TARGET) ** 2))


def _qp_grad(x):
    return 2.0 * (x - TARGET)


def _qp_hess(x):
    del x
    return 2.0 * np.eye(4)


def _mixed_linear_constraint() -> LinearConstraint:
    # One equality row, one two-sided row, one upper-only row, one free row.
    A = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    return LinearConstraint(A, [2.0, -0.5, -np.inf, -np.inf], [2.0, 0.5, 0.2, np.inf])


def _circle() -> NonlinearConstraint:
    # x0² + x1² ≤ 1 (upper side only).
    return NonlinearConstraint(
        lambda x: np.array([x[0] ** 2 + x[1] ** 2]),
        -np.inf,
        1.0,
        jac=lambda x: np.array([[2.0 * x[0], 2.0 * x[1], 0.0, 0.0]]),
        hess=lambda x, v: v[0] * np.diag([2.0, 2.0, 0.0, 0.0]),
    )


# ---------------------------------------------------------------------------
# Entry points: standalone and as ``minimize(method=...)``
# ---------------------------------------------------------------------------


def test_minimize_method_callable_solves_rosenbrock():
    res = minimize(rosen, np.array([-1.2, 1.0]), jac=rosen_der, method=minimize_ipax)

    assert isinstance(res, OptimizeResult)
    assert res.success
    assert res.status == STATUS_CODES["optimal"]
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-6)
    assert res.fun == pytest.approx(0.0, abs=1e-10)
    np.testing.assert_allclose(res.jac, rosen_der(res.x))
    assert res.nit > 0
    assert res.nfev > 0
    assert res.njev > 0
    assert isinstance(res.message, str) and res.message


def test_standalone_matches_minimize_method_bitwise():
    x0 = np.array([-1.2, 1.0])
    direct = minimize_ipax(rosen, x0, jac=rosen_der)
    via_scipy = minimize(rosen, x0, jac=rosen_der, method=minimize_ipax)

    np.testing.assert_array_equal(direct.x, via_scipy.x)
    assert direct.nit == via_scipy.nit


@pytest.mark.parametrize("entry", ["standalone", "minimize"])
def test_jac_true_means_fun_returns_value_and_gradient(entry):
    calls = {"n": 0}

    def fun_and_grad(x):
        calls["n"] += 1
        return rosen(x), rosen_der(x)

    x0 = np.array([-1.2, 1.0])
    if entry == "standalone":
        res = minimize_ipax(fun_and_grad, x0, jac=True)
    else:
        res = minimize(fun_and_grad, x0, jac=True, method=minimize_ipax)

    assert res.success
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-6)
    assert np.ndim(res.fun) == 0
    assert calls["n"] > 0


def test_args_reach_fun_jac_and_hess():
    seen = set()

    def fun(x, a, b):
        seen.add(("fun", a, b))
        return float(np.sum((x - a) ** 2)) + b

    def jac(x, a, b):
        seen.add(("jac", a, b))
        return 2.0 * (x - a)

    def hess(x, a, b):
        seen.add(("hess", a, b))
        return 2.0 * np.eye(x.shape[0])

    res = minimize(
        fun, np.zeros(3), args=(1.5, 4.0), jac=jac, hess=hess, method=minimize_ipax
    )

    assert res.success
    np.testing.assert_allclose(res.x, [1.5, 1.5, 1.5], atol=1e-7)
    assert res.fun == pytest.approx(4.0)
    assert seen == {("fun", 1.5, 4.0), ("jac", 1.5, 4.0), ("hess", 1.5, 4.0)}


def test_single_non_tuple_arg_is_wrapped_like_scipy():
    res = minimize_ipax(lambda x, a: float(np.sum((x - a) ** 2)), np.zeros(2), args=2.0)
    np.testing.assert_allclose(res.x, [2.0, 2.0], atol=1e-6)


@pytest.mark.parametrize("jac", [None, "2-point", "3-point", "cs"])
def test_missing_or_fd_jac_falls_back_to_ipax_derivatives(jac):
    res = minimize_ipax(_qp, np.zeros(4), jac=jac)

    assert res.success
    np.testing.assert_allclose(res.x, TARGET, atol=1e-5)
    assert res.ipax_result.derivative_sources.gradient == "finite-diff"
    assert res.njev == 0


def test_integer_x0_is_promoted_and_2d_x0_rejected():
    res = minimize_ipax(_qp, [0, 0, 0, 0], jac=_qp_grad)
    assert res.x.dtype == np.float64

    with pytest.raises(ValueError, match="one dimension"):
        minimize_ipax(_qp, np.zeros((2, 2)), jac=_qp_grad)


def test_result_exposes_ipax_result_and_status_name():
    res = minimize_ipax(_qp, np.zeros(4), jac=_qp_grad)
    assert res.ipax_status == "optimal"
    assert res.ipax_result.status.value == "optimal"
    # Result.n_iter counts history rows incl. the starting point; SciPy's nit
    # counts iterations performed.
    assert res.nit == res.ipax_result.history[-1].iteration
    assert res.ipax_result.n_iter == res.nit + 1


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bounds",
    [
        Bounds([0.0, -np.inf, 0.0, -np.inf], [np.inf, -2.5, 0.25, np.inf]),
        [(0.0, None), (None, -2.5), (0.0, 0.25), (None, None)],
    ],
    ids=["Bounds", "pairs"],
)
def test_bounds_object_and_pairs_activate_expected_sides(bounds):
    res = minimize_ipax(
        _qp, np.array([1.0, -3.0, 0.1, 0.0]), jac=_qp_grad, bounds=bounds
    )

    assert res.success
    np.testing.assert_allclose(res.x, [1.0, -2.5, 0.25, 3.0], atol=1e-6)
    # trust-constr sign: z_U − z_L, so an active upper side is positive.
    assert res.bound_multipliers[1] == pytest.approx(1.0, abs=1e-5)
    assert res.bound_multipliers[2] == pytest.approx(0.5, abs=1e-5)
    assert res.bound_multipliers[0] == pytest.approx(0.0, abs=1e-6)


def test_bounds_with_wrong_length_are_rejected():
    with pytest.raises(ValueError, match="bounds"):
        minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, bounds=[(0.0, 1.0)] * 3)


# ---------------------------------------------------------------------------
# Constraint translation
# ---------------------------------------------------------------------------


def test_mixed_linear_constraint_matches_trust_constr():
    con = _mixed_linear_constraint()
    res = minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, constraints=con)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = minimize(
            _qp,
            np.zeros(4),
            jac=_qp_grad,
            hess=_qp_hess,
            method="trust-constr",
            constraints=con,
            options={"gtol": 1e-12, "xtol": 1e-14},
        )

    assert res.success
    np.testing.assert_allclose(res.x, ref.x, atol=1e-6)
    assert len(res.v) == 1
    np.testing.assert_allclose(res.v[0], ref.v[0], atol=1e-5)
    np.testing.assert_allclose(res.constr[0], con.A @ res.x, atol=1e-12)
    assert res.constr_violation <= 1e-8


def test_sparse_linear_constraint_matches_dense():
    dense = _mixed_linear_constraint()
    sparse = LinearConstraint(sp.csr_matrix(dense.A), dense.lb, dense.ub)

    res_dense = minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, constraints=dense)
    res_sparse = minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, constraints=sparse)

    assert res_sparse.success
    np.testing.assert_allclose(res_sparse.x, res_dense.x, atol=1e-8)
    np.testing.assert_allclose(res_sparse.v[0], res_dense.v[0], atol=1e-6)


def test_nonlinear_constraint_with_hessians_uses_exact_hessian():
    res = minimize_ipax(
        _qp, np.zeros(4), jac=_qp_grad, hess=_qp_hess, constraints=[_circle()]
    )

    assert res.success
    assert res.ipax_result.routes.hessian == "exact"
    # Projection of (1, -2) onto the unit disc; x2, x3 are free.
    expected = np.array([1.0, -2.0]) / np.sqrt(5.0)
    np.testing.assert_allclose(res.x[:2], expected, atol=1e-6)
    np.testing.assert_allclose(res.x[2:], TARGET[2:], atol=1e-6)
    # Active upper side ⇒ positive multiplier; stationarity on x0:
    # 2(x0 − 1) + v·2x0 = 0.
    v = res.v[0][0]
    assert v > 0
    assert 2.0 * (res.x[0] - 1.0) + 2.0 * v * res.x[0] == pytest.approx(0.0, abs=1e-6)
    assert res.nhev > 0


def test_missing_constraint_hessian_falls_back_to_lbfgs_with_warning():
    con = NonlinearConstraint(
        lambda x: np.array([x[0] ** 2 + x[1] ** 2]),
        -np.inf,
        1.0,
        jac=lambda x: np.array([[2.0 * x[0], 2.0 * x[1], 0.0, 0.0]]),
    )
    with pytest.warns(OptimizeWarning, match="L-BFGS"):
        res = minimize_ipax(
            _qp, np.zeros(4), jac=_qp_grad, hess=_qp_hess, constraints=con
        )

    assert res.success
    assert res.ipax_result.routes.hessian == "lbfgs"


def test_no_hessian_anywhere_uses_lbfgs_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        res = minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, constraints=[_circle()])
    assert res.ipax_result.routes.hessian == "lbfgs"


@pytest.mark.parametrize("hess", [BFGS(), "2-point"], ids=["update-strategy", "fd"])
def test_quasi_newton_or_fd_hess_is_replaced_by_lbfgs_with_warning(hess):
    with pytest.warns(OptimizeWarning, match="L-BFGS"):
        res = minimize_ipax(rosen, np.array([-1.2, 1.0]), jac=rosen_der, hess=hess)
    assert res.success
    assert res.ipax_result.routes.hessian == "lbfgs"


def test_hessp_supplies_matrix_free_exact_hessian():
    res = minimize_ipax(
        rosen, np.array([-1.2, 1.0, 0.8]), jac=rosen_der, hessp=rosen_hess_prod
    )
    assert res.success
    assert res.ipax_result.routes.hessian != "lbfgs"
    np.testing.assert_allclose(res.x, np.ones(3), atol=1e-6)


def test_dense_hess_matches_rosen_hess():
    res = minimize_ipax(rosen, np.array([-1.2, 1.0]), jac=rosen_der, hess=rosen_hess)
    assert res.success
    assert res.ipax_result.routes.hessian == "exact"
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-7)


def test_sparse_constraint_jacobian_matches_dense():
    def jac_sparse(x):
        return sp.csr_matrix(np.array([[2.0 * x[0], 2.0 * x[1], 0.0, 0.0]]))

    con = NonlinearConstraint(
        lambda x: np.array([x[0] ** 2 + x[1] ** 2]), -np.inf, 1.0, jac=jac_sparse
    )
    res_sparse = minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, constraints=con)
    res_dense = minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, constraints=[_circle()])

    assert res_sparse.success
    np.testing.assert_allclose(res_sparse.x, res_dense.x, atol=1e-6)


def test_dict_constraints_use_scipy_sign_and_args():
    # SLSQP doc example: 'ineq' means fun(x) ≥ 0.
    cons = [
        {"type": "ineq", "fun": lambda x, k: x[0] - 2 * x[1] + k, "args": (2.0,)},
        {"type": "ineq", "fun": lambda x: -x[0] - 2 * x[1] + 6},
        {
            "type": "ineq",
            "fun": lambda x: -x[0] + 2 * x[1] + 2,
            "jac": lambda x: np.array([-1.0, 2.0]),
        },
    ]
    res = minimize_ipax(
        lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2,
        np.array([2.0, 0.0]),
        bounds=((0, None), (0, None)),
        constraints=cons,
    )
    assert res.success
    np.testing.assert_allclose(res.x, [1.4, 1.7], atol=1e-6)
    # Active lower side (fun ≥ 0) ⇒ negative multiplier, as trust-constr reports.
    assert res.v[0][0] == pytest.approx(-0.8, abs=1e-5)
    assert res.v[1][0] == pytest.approx(0.0, abs=1e-6)


def test_dict_equality_constraint():
    con = {"type": "eq", "fun": lambda x: np.array([x[0] + x[1] - 1.0])}
    res = minimize_ipax(lambda x: float(x @ x), np.array([3.0, -1.0]), constraints=con)
    assert res.success
    np.testing.assert_allclose(res.x, [0.5, 0.5], atol=1e-6)
    assert res.v[0][0] == pytest.approx(-1.0, abs=1e-5)


def test_constraint_values_and_multipliers_follow_input_order():
    cons = [
        _circle(),
        {"type": "eq", "fun": lambda x: np.array([x[2] - 0.0])},
        _mixed_linear_constraint(),
    ]
    res = minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, constraints=cons)
    assert res.success
    assert [np.shape(v) for v in res.v] == [(1,), (1,), (4,)]
    assert [np.shape(c) for c in res.constr] == [(1,), (1,), (4,)]
    assert res.constr[1][0] == pytest.approx(0.0, abs=1e-8)


def test_unknown_dict_constraint_type_is_rejected():
    with pytest.raises(ValueError, match="type"):
        minimize_ipax(_qp, np.zeros(4), constraints={"type": "le", "fun": _qp})


def test_keep_feasible_on_constraints_warns():
    con = LinearConstraint(np.ones((1, 4)), -np.inf, 1.0, keep_feasible=True)
    with pytest.warns(OptimizeWarning, match="keep_feasible"):
        minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, constraints=con)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


def test_intermediate_result_callback_receives_optimize_result():
    seen = []

    def callback(intermediate_result):
        seen.append(intermediate_result)

    res = minimize(
        rosen,
        np.array([-1.2, 1.0]),
        jac=rosen_der,
        method=minimize_ipax,
        callback=callback,
    )
    assert res.success
    assert seen
    assert all(isinstance(r, OptimizeResult) for r in seen)
    assert seen[-1].x.shape == (2,)
    assert np.isfinite(seen[-1].fun)


def test_legacy_callback_receives_copy_of_x():
    seen = []

    def callback(xk):
        seen.append(xk)
        xk[:] = 1e6  # must not disturb the solve

    res = minimize_ipax(rosen, np.array([-1.2, 1.0]), jac=rosen_der, callback=callback)
    assert res.success
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-6)
    assert seen


def test_stop_iteration_in_callback_stops_the_solve():
    calls = {"n": 0}

    def callback(intermediate_result):
        calls["n"] += 1
        if calls["n"] == 3:
            raise StopIteration

    res = minimize_ipax(rosen, np.array([-1.2, 1.0]), jac=rosen_der, callback=callback)
    assert not res.success
    assert res.status == STATUS_CODES["stopped"]
    assert res.ipax_status == "stopped"
    assert calls["n"] == 3


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def test_maxiter_maps_to_max_iter():
    res = minimize(
        rosen,
        np.array([-1.2, 1.0]),
        jac=rosen_der,
        method=minimize_ipax,
        options={"maxiter": 2},
    )
    assert not res.success
    assert res.status == STATUS_CODES["max_iter"]
    assert res.nit == 2


def test_tol_sets_kkt_tolerances():
    loose = minimize(
        rosen, np.array([-1.2, 1.0]), jac=rosen_der, method=minimize_ipax, tol=1e-2
    )
    tight = minimize(rosen, np.array([-1.2, 1.0]), jac=rosen_der, method=minimize_ipax)
    assert loose.success
    assert loose.ipax_result.kkt_error <= 1e-2
    assert loose.nit < tight.nit


def test_ipax_options_and_field_overrides():
    base = Options(globalization="breedveld", max_iter=5)
    res = minimize_ipax(
        _qp, np.zeros(4), jac=_qp_grad, options={"ipax_options": base, "maxiter": 50}
    )
    assert res.ipax_result.routes.globalization == "breedveld"
    assert res.success  # maxiter=50 overrode max_iter=5

    res = minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, globalization="breedveld")
    assert res.ipax_result.routes.globalization == "breedveld"


def test_unknown_option_warns_like_scipy():
    with pytest.warns(OptimizeWarning, match="Unknown solver options: bogus"):
        minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, options={"bogus": 1})


def test_disp_enables_progress_logging(caplog):
    with caplog.at_level("INFO", logger="ipax"):
        minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, options={"disp": True})
    assert caplog.records


# ---------------------------------------------------------------------------
# Contract: the translated problem is a well-formed ipax Problem
# ---------------------------------------------------------------------------


def _hs71_like_translation():
    def fun(x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def jac(x):
        return np.array(
            [
                x[3] * (2 * x[0] + x[1] + x[2]),
                x[0] * x[3],
                x[0] * x[3] + 1.0,
                x[0] * (x[0] + x[1] + x[2]),
            ]
        )

    def hess(x):
        s = 2 * x[0] + x[1] + x[2]
        return np.array(
            [
                [2 * x[3], x[3], x[3], s],
                [x[3], 0.0, 0.0, x[0]],
                [x[3], 0.0, 0.0, x[0]],
                [s, x[0], x[0], 0.0],
            ]
        )

    def c(x):
        return np.array([np.prod(x), np.sum(x**2)])

    def c_jac(x):
        p = np.prod(x)
        return np.vstack([p / x, 2.0 * x])

    def c_hess(x, v):
        H = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                if i != j:
                    H[i, j] = np.prod([x[k] for k in range(4) if k not in (i, j)])
        return v[0] * H + v[1] * 2.0 * np.eye(4)

    con = NonlinearConstraint(c, [25.0, 40.0], [np.inf, 40.0], jac=c_jac, hess=c_hess)
    lin = LinearConstraint(np.array([[1.0, -1.0, 0.0, 0.0]]), -1.0, 1.0)
    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    return (
        _translate(
            fun,
            x0,
            jac=jac,
            hess=hess,
            bounds=[(1.0, 5.0)] * 4,
            constraints=[con, lin],
        ),
        np.array([1.3, 4.2, 3.9, 1.6]),
    )


def test_translated_hessian_matches_split_constraint_hessians():
    translation, x = _hs71_like_translation()
    problem = translation.problem
    # Rows: equality (Σx² = 40); inequalities: 25 − Πx ≤ 0 (the nonlinear
    # lower side). The linear block carries no Hessian term.
    y_eq = np.array([0.7])
    y_ineq = np.array([0.3])
    W = problem.lagrangian_hessian(x, y_eq, y_ineq, 2.0)

    H_prod = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i != j:
                H_prod[i, j] = np.prod([x[k] for k in range(4) if k not in (i, j)])
    s = 2 * x[0] + x[1] + x[2]
    H_f = np.array(
        [
            [2 * x[3], x[3], x[3], s],
            [x[3], 0.0, 0.0, x[0]],
            [x[3], 0.0, 0.0, x[0]],
            [s, x[0], x[0], 0.0],
        ]
    )
    expected = 2.0 * H_f + 0.7 * 2.0 * np.eye(4) - 0.3 * H_prod
    np.testing.assert_allclose(np.asarray(W), expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Hessian assembly variants and input errors
# ---------------------------------------------------------------------------


def _circle_with_hess(hess):
    return NonlinearConstraint(
        lambda x: np.array([x[0] ** 2 + x[1] ** 2]),
        -np.inf,
        1.0,
        jac=lambda x: np.array([[2.0 * x[0], 2.0 * x[1], 0.0, 0.0]]),
        hess=hess,
    )


@pytest.mark.parametrize(
    ("objective_hess", "constraint_hess"),
    [
        (
            lambda x: sp.csr_matrix(2.0 * np.eye(4)),
            lambda x, v: sp.csr_matrix(v[0] * np.diag([2.0, 2.0, 0.0, 0.0])),
        ),
        (
            lambda x: sp.csr_matrix(2.0 * np.eye(4)),
            lambda x, v: v[0] * np.diag([2.0, 2.0, 0.0, 0.0]),
        ),
        (
            lambda x: aslinearoperator(2.0 * np.eye(4)),
            lambda x, v: v[0] * np.diag([2.0, 2.0, 0.0, 0.0]),
        ),
    ],
    ids=["sparse+sparse", "sparse+dense", "operator+dense"],
)
def test_sparse_and_operator_hessians_match_dense(objective_hess, constraint_hess):
    dense = minimize_ipax(
        _qp, np.zeros(4), jac=_qp_grad, hess=_qp_hess, constraints=[_circle()]
    )
    res = minimize_ipax(
        _qp,
        np.zeros(4),
        jac=_qp_grad,
        hess=objective_hess,
        constraints=[_circle_with_hess(constraint_hess)],
    )
    assert res.success
    assert res.ipax_result.routes.hessian == "exact"
    np.testing.assert_allclose(res.x, dense.x, atol=1e-7)


def test_zero_sigma_skips_the_objective_hessian():
    translation, x = _hs71_like_translation()
    before = translation.counts.nhev
    translation.problem.lagrangian_hessian(x, np.array([0.5]), np.array([0.2]), 0.0)
    assert translation.counts.nhev == before


def test_operator_constraint_jacobian_is_rejected():
    con = NonlinearConstraint(
        lambda x: np.array([x[0]]),
        -np.inf,
        1.0,
        jac=lambda x: aslinearoperator(np.array([[1.0, 0.0, 0.0, 0.0]])),
    )
    with pytest.raises(TypeError, match="LinearOperator"):
        minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, constraints=con)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"hess": 3.0}, TypeError, "hess"),
        (
            {"constraints": LinearConstraint(np.ones((1, 4)), 1.0, 0.0)},
            ValueError,
            "lb > ub",
        ),
        (
            {"constraints": LinearConstraint(np.ones((1, 3)), 0.0, 1.0)},
            ValueError,
            "columns",
        ),
        ({"constraints": [object()]}, TypeError, "unsupported constraint"),
        ({"bounds": Bounds(np.zeros(3), np.ones(3))}, ValueError, "bounds"),
        ({"options": {"ipax_options": {"max_iter": 3}}}, TypeError, "ipax.Options"),
    ],
    ids=[
        "hess",
        "lb>ub",
        "columns",
        "constraint-type",
        "Bounds-length",
        "ipax_options",
    ],
)
def test_invalid_inputs_are_rejected(kwargs, error, match):
    with pytest.raises(error, match=match):
        minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, **kwargs)


def test_integer_disp_sets_verbosity(caplog):
    with caplog.at_level("DEBUG", logger="ipax"):
        res = minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, disp=2)
    assert res.success
    assert any(r.levelname == "DEBUG" for r in caplog.records)


def test_supplied_jacobians_are_kept_when_another_constraint_has_none():
    calls = {"jac": 0}

    def jac(x):
        calls["jac"] += 1
        return np.array([-1.0, 2.0])

    cons = [
        {"type": "ineq", "fun": lambda x: x[0] - 2 * x[1] + 2},  # no jac
        {"type": "ineq", "fun": lambda x: -x[0] + 2 * x[1] + 2, "jac": jac},
        NonlinearConstraint(lambda x: -x[0] - 2 * x[1], -6.0, np.inf),  # '2-point'
    ]
    with pytest.warns(OptimizeWarning, match="central differences"):
        res = minimize_ipax(
            lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2,
            np.array([2.0, 0.0]),
            jac=lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] - 2.5)]),
            bounds=((0, None), (0, None)),
            constraints=cons,
        )
    assert res.success
    np.testing.assert_allclose(res.x, [1.4, 1.7], atol=1e-6)
    assert calls["jac"] > 0
    assert res.ipax_result.derivative_sources.ineq_jacobian == "analytic"


def test_no_constraints_none_is_accepted():
    res = minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, constraints=None)
    np.testing.assert_allclose(res.x, TARGET, atol=1e-6)


def test_zero_sigma_without_constraint_curvature_is_a_zero_hessian():
    translation = _translate(
        rosen, np.array([-1.2, 1.0]), jac=rosen_der, hess=rosen_hess
    )
    W = translation.problem.lagrangian_hessian(
        np.array([0.3, 0.4]), np.zeros(0), np.zeros(0), 0.0
    )
    np.testing.assert_array_equal(np.asarray(W), np.zeros((2, 2)))
    assert translation.counts.nhev == 0


# ---------------------------------------------------------------------------
# Regressions from the invariant audit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "constraints",
    [
        (),
        [_circle()],
        [LinearConstraint(np.ones((1, 4)), -np.inf, 1.0)],
    ],
    ids=["none", "nonlinear", "linear"],
)
def test_bounds_with_lb_above_ub_are_rejected(constraints):
    # minimize() skips its own _validate_bounds for a callable method, and
    # ipax.solve's early infeasible-bounds exit has no multipliers, which used
    # to crash the per-constraint multiplier mapping.
    with pytest.raises(ValueError, match="lb > ub"):
        minimize_ipax(
            _qp,
            np.zeros(4),
            jac=_qp_grad,
            bounds=Bounds(np.ones(4), np.zeros(4)),
            constraints=constraints,
        )


def test_result_without_multipliers_maps_to_zero_multipliers():
    from ipax import Result, Status
    from ipax.interfaces.scipy import _to_optimize_result

    translation = _translate(
        _qp,
        np.zeros(4),
        jac=_qp_grad,
        constraints=[_circle(), _mixed_linear_constraint()],
    )
    early = Result(status=Status.INFEASIBLE, x=np.zeros(4), objective=0.0)
    res = _to_optimize_result(early, translation)
    assert [v.tolist() for v in res.v] == [[0.0], [0.0, 0.0, 0.0, 0.0]]
    assert res.nit == 0
    assert res.status == STATUS_CODES["infeasible"]


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_equality_rows_are_rejected(value):
    con = LinearConstraint(np.ones((1, 4)), value, value)
    with pytest.raises(ValueError, match="infinite"):
        minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, constraints=con)


def test_status_codes_cover_every_ipax_status():
    from ipax import Status

    assert set(STATUS_CODES) == {s.value for s in Status}
    assert len(set(STATUS_CODES.values())) == len(STATUS_CODES)


def test_trust_constr_style_callback_receives_x_and_state():
    seen = []

    def callback(xk, state):
        seen.append((xk.copy(), state.nit))

    res = minimize_ipax(rosen, np.array([-1.2, 1.0]), jac=rosen_der, callback=callback)
    assert res.success
    assert seen
    assert seen[-1][1] == res.nit


def test_keep_feasible_warning_points_at_the_caller():
    con = LinearConstraint(np.ones((1, 4)), -np.inf, 1.0, keep_feasible=True)
    with pytest.warns(OptimizeWarning, match="keep_feasible") as record:
        minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, constraints=con)
    assert record[0].filename == __file__


def test_zero_hessian_keeps_the_problem_dtype():
    x0 = np.array([-1.2, 1.0], dtype=np.float32)
    translation = _translate(rosen, x0, jac=rosen_der, hess=rosen_hess)
    W = translation.problem.lagrangian_hessian(x0, np.zeros(0), np.zeros(0), 0.0)
    assert np.asarray(W).dtype == np.float32


# ---------------------------------------------------------------------------
# Regressions from review round 1
# ---------------------------------------------------------------------------


def test_legacy_callback_with_defaulted_second_parameter_gets_only_x():
    # A second parameter with a default is not trust-constr's (xk, state).
    seen = []

    def callback(xk, history=seen):
        history.append(xk.copy())

    res = minimize_ipax(rosen, np.array([-1.2, 1.0]), jac=rosen_der, callback=callback)
    assert res.success
    assert seen
    assert all(isinstance(x, np.ndarray) and x.shape == (2,) for x in seen)


def test_trust_constr_style_callback_returning_true_stops_the_solve():
    # trust-constr ends the solve when callback(xk, state) returns True.
    calls = {"n": 0}

    def callback(xk, state):
        calls["n"] += 1
        return calls["n"] == 3

    res = minimize_ipax(rosen, np.array([-1.2, 1.0]), jac=rosen_der, callback=callback)
    assert not res.success
    assert res.status == STATUS_CODES["stopped"]
    assert calls["n"] == 3


def test_legacy_callback_return_value_is_ignored_like_scipy():
    res = minimize_ipax(
        rosen, np.array([-1.2, 1.0]), jac=rosen_der, callback=lambda xk: True
    )
    assert res.success


def test_array_like_constraint_hessian_is_accepted():
    # SciPy accepts array-likes (here a nested list) from a constraint hess.
    con = NonlinearConstraint(
        lambda x: np.array([x[0] ** 2 + x[1] ** 2]),
        -np.inf,
        1.0,
        jac=lambda x: np.array([[2.0 * x[0], 2.0 * x[1], 0.0, 0.0]]),
        hess=lambda x, v: [
            [2.0 * v[0], 0.0, 0.0, 0.0],
            [0.0, 2.0 * v[0], 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    )
    reference = minimize_ipax(
        _qp, np.zeros(4), jac=_qp_grad, hess=_qp_hess, constraints=[_circle()]
    )
    res = minimize_ipax(_qp, np.zeros(4), jac=_qp_grad, hess=_qp_hess, constraints=con)
    assert res.success
    assert res.ipax_result.routes.hessian == "exact"
    np.testing.assert_allclose(res.x, reference.x, atol=1e-10)


@pytest.mark.parametrize("jac_true", [False, True], ids=["callable-jac", "jac=True"])
def test_counters_include_the_final_gradient(jac_true):
    calls = {"fun": 0, "jac": 0}

    def fun(x):
        calls["fun"] += 1
        return rosen(x)

    def jac(x):
        calls["jac"] += 1
        return rosen_der(x)

    def fun_and_grad(x):
        calls["fun"] += 1
        calls["jac"] += 1
        return rosen(x), rosen_der(x)

    if jac_true:
        res = minimize_ipax(fun_and_grad, np.array([-1.2, 1.0]), jac=True)
        assert res.nfev == calls["fun"]
    else:
        res = minimize_ipax(fun, np.array([-1.2, 1.0]), jac=jac)
        assert res.nfev == calls["fun"]
        assert res.njev == calls["jac"]


def test_scalar_x0_becomes_a_one_element_vector_like_scipy():
    res = minimize_ipax(lambda x: float((x[0] - 2.0) ** 2), 0.0)
    assert res.x.shape == (1,)
    assert res.x[0] == pytest.approx(2.0, abs=1e-6)


def test_sparse_and_device_dense_jacobians_cannot_be_stacked():
    # A scipy.sparse Jacobian lives on the host; a dense one that NumPy cannot
    # read (CuPy / CUDA tensors refuse np.asarray) gets a clear error instead
    # of an obscure conversion failure.
    from ipax.interfaces.scipy import _on_host

    class DeviceOnly:
        def __array__(self, *args, **kwargs):
            raise TypeError("implicit host conversion is not allowed")

    with pytest.raises(TypeError, match=r"scipy.sparse constraint Jacobian"):
        _on_host(DeviceOnly())
