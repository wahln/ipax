"""``minimize_ipax`` on every Array-API backend, not just NumPy.

SciPy is moving its optimizers onto the Array API standard. Today
``scipy.optimize.minimize`` still converts ``x0`` with ``np.asarray`` before it
calls a custom method, but the adapter must not depend on that: it takes the
namespace, device and dtype from ``x0`` and keeps every per-iteration value in
it. Inputs that are NumPy by construction — the data ``LinearConstraint`` and
``Bounds`` coerce to float64 NumPy arrays, ``scipy.sparse`` matrices — are
converted once into ``x0``'s namespace. Calling ``minimize_ipax`` directly with
a non-NumPy ``x0`` is exactly what an array-API-aware ``minimize`` would do.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

from ipax.backend.namespace import array_namespace, device
from ipax.interfaces.scipy import _translate, minimize_ipax
from tests._helpers import array, assert_allclose, to_float, to_numpy
from tests.contracts.test_problem_contract import ProblemContract


def _same_namespace(xp, value) -> bool:
    return array_namespace(value).__name__ == xp.__name__


def _qp(xp):
    target = array(xp, [1.0, -2.0, 0.5, 3.0])

    def fun(x):
        return xp.sum((x - target) ** 2)

    def jac(x):
        return 2.0 * (x - target)

    return fun, jac, target


def _circle(xp):
    # x0² + x1² ≤ 1, with exact Jacobian and Hessian written in xp.
    mask = array(xp, [1.0, 1.0, 0.0, 0.0])
    return NonlinearConstraint(
        lambda x: xp.reshape(xp.sum(mask * x**2), (1,)),
        -np.inf,
        1.0,
        jac=lambda x: xp.reshape(2.0 * mask * x, (1, 4)),
        hess=lambda x, v: v[0] * 2.0 * xp.eye(4, dtype=x.dtype) * mask,
    )


def test_result_stays_in_the_input_namespace(namespace):
    fun, jac, target = _qp(namespace)
    x0 = array(namespace, [0.0, 0.0, 0.0, 0.0])
    res = minimize_ipax(fun, x0, jac=jac)

    assert res.success
    assert _same_namespace(namespace, res.x)
    assert res.x.dtype == x0.dtype
    assert _same_namespace(namespace, res.jac)
    assert_allclose(namespace, res.x, target, atol=1e-7, rtol=0.0)
    assert isinstance(res.fun, float)


def test_missing_jac_resolves_in_the_input_namespace(namespace, backend_name):
    fun, _, target = _qp(namespace)
    res = minimize_ipax(fun, array(namespace, [0.0, 0.0, 0.0, 0.0]))

    assert res.success
    assert_allclose(namespace, res.x, target, atol=1e-5, rtol=0.0)
    expected = "autodiff" if backend_name in ("torch", "jax") else "finite-diff"
    assert res.ipax_result.derivative_sources.gradient == expected


def test_nonlinear_constraint_with_exact_hessians(namespace):
    fun, jac, _ = _qp(namespace)
    res = minimize_ipax(
        fun,
        array(namespace, [0.0, 0.0, 0.0, 0.0]),
        jac=jac,
        hess=lambda x: 2.0 * namespace.eye(4, dtype=x.dtype),
        constraints=[_circle(namespace)],
    )

    assert res.success
    assert res.ipax_result.routes.hessian == "exact"
    corner = 1.0 / np.sqrt(5.0)
    assert_allclose(
        namespace, res.x, array(namespace, [corner, -2 * corner, 0.5, 3.0]), atol=1e-6
    )
    assert _same_namespace(namespace, res.v[0])
    assert _same_namespace(namespace, res.constr[0])
    assert to_float(res.v[0][0]) > 0.0
    assert _same_namespace(namespace, res.bound_multipliers)


def test_scipy_constraint_objects_are_converted_to_the_input_namespace(namespace):
    # LinearConstraint and Bounds coerce their data to float64 NumPy arrays.
    fun, jac, _ = _qp(namespace)
    A = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 0.0, 0.0]])
    res = minimize_ipax(
        fun,
        array(namespace, [0.0, 0.0, 0.0, 0.0]),
        jac=jac,
        bounds=Bounds([0.0, -np.inf, 0.0, -np.inf], [np.inf, -2.5, 0.25, np.inf]),
        constraints=[LinearConstraint(A, [2.0, -np.inf], [2.0, 4.0])],
    )

    assert res.success
    assert _same_namespace(namespace, res.x)
    assert _same_namespace(namespace, res.v[0])
    assert_allclose(namespace, res.constr[0][:1], array(namespace, [2.0]), atol=1e-8)
    assert res.constr_violation <= 1e-8
    x = to_numpy(res.x)
    assert x[1] <= -2.5 + 1e-8 and 0.0 - 1e-8 <= x[2] <= 0.25 + 1e-8


def test_translated_data_keeps_the_input_dtype(namespace):
    # LinearConstraint has already cast A to float64; float32 input stays float32.
    x0 = namespace.asarray([0.0, 0.0], dtype=namespace.float32)
    translation = _translate(
        lambda x: namespace.sum(x**2),
        x0,
        bounds=[(0.0, None), (None, 1.0)],
        constraints=[LinearConstraint(np.array([[1.0, 2.0]]), 0.0, 1.0)],
    )
    problem = translation.problem
    A, lower, upper = problem.linear_ineq()
    assert A.dtype == namespace.float32
    assert lower.dtype == upper.dtype == namespace.float32
    x_lower, x_upper = problem.bounds()
    assert x_lower.dtype == x_upper.dtype == namespace.float32
    assert _same_namespace(namespace, x_lower)


def test_dict_constraints_and_callback_see_the_input_namespace(namespace):
    seen = []

    def callback(intermediate_result):
        seen.append(intermediate_result.x)

    cons = [
        {"type": "ineq", "fun": lambda x: x[0] - 2 * x[1] + 2},
        {"type": "ineq", "fun": lambda x: -x[0] - 2 * x[1] + 6},
        {"type": "ineq", "fun": lambda x: -x[0] + 2 * x[1] + 2},
    ]
    res = minimize_ipax(
        lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2,
        array(namespace, [2.0, 0.0]),
        bounds=((0, None), (0, None)),
        constraints=cons,
        callback=callback,
    )

    assert res.success
    assert_allclose(namespace, res.x, array(namespace, [1.4, 1.7]), atol=1e-6)
    assert to_float(res.v[0][0]) == pytest.approx(-0.8, abs=1e-5)
    assert seen and all(_same_namespace(namespace, x) for x in seen)


def test_array_like_constraint_hessian_is_converted(namespace):
    fun, jac, _ = _qp(namespace)
    mask = array(namespace, [1.0, 1.0, 0.0, 0.0])

    def nested_list_hess(x, v):
        h = 2.0 * float(v[0])
        return [
            [h, 0.0, 0.0, 0.0],
            [0.0, h, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]

    con = NonlinearConstraint(
        lambda x: namespace.reshape(namespace.sum(mask * x**2), (1,)),
        -np.inf,
        1.0,
        jac=lambda x: namespace.reshape(2.0 * mask * x, (1, 4)),
        hess=nested_list_hess,
    )
    res = minimize_ipax(
        fun,
        array(namespace, [0.0, 0.0, 0.0, 0.0]),
        jac=jac,
        hess=lambda x: 2.0 * namespace.eye(4, dtype=x.dtype),
        constraints=con,
    )
    assert res.success
    assert res.ipax_result.routes.hessian == "exact"


def test_list_x0_falls_back_to_numpy_like_scipy():
    res = minimize_ipax(lambda x: float(np.sum((x - 1.0) ** 2)), [0, 0])
    assert isinstance(res.x, np.ndarray)
    assert res.x.dtype == np.float64


# ---------------------------------------------------------------------------
# Contract battery on every backend: the translated problem is a valid Problem
# ---------------------------------------------------------------------------


def _hs71_translation(xp):
    def fun(x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def jac(x):
        return xp.stack(
            [
                x[3] * (2 * x[0] + x[1] + x[2]),
                x[0] * x[3],
                x[0] * x[3] + 1.0,
                x[0] * (x[0] + x[1] + x[2]),
            ]
        )

    def hess(x):
        s = 2 * x[0] + x[1] + x[2]
        zero = x[0] * 0.0
        return xp.stack(
            [
                xp.stack([2 * x[3], x[3], x[3], s]),
                xp.stack([x[3], zero, zero, x[0]]),
                xp.stack([x[3], zero, zero, x[0]]),
                xp.stack([s, x[0], x[0], zero]),
            ]
        )

    def prod_hess(x):
        # ∂²(Πx)/∂xᵢ∂xⱼ = Πx / (xᵢ xⱼ) off the diagonal, 0 on it.
        n = x.shape[0]
        off = 1.0 - xp.eye(n, dtype=x.dtype)
        return xp.prod(x) / (x[:, None] * x[None, :]) * off

    con = NonlinearConstraint(
        lambda x: xp.stack([xp.prod(x), xp.sum(x**2)]),
        [25.0, 40.0],
        [np.inf, 40.0],
        jac=lambda x: xp.stack([xp.prod(x) / x, 2.0 * x]),
        hess=lambda x, v: v[0] * prod_hess(x) + v[1] * 2.0 * xp.eye(4, dtype=x.dtype),
    )
    lin = LinearConstraint(np.array([[1.0, -1.0, 0.0, 0.0]]), -1.0, 1.0)
    x0 = array(xp, [1.0, 5.0, 5.0, 1.0])
    translation = _translate(
        fun, x0, jac=jac, hess=hess, bounds=[(1.0, 5.0)] * 4, constraints=[con, lin]
    )
    return translation, array(xp, [1.3, 4.2, 3.9, 1.6]), hess, prod_hess


class TestTranslatedProblemContractAllBackends(ProblemContract):
    def make_problem(self, namespace):
        translation, x, _, _ = _hs71_translation(namespace)
        return translation.problem, x


def test_translated_hessian_matches_split_constraint_hessians(namespace):
    translation, x, hess, prod_hess = _hs71_translation(namespace)
    W = translation.problem.lagrangian_hessian(
        x, array(namespace, [0.7]), array(namespace, [0.3]), 2.0
    )
    # Equality Σx² = 40 → +0.7·2I; inequality 25 − Πx ≤ 0 → −0.3·∇²Πx.
    expected = (
        2.0 * hess(x) + 0.7 * 2.0 * namespace.eye(4, dtype=x.dtype) - 0.3 * prod_hess(x)
    )
    assert _same_namespace(namespace, W)
    assert_allclose(namespace, W, expected, rtol=1e-12, atol=1e-12)


def test_constraint_cache_does_not_serve_autodiff_a_stale_result(
    namespace, backend_name
):
    # Regression: the per-point cache used to key on values, so the autodiff
    # adapter's tracked copy of the iterate was served the untracked constraint
    # values cached for the equal-valued iterate — a wrong Jacobian, and the
    # solve stalled. Equality + inequality rows exercise the shared cache.
    if backend_name not in ("torch", "jax"):
        pytest.skip("needs an autodiff backend")
    cons = [
        {"type": "eq", "fun": lambda x: x[0] + x[1] - 3.0},
        {"type": "ineq", "fun": lambda x: x[0] - 2 * x[1] + 2},
    ]
    res = minimize_ipax(
        lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2,
        array(namespace, [2.0, 0.0]),
        constraints=cons,
    )
    sources = res.ipax_result.derivative_sources
    assert (sources.eq_jacobian, sources.ineq_jacobian) == ("autodiff", "autodiff")
    assert res.success
    assert_allclose(
        namespace, res.x, array(namespace, [4.0 / 3.0, 5.0 / 3.0]), atol=1e-6
    )


def _gpu_start():
    """A float64 GPU array from whichever GPU backend is usable, else skip."""
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy.zeros(4, dtype=cupy.float64)
    except Exception:  # any import/driver failure means no usable CuPy GPU
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return torch.zeros(4, dtype=torch.float64, device="cuda")
    except ImportError:
        pass
    pytest.skip("needs a CUDA device (CuPy or a CUDA build of PyTorch)")


def test_gpu_end_to_end_stays_on_the_device():
    x0 = _gpu_start()
    xp = array_namespace(x0)
    on = device(x0)
    target = xp.asarray([1.0, -2.0, 0.5, 3.0], dtype=xp.float64, device=on)
    mask = xp.asarray([1.0, 1.0, 0.0, 0.0], dtype=xp.float64, device=on)
    res = minimize_ipax(
        lambda x: xp.sum((x - target) ** 2),
        x0,
        jac=lambda x: 2.0 * (x - target),
        bounds=Bounds([-np.inf, -np.inf, 0.0, -np.inf], [np.inf, np.inf, 0.25, np.inf]),
        constraints=[
            NonlinearConstraint(
                lambda x: xp.reshape(xp.sum(mask * x**2), (1,)),
                -np.inf,
                1.0,
                jac=lambda x: xp.reshape(2.0 * mask * x, (1, 4)),
            ),
            LinearConstraint(np.array([[0.0, 0.0, 1.0, 1.0]]), 1.0, 1.0),
        ],
    )
    assert res.success, res.message
    for value in (res.x, res.v[0], res.v[1], res.constr[1], res.bound_multipliers):
        assert type(value) is type(x0)
        assert device(value) == on
    # (x2, x3): the line x2 + x3 = 1 minimizes (x2 − ½)² + (x3 − 3)² at x2 = −¾,
    # so the bound x2 ≥ 0 is active.
    corner = 1.0 / np.sqrt(5.0)
    np.testing.assert_allclose(
        to_numpy(res.x), [corner, -2 * corner, 0.0, 1.0], atol=1e-6
    )


# ---------------------------------------------------------------------------
# Audit round: paths that must hold on every backend, not only NumPy
# ---------------------------------------------------------------------------


def test_objective_hess_may_return_an_ipax_operator(namespace):
    from ipax.backend.operators import Dense

    fun, jac, target = _qp(namespace)
    res = minimize_ipax(
        fun,
        array(namespace, [0.0, 0.0, 0.0, 0.0]),
        jac=jac,
        hess=lambda x: Dense(2.0 * namespace.eye(4, dtype=x.dtype)),
    )
    assert res.success
    assert res.ipax_result.routes.hessian != "lbfgs"
    assert_allclose(namespace, res.x, target, atol=1e-7, rtol=0.0)


def test_values_already_in_the_space_are_passed_through(namespace):
    # No conversion call on values that already are x0's array type, dtype and
    # device: autodiff then never depends on how a backend's asarray treats a
    # tracked tensor.
    x = array(namespace, [1.0, 2.0])
    translation = _translate(lambda v: namespace.sum(v**2), x)
    assert translation.space.asarray(x) is x


def test_jac_true_and_hessp_on_every_backend(namespace):
    _, _, target = _qp(namespace)

    def fun_and_grad(x):
        return namespace.sum((x - target) ** 2), 2.0 * (x - target)

    res = minimize_ipax(
        fun_and_grad,
        array(namespace, [0.0, 0.0, 0.0, 0.0]),
        jac=True,
        hessp=lambda x, p: 2.0 * p,
    )
    assert res.success
    assert res.ipax_result.routes.hessian != "lbfgs"
    assert_allclose(namespace, res.x, target, atol=1e-7, rtol=0.0)
    assert _same_namespace(namespace, res.jac)


def test_sparse_linear_constraint_on_backends_with_a_sparse_adapter(
    namespace, backend_name
):
    if backend_name == "array_api_strict":
        pytest.skip("array-api-strict has no sparse adapter (sparse needs one)")
    import scipy.sparse as sp

    fun, jac, _ = _qp(namespace)
    A = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 0.0, 0.0]])
    dense = LinearConstraint(A, [2.0, -np.inf], [2.0, 4.0])
    sparse = LinearConstraint(sp.csr_matrix(A), [2.0, -np.inf], [2.0, 4.0])
    x0 = array(namespace, [0.0, 0.0, 0.0, 0.0])
    ref = minimize_ipax(fun, x0, jac=jac, constraints=dense)
    res = minimize_ipax(fun, x0, jac=jac, constraints=sparse)
    assert res.success
    assert _same_namespace(namespace, res.x)
    assert_allclose(namespace, res.x, ref.x, atol=1e-8, rtol=0.0)


def test_float32_solve_stays_float32_end_to_end(namespace):
    target = namespace.asarray([1.0, -2.0], dtype=namespace.float32)
    res = minimize_ipax(
        lambda x: namespace.sum((x - target) ** 2),
        namespace.zeros(2, dtype=namespace.float32),
        jac=lambda x: 2.0 * (x - target),
        bounds=Bounds([-np.inf, -1.5], [np.inf, np.inf]),
        constraints=[LinearConstraint(np.array([[1.0, 1.0]]), -np.inf, 1.0)],
        tol=1e-4,
    )
    assert res.success, res.message
    assert res.x.dtype == namespace.float32
    assert res.v[0].dtype == res.bound_multipliers.dtype == namespace.float32
    assert_allclose(
        namespace,
        res.x,
        namespace.asarray([1.0, -1.5], dtype=namespace.float32),
        atol=1e-3,
        rtol=0.0,
    )
