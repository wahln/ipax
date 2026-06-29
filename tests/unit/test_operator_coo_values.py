"""``coo_values`` must equal ``to_coo()[2]`` exactly (values-only refactor #2).

The sparse solver caches the COO row/column structure across interior-point
iterations and recomputes only the value vector via ``coo_values``. That is sound
only if ``coo_values`` reproduces ``to_coo``'s value array element-for-element, in
the same order — these contract tests pin that for the operator hierarchy and the
condensed/saddle KKT blocks, on every backend (pure Array API).
"""

from __future__ import annotations

from ipax.backend.operators import Dense, Diagonal, Identity, VStack
from ipax.ipm.hessian import LBFGSOperator
from ipax.ipm.kkt import build_condensed_operator, build_saddle_operator
from ipax.linalg.regularize import RegularizationState
from ipax.options import LBFGSOptions
from tests._helpers import array, assert_allclose, float_dtype


def _assert_values_match(namespace, op, tol, *, like=None):
    expected = op.to_coo(like)[2] if like is not None else op.to_coo()[2]
    actual = op.coo_values(like) if like is not None else op.coo_values()
    assert int(actual.shape[0]) == int(expected.shape[0])
    assert_allclose(namespace, actual, expected, **tol)


def _lbfgs(namespace, n):
    op = LBFGSOperator(n, LBFGSOptions(memory=5))
    op.update(
        array(namespace, [0.1 * (k + 1) for k in range(n)]),
        array(namespace, [0.2 * (k + 1) for k in range(n)]),
    )
    return op


def test_dense_coo_values_match(namespace, tol):
    op = Dense(array(namespace, [[2.0, -1.0, 0.5], [0.0, 3.0, 4.0]]))
    _assert_values_match(namespace, op, tol)


def test_diagonal_coo_values_match(namespace, tol):
    _assert_values_match(namespace, Diagonal(array(namespace, [2.0, -3.0, 4.0])), tol)


def test_identity_coo_values_match(namespace, tol):
    like = array(namespace, [0.0, 0.0, 0.0])
    _assert_values_match(namespace, Identity(3), tol, like=like)


def test_vstack_coo_values_match(namespace, tol):
    op = VStack(
        (
            Dense(array(namespace, [[1.0, 2.0], [3.0, 4.0]])),
            Diagonal(array(namespace, [5.0, 6.0])),
        )
    )
    _assert_values_match(namespace, op, tol)


def test_condensed_assemblable_with_inequalities_coo_values_match(namespace, tol):
    w = Dense(array(namespace, [[3.0, 1.0], [1.0, 2.0]]))
    sigma_x = Diagonal(array(namespace, [0.5, 0.25]))
    sigma_s = Diagonal(array(namespace, [1.5, 2.0]))
    ineq = Dense(array(namespace, [[1.0, 0.0], [0.0, 1.0]]))
    op = build_condensed_operator(
        w, sigma_x, sigma_s, ineq, RegularizationState(delta_w=1e-3)
    )
    _assert_values_match(namespace, op, tol)


def test_condensed_lbfgs_with_inequalities_coo_values_match(namespace, tol):
    w = _lbfgs(namespace, 2)
    sigma_x = Diagonal(array(namespace, [0.5, 0.25]))
    sigma_s = Diagonal(array(namespace, [1.5, 2.0]))
    ineq = Dense(array(namespace, [[1.0, 2.0], [0.0, 1.0]]))
    op = build_condensed_operator(
        w, sigma_x, sigma_s, ineq, RegularizationState(delta_w=1e-3)
    )
    _assert_values_match(namespace, op, tol)


def test_condensed_lbfgs_without_pairs_coo_values_match(namespace, tol):
    # Before the first curvature pair B = I: the low-rank branch with no border.
    w = LBFGSOperator(2, LBFGSOptions(memory=5))
    sigma_x = Diagonal(array(namespace, [0.5, 0.25]))
    sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, 2), dtype=float_dtype(namespace)))
    op = build_condensed_operator(w, sigma_x, sigma_s, empty_jac, RegularizationState())
    _assert_values_match(namespace, op, tol)


def test_saddle_lbfgs_with_equalities_and_inequalities_coo_values_match(namespace, tol):
    w = _lbfgs(namespace, 3)
    sigma_x = Diagonal(array(namespace, [0.5, 0.25, 0.75]))
    sigma_s = Diagonal(array(namespace, [1.5]))
    ineq = Dense(array(namespace, [[1.0, 0.0, 2.0]]))
    condensed = build_condensed_operator(
        w, sigma_x, sigma_s, ineq, RegularizationState(delta_w=1e-3)
    )
    eq = Dense(array(namespace, [[1.0, 1.0, 0.0], [0.0, 2.0, 1.0]]))
    op = build_saddle_operator(condensed, eq, delta_c=1e-4)
    _assert_values_match(namespace, op, tol)


def test_saddle_assemblable_no_inequalities_coo_values_match(namespace, tol):
    w = Dense(array(namespace, [[3.0, 1.0], [1.0, 2.0]]))
    sigma_x = Diagonal(array(namespace, [0.5, 0.25]))
    sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, 2), dtype=float_dtype(namespace)))
    condensed = build_condensed_operator(
        w, sigma_x, sigma_s, empty_jac, RegularizationState(delta_w=1e-3)
    )
    eq = Dense(array(namespace, [[1.0, 1.0]]))
    op = build_saddle_operator(condensed, eq, delta_c=1e-4)
    _assert_values_match(namespace, op, tol)
