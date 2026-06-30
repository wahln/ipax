"""A ``Problem`` supplying its Jacobian/Hessian as public sparse operators.

This is the end-to-end check that ``COOOperator`` is first-class on the
sparse-direct route: the driver pulls the constraint Jacobian's ``rmatvec`` into
the Lagrangian gradient, the condensed/saddle assembly its ``to_coo`` /
``coo_values`` / Gram diagonal, and the adapter factorizes the result. The COO
operator solve must reach the same optimum as the equivalent dense-array model.
"""

from __future__ import annotations

import pytest

from ipax import COOOperator, CSCOperator, Options, Problem
from ipax.backend.operators import Dense
from ipax.backend.sparse import get_sparse_adapter
from ipax.result import Status
from ipax.solve import solve
from tests._helpers import array, assert_allclose, norm_inf

pytestmark = pytest.mark.sparse


class _EqualityQP(Problem):
    """min ½xᵀQx + bᵀx s.t. Cx = d, with structure supplied as operators.

    ``Q = [[4,1,0],[1,3,1],[0,1,2]]`` (SPD), ``b = [1,2,3]``,
    ``C = [[1,1,1]]``, ``d = [1]``. With ``use_sparse_ops`` the Jacobian and
    Hessian are returned as :class:`COOOperator`; otherwise as dense arrays.
    """

    def __init__(self, namespace, *, use_sparse_ops: bool) -> None:
        self._xp = namespace
        self._sparse = use_sparse_ops
        self._Q = array(namespace, [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
        self._b = array(namespace, [1.0, 2.0, 3.0])
        self._C = array(namespace, [[1.0, 1.0, 1.0]])
        self._d = array(namespace, [1.0])
        # COO triplets (Q symmetric, full; C dense row).
        self._q_rows = namespace.asarray([0, 0, 1, 1, 1, 2, 2])
        self._q_cols = namespace.asarray([0, 1, 0, 1, 2, 1, 2])
        self._q_vals = array(namespace, [4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0])
        self._c_rows = namespace.asarray([0, 0, 0])
        self._c_cols = namespace.asarray([0, 1, 2])
        self._c_vals = array(namespace, [1.0, 1.0, 1.0])

    @property
    def n_vars(self) -> int:
        return 3

    def objective(self, x):
        return 0.5 * self._xp.sum(x * self._xp.matmul(self._Q, x)) + self._xp.sum(
            self._b * x
        )

    def gradient(self, x):
        return self._xp.matmul(self._Q, x) + self._b

    def eq_constraints(self, x):
        return self._xp.matmul(self._C, x) - self._d

    def eq_jacobian(self, x):
        del x
        if self._sparse:
            return COOOperator(
                self._c_rows, self._c_cols, self._c_vals, (1, 3), pattern_key="C"
            )
        return Dense(self._C)

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del x, y_eq, y_ineq
        if self._sparse:
            return COOOperator(
                self._q_rows,
                self._q_cols,
                sigma * self._q_vals,
                (3, 3),
                symmetric=True,
                pattern_key="Q",
            )
        return Dense(sigma * self._Q)


def _require_sparse(namespace):
    if get_sparse_adapter(namespace) is None:
        pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")


def test_coo_operator_problem_reaches_optimum_via_sparse_route(namespace):
    _require_sparse(namespace)
    x0 = array(namespace, [0.2, 0.1, 0.3])

    sparse_ops = solve(
        _EqualityQP(namespace, use_sparse_ops=True),
        x0,
        options=Options(linsolve="sparse", hessian="exact"),
    )
    dense_ref = solve(
        _EqualityQP(namespace, use_sparse_ops=False),
        x0,
        options=Options(linsolve="dense", hessian="exact"),
    )

    assert sparse_ops.status is Status.OPTIMAL
    assert dense_ref.status is Status.OPTIMAL
    diff = norm_inf(namespace, sparse_ops.x - dense_ref.x)
    assert diff <= 1e-6, f"|Δx|∞={diff}"
    # The single equality Cx = d = 1 must hold at the solution.
    assert_allclose(
        namespace,
        namespace.matmul(array(namespace, [[1.0, 1.0, 1.0]]), sparse_ops.x),
        array(namespace, [1.0]),
        rtol=1e-6,
        atol=1e-6,
    )


class _BoundQP(Problem):
    """min ½xᵀQx + bᵀx s.t. x ≥ 1, with Q as a ``CSCOperator``.

    Bound-only and exact-sparse: exercises a CSC-supplied Hessian end-to-end
    through the sparse-direct route against a dense-array reference.
    """

    def __init__(self, namespace, *, use_sparse_ops: bool) -> None:
        self._xp = namespace
        self._sparse = use_sparse_ops
        # Q = [[4,1,0],[1,3,1],[0,1,2]] (SPD, tridiagonal, full diagonal), b = -1.
        self._Q = array(namespace, [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
        self._b = array(namespace, [-1.0, -1.0, -1.0])
        # Canonical CSC of Q (column-major; full diagonal present).
        self._indptr = namespace.asarray([0, 2, 5, 7])
        self._indices = namespace.asarray([0, 1, 0, 1, 2, 1, 2])
        self._data = array(namespace, [4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0])

    @property
    def n_vars(self) -> int:
        return 3

    def bounds(self):
        return (array(self._xp, [1.0, 1.0, 1.0]), None)

    def objective(self, x):
        return 0.5 * self._xp.sum(x * self._xp.matmul(self._Q, x)) + self._xp.sum(
            self._b * x
        )

    def gradient(self, x):
        return self._xp.matmul(self._Q, x) + self._b

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del x, y_eq, y_ineq
        if self._sparse:
            return CSCOperator(
                self._indptr,
                self._indices,
                sigma * self._data,
                (3, 3),
                symmetric=True,
                pattern_key="Q",
            )
        return Dense(sigma * self._Q)


def test_bound_only_csc_operator_matches_dense(namespace):
    _require_sparse(namespace)
    x0 = array(namespace, [2.0, 2.0, 2.0])

    sparse_ops = solve(
        _BoundQP(namespace, use_sparse_ops=True),
        x0,
        options=Options(linsolve="sparse", hessian="exact"),
    )
    dense = solve(
        _BoundQP(namespace, use_sparse_ops=False),
        x0,
        options=Options(linsolve="dense", hessian="exact"),
    )

    assert sparse_ops.status is Status.OPTIMAL
    assert dense.status is Status.OPTIMAL
    assert norm_inf(namespace, sparse_ops.x - dense.x) <= 1e-6
