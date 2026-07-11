"""End-to-end solves through the sparse normal-equations route.

A small banded tall QP (m = 5n, two entries per row around the diagonal) —
the localized-row structure the route exists for. The sparse-NE solve must
reach the same optimum as the dense reference on both the exact (sparse
Hessian) and L-BFGS (low-rank border retained) paths.
"""

from __future__ import annotations

import pytest

from ipax import COOOperator, Options, Problem, solve
from ipax.backend.sparse import get_sparse_adapter
from ipax.options import SparseOptions
from tests._helpers import array

pytestmark = pytest.mark.sparse

_N = 12
_M = 5 * _N


class _BandedTallQP(Problem):
    """min ½‖x − 1.5‖² s.t. banded A x ≤ b, x ≥ 0 (rows hit adjacent columns).

    With ``with_equality=True`` a sparse two-row equality block is added, so
    the NE route factors the Schur/equality-bordered saddle.
    """

    def __init__(self, namespace, *, with_hessian: bool, with_equality=False) -> None:
        self._xp = namespace
        self._with_hessian = with_hessian
        self._with_equality = with_equality
        rows_l, cols_l, vals_l = [], [], []
        for i in range(_M):
            center = (i * _N) // _M
            for k in range(2):
                j = min(_N - 1, center + k)
                rows_l.append(i)
                cols_l.append(j)
                vals_l.append(1.0 + 0.1 * ((i + k) % 5))
        self._rows = namespace.asarray(rows_l)
        self._cols = namespace.asarray(cols_l)
        self._vals = array(namespace, vals_l)
        dense = COOOperator(self._rows, self._cols, self._vals, (_M, _N)).matmat(
            namespace.eye(_N, dtype=self._vals.dtype)
        )
        x_feas = namespace.full((_N,), 0.5, dtype=self._vals.dtype)
        margins = array(namespace, [0.05 + 0.4 * ((i % 7) / 7.0) for i in range(_M)])
        row_scale = namespace.matmul(dense, namespace.full((_N,), 0.5))
        self._b = namespace.matmul(dense, x_feas) + margins * row_scale
        self._target = namespace.full((_N,), 1.5, dtype=self._vals.dtype)

    @property
    def n_vars(self) -> int:
        return _N

    def objective(self, x):
        d = x - self._target
        return 0.5 * self._xp.sum(d * d)

    def gradient(self, x):
        return x - self._target

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        if not self._with_hessian:
            raise NotImplementedError
        xp = self._xp
        idx = xp.arange(_N)
        return COOOperator(
            idx,
            idx,
            xp.full((_N,), float(sigma), dtype=self._vals.dtype),
            (_N, _N),
            symmetric=True,
            pattern_key="H",
        )

    def ineq_constraints(self, x):
        op = COOOperator(self._rows, self._cols, self._vals, (_M, _N))
        return op.matvec(x) - self._b

    def ineq_jacobian(self, x):
        del x
        return COOOperator(
            self._rows, self._cols, self._vals, (_M, _N), pattern_key="A"
        )

    def bounds(self):
        return self._xp.zeros((_N,), dtype=self._vals.dtype), None

    def _eq_operator(self):
        xp = self._xp
        rows = xp.asarray([0, 0, 1, 1])
        cols = xp.asarray([0, 2, 1, _N - 1])
        vals = array(xp, [1.0, 1.0, 1.0, -1.0])
        return COOOperator(rows, cols, vals, (2, _N), pattern_key="C")

    def eq_constraints(self, x):
        if not self._with_equality:
            raise NotImplementedError
        # x0 + x2 = 0.9 and x1 = x_{N-1}: satisfied strictly inside the
        # inequality margins at the feasible seed x = 0.5.
        rhs = array(self._xp, [0.9, 0.0])
        return self._eq_operator().matvec(x) - rhs

    def eq_jacobian(self, x):
        if not self._with_equality:
            raise NotImplementedError
        del x
        return self._eq_operator()


def _require_sparse(namespace):
    if get_sparse_adapter(namespace) is None:
        pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")


@pytest.mark.parametrize("hessian", ["exact", "lbfgs"])
def test_sparse_normal_equations_reaches_the_dense_optimum(namespace, hessian):
    _require_sparse(namespace)
    problem = _BandedTallQP(namespace, with_hessian=(hessian == "exact"))
    x0 = namespace.full((_N,), 0.4)

    reference = solve(problem, x0, options=Options(linsolve="dense", hessian=hessian))
    assert reference.status.is_success

    result = solve(
        problem,
        x0,
        options=Options(
            linsolve="sparse",
            sparse=SparseOptions(kkt_route="normal_equations"),
            hessian=hessian,
        ),
    )
    assert result.status.is_success
    assert result.kkt_error <= 1e-6
    assert abs(float(result.objective) - float(reference.objective)) <= 1e-6


@pytest.mark.parametrize("hessian", ["exact", "lbfgs"])
def test_sparse_normal_equations_solves_equality_constrained_problems(
    namespace, hessian
):
    # The Schur/equality border: ∇c stays explicit next to the sparsely
    # condensed n×n block, so equality-constrained problems now go through
    # the NE route instead of being rejected.
    _require_sparse(namespace)
    problem = _BandedTallQP(
        namespace, with_hessian=(hessian == "exact"), with_equality=True
    )
    x0 = namespace.full((_N,), 0.4)

    reference = solve(problem, x0, options=Options(linsolve="dense", hessian=hessian))
    assert reference.status.is_success

    result = solve(
        problem,
        x0,
        options=Options(
            linsolve="sparse",
            sparse=SparseOptions(kkt_route="normal_equations"),
            hessian=hessian,
        ),
    )
    assert result.status.is_success
    assert result.kkt_error <= 1e-6
    assert abs(float(result.objective) - float(reference.objective)) <= 1e-6


def test_normal_equations_route_solves_a_pure_equality_problem(namespace):
    # No inequalities at all: the NE form degenerates to the equality saddle.
    _require_sparse(namespace)

    class _EqQP(Problem):
        @property
        def n_vars(self) -> int:
            return 2

        def objective(self, x):
            return namespace.sum(x * x)

        def gradient(self, x):
            return 2.0 * x

        def eq_constraints(self, x):
            return namespace.stack((x[0] + x[1] - 1.0,))

        def eq_jacobian(self, x):
            return array(namespace, [[1.0, 1.0]])

        def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
            idx = namespace.arange(2)
            return COOOperator(
                idx,
                idx,
                namespace.full((2,), 2.0 * float(sigma)),
                (2, 2),
                symmetric=True,
                pattern_key="H",
            )

    result = solve(
        _EqQP(),
        array(namespace, [0.3, 0.4]),
        options=Options(
            linsolve="sparse",
            sparse=SparseOptions(kkt_route="normal_equations"),
            hessian="exact",
        ),
    )
    assert result.status.is_success
    # Analytic optimum of min ‖x‖² s.t. x0 + x1 = 1 is x = (0.5, 0.5).
    assert abs(float(result.x[0]) - 0.5) <= 1e-6
    assert abs(float(result.x[1]) - 0.5) <= 1e-6
