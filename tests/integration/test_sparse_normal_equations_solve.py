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
    """min ½‖x − 1.5‖² s.t. banded A x ≤ b, x ≥ 0 (rows hit adjacent columns)."""

    def __init__(self, namespace, *, with_hessian: bool) -> None:
        self._xp = namespace
        self._with_hessian = with_hessian
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


def test_normal_equations_route_rejects_equality_problems(namespace):
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

    with pytest.raises(RuntimeError, match=r"normal.equations"):
        solve(
            _EqQP(),
            array(namespace, [0.3, 0.4]),
            options=Options(
                linsolve="sparse",
                sparse=SparseOptions(kkt_route="normal_equations"),
                hessian="exact",
            ),
        )
