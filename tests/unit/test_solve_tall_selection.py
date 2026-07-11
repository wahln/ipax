"""``solve()`` wires the lazy tall-selection probes into ``select_solver``.

The real tall thresholds engage at ``n >= 10_000`` (``ipax/linalg/solver.py``),
which is benchmark scale, so they are monkeypatched small here and the probe
closures in ``ipax.solve`` (``_ineq_jac_probe`` / ``_ineq_gram_capable`` /
``_ineq_density``) run against a tiny banded tall QP instead. The threshold
*values* themselves are measurement-derived and exercised in
``tests/unit/test_solver_selection.py``; this file covers the wiring.
"""

from __future__ import annotations

from importlib import import_module

import pytest

import ipax.linalg.solver as solver_mod
from ipax import COOOperator, Options, Problem, solve
from ipax.linalg.dense import DenseSolver
from ipax.linalg.krylov import KrylovSolver
from ipax.testing.backends import import_namespace

# ``ipax.solve`` the *module* — the package re-exports the ``solve`` function
# under the same name, shadowing the submodule attribute.
solve_mod = import_module("ipax.solve")

_N = 8
_M = 96  # 12x the variable count: comfortably past _TALL_ROW_EXCESS = 10


class _TinyTallQP(Problem):
    """min ½‖x − 1.5‖² s.t. banded A x ≤ b, x ≥ 0 (rows hit adjacent columns)."""

    def __init__(self, xp, *, jacobian: str) -> None:
        assert jacobian in ("coo", "dense")
        self._xp = xp
        self._jacobian = jacobian
        rows_l, cols_l, vals_l = [], [], []
        for i in range(_M):
            center = (i * _N) // _M
            for k in range(2):
                j = min(_N - 1, center + k)
                rows_l.append(i)
                cols_l.append(j)
                vals_l.append(1.0 + 0.1 * ((i + k) % 5))
        self._rows = xp.asarray(rows_l)
        self._cols = xp.asarray(cols_l)
        self._vals = xp.asarray(vals_l, dtype=xp.float64)
        self._dense = COOOperator(self._rows, self._cols, self._vals, (_M, _N)).matmat(
            xp.eye(_N, dtype=xp.float64)
        )
        x_feas = xp.full((_N,), 0.5, dtype=xp.float64)
        self._b = xp.matmul(self._dense, x_feas) + 0.3
        self._target = xp.full((_N,), 1.5, dtype=xp.float64)

    @property
    def n_vars(self) -> int:
        return _N

    def objective(self, x):
        d = x - self._target
        return 0.5 * self._xp.sum(d * d)

    def gradient(self, x):
        return x - self._target

    def ineq_constraints(self, x):
        return self._xp.matmul(self._dense, x) - self._b

    def ineq_jacobian(self, x):
        del x
        if self._jacobian == "dense":
            return self._dense
        return COOOperator(
            self._rows, self._cols, self._vals, (_M, _N), pattern_key="A"
        )

    def bounds(self):
        return self._xp.zeros((_N,), dtype=self._xp.float64), None


@pytest.fixture
def selected_solvers(monkeypatch):
    """Shrink the tall thresholds and record what ``solve`` selects."""
    monkeypatch.setattr(solver_mod, "_DENSE_AUTO_MAX_VARS", 2)
    monkeypatch.setattr(solver_mod, "_TALL_DENSE_MAX_VARS", 1_000)

    chosen: list[object] = []
    real_select = solve_mod.select_solver

    def spy(**kwargs):
        solver = real_select(**kwargs)
        chosen.append(solver)
        return solver

    monkeypatch.setattr(solve_mod, "select_solver", spy)
    return chosen


def _run(problem) -> None:
    xp = import_namespace("numpy")
    x0 = xp.full((_N,), 0.4)
    solve(problem, x0, options=Options(max_iter=3))


def test_tall_gram_capable_sparse_jacobian_selects_the_dense_route(selected_solvers):
    # Banded COO rows: density 2/8 = 0.25, above _TALL_DENSE_MIN_DENSITY (0.05).
    _run(_TinyTallQP(import_namespace("numpy"), jacobian="coo"))
    assert len(selected_solvers) == 1
    assert isinstance(selected_solvers[0], DenseSolver)


def test_tall_sparse_jacobian_below_min_density_keeps_krylov(
    selected_solvers, monkeypatch
):
    monkeypatch.setattr(solver_mod, "_TALL_DENSE_MIN_DENSITY", 0.5)
    _run(_TinyTallQP(import_namespace("numpy"), jacobian="coo"))
    assert len(selected_solvers) == 1
    assert isinstance(selected_solvers[0], KrylovSolver)


def test_tall_gram_incapable_dense_jacobian_keeps_krylov(selected_solvers):
    # A raw dense-array Jacobian wraps into a Dense operator, which cannot form
    # the Gram without densify — the gram_capable probe must veto the route.
    _run(_TinyTallQP(import_namespace("numpy"), jacobian="dense"))
    assert len(selected_solvers) == 1
    assert isinstance(selected_solvers[0], KrylovSolver)


# --- sparse normal-equations auto-selection wiring ---------------------------


class _TinyTallQPWithHessian(_TinyTallQP):
    """The same QP with an analytic (identity) Lagrangian Hessian."""

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del y_eq, y_ineq
        return sigma * self._xp.eye(_N, dtype=self._xp.float64)


@pytest.fixture
def ne_zone(selected_solvers, monkeypatch):
    """Push the tiny banded QP into the sparse-NE selection zone: its density
    (0.25) is above the real dense-GEMM crossover and its Gram fill (~0.4 at
    n=8) above the real NE threshold, so both measurement-derived cutoffs are
    widened; the *wiring* (probes → select_solver → SparseDirectSolver) is
    what runs for real here."""
    monkeypatch.setattr(solver_mod, "_TALL_DENSE_MIN_DENSITY", 0.5)
    monkeypatch.setattr(solver_mod, "_TALL_SPARSE_NE_MAX_FILL", 0.9)
    return selected_solvers


def test_tall_sparse_gram_selects_normal_equations(ne_zone):
    from ipax.linalg.sparse import SparseDirectSolver

    pytest.importorskip("scipy")
    _run(_TinyTallQP(import_namespace("numpy"), jacobian="coo"))
    assert len(ne_zone) == 1
    assert isinstance(ne_zone[0], SparseDirectSolver)
    assert ne_zone[0].form == "normal_equations"


def test_tall_analytic_hessian_keeps_krylov(ne_zone):
    # The NE form folds the Gram into the condensed block only for an L-BFGS
    # (diagonal + low-rank) Hessian; with an analytic Hessian the auto route
    # must not gamble on the operator being COO-emittable.
    pytest.importorskip("scipy")
    _run(_TinyTallQPWithHessian(import_namespace("numpy"), jacobian="coo"))
    assert len(ne_zone) == 1
    assert isinstance(ne_zone[0], KrylovSolver)
