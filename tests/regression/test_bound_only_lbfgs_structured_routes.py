"""Regression: bound-only L-BFGS problems must stay on the structured routes.

2026-08 RT-scale CPU study (n = 50k, ``f(Dx)`` piecewise least squares, box
bounds, default L-BFGS Hessian):

* ``linsolve="dense"`` materialized the n×n condensed block at iteration 0 —
  before the first curvature pair ``LBFGSOperator.compact_form`` raised, the
  structured solve propagated ``NotImplementedError``, and the dense solver
  fell through to a 20 GB matrix and an LU per solve (488 s/iteration).
* ``linsolve="krylov"`` with the default Jacobi preconditioner burned ~30 CG
  iterations per solve plus an O(n·k²) L-BFGS diagonal, although the condensed
  Woodbury inverse is the *exact* ``N⁻¹`` for a bound-only system (one CG
  iteration; 34 → 6 ms per step at n = 50k).
"""

from __future__ import annotations

from ipax import Options, Problem, solve
from ipax.linalg.dense import DenseSolver
from ipax.linalg.krylov import KrylovSolver
from tests._helpers import array, assert_allclose


class _BoxLeastSquares(Problem):
    """``min 0.5 * ||A x - b||^2`` over ``0 <= x <= 1`` with active bounds."""

    def __init__(self, xp, n: int = 12) -> None:
        self.xp = xp
        rows = [
            [1.0 / (1.0 + abs(i - j)) if abs(i - j) <= 2 else 0.0 for j in range(n)]
            for i in range(n + 4)
        ]
        self._A = array(xp, rows)
        # Pull target outside the box on both sides so bounds become active.
        self._b = xp.matmul(
            self._A, array(xp, [1.5 if k % 3 == 0 else -0.5 for k in range(n)])
        )
        self._n = n

    @property
    def n_vars(self) -> int:
        return self._n

    def bounds(self):
        xp = self.xp
        dtype = self._b.dtype
        return xp.zeros((self._n,), dtype=dtype), xp.ones((self._n,), dtype=dtype)

    def objective(self, x):
        r = self.xp.matmul(self._A, x) - self._b
        return 0.5 * self.xp.sum(r * r)

    def gradient(self, x):
        r = self.xp.matmul(self._A, x) - self._b
        return self.xp.matmul(self.xp.permute_dims(self._A, (1, 0)), r)


def _x0(namespace, n):
    return array(namespace, [0.5] * n)


def test_dense_route_never_materializes_bound_only_lbfgs(namespace, monkeypatch):
    def _boom(self, *args, **kwargs):
        raise AssertionError("bound-only L-BFGS must use the structured solve")

    monkeypatch.setattr(DenseSolver, "_materialize_and_guard", _boom)
    problem = _BoxLeastSquares(namespace)
    result = solve(
        problem,
        _x0(namespace, problem.n_vars),
        options=Options(linsolve="dense", hessian="lbfgs"),
    )
    assert result.success, f"got {result.status}: {result.message}"
    assert result.linear_solver == "dense"


def test_krylov_route_uses_exact_lbfgs_inverse_on_bound_only(namespace, monkeypatch):
    iterations: list[int] = []
    original = KrylovSolver.solve

    def _spy(self, rhs):
        out = original(self, rhs)
        iterations.append(self.last_iterations)
        return out

    monkeypatch.setattr(KrylovSolver, "solve", _spy)
    problem = _BoxLeastSquares(namespace)
    result = solve(
        problem,
        _x0(namespace, problem.n_vars),
        options=Options(linsolve="krylov", hessian="lbfgs"),
    )
    assert result.success, f"got {result.status}: {result.message}"
    assert "pc=lbfgs-exact" in result.linear_solver
    # Iteration 0 has no pair (Jacobi on a diagonal N: 1 iteration anyway);
    # every later solve applies the exact Woodbury inverse.
    assert iterations and max(iterations) <= 3, iterations


def test_structured_routes_agree_with_each_other(namespace, tol):
    problem = _BoxLeastSquares(namespace)
    x0 = _x0(namespace, problem.n_vars)
    dense = solve(problem, x0, options=Options(linsolve="dense", hessian="lbfgs"))
    krylov = solve(problem, x0, options=Options(linsolve="krylov", hessian="lbfgs"))
    assert dense.success and krylov.success
    assert_allclose(namespace, dense.x, krylov.x, rtol=1e-5, atol=1e-6)
