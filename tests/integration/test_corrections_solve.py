"""Integration: Mehrotra/Gondzio corrections converge correctly."""

from __future__ import annotations

import pytest

from ipax import Options, Status, solve
from ipax.backend.namespace import capabilities
from ipax.problem.base import Problem
from ipax.testing.problems import BoundConstrainedQP
from tests._helpers import array, assert_allclose, implemented


class _InequalityQP(Problem):
    """``min 0.5‖x‖² − bᵀx`` s.t. ``Σx ≤ 1``, ``x ≥ 0`` — convex, unique optimum."""

    def __init__(self, xp, n: int = 5) -> None:
        self._xp = xp
        self._n = n
        self._b = array(xp, [1.0 + i / (n - 1) for i in range(n)])

    @property
    def n_vars(self) -> int:
        return self._n

    def bounds(self):
        return (self._xp.zeros((self._n,), dtype=self._b.dtype), None)

    def objective(self, x):
        return 0.5 * self._xp.sum(x * x) - self._xp.sum(self._b * x)

    def gradient(self, x):
        return x - self._b

    def ineq_constraints(self, x):
        return self._xp.reshape(self._xp.sum(x) - 1.0, (1,))

    def ineq_jacobian(self, x):
        return self._xp.ones((1, self._n), dtype=x.dtype)

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        return sigma * self._xp.eye(self._n, dtype=x.dtype)


def _opts(method: str, linsolve: str = "dense") -> Options:
    return Options(hessian="exact", linsolve=linsolve, corrections=method)


@pytest.mark.parametrize("method", ["mehrotra", "gondzio"])
def test_corrections_match_optimum_on_bound_qp(namespace, method):
    problem = BoundConstrainedQP(namespace)
    x0 = array(namespace, [0.25, 0.75])

    with implemented("dense solver"):
        base = solve(problem, x0, options=_opts("none"))
        corrected = solve(problem, x0, options=_opts(method))

    assert corrected.status is Status.OPTIMAL
    assert corrected.kkt_error <= 1e-6
    assert_allclose(namespace, corrected.x, base.x, rtol=1e-6, atol=1e-6)
    # Corrections must not cost extra outer iterations on a convex QP.
    assert corrected.n_iter <= base.n_iter


@pytest.mark.parametrize("method", ["mehrotra", "gondzio"])
def test_corrections_solve_inequality_qp(namespace, method):
    problem = _InequalityQP(namespace)
    x0 = namespace.full((5,), 0.1, dtype=array(namespace, [0.0]).dtype)

    with implemented("dense solver"):
        base = solve(problem, x0, options=_opts("none"))
        corrected = solve(problem, x0, options=_opts(method))

    assert corrected.status is Status.OPTIMAL
    assert corrected.kkt_error <= 1e-6
    assert_allclose(namespace, corrected.x, base.x, rtol=1e-6, atol=1e-6)
    assert corrected.n_iter <= base.n_iter
    # The slack inequality is active at the optimum (Σx* = 1).
    assert float(namespace.sum(corrected.x)) == pytest.approx(1.0, abs=1e-6)


def test_corrections_reduce_outer_iterations_on_inequality_qp(namespace):
    # This checks outer convergence, not wall time: dense corrector solves are
    # full solves, while sparse-direct can reuse its numeric factorization.
    problem = _InequalityQP(namespace, n=8)
    x0 = namespace.full((8,), 0.05, dtype=array(namespace, [0.0]).dtype)

    with implemented("dense solver"):
        base = solve(problem, x0, options=_opts("none"))
        mehrotra = solve(problem, x0, options=_opts("mehrotra"))

    assert mehrotra.status is Status.OPTIMAL
    assert mehrotra.n_iter < base.n_iter


@pytest.mark.parametrize("method", ["mehrotra", "gondzio"])
@pytest.mark.parametrize("linsolve", ["krylov", "sparse"])
def test_corrections_work_across_repeated_solve_routes(namespace, method, linsolve):
    if linsolve == "sparse" and not capabilities(namespace).has_sparse_adapter:
        pytest.skip(f"no sparse adapter for {namespace.__name__!r}")
    problem = _InequalityQP(namespace)
    x0 = namespace.full((5,), 0.1, dtype=array(namespace, [0.0]).dtype)

    corrected = solve(problem, x0, options=_opts(method, linsolve))

    assert corrected.status is Status.OPTIMAL
    assert float(namespace.sum(corrected.x)) == pytest.approx(1.0, abs=1e-5)
