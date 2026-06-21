"""Dense reference route on a nonconvex (indefinite-Hessian) problem.

``xp.linalg.solve`` is an LU factorization, so it accepts an indefinite condensed
``N`` and returns a non-descent step. The dense route guards against that with a
Cholesky PD-probe on ``N`` that drives δ_w escalation — without it the solve
stalls at a saddle point. This is the dense analogue of the sparse route's
inertia check (``tests/integration/test_sparse_solve.py``).
"""

from __future__ import annotations

from ipax import FunctionProblem, Options
from ipax.result import Status
from ipax.solve import solve
from tests._helpers import array, float_dtype


def _saddle_trap_qp(namespace):
    """``f(x) = x0 x1`` on ``[-1, 1]^2``: minimizers at (1,-1)/(-1,1) (f = -1),
    saddle KKT point at the origin (f = 0). The Hessian ``[[0, 1], [1, 0]]`` is
    indefinite (eigenvalues ±1)."""
    hess = array(namespace, [[0.0, 1.0], [1.0, 0.0]])

    def objective(x):
        return x[0] * x[1]

    def gradient(x):
        return namespace.stack((x[1], x[0]))

    def lagrangian_hessian(x, y_eq, y_ineq, sigma=1.0):
        del x, y_eq, y_ineq
        return sigma * hess

    problem = FunctionProblem(
        2,
        objective,
        gradient=gradient,
        bounds=(
            -namespace.ones((2,), dtype=float_dtype(namespace)),
            namespace.ones((2,), dtype=float_dtype(namespace)),
        ),
        lagrangian_hessian=lagrangian_hessian,
    )
    return problem, array(namespace, [0.3, 0.31])


def test_dense_route_pd_guard_escapes_saddle(namespace):
    problem, x0 = _saddle_trap_qp(namespace)
    result = solve(
        problem, x0, options=Options(linsolve="dense", hessian="exact", max_iter=300)
    )

    assert result.status is Status.OPTIMAL
    # Reaches a true minimizer (f = -1, a box vertex), not the saddle at f = 0.
    assert abs(float(result.objective) - (-1.0)) <= 1e-5
    assert abs(float(result.x[0] * result.x[1]) - (-1.0)) <= 1e-5
