"""An overflowing bound diagonal must not yield an accepted finite direction."""

from __future__ import annotations

import pytest

from ipax.backend.operators import Dense, Diagonal
from ipax.ipm.hessian import LBFGSOperator
from ipax.ipm.kkt import _add_dense_diagonal, build_condensed_operator
from ipax.linalg.dense import DenseSolver
from ipax.linalg.regularize import RegularizationState
from ipax.linalg.solver import LinearSolveError
from ipax.options import LBFGSOptions
from tests._helpers import array


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("scalar", [False, True])
def test_nonfinite_dense_diagonal_marks_invalid_pivots(namespace, scalar):
    xp = namespace
    matrix = array(xp, [[2.0, 0.5], [0.5, 2.0]])
    diagonal = float("inf") if scalar else array(xp, [float("inf"), 1.0])

    result = _add_dense_diagonal(matrix, diagonal)

    # NaN pivots cannot be back-substituted into finite zero steps, unlike
    # +inf pivots. Finite source entries must remain untouched.
    assert bool(xp.isnan(result[0, 0]))
    assert bool(xp.all(xp.isfinite(matrix)))
    if not scalar:
        assert float(result[1, 1]) == 3.0
        assert float(diagonal[1]) == 1.0
        assert bool(xp.isinf(diagonal[0]))


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("hessian", ["exact", "lbfgs"])
def test_overflowing_bound_diagonal_cannot_return_a_finite_step(namespace, hessian):
    xp = namespace
    # Finite bound data, but Sigma_L overflows. A bare +inf pivot can yield
    # dx[0] = 0, passing the driver's finite-solution guard before the bound
    # recovery computes -Sigma_L * dx = NaN (Breedveld 2017, eqs. 14-17).
    z_lower = array(xp, [1e308, 1.0])
    gap = array(xp, [0.1, 1.0])
    sigma_l = z_lower / gap
    assert bool(xp.all(xp.isfinite(z_lower)))
    assert bool(xp.all(xp.isfinite(gap)))
    assert not bool(xp.all(xp.isfinite(sigma_l)))

    W = (
        Dense(array(xp, [[2.0, 0.5], [0.5, 2.0]]))
        if hessian == "exact"
        else LBFGSOperator(2, LBFGSOptions())
    )
    # Inequality rows force the L-BFGS case through dense materialization,
    # including its Cholesky-to-LU fallback, rather than a structured solve.
    op = build_condensed_operator(
        W,
        Diagonal(sigma_l),
        Diagonal(array(xp, [1.0])),
        Dense(array(xp, [[1.0, 1.0]])),
        RegularizationState(),
    )
    solver = DenseSolver()
    try:
        solver.factor(op)
        dx = solver.solve(array(xp, [1.0, 1.0]))
    except LinearSolveError:
        return
    # Both outcomes are part of the LinearSolver numerical-failure contract;
    # the driver rejects a non-finite solution before recovering bound duals.
    assert not bool(xp.all(xp.isfinite(dx)))
