"""Regression: a singular KKT system must not crash the GMRES fallback.

QC sweep — HS9 (``min sin(πx₁/12)·cos(πx₂/16)`` s.t. ``4x₁ − 3x₂ = 0``) from the
standard start ``(0, 0)`` with ``hessian="exact", linsolve="krylov"``. At that
point the exact Lagrangian Hessian is *identically zero* (every entry carries a
``sin(πxᵢ/·)`` factor), so with no bounds or inequalities the condensed (1,1)
block is the zero operator and the first saddle system is singular. MINRES hits
a Lanczos breakdown, and the new MINRES→GMRES fallback then divided by an exactly
zero rotated-Hessenberg diagonal in the back-substitution — a raw
``ZeroDivisionError`` escaping the solver instead of the controlled
``KrylovConvergenceError`` that lets the driver escalate δ_w and recover.
"""

from __future__ import annotations

import math

import pytest

from ipax import Options, solve
from ipax.testing.problems import HS9
from tests._helpers import array


def test_hs9_exact_krylov_recovers_from_zero_hessian_start(namespace):
    result = solve(
        HS9(namespace),
        array(namespace, [0.0, 0.0]),
        options=Options(hessian="exact", linsolve="krylov"),
    )
    assert result.success, f"got {result.status}"
    # Optimum is non-unique (periodic); assert f* rather than a specific x*.
    assert math.isclose(float(result.objective), -0.5, abs_tol=1e-6)


def test_gmres_zero_operator_raises_convergence_error(namespace):
    """Direct kernel check: GMRES on ``K = 0`` must fail *controlledly*."""
    from ipax.backend.operators import Dense
    from ipax.linalg.krylov import KrylovConvergenceError, KrylovSolver
    from ipax.options import KrylovOptions

    xp = namespace
    K = Dense(xp.zeros((3, 3), dtype=xp.float64))
    solver = KrylovSolver(KrylovOptions(method="gmres"))
    solver.factor(K)
    with pytest.raises(KrylovConvergenceError):
        solver.solve(array(xp, [1.0, 2.0, 3.0]))
