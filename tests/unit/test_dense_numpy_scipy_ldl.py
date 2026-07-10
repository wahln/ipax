"""Unit tests for the SciPy/LAPACK Bunch-Kaufman LDLT dense adapter."""

from __future__ import annotations

import numpy as np
import pytest

from ipax.linalg.solver import LinearSolveError

pytest.importorskip("scipy")  # the NumPy adapter wraps scipy.linalg.ldl

from ipax.backend.dense.numpy_scipy import ScipyLDLFactorization


def test_factor_solve_matches_reference_on_indefinite_matrix():
    # Symmetric indefinite (mixed-sign diagonal), no special structure.
    A = np.array(
        [[4.0, 1.0, -1.0], [1.0, -2.0, 0.5], [-1.0, 0.5, 3.0]], dtype=np.float64
    )
    rng = np.random.default_rng(0)
    x_exact = rng.normal(size=3)
    b = A @ x_exact

    solver = ScipyLDLFactorization()
    solver.factor(A)
    x = solver.solve(b)

    np.testing.assert_allclose(x, x_exact, rtol=1e-9, atol=1e-9)


def test_factor_solve_handles_a_genuine_2x2_pivot_block():
    # [[0,1],[1,0]] forces Bunch-Kaufman to choose a 2x2 pivot.
    A = np.array(
        [[0.0, 1.0, 0.5], [1.0, 0.0, -0.5], [0.5, -0.5, 2.0]], dtype=np.float64
    )
    x_exact = np.array([1.0, -2.0, 0.5])
    b = A @ x_exact

    solver = ScipyLDLFactorization()
    solver.factor(A)
    x = solver.solve(b)

    np.testing.assert_allclose(x, x_exact, rtol=1e-9, atol=1e-9)


def test_solve_handles_matrix_rhs():
    A = np.array(
        [[4.0, 1.0, -1.0], [1.0, -2.0, 0.5], [-1.0, 0.5, 3.0]], dtype=np.float64
    )
    rng = np.random.default_rng(1)
    X_exact = rng.normal(size=(3, 4))
    B = A @ X_exact

    solver = ScipyLDLFactorization()
    solver.factor(A)
    X = solver.solve(B)

    np.testing.assert_allclose(X, X_exact, rtol=1e-9, atol=1e-9)


def test_inertia_matches_eigenvalue_sign_count_diagonal():
    A = np.diag([2.0, -3.0, 0.5, -1.0])
    solver = ScipyLDLFactorization()
    solver.factor(A)
    assert solver.inertia_or_none() == (2, 2, 0)


def test_inertia_matches_eigenvalue_sign_count_with_2x2_pivot():
    # From the [[0,1],[1,0]] + bordering example: eigenvalues are
    # (-1.158, 1.0, 2.158) -> 2 positive, 1 negative (verified against
    # np.linalg.eigvalsh independently).
    A = np.array(
        [[0.0, 1.0, 0.5], [1.0, 0.0, -0.5], [0.5, -0.5, 2.0]], dtype=np.float64
    )
    true_eig = np.linalg.eigvalsh(A)
    expected_pos = int(np.sum(true_eig > 1e-9))
    expected_neg = int(np.sum(true_eig < -1e-9))

    solver = ScipyLDLFactorization()
    solver.factor(A)
    pos, neg, zero = solver.inertia_or_none()

    assert (pos, neg) == (expected_pos, expected_neg)
    assert zero == 0


def test_inertia_none_before_factor():
    solver = ScipyLDLFactorization()
    assert solver.inertia_or_none() is None


def test_factor_wraps_non_finite_input_as_linear_solve_error():
    A = np.array([[1.0, np.nan], [np.nan, 1.0]])
    solver = ScipyLDLFactorization()
    with pytest.raises(LinearSolveError):
        solver.factor(A)


def test_solve_before_factor_raises():
    solver = ScipyLDLFactorization()
    with pytest.raises(RuntimeError, match="factor"):
        solver.solve(np.array([1.0, 2.0]))
