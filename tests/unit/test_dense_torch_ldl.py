"""Unit tests for the Torch Bunch-Kaufman LDLT dense adapter."""

from __future__ import annotations

import pytest

from ipax.linalg.solver import LinearSolveError
from ipax.testing.backends import import_namespace

torch = pytest.importorskip("torch")
import_namespace("torch")  # skip cleanly if the torch namespace can't be imported

from ipax.backend.dense.torch import TorchLDLFactorization  # noqa: E402


def test_factor_solve_matches_reference_on_indefinite_matrix():
    A = torch.tensor(
        [[4.0, 1.0, -1.0], [1.0, -2.0, 0.5], [-1.0, 0.5, 3.0]], dtype=torch.float64
    )
    x_exact = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    b = A @ x_exact

    solver = TorchLDLFactorization()
    solver.factor(A)
    x = solver.solve(b)

    torch.testing.assert_close(x, x_exact, rtol=1e-9, atol=1e-9)


def test_factor_solve_handles_a_genuine_2x2_pivot_block():
    A = torch.tensor(
        [[0.0, 1.0, 0.5], [1.0, 0.0, -0.5], [0.5, -0.5, 2.0]], dtype=torch.float64
    )
    x_exact = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    b = A @ x_exact

    solver = TorchLDLFactorization()
    solver.factor(A)
    x = solver.solve(b)

    torch.testing.assert_close(x, x_exact, rtol=1e-9, atol=1e-9)


def test_solve_handles_matrix_rhs():
    A = torch.tensor(
        [[4.0, 1.0, -1.0], [1.0, -2.0, 0.5], [-1.0, 0.5, 3.0]], dtype=torch.float64
    )
    X_exact = torch.tensor(
        [[1.0, 0.5, -0.5, 2.0], [-2.0, 1.0, 0.0, -1.0], [0.5, -1.5, 2.0, 0.5]],
        dtype=torch.float64,
    )
    B = A @ X_exact

    solver = TorchLDLFactorization()
    solver.factor(A)
    X = solver.solve(B)

    torch.testing.assert_close(X, X_exact, rtol=1e-9, atol=1e-9)


def test_inertia_matches_eigenvalue_sign_count_diagonal():
    A = torch.diag(torch.tensor([2.0, -3.0, 0.5, -1.0], dtype=torch.float64))
    solver = TorchLDLFactorization()
    solver.factor(A)
    assert solver.inertia_or_none() == (2, 2, 0)


def test_inertia_matches_eigenvalue_sign_count_with_2x2_pivot():
    A = torch.tensor(
        [[0.0, 1.0, 0.5], [1.0, 0.0, -0.5], [0.5, -0.5, 2.0]], dtype=torch.float64
    )
    true_eig = torch.linalg.eigvalsh(A)
    expected_pos = int((true_eig > 1e-9).sum())
    expected_neg = int((true_eig < -1e-9).sum())

    solver = TorchLDLFactorization()
    solver.factor(A)
    pos, neg, zero = solver.inertia_or_none()

    assert (pos, neg) == (expected_pos, expected_neg)
    assert zero == 0


def test_inertia_none_before_factor():
    solver = TorchLDLFactorization()
    assert solver.inertia_or_none() is None


def test_factor_wraps_singular_input_as_linear_solve_error():
    # info != 0 in ldl_factor_ex signals a zero pivot -> classified failure.
    A = torch.zeros((2, 2), dtype=torch.float64)
    solver = TorchLDLFactorization()
    with pytest.raises(LinearSolveError):
        solver.factor(A)


def test_solve_before_factor_raises():
    solver = TorchLDLFactorization()
    with pytest.raises(RuntimeError, match="factor"):
        solver.solve(torch.tensor([1.0, 2.0], dtype=torch.float64))
