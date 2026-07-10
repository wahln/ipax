"""Unit tests for the CuPy/cuSOLVER Bunch-Kaufman LDLT dense adapter.

Gated on a real CUDA device (``@pytest.mark.gpu``): the numerics here go
through actual nvmath-python ``cusolverDn`` calls (device pointers, handle
lifetime, int32/int64 pivot marshaling), which a NumPy-backed fake module
cannot exercise meaningfully — unlike the CuPy *sparse* adapter's fake-module
tests, faking this would mean re-implementing LAPACK ``sytrf`` in the test
double. Skips cleanly where no CUDA device is available (CI has none).
"""

from __future__ import annotations

import pytest

cupy = pytest.importorskip("cupy")

from ipax.linalg.solver import LinearSolveError  # noqa: E402

pytestmark = pytest.mark.gpu


def _cuda_available() -> bool:
    try:
        return bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


if not _cuda_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

from ipax.backend.dense.cupy import CuPyLDLFactorization  # noqa: E402


def test_factor_solve_matches_reference_on_indefinite_matrix():
    A = cupy.array(
        [[4.0, 1.0, -1.0], [1.0, -2.0, 0.5], [-1.0, 0.5, 3.0]], dtype=cupy.float64
    )
    x_exact = cupy.array([1.0, -2.0, 0.5], dtype=cupy.float64)
    b = A @ x_exact

    solver = CuPyLDLFactorization()
    solver.factor(A)
    x = solver.solve(b)

    cupy.testing.assert_allclose(x, x_exact, rtol=1e-9, atol=1e-9)


def test_factor_solve_handles_a_genuine_2x2_pivot_block():
    A = cupy.array(
        [[0.0, 1.0, 0.5], [1.0, 0.0, -0.5], [0.5, -0.5, 2.0]], dtype=cupy.float64
    )
    x_exact = cupy.array([1.0, -2.0, 0.5], dtype=cupy.float64)
    b = A @ x_exact

    solver = CuPyLDLFactorization()
    solver.factor(A)
    x = solver.solve(b)

    cupy.testing.assert_allclose(x, x_exact, rtol=1e-9, atol=1e-9)


def test_solve_handles_matrix_rhs():
    A = cupy.array(
        [[4.0, 1.0, -1.0], [1.0, -2.0, 0.5], [-1.0, 0.5, 3.0]], dtype=cupy.float64
    )
    X_exact = cupy.array(
        [[1.0, 0.5, -0.5, 2.0], [-2.0, 1.0, 0.0, -1.0], [0.5, -1.5, 2.0, 0.5]],
        dtype=cupy.float64,
    )
    B = A @ X_exact

    solver = CuPyLDLFactorization()
    solver.factor(A)
    X = solver.solve(B)

    cupy.testing.assert_allclose(X, X_exact, rtol=1e-9, atol=1e-9)


def test_inertia_matches_eigenvalue_sign_count_diagonal():
    A = cupy.diag(cupy.array([2.0, -3.0, 0.5, -1.0], dtype=cupy.float64))
    solver = CuPyLDLFactorization()
    solver.factor(A)
    assert solver.inertia_or_none() == (2, 2, 0)


def test_inertia_matches_eigenvalue_sign_count_with_2x2_pivot():
    A = cupy.array(
        [[0.0, 1.0, 0.5], [1.0, 0.0, -0.5], [0.5, -0.5, 2.0]], dtype=cupy.float64
    )
    true_eig = cupy.linalg.eigvalsh(A)
    expected_pos = int((true_eig > 1e-9).sum())
    expected_neg = int((true_eig < -1e-9).sum())

    solver = CuPyLDLFactorization()
    solver.factor(A)
    pos, neg, zero = solver.inertia_or_none()

    assert (pos, neg) == (expected_pos, expected_neg)
    assert zero == 0


def test_inertia_none_before_factor():
    solver = CuPyLDLFactorization()
    assert solver.inertia_or_none() is None


def test_factor_wraps_singular_input_as_linear_solve_error():
    A = cupy.zeros((2, 2), dtype=cupy.float64)
    solver = CuPyLDLFactorization()
    with pytest.raises(LinearSolveError):
        solver.factor(A)


def test_solve_before_factor_raises():
    solver = CuPyLDLFactorization()
    with pytest.raises(RuntimeError, match="factor"):
        solver.solve(cupy.array([1.0, 2.0], dtype=cupy.float64))


def test_factor_reused_across_multiple_solves():
    # The same factorization must serve repeated solve() calls (mirrors the
    # DenseSolver caching contract) without re-factoring.
    A = cupy.array(
        [[4.0, 1.0, -1.0], [1.0, -2.0, 0.5], [-1.0, 0.5, 3.0]], dtype=cupy.float64
    )
    solver = CuPyLDLFactorization()
    solver.factor(A)

    x1_exact = cupy.array([1.0, -2.0, 0.5], dtype=cupy.float64)
    x2_exact = cupy.array([0.5, 1.0, -1.0], dtype=cupy.float64)
    x1 = solver.solve(A @ x1_exact)
    x2 = solver.solve(A @ x2_exact)

    cupy.testing.assert_allclose(x1, x1_exact, rtol=1e-9, atol=1e-9)
    cupy.testing.assert_allclose(x2, x2_exact, rtol=1e-9, atol=1e-9)
