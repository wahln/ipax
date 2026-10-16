"""Unit tests for the NumPy/SciPy dense Cholesky factorization gap-filler.

``ipax.backend.dense.numpy_scipy.cholesky`` factors through LAPACK ``?potrf``
directly instead of ``numpy.linalg.cholesky``. NumPy's gufunc linearizes its
C-ordered input into a Fortran-ordered scratch buffer, factors, zeroes the
upper triangle and delinearizes back into a fresh C-ordered output: two full
transposing n×n copies around the factorization, which at n≈10⁴ cost as much
as the factorization itself. The gap-filler pays a single cache-blocked copy
of the lower triangle into a Fortran-ordered buffer that ``potrf`` then
factors in place. These tests pin its contract:

* same factor as ``numpy.linalg.cholesky`` to round-off (bit for bit on the
  NumPy 2.4.6 / SciPy 1.17.1 wheels, OpenBLAS 0.3.31 / 0.3.30; a build fact,
  not asserted);
* reads only the lower triangle, like NumPy;
* never writes to its input (``DenseSolver`` keeps the matrix for the LU
  fallback, the refinement residual and the mixed-precision rebuild);
* the factor is Fortran-ordered and exactly lower triangular, so
  ``cholesky_solve`` back-substitutes it without an O(n²) copy;
* a non-positive-definite matrix raises ``numpy.linalg.LinAlgError`` — the
  same type (and message) as ``numpy.linalg.cholesky`` — which
  ``numerical_failure_types()`` classifies;
* inputs outside the float64 2-D fast path (empty, batched, float32,
  complex, integer) go to ``numpy.linalg.cholesky`` unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")  # the NumPy adapter wraps scipy.linalg.lapack

import ipax.backend.dense.numpy_scipy as adapter_module
from ipax.backend.dense.numpy_scipy import cholesky, cholesky_solve
from ipax.linalg.solver import numerical_failure_types

# Sizes straddling the copy block: below, at, just above, and a ragged tail.
_SIZES = (1, 2, 7, 64, 127, 128, 129, 300, 513, 1000)


def _spd(n: int, dtype: type, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + n)
    m = rng.standard_normal((n, n))
    a = m @ m.T + n * np.eye(n)
    return a.astype(dtype)


@pytest.mark.parametrize("n", _SIZES)
def test_cholesky_matches_numpy_linalg_cholesky(n):
    a = _spd(n, np.float64)
    reference = np.linalg.cholesky(a)
    factor = cholesky(a)
    assert factor.dtype == reference.dtype
    assert factor.shape == reference.shape
    scale = float(np.max(np.abs(reference)))
    assert float(np.max(np.abs(factor - reference))) <= 1e-12 * scale


@pytest.mark.parametrize("n", (5, 129))
def test_cholesky_reads_only_the_lower_triangle(n):
    a = _spd(n, np.float64)
    garbled = a.copy()
    garbled[np.triu_indices(n, 1)] = np.nan  # NumPy ignores the upper triangle too
    np.testing.assert_array_equal(cholesky(garbled), cholesky(a))


@pytest.mark.parametrize("n", (5, 300))
def test_cholesky_does_not_write_to_its_input(n):
    a = _spd(n, np.float64)
    before = a.copy()
    a.flags.writeable = False  # any write would raise
    cholesky(a)
    np.testing.assert_array_equal(a, before)


@pytest.mark.parametrize("n", (1, 5, 300))
def test_cholesky_factor_is_fortran_ordered_and_exactly_lower(n):
    factor = cholesky(_spd(n, np.float64))
    assert factor.flags["F_CONTIGUOUS"]
    np.testing.assert_array_equal(np.triu(factor, 1), np.zeros((n, n)))


def test_cholesky_factors_in_place_on_its_single_copy(monkeypatch):
    # One n×n allocation: the copy handed to potrf is factored in place
    # (``overwrite_a``) and returned, not copied again by the f2py wrapper.
    n = 200
    seen: dict[str, object] = {}
    real_potrf = adapter_module._dpotrf

    def spy(a, **kwargs):
        seen["input"] = a
        seen["kwargs"] = kwargs
        return real_potrf(a, **kwargs)

    monkeypatch.setattr(adapter_module, "_dpotrf", spy)
    a = _spd(n, np.float64)
    factor = cholesky(a)
    work = seen["input"]
    assert isinstance(work, np.ndarray)
    assert work.flags["F_CONTIGUOUS"]
    assert not np.shares_memory(work, a)
    assert np.shares_memory(factor, work)
    assert seen["kwargs"] == {"lower": 1, "overwrite_a": 1, "clean": 1}


def test_cholesky_factor_backsubstitutes_without_a_copy(monkeypatch):
    # The layout contract with ``cholesky_solve``: an F-ordered lower factor
    # takes the ``(L, lower=True)`` ``potrs`` path directly.
    n = 200
    a = _spd(n, np.float64)
    factor = cholesky(a)
    captured: dict[str, object] = {}
    real_cho_solve = adapter_module._scipy_cho_solve

    def spy(c_and_lower, b, **kwargs):
        captured["c"], captured["lower"] = c_and_lower
        return real_cho_solve(c_and_lower, b, **kwargs)

    monkeypatch.setattr(adapter_module, "_scipy_cho_solve", spy)
    rhs = np.random.default_rng(3).standard_normal(n)
    x = cholesky_solve(factor, rhs)
    assert captured["lower"] is True
    assert captured["c"] is factor
    np.testing.assert_allclose(a @ x, rhs, rtol=0, atol=1e-10 * np.abs(rhs).max())


def test_cholesky_accepts_a_strided_view():
    # The equality-saddle guard factors ``matrix[:n, :n]``, a non-contiguous view.
    big = _spd(9, np.float64)
    view = big[:6, :6]
    assert not view.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(
        cholesky(view), np.linalg.cholesky(view), rtol=0, atol=1e-13 * big.max()
    )


def test_cholesky_accepts_a_fortran_ordered_input():
    a = np.asfortranarray(_spd(130, np.float64))
    np.testing.assert_allclose(
        cholesky(a), np.linalg.cholesky(a), rtol=0, atol=1e-12 * a.max()
    )


@pytest.mark.parametrize(
    "matrix",
    [
        [[-1.0]],
        [[0.0]],
        [[1.0, 2.0], [2.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1e-300]],
    ],
    ids=["negative", "zero", "indefinite", "last-pivot"],
)
def test_cholesky_breakdown_raises_numpys_linalg_error(matrix):
    a = np.asarray(matrix, dtype=np.float64)
    with pytest.raises(np.linalg.LinAlgError) as expected:
        np.linalg.cholesky(a)
    with pytest.raises(np.linalg.LinAlgError) as got:
        cholesky(a)
    assert str(got.value) == str(expected.value)
    assert isinstance(got.value, numerical_failure_types())


def test_cholesky_large_indefinite_raises():
    a = _spd(300, np.float64)
    a[150, 150] = -1.0e6
    with pytest.raises(np.linalg.LinAlgError):
        cholesky(a)


def test_cholesky_nan_input_does_not_raise_or_hide_the_nan():
    # Whether ``potrf`` flags a NaN pivot is a LAPACK build property
    # (reference ``dpotrf2`` does, OpenBLAS does not). The contract the dense
    # route needs either way: never a non-numerical exception, and a NaN that
    # is not flagged must stay in the factor, so it reaches the solution.
    a = np.array([[4.0, 2.0], [2.0, np.nan]])
    try:
        factor = cholesky(a)
    except np.linalg.LinAlgError:
        return
    assert factor[0, 0] == 2.0 and factor[1, 0] == 1.0
    assert np.isnan(factor[1, 1])


@pytest.mark.parametrize(
    "a",
    [
        np.zeros((0, 0)),
        np.stack([np.eye(3) * 2.0, np.eye(3) * 3.0]),
        np.array([[4.0 + 0j, 1.0j], [-1.0j, 3.0]]),
        np.array([[4, 2], [2, 3]]),
        np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float32),
    ],
    ids=["empty", "batched", "complex", "integer", "float32"],
)
def test_cholesky_outside_the_fast_path_defers_to_numpy(a, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("potrf reached outside the float64 2-D fast path")

    monkeypatch.setattr(adapter_module, "_dpotrf", unexpected)
    np.testing.assert_array_equal(cholesky(a), np.linalg.cholesky(a))
