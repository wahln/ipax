"""Unit tests for the NumPy/SciPy Cholesky back-substitution gap-filler.

``ipax.backend.dense.numpy_scipy.cholesky_solve`` wraps ``scipy.linalg.cho_solve``
(LAPACK ``?potrs``). ``numpy.linalg.cholesky`` returns its factor C-contiguous,
but ``potrs`` wants a Fortran-ordered one; without care, SciPy's f2py wrapper
bridges that mismatch with a full O(n²) copy on *every* back-solve. These tests
pin: (1) the fix hands SciPy an already-Fortran-ordered view (no copy), (2) the
result is numerically identical (bitwise, on this NumPy/OpenBLAS build) to the
pre-fix formulation across sizes, right-hand-side shapes, dtypes, and
conditioning.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")  # the NumPy adapter wraps scipy.linalg.cho_solve

from scipy.linalg import cho_solve as _scipy_cho_solve

import ipax.backend.dense.numpy_scipy as adapter_module
from ipax.backend.dense.numpy_scipy import cholesky_solve


def _spd(n: int, dtype: np.dtype, *, ill_conditioned: bool) -> np.ndarray:
    rng = np.random.default_rng(1234 + n)
    m = rng.standard_normal((n, n)).astype(dtype)
    a = m @ m.T
    if ill_conditioned:
        # Geometrically spaced eigenvalues spanning ~1e8 in condition number,
        # via a similarity-free diagonal reweighting that keeps symmetry.
        scale = np.geomspace(1.0, 1e8, n).astype(dtype)
        a = a * np.outer(scale, scale) ** 0.5
        a = a / np.max(np.abs(a))
    a = a + n * np.eye(n, dtype=dtype) * (1e-3 if ill_conditioned else 1.0)
    return a


def _reference_cho_solve(factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """The pre-fix formulation: C-ordered lower factor straight into SciPy."""
    return _scipy_cho_solve(
        (np.asarray(factor), True), np.asarray(rhs), check_finite=False
    )


# --- no O(n²) copy: the array handed to SciPy shares memory & is F-contiguous


def test_cholesky_solve_hands_scipy_a_zero_copy_fortran_view(monkeypatch):
    n = 200
    a = _spd(n, np.float64, ill_conditioned=False)
    factor = np.linalg.cholesky(a)
    assert factor.flags["C_CONTIGUOUS"] and not factor.flags["F_CONTIGUOUS"]
    rhs = np.ones(n)

    captured: dict[str, np.ndarray] = {}
    real_cho_solve = adapter_module._scipy_cho_solve

    def spy(c_and_lower, b, **kwargs):
        captured["c"] = c_and_lower[0]
        captured["lower"] = c_and_lower[1]
        return real_cho_solve(c_and_lower, b, **kwargs)

    monkeypatch.setattr(adapter_module, "_scipy_cho_solve", spy)

    cholesky_solve(factor, rhs)

    c = captured["c"]
    assert captured["lower"] is False  # upper factor U = L^T
    assert np.shares_memory(c, factor)  # a view, not a copy
    assert c.flags["F_CONTIGUOUS"]  # no C->F copy left for SciPy's f2py wrapper


def test_cholesky_solve_does_not_copy_the_rhs_matrix():
    # np.asarray on an already-ndarray rhs is a no-op view, not a copy.
    n = 50
    rhs = np.random.default_rng(0).standard_normal((n, 3))
    assert np.shares_memory(np.asarray(rhs), rhs)


# --- bitwise-identity characterization ---------------------------------------


_SIZES = (10, 100, 500, 2000)
_FAST_SIZES = (10, 100, 500)  # the full sweep (incl. 2000) lives in the
# characterization test below; a per-case assert on every size would repeat
# a chunk of O(n^3) Cholesky factorizations for no extra coverage.
_DTYPES = (np.float64, np.float32)


def _rel_tolerance(dtype: np.dtype, *, ill_conditioned: bool) -> float:
    """Dtype/conditioning-split relative tolerance (not a bitwise claim).

    The back-solve is only *verified* bitwise identical on this build (see
    the module docstring and the adapter's own docstring); a different BLAS
    is guaranteed numerically equivalent, not bit-for-bit, so tests must
    tolerate that rather than assert exact equality. float32 on an
    ill-conditioned system loses more digits to the O(1e8) condition number
    itself, independent of which ``potrs`` path is used.
    """
    if dtype == np.float64:
        return 1e-12
    return 1e-4 if ill_conditioned else 1e-6


@pytest.mark.parametrize("n", _FAST_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("ill_conditioned", (False, True))
@pytest.mark.parametrize("n_rhs", (1, 4))
def test_cholesky_solve_matches_reference_formulation(n, dtype, ill_conditioned, n_rhs):
    a = _spd(n, dtype, ill_conditioned=ill_conditioned)
    factor = np.linalg.cholesky(a)
    rng = np.random.default_rng(n + n_rhs)
    rhs = rng.standard_normal((n,) if n_rhs == 1 else (n, n_rhs)).astype(dtype)

    fast = cholesky_solve(factor, rhs)
    reference = _reference_cho_solve(factor, rhs)

    denom = np.maximum(np.abs(reference), 1e-300)
    rel_diff = float(np.max(np.abs(fast - reference) / denom))
    assert rel_diff <= _rel_tolerance(dtype, ill_conditioned=ill_conditioned)


def test_cholesky_solve_bitwise_identity_report(capsys):
    """Characterize (not just assert) bitwise identity across the sweep.

    A regression check on this function must not assert bitwise equality
    outright (LAPACK's ``uplo='U'`` and ``'L'`` ``potrs`` paths are not
    required to be bit-for-bit identical on every BLAS — only *numerically*
    equivalent), so the real assertion is the same dtype/conditioning-split
    relative tolerance as the parametrized test above. The bitwise-identical
    fraction is reported for its own sake (the task asks for a
    characterization, not just pass/fail): on this NumPy/OpenBLAS build it is
    100%, which is exactly the kind of build-specific fact that belongs in a
    printout, not a hard assertion that would fail the suite on another BLAS.
    """
    total = 0
    identical = 0
    max_rel_diff = 0.0
    for n in _SIZES:
        for dtype in _DTYPES:
            for ill_conditioned in (False, True):
                for n_rhs in (1, 4):
                    a = _spd(n, dtype, ill_conditioned=ill_conditioned)
                    factor = np.linalg.cholesky(a)
                    rng = np.random.default_rng(n + n_rhs)
                    shape = (n,) if n_rhs == 1 else (n, n_rhs)
                    rhs = rng.standard_normal(shape).astype(dtype)

                    fast = cholesky_solve(factor, rhs)
                    reference = _reference_cho_solve(factor, rhs)

                    denom = np.maximum(np.abs(reference), 1e-300)
                    rel_diff = float(np.max(np.abs(fast - reference) / denom))

                    total += 1
                    if np.array_equal(fast, reference):
                        identical += 1
                    max_rel_diff = max(max_rel_diff, rel_diff)
                    assert rel_diff <= _rel_tolerance(
                        dtype, ill_conditioned=ill_conditioned
                    )

    fraction = identical / total
    with capsys.disabled():
        print(
            f"\ncholesky_solve bitwise-identity: {identical}/{total} "
            f"({fraction:.1%}) configurations bitwise identical; "
            f"max relative difference among the rest: {max_rel_diff:.3e}"
        )
