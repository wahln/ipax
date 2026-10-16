# Copyright 2026 Niklas Wahl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NumPy/SciPy dense adapters: Cholesky (``?potrf``/``?potrs``) and LDLT (``?sytrf``)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import cho_solve as _scipy_cho_solve
from scipy.linalg import ldl as _scipy_ldl
from scipy.linalg import solve_triangular
from scipy.linalg.lapack import dpotrf as _dpotrf

from ipax.linalg.solver import LinearSolveError


def _float64_potrf(dtype: np.dtype[Any]) -> Any:
    """LAPACK ``dpotrf`` for a float64 ``dtype``, else ``None``.

    float64 only — the dense IPM's working precision, and the dtype on which
    the factor was verified bit-for-bit identical to ``numpy.linalg.
    cholesky``'s (NumPy and SciPy wheels each ship their own OpenBLAS build;
    for float32 their ``spotrf`` results differ in the last bits, so a
    float32 problem would see a changed trajectory for a lever nobody has
    measured there). Every other dtype goes to ``numpy.linalg.cholesky``
    unchanged. Resolved through the module global at call time (not a
    per-call ``get_lapack_funcs``), so tests can spy on the factorization.
    """
    return _dpotrf if dtype == np.float64 else None


# Edge of the square blocks the lower triangle is copied in (elements). Each
# block is read row-major and written column-major; at 128 both 128 KiB fp64
# blocks stay in a P-core's L2, so the transposing copy runs at near-streaming
# speed (measured 0.15-0.18 s at n = 9984 for block edges 64-128 on an
# i9-13900HX, against 0.9 s for NumPy's unblocked ``asfortranarray``).
_TRIANGLE_COPY_BLOCK = 128

_NOT_POSITIVE_DEFINITE = "Matrix is not positive definite"  # NumPy's own message


def _copy_lower_triangle_fortran(a: np.ndarray) -> np.ndarray:
    """A Fortran-ordered ``n x n`` buffer holding ``a``'s lower triangle.

    Copies only the block lower triangle (diagonal blocks whole), in square
    cache-sized blocks so the C-to-Fortran transposition of each block stays
    in cache; the strictly-upper blocks are left uninitialized, which is safe
    because ``potrf(uplo='L')`` never reads them and ``clean=1`` zeroes them.
    Works for any input strides (a C-ordered matrix, a ``[:n, :n]`` view of an
    equality saddle, a Fortran-ordered matrix) and never writes to ``a``.
    """
    n = a.shape[0]
    work = np.empty((n, n), dtype=a.dtype, order="F")
    block = _TRIANGLE_COPY_BLOCK
    for j in range(0, n, block):
        j_end = min(j + block, n)
        for i in range(j, n, block):
            i_end = min(i + block, n)
            work[i:i_end, j:j_end] = a[i:i_end, j:j_end]
    return work


def cholesky(matrix: Any) -> np.ndarray:
    """Lower Cholesky factor ``L`` (``A = L Lᵀ``) of a symmetric PD matrix.

    Same factorization as ``numpy.linalg.cholesky`` — LAPACK ``?potrf`` with
    ``uplo='L'`` on a Fortran-ordered copy of ``A``'s lower triangle (Golub &
    Van Loan 2013, §4.2) — without NumPy's layout round trip. The Array-API
    ``linalg`` extension specifies ``cholesky`` but not its memory layout, and
    NumPy's gufunc returns a fresh C-ordered factor: it linearizes ``A`` into a
    Fortran-ordered scratch buffer, factors, zeroes the upper triangle and
    transposes the result back, i.e. two unblocked transposing ``n x n``
    copies around the ``n³/3``-flop factorization. At ``n = 9984`` (TROTS
    Head-and-Neck_01's condensed ``N``) each copy costs about as much as
    ``potrf`` itself (0.9 s vs 0.87 s on an i9-13900HX, see
    ``docs/benchmarks/trots.md``). Here the only copy is a cache-blocked one of
    the lower triangle into a Fortran-ordered buffer that ``potrf`` factors in
    place (``overwrite_a=1``), so the call allocates one ``n x n`` array
    instead of two and returns ``L`` Fortran-ordered — the layout
    :func:`cholesky_solve`'s ``potrs`` takes without a copy.

    Contract, matching ``numpy.linalg.cholesky``:

    * only the lower triangle of ``matrix`` is read, and ``matrix`` is never
      written to (``DenseSolver`` keeps it for the LU fallback, the
      refinement residual and the mixed-precision rebuild);
    * the returned factor is exactly lower triangular (``clean=1``);
    * a breakdown (``potrf`` ``INFO > 0``: a pivot ``l_kk² ≤ 0``) raises
      ``numpy.linalg.LinAlgError("Matrix is not positive definite")`` — the
      type and message NumPy raises, which
      :func:`~ipax.linalg.solver.numerical_failure_types` classifies. A NaN
      pivot is flagged only if SciPy's LAPACK flags it: reference LAPACK's
      ``dpotrf2`` tests ``DISNAN(AJJ)``, OpenBLAS's own ``potrf`` (the
      NumPy and SciPy wheels) does not and returns a NaN factor — as
      ``numpy.linalg.cholesky`` does on those wheels. Either way a NaN
      reaches the solution, which the ``LinearSolver`` contract already
      treats as a failed solve;
    * inputs outside the float64 2-D fast path (empty, batched, float32,
      complex, integer and other dtypes) are passed to
      ``numpy.linalg.cholesky``.

    Both run ``dpotrf('L')`` on the same Fortran-ordered data, so the factor
    is bit-for-bit NumPy's wherever the two OpenBLAS builds agree on it — as
    they do for the NumPy 2.4.6 / SciPy 1.17.1 wheels (OpenBLAS 0.3.31 / 0.3.30,
    both with Haswell kernels) this was measured on; with other builds (e.g.
    an MKL NumPy) it agrees to round-off.
    """
    a = np.asarray(matrix)
    potrf = _float64_potrf(a.dtype) if a.ndim == 2 else None
    if potrf is None or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        result: np.ndarray = np.linalg.cholesky(a)
        return result
    work = _copy_lower_triangle_fortran(a)
    factor, info = potrf(work, lower=1, overwrite_a=1, clean=1)
    if info > 0:
        raise np.linalg.LinAlgError(_NOT_POSITIVE_DEFINITE)
    if info < 0:  # pragma: no cover - an illegal argument is a wrapper bug
        raise ValueError(f"LAPACK potrf: illegal value in argument {-info}")
    return np.asarray(factor)


def cholesky_solve(factor: Any, rhs: Any) -> np.ndarray:
    """Solve ``(L Lᵀ) x = rhs`` given the lower Cholesky factor ``L``.

    Gap-filler for the missing Array-API primitive: the ``linalg`` extension
    has no triangular solve (BLAS ``trsm`` / LAPACK ``potrs``), so a Cholesky
    factor cannot be back-substituted portably. Wraps ``scipy.linalg.cho_solve``
    (LAPACK ``?potrs``), O(n²) per right-hand side. ``check_finite=False``:
    the factor came from a ``cholesky`` that already succeeded on this data.

    ``numpy.linalg.cholesky`` returns ``L`` C-contiguous, but LAPACK's
    ``potrs`` wants a Fortran-ordered factor; SciPy's f2py wrapper bridges the
    mismatch with a full ``O(n²)`` copy on *every* back-solve (measured 0.91 s
    vs 0.053 s at n≈10⁴; see CHANGELOG). ``L``'s transpose view ``Lᵀ`` is
    Fortran-contiguous over the same data at zero cost, and is exactly the
    upper factor ``U`` with ``A = Uᵀ U``: passing ``(U, lower=False)`` runs
    the mathematically identical ``potrs`` with ``uplo='U'`` (solves
    ``Uᵀ y = b``, ``U x = y``) without the copy. If a caller ever hands in a
    factor that is already Fortran-ordered (not the case for the
    ``xp.linalg.cholesky`` caller today, but not guaranteed by the gap-filler
    contract either), the plain lower-``L`` call is used instead so neither
    orientation pays a copy. :func:`cholesky` (the NumPy route's factorization
    since it replaced ``numpy.linalg.cholesky``) returns exactly such a
    Fortran-ordered factor, so the dense route takes that branch.
    """
    ell = np.asarray(factor)
    b = np.asarray(rhs)
    if ell.flags["F_CONTIGUOUS"] and not ell.flags["C_CONTIGUOUS"]:
        result: np.ndarray = _scipy_cho_solve((ell, True), b, check_finite=False)
    else:
        result = _scipy_cho_solve((ell.T, False), b, check_finite=False)
    return result


def add_diagonal(matrix: Any, diagonal: Any) -> np.ndarray:
    """Return ``matrix + diag(diagonal)`` in O(n) beyond the unavoidable copy.

    Gap-filler for a missing Array-API primitive: the standard has no
    strided diagonal setter (no portable way to add to a matrix's diagonal
    without either mutating in place through non-standard fancy indexing, or
    building a full ``eye(n) * diagonal`` term — an extra O(n²) allocation
    and elementwise multiply on top of the O(n²) copy that is unavoidable to
    return a new matrix). ``numpy.einsum("ii->i", a)`` is NumPy's documented
    idiom for a writable O(n) view of ``a``'s diagonal (unlike
    ``a.diagonal()``, which is read-only since NumPy 1.9); ``diagonal`` may
    be a vector or a scalar (broadcasts, e.g. a ``delta_w * I`` term).

    Bitwise identical (for finite inputs) to ``matrix + xp.eye(n) *
    diagonal``: every off-diagonal term is ``x + 0 = x`` exactly, and every
    diagonal term is the same single float addition — *at the same dtype*
    the old expression would have promoted to. That promotion is not always
    ``matrix.dtype``: under NEP 50, ``eye(dtype=matrix.dtype) * diagonal``
    keeps ``matrix.dtype`` when ``diagonal`` is a Python scalar (a "weak"
    type) but promotes to ``diagonal``'s dtype when it is a wider-precision
    array — e.g. a float64 vector against a float32 ``matrix``. Casting
    ``diagonal`` to ``matrix.dtype`` unconditionally silently narrows that
    case, and double-rounds a float32 scalar delta computed via a float64
    intermediate. ``np.result_type`` reproduces the old expression's exact
    NEP 50 promotion decision without paying for the O(n²) multiply.

    Non-finite inputs need not reproduce the old formula's off-diagonal
    ``0 * inf/nan`` footprint. KKT assembly marks non-finite addends as NaN
    before calling this helper: an infinite pivot alone can back-substitute
    to a finite zero and bypass the driver's finite-solution check.
    """
    dtype = np.result_type(np.asarray(matrix).dtype, diagonal)
    result = np.array(matrix, dtype=dtype, copy=True)
    np.einsum("ii->i", result)[...] += np.asarray(diagonal, dtype=dtype)
    return result


def _ldl_blocks(d: np.ndarray) -> list[tuple[int, int]]:
    """Identify the ``(start, size)`` of each 1x1/2x2 block along ``d``.

    ``scipy.linalg.ldl``'s block-diagonal ``d`` signals a 2x2 Bunch-Kaufman
    pivot with a nonzero first off-diagonal entry ``d[i, i+1]``.
    """
    n = d.shape[0]
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if i + 1 < n and d[i, i + 1] != 0.0:
            blocks.append((i, 2))
            i += 2
        else:
            blocks.append((i, 1))
            i += 1
    return blocks


def _block_diagonal_solve(
    d: np.ndarray, blocks: list[tuple[int, int]], rhs: np.ndarray
) -> np.ndarray:
    """Solve ``d @ x = rhs`` in ``O(n)`` by walking the 1x1/2x2 blocks."""
    out = np.empty_like(rhs)
    for start, size in blocks:
        if size == 1:
            out[start] = rhs[start] / d[start, start]
        else:
            block = d[start : start + 2, start : start + 2]
            out[start : start + 2] = np.linalg.solve(block, rhs[start : start + 2])
    return out


def _sign_count_blocks(
    d: np.ndarray, blocks: list[tuple[int, int]]
) -> tuple[int, int, int]:
    """Sign-count the block-diagonal ``d`` into ``(n_pos, n_neg, n_zero)``.

    A 1x1 block's sign is its (tolerance-gated) diagonal entry. A genuine 2x2
    Bunch-Kaufman pivot always contributes exactly one positive and one
    negative eigenvalue (its determinant is negative by construction — that
    is precisely why the algorithm chose a 2x2 pivot over a 1x1 one); this is
    verified via the block's determinant rather than assumed.
    """
    n = d.shape[0]
    scale = float(np.max(np.abs(d))) if n else 0.0
    tol = max(1.0, scale) * n * float(np.finfo(d.dtype).eps)
    pos = neg = zero = 0
    for start, size in blocks:
        if size == 1:
            value = d[start, start]
            if value > tol:
                pos += 1
            elif value < -tol:
                neg += 1
            else:
                zero += 1
        else:
            a, b, c = d[start, start], d[start, start + 1], d[start + 1, start + 1]
            det = a * c - b * b
            if det < -tol:
                pos += 1
                neg += 1
            else:
                # Should never happen for a valid Bunch-Kaufman block; treat
                # defensively as numerically singular rather than guessing.
                zero += 2
    return pos, neg, zero


class ScipyLDLFactorization:
    """Bunch-Kaufman LDLᵀ factorization of a dense symmetric matrix.

    Wraps ``scipy.linalg.ldl`` (LAPACK ``?sytrf``): a genuinely pivoted
    symmetric-indefinite factorization, unlike a plain ``eigh``, which does
    not reorder around ill-scaled entries. Solves via the permuted
    triangular factors directly (``O(n^2)`` per solve) instead of a generic
    ``O(n^3)`` solve on the unpermuted factor, and exposes the factorization's
    inertia — the dense analogue of ``FeralSparseSolver``.
    """

    def __init__(self) -> None:
        self._l: np.ndarray | None = None
        self._d: np.ndarray | None = None
        self._perm: np.ndarray | None = None
        self._blocks: list[tuple[int, int]] | None = None
        self._inertia: tuple[int, int, int] | None = None

    def factor(self, matrix: Any) -> None:
        a = np.asarray(matrix)
        try:
            lu, d, perm = _scipy_ldl(a)
        except Exception as exc:
            raise LinearSolveError("dense LDL^T factorization failed") from exc
        self._l = lu[perm, :]
        self._d = d
        self._perm = perm
        self._blocks = _ldl_blocks(d)
        self._inertia = _sign_count_blocks(d, self._blocks)

    def solve(self, rhs: Any) -> np.ndarray:
        if self._l is None or self._d is None or self._perm is None:
            raise RuntimeError("factor() must be called before solve()")
        assert self._blocks is not None
        b = np.asarray(rhs)
        perm = self._perm
        try:
            w = solve_triangular(self._l, b[perm], lower=True, unit_diagonal=True)
            z = _block_diagonal_solve(self._d, self._blocks, w)
            y = solve_triangular(self._l.T, z, lower=False, unit_diagonal=True)
        except Exception as exc:
            raise LinearSolveError("dense LDL^T solve failed") from exc
        x = np.empty_like(b)
        x[perm] = y
        return x

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """Inertia ``(n_pos, n_neg, n_zero)`` of the factored matrix, or ``None``."""
        return self._inertia


__all__ = ["ScipyLDLFactorization", "add_diagonal", "cholesky", "cholesky_solve"]
