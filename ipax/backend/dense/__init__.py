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

"""Per-backend dense adapters: Bunch-Kaufman LDLT, Cholesky factor/back-solve.

.. warning::
   These modules are the **only** place under ``backend/`` (besides
   ``backend/sparse/``) allowed to import a concrete array library (invariant
   #1/#4).

The augmented dense KKT route (``DenseOptions(kkt_route="augmented")``) keeps
``∇g``/``−Σ_s⁻¹`` as an explicit border instead of condensing the inequality
Gram term into the ``N`` block (see ``ipax.ipm.kkt``). Factoring that bordered
matrix with a pivoted Bunch-Kaufman LDLᵀ (LAPACK/cuSOLVER ``sytrf``) is both a
genuinely pivoted factorization — unlike a plain symmetric eigendecomposition,
which inherits the border's raw entry-scale conditioning — and exposes real
inertia for free, mirroring what Feral/cuDSS already give the sparse-direct
route (Wächter & Biegler 2006 §3.1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from ipax.typing import Array, Namespace


def get_dense_symmetric_indefinite_adapter(xp: Namespace) -> object | None:
    """Return a fresh Bunch-Kaufman LDLᵀ factorization for ``xp``, or ``None``.

    ===========  ================================================
    backend      factorization
    ===========  ================================================
    numpy        ``scipy.linalg.ldl`` (LAPACK ``?sytrf``)
    torch        ``torch.linalg.ldl_factor_ex``/``ldl_solve`` (CPU + CUDA)
    cupy         ``cusolverDnDsytrf``/``Xsytrs`` via nvmath-python (CUDA)
    ===========  ================================================

    ``None`` for any other backend (JAX, or array-api-strict): ``DenseSolver``'s
    augmented route falls back to the Array-API-pure ``eigh``-based path, which
    is always available but not pivoted (a plain eigendecomposition inherits
    the bordered matrix's raw entry-scale conditioning rather than reordering
    around it).

    Import errors for the concrete library (SciPy/Torch/nvmath) yield ``None``
    too — CuPy is user-managed CUDA, so a missing nvmath binding or cuSOLVER
    runtime degrades to the same eigh fallback rather than raising.
    """
    from ipax.backend.namespace import _namespace_name

    name = _namespace_name(xp)
    if name == "numpy":
        try:
            from ipax.backend.dense.numpy_scipy import ScipyLDLFactorization
        except ImportError:
            return None
        return ScipyLDLFactorization()
    if name == "torch":
        try:
            from ipax.backend.dense.torch import TorchLDLFactorization
        except ImportError:
            return None
        return TorchLDLFactorization()
    if name == "cupy":
        try:
            from ipax.backend.dense.cupy import CuPyLDLFactorization
        except ImportError:
            return None
        return CuPyLDLFactorization()
    return None


def get_dense_cholesky(xp: Namespace) -> Callable[[Array], Array] | None:
    """Return ``matrix -> L`` (lower Cholesky factor, raising on breakdown), or ``None``.

    Gap-filler for a missing Array-API guarantee: the ``linalg`` extension
    specifies ``cholesky``'s result but not its memory layout or how many
    copies an implementation makes, and NumPy's gufunc wraps LAPACK ``potrf``
    in two unblocked transposing ``n x n`` copies (C-ordered in, Fortran-ordered
    scratch, C-ordered out) that at ``n ≈ 10⁴`` cost twice the factorization
    itself. The adapter calls ``potrf`` directly on a single cache-blocked
    Fortran-ordered copy and returns the Fortran-ordered factor that the
    matching :func:`get_dense_cholesky_solve` back-substitutes without a copy:

    ===========  ================================================
    backend      primitive
    ===========  ================================================
    numpy        ``scipy.linalg.lapack.dpotrf`` (float64, one blocked copy;
                 other dtypes: ``numpy.linalg.cholesky``)
    ===========  ================================================

    Contract (what ``DenseSolver`` relies on): reads only the lower triangle,
    never writes to its input, returns an exactly lower-triangular factor
    that :func:`get_dense_cholesky_solve`'s adapter accepts, and *raises* the
    backend's own numerical failure type on a breakdown (a member of
    :func:`~ipax.linalg.solver.numerical_failure_types`) — so callers need no
    silent-breakdown pivot check for it.

    ``None`` for every other backend or when SciPy is unimportable: callers
    use ``xp.linalg.cholesky``. The two other adapter backends already factor
    without NumPy's double transposition: ``cupy.linalg.cholesky`` (CuPy 14)
    makes one C-ordered copy, runs cuSOLVER ``potrf`` with ``uplo='U'`` on it
    (the upper triangle of the Fortran view ``Aᵀ``, i.e. ``A``'s lower one)
    and zeroes the other triangle in one elementwise kernel, and
    ``cupyx.scipy.linalg`` has no other Cholesky entry point;
    ``torch.linalg.cholesky`` returns a column-major factor computed by
    LAPACK/MAGMA ``potrf`` in 0.87 s at ``n = 9984`` (CPU, MKL) — what
    ``potrf`` alone costs there. Neither has a cheaper call to inject.
    """
    from ipax.backend.namespace import _namespace_name

    if _namespace_name(xp) == "numpy":
        try:
            from ipax.backend.dense.numpy_scipy import cholesky as _numpy_cholesky
        except ImportError:
            return None
        return _numpy_cholesky
    return None


def get_dense_cholesky_solve(xp: Namespace) -> Callable[[Array, Array], Array] | None:
    """Return ``(L, rhs) -> x`` solving ``(L Lᵀ) x = rhs``, or ``None``.

    Gap-filler for a missing Array-API primitive: the ``linalg`` extension has
    **no triangular solve** (BLAS ``trsm`` / LAPACK ``potrs``), so the Cholesky
    factor that ``DenseSolver``'s PD guard already computed cannot be
    back-substituted portably — a pure-standard caller would have to refactor
    the matrix with ``xp.linalg.solve`` (LU), paying O(n³) twice per KKT
    factorization (and once more per extra back-solve). Per-backend adapters
    provide the O(n²) back-substitution instead:

    ===========  ================================================
    backend      primitive
    ===========  ================================================
    numpy        ``scipy.linalg.cho_solve`` (LAPACK ``?potrs``)
    torch        ``torch.cholesky_solve`` (CPU + CUDA)
    cupy         ``cupyx.scipy.linalg.solve_triangular`` ×2 (CUDA)
    ===========  ================================================

    ``None`` for any other backend (JAX, array-api-strict) or when the
    concrete library import fails: the caller falls back to the LU solve,
    losing only the reuse speedup, never correctness.
    """
    from ipax.backend.namespace import _namespace_name

    name = _namespace_name(xp)
    if name == "numpy":
        try:
            from ipax.backend.dense.numpy_scipy import (
                cholesky_solve as _numpy_cholesky_solve,
            )
        except ImportError:
            return None
        return _numpy_cholesky_solve
    if name == "torch":
        try:
            from ipax.backend.dense.torch import (
                cholesky_solve as _torch_cholesky_solve,
            )
        except ImportError:
            return None
        return _torch_cholesky_solve
    if name == "cupy":
        try:
            from ipax.backend.dense.cupy import (
                cholesky_solve as _cupy_cholesky_solve,
            )
        except ImportError:
            return None
        return _cupy_cholesky_solve
    return None


def get_dense_diagonal_adder(xp: Namespace) -> Callable[[Array, Array], Array] | None:
    """Return ``(matrix, diagonal) -> matrix + diag(diagonal)`` in O(n), or ``None``.

    Gap-filler for a missing Array-API primitive: the standard has no
    strided diagonal setter, so a pure-standard caller must build the
    diagonal update via ``xp.eye(n) * diagonal`` — an extra O(n²) identity
    allocation and elementwise multiply on top of the O(n²) copy already
    needed to return a new matrix. Per-backend adapters instead mutate a
    writable O(n) diagonal view of a fresh copy:

    ===========  ================================================
    backend      primitive
    ===========  ================================================
    numpy        ``numpy.einsum("ii->i", a)`` (writable diagonal view)
    torch        ``torch.Tensor.diagonal()`` (writable view) + ``add_``
    ===========  ================================================

    ``None`` for any other backend (CuPy, JAX, array-api-strict) or when the
    concrete library import fails: the caller falls back to the ``eye``-based
    formula, losing only the speedup, never correctness (both formulas are
    bitwise identical for finite inputs, at the same promoted dtype — see
    each adapter's docstring for the NEP 50 / Torch promotion subtlety this
    must reproduce). CuPy is left on the fallback here: its
    ``ndarray.diagonal()`` is documented as a writable view (unlike NumPy's
    own method), which would make the same O(n) update correct, but this has
    not been exercised on real CUDA hardware.
    """
    from ipax.backend.namespace import _namespace_name

    name = _namespace_name(xp)
    if name == "numpy":
        try:
            from ipax.backend.dense.numpy_scipy import (
                add_diagonal as _numpy_add_diagonal,
            )
        except ImportError:
            return None
        return _numpy_add_diagonal
    if name == "torch":
        try:
            from ipax.backend.dense.torch import add_diagonal as _torch_add_diagonal
        except ImportError:
            return None
        return _torch_add_diagonal
    return None


__all__ = [
    "get_dense_cholesky",
    "get_dense_cholesky_solve",
    "get_dense_diagonal_adder",
    "get_dense_symmetric_indefinite_adapter",
]
