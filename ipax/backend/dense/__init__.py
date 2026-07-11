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

"""Per-backend dense symmetric-indefinite (Bunch-Kaufman LDLT) adapters.

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


__all__ = ["get_dense_cholesky_solve", "get_dense_symmetric_indefinite_adapter"]
