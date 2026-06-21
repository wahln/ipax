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

"""Per-backend sparse adapters.

.. warning::
   These modules are the **only** place under ``backend/`` allowed to import a
   concrete sparse library (invariant #1/#4). The core emits structure as
   Array-API integer/value vectors (row, col, value); each adapter converts and
   factors. The IPM never sees a backend-specific sparse object.

Each adapter implements :class:`~ipax.backend.operators.LinearOperator`
(``SparseOperator``) plus the ``SparseDirectSolver`` protocol from
``ipax.linalg.solver``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipax.typing import Namespace


def get_sparse_adapter(xp: Namespace) -> object | None:
    """Return the sparse adapter registered for ``xp``'s backend, or ``None``.

    Dispatch table (loaded lazily):

    ===========  =================================  ==========================
    backend      sparse build                       factorization
    ===========  =================================  ==========================
    numpy/scipy  ``scipy.sparse`` (COO→CSC)         Feral LDLᵀ / SuperLU fallback
    cupy         ``cupyx.scipy.sparse``             cuDSS / ``cupyx…spsolve``
    torch        device-routed (DLPack)             SciPy (CPU) / cuDSS (CUDA)
    jax          device-routed (DLPack)             SciPy (CPU) / cuDSS (CUDA)
    ===========  =================================  ==========================

    Torch and JAX have no native sparse-direct path of their own; instead the
    :class:`~ipax.backend.sparse._routing.DeviceRoutingSparseAdapter` reinterprets
    their COO buffer (via DLPack) onto the SciPy adapter for CPU arrays and the
    CuPy/cuDSS adapter for CUDA arrays — zero-copy where the libraries allow it.
    The actual SciPy/CuPy import (and its possible failure) is deferred to
    ``from_coo``, once the value array's device is known.

    Import errors for the sparse array library (e.g. SciPy/CuPy) yield ``None``.
    A missing nvmath binding or user-managed cuDSS runtime selects the CuPy
    adapter's solve-only fallback instead.
    """
    from ipax.backend.namespace import _namespace_name

    name = _namespace_name(xp)
    if name == "numpy":
        try:
            from ipax.backend.sparse.numpy_scipy import SciPySparseAdapter
        except ImportError:
            return None
        return SciPySparseAdapter()
    if name == "cupy":
        try:
            from ipax.backend.sparse.cupy import CuPySparseAdapter
        except ImportError:
            return None
        return CuPySparseAdapter()
    if name in ("torch", "jax"):
        from ipax.backend.sparse._routing import DeviceRoutingSparseAdapter

        return DeviceRoutingSparseAdapter()
    return None


__all__ = ["get_sparse_adapter"]
