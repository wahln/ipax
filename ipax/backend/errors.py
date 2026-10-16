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

"""Backend linear-algebra failure types (the core/backend boundary, invariant #1).

Gap-filler for a missing Array-API guarantee: the standard's ``linalg``
extension does not specify *how* ``cholesky``/``solve``/``eigh`` report a
numerical failure (a non-PD matrix, an exactly singular one), so every library
raises its own type — or none at all:

==================  ==========================================================
namespace           numerical failure of ``xp.linalg.cholesky``/``solve``
==================  ==========================================================
numpy               ``numpy.linalg.LinAlgError`` (a ``ValueError`` subclass);
                    NaN input returns NaN, no exception
array-api-strict    NumPy underneath: ``numpy.linalg.LinAlgError``
SciPy (adapters)    ``scipy.linalg.LinAlgError`` *is* NumPy's class
torch               ``torch.linalg.LinAlgError`` (a ``RuntimeError`` subclass;
                    also raised for a NaN Cholesky input)
cupy                ``cupy.linalg.LinAlgError`` *is* NumPy's class; raised only
                    under ``cupyx.errstate(linalg="raise")`` — by default a
                    failed cuSOLVER call returns garbage/NaN silently
jax                 none — ``jnp.linalg`` returns NaN; no exception to classify
==================  ==========================================================

The core may not import the libraries (invariant #1), so it asks this module,
which names the types only as ``sys.modules`` keys and must stay import-free:
it lives outside the adapter directories on purpose, so the import-purity gate
(``scripts/check_purity.py``) rejects any concrete-library import added here —
one would otherwise run on the first solve failure of every run. :func:`ipax.linalg.solver.numerical_failure_types` combines
the result with ipax's own :class:`~ipax.linalg.solver.LinearSolveError` and
Python's ``ArithmeticError`` family into the set the KKT solve sites translate
into ``LinearSolveError`` — anything outside it (a ``TypeError`` from a user
operator, say) is a bug and propagates unchanged.
"""

from __future__ import annotations

import sys

# Libraries whose ``<module>.linalg.LinAlgError`` marks a numerical failure.
# ``scipy`` and ``cupy`` currently re-export NumPy's class; they are listed
# anyway so a future divergence is still recognized. JAX has no such type.
_LINALG_ERROR_MODULES = ("numpy", "scipy", "torch", "cupy")


def backend_linalg_error_types() -> tuple[type[BaseException], ...]:
    """The ``LinAlgError`` classes of every array library *already imported*.

    Reads ``sys.modules`` instead of importing: an exception instance of a
    library's type cannot exist unless that library is loaded, so a lookup
    never pays (or triggers) a multi-second Torch/CuPy import on a NumPy run.
    Called only on the failure path (inside an ``except`` clause), so it costs
    nothing on a successful solve.
    """
    found: list[type[BaseException]] = []
    for name in _LINALG_ERROR_MODULES:
        # The ``linalg`` submodule itself, not ``getattr(module, "linalg")``:
        # SciPy loads submodules lazily on attribute access, which would turn
        # a lookup into an import.
        linalg = sys.modules.get(f"{name}.linalg")
        error = getattr(linalg, "LinAlgError", None) if linalg is not None else None
        if (
            isinstance(error, type)
            and issubclass(error, BaseException)
            and error not in found
        ):
            found.append(error)
    return tuple(found)


__all__ = ["backend_linalg_error_types"]
