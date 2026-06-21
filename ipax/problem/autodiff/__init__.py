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

"""Autodiff adapters (one per capable backend).

.. warning::
   These modules are the **only** place under ``problem/`` allowed to import a
   concrete array library (invariant #1). Each adapter is loaded lazily, by
   namespace, and exposes grad / Jacobian / Hessian-vector-product callbacks.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from ipax.backend.namespace import capabilities

if TYPE_CHECKING:
    from ipax.typing import Namespace


# Backend name → adapter module (relative to this package).
_ADAPTER_MODULES = {
    "torch": "ipax.problem.autodiff.torch",
    "jax": "ipax.problem.autodiff.jax",
}


def get_autodiff_adapter(xp: Namespace) -> object | None:
    """Return the autodiff adapter for ``xp``'s backend, or ``None`` if absent.

    Dispatches on the namespace name (``torch`` → ``.torch``, ``jax`` →
    ``.jax``) and imports the adapter lazily so the concrete library is only
    touched when actually used. Returns ``None`` for backends without an
    autodiff adapter (e.g. NumPy, ``array_api_strict``), letting the caller fall
    back to finite differences.
    """
    name = capabilities(xp).name
    module_path = _ADAPTER_MODULES.get(name)
    if module_path is None:
        return None
    try:
        return import_module(module_path)
    except ImportError:  # pragma: no cover - backend present but adapter deps absent
        return None


__all__ = ["get_autodiff_adapter"]
