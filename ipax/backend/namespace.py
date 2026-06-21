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

"""Array-API namespace resolution and capability detection (§5.4).

This is the single place the core asks "what backend am I on, and what can it
do?". It does **not** import any concrete array library — it uses
``array-api-compat`` to resolve the namespace from the input arrays and probes
the resolved namespace for optional features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipax.typing import Array, Namespace


def array_namespace(*arrays: Array) -> Namespace:
    """Return the common Array-API namespace for ``arrays``.

    Thin wrapper over ``array_api_compat.array_namespace`` so the rest of the
    core has a single import site. Raises if the arrays disagree on backend.
    """
    # Deferred to keep the top-level import free of the compat shim until used.
    from array_api_compat import array_namespace as _ns

    return _ns(*arrays)


_ARRAY_API_LINALG_FUNCTIONS = frozenset(
    {
        "cholesky",
        "cross",
        "det",
        "diagonal",
        "eigh",
        "eigvalsh",
        "inv",
        "matmul",
        "matrix_norm",
        "matrix_power",
        "matrix_rank",
        "matrix_transpose",
        "outer",
        "pinv",
        "qr",
        "slogdet",
        "solve",
        "svd",
        "svdvals",
        "tensordot",
        "trace",
        "vecdot",
        "vector_norm",
    }
)

_SPARSE_ADAPTER_BACKENDS = frozenset({"numpy", "torch", "cupy", "jax"})
_AUTODIFF_BACKENDS = frozenset({"torch", "jax"})


def _namespace_name(xp: Namespace) -> str:
    """Canonical backend name from a namespace module's ``__name__``.

    The backend is identified by the *leading* package, not the trailing one:
    JAX resolves to the ``jax.numpy`` module (``__name__ == "jax.numpy"``), so a
    trailing-segment rule would mislabel it as ``"numpy"`` and skip its autodiff
    adapter. We strip an optional ``array_api_compat.`` wrapper prefix and take
    the first component: ``jax.numpy`` → ``jax``, ``array_api_compat.torch`` →
    ``torch``, ``array_api_strict`` → ``array_api_strict``.
    """
    module_name = getattr(xp, "__name__", "")
    prefix = "array_api_compat."
    if module_name.startswith(prefix):
        module_name = module_name[len(prefix) :]
    return module_name.split(".", 1)[0] or "unknown"


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What the current namespace/device supports (§5.4)."""

    name: str  # "numpy", "torch", "array_api_strict", ...
    has_linalg: bool
    linalg_functions: frozenset[str]
    has_sparse_adapter: bool
    supports_autodiff: bool
    devices: tuple[str, ...]
    default_float: str  # "float64" preferred; read from inputs in practice


def capabilities(xp: Namespace) -> Capabilities:
    """Probe ``xp`` for optional Array-API features and adapter availability.

    Records presence of ``xp.linalg`` and which functions exist, whether a
    sparse adapter is registered for this namespace, device info, and autodiff
    support. Missing standard pieces (triangular solve, ``lstsq``) are filled by
    labeled helpers elsewhere in ``backend``/``linalg``.
    """
    linalg = getattr(xp, "linalg", None)
    linalg_functions = frozenset(
        name
        for name in _ARRAY_API_LINALG_FUNCTIONS
        if linalg is not None and hasattr(linalg, name)
    )
    name = _namespace_name(xp)
    if hasattr(xp, "float64"):
        default_float = "float64"
    elif hasattr(xp, "float32"):
        default_float = "float32"
    else:
        default_float = "unknown"

    return Capabilities(
        name=name,
        has_linalg=linalg is not None,
        linalg_functions=linalg_functions,
        has_sparse_adapter=name in _SPARSE_ADAPTER_BACKENDS,
        supports_autodiff=name in _AUTODIFF_BACKENDS,
        devices=("cpu",),
        default_float=default_float,
    )


__all__ = ["Capabilities", "array_namespace", "capabilities"]
