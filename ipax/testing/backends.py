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

"""Backend discovery helpers for tests and benchmarks."""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipax.typing import Namespace

DEFAULT_BACKENDS = ("numpy", "torch", "array_api_strict")

_NAMESPACE_MODULES = {
    "numpy": "array_api_compat.numpy",
    "torch": "array_api_compat.torch",
    "array_api_strict": "array_api_strict",
    "strict": "array_api_strict",
    "jax": "array_api_compat.jax",
    "cupy": "array_api_compat.cupy",
}


def requested_backends() -> tuple[str, ...]:
    """Backend names from ``IPAX_BACKENDS`` or the default CI pair."""
    raw = os.environ.get("IPAX_BACKENDS")
    if not raw:
        return DEFAULT_BACKENDS
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def import_namespace(name: str) -> Namespace:
    """Import and return the Array-API namespace module for ``name``."""
    try:
        module_name = _NAMESPACE_MODULES[name]
    except KeyError as exc:
        known = ", ".join(sorted(_NAMESPACE_MODULES))
        raise ValueError(f"unknown backend {name!r}; expected one of: {known}") from exc

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(f"backend {name!r} is not installed") from exc


__all__ = ["DEFAULT_BACKENDS", "import_namespace", "requested_backends"]
