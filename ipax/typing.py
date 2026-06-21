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

"""Shared type aliases.

Array API backends are intentionally duck-typed: NumPy, PyTorch, JAX, CuPy, and
``array-api-strict`` expose similar runtime surfaces but do not share a useful
static base class. Until the package grows a full local protocol for the subset
we use, these aliases keep public signatures readable while letting mypy type
check the solver's own control flow and dataclasses.
"""

from __future__ import annotations

from typing import Any, TypeAlias

Array: TypeAlias = Any
"""An Array-API array object from any supported backend."""

Scalar: TypeAlias = Any
"""A scalar objective value: usually a 0-d array or a Python float."""

Namespace: TypeAlias = Any
"""An Array-API namespace module such as ``array_api_compat.numpy``."""

__all__ = ["Array", "Namespace", "Scalar"]
