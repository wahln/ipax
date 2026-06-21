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

"""JAX sparse adapter (device-routed; no native factorization).

JAX's ``jax.experimental.sparse`` provides no general sparse-direct factorization
ipax can depend on, so the JAX backend *reinterprets* its COO buffers onto the
existing SciPy (CPU) and CuPy/cuDSS (CUDA) adapters via DLPack — see
:mod:`ipax.backend.sparse._routing`. A JAX-CPU triplet is the same host buffer
SciPy factors; a JAX-GPU triplet is the same CUDA buffer cuDSS factors. Dispatch
(``get_sparse_adapter``) selects this adapter for the JAX namespace; the device
decision happens inside ``from_coo``.

This module imports **no** concrete array library: the routing adapter inspects
``__dlpack_device__`` on the value array and defers the SciPy/CuPy import to the
chosen device path, so importing it never requires JAX to be installed.
"""

from __future__ import annotations

# Allowed import boundary (invariants #1, #4): backend/sparse/ adapters. JAX
# itself is never imported here — see the module docstring.
from ipax.backend.sparse._routing import DeviceRoutingSparseAdapter

# The JAX sparse adapter is the generic DLPack device-routing adapter.
JaxSparseAdapter = DeviceRoutingSparseAdapter

__all__ = ["JaxSparseAdapter"]
