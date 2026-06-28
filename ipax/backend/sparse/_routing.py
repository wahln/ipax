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

"""Device-routing sparse adapter and namespace round-trip helper.

This module is intentionally **library-free** (no concrete array import): it
routes a backend that has no native sparse-direct path of its own — Torch, JAX —
onto the existing SciPy (CPU) and CuPy/cuDSS (CUDA) adapters by *reinterpreting
the COO buffer* rather than rebuilding a native sparse factorization.

The key observation is that the SciPy and CuPy adapters are already
namespace-agnostic on their input side (``_to_numpy``/``_to_cupy`` ingest any
array exposing DLPack), and capture the original namespace to return through it.
So a Torch-CPU or JAX-CPU triplet is the *same host buffer* SciPy consumes, and a
Torch-CUDA or JAX-GPU triplet is the *same device buffer* cuDSS consumes — we
just need to dispatch by **DLPack device type** instead of by library name. This
keeps "sparsity is an adapter concern" (invariant #4) honest without adding a
fourth and fifth concrete factorization backend.

Routing is by ``__dlpack_device__()`` (Array-API standard): device kind ``1``
(CPU) → SciPy adapter, kind ``2`` (CUDA) → CuPy/cuDSS adapter. Other device kinds
(ROCm, Metal, …) have no direct solver and raise a clear error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ipax.typing import Array, Namespace

# DLPack ``DLDeviceType`` values (stable across the standard); the enum members
# Torch/JAX return compare equal to these ints once coerced with ``int()``.
_DLPACK_CPU = 1
_DLPACK_CUDA = 2


def to_xp_array(out: Any, xp: Namespace) -> Any:
    """Return ``out`` in namespace ``xp``, preferring a zero-copy DLPack import.

    The SciPy/CuPy adapters compute results as NumPy/CuPy arrays and must hand
    them back in the *caller's* namespace (Torch, JAX, …). ``xp.from_dlpack``
    reinterprets the existing buffer in place — crucially this preserves the
    CUDA device on the GPU round-trip (Torch-CUDA ← CuPy), which a plain
    ``xp.asarray`` would not. We fall back to ``asarray`` for any array that does
    not expose a consumable DLPack capsule.
    """
    try:
        return xp.from_dlpack(out)
    except (TypeError, RuntimeError, BufferError, ValueError, NotImplementedError):
        return xp.asarray(out)


def _dlpack_device_kind(arr: Array) -> int | None:
    """DLPack device *kind* for ``arr`` (1=CPU, 2=CUDA), or ``None``."""
    dlpack_device = getattr(arr, "__dlpack_device__", None)
    if dlpack_device is None:
        return None
    try:
        kind, _ = dlpack_device()
    except Exception:
        return None
    return int(kind)


def _cpu_adapter() -> Any:
    """The SciPy CPU sparse adapter, with a clear error if SciPy is missing."""
    try:
        from ipax.backend.sparse.numpy_scipy import SciPySparseAdapter
    except ImportError as exc:
        raise RuntimeError(
            "the CPU sparse-direct route needs SciPy; install `ipax[sparse-cpu]`"
        ) from exc
    return SciPySparseAdapter()


def _cuda_adapter() -> Any:
    """The CuPy/cuDSS CUDA sparse adapter, with a clear error if CuPy is missing."""
    try:
        from ipax.backend.sparse.cupy import CuPySparseAdapter
    except ImportError as exc:
        raise RuntimeError(
            "the CUDA sparse-direct route needs CuPy; install a CUDA-compatible "
            "CuPy package and `ipax[sparse-cuda]`"
        ) from exc
    return CuPySparseAdapter()


class DeviceRoutingSparseAdapter:
    """Reinterpret COO triplets onto the SciPy/cuDSS adapters by DLPack device.

    Used for backends (Torch, JAX) whose arrays carry the same buffers the
    CPU/CUDA adapters already consume. The device decision is made in
    :meth:`from_coo`, where the value array is in hand, and remembered so that
    :meth:`solver` returns the matching backend solver. ``SparseDirectSolver``
    always calls ``from_coo`` before ``solver`` within a single ``factor`` call,
    and the device is fixed across an interior-point solve, so the pairing is
    consistent.
    """

    def __init__(self) -> None:
        self._delegate: Any = None

    def _resolve(self, like: Array) -> Any:
        kind = _dlpack_device_kind(like)
        if kind == _DLPACK_CPU:
            return _cpu_adapter()
        if kind == _DLPACK_CUDA:
            return _cuda_adapter()
        raise RuntimeError(
            f"no sparse-direct adapter for DLPack device kind {kind}; ipax "
            "reinterprets COO buffers onto the SciPy (CPU) and cuDSS (CUDA) "
            "adapters, which covers host and NVIDIA CUDA arrays only"
        )

    def from_coo(
        self,
        rows: Array,
        cols: Array,
        values: Array,
        *,
        shape: tuple[int, int],
        symmetric: bool | None = None,
        pattern_signature: object | None = None,
    ) -> Any:
        """Build a ``SparseOperator`` via the device-appropriate adapter."""
        self._delegate = self._resolve(values)
        return self._delegate.from_coo(
            rows,
            cols,
            values,
            shape=shape,
            symmetric=symmetric,
            pattern_signature=pattern_signature,
        )

    def solver(self, *, require_inertia: bool = False) -> Any:
        """Return the resolved backend's sparse-direct solver."""
        if self._delegate is None:
            raise RuntimeError("from_coo() must be called before solver()")
        return self._delegate.solver(require_inertia=require_inertia)


__all__ = ["DeviceRoutingSparseAdapter", "to_xp_array"]
