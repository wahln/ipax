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

"""CuPy Bunch-Kaufman LDLT dense adapter via NVIDIA cuSOLVER (nvmath-python).

Wraps ``cusolverDnDsytrf``/``cusolverDnXsytrs`` through the official
``nvmath-python`` bindings (``nvmath.bindings.cusolverDn``), the same
lazily-imported, user-managed-CUDA convention ``ipax/backend/sparse/cupy.py``
uses for cuDSS. ``dsytrf`` is the classic (32-bit-indexed) per-dtype
factorization, writing pivots as ``int32``; ``xsytrs`` is the newer generic
64-bit solve routine and expects ``int64`` pivots — the mismatch between the
two generations of the cuSOLVER Dense API is real (verified directly against
the installed bindings' Cython declarations and a working GPU probe) and is
bridged here with one explicit ``int32 -> int64`` cast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Allowed import boundary (invariants #1, #4): backend/dense/ adapters only.
import cupy
import numpy as np

from ipax.linalg.solver import LinearSolveError

if TYPE_CHECKING:
    from ipax.typing import Array

# Cached nvmath cusolverDn bindings module + the cublas FillMode/CudaDataType
# enums it depends on (loaded lazily so importing this module never requires
# nvmath to be installed unless the augmented dense route is actually used).
_CUSOLVER: Any | None = None
_FILL_MODE_LOWER: int | None = None
_DATA_TYPE_R64F: int | None = None


def _load_cusolver() -> Any:
    """Import the nvmath-python cuSOLVER dense bindings lazily."""
    global _CUSOLVER, _FILL_MODE_LOWER, _DATA_TYPE_R64F
    if _CUSOLVER is not None:
        return _CUSOLVER
    try:
        import nvmath
        from nvmath.bindings import cublas, cusolverDn
    except ImportError as exc:
        raise ImportError(
            "the NVIDIA cuSOLVER dense bindings are unavailable; after "
            "installing a CUDA-compatible CuPy package, install "
            "`ipax[sparse-cuda]` for the nvmath bindings"
        ) from exc
    _CUSOLVER = cusolverDn
    _FILL_MODE_LOWER = int(cublas.FillMode.LOWER)
    _DATA_TYPE_R64F = int(nvmath.CudaDataType.CUDA_R_64F)
    return cusolverDn


def _to_cupy(arr: Any) -> Any:
    """Return ``arr`` as a CuPy array, using DLPack where available."""
    if isinstance(arr, cupy.ndarray):
        return arr
    try:
        return cupy.from_dlpack(arr)
    except (TypeError, RuntimeError, BufferError, ValueError):
        return cupy.asarray(arr)


def _ptr(arr: Any) -> int:
    """Device pointer (``intptr_t``) for a CuPy array; nvmath takes plain ints."""
    return int(arr.data.ptr)


def _device_id(arr: Any) -> int:
    """CUDA device containing a CuPy array's storage."""
    device = getattr(arr, "device", None)
    if device is not None and hasattr(device, "id"):
        return int(device.id)
    return int(cupy.cuda.runtime.getDevice())


def _ldl_blocks_from_pivots(pivots_host: list[int]) -> list[tuple[int, int]]:
    """Identify the ``(start, size)`` of each 1x1/2x2 block from LAPACK ``ipiv``.

    A 2x2 Bunch-Kaufman pivot is signalled by two consecutive equal *negative*
    entries in the pivot array (the standard ``?sytrf`` ``ipiv`` convention;
    verified identical on cuSOLVER's ``dsytrf`` and LAPACK's via a GPU probe).
    """
    n = len(pivots_host)
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if pivots_host[i] < 0:
            blocks.append((i, 2))
            i += 2
        else:
            blocks.append((i, 1))
            i += 1
    return blocks


def _sign_count_blocks(
    diag_host: np.ndarray, subdiag_host: np.ndarray, blocks: list[tuple[int, int]]
) -> tuple[int, int, int]:
    """Sign-count the packed factor's diagonal/subdiagonal blocks.

    Mirrors ``ipax.backend.dense.numpy_scipy``/``torch``'s tolerance
    convention. A genuine 2x2 Bunch-Kaufman pivot always contributes exactly
    one positive and one negative eigenvalue (verified via the block's own
    determinant, not assumed).
    """
    n = len(diag_host)
    scale = float(np.max(np.abs(diag_host))) if n else 0.0
    eps = float(np.finfo(np.float64).eps)
    tol = max(1.0, scale) * n * eps
    pos = neg = zero = 0
    for start, size in blocks:
        if size == 1:
            value = float(diag_host[start])
            if value > tol:
                pos += 1
            elif value < -tol:
                neg += 1
            else:
                zero += 1
        else:
            a = float(diag_host[start])
            b = float(subdiag_host[start])
            c = float(diag_host[start + 1])
            det = a * c - b * b
            if det < -tol:
                pos += 1
                neg += 1
            else:
                # Should never happen for a valid Bunch-Kaufman block; treat
                # defensively as numerically singular rather than guessing.
                zero += 2
    return pos, neg, zero


class CuPyLDLFactorization:
    """Bunch-Kaufman LDLᵀ factorization of a dense symmetric matrix on CUDA.

    Keeps one cuSOLVER handle for the instance's lifetime (created lazily on
    first :meth:`factor`, reused across repeated factorizations — a fresh
    handle per Newton iteration would be wasteful GPU context churn) and
    releases it in :meth:`close`/``__del__``. ``factor`` overwrites a fresh
    device copy of the input in place with the packed ``LD`` factor;
    ``solve`` may be called repeatedly against the same factorization.
    """

    def __init__(self) -> None:
        self._cusolver: Any | None = None
        self._handle: int | None = None
        self._device_id: int | None = None
        self._n = 0
        self._a: Any | None = None  # packed LD, device buffer
        self._ipiv64: Any | None = None
        self._inertia: tuple[int, int, int] | None = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Release the cuSOLVER handle owned by this instance."""
        if self._cusolver is None or self._handle is None:
            self._cusolver = None
            self._handle = None
            return
        cusolver = self._cusolver
        handle = self._handle
        device_id = self._device_id
        try:
            if device_id is None:
                cusolver.destroy(handle)
            else:
                with cupy.cuda.Device(device_id):
                    cusolver.destroy(handle)
        finally:
            self._cusolver = None
            self._handle = None
            self._device_id = None
            self._a = None
            self._ipiv64 = None
            self._inertia = None

    def _guard(self, fn: Any, *args: Any) -> Any:
        """Call an nvmath cuSOLVER function, mapping failures to LinearSolveError."""
        try:
            return fn(*args)
        except Exception as exc:
            raise LinearSolveError(f"cuSOLVER {fn.__name__} failed") from exc

    def _ensure_handle(self, device_id: int) -> Any:
        if self._cusolver is not None and self._device_id == device_id:
            return self._cusolver
        if self._cusolver is not None:
            self.close()
        cusolver = _load_cusolver()
        self._cusolver = cusolver
        self._device_id = device_id
        self._handle = self._guard(cusolver.create)
        return cusolver

    def factor(self, matrix: Array) -> None:
        cupy_matrix = _to_cupy(matrix)
        device_id = _device_id(cupy_matrix)
        with cupy.cuda.Device(device_id):
            cusolver = self._ensure_handle(device_id)
            handle = self._handle
            n = int(cupy_matrix.shape[0])
            uplo = _FILL_MODE_LOWER
            assert handle is not None and uplo is not None

            # dsytrf overwrites A in place: work on a fresh contiguous copy.
            # Symmetric ⇒ a C-contiguous (row-major) buffer is bit-identical
            # to the column-major (Fortran) layout cuSOLVER expects, so no
            # explicit transpose is needed.
            a = cupy.ascontiguousarray(cupy_matrix.astype(cupy.float64, copy=True))
            lwork = self._guard(cusolver.dsytrf_buffer_size, handle, n, _ptr(a), n)
            work = cupy.empty((max(int(lwork), 1),), dtype=cupy.float64)
            ipiv32 = cupy.empty((n,), dtype=cupy.int32)
            info = cupy.zeros((1,), dtype=cupy.int32)
            self._guard(
                cusolver.dsytrf,
                handle,
                uplo,
                n,
                _ptr(a),
                n,
                _ptr(ipiv32),
                _ptr(work),
                int(lwork),
                _ptr(info),
            )
            info_value = int(cupy.asnumpy(info)[0])
            if info_value != 0:
                raise LinearSolveError(
                    f"dense LDL^T factorization found a zero pivot (info={info_value})"
                )

            self._n = n
            self._a = a
            self._ipiv64 = ipiv32.astype(cupy.int64)
            self._inertia = self._compute_inertia(a, ipiv32, n)

    def _compute_inertia(self, a: Any, ipiv32: Any, n: int) -> tuple[int, int, int]:
        pivots_host = cupy.asnumpy(ipiv32).tolist()
        blocks = _ldl_blocks_from_pivots(pivots_host)
        diag_host = cupy.asnumpy(cupy.diagonal(a))
        sub_host = cupy.asnumpy(cupy.diagonal(a, -1)) if n > 1 else np.empty((0,))
        return _sign_count_blocks(diag_host, sub_host, blocks)

    def solve(self, rhs: Array) -> Any:
        if self._a is None or self._ipiv64 is None or self._handle is None:
            raise RuntimeError("factor() must be called before solve()")
        cusolver = self._cusolver
        assert cusolver is not None
        with cupy.cuda.Device(self._device_id):
            b = _to_cupy(rhs)
            vector_rhs = len(b.shape) == 1
            n = self._n
            # xsytrs expects column-major (Fortran) B with leading dimension
            # n, unlike A (safe as C-contiguous only because A is symmetric —
            # B in general is not, so this one genuinely needs a real
            # transpose-in-storage, not just a pointer reinterpretation.
            b2d = cupy.asfortranarray(
                (cupy.reshape(b, (n, 1)) if vector_rhs else b).astype(
                    cupy.float64, copy=True
                )
            )
            nrhs = int(b2d.shape[1])
            uplo = _FILL_MODE_LOWER
            dt = _DATA_TYPE_R64F
            assert uplo is not None and dt is not None

            dev_bytes, host_bytes = self._guard(
                cusolver.xsytrs_buffer_size,
                self._handle,
                uplo,
                n,
                nrhs,
                dt,
                _ptr(self._a),
                n,
                _ptr(self._ipiv64),
                dt,
                _ptr(b2d),
                n,
            )
            dev_buf = cupy.empty((max(int(dev_bytes), 1),), dtype=cupy.uint8)
            host_buf = np.empty((max(int(host_bytes), 1),), dtype=np.uint8)
            info = cupy.zeros((1,), dtype=cupy.int32)
            self._guard(
                cusolver.xsytrs,
                self._handle,
                uplo,
                n,
                nrhs,
                dt,
                _ptr(self._a),
                n,
                _ptr(self._ipiv64),
                dt,
                _ptr(b2d),
                n,
                _ptr(dev_buf),
                int(dev_bytes),
                host_buf.ctypes.data,
                int(host_bytes),
                _ptr(info),
            )
            info_value = int(cupy.asnumpy(info)[0])
            if info_value != 0:
                raise LinearSolveError(f"dense LDL^T solve failed (info={info_value})")
            return b2d.reshape(-1) if vector_rhs else b2d

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """Inertia ``(n_pos, n_neg, n_zero)`` of the factored matrix, or ``None``."""
        return self._inertia


__all__ = ["CuPyLDLFactorization"]
