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

"""CuPy sparse adapter (ADAPTER -- concrete-library import allowed here).

Builds ``cupyx.scipy.sparse`` matrices from Array-API COO triplets and solves
them on CUDA. The preferred direct route is NVIDIA cuDSS, accessed through the
official ``nvmath-python`` bindings (``nvmath.bindings.cudss``), which handle
library discovery and status checking and report inertia for symmetric/Hermitian
indefinite factorizations. If the bindings (or the cuDSS library they wrap) are
not installed, the adapter falls back to ``cupyx.scipy.sparse.linalg.spsolve``
for solve-only use.

``ipax`` deliberately does not choose a CUDA toolkit, CuPy wheel, or cuDSS
runtime for the user. The ``[sparse-cuda]`` extra installs only the
``nvmath-python`` bindings; see the installation documentation for the
user-managed CUDA dependencies.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any, cast

# Allowed import boundary (invariants #1, #4): backend/sparse/ adapters only.
import cupy
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg

from ipax.backend.operators import LinearOperator
from ipax.backend.sparse._routing import to_xp_array
from ipax.linalg.solver import LinearSolveError

if TYPE_CHECKING:
    from ipax.typing import Array, Namespace


# ``cudaDataType_t`` values from CUDA ``library_types.h``. These are ABI-stable
# CUDA enum values (not cuDSS-specific) and are what ``matrix_create_csr`` /
# ``matrix_create_dn`` expect for their ``value_type`` / ``index_type`` args.
_CUDSS_R_32F = 0
_CUDSS_R_64F = 1
_CUDSS_R_32I = 10
_CUDSS_R_64I = 24

# Cached nvmath cuDSS bindings module (``nvmath.bindings.cudss``).
_CUDSS_BINDINGS: Any | None = None


class _CuDSSUnavailableError(ImportError):
    """The nvmath binding or user-managed cuDSS runtime cannot be loaded."""


def _to_cupy(arr: Array) -> Any:
    """Return ``arr`` as a CuPy array, using DLPack where available."""
    if isinstance(arr, cupy.ndarray):
        return arr
    try:
        return cupy.from_dlpack(arr)
    except (TypeError, RuntimeError, BufferError, ValueError):
        return cupy.asarray(arr)


def _to_index(arr: Array) -> Any:
    """Device integer-index vector for COO/CSR assembly."""
    return _to_cupy(arr).astype(cupy.int64, copy=False)


def _to_float(value: Any) -> float:
    """Bring a scalar-like CuPy value to a Python ``float``."""
    return float(cupy.asnumpy(value))


def _ptr(arr: Any) -> int:
    """Device pointer (``intptr_t``) for a CuPy array; nvmath takes plain ints."""
    return int(arr.data.ptr)


def _value_type(dtype: Any) -> int:
    """cuDSS scalar type for real CuPy matrix values."""
    normalized = cupy.dtype(dtype)
    if normalized == cupy.dtype(cupy.float32):
        return _CUDSS_R_32F
    if normalized == cupy.dtype(cupy.float64):
        return _CUDSS_R_64F
    raise TypeError("cuDSS sparse solver supports only float32/float64 values")


def _index_type(dtype: Any) -> int:
    """cuDSS scalar type for integer CSR offsets/indices."""
    normalized = cupy.dtype(dtype)
    if normalized == cupy.dtype(cupy.int32):
        return _CUDSS_R_32I
    if normalized == cupy.dtype(cupy.int64):
        return _CUDSS_R_64I
    raise TypeError("cuDSS sparse solver supports only int32/int64 indices")


def _is_symmetric(matrix: Any) -> bool:
    """Return whether a CuPy sparse matrix is numerically symmetric."""
    if matrix.shape[0] != matrix.shape[1]:
        return False
    diff = (matrix - matrix.T).tocoo()
    diff.eliminate_zeros()
    if int(diff.nnz) == 0:
        return True
    matrix_scale = _to_float(cupy.max(cupy.abs(matrix.data))) if matrix.nnz else 0.0
    diff_scale = _to_float(cupy.max(cupy.abs(diff.data)))
    n = int(matrix.shape[0])
    tol = max(1.0, matrix_scale) * n * float(cupy.finfo(cupy.float64).eps)
    return diff_scale <= tol


def _canonical_csr(matrix: Any) -> Any:
    """CSR with summed duplicates and sorted column indices."""
    csr = matrix.tocsr()
    csr.sum_duplicates()
    csr.sort_indices()
    return csr


def _device_id(matrix: Any) -> int:
    """CUDA device containing a CuPy sparse matrix's value storage."""
    device = getattr(matrix.data, "device", None)
    if device is not None and hasattr(device, "id"):
        return int(device.id)
    # Test doubles and third-party CuPy-compatible arrays may not expose
    # ``ndarray.device``; their active CUDA device is the best available signal.
    return int(cupy.cuda.runtime.getDevice())


def _is_cudss_load_error(exc: BaseException) -> bool:
    """Return whether ``exc`` means the optional cuDSS runtime is unavailable."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ImportError, OSError)):
            return True
        cls = type(current)
        if cls.__name__ in {
            "DynamicLibNotFoundError",
            "LibraryNotFoundError",
        } and cls.__module__.startswith("cuda.pathfinder"):
            return True
        current = current.__cause__ or current.__context__
    return False


class SparseOperator(LinearOperator):
    """``LinearOperator`` backed by a ``cupyx.scipy.sparse`` CSR matrix.

    Unlike the SciPy adapter the CSR form is kept eager: cuDSS consumes CSR, which
    is also the matvec format, so it is not wasted on the direct-solve route. The
    only per-iteration trim here is caching the symmetry verdict (and honoring the
    assembler's structural hint) instead of recomputing ``A − Aᵀ`` every factor.
    """

    def __init__(
        self,
        matrix: Any,
        xp: Namespace,
        *,
        symmetric: bool | None = None,
        pattern_signature: object | None = None,
    ) -> None:
        self._matrix = _canonical_csr(matrix)
        self._xp = xp
        # Structural symmetry hint from the assembler (None ⇒ test numerically).
        self._symmetric_hint = symmetric
        self._pattern_signature = pattern_signature
        self._symmetric: bool | None = None

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self._matrix.shape
        return int(rows), int(cols)

    @property
    def cupy_matrix(self) -> Any:
        """The wrapped device CSR matrix (consumed by CUDA sparse solvers)."""
        return self._matrix

    def is_symmetric(self) -> bool:
        """Whether ``A == Aᵀ``: honor the assembler's hint, else test numerically.

        The structural hint (set by the KKT condensed/saddle assembler) is
        authoritative when present, sparing the per-iteration O(nnz) device-side
        ``A − Aᵀ`` test and its host syncs.
        """
        if self._symmetric_hint is not None:
            return self._symmetric_hint
        if self._symmetric is None:
            self._symmetric = _is_symmetric(self._matrix)
        return self._symmetric

    def coo_pattern_signature(self) -> object | None:
        return self._pattern_signature

    def matvec(self, v: Array) -> Array:
        out = self._matrix @ _to_cupy(v)
        return cast("Array", to_xp_array(out, self._xp))

    def rmatvec(self, v: Array) -> Array:
        out = self._matrix.T @ _to_cupy(v)
        return cast("Array", to_xp_array(out, self._xp))

    def matmat(self, V: Array) -> Array:
        out = self._matrix @ _to_cupy(V)
        return cast("Array", to_xp_array(out, self._xp))

    def diagonal(self, like: Array | None = None) -> Array:
        del like
        return cast("Array", to_xp_array(self._matrix.diagonal(), self._xp))

    def row_inf_norms(self, like: Array | None = None) -> Array:
        del like
        m, n = self.shape
        if n == 0:
            norms = cupy.zeros((m,), dtype=self._matrix.dtype)
        else:
            norms = abs(self._matrix).max(axis=1).toarray().reshape((m,))
        return cast("Array", to_xp_array(norms, self._xp))

    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        del like
        coo = self._matrix.tocoo()
        return (
            cast("Array", to_xp_array(coo.row, self._xp)),
            cast("Array", to_xp_array(coo.col, self._xp)),
            cast("Array", to_xp_array(coo.data, self._xp)),
            self.shape,
        )


def _require_csr(K: LinearOperator) -> tuple[Any, Namespace]:
    """Extract a square CuPy CSR matrix + namespace from ``SparseOperator``."""
    if not isinstance(K, SparseOperator):
        raise TypeError(
            "CuPy sparse solver requires a SparseOperator built from COO "
            f"triplets; got {type(K).__name__}"
        )
    matrix = K.cupy_matrix
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("sparse solver requires a square operator")
    return matrix, K._xp


def _load_cudss() -> Any:
    """Import the nvmath-python cuDSS bindings lazily.

    ``nvmath.bindings.cudss`` wraps NVIDIA cuDSS, dlopening the library and
    raising :class:`cuDSSError` on non-success statuses. Kept lazy (not a
    module-level import) so the CuPy adapter still degrades to the spsolve
    fallback when nvmath/cuDSS is absent.
    """
    global _CUDSS_BINDINGS
    if _CUDSS_BINDINGS is not None:
        return _CUDSS_BINDINGS
    try:
        from nvmath.bindings import cudss
    except ImportError as exc:
        raise ImportError(
            "NVIDIA cuDSS bindings are unavailable; after installing a "
            "CUDA-compatible CuPy package, install `ipax[sparse-cuda]` for "
            "the nvmath bindings"
        ) from exc
    _CUDSS_BINDINGS = cudss
    return cudss


class CuPySpsolveSolver:
    """Solve-only fallback backed by ``cupyx.scipy.sparse.linalg.spsolve``."""

    def __init__(self, *, require_inertia: bool = False) -> None:
        self._require_inertia = require_inertia
        self._matrix: Any | None = None
        self._xp: Namespace | None = None

    def describe(self) -> str:
        return "CuPy spsolve (GPU)"

    def factor(self, K: LinearOperator) -> None:
        if self._require_inertia:
            raise NotImplementedError(
                "GPU sparse inertia requires a user-managed NVIDIA cuDSS runtime "
                "compatible with the installed CUDA toolkit"
            )
        matrix, xp = _require_csr(K)
        self._matrix = matrix
        self._xp = xp

    def solve(self, rhs: Array) -> Array:
        if self._matrix is None or self._xp is None:
            raise RuntimeError("factor() must be called before solve()")
        try:
            x = cupyx.scipy.sparse.linalg.spsolve(self._matrix, _to_cupy(rhs))
        except RuntimeError as exc:
            raise LinearSolveError("CuPy sparse solve failed") from exc
        return cast("Array", to_xp_array(x, self._xp))

    @property
    def inertia(self) -> tuple[int, int, int]:
        """Inertia is unavailable from the ``spsolve`` fallback."""
        raise RuntimeError("inertia is unavailable without cuDSS")

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """The ``spsolve`` fallback is not inertia-revealing."""
        return None


class CuDSSSparseSolver:
    """Sparse CUDA direct solver backed by NVIDIA cuDSS via nvmath-python.

    cuDSS can solve general sparse systems. Inertia is available only for
    symmetric/Hermitian non-positive-definite matrix types, so this wrapper
    requests a symmetric lower-triangle LDL^T factorization whenever inertia is
    required.
    """

    def __init__(self, *, require_inertia: bool = False) -> None:
        self._require_inertia = require_inertia
        self._cudss: Any | None = None
        self._handle: int | None = None
        self._config: int | None = None
        self._data: int | None = None
        self._matrix: int | None = None
        self._row_offsets: Any | None = None
        self._col_indices: Any | None = None
        self._values: Any | None = None
        self._pattern: tuple[int, int, int, int, int, int, int] | None = None
        self._pattern_key: object | None = None
        self._xp: Namespace | None = None
        self._dtype: Any | None = None
        self._inertia: tuple[int, int, int] | None = None
        self._device_id: int | None = None

    def describe(self) -> str:
        return "cuDSS (GPU)"

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Release cuDSS resources owned by the solver."""
        cudss = self._cudss
        if cudss is None:
            return
        device_id = self._device_id

        def release() -> None:
            self._destroy_matrix()
            if self._data is not None and self._handle is not None:
                cudss.data_destroy(self._handle, self._data)
                self._data = None
            if self._config is not None:
                cudss.config_destroy(self._config)
                self._config = None
            if self._handle is not None:
                cudss.destroy(self._handle)
                self._handle = None

        try:
            if device_id is None:
                release()
            else:
                with cupy.cuda.Device(device_id):
                    release()
        finally:
            self._cudss = None
            self._device_id = None
            self._xp = None
            self._dtype = None
            self._inertia = None

    def factor(self, K: LinearOperator) -> None:
        matrix, xp = _require_csr(K)
        assert isinstance(K, SparseOperator)  # narrowed by _require_csr
        device_id = _device_id(matrix)
        with cupy.cuda.Device(device_id):
            symmetric = K.is_symmetric()
            if self._require_inertia and not symmetric:
                raise ValueError("cuDSS inertia is defined only for symmetric matrices")

            self._ensure_context(device_id)
            self._bind_current_stream()
            cudss = self._cudss
            assert cudss is not None

            factor_matrix = matrix
            matrix_type = int(cudss.MatrixType.GENERAL)
            matrix_view = int(cudss.MatrixViewType.FULL)
            if symmetric:
                factor_matrix = _canonical_csr(cupyx.scipy.sparse.tril(matrix))
                matrix_type = int(cudss.MatrixType.SYMMETRIC)
                matrix_view = int(cudss.MatrixViewType.LOWER)

            self._xp = xp
            pattern_signature = K.coo_pattern_signature()
            self._factor_csr(
                factor_matrix,
                matrix_type,
                matrix_view,
                pattern_signature=pattern_signature,
            )
            self._inertia = (
                self._read_inertia(int(matrix.shape[0]))
                if self._require_inertia
                else None
            )

    def solve(self, rhs: Array) -> Array:
        if (
            self._cudss is None
            or self._handle is None
            or self._config is None
            or self._data is None
            or self._matrix is None
            or self._xp is None
            or self._dtype is None
            or self._device_id is None
        ):
            raise RuntimeError("factor() must be called before solve()")
        with cupy.cuda.Device(self._device_id):
            self._bind_current_stream()
            cudss = self._cudss

            b = cupy.asarray(_to_cupy(rhs), dtype=self._dtype)
            was_vector = len(b.shape) == 1
            if was_vector:
                b = cupy.reshape(b, (int(b.shape[0]), 1))
            b = cupy.ascontiguousarray(b)
            x = cupy.empty_like(b)

            rhs_matrix = self._create_dense_matrix(x.shape[0], x.shape[1], b)
            sol_matrix = self._create_dense_matrix(x.shape[0], x.shape[1], x)
            try:
                self._guard(
                    cudss.execute,
                    self._handle,
                    int(cudss.Phase.SOLVE),
                    self._config,
                    self._data,
                    self._matrix,
                    sol_matrix,
                    rhs_matrix,
                )
            finally:
                self._destroy_temp_matrix(sol_matrix)
                self._destroy_temp_matrix(rhs_matrix)

            result = x[:, 0] if was_vector else x
            return cast("Array", to_xp_array(result, self._xp))

    @property
    def inertia(self) -> tuple[int, int, int]:
        """The factored operator's inertia ``(n_pos, n_neg, n_zero)``."""
        if self._inertia is None:
            raise RuntimeError(
                "inertia is unavailable: create the solver with "
                "require_inertia=True and call factor() first"
            )
        return self._inertia

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """Best-effort inertia; only populated under ``require_inertia``."""
        return self._inertia

    def _guard(self, fn: Any, *args: Any) -> Any:
        """Call an nvmath cuDSS function, mapping ``cuDSSError`` failures."""
        assert self._cudss is not None
        try:
            return fn(*args)
        except self._cudss.cuDSSError as exc:
            raise LinearSolveError(f"cuDSS {fn.__name__} failed: {exc}") from exc

    def _ensure_context(self, device_id: int) -> None:
        if self._cudss is not None and self._device_id == device_id:
            return
        if self._cudss is not None:
            self.close()
        cudss = _load_cudss()
        self._cudss = cudss
        self._device_id = device_id
        try:
            self._handle = self._guard(cudss.create)
            self._config = self._guard(cudss.config_create)
            self._data = self._guard(cudss.data_create, self._handle)
        except Exception as exc:
            self.close()
            if _is_cudss_load_error(exc):
                raise _CuDSSUnavailableError(
                    "the user-managed NVIDIA cuDSS runtime could not be loaded"
                ) from exc
            raise

    def _bind_current_stream(self) -> None:
        """Run cuDSS on CuPy's current stream for correct dependency ordering."""
        if self._cudss is None or self._handle is None:
            raise RuntimeError("cuDSS context was not initialized")
        stream = cupy.cuda.get_current_stream()
        self._guard(self._cudss.set_stream, self._handle, int(stream.ptr))

    def _factor_csr(
        self,
        matrix: Any,
        matrix_type: int,
        matrix_view: int,
        *,
        pattern_signature: object | None,
    ) -> None:
        cudss = self._cudss
        if (
            cudss is None
            or self._handle is None
            or self._config is None
            or self._data is None
        ):
            raise RuntimeError("cuDSS context was not initialized")

        values = cupy.ascontiguousarray(matrix.data)
        index_type = _index_type(cupy.dtype(cupy.int64))
        pattern = (
            int(matrix.shape[0]),
            int(matrix.shape[1]),
            int(matrix.nnz),
            matrix_type,
            matrix_view,
            index_type,
            _value_type(values.dtype),
        )

        self._dtype = values.dtype
        structure_key = (
            pattern_signature,
            matrix_type,
            matrix_view,
        )
        same_pattern = (
            pattern_signature is not None
            and pattern == self._pattern
            and structure_key == self._pattern_key
            and self._matrix is not None
            and self._row_offsets is not None
            and self._col_indices is not None
        )
        if same_pattern:
            # Deliberately no ``array_equal`` on CSR indices here: on CUDA that
            # comparison is device work plus a host sync. A non-None signature is
            # a contract that row/column coordinates are structural; value-
            # dependent sparse operators must return ``None`` and reanalyze.
            self._values = values
            self._guard(cudss.matrix_set_values, self._matrix, _ptr(values))
            phase = int(cudss.Phase.FACTORIZATION)
        else:
            row_offsets = cupy.ascontiguousarray(
                matrix.indptr.astype(cupy.int64, copy=False)
            )
            indices = cupy.ascontiguousarray(
                matrix.indices.astype(cupy.int64, copy=False)
            )
            self._destroy_matrix()
            self._row_offsets = row_offsets
            self._col_indices = indices
            self._values = values
            self._pattern = pattern
            self._pattern_key = structure_key if pattern_signature is not None else None
            self._matrix = self._create_csr_matrix(
                matrix,
                row_offsets,
                indices,
                values,
                matrix_type,
                matrix_view,
            )
            phase = int(cudss.Phase.ANALYSIS) | int(cudss.Phase.FACTORIZATION)

        n = int(matrix.shape[0])
        dummy = cupy.zeros((n, 1), dtype=values.dtype)
        rhs_matrix = self._create_dense_matrix(n, 1, dummy)
        sol_matrix = self._create_dense_matrix(n, 1, dummy)
        try:
            self._guard(
                cudss.execute,
                self._handle,
                phase,
                self._config,
                self._data,
                self._matrix,
                sol_matrix,
                rhs_matrix,
            )
        finally:
            self._destroy_temp_matrix(sol_matrix)
            self._destroy_temp_matrix(rhs_matrix)

    def _create_csr_matrix(
        self,
        matrix: Any,
        row_offsets: Any,
        indices: Any,
        values: Any,
        matrix_type: int,
        matrix_view: int,
    ) -> int:
        cudss = self._cudss
        if cudss is None:
            raise RuntimeError("cuDSS context was not initialized")
        # cuDSS shares one index type for row offsets and column indices; our
        # COO assembly keeps both as int64 (see ``_factor_csr``).
        return cast(
            int,
            self._guard(
                cudss.matrix_create_csr,
                int(matrix.shape[0]),
                int(matrix.shape[1]),
                int(matrix.nnz),
                _ptr(row_offsets),
                0,
                _ptr(indices),
                _ptr(values),
                _index_type(indices.dtype),
                _value_type(values.dtype),
                matrix_type,
                matrix_view,
                int(cudss.IndexBase.ZERO),
            ),
        )

    def _create_dense_matrix(self, nrows: int, ncols: int, values: Any) -> int:
        cudss = self._cudss
        if cudss is None:
            raise RuntimeError("cuDSS context was not initialized")
        return cast(
            int,
            self._guard(
                cudss.matrix_create_dn,
                int(nrows),
                int(ncols),
                int(ncols),
                _ptr(values),
                _value_type(values.dtype),
                int(cudss.Layout.ROW_MAJOR),
            ),
        )

    def _read_inertia(self, n: int) -> tuple[int, int, int]:
        cudss = self._cudss
        if cudss is None or self._handle is None or self._data is None:
            raise RuntimeError("cuDSS context was not initialized")
        counts = (ctypes.c_int64 * 2)()
        written = ctypes.c_size_t()
        self._guard(
            cudss.data_get,
            self._handle,
            self._data,
            int(cudss.DataParam.INERTIA),
            ctypes.addressof(counts),
            ctypes.sizeof(counts),
            ctypes.addressof(written),
        )
        pos = int(counts[0])
        neg = int(counts[1])
        zero = n - pos - neg
        if pos < 0 or neg < 0 or zero < 0:
            raise LinearSolveError("cuDSS returned an implausible inertia")
        return pos, neg, zero

    def _destroy_matrix(self) -> None:
        if self._cudss is not None and self._matrix is not None:
            self._cudss.matrix_destroy(self._matrix)
        self._matrix = None
        self._row_offsets = None
        self._col_indices = None
        self._values = None
        self._pattern = None
        self._pattern_key = None

    def _destroy_temp_matrix(self, matrix: int) -> None:
        if self._cudss is not None and matrix:
            self._cudss.matrix_destroy(matrix)


class CuPySparseSolver:
    """CUDA sparse-direct solver preferring cuDSS and falling back to spsolve."""

    def __init__(
        self, *, require_inertia: bool = False, prefer_cudss: bool = True
    ) -> None:
        self._require_inertia = require_inertia
        self._prefer_cudss = prefer_cudss
        self._inner: CuDSSSparseSolver | CuPySpsolveSolver | None = None
        self._cudss_unavailable = False

    def describe(self) -> str:
        return self._inner.describe() if self._inner is not None else "CuPy (GPU)"

    def factor(self, K: LinearOperator) -> None:
        if self._prefer_cudss and not self._cudss_unavailable:
            if not isinstance(self._inner, CuDSSSparseSolver):
                self._inner = CuDSSSparseSolver(require_inertia=self._require_inertia)
            try:
                self._inner.factor(K)
            except ImportError as exc:
                self._cudss_unavailable = True
                self._inner = None
                if self._require_inertia:
                    raise NotImplementedError(
                        "GPU sparse inertia requires a user-managed NVIDIA cuDSS "
                        "runtime compatible with the installed CUDA toolkit"
                    ) from exc
            else:
                return

        if not isinstance(self._inner, CuPySpsolveSolver):
            self._inner = CuPySpsolveSolver(require_inertia=self._require_inertia)
        self._inner.factor(K)

    def solve(self, rhs: Array) -> Array:
        if self._inner is None:
            raise RuntimeError("factor() must be called before solve()")
        return self._inner.solve(rhs)

    @property
    def inertia(self) -> tuple[int, int, int]:
        """The factored operator's inertia ``(n_pos, n_neg, n_zero)``."""
        if self._inner is None:
            raise RuntimeError("factor() must be called before reading inertia")
        return self._inner.inertia

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """Best-effort inertia of the dispatched inner solver; ``None`` if absent."""
        if self._inner is None:
            return None
        return self._inner.inertia_or_none()


class CuPySparseAdapter:
    """Factory pairing CuPy COO assembly with CUDA sparse-direct solvers."""

    def from_coo(
        self,
        rows: Array,
        cols: Array,
        values: Array,
        *,
        shape: tuple[int, int],
        symmetric: bool | None = None,
        pattern_signature: object | None = None,
    ) -> SparseOperator:
        """Build a :class:`SparseOperator` from Array-API COO triplets.

        ``symmetric`` is an optional structural hint from the assembler; ``None``
        leaves the operator to test symmetry numerically when first asked.
        ``pattern_signature`` is a conservative stable-pattern key used by cuDSS
        to reuse symbolic analysis without comparing device index arrays.
        """
        from ipax.backend.namespace import array_namespace

        xp = array_namespace(values)
        matrix = cupyx.scipy.sparse.coo_matrix(
            (_to_cupy(values), (_to_index(rows), _to_index(cols))),
            shape=shape,
        )
        return SparseOperator(
            matrix,
            xp,
            symmetric=symmetric,
            pattern_signature=pattern_signature,
        )

    def solver(self, *, require_inertia: bool = False) -> CuPySparseSolver:
        """Return the CUDA sparse-direct solver (cuDSS default, spsolve fallback)."""
        return CuPySparseSolver(require_inertia=require_inertia)


__all__ = [
    "CuDSSSparseSolver",
    "CuPySparseAdapter",
    "CuPySparseSolver",
    "CuPySpsolveSolver",
    "SparseOperator",
]
