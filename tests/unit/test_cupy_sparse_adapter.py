"""Unit coverage for the CuPy sparse adapter without requiring a CUDA device."""

from __future__ import annotations

import ctypes
import enum
import importlib
import sys
import types
from typing import Any, ClassVar

import numpy as np
import pytest

scipy_sparse = pytest.importorskip("scipy.sparse")
scipy_sparse_linalg = pytest.importorskip("scipy.sparse.linalg")


@pytest.fixture
def cupy_sparse_module(monkeypatch: pytest.MonkeyPatch):
    """Import the CuPy adapter against fake CuPy modules backed by NumPy/SciPy."""
    sys.modules.pop("ipax.backend.sparse.cupy", None)

    cupy = types.ModuleType("cupy")
    cupy.__name__ = "cupy"
    cupy.ndarray = np.ndarray
    cupy.float32 = np.float32
    cupy.float64 = np.float64
    cupy.int32 = np.int32
    cupy.int64 = np.int64
    cupy.asarray = np.asarray
    cupy.ascontiguousarray = np.ascontiguousarray
    cupy.asnumpy = np.asarray
    cupy.abs = np.abs
    cupy.max = np.max
    cupy.finfo = np.finfo
    cupy.dtype = np.dtype
    cupy.reshape = np.reshape
    cupy.zeros = np.zeros
    cupy.empty_like = np.empty_like
    cupy.array_equal = np.array_equal
    # Array ops used by the compiled COO→canonical-CSR fast path.
    cupy.unique = np.unique
    cupy.bincount = np.bincount
    cupy.cumsum = np.cumsum
    cupy.arange = np.arange
    cupy.diff = np.diff
    cupy.repeat = np.repeat

    class FakeRuntime:
        current_device = 0

        @classmethod
        def getDevice(cls) -> int:
            return cls.current_device

    class FakeDevice:
        entered: ClassVar[list[int]] = []

        def __init__(self, device_id: int) -> None:
            self.id = int(device_id)

        def __enter__(self) -> FakeDevice:
            self.entered.append(self.id)
            return self

        def __exit__(self, *args: object) -> None:
            del args

    cupy.cuda = types.SimpleNamespace(
        Device=FakeDevice,
        get_current_stream=lambda: types.SimpleNamespace(ptr=1234),
        runtime=FakeRuntime,
    )

    def from_dlpack(arr: object) -> np.ndarray:
        return np.from_dlpack(arr)

    cupy.from_dlpack = from_dlpack

    cupyx = types.ModuleType("cupyx")
    cupyx.scatter_add = lambda out, idx, vals: np.add.at(out, idx, vals)
    cupyx_scipy = types.ModuleType("cupyx.scipy")
    cupyx_sparse = types.ModuleType("cupyx.scipy.sparse")
    cupyx_sparse.coo_matrix = scipy_sparse.coo_matrix
    cupyx_sparse.csr_matrix = scipy_sparse.csr_matrix
    cupyx_sparse.tril = scipy_sparse.tril
    cupyx_sparse_linalg = types.ModuleType("cupyx.scipy.sparse.linalg")
    cupyx_sparse_linalg.spsolve = scipy_sparse_linalg.spsolve
    cupyx_sparse.linalg = cupyx_sparse_linalg
    cupyx_scipy.sparse = cupyx_sparse
    cupyx.scipy = cupyx_scipy

    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "cupyx", cupyx)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", cupyx_scipy)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.sparse", cupyx_sparse)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.sparse.linalg", cupyx_sparse_linalg)

    module = importlib.import_module("ipax.backend.sparse.cupy")
    yield module
    sys.modules.pop("ipax.backend.sparse.cupy", None)


def test_cupy_adapter_builds_and_spsolve_fallback(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cupy_sparse_module,
        "_load_cudss",
        lambda: (_ for _ in ()).throw(ImportError("missing cuDSS")),
    )

    adapter = cupy_sparse_module.CuPySparseAdapter()
    rows = np.asarray([0, 1, 1, 2])
    cols = np.asarray([0, 0, 1, 2])
    values = np.asarray([2.0, 1.0, 3.0, 4.0])
    operator = adapter.from_coo(rows, cols, values, shape=(3, 3))

    actual_matvec = operator.matvec(np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(actual_matvec, np.asarray([2.0, 7.0, 12.0]))

    solver = adapter.solver()
    solver.factor(operator)
    actual = solver.solve(np.asarray([2.0, 7.0, 8.0]))
    np.testing.assert_allclose(actual, np.asarray([1.0, 2.0, 2.0]))


def test_cupy_solver_requires_cudss_for_inertia(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cupy_sparse_module,
        "_load_cudss",
        lambda: (_ for _ in ()).throw(ImportError("missing cuDSS")),
    )

    adapter = cupy_sparse_module.CuPySparseAdapter()
    rows = np.asarray([0, 1])
    cols = np.asarray([0, 1])
    values = np.asarray([1.0, -1.0])
    operator = adapter.from_coo(rows, cols, values, shape=(2, 2))
    solver = adapter.solver(require_inertia=True)

    with pytest.raises(NotImplementedError, match="GPU sparse inertia requires"):
        solver.factor(operator)


class _CuDSSError(Exception):
    """Stand-in for ``nvmath.bindings.cudss.cuDSSError``."""


class _Phase(enum.IntEnum):
    # cuDSS phase bitmask (CUDA ``cudss.h``): ANALYSIS = REORDERING |
    # SYMBOLIC_FACTORIZATION, SOLVE = OR of the six solve sub-phases.
    ANALYSIS = (1 << 0) | (1 << 1)
    FACTORIZATION = 1 << 2
    SOLVE = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9)


class _MatrixType(enum.IntEnum):
    GENERAL = 0
    SYMMETRIC = 1


class _MatrixViewType(enum.IntEnum):
    FULL = 0
    LOWER = 1


class _IndexBase(enum.IntEnum):
    ZERO = 0


class _Layout(enum.IntEnum):
    ROW_MAJOR = 1


class _DataParam(enum.IntEnum):
    INERTIA = 3


class _FakeCudss:
    """Fake ``nvmath.bindings.cudss`` module: records phases, fixed inertia.

    nvmath bindings return opaque handles as plain ``intptr_t`` ints and raise
    :class:`cuDSSError` on failure, so the fake mirrors that calling convention.
    """

    cuDSSError = _CuDSSError
    Phase = _Phase
    MatrixType = _MatrixType
    MatrixViewType = _MatrixViewType
    IndexBase = _IndexBase
    Layout = _Layout
    DataParam = _DataParam

    def __init__(self) -> None:
        self.next_handle = 100
        self.phases: list[int] = []
        self.streams: list[int] = []
        self.dense_matrices: dict[int, tuple[int, int, int, int]] = {}
        self.matrix_set_values_calls = 0
        self.create_calls = 0

    def _handle(self) -> int:
        self.next_handle += 1
        return self.next_handle

    def create(self) -> int:
        self.create_calls += 1
        return self._handle()

    def config_create(self) -> int:
        return self._handle()

    def data_create(self, _handle: int) -> int:
        return self._handle()

    def matrix_create_csr(self, *args: Any) -> int:
        del args
        return self._handle()

    def matrix_create_dn(
        self,
        nrows: int,
        ncols: int,
        _ld: int,
        values: int,
        value_type: int,
        _layout: int,
    ) -> int:
        handle = self._handle()
        self.dense_matrices[handle] = (nrows, ncols, values, value_type)
        return handle

    def matrix_set_values(self, *args: Any) -> None:
        del args
        self.matrix_set_values_calls += 1

    def execute(
        self,
        _handle: int,
        phase: int,
        _config: int,
        _data: int,
        _matrix: int,
        solution: int,
        rhs: int,
    ) -> None:
        self.phases.append(int(phase))
        if int(phase) == int(_Phase.SOLVE):
            nrows, ncols, solution_ptr, value_type = self.dense_matrices[solution]
            _, _, rhs_ptr, _ = self.dense_matrices[rhs]
            scalar = ctypes.c_float if value_type == 0 else ctypes.c_double
            length = nrows * ncols
            solution_values = (scalar * length).from_address(solution_ptr)
            rhs_values = (scalar * length).from_address(rhs_ptr)
            solution_values[:] = rhs_values[:]

    def set_stream(self, _handle: int, stream: int) -> None:
        self.streams.append(int(stream))

    def data_get(
        self,
        _handle: int,
        _data: int,
        param: int,
        value: int,
        _size: int,
        written: int,
    ) -> None:
        assert int(param) == int(_DataParam.INERTIA)
        inertia = (ctypes.c_int64 * 2).from_address(value)
        inertia[0] = 1
        inertia[1] = 1
        ctypes.c_size_t.from_address(written).value = ctypes.sizeof(inertia)

    def matrix_destroy(self, *args: Any) -> None:
        del args

    def data_destroy(self, *args: Any) -> None:
        del args

    def config_destroy(self, *args: Any) -> None:
        del args

    def destroy(self, *args: Any) -> None:
        del args


def test_cudss_solver_reports_inertia(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCudss()
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))

    adapter = cupy_sparse_module.CuPySparseAdapter()
    rows = np.asarray([0, 1])
    cols = np.asarray([0, 1])
    values = np.asarray([1.0, -1.0])
    operator = adapter.from_coo(rows, cols, values, shape=(2, 2))
    solver = adapter.solver(require_inertia=True)

    solver.factor(operator)

    assert solver.inertia == (1, 1, 0)
    assert fake.phases == [int(_Phase.ANALYSIS) | int(_Phase.FACTORIZATION)]


def test_cudss_solver_binds_current_stream_and_solves(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCudss()
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))

    adapter = cupy_sparse_module.CuPySparseAdapter()
    rows = np.asarray([0, 1])
    cols = np.asarray([0, 1])
    values = np.asarray([1.0, 1.0])
    operator = adapter.from_coo(rows, cols, values, shape=(2, 2))
    solver = adapter.solver()

    solver.factor(operator)
    actual = solver.solve(np.asarray([2.0, 3.0]))

    np.testing.assert_allclose(actual, np.asarray([2.0, 3.0]))
    assert fake.streams == [1234, 1234]
    assert fake.phases[-1] == int(_Phase.SOLVE)


def test_cudss_solver_reuses_symbolic_analysis_without_host_pattern_copy(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCudss()
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))

    adapter = cupy_sparse_module.CuPySparseAdapter()
    rows = np.asarray([0, 1])
    cols = np.asarray([0, 1])
    signature = ("stable-diagonal", 2)
    first = adapter.from_coo(
        rows,
        cols,
        np.asarray([1.0, 2.0]),
        shape=(2, 2),
        pattern_signature=signature,
    )
    second = adapter.from_coo(
        rows,
        cols,
        np.asarray([3.0, 4.0]),
        shape=(2, 2),
        pattern_signature=signature,
    )
    solver = adapter.solver()
    monkeypatch.setattr(
        cupy_sparse_module.cupy,
        "array_equal",
        lambda *args: (_ for _ in ()).throw(AssertionError(args)),
    )

    solver.factor(first)
    solver.factor(second)

    assert fake.phases == [
        int(_Phase.ANALYSIS) | int(_Phase.FACTORIZATION),
        int(_Phase.FACTORIZATION),
    ]
    assert fake.matrix_set_values_calls == 1


def test_cudss_unknown_pattern_reanalyzes_without_device_comparison(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCudss()
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))
    monkeypatch.setattr(
        cupy_sparse_module.cupy,
        "array_equal",
        lambda *args: (_ for _ in ()).throw(AssertionError(args)),
    )

    adapter = cupy_sparse_module.CuPySparseAdapter()
    rows = np.asarray([0, 1])
    cols = np.asarray([0, 1])
    first = adapter.from_coo(rows, cols, np.asarray([1.0, 2.0]), shape=(2, 2))
    second = adapter.from_coo(rows, cols, np.asarray([3.0, 4.0]), shape=(2, 2))
    solver = adapter.solver()

    solver.factor(first)
    solver.factor(second)

    assert fake.phases == [
        int(_Phase.ANALYSIS) | int(_Phase.FACTORIZATION),
        int(_Phase.ANALYSIS) | int(_Phase.FACTORIZATION),
    ]
    assert fake.matrix_set_values_calls == 0


def test_cudss_signature_reuse_requires_matching_shape_nnz_type_metadata(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCudss()
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))
    monkeypatch.setattr(
        cupy_sparse_module.cupy,
        "array_equal",
        lambda *args: (_ for _ in ()).throw(AssertionError(args)),
    )

    adapter = cupy_sparse_module.CuPySparseAdapter()
    signature = ("stable", "but-metadata-changed")
    first = adapter.from_coo(
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        np.asarray([1.0, 2.0]),
        shape=(2, 2),
        pattern_signature=signature,
    )
    second = adapter.from_coo(
        np.asarray([0, 1, 1]),
        np.asarray([0, 0, 1]),
        np.asarray([3.0, 0.5, 4.0]),
        shape=(2, 2),
        pattern_signature=signature,
    )
    solver = adapter.solver()

    solver.factor(first)
    solver.factor(second)

    assert fake.phases == [
        int(_Phase.ANALYSIS) | int(_Phase.FACTORIZATION),
        int(_Phase.ANALYSIS) | int(_Phase.FACTORIZATION),
    ]
    assert fake.matrix_set_values_calls == 0


def test_cudss_missing_runtime_uses_spsolve_fallback(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DynamicLibNotFoundError(Exception):
        pass

    DynamicLibNotFoundError.__module__ = "cuda.pathfinder._dynamic_libs"

    class MissingRuntimeCudss(_FakeCudss):
        def create(self) -> int:
            raise DynamicLibNotFoundError("cudss library is absent")

    monkeypatch.setattr(
        cupy_sparse_module, "_load_cudss", lambda: MissingRuntimeCudss()
    )

    adapter = cupy_sparse_module.CuPySparseAdapter()
    operator = adapter.from_coo(
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        np.asarray([2.0, 4.0]),
        shape=(2, 2),
    )
    solver = adapter.solver()

    solver.factor(operator)
    actual = solver.solve(np.asarray([2.0, 8.0]))

    np.testing.assert_allclose(actual, np.asarray([1.0, 2.0]))


def test_from_coo_signature_fast_path_matches_slow_path(
    cupy_sparse_module: types.ModuleType,
) -> None:
    # The compiled-map fast path (pattern_signature set) must build the same
    # canonical matrix as the from-scratch COO build (signature absent).
    adapter = cupy_sparse_module.CuPySparseAdapter()
    rows = np.asarray([0, 0, 1, 1, 0])  # (0,0) duplicated ⇒ summed
    cols = np.asarray([0, 1, 0, 1, 0])
    values = np.asarray([2.0, 1.0, 1.0, 3.0, 0.5])

    slow = adapter.from_coo(rows, cols, values, shape=(2, 2))
    fast = cupy_sparse_module.CuPySparseAdapter().from_coo(
        rows, cols, values, shape=(2, 2), pattern_signature=("p", (2, 2))
    )

    probe = np.asarray([1.0, -2.0])
    np.testing.assert_allclose(fast.matvec(probe), slow.matvec(probe))


def test_from_coo_signature_reuse_solves_correctly_via_cudss(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCudss()
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))

    adapter = cupy_sparse_module.CuPySparseAdapter()
    rows = np.asarray([0, 0, 1, 1])
    cols = np.asarray([0, 1, 0, 1])
    signature = ("kkt", (2, 2))
    solver = adapter.solver()

    first = adapter.from_coo(
        rows,
        cols,
        np.asarray([2.0, 1.0, 1.0, 3.0]),
        shape=(2, 2),
        pattern_signature=signature,
    )
    solver.factor(first)
    second = adapter.from_coo(
        rows,
        cols,
        np.asarray([5.0, 1.0, 1.0, 4.0]),
        shape=(2, 2),
        pattern_signature=signature,
    )
    solver.factor(second)

    # Symbolic analysis reused (one analysis, one values-only factor); the cached
    # lower-triangle map serves the symmetric route without a per-iter tril.
    assert fake.phases == [
        int(_Phase.ANALYSIS) | int(_Phase.FACTORIZATION),
        int(_Phase.FACTORIZATION),
    ]
    assert fake.matrix_set_values_calls == 1


def test_cudss_uses_int32_indices_for_small_system(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCudss()
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))

    adapter = cupy_sparse_module.CuPySparseAdapter()
    operator = adapter.from_coo(
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        np.asarray([1.0, 1.0]),
        shape=(2, 2),
        pattern_signature=("diag", (2, 2)),
    )
    solver = adapter.solver()
    solver.factor(operator)

    # Optimization #3: the uploaded CSR offsets/indices are int32 when the system
    # fits a signed 32-bit integer.
    assert solver._inner._row_offsets.dtype == np.int32
    assert solver._inner._col_indices.dtype == np.int32


def test_cudss_solver_recreates_context_on_device_change(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCudss()
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))
    runtime = cupy_sparse_module.cupy.cuda.runtime

    adapter = cupy_sparse_module.CuPySparseAdapter()
    operator = adapter.from_coo(
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        np.asarray([1.0, 1.0]),
        shape=(2, 2),
    )
    solver = adapter.solver()

    runtime.current_device = 0
    solver.factor(operator)
    runtime.current_device = 1
    solver.factor(operator)

    assert fake.create_calls == 2
