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

    # Model the most restrictive supported CuPy: array-valued ``repeats`` are
    # only accepted from CuPy v14, so the fake rejects them like older CuPy
    # (message verbatim, typo included) instead of inheriting NumPy's leniency.
    def _repeat(a: np.ndarray, repeats: object, axis: int | None = None) -> np.ndarray:
        if isinstance(repeats, np.ndarray):
            raise ValueError("cupy.ndaray cannot be specified as `repeats` argument.")
        return np.repeat(a, repeats, axis)

    cupy.repeat = _repeat
    cupy.searchsorted = np.searchsorted
    cupy.result_type = np.result_type
    # Unbuffered scatter-add used by the canonical COO map (cupy.add.at is
    # the documented replacement for the deprecated cupyx.scatter_add).
    cupy.add = np.add
    # Weight gating + √w scaling + triangle mirror for the syrk Gram path.
    cupy.sqrt = np.sqrt
    cupy.isfinite = np.isfinite
    cupy.all = np.all
    cupy.where = np.where
    cupy.isnan = np.isnan
    cupy.inf = np.inf
    cupy.minimum = np.minimum
    cupy.stack = np.stack
    cupy.any = np.any
    cupy.triu_indices = np.triu_indices

    # ``cupy.cublas.syrk`` fake with real cuBLAS semantics: only the requested
    # triangle of ``out`` is written; the other one keeps its stale contents —
    # so a missing mirror step in the adapter corrupts the result visibly.
    cublas = types.ModuleType("cupy.cublas")

    def _syrk(
        trans: str,
        a: np.ndarray,
        out: np.ndarray | None = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        lower: bool = False,
    ) -> np.ndarray:
        assert trans == "T" and lower and out is not None
        full = alpha * (a.T @ a) + beta * out
        tri = np.tril_indices(out.shape[0])
        out[tri] = full[tri]
        return out

    cublas.syrk = _syrk
    cupy.cublas = cublas

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
    monkeypatch.setitem(sys.modules, "cupy.cublas", cublas)
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
    """Stand-in for ``nvmath.bindings.cudss.cuDSSError`` (carries ``status``)."""

    def __init__(self, *args: object, status: int | None = None) -> None:
        super().__init__(*args)
        self.status = status


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
    COL_MAJOR = 0
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
        self.dense_matrices: dict[int, tuple[int, int, int, int, int, int]] = {}
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
        ld: int,
        values: int,
        value_type: int,
        layout: int,
    ) -> int:
        handle = self._handle()
        self.dense_matrices[handle] = (nrows, ncols, ld, values, value_type, layout)
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
            nrows, ncols, _, solution_ptr, value_type, _ = self.dense_matrices[solution]
            _, _, _, rhs_ptr, _, _ = self.dense_matrices[rhs]
            scalar = ctypes.c_float if value_type == 0 else ctypes.c_double
            length = nrows * ncols
            solution_values = (scalar * length).from_address(solution_ptr)
            rhs_values = (scalar * length).from_address(rhs_ptr)
            solution_values[:] = rhs_values[:]

    def set_stream(self, _handle: int, stream: int) -> None:
        self.streams.append(int(stream))

    def get_property(self, prop: int) -> int:
        # libraryPropertyType: MAJOR_VERSION=0, MINOR_VERSION=1, PATCH_LEVEL=2.
        return (0, 7, 1)[int(prop)]

    def data_get(
        self,
        _handle: int,
        _data: int,
        param: int,
        value: int,
        size: int,
        written: int,
    ) -> None:
        assert int(param) == int(_DataParam.INERTIA)
        # Real cuDSS (observed on 0.7) fills INERTIA as int32[2] and rejects
        # any other buffer size with INVALID_VALUE; the fake mirrors that.
        if int(size) != 8:
            raise _CuDSSError("INVALID_VALUE (3)", status=3)
        inertia = (ctypes.c_int32 * 2).from_address(value)
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


def test_cudss_inertia_is_best_effort_on_symmetric_factors(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Parity with the Feral adapter: LDLᵀ inertia is a free by-product of a
    # symmetric cuDSS factorization and must be reported best-effort through
    # ``inertia_or_none`` even without ``require_inertia`` — the IPM's
    # inertia-guided δ_w correction depends on it (a cuDSS route returning None
    # converges to saddle points the CPU route escapes). It is read lazily so
    # solves that never consult it (L-BFGS) pay no per-factor host sync.
    fake = _FakeCudss()
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))

    adapter = cupy_sparse_module.CuPySparseAdapter()
    operator = adapter.from_coo(
        np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([1.0, -1.0]), shape=(2, 2)
    )
    solver = adapter.solver()  # note: no require_inertia
    solver.factor(operator)

    assert solver.inertia_or_none() == (1, 1, 0)


def test_cudss_inertia_or_none_is_none_for_general_factors(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # cuDSS defines inertia only for symmetric factorizations; a GENERAL-route
    # factor must report None, not read garbage.
    fake = _FakeCudss()
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))

    adapter = cupy_sparse_module.CuPySparseAdapter()
    operator = adapter.from_coo(
        np.asarray([0, 0, 1]),
        np.asarray([0, 1, 1]),
        np.asarray([1.0, 2.0, 1.0]),  # (0,1) has no (1,0) mirror ⇒ not symmetric
        shape=(2, 2),
    )
    solver = adapter.solver()
    solver.factor(operator)

    assert solver.inertia_or_none() is None


class _DataParamWithPivots(enum.IntEnum):
    INERTIA = 3
    NPIVOTS = 7


class _PivotingFakeCudss(_FakeCudss):
    """Fake cuDSS reporting a configurable perturbed-pivot count (``NPIVOTS``).

    Its SOLVE copies the right-hand side (``x = b``), so a test picks the
    matrix to make that "solution" consistent or not.
    """

    DataParam = _DataParamWithPivots

    def __init__(self, npivots: int) -> None:
        super().__init__()
        self.npivots = npivots
        self.npivot_reads = 0

    def data_get(
        self, handle: int, data: int, param: int, value: int, size: int, written: int
    ) -> None:
        if int(param) != int(_DataParamWithPivots.NPIVOTS):
            super().data_get(
                handle, data, int(_DataParam.INERTIA), value, size, written
            )
            return
        if int(size) != 4:  # int32, like INERTIA's entries on cuDSS 0.7
            raise _CuDSSError("INVALID_VALUE (3)", status=3)
        ctypes.c_int32.from_address(value).value = self.npivots
        ctypes.c_size_t.from_address(written).value = 4
        self.npivot_reads += 1


def _cudss_solver(module, monkeypatch, fake, values, *, require_inertia=False):
    monkeypatch.setattr(module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(module, "_ptr", lambda arr: int(arr.ctypes.data))
    rows = np.asarray([0, 0, 1, 1])
    cols = np.asarray([0, 1, 0, 1])
    operator = module.CuPySparseAdapter().from_coo(
        rows, cols, np.asarray(values, dtype=np.float64), shape=(2, 2), symmetric=True
    )
    solver = module.CuPySparseSolver(require_inertia=require_inertia)
    solver.factor(operator)
    return solver


@pytest.mark.parametrize("fake_factory", [_FakeCudss, lambda: _PivotingFakeCudss(0)])
def test_cudss_unperturbed_factor_is_never_certified(
    cupy_sparse_module, monkeypatch, fake_factory
):
    # No perturbed pivot (or a runtime that cannot report NPIVOTS): the solve
    # is returned as cuDSS produced it, with no residual SpMV — even when, as
    # with this identity-solve fake, it does not solve the system.
    fake = fake_factory()
    calls = {"n": 0}
    original = cupy_sparse_module.CuDSSSparseSolver._certify_perturbed_solve

    def spy(self, b, x):
        calls["n"] += 1
        return original(self, b, x)

    monkeypatch.setattr(
        cupy_sparse_module.CuDSSSparseSolver, "_certify_perturbed_solve", spy
    )
    solver = _cudss_solver(cupy_sparse_module, monkeypatch, fake, [2.0, 0.0, 0.0, 3.0])
    for _ in range(2):
        np.testing.assert_array_equal(
            solver.solve(np.asarray([1.0, 2.0])), np.asarray([1.0, 2.0])
        )
    assert calls["n"] == 0
    if isinstance(fake, _PivotingFakeCudss):
        assert fake.npivot_reads == 1  # once per factorization, not per solve


def test_cudss_perturbed_consistent_solve_is_returned_unchanged(
    cupy_sparse_module, monkeypatch
):
    # A perturbed pivot on a system the returned vector does solve (here
    # A = I): accepted as is — the certificate never alters a solution.
    fake = _PivotingFakeCudss(1)
    solver = _cudss_solver(cupy_sparse_module, monkeypatch, fake, [1.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(
        solver.solve(np.asarray([1.0, 2.0])), np.asarray([1.0, 2.0])
    )


def test_cudss_perturbed_inconsistent_solve_is_a_linear_solve_error(
    cupy_sparse_module, monkeypatch
):
    # Singular A = [[1, 1], [1, 1]] with b outside its range: what cuDSS's
    # static pivoting hands back cannot solve the system (relative residual
    # O(1)), so it is classified as the numerical failure it is.
    from ipax.linalg.solver import LinearSolveError

    fake = _PivotingFakeCudss(1)
    solver = _cudss_solver(cupy_sparse_module, monkeypatch, fake, [1.0, 1.0, 1.0, 1.0])
    with pytest.raises(LinearSolveError, match="perturbed 1 pivot"):
        solver.solve(np.asarray([1.0, 2.0]))


def test_cudss_perturbed_certificate_is_per_column(cupy_sparse_module, monkeypatch):
    # A = diag(1, 0): column 0 of B is consistent and large, column 1
    # inconsistent and small. A global max-norm test would let column 1's
    # O(1) residual hide behind column 0's 1e6 scale.
    from ipax.linalg.solver import LinearSolveError

    fake = _PivotingFakeCudss(1)
    solver = _cudss_solver(cupy_sparse_module, monkeypatch, fake, [1.0, 0.0, 0.0, 0.0])
    rhs = np.asarray([[1e6, 0.0], [0.0, 1.0]])
    with pytest.raises(LinearSolveError):
        solver.solve(rhs)


class _RefiningFakeCudss(_PivotingFakeCudss):
    """Fake cuDSS whose SOLVE uses a *perturbed* factor ``A + E``.

    Models static pivoting: every SOLVE phase returns ``(A + E)⁻¹ b`` while the
    adapter measures residuals against the true ``A`` it assembled. ``A`` and
    ``E`` are dense NumPy arrays the test chooses; ``solves`` counts SOLVE
    phases (the initial solve plus one per refinement step).
    """

    def __init__(self, npivots: int, A: np.ndarray, E: np.ndarray) -> None:
        super().__init__(npivots)
        self.A = np.asarray(A, dtype=np.float64)
        self.perturbed = self.A + np.asarray(E, dtype=np.float64)
        self.solves = 0

    def execute(self, _handle, phase, _config, _data, _matrix, solution, rhs) -> None:
        self.phases.append(int(phase))
        if int(phase) != int(_Phase.SOLVE):
            return
        self.solves += 1
        nrows, ncols, _, solution_ptr, _, _ = self.dense_matrices[solution]
        _, _, _, rhs_ptr, _, _ = self.dense_matrices[rhs]
        length = nrows * ncols
        # Column-major buffers (cuDSS supports COL_MAJOR only).
        b = np.array((ctypes.c_double * length).from_address(rhs_ptr))
        x = np.linalg.solve(self.perturbed, b.reshape(ncols, nrows).T)
        out = (ctypes.c_double * length).from_address(solution_ptr)
        out[:] = np.asarray(x.T, dtype=np.float64).ravel().tolist()


def _refining_solver(module, monkeypatch, fake, **policy):
    monkeypatch.setattr(module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(module, "_ptr", lambda arr: int(arr.ctypes.data))
    n = fake.A.shape[0]
    rows, cols = np.nonzero(np.ones((n, n)))
    adapter = module.CuPySparseAdapter(**policy)
    operator = adapter.from_coo(
        rows, cols, fake.A[rows, cols], shape=(n, n), symmetric=True
    )
    solver = adapter.solver()
    solver.factor(operator)
    return solver


# A saddle point whose (0, 0) entry is exactly zero: a symmetric factorization
# without 2×2 pivots meets a zero pivot there and static pivoting perturbs it
# (the minimax epigraph variable of MAKELA1 / MIFFLIN2 has this structure).
# Well conditioned, so the unperturbed system is benign.
_SADDLE = np.asarray([[0.0, 1.0, 1.0], [1.0, 4.0, 0.0], [1.0, 0.0, 3.0]])
_SADDLE_RHS = np.asarray([1.0, 2.0, 3.0])


def _perturbation(value: float) -> np.ndarray:
    E = np.zeros((3, 3))
    E[0, 0] = value
    return E


def _relative_residual(A, b, x):
    return float(np.max(np.abs(b - A @ np.asarray(x))) / np.max(np.abs(b)))


def test_cudss_refines_a_perturbed_factor_to_the_unperturbed_solution(
    cupy_sparse_module, monkeypatch
):
    # E = 1e-3 on the zero pivot: unrefined, the solve misses the system by a
    # relative residual ~3e-4 — accepted by the 1e-2 certificate, but a poor
    # Newton step (the CuPy sweep's accepted-but-inaccurate class). Refining
    # against A contracts the error by ≈ ‖(A+E)⁻¹E‖ ≈ 2e-3 per step, so the
    # IPOPT target 1e-10 is reached in three steps and the loop stops there.
    reference = np.linalg.solve(_SADDLE, _SADDLE_RHS)
    unrefined = _refining_solver(
        cupy_sparse_module,
        monkeypatch,
        _RefiningFakeCudss(1, _SADDLE, _perturbation(1e-3)),
        max_refinement_steps=0,
    )
    x0 = unrefined.solve(_SADDLE_RHS)
    assert 1e-5 < _relative_residual(_SADDLE, _SADDLE_RHS, x0) < 1e-2
    assert unrefined._inner.last_refinement_steps == 0

    fake = _RefiningFakeCudss(1, _SADDLE, _perturbation(1e-3))
    solver = _refining_solver(cupy_sparse_module, monkeypatch, fake)
    x = solver.solve(_SADDLE_RHS)
    assert _relative_residual(_SADDLE, _SADDLE_RHS, x) <= 1e-10
    np.testing.assert_allclose(x, reference, rtol=1e-9, atol=0.0)
    assert solver._inner.last_refinement_steps == 3
    assert fake.solves == 1 + 3
    assert solver._inner.last_residual_ratio <= 1e-10


def test_cudss_refinement_rescues_a_consistent_solve_the_certificate_rejected(
    cupy_sparse_module, monkeypatch
):
    # A larger perturbation (contraction ≈ 0.2) leaves the unrefined solve
    # above the 1e-2 certificate — a false positive on a consistent system;
    # refinement brings it well below, within the step cap.
    from ipax.linalg.solver import LinearSolveError

    unrefined = _refining_solver(
        cupy_sparse_module,
        monkeypatch,
        _RefiningFakeCudss(1, _SADDLE, _perturbation(0.1)),
        max_refinement_steps=0,
    )
    with pytest.raises(LinearSolveError, match="perturbed 1 pivot"):
        unrefined.solve(_SADDLE_RHS)

    fake = _RefiningFakeCudss(1, _SADDLE, _perturbation(0.1))
    solver = _refining_solver(cupy_sparse_module, monkeypatch, fake)
    x = solver.solve(_SADDLE_RHS)
    assert _relative_residual(_SADDLE, _SADDLE_RHS, x) <= 1e-4
    steps = cupy_sparse_module._PERTURBED_MAX_REFINEMENT_STEPS
    assert solver._inner.last_refinement_steps == steps  # capped, still improving
    assert fake.solves == 1 + steps


def test_cudss_refinement_cannot_rescue_an_inconsistent_system(
    cupy_sparse_module, monkeypatch
):
    # Singular A = [[1, 1], [1, 1]], b outside its range: the ±1/ε spike's
    # residual is b's component outside range(A), which no refinement step
    # reduces — one non-improving step, then the LinearSolveError the δ_w/δ_c
    # ladder acts on.
    from ipax.linalg.solver import LinearSolveError

    fake = _RefiningFakeCudss(1, np.ones((2, 2)), np.diag([0.0, 1e-13]))
    solver = _refining_solver(cupy_sparse_module, monkeypatch, fake)
    with pytest.raises(LinearSolveError, match="after 1 refinement step"):
        solver.solve(np.asarray([1.0, 2.0]))
    assert fake.solves == 2


def test_cudss_refinement_keeps_the_better_iterate_when_it_diverges(
    cupy_sparse_module, monkeypatch
):
    # ‖(A+E)⁻¹E‖ > 1 (E = 1 on the zero pivot): a refinement step *raises*
    # the residual (0.4 → 0.96). The loop stops at that step and returns the
    # unrefined solution, never the worse one (a threshold of 1.0 accepts it,
    # isolating the keep-best rule from the certificate).
    policy = {"perturbed_max_residual": 1.0}
    unrefined = _refining_solver(
        cupy_sparse_module,
        monkeypatch,
        _RefiningFakeCudss(1, _SADDLE, _perturbation(1.0)),
        max_refinement_steps=0,
        **policy,
    )
    x0 = unrefined.solve(_SADDLE_RHS)

    fake = _RefiningFakeCudss(1, _SADDLE, _perturbation(1.0))
    solver = _refining_solver(cupy_sparse_module, monkeypatch, fake, **policy)
    x = solver.solve(_SADDLE_RHS)
    np.testing.assert_array_equal(x, x0)
    assert solver._inner.last_refinement_steps == 1
    assert fake.solves == 2


def test_cudss_unperturbed_factor_pays_no_refinement(cupy_sparse_module, monkeypatch):
    # NPIVOTS = 0: one SOLVE phase per solve and no residual SpMV — the path
    # is exactly what it was before refinement existed.
    fake = _RefiningFakeCudss(0, _SADDLE + np.eye(3), np.zeros((3, 3)))
    calls = {"n": 0}
    original = cupy_sparse_module.CuDSSSparseSolver._certify_perturbed_solve

    def spy(self, b, x):
        calls["n"] += 1
        return original(self, b, x)

    monkeypatch.setattr(
        cupy_sparse_module.CuDSSSparseSolver, "_certify_perturbed_solve", spy
    )
    solver = _refining_solver(cupy_sparse_module, monkeypatch, fake)
    for _ in range(3):
        solver.solve(_SADDLE_RHS)
    assert fake.solves == 3
    assert calls["n"] == 0


def test_cudss_refinement_policy_is_threaded_and_validated(
    cupy_sparse_module, monkeypatch
):
    fake = _RefiningFakeCudss(1, _SADDLE, _perturbation(0.1))
    solver = _refining_solver(
        cupy_sparse_module,
        monkeypatch,
        fake,
        max_refinement_steps=2,
        perturbed_max_residual=1e-5,
    )
    inner = solver._inner
    assert (inner._max_refinement_steps, inner._perturbed_max_residual) == (2, 1e-5)
    from ipax.linalg.solver import LinearSolveError

    # Two steps of contraction ≈ 0.2 leave ~1e-3: above the tighter threshold.
    with pytest.raises(LinearSolveError, match="after 2 refinement step"):
        solver.solve(_SADDLE_RHS)

    adapter_cls = cupy_sparse_module.CuPySparseAdapter
    for bad in ({"max_refinement_steps": -1}, {"max_refinement_steps": 1.5}):
        with pytest.raises(ValueError, match="max_refinement_steps"):
            adapter_cls(**bad)
    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="perturbed_max_residual"):
            adapter_cls(perturbed_max_residual=value)
    with pytest.raises(ValueError, match="max_refinement_steps"):
        cupy_sparse_module.CuDSSSparseSolver(max_refinement_steps=True)


def test_cudss_refinement_keeps_the_best_iterate_per_column(
    cupy_sparse_module, monkeypatch
):
    # Two decoupled saddle blocks with different perturbations: refinement
    # contracts block 1 (E = 1e-3) and diverges on block 2 (E = 1). Column 0
    # lives on block 1, column 1 on block 2. The diverging column must come
    # back bit-equal to its unrefined solve while the converging one is
    # refined; the loop stops at the first step where the *worst* column
    # does not improve (here: step 1). Threshold 1.0 isolates keep-best from
    # the certificate.
    A = np.zeros((6, 6))
    A[:3, :3] = _SADDLE
    A[3:, 3:] = _SADDLE
    E = np.zeros((6, 6))
    E[0, 0] = 1e-3
    E[3, 3] = 1.0
    rhs = np.zeros((6, 2))
    rhs[:3, 0] = _SADDLE_RHS
    rhs[3:, 1] = _SADDLE_RHS
    policy = {"perturbed_max_residual": 1.0}
    unrefined = _refining_solver(
        cupy_sparse_module,
        monkeypatch,
        _RefiningFakeCudss(1, A, E),
        max_refinement_steps=0,
        **policy,
    )
    x0 = unrefined.solve(rhs)

    fake = _RefiningFakeCudss(1, A, E)
    solver = _refining_solver(cupy_sparse_module, monkeypatch, fake, **policy)
    x = solver.solve(rhs)
    np.testing.assert_array_equal(x[:, 1], x0[:, 1])
    r0 = _relative_residual(A, rhs[:, 0], x0[:, 0])
    r1 = _relative_residual(A, rhs[:, 0], x[:, 0])
    assert r1 < 1e-2 * r0
    assert solver._inner.last_refinement_steps == 1
    assert fake.solves == 2


def test_cudss_non_finite_perturbed_solve_is_rejected_without_refinement(
    cupy_sparse_module, monkeypatch
):
    # A solve that comes back non-finite has an ∞ ratio: nothing to refine
    # (no extra SOLVE phase), rejected as before.
    from ipax.linalg.solver import LinearSolveError

    class NaNSolve(_RefiningFakeCudss):
        def execute(self, _h, phase, _c, _d, _m, solution, rhs) -> None:
            super().execute(_h, phase, _c, _d, _m, solution, rhs)
            if int(phase) == int(_Phase.SOLVE):
                nrows, ncols, _, ptr, _, _ = self.dense_matrices[solution]
                (ctypes.c_double * (nrows * ncols)).from_address(ptr)[0] = float("nan")

    fake = NaNSolve(1, _SADDLE, _perturbation(1e-3))
    solver = _refining_solver(cupy_sparse_module, monkeypatch, fake)
    with pytest.raises(LinearSolveError, match="after 0 refinement step"):
        solver.solve(_SADDLE_RHS)
    assert fake.solves == 1


def test_cudss_refinement_target_is_floored_at_the_dtype_roundoff(
    cupy_sparse_module, monkeypatch
):
    # A float32 factorization cannot reach 1e-10: the target is floored at
    # _PERTURBED_REFINEMENT_EPS_FACTOR float32 roundoffs (≈1.2e-5), so a
    # solve already below that takes no refinement step.
    fake = _RefiningFakeCudss(1, _SADDLE, _perturbation(1e-3))
    solver = _refining_solver(cupy_sparse_module, monkeypatch, fake)
    inner = solver._inner
    monkeypatch.setattr(inner, "_dtype", np.dtype(np.float32))
    b = np.asarray(_SADDLE_RHS)[:, None]
    x = np.linalg.solve(_SADDLE, b) + 1e-8  # ratio ≈ 1e-8: above 1e-10
    inner._certify_perturbed_solve(b, x)
    assert inner.last_refinement_steps == 0
    monkeypatch.setattr(inner, "_dtype", np.dtype(np.float64))
    inner._certify_perturbed_solve(b, x)
    assert inner.last_refinement_steps >= 1


def test_cudss_certificate_counter_sees_the_refined_solve(
    cupy_sparse_module, monkeypatch
):
    # The benchmark-side counter wraps the certificate: it must hand the
    # refined solution back (the wrapper's return is load-bearing) and bin
    # the unrefined vs the certified (refined) residual.
    from benchmarks.runners._cupy_certificates import CertificateCounter

    fake = _RefiningFakeCudss(1, _SADDLE, _perturbation(0.1))
    solver = _refining_solver(cupy_sparse_module, monkeypatch, fake)
    with CertificateCounter() as counter:
        x = solver.solve(_SADDLE_RHS)
    assert _relative_residual(_SADDLE, _SADDLE_RHS, x) <= 1e-4
    summary = counter.summary()
    assert summary["perturbed_solves"] == 1
    assert summary["rejected_solves"] == 0
    assert summary["rescued_solves"] == 1  # 3.4e-2 unrefined > 1e-2 threshold
    assert summary["unrefined_bins"] == {"1e-2": 1}
    steps = cupy_sparse_module._PERTURBED_MAX_REFINEMENT_STEPS
    assert summary["refinement_steps"] == {str(steps): 1}
    assert summary["max_refinement_steps"] == steps
    assert summary["perturbed_max_residual"] == 1e-2


def test_cudss_non_finite_matrix_is_a_linear_solve_error(
    cupy_sparse_module, monkeypatch
):
    # Deferred to the first solve (a factor stays sync-free)...
    from ipax.linalg.solver import LinearSolveError

    values = [float("nan"), 0.0, 0.0, 1.0]
    solver = _cudss_solver(cupy_sparse_module, monkeypatch, _FakeCudss(), values)
    with pytest.raises(LinearSolveError, match="non-finite"):
        solver.solve(np.asarray([1.0, 2.0]))
    # ...but raised by factor itself when require_inertia already syncs there.
    with pytest.raises(LinearSolveError, match="non-finite"):
        _cudss_solver(
            cupy_sparse_module, monkeypatch, _FakeCudss(), values, require_inertia=True
        )


def test_spsolve_fallback_rejects_a_non_finite_matrix(cupy_sparse_module):
    from ipax.linalg.solver import LinearSolveError

    operator = cupy_sparse_module.CuPySparseAdapter().from_coo(
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        np.asarray([float("inf"), 1.0]),
        shape=(2, 2),
    )
    with pytest.raises(LinearSolveError, match="non-finite"):
        cupy_sparse_module.CuPySpsolveSolver().factor(operator)


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


def test_cudss_reuses_dense_descriptors_across_solves(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The persistent RHS/solution descriptors are created once (factor binds them
    # for the analysis phase) and reused for every solve of the same shape — no
    # per-solve create/destroy churn (#4).
    fake = _FakeCudss()
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))

    adapter = cupy_sparse_module.CuPySparseAdapter()
    operator = adapter.from_coo(
        np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([2.0, 4.0]), shape=(2, 2)
    )
    solver = adapter.solver()
    solver.factor(operator)
    dense_after_factor = len(fake.dense_matrices)

    # The fake cuDSS echoes the RHS into the solution (it does not really solve),
    # so the asserted values mirror the inputs; the point is the descriptor reuse.
    first = solver.solve(np.asarray([2.0, 8.0]))
    second = solver.solve(np.asarray([6.0, 4.0]))

    np.testing.assert_allclose(first, np.asarray([2.0, 8.0]))
    np.testing.assert_allclose(second, np.asarray([6.0, 4.0]))
    # Two dense descriptors total (rhs + sol), created at factor time and reused;
    # neither solve allocated a fresh pair.
    assert dense_after_factor == 2
    assert len(fake.dense_matrices) == 2


def test_cudss_dense_descriptors_are_column_major(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # cuDSS supports only column-major dense matrices (``cudssMatrixCreateDn``
    # returns NOT_SUPPORTED for ROW_MAJOR on a real runtime), so every dense
    # RHS/solution descriptor must be created COL_MAJOR with ld = nrows.
    fake = _FakeCudss()
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))

    adapter = cupy_sparse_module.CuPySparseAdapter()
    operator = adapter.from_coo(
        np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([2.0, 4.0]), shape=(2, 2)
    )
    solver = adapter.solver()
    solver.factor(operator)
    solver.solve(np.asarray([2.0, 8.0]))

    assert fake.dense_matrices  # descriptors were created through the fake
    for nrows, _ncols, ld, _values, _vt, layout in fake.dense_matrices.values():
        assert layout == int(_Layout.COL_MAJOR)
        assert ld == nrows


def test_cudss_creation_rejection_degrades_to_spsolve_with_warning(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A cuDSS runtime that rejects matrix creation outright (INVALID_VALUE /
    # NOT_SUPPORTED — the signature of an ABI mismatch between the installed
    # cuDSS library and the nvmath-python bindings) is unusable, not a numerical
    # failure: the solver must degrade to spsolve with a loud warning instead of
    # letting the driver burn its regularization ladder on identical errors.
    fake = _FakeCudss()

    def rejecting_create_csr(*args: Any) -> int:
        del args
        raise _CuDSSError("INVALID_VALUE (3)", status=3)

    monkeypatch.setattr(fake, "matrix_create_csr", rejecting_create_csr)
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))

    adapter = cupy_sparse_module.CuPySparseAdapter()
    rows = np.asarray([0, 1, 1, 2])
    cols = np.asarray([0, 0, 1, 2])
    values = np.asarray([2.0, 1.0, 3.0, 4.0])
    operator = adapter.from_coo(rows, cols, values, shape=(3, 3))
    solver = adapter.solver()

    with pytest.warns(RuntimeWarning, match="spsolve"):
        solver.factor(operator)

    assert "spsolve" in solver.describe()
    actual = solver.solve(np.asarray([2.0, 7.0, 8.0]))
    np.testing.assert_allclose(actual, np.asarray([1.0, 2.0, 2.0]))


def test_spsolve_fallback_on_missing_cudss_warns(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Degrading to spsolve costs a full factorization per solve; it must never
    # happen silently (two real-world environments hit this without noticing).
    monkeypatch.setattr(
        cupy_sparse_module,
        "_load_cudss",
        lambda: (_ for _ in ()).throw(ImportError("missing cuDSS")),
    )

    adapter = cupy_sparse_module.CuPySparseAdapter()
    operator = adapter.from_coo(
        np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([2.0, 4.0]), shape=(2, 2)
    )
    solver = adapter.solver()

    with pytest.warns(RuntimeWarning, match="spsolve"):
        solver.factor(operator)

    assert "spsolve" in solver.describe()


def test_cudss_solve_result_is_independent_of_reused_buffer(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The returned array must be a copy: a later solve reuses the solution buffer
    # in place, and must not retroactively mutate an earlier result.
    fake = _FakeCudss()
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))

    adapter = cupy_sparse_module.CuPySparseAdapter()
    operator = adapter.from_coo(
        np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([2.0, 4.0]), shape=(2, 2)
    )
    solver = adapter.solver()
    solver.factor(operator)

    first = solver.solve(np.asarray([2.0, 8.0]))
    # The second solve overwrites the shared solution buffer in place; ``first``
    # must keep its own values (it is a copy), not track the buffer.
    solver.solve(np.asarray([6.0, 4.0]))

    np.testing.assert_allclose(first, np.asarray([2.0, 8.0]))


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


def test_cupy_gram_memo_hits_for_equal_weights(
    cupy_sparse_module: types.ModuleType,
) -> None:
    # Mirrors the SciPy adapter's per-operator Gram cache (n ≪ m fast path):
    # equal weights reuse the memoized n×n result; changed weights recompute.
    adapter = cupy_sparse_module.CuPySparseAdapter()
    rows = np.asarray([0, 0, 1, 2, 2, 3])
    cols = np.asarray([0, 2, 1, 0, 1, 2])
    values = np.asarray([2.0, 1.0, 3.0, -1.0, 0.5, 4.0])
    operator = adapter.from_coo(rows, cols, values, shape=(4, 3))

    w1 = np.asarray([0.5, 2.0, 1.0, 0.25])
    first = np.asarray(operator.gram(w1))
    second = np.asarray(operator.gram(np.array(w1, copy=True)))
    assert operator._gram_compute_count == 1
    np.testing.assert_array_equal(first, second)

    dense = np.zeros((4, 3))
    dense[rows, cols] = values
    w2 = 2.0 * w1
    third = np.asarray(operator.gram(w2))
    assert operator._gram_compute_count == 2
    np.testing.assert_allclose(third, dense.T @ (w2[:, None] * dense), rtol=1e-14)


def _syrk_spy(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Count ``cublas.syrk`` calls to pin which accumulation branch ran."""
    calls: list[str] = []
    real = cupy_sparse_module._cublas.syrk

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append("syrk")
        return real(*args, **kwargs)

    monkeypatch.setattr(cupy_sparse_module._cublas, "syrk", spy)
    return calls


def _dense_gram_operator(cupy_sparse_module: types.ModuleType, matrix, weights):
    coo = matrix.tocoo()
    adapter = cupy_sparse_module.CuPySparseAdapter()
    return adapter.from_coo(coo.row, coo.col, coo.data, shape=coo.shape)


def test_cupy_gram_dense_strategy_uses_syrk_and_mirrors(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nonnegative finite weights take the cuBLAS syrk branch: half the GEMM
    # FLOPs, and the mirrored result is *bitwise* symmetric. The fake syrk
    # writes only the lower triangle, so this also proves the upper triangle
    # comes from the adapter's mirror, not from the BLAS call.
    monkeypatch.setattr(cupy_sparse_module, "_GRAM_DENSE_MIN_DENSITY", 0.0)
    monkeypatch.setattr(cupy_sparse_module, "_GRAM_DENSE_CHUNK_ELEMENTS", 40)
    monkeypatch.setattr(cupy_sparse_module, "_GRAM_MIRROR_BLOCK", 3)
    calls = _syrk_spy(cupy_sparse_module, monkeypatch)

    rng = np.random.default_rng(11)
    matrix = scipy_sparse.csr_matrix(rng.standard_normal((23, 7)))
    weights = rng.uniform(0.0, 1e3, size=23)  # includes the w == 0 boundary
    weights[3] = 0.0
    operator = _dense_gram_operator(cupy_sparse_module, matrix, weights)

    actual = np.asarray(operator.gram(weights))

    assert calls and all(c == "syrk" for c in calls)
    assert len(calls) > 1  # chunked: the accumulation loop really ran
    np.testing.assert_array_equal(actual, actual.T)
    dense = matrix.toarray()
    np.testing.assert_allclose(
        actual, dense.T @ (weights[:, None] * dense), rtol=1e-12, atol=1e-12
    )


def test_cupy_gram_accumulate_dtype_float32(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mirror of the SciPy adapter's reduced-precision accumulate: float32
    # accumulation (the ~10x lever on consumer GPUs), float64 result.
    monkeypatch.setattr(cupy_sparse_module, "_GRAM_DENSE_MIN_DENSITY", 0.0)
    monkeypatch.setattr(cupy_sparse_module, "_GRAM_DENSE_CHUNK_ELEMENTS", 40)

    rng = np.random.default_rng(15)
    matrix = scipy_sparse.csr_matrix(rng.standard_normal((19, 5)) * 1e3)
    weights = rng.uniform(1e-3, 1e2, size=19)
    operator = _dense_gram_operator(cupy_sparse_module, matrix, weights)

    exact = np.asarray(operator.gram(weights))
    reduced = np.asarray(operator.gram(weights, accumulate_dtype="float32"))

    assert reduced.dtype == np.float64
    rel = np.max(np.abs(reduced - exact)) / np.max(np.abs(exact))
    assert 1e-12 < rel < 1e-4
    np.testing.assert_array_equal(reduced, reduced.T)
    assert operator._gram_compute_count == 2  # memo keyed by dtype


def test_cupy_gram_dense_strategy_negative_weights_fall_back(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mixed-sign weights cannot take the √w route; the general GEMM
    # accumulation must produce the same values with no syrk call.
    monkeypatch.setattr(cupy_sparse_module, "_GRAM_DENSE_MIN_DENSITY", 0.0)
    monkeypatch.setattr(cupy_sparse_module, "_GRAM_DENSE_CHUNK_ELEMENTS", 40)
    calls = _syrk_spy(cupy_sparse_module, monkeypatch)

    rng = np.random.default_rng(12)
    matrix = scipy_sparse.csr_matrix(rng.standard_normal((17, 5)))
    weights = rng.uniform(0.1, 2.0, size=17)
    weights[2] = -1.5
    operator = _dense_gram_operator(cupy_sparse_module, matrix, weights)

    actual = np.asarray(operator.gram(weights))

    assert calls == []
    dense = matrix.toarray()
    np.testing.assert_allclose(
        actual, dense.T @ (weights[:, None] * dense), rtol=1e-12, atol=1e-12
    )


def test_cupy_gram_dense_strategy_nonfinite_weights_fall_back(
    cupy_sparse_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # +inf passes ``w >= 0`` but must not take the √w/syrk path (different
    # overflow envelope) — same finiteness gate as the SciPy adapter.
    monkeypatch.setattr(cupy_sparse_module, "_GRAM_DENSE_MIN_DENSITY", 0.0)
    monkeypatch.setattr(cupy_sparse_module, "_GRAM_DENSE_CHUNK_ELEMENTS", 40)
    calls = _syrk_spy(cupy_sparse_module, monkeypatch)

    rng = np.random.default_rng(13)
    matrix = scipy_sparse.csr_matrix(rng.standard_normal((9, 4)))
    weights = rng.uniform(0.1, 2.0, size=9)
    weights[5] = np.inf
    operator = _dense_gram_operator(cupy_sparse_module, matrix, weights)

    actual = np.asarray(operator.gram(weights))

    assert calls == []
    dense = matrix.toarray()
    expected = dense.T @ (weights[:, None] * dense)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.isfinite(actual), finite)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-12)


def test_sparse_options_defaults_match_the_cudss_adapter_policy(cupy_sparse_module):
    # SparseOptions documents the adapter defaults as literals (the core may
    # not import an adapter); this pins the two against drift.
    from ipax.options import SparseOptions

    options = SparseOptions()
    assert (
        options.perturbed_pivot_refinement_steps
        == cupy_sparse_module._PERTURBED_MAX_REFINEMENT_STEPS
    )
    assert (
        options.perturbed_pivot_max_residual
        == cupy_sparse_module._PERTURBED_SOLVE_MAX_RESIDUAL
    )


def test_cupy_adapter_hook_sets_the_policy_of_later_solvers(
    cupy_sparse_module, monkeypatch
):
    # ``configure_perturbed_pivots`` is how SparseOptions reaches the adapter
    # (the SparseDirectSolver facade calls it once, before ``solver()``).
    fake = _RefiningFakeCudss(1, _SADDLE, _perturbation(1e-3))
    monkeypatch.setattr(cupy_sparse_module, "_load_cudss", lambda: fake)
    monkeypatch.setattr(cupy_sparse_module, "_ptr", lambda arr: int(arr.ctypes.data))
    adapter = cupy_sparse_module.CuPySparseAdapter()
    adapter.configure_perturbed_pivots(refinement_steps=0, max_residual=0.5)
    rows, cols = np.nonzero(np.ones((3, 3)))
    operator = adapter.from_coo(
        rows, cols, _SADDLE[rows, cols], shape=(3, 3), symmetric=True
    )
    solver = adapter.solver()
    solver.factor(operator)
    x = solver.solve(_SADDLE_RHS)
    inner = solver._inner
    assert (inner._max_refinement_steps, inner._perturbed_max_residual) == (0, 0.5)
    assert inner.last_refinement_steps == 0
    assert fake.solves == 1
    # Unrefined: the perturbed factor's inexact solve is returned as is.
    assert _relative_residual(_SADDLE, _SADDLE_RHS, x) > 1e-6


def test_cupy_adapter_hook_none_keeps_the_module_default(cupy_sparse_module):
    adapter = cupy_sparse_module.CuPySparseAdapter(
        max_refinement_steps=2, perturbed_max_residual=1e-5
    )
    adapter.configure_perturbed_pivots(refinement_steps=None, max_residual=None)
    assert (adapter._max_refinement_steps, adapter._perturbed_max_residual) == (
        cupy_sparse_module._PERTURBED_MAX_REFINEMENT_STEPS,
        cupy_sparse_module._PERTURBED_SOLVE_MAX_RESIDUAL,
    )


def test_cupy_adapter_hook_validates_the_policy(cupy_sparse_module):
    adapter = cupy_sparse_module.CuPySparseAdapter()
    with pytest.raises(ValueError, match="max_refinement_steps"):
        adapter.configure_perturbed_pivots(refinement_steps=-1, max_residual=None)
    with pytest.raises(ValueError, match="perturbed_max_residual"):
        adapter.configure_perturbed_pivots(refinement_steps=None, max_residual=0.0)
    # A rejected policy leaves the previous one in place.
    assert (adapter._max_refinement_steps, adapter._perturbed_max_residual) == (
        cupy_sparse_module._PERTURBED_MAX_REFINEMENT_STEPS,
        cupy_sparse_module._PERTURBED_SOLVE_MAX_RESIDUAL,
    )
