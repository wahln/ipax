"""Unit tests for DLPack device-routed sparse dispatch (Torch/JAX reuse).

Torch and JAX have no native sparse-direct path; the device-routing adapter
reinterprets their COO buffers onto the SciPy (CPU) and CuPy/cuDSS (CUDA)
adapters by ``__dlpack_device__``. These tests pin the routing decision and the
namespace round-trip without requiring a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from ipax.backend.namespace import _namespace_name
from ipax.backend.sparse import get_sparse_adapter
from ipax.backend.sparse._routing import (
    DeviceRoutingSparseAdapter,
    _dlpack_device_kind,
    to_xp_array,
)


class _FakeDeviceArray:
    """Minimal array stand-in exposing only a DLPack device kind."""

    def __init__(self, kind: int) -> None:
        self._kind = kind

    def __dlpack_device__(self) -> tuple[int, int]:
        return (self._kind, 0)


def test_torch_and_jax_dispatch_to_routing_adapter():
    import ipax.testing.backends as backends

    for name in ("torch", "jax"):
        try:
            xp = backends.import_namespace(name)
        except ImportError:
            continue
        assert isinstance(get_sparse_adapter(xp), DeviceRoutingSparseAdapter)


def test_dlpack_device_kind_reads_cpu_for_numpy():
    assert _dlpack_device_kind(np.arange(3.0)) == 1


def test_routing_cuda_without_cupy_errors_clearly():
    # No CUDA array library in CI: a CUDA-device buffer must fail with guidance,
    # not silently fall back to a wrong (host) factorization.
    try:
        import cupy  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("CuPy is installed; the missing-library branch is unreachable")

    adapter = DeviceRoutingSparseAdapter()
    cuda = _FakeDeviceArray(2)
    with pytest.raises(RuntimeError, match="CuPy"):
        adapter.from_coo(cuda, cuda, cuda, shape=(1, 1))


def test_routing_unknown_device_errors_with_kind():
    adapter = DeviceRoutingSparseAdapter()
    rocm = _FakeDeviceArray(10)  # DLPack ROCm: no direct solver
    with pytest.raises(RuntimeError, match="device kind 10"):
        adapter.from_coo(rocm, rocm, rocm, shape=(1, 1))


def test_solver_before_from_coo_errors():
    with pytest.raises(RuntimeError, match="from_coo"):
        DeviceRoutingSparseAdapter().solver()


def test_routing_reuses_delegate_across_calls_on_same_device():
    # The delegate adapter must persist across iterations so its compiled
    # COO→canonical map cache (values-only refactor fast path) survives; a fresh
    # delegate every call would silently disable that reuse for Torch/JAX.
    adapter = DeviceRoutingSparseAdapter()
    rows = np.asarray([0, 1])
    cols = np.asarray([0, 1])
    adapter.from_coo(rows, cols, np.asarray([1.0, 2.0]), shape=(2, 2))
    first_delegate = adapter._delegate
    adapter.from_coo(rows, cols, np.asarray([3.0, 4.0]), shape=(2, 2))

    assert adapter._delegate is first_delegate


def test_to_xp_array_round_trips_into_namespace(namespace):
    # ``to_xp_array`` is what returns SciPy/CuPy results to the caller's
    # namespace; for any backend with a sparse adapter it must yield an array of
    # that namespace carrying the same values.
    if get_sparse_adapter(namespace) is None:
        pytest.skip(f"no sparse adapter for {namespace.__name__!r}")
    source = np.asarray([1.0, 2.0, 3.0])
    out = to_xp_array(source, namespace)
    from ipax.backend.namespace import array_namespace

    assert _namespace_name(array_namespace(out)) == _namespace_name(namespace)
    assert [float(v) for v in out] == [1.0, 2.0, 3.0]


def test_sparse_operator_results_stay_in_caller_namespace(namespace):
    # A Torch/JAX COO triplet routed onto SciPy/cuDSS must hand matvec results
    # back in the *original* namespace, not NumPy/CuPy.
    adapter = get_sparse_adapter(namespace)
    if adapter is None:
        pytest.skip(f"no sparse adapter for {namespace.__name__!r}")
    from ipax.backend.namespace import array_namespace
    from tests._helpers import array

    rows = namespace.asarray([0, 1])
    cols = namespace.asarray([0, 1])
    values = array(namespace, [2.0, 3.0])
    operator = adapter.from_coo(rows, cols, values, shape=(2, 2))
    out = operator.matvec(array(namespace, [1.0, 1.0]))

    assert _namespace_name(array_namespace(out)) == _namespace_name(namespace)
    assert [float(v) for v in out] == [2.0, 3.0]
