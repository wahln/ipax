"""Unit tests for sparse adapter dispatch."""

from __future__ import annotations

import pytest

from ipax.backend.operators import Dense
from ipax.backend.sparse import get_sparse_adapter
from ipax.linalg.sparse import SparseDirectSolver
from ipax.testing.backends import import_namespace
from tests._helpers import array, implemented


def test_get_sparse_adapter_returns_none_for_unregistered_namespace(namespace):
    with implemented("sparse dispatch"):
        adapter = get_sparse_adapter(namespace)

    if getattr(namespace, "__name__", "") not in {
        "array_api_compat.numpy",
        "array_api_compat.torch",
        "array_api_compat.cupy",
        "array_api_compat.jax",
    }:
        assert adapter is None


def test_sparse_direct_solver_rejects_cached_adapter_backend_change(monkeypatch):
    class FakeInner:
        def factor(self, operator):
            del operator

    class FakeAdapter:
        def from_coo(self, rows, cols, values, *, shape, symmetric, pattern_signature):
            del rows, cols, values, shape, symmetric, pattern_signature
            return object()

        def solver(self, *, require_inertia=False):
            del require_inertia
            return FakeInner()

    import ipax.backend.sparse as sparse_module

    monkeypatch.setattr(sparse_module, "get_sparse_adapter", lambda xp: FakeAdapter())
    numpy = import_namespace("numpy")
    try:
        torch = import_namespace("torch")
    except ImportError as exc:
        pytest.skip(f"torch namespace unavailable: {exc}")

    solver = SparseDirectSolver()
    solver.factor(Dense(array(numpy, [[1.0]])))

    with pytest.raises(RuntimeError, match="cached sparse adapter"):
        solver.factor(Dense(array(torch, [[1.0]])))


def test_sparse_direct_solver_errors_without_adapter(monkeypatch):
    import ipax.backend.sparse as sparse_module

    monkeypatch.setattr(sparse_module, "get_sparse_adapter", lambda xp: None)
    numpy = import_namespace("numpy")

    solver = SparseDirectSolver()
    with pytest.raises(RuntimeError, match="no sparse-direct adapter"):
        solver.factor(Dense(array(numpy, [[1.0]])))


def test_dlpack_device_kind_tolerates_missing_or_failing_dlpack():
    from ipax.linalg.sparse import _dlpack_device_kind

    # No __dlpack_device__ at all: unknown device, cache key falls back to None.
    assert _dlpack_device_kind(object()) is None

    class _Boom:
        def __dlpack_device__(self):
            raise RuntimeError("device query failed")

    assert _dlpack_device_kind(_Boom()) is None
