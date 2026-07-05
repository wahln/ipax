"""Unit tests for the dense symmetric-indefinite (LDLT) adapter dispatch."""

from __future__ import annotations

from ipax.backend.dense import get_dense_symmetric_indefinite_adapter
from ipax.testing.backends import import_namespace


def test_get_dense_symmetric_indefinite_adapter_numpy_returns_scipy_adapter():
    numpy = import_namespace("numpy")
    adapter = get_dense_symmetric_indefinite_adapter(numpy)
    assert adapter is not None
    assert adapter.__class__.__name__ == "ScipyLDLFactorization"


def test_get_dense_symmetric_indefinite_adapter_torch_returns_torch_adapter():
    try:
        torch = import_namespace("torch")
    except ImportError:
        return
    adapter = get_dense_symmetric_indefinite_adapter(torch)
    assert adapter is not None
    assert adapter.__class__.__name__ == "TorchLDLFactorization"


def test_get_dense_symmetric_indefinite_adapter_returns_none_for_unregistered_namespace():
    strict = import_namespace("array_api_strict")
    assert get_dense_symmetric_indefinite_adapter(strict) is None


def test_get_dense_symmetric_indefinite_adapter_cupy_returns_cupy_adapter():
    try:
        cupy = import_namespace("cupy")
    except ImportError:
        return
    adapter = get_dense_symmetric_indefinite_adapter(cupy)
    assert adapter is not None
    assert adapter.__class__.__name__ == "CuPyLDLFactorization"
