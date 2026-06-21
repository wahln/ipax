"""Unit tests for sparse adapter dispatch."""

from __future__ import annotations

from ipax.backend.sparse import get_sparse_adapter
from tests._helpers import implemented


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
