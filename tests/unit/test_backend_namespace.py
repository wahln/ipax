"""Unit tests for namespace resolution and capability probing."""

from __future__ import annotations

import types

import pytest

from ipax.backend.namespace import array_namespace, capabilities
from ipax.testing.backends import import_namespace, requested_backends
from tests._helpers import array, implemented


def test_requested_backends_round_trip_env(monkeypatch):
    monkeypatch.setenv("IPAX_BACKENDS", "numpy, torch, array_api_strict")
    assert requested_backends() == ("numpy", "torch", "array_api_strict")


def test_import_namespace_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        import_namespace("not-a-backend")


def test_array_namespace_resolves_backend(namespace):
    x = array(namespace, [1.0, 2.0])
    resolved = array_namespace(x)
    assert hasattr(resolved, "asarray")
    assert hasattr(resolved, "linalg")


def test_array_namespace_rejects_mixed_backends(all_available_backends):
    if len(all_available_backends) < 2:
        pytest.skip("needs at least two installed Array-API backends")

    xp_a = import_namespace(all_available_backends[0])
    xp_b = import_namespace(all_available_backends[1])
    with pytest.raises((TypeError, ValueError)):
        array_namespace(array(xp_a, [1.0]), array(xp_b, [1.0]))


def test_capabilities_reports_linalg_functions(namespace):
    with implemented("capabilities probe"):
        caps = capabilities(namespace)

    assert caps.has_linalg is hasattr(namespace, "linalg")
    assert "solve" in caps.linalg_functions
    assert "cholesky" in caps.linalg_functions
    assert "lstsq" not in caps.linalg_functions


@pytest.mark.parametrize(
    ("module_name", "expected"),
    [
        ("jax.numpy", "jax"),  # JAX resolves to the jax.numpy module
        ("array_api_compat.jax", "jax"),
        ("array_api_compat.jax.numpy", "jax"),
        ("array_api_compat.numpy", "numpy"),
        ("array_api_compat.torch", "torch"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("array_api_strict", "array_api_strict"),
    ],
)
def test_namespace_name_keys_on_leading_package(module_name, expected):
    """JAX (``jax.numpy``) must not be mislabeled as NumPy (regression)."""
    stub = types.ModuleType(module_name)
    caps = capabilities(stub)
    assert caps.name == expected


def test_capabilities_flags_jax_as_autodiff_not_sparse_numpy():
    """A jax.numpy namespace enables autodiff and is not treated as NumPy."""
    caps = capabilities(types.ModuleType("jax.numpy"))
    assert caps.name == "jax"
    assert caps.supports_autodiff is True
