"""Shared fixtures: the namespace fixture and the tolerance ladder (§8.2).

All numerical tests parametrize over ``namespace`` so each assertion runs on
every available backend. ``array-api-strict`` is the purity gate — it raises on
any out-of-standard call.
"""

from __future__ import annotations

import pytest

from ipax.testing.backends import import_namespace, requested_backends


def _available_backends() -> list[str]:
    names: list[str] = []
    for name in requested_backends():
        try:
            import_namespace(name)
        except ImportError:
            continue
        names.append(name)
    return names


@pytest.fixture(params=_available_backends(), ids=str)
def backend_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def namespace(backend_name: str):
    """An Array-API namespace module (numpy / torch / array_api_strict / ...)."""
    return import_namespace(backend_name)


@pytest.fixture
def all_available_backends() -> list[str]:
    return _available_backends()


@pytest.fixture
def tol(namespace) -> dict[str, float]:
    """Tolerance ladder keyed by dtype/device (relaxed for float32 backends)."""
    del namespace
    return {"rtol": 1e-9, "atol": 1e-9}
