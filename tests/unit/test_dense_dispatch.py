"""Unit tests for the dense symmetric-indefinite (LDLT) adapter dispatch."""

from __future__ import annotations

import sys
import types
from importlib.util import find_spec

import pytest

from ipax.backend.dense import (
    get_dense_cholesky_solve,
    get_dense_symmetric_indefinite_adapter,
)
from ipax.testing.backends import import_namespace


def test_get_dense_symmetric_indefinite_adapter_numpy_returns_scipy_adapter():
    pytest.importorskip("scipy")  # the NumPy adapter wraps scipy.linalg.ldl
    numpy = import_namespace("numpy")
    adapter = get_dense_symmetric_indefinite_adapter(numpy)
    assert adapter is not None
    assert adapter.__class__.__name__ == "ScipyLDLFactorization"


def test_get_dense_symmetric_indefinite_adapter_torch_returns_torch_adapter():
    pytest.importorskip("torch")
    torch = import_namespace("torch")
    adapter = get_dense_symmetric_indefinite_adapter(torch)
    assert adapter is not None
    assert adapter.__class__.__name__ == "TorchLDLFactorization"


def test_get_dense_symmetric_indefinite_adapter_returns_none_for_unregistered_namespace():
    strict = import_namespace("array_api_strict")
    assert get_dense_symmetric_indefinite_adapter(strict) is None


def test_get_dense_symmetric_indefinite_adapter_cupy_returns_cupy_adapter():
    # nvmath/cuSOLVER is loaded lazily at factor time, so an importable CuPy is
    # sufficient for dispatch to return the adapter object itself.
    pytest.importorskip("cupy")
    cupy = import_namespace("cupy")
    adapter = get_dense_symmetric_indefinite_adapter(cupy)
    assert adapter is not None
    assert adapter.__class__.__name__ == "CuPyLDLFactorization"


# --- ImportError degradation: dispatch returns None when the adapter module ---
# --- cannot be imported (missing SciPy / Torch / CuPy).                     ---
# A ``None`` entry in ``sys.modules`` makes the import system raise
# ``ImportError`` for that module, simulating the missing concrete library
# without uninstalling anything.


def test_dispatch_degrades_to_none_when_scipy_adapter_unimportable(monkeypatch):
    numpy = import_namespace("numpy")
    monkeypatch.setitem(sys.modules, "ipax.backend.dense.numpy_scipy", None)
    assert get_dense_symmetric_indefinite_adapter(numpy) is None


def test_dispatch_degrades_to_none_when_torch_adapter_unimportable(monkeypatch):
    pytest.importorskip("torch")
    torch = import_namespace("torch")
    monkeypatch.setitem(sys.modules, "ipax.backend.dense.torch", None)
    assert get_dense_symmetric_indefinite_adapter(torch) is None


def test_dispatch_degrades_to_none_when_cupy_adapter_unimportable(monkeypatch):
    # Dispatch keys on the namespace's ``__name__`` only, so a stub namespace
    # exercises the cupy branch without CuPy installed.
    fake_namespace = types.SimpleNamespace(__name__="cupy")
    monkeypatch.setitem(sys.modules, "ipax.backend.dense.cupy", None)
    assert get_dense_symmetric_indefinite_adapter(fake_namespace) is None


def test_dispatch_cupy_branch_reaches_adapter_without_real_cupy(monkeypatch):
    # The adapter module's only top-level concrete import is ``cupy`` itself
    # (nvmath is lazy), so an empty stub module suffices to import it where
    # CuPy is absent (CI) — the real-CuPy case is covered above.
    stubbed = "cupy" not in sys.modules and find_spec("cupy") is None
    if stubbed:
        monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    fake_namespace = types.SimpleNamespace(__name__="cupy")
    try:
        adapter = get_dense_symmetric_indefinite_adapter(fake_namespace)
        assert adapter is not None
        assert adapter.__class__.__name__ == "CuPyLDLFactorization"
    finally:
        if stubbed:
            # Drop the adapter module bound against the stub so later imports
            # (in environments that gain a real CuPy) start fresh.
            sys.modules.pop("ipax.backend.dense.cupy", None)


# --- the Cholesky back-substitution gap-filler shares the dispatch shape -----


def test_cholesky_solve_dispatch_degrades_when_scipy_unimportable(monkeypatch):
    numpy = import_namespace("numpy")
    monkeypatch.setitem(sys.modules, "ipax.backend.dense.numpy_scipy", None)
    assert get_dense_cholesky_solve(numpy) is None


def test_cholesky_solve_dispatch_degrades_when_torch_unimportable(monkeypatch):
    pytest.importorskip("torch")
    torch = import_namespace("torch")
    monkeypatch.setitem(sys.modules, "ipax.backend.dense.torch", None)
    assert get_dense_cholesky_solve(torch) is None


def test_cholesky_solve_dispatch_degrades_when_cupy_unimportable(monkeypatch):
    fake_namespace = types.SimpleNamespace(__name__="cupy")
    monkeypatch.setitem(sys.modules, "ipax.backend.dense.cupy", None)
    assert get_dense_cholesky_solve(fake_namespace) is None


def test_cholesky_solve_dispatch_cupy_branch_returns_the_adapter(monkeypatch):
    # Mirrors the LDL^T stub trick above: dispatch only needs the adapter
    # module importable, so the cupy branch is exercised without a GPU.
    stubbed = "cupy" not in sys.modules and find_spec("cupy") is None
    if stubbed:
        monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    fake_namespace = types.SimpleNamespace(__name__="cupy")
    try:
        solve_fn = get_dense_cholesky_solve(fake_namespace)
        assert solve_fn is not None
        assert solve_fn.__name__ == "cholesky_solve"
    finally:
        if stubbed:
            sys.modules.pop("ipax.backend.dense.cupy", None)
