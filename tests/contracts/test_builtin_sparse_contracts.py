"""Contract coverage for the built-in ``SparseDirectSolver`` adapters."""

from __future__ import annotations

import pytest

from ipax.backend.sparse import get_sparse_adapter
from tests.contracts.test_sparse_contract import SparseDirectSolverContract

pytestmark = pytest.mark.sparse


class TestSciPySparseAdapter(SparseDirectSolverContract):
    """Run the shared battery against the NumPy/SciPy CPU adapter."""

    implementation_reason = "SciPy sparse adapter"

    def make_adapter(self, namespace):
        adapter = get_sparse_adapter(namespace)
        if adapter is None:
            pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")
        return adapter
