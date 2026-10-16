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


def _cupy_cuda_available() -> bool:
    """Whether CuPy is importable and sees at least one CUDA device.

    Mirrors the gating in ``tests/unit/test_cudss_refinement_gpu.py``; kept as
    a plain predicate (not a module-level ``importorskip``) because this
    module also hosts the CPU-only :class:`TestSciPySparseAdapter`.
    """
    try:
        import cupy
    except ImportError:
        return False
    try:
        return bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


@pytest.mark.gpu
class TestCuPySparseAdapter(SparseDirectSolverContract):
    """Run the shared battery against the CuPy/cuDSS GPU adapter.

    Unlike :class:`TestSciPySparseAdapter` this does not depend on
    ``IPAX_BACKENDS`` listing ``cupy`` — it targets the CuPy adapter directly
    so the battery (including the optional ``configure_perturbed_pivots``
    hook case) always runs when CuPy and a CUDA device are present, and
    skips cleanly otherwise (GPU-gating pattern of
    ``tests/unit/test_cudss_refinement_gpu.py`` /
    ``tests/unit/test_cupy_sparse_adapter.py``). This is the only adapter
    where the perturbed-pivot hook does anything, so this is the only place
    that case is actually exercised rather than skipped.
    """

    implementation_reason = "CuPy sparse adapter"

    @pytest.fixture
    def namespace(self):
        if not _cupy_cuda_available():
            pytest.skip("CuPy / CUDA is not available")
        from ipax.testing.backends import import_namespace

        return import_namespace("cupy")

    def make_adapter(self, namespace):
        adapter = get_sparse_adapter(namespace)
        if adapter is None:
            pytest.skip("no sparse adapter for backend 'cupy'")
        return adapter
