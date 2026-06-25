"""``SparseOperator`` symmetry verdict and lazy matvec-format behavior.

These cover the per-iteration cost trims on the sparse-direct route: the operator
honors a structural symmetry *hint* (set by the KKT assembler, which knows the
matrix is symmetric by construction) instead of always running the O(nnz)
``A - Aᵀ`` numerical test, and it only materializes the CSR matvec format on demand
(the direct-solve route consumes CSC, never CSR).
"""

from __future__ import annotations

import pytest

from ipax.backend.sparse import get_sparse_adapter
from tests._helpers import array, assert_allclose

pytestmark = pytest.mark.sparse


def _adapter(namespace):
    adapter = get_sparse_adapter(namespace)
    if adapter is None:
        pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")
    return adapter


def _symmetric_op(namespace, *, symmetric=None):
    # [[2, 1], [1, 3]] — numerically symmetric.
    return _adapter(namespace).from_coo(
        namespace.asarray([0, 0, 1, 1]),
        namespace.asarray([0, 1, 0, 1]),
        array(namespace, [2.0, 1.0, 1.0, 3.0]),
        shape=(2, 2),
        symmetric=symmetric,
    )


def _asymmetric_op(namespace, *, symmetric=None):
    # [[2, 1], [0, 3]] — not symmetric.
    return _adapter(namespace).from_coo(
        namespace.asarray([0, 0, 1]),
        namespace.asarray([0, 1, 1]),
        array(namespace, [2.0, 1.0, 3.0]),
        shape=(2, 2),
        symmetric=symmetric,
    )


def test_is_symmetric_numerical_when_no_hint(namespace):
    assert _symmetric_op(namespace).is_symmetric() is True
    assert _asymmetric_op(namespace).is_symmetric() is False


def test_is_symmetric_honors_hint_over_numerical_verdict(namespace):
    # The hint is authoritative: an explicit False suppresses the (otherwise True)
    # numerical verdict, proving the numerical path is not consulted when hinted.
    assert _symmetric_op(namespace, symmetric=False).is_symmetric() is False
    assert _asymmetric_op(namespace, symmetric=True).is_symmetric() is True


def test_lazy_csr_matvec_matches_dense(namespace, tol):
    op = _symmetric_op(namespace)
    v = array(namespace, [1.0, -2.0])
    assert_allclose(namespace, op.matvec(v), array(namespace, [0.0, -5.0]), **tol)
