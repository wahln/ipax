"""``gram_fill_estimate`` — the cheap Gram fill-in probe for NE auto-selection.

The sparse normal-equations route wins only when the Gram pattern ``AᵀA``
stays sparse (localized/banded rows); on scattered sparsity it fills in
catastrophically. ``gram_fill_estimate`` answers "how dense is the Gram
pattern?" *without* forming the Gram: the nonzero pattern of Gram column
``j`` is exactly the union of the column patterns of every row touching
``j``, so sampling columns and averaging the union sizes estimates
``nnz(AᵀA)/n²``. These tests pin the estimator against the exact fill
computed from ``gram_coo``.
"""

from __future__ import annotations

import pytest

from ipax.backend.operators import COOOperator, Dense, VStack
from ipax.backend.sparse import get_sparse_adapter
from tests._helpers import array

pytestmark = pytest.mark.sparse


def _require_sparse(namespace):
    if get_sparse_adapter(namespace) is None:
        pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")


def _banded_coo(namespace, m: int, n: int, width: int = 2):
    """Tall banded m×n: row i hits ``width`` columns near i·n/m (sparse Gram)."""
    rows_l, cols_l, vals_l = [], [], []
    for i in range(m):
        center = (i * n) // m
        for k in range(width):
            j = min(n - 1, center + k)
            rows_l.append(i)
            cols_l.append(j)
            vals_l.append(1.0 + 0.1 * ((i + k) % 5))
    return (
        namespace.asarray(rows_l),
        namespace.asarray(cols_l),
        array(namespace, vals_l),
    )


def _scattered_coo(namespace, m: int, n: int, per_row: int = 4):
    """Tall m×n with pseudo-random column scatter per row (Gram fills in)."""
    rows_l, cols_l, vals_l = [], [], []
    state = 1
    for i in range(m):
        for k in range(per_row):
            state = (state * 1103515245 + 12345) % (2**31)
            rows_l.append(i)
            # High bits: an LCG's low bits cycle with a short period.
            cols_l.append((state >> 16) % n)
            vals_l.append(1.0 + 0.01 * ((i + k) % 7))
    return (
        namespace.asarray(rows_l),
        namespace.asarray(cols_l),
        array(namespace, vals_l),
    )


def _exact_gram_fill(op, namespace, m: int, n: int) -> float:
    """Exact ``nnz(AᵀA)/n²`` from the gram_coo pattern (unique pairs)."""
    weights = namespace.ones((m,), dtype=array(namespace, [0.0]).dtype)
    grows, gcols, _, _ = op.gram_coo(weights)
    pairs = {(int(grows[k]), int(gcols[k])) for k in range(int(grows.shape[0]))}
    return len(pairs) / (n * n)


def test_banded_estimate_tracks_exact_fill(namespace):
    _require_sparse(namespace)
    m, n = 640, 64
    rows, cols, vals = _banded_coo(namespace, m, n)
    op = COOOperator(rows, cols, vals, (m, n), pattern_key="banded")

    estimate = op.gram_fill_estimate()

    assert estimate is not None
    exact = _exact_gram_fill(op, namespace, m, n)
    assert exact < 0.15  # the banded Gram really is sparse
    assert 0.5 * exact <= estimate <= 2.0 * exact


def test_scattered_estimate_reports_dense_gram(namespace):
    _require_sparse(namespace)
    m, n = 640, 32
    rows, cols, vals = _scattered_coo(namespace, m, n)
    op = COOOperator(rows, cols, vals, (m, n), pattern_key="scatter")

    estimate = op.gram_fill_estimate()

    assert estimate is not None
    exact = _exact_gram_fill(op, namespace, m, n)
    assert exact > 0.5  # scattered rows saturate the Gram
    assert estimate >= 0.5 * exact


def test_dense_operator_reports_unknown_fill(namespace):
    op = Dense(array(namespace, [[1.0, 0.0], [0.0, 1.0]]))
    assert op.gram_fill_estimate() is None


def test_vstack_estimate_bounds_the_union(namespace):
    _require_sparse(namespace)
    m, n = 320, 32
    r1, c1, v1 = _banded_coo(namespace, m, n)
    top = COOOperator(r1, c1, v1, (m, n), pattern_key="b1")
    bottom = COOOperator(r1, c1, v1, (m, n), pattern_key="b2")
    stacked = VStack((top, bottom))

    e_top = top.gram_fill_estimate()
    estimate = stacked.gram_fill_estimate()

    assert e_top is not None and estimate is not None
    # The stacked Gram pattern is the union of the block patterns: at least
    # one block's fill, at most the (capped) sum.
    assert e_top <= estimate <= min(1.0, 2.0 * e_top) + 1e-12


def test_vstack_with_unknown_block_reports_unknown(namespace):
    _require_sparse(namespace)
    m, n = 32, 8
    rows, cols, vals = _banded_coo(namespace, m, n)
    coo = COOOperator(rows, cols, vals, (m, n), pattern_key="b")
    dense = Dense(namespace.ones((4, n), dtype=vals.dtype))
    assert VStack((coo, dense)).gram_fill_estimate() is None


def test_row_scaled_wrapper_forwards_estimate(namespace):
    # Gradient-based scaling wraps ∇g in a row-scaling operator; the scaling
    # is pattern-preserving, so the probe must survive it unchanged.
    _require_sparse(namespace)
    from ipax.problem.scaling import _RowScaled

    m, n = 320, 32
    rows, cols, vals = _banded_coo(namespace, m, n)
    op = COOOperator(rows, cols, vals, (m, n), pattern_key="banded")
    d = namespace.full((m,), 0.5, dtype=vals.dtype)

    assert _RowScaled(op, d).gram_fill_estimate() == op.gram_fill_estimate()


def test_empty_operator_reports_zero_fill(namespace):
    _require_sparse(namespace)
    dtype = array(namespace, [0.0]).dtype
    op = COOOperator(
        namespace.zeros((0,), dtype=namespace.int64),
        namespace.zeros((0,), dtype=namespace.int64),
        namespace.zeros((0,), dtype=dtype),
        (0, 4),
        pattern_key="empty",
    )
    assert op.gram_fill_estimate() == 0.0
