"""Sparse condensed normal-equations route: gram_coo + solver form.

The condensed Newton matrix ``N = W + Σ_x + δ_w I + ∇gᵀ Σ_s ∇g`` stays sparse
when the inequality Jacobian has *localized* structure (banded/block rows —
e.g. dose-influence at clinic scale), so for tall ``n ≫ 2e4`` problems a
sparse ``n×n`` factorization of ``N`` beats both the dense condensed route
(O(n²) memory) and matrix-free Krylov (whose iteration count blows up on the
ill-conditioned late-IPM Σ). The route is **opt-in**
(``SparseOptions(kkt_route="normal_equations")``): on non-localized sparsity
``AᵀA`` fills in catastrophically, and no cheap structural probe can see that
in advance.
"""

from __future__ import annotations

import pytest

from ipax.backend.operators import COOOperator, Dense, Diagonal, VStack
from ipax.backend.sparse import get_sparse_adapter
from ipax.ipm.kkt import build_condensed_operator
from ipax.linalg.regularize import RegularizationState
from ipax.linalg.sparse import SparseDirectSolver
from ipax.options import Options, SparseOptions
from tests._helpers import array, assert_allclose

pytestmark = pytest.mark.sparse


def _require_sparse(namespace):
    if get_sparse_adapter(namespace) is None:
        pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")


def _banded_coo(namespace, m: int, n: int):
    """A tall banded m×n matrix: row i hits columns near i·n/m (sparse Gram)."""
    rows_l, cols_l, vals_l = [], [], []
    for i in range(m):
        center = (i * n) // m
        for k in range(2):
            j = min(n - 1, center + k)
            rows_l.append(i)
            cols_l.append(j)
            vals_l.append(1.0 + 0.1 * ((i + k) % 5))
    rows = namespace.asarray(rows_l)
    cols = namespace.asarray(cols_l)
    vals = array(namespace, vals_l)
    return rows, cols, vals


def _dense_reference(namespace, rows, cols, vals, m, n, weights):
    # Densify the triplets through the operator itself (identity probe), then
    # form Aᵀ diag(w) A densely as the reference.
    op = COOOperator(rows, cols, vals, (m, n))
    a = op.matmat(namespace.eye(n, dtype=vals.dtype))
    wa = namespace.expand_dims(weights, axis=1) * a
    return namespace.matmul(namespace.permute_dims(a, (1, 0)), wa)


def test_gram_coo_matches_dense_gram(namespace):
    _require_sparse(namespace)
    m, n = 12, 5
    rows, cols, vals = _banded_coo(namespace, m, n)
    op = COOOperator(rows, cols, vals, (m, n), pattern_key="banded")
    weights = array(namespace, [float(2 + (i % 3)) for i in range(m)])

    grows, gcols, gvals, gshape = op.gram_coo(weights)
    assert gshape == (n, n)
    ref = _dense_reference(namespace, rows, cols, vals, m, n, weights)
    # Scatter the triplets back to dense through a COO operator's matmat.
    gop = COOOperator(grows, gcols, gvals, gshape)
    dense = gop.matmat(namespace.eye(n, dtype=vals.dtype))
    assert_allclose(namespace, dense, ref, atol=1e-12)


def test_gram_coo_pattern_is_stable_across_weight_changes(namespace):
    _require_sparse(namespace)
    m, n = 12, 5
    rows, cols, vals = _banded_coo(namespace, m, n)
    op = COOOperator(rows, cols, vals, (m, n), pattern_key="banded")
    w1 = array(namespace, [1.0] * m)
    w2 = array(namespace, [float(1 + i) for i in range(m)])

    r1, c1, v1, _ = op.gram_coo(w1)
    r2, c2, v2, _ = op.gram_coo(w2)
    assert v1.shape == v2.shape
    assert bool(namespace.all(r1 == r2))
    assert bool(namespace.all(c1 == c2))


def test_gram_coo_capable_probes(namespace):
    _require_sparse(namespace)
    m, n = 12, 5
    rows, cols, vals = _banded_coo(namespace, m, n)
    sparse_op = COOOperator(rows, cols, vals, (m, n))
    assert sparse_op.gram_coo_capable()
    dense_op = Dense(namespace.zeros((m, n), dtype=vals.dtype))
    assert not dense_op.gram_coo_capable()


def test_vstack_gram_coo_sums_the_blocks(namespace):
    _require_sparse(namespace)
    m, n = 12, 5
    rows, cols, vals = _banded_coo(namespace, m, n)
    block = COOOperator(rows, cols, vals, (m, n), pattern_key="banded")
    stacked = VStack((block, block))
    weights = array(namespace, [1.0] * (2 * m))

    grows, gcols, gvals, gshape = stacked.gram_coo(weights)
    ref = _dense_reference(
        namespace, rows, cols, vals, m, n, array(namespace, [1.0] * m)
    )
    gop = COOOperator(grows, gcols, gvals, gshape)
    dense = gop.matmat(namespace.eye(n, dtype=vals.dtype))
    assert_allclose(namespace, dense, 2.0 * ref, atol=1e-12)


def _condensed_sparse_system(namespace):
    """Condensed operator whose W and ∇g are sparse operators (NE-capable)."""
    n = 4
    m = 8
    w_rows = namespace.asarray([0, 1, 2, 3, 0, 1])
    w_cols = namespace.asarray([0, 1, 2, 3, 1, 0])
    w_vals = array(namespace, [4.0, 3.0, 2.5, 2.0, 0.5, 0.5])
    W = COOOperator(w_rows, w_cols, w_vals, (n, n), symmetric=True, pattern_key="W")
    rows, cols, vals = _banded_coo(namespace, m, n)
    jac = COOOperator(rows, cols, vals, (m, n), pattern_key="G")
    op = build_condensed_operator(
        W,
        Diagonal(array(namespace, [0.25, 0.75, 0.5, 1.0])),
        Diagonal(array(namespace, [float(1 + (i % 4)) for i in range(m)])),
        jac,
        RegularizationState(delta_w=1e-6),
    )
    x_exact = array(namespace, [1.0, -2.0, 0.5, 3.0])
    rhs = op.matvec(x_exact)
    return op, rhs, x_exact


def test_normal_equations_solver_solves_the_condensed_system(namespace):
    _require_sparse(namespace)
    op, rhs, x_exact = _condensed_sparse_system(namespace)
    solver = SparseDirectSolver(form="normal_equations")
    solver.factor(op)
    x = solver.solve(rhs)
    assert_allclose(namespace, x, x_exact, atol=1e-8)
    # Refactor with fresh values (same pattern) — the cached-structure path.
    solver.factor(op)
    assert_allclose(namespace, solver.solve(rhs), x_exact, atol=1e-8)


def test_normal_equations_form_requires_a_capable_operator(namespace):
    _require_sparse(namespace)
    # A dense (non-gram_coo) inequality Jacobian cannot feed the NE form.
    op = build_condensed_operator(
        Dense(array(namespace, [[4.0, 0.5], [0.5, 3.0]])),
        Diagonal(array(namespace, [0.25, 0.75])),
        Diagonal(array(namespace, [2.0, 0.5, 1.0])),
        Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5], [0.3, 0.1]])),
        RegularizationState(delta_w=1e-6),
    )
    solver = SparseDirectSolver(form="normal_equations")
    with pytest.raises(RuntimeError, match=r"normal.equations"):
        solver.factor(op)


def test_sparse_options_validate_the_route():
    assert SparseOptions().kkt_route == "augmented"
    assert Options(sparse=SparseOptions(kkt_route="normal_equations"))
    with pytest.raises(ValueError, match="kkt_route"):
        SparseOptions(kkt_route="bogus")  # type: ignore[arg-type]
