"""``to_coo`` structure emission round-trips to the dense matrix (invariant #4).

Pure Array API, so these run on every backend (including the ``array-api-strict``
purity gate). The COO triplets are reconstructed into a dense array with plain
Python arithmetic — duplicate ``(row, col)`` entries sum, matching the adapter.
"""

from __future__ import annotations

import pytest

from ipax.backend.operators import Dense, Diagonal, Identity
from ipax.ipm.hessian import LBFGSOperator
from ipax.ipm.kkt import build_condensed_operator, build_saddle_operator
from ipax.linalg.regularize import RegularizationState
from ipax.options import LBFGSOptions
from tests._helpers import array, assert_allclose, float_dtype


def _coo_to_dense(namespace, rows, cols, values, shape):
    m, n = shape
    grid = [[0.0] * n for _ in range(m)]
    nnz = int(rows.shape[0])
    for k in range(nnz):
        grid[int(rows[k])][int(cols[k])] += float(values[k])
    return array(namespace, grid)


def _identity_dense(namespace, op):
    n = op.shape[1]
    columns = [
        op.matvec(array(namespace, [1.0 if k == j else 0.0 for k in range(n)]))
        for j in range(n)
    ]
    return namespace.stack(columns, axis=1)


def test_dense_to_coo_round_trips(namespace, tol):
    A = array(namespace, [[2.0, -1.0, 0.5], [0.0, 3.0, 4.0]])
    op = Dense(A)
    rows, cols, values, shape = op.to_coo()
    assert shape == (2, 3)
    assert_allclose(
        namespace, _coo_to_dense(namespace, rows, cols, values, shape), A, **tol
    )


def test_diagonal_to_coo_round_trips(namespace, tol):
    d = array(namespace, [2.0, -3.0, 4.0])
    rows, cols, values, shape = Diagonal(d).to_coo()
    expected = array(namespace, [[2.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, 4.0]])
    assert_allclose(
        namespace, _coo_to_dense(namespace, rows, cols, values, shape), expected, **tol
    )


def test_identity_to_coo_requires_template(namespace, tol):
    op = Identity(3)
    with pytest.raises(NotImplementedError):
        op.to_coo()
    like = array(namespace, [0.0, 0.0, 0.0])
    rows, cols, values, shape = op.to_coo(like)
    expected = namespace.eye(3, dtype=like.dtype)
    assert_allclose(
        namespace, _coo_to_dense(namespace, rows, cols, values, shape), expected, **tol
    )


def test_condensed_operator_to_coo_matches_matvec(namespace, tol):
    # Bound/equality block: N = W + Σ_x + δ_w I (no inequalities ⇒ assemblable).
    w = Dense(array(namespace, [[3.0, 1.0], [1.0, 2.0]]))
    sigma_x = Diagonal(array(namespace, [0.5, 0.25]))
    sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, 2), dtype=float_dtype(namespace)))
    reg = RegularizationState(delta_w=1e-3)
    op = build_condensed_operator(w, sigma_x, sigma_s, empty_jac, reg)

    rows, cols, values, shape = op.to_coo()
    assert_allclose(
        namespace,
        _coo_to_dense(namespace, rows, cols, values, shape),
        _identity_dense(namespace, op),
        **tol,
    )


def test_condensed_operator_to_coo_emits_inequality_border(namespace, tol):
    # Inequalities are kept explicit as the −Σ_s⁻¹ augmented border (no dense
    # ∇gᵀΣ_s∇g product). Solving the bordered system and dropping the auxiliary
    # slack-multiplier rows must reproduce the dense condensed Newton step.
    n = 3
    w = Dense(array(namespace, [[2.0, 0.5, 0.0], [0.5, 3.0, 1.0], [0.0, 1.0, 2.0]]))
    sigma_x = Diagonal(array(namespace, [0.4, 0.2, 0.6]))
    sigma_s = Diagonal(array(namespace, [1.5, 2.5]))
    ineq_jac = Dense(array(namespace, [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]))
    op = build_condensed_operator(
        w, sigma_x, sigma_s, ineq_jac, RegularizationState(delta_w=1e-2)
    )

    dense_n = _identity_dense(namespace, op)  # exact N incl. ∇gᵀ Σ_s ∇g
    rows, cols, values, shape = op.to_coo()
    assert shape[0] == n + 2  # two slack-multiplier border rows appended
    bordered = _coo_to_dense(namespace, rows, cols, values, shape)
    assert_allclose(
        namespace, bordered, namespace.permute_dims(bordered, (1, 0)), **tol
    )

    rhs = array(namespace, [1.0, -2.0, 0.5])
    pad = namespace.zeros((shape[0] - n,), dtype=rhs.dtype)
    dx = namespace.linalg.solve(bordered, namespace.concat((rhs, pad)))[:n]
    expected = namespace.linalg.solve(dense_n, rhs)
    assert_allclose(namespace, dx, expected, **tol)


def _lbfgs_with_pairs(namespace, n):
    lbfgs = LBFGSOperator(n, LBFGSOptions(memory=5))
    lbfgs.update(
        array(namespace, [0.5, -0.2, 0.3, 0.1]),
        array(namespace, [0.6, 0.1, 0.4, 0.2]),
    )
    lbfgs.update(
        array(namespace, [0.1, 0.4, -0.1, 0.2]),
        array(namespace, [0.2, 0.5, 0.1, 0.3]),
    )
    return lbfgs


def test_condensed_lbfgs_emits_low_rank_border(namespace, tol):
    # IPOPT limited-memory trick: the dense −U M⁻¹ Uᵀ term becomes a thin border,
    # never densifying the (1,1) block. Solving the bordered system and dropping
    # the auxiliary rows must reproduce the dense condensed Newton step.
    n = 4
    lbfgs = _lbfgs_with_pairs(namespace, n)
    sigma_x = Diagonal(array(namespace, [0.7, 0.3, 0.5, 0.9]))
    sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, n), dtype=float_dtype(namespace)))
    op = build_condensed_operator(
        lbfgs, sigma_x, sigma_s, empty_jac, RegularizationState(delta_w=1e-2)
    )

    # op.matvec is exact, so this is the true dense condensed matrix N.
    dense_n = _identity_dense(namespace, op)
    rows, cols, values, shape = op.to_coo()
    assert shape[0] > n  # a low-rank border was appended
    bordered = _coo_to_dense(namespace, rows, cols, values, shape)
    # The bordered matrix is symmetric (so the Feral LDLᵀ path engages).
    assert_allclose(
        namespace, bordered, namespace.permute_dims(bordered, (1, 0)), **tol
    )

    rhs = array(namespace, [1.0, -2.0, 0.5, 3.0])
    pad = namespace.zeros((shape[0] - n,), dtype=rhs.dtype)
    dx = namespace.linalg.solve(bordered, namespace.concat((rhs, pad)))[:n]
    expected = namespace.linalg.solve(dense_n, rhs)
    assert_allclose(namespace, dx, expected, **tol)


def test_condensed_diag_plus_low_rank_emits_border(namespace, tol):
    # A matrix-free diagonal+low-rank Hessian W = diag(h) + C Cᵀ factors through
    # the same low-rank border as L-BFGS (the generic diagonal_low_rank_form hook).
    from ipax.testing.problems import _DiagPlusLowRank

    n = 4
    h = array(namespace, [1.0, 2.0, 1.5, 3.0])
    c = array(namespace, [[0.5, 0.0], [0.2, 0.4], [0.0, 0.3], [0.1, 0.1]])
    w = _DiagPlusLowRank(h, c, sigma=1.0)
    sigma_x = Diagonal(array(namespace, [0.3, 0.2, 0.1, 0.4]))
    sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, n), dtype=float_dtype(namespace)))
    op = build_condensed_operator(
        w, sigma_x, sigma_s, empty_jac, RegularizationState(delta_w=1e-3)
    )

    dense_n = _identity_dense(namespace, op)  # exact N incl. the dense C Cᵀ term
    rows, cols, values, shape = op.to_coo()
    assert shape[0] == n + 2  # rank-2 low-rank border
    bordered = _coo_to_dense(namespace, rows, cols, values, shape)

    rhs = array(namespace, [1.0, -1.0, 0.5, 2.0])
    pad = namespace.zeros((shape[0] - n,), dtype=rhs.dtype)
    dx = namespace.linalg.solve(bordered, namespace.concat((rhs, pad)))[:n]
    expected = namespace.linalg.solve(dense_n, rhs)
    assert_allclose(namespace, dx, expected, **tol)


def test_saddle_lbfgs_border_solves_like_dense(namespace, tol):
    # Same equivalence with equalities: the L-BFGS tail rides along after the
    # [Δx | Δy] saddle, and the truncated solve matches the dense saddle.
    n = 4
    lbfgs = _lbfgs_with_pairs(namespace, n)
    sigma_x = Diagonal(array(namespace, [0.2, 0.1, 0.3, 0.4]))
    sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, n), dtype=float_dtype(namespace)))
    condensed = build_condensed_operator(
        lbfgs, sigma_x, sigma_s, empty_jac, RegularizationState(delta_w=1e-3)
    )
    eq_jac = Dense(array(namespace, [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 0.0, 2.0]]))
    saddle = build_saddle_operator(condensed, eq_jac, delta_c=1e-2)

    dense_k = _identity_dense(namespace, saddle)  # exact (n+m) saddle
    rows, cols, values, shape = saddle.to_coo()
    assert shape[0] > saddle.shape[0]  # border beyond the logical [Δx | Δy]
    bordered = _coo_to_dense(namespace, rows, cols, values, shape)

    logical = saddle.shape[0]
    rhs = array(namespace, [1.0, -2.0, 0.5, 3.0, 0.7, -0.4])
    pad = namespace.zeros((shape[0] - logical,), dtype=rhs.dtype)
    sol = namespace.linalg.solve(bordered, namespace.concat((rhs, pad)))[:logical]
    expected = namespace.linalg.solve(dense_k, rhs)
    assert_allclose(namespace, sol, expected, **tol)


def test_saddle_operator_to_coo_matches_matvec(namespace, tol):
    w = Dense(array(namespace, [[2.0, 0.0], [0.0, 2.0]]))
    sigma_x = Diagonal(array(namespace, [0.0, 0.0]))
    sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, 2), dtype=float_dtype(namespace)))
    condensed = build_condensed_operator(
        w, sigma_x, sigma_s, empty_jac, RegularizationState()
    )
    eq_jac = Dense(array(namespace, [[1.0, 1.0]]))
    saddle = build_saddle_operator(condensed, eq_jac, delta_c=1e-2)

    rows, cols, values, shape = saddle.to_coo()
    assert shape == (3, 3)
    assert_allclose(
        namespace,
        _coo_to_dense(namespace, rows, cols, values, shape),
        _identity_dense(namespace, saddle),
        **tol,
    )
