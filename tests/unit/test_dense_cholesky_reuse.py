"""Cholesky-factor reuse in the dense solver route.

The PD guard's ``xp.linalg.cholesky`` probe already pays the O(n³)
factorization; these tests pin that the factor is *reused* for the solve via
the ``get_dense_cholesky_solve`` backend gap-filler (the Array API ``linalg``
extension has no triangular solve — BLAS ``trsm`` / LAPACK ``potrs``), instead
of refactoring the same matrix with LU.
"""

from __future__ import annotations

import pytest

from ipax.backend.dense import get_dense_cholesky_solve
from ipax.backend.operators import Dense, Diagonal
from ipax.ipm.kkt import build_condensed_operator, build_saddle_operator
from ipax.linalg.dense import DenseSolver
from ipax.linalg.regularize import RegularizationState
from ipax.linalg.solver import LinearSolveError
from tests._helpers import array, assert_allclose, transpose


def _spd_matrix(namespace):
    b = array(namespace, [[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    eye = namespace.eye(3, dtype=b.dtype)
    return namespace.matmul(transpose(namespace, b), b) + 3.0 * eye


def _condensed_with_inequalities(namespace, *, delta_w=1e-6):
    W_dense = array(namespace, [[4.0, 0.5], [0.5, 3.0]])
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    jac = Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]]))
    return build_condensed_operator(
        Dense(W_dense), sigma_x, sigma_s, jac, RegularizationState(delta_w=delta_w)
    )


def _require_gap_filler(namespace):
    solve_fn = get_dense_cholesky_solve(namespace)
    if solve_fn is None:
        pytest.skip("backend has no triangular-solve primitive")
    return solve_fn


# --- the gap-filler itself ---------------------------------------------------


def test_cholesky_solve_gap_filler_solves_spd_system(namespace, tol):
    solve_fn = _require_gap_filler(namespace)
    a = _spd_matrix(namespace)
    factor = namespace.linalg.cholesky(a)
    rhs = array(namespace, [1.0, -2.0, 0.5])

    x = solve_fn(factor, rhs)

    assert_allclose(namespace, namespace.matmul(a, x), rhs, **tol)


def test_cholesky_solve_gap_filler_accepts_matrix_rhs(namespace, tol):
    solve_fn = _require_gap_filler(namespace)
    a = _spd_matrix(namespace)
    factor = namespace.linalg.cholesky(a)
    rhs = array(namespace, [[1.0, 0.5], [-2.0, 1.5], [0.5, -1.0]])

    x = solve_fn(factor, rhs)

    assert x.shape == rhs.shape
    assert_allclose(namespace, namespace.matmul(a, x), rhs, **tol)


def test_cholesky_solve_gap_filler_availability(backend_name, namespace):
    # numpy (SciPy) and torch are the CI backends and must provide the
    # primitive; array-api-strict deliberately has none (the pure fallback is
    # the LU path, exercised below).
    expected = {"numpy": True, "torch": True, "cupy": True, "array_api_strict": False}
    if backend_name not in expected:
        pytest.skip(f"no availability expectation for {backend_name}")
    assert (get_dense_cholesky_solve(namespace) is not None) is expected[backend_name]


# --- DenseSolver reuse -------------------------------------------------------


def test_dense_solver_reuses_probe_factor_for_condensed_solves(
    namespace, tol, monkeypatch
):
    # The PD probe already factored N; the solve must back-substitute that
    # factor, not refactor via LU. Both solves after one factor() must go
    # through the reused factor (the corrector/SOC back-solve pattern).
    _require_gap_filler(namespace)
    monkeypatch.setattr(
        DenseSolver,
        "_solve_lu",
        lambda self, matrix, rhs, xp: pytest.fail(
            "LU refactor reached despite an available Cholesky factor"
        ),
    )
    op = _condensed_with_inequalities(namespace)
    solver = DenseSolver()
    solver.factor(op)

    rhs1 = array(namespace, [1.0, -2.0])
    rhs2 = array(namespace, [0.5, 1.5])
    x1 = solver.solve(rhs1)
    x2 = solver.solve(rhs2)

    dense = op.matmat(namespace.eye(2, dtype=rhs1.dtype))
    assert_allclose(namespace, namespace.matmul(dense, x1), rhs1, **tol)
    assert_allclose(namespace, namespace.matmul(dense, x2), rhs2, **tol)


def test_dense_solver_falls_back_to_lu_without_gap_filler(namespace, tol, monkeypatch):
    import ipax.backend.dense as dense_backend

    monkeypatch.setattr(dense_backend, "get_dense_cholesky_solve", lambda xp: None)
    calls = {"n": 0}
    original = DenseSolver._solve_lu

    def counting(self, matrix, rhs, xp):
        calls["n"] += 1
        return original(self, matrix, rhs, xp)

    monkeypatch.setattr(DenseSolver, "_solve_lu", counting)
    op = _condensed_with_inequalities(namespace)
    rhs = array(namespace, [1.0, -2.0])
    solver = DenseSolver()
    solver.factor(op)

    x = solver.solve(rhs)

    assert calls["n"] == 1
    dense = op.matmat(namespace.eye(2, dtype=rhs.dtype))
    assert_allclose(namespace, namespace.matmul(dense, x), rhs, **tol)


def test_dense_solver_saddle_route_keeps_lu(namespace, tol, monkeypatch):
    # For an equality saddle the probe factors only the leading N block, so
    # the bordered (indefinite) system must still go through LU.
    calls = {"n": 0}
    original = DenseSolver._solve_lu

    def counting(self, matrix, rhs, xp):
        calls["n"] += 1
        return original(self, matrix, rhs, xp)

    monkeypatch.setattr(DenseSolver, "_solve_lu", counting)
    condensed = _condensed_with_inequalities(namespace)
    saddle = build_saddle_operator(
        condensed, Dense(array(namespace, [[1.0, -1.0]])), 1e-4
    )
    rhs = array(namespace, [1.0, -2.0, 0.5])
    solver = DenseSolver()
    solver.factor(saddle)

    x = solver.solve(rhs)

    assert calls["n"] == 1
    dense = saddle.matmat(namespace.eye(3, dtype=rhs.dtype))
    assert_allclose(namespace, namespace.matmul(dense, x), rhs, **tol)


def test_dense_solver_falls_back_to_lu_when_backsub_fails(namespace, tol, monkeypatch):
    # The reuse is purely an optimization: an unexpected back-substitution
    # failure must not escalate delta_w — the materialized matrix is still in
    # hand, so the solver falls back to LU and drops the factor so later
    # solves skip the retry.
    import ipax.backend.dense as dense_backend

    calls = {"boom": 0}

    def boom(factor, rhs):
        calls["boom"] += 1
        raise RuntimeError("backend blew up")

    monkeypatch.setattr(dense_backend, "get_dense_cholesky_solve", lambda xp: boom)
    op = _condensed_with_inequalities(namespace)
    rhs = array(namespace, [1.0, -2.0])
    solver = DenseSolver()
    solver.factor(op)

    x1 = solver.solve(rhs)
    x2 = solver.solve(rhs)

    assert calls["boom"] == 1
    dense = op.matmat(namespace.eye(2, dtype=rhs.dtype))
    assert_allclose(namespace, namespace.matmul(dense, x1), rhs, **tol)
    assert_allclose(namespace, namespace.matmul(dense, x2), rhs, **tol)


def test_dense_solver_still_rejects_indefinite_block_before_reuse(namespace):
    # The escalation contract is untouched: a non-PD N raises LinearSolveError
    # from the probe itself, so no factor is ever kept for reuse.
    dtype = array(namespace, [0.0]).dtype
    op = build_condensed_operator(
        Dense(array(namespace, [[1.0, 0.0], [0.0, -1.0]])),
        Diagonal(array(namespace, [0.0, 0.0])),
        Diagonal(array(namespace, [])),
        Dense(namespace.zeros((0, 2), dtype=dtype)),
        RegularizationState(delta_w=0.0),
    )
    solver = DenseSolver()
    solver.factor(op)

    with pytest.raises(LinearSolveError, match="not positive definite"):
        solver.solve(array(namespace, [1.0, 1.0]))


def test_dense_solver_factor_resets_cached_cholesky(namespace, tol):
    # factor() with a new operator must not back-substitute the previous
    # operator's factor.
    op1 = _condensed_with_inequalities(namespace, delta_w=1e-6)
    op2 = _condensed_with_inequalities(namespace, delta_w=10.0)
    rhs = array(namespace, [1.0, -2.0])
    solver = DenseSolver()

    solver.factor(op1)
    x1 = solver.solve(rhs)
    solver.factor(op2)
    x2 = solver.solve(rhs)

    dense1 = op1.matmat(namespace.eye(2, dtype=rhs.dtype))
    dense2 = op2.matmat(namespace.eye(2, dtype=rhs.dtype))
    assert_allclose(namespace, namespace.matmul(dense1, x1), rhs, **tol)
    assert_allclose(namespace, namespace.matmul(dense2, x2), rhs, **tol)
