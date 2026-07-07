"""Unit tests for the dense reference linear solver."""

from __future__ import annotations

import pytest

from ipax.backend.namespace import array_namespace
from ipax.backend.operators import Dense, Diagonal, LinearOperator
from ipax.ipm.hessian import LBFGSOperator
from ipax.ipm.kkt import build_condensed_operator, build_saddle_operator
from ipax.linalg.dense import DenseSolver
from ipax.linalg.regularize import RegularizationState
from ipax.linalg.solver import LinearSolveError
from ipax.options import DenseOptions, LBFGSOptions
from tests._helpers import array, assert_allclose


def test_dense_solver_wraps_backend_singular_failure(namespace):
    operator = Dense(array(namespace, [[1.0, 2.0], [2.0, 4.0]]))
    solver = DenseSolver()
    solver.factor(operator)

    with pytest.raises(LinearSolveError, match="dense linear solve failed"):
        solver.solve(array(namespace, [1.0, 2.0]))


def _condensed(namespace, w_dense, *, delta_w=0.0):
    dtype = array(namespace, [0.0]).dtype
    return build_condensed_operator(
        Dense(w_dense),
        Diagonal(array(namespace, [0.0, 0.0])),
        Diagonal(array(namespace, [])),
        Dense(namespace.zeros((0, 2), dtype=dtype)),
        RegularizationState(delta_w=delta_w),
    )


def test_dense_solver_rejects_indefinite_condensed_block(namespace):
    # N = diag(1, -1) is indefinite: xp.linalg.solve (LU) would accept it, but the
    # Cholesky PD guard must reject it so the IPM escalates δ_w.
    op = _condensed(namespace, array(namespace, [[1.0, 0.0], [0.0, -1.0]]))
    solver = DenseSolver()
    solver.factor(op)

    with pytest.raises(LinearSolveError, match="not positive definite"):
        solver.solve(array(namespace, [1.0, 1.0]))


def test_dense_solver_accepts_pd_condensed_block(namespace, tol):
    op = _condensed(namespace, array(namespace, [[2.0, 0.0], [0.0, 3.0]]))
    solver = DenseSolver()
    solver.factor(op)
    actual = solver.solve(array(namespace, [2.0, 3.0]))

    assert_allclose(namespace, actual, array(namespace, [1.0, 1.0]), **tol)


def test_dense_solver_accepts_matrix_rhs(namespace, tol):
    op = Dense(array(namespace, [[2.0, 0.0], [0.0, 4.0]]))
    rhs = array(namespace, [[2.0, 4.0], [8.0, 12.0]])
    solver = DenseSolver()
    solver.factor(op)

    actual = solver.solve(rhs)

    assert_allclose(
        namespace, actual, array(namespace, [[1.0, 2.0], [2.0, 3.0]]), **tol
    )


def test_dense_solver_prefers_structured_solve(namespace, tol):
    class _StructuredOnly:
        shape = (2, 2)

        def __init__(self):
            self.called = False

        def dense_structured_solve(self, rhs):
            self.called = True
            return 2.0 * rhs

        def matmat(self, V):
            raise AssertionError("structured solve should avoid materialization")

    op = _StructuredOnly()
    solver = DenseSolver()
    solver.factor(op)

    actual = solver.solve(array(namespace, [1.0, 2.0]))

    assert op.called
    assert_allclose(namespace, actual, array(namespace, [2.0, 4.0]), **tol)


def test_dense_solver_prefers_dense_matrix_hook(namespace, tol):
    class _DenseMatrixOnly(LinearOperator):
        def __init__(self, matrix):
            self._matrix = matrix
            self.called = False

        @property
        def shape(self):
            return int(self._matrix.shape[0]), int(self._matrix.shape[1])

        def matvec(self, v):
            xp = array_namespace(self._matrix, v)
            return xp.matmul(self._matrix, v)

        def matmat(self, V):
            raise AssertionError("dense_matrix should avoid identity matmat")

        def dense_matrix(self, like=None):
            del like
            self.called = True
            return self._matrix

    matrix = array(namespace, [[2.0, 0.0], [0.0, 4.0]])
    op = _DenseMatrixOnly(matrix)
    solver = DenseSolver()
    solver.factor(op)

    actual = solver.solve(array(namespace, [2.0, 8.0]))

    assert op.called
    assert_allclose(namespace, actual, array(namespace, [1.0, 2.0]), **tol)


def test_dense_solver_caches_materialized_matrix(namespace, tol):
    class _CountingOperator(LinearOperator):
        def __init__(self, matrix):
            self._matrix = matrix
            self.matmat_calls = 0

        @property
        def shape(self):
            return int(self._matrix.shape[0]), int(self._matrix.shape[1])

        def matvec(self, v):
            xp = array_namespace(self._matrix, v)
            return xp.matmul(self._matrix, v)

        def matmat(self, V):
            self.matmat_calls += 1
            xp = array_namespace(self._matrix, V)
            return xp.matmul(self._matrix, V)

    matrix = array(namespace, [[2.0, 0.0], [0.0, 4.0]])
    op = _CountingOperator(matrix)
    solver = DenseSolver()
    solver.factor(op)

    first = solver.solve(array(namespace, [2.0, 8.0]))
    second = solver.solve(array(namespace, [4.0, 12.0]))

    assert op.matmat_calls == 1
    assert_allclose(namespace, first, array(namespace, [1.0, 2.0]), **tol)
    assert_allclose(namespace, second, array(namespace, [2.0, 3.0]), **tol)


def test_dense_solver_reuses_materialized_leading_primal_block(namespace, tol):
    class _ExplodingPrimal:
        shape = (2, 2)

        def matmat(self, V):
            raise AssertionError("primal block should already be materialized")

    class _LeadingPrimalSaddle(LinearOperator):
        shape = (3, 3)

        def __init__(self, matrix):
            self._matrix = matrix

        def matvec(self, v):
            xp = array_namespace(self._matrix, v)
            return xp.matmul(self._matrix, v)

        def matmat(self, V):
            xp = array_namespace(self._matrix, V)
            return xp.matmul(self._matrix, V)

        def primal_block(self):
            return _ExplodingPrimal()

    matrix = array(
        namespace,
        [[2.0, 0.25, 1.0], [0.25, 3.0, -1.0], [1.0, -1.0, -1.0]],
    )
    rhs = array(namespace, [1.0, 2.0, 3.0])
    solver = DenseSolver()
    solver.factor(_LeadingPrimalSaddle(matrix))

    actual = solver.solve(rhs)

    assert_allclose(namespace, namespace.matmul(matrix, actual), rhs, **tol)


def _lbfgs_operator(namespace, n):
    op = LBFGSOperator(n, LBFGSOptions(memory=5))
    delta = array(namespace, [1.0, 0.5] + [0.0] * (n - 2))
    gamma = array(namespace, [2.0, 1.0] + [0.5] * (n - 2))
    op.update(delta, gamma)
    return op


def test_dense_solver_falls_back_when_condensed_has_inequalities(namespace, tol):
    # An L-BFGS condensed block *with* an inequality Gram term has no exact
    # structured solve, so the solver must materialize and factor instead.
    W = _lbfgs_operator(namespace, 2)
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    jac = Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]]))
    op = build_condensed_operator(
        W, sigma_x, sigma_s, jac, RegularizationState(delta_w=1e-6)
    )
    rhs = array(namespace, [1.0, -2.0])
    solver = DenseSolver()
    solver.factor(op)

    with pytest.raises(NotImplementedError):
        op.dense_structured_solve(rhs)  # structured path unavailable -> fall back
    actual = solver.solve(rhs)

    expected = namespace.linalg.solve(op.matmat(namespace.eye(2, dtype=rhs.dtype)), rhs)
    assert_allclose(namespace, actual, expected, **tol)


def test_dense_solver_falls_back_for_saddle_without_structured_condensed(
    namespace, tol
):
    # A saddle whose condensed block is a plain (non-L-BFGS) Hessian cannot use the
    # Woodbury Schur path; the NotImplementedError propagates and the solver
    # materializes the bordered system.
    condensed = _condensed(
        namespace, array(namespace, [[2.0, 0.5], [0.5, 3.0]]), delta_w=1e-6
    )
    saddle = build_saddle_operator(
        condensed, Dense(array(namespace, [[1.0, -1.0]])), 1e-4
    )
    rhs = array(namespace, [1.0, -2.0, 0.5])
    solver = DenseSolver()
    solver.factor(saddle)

    with pytest.raises(NotImplementedError):
        saddle.dense_structured_solve(rhs)  # condensed has no compact form
    actual = solver.solve(rhs)

    expected = namespace.linalg.solve(
        saddle.matmat(namespace.eye(3, dtype=rhs.dtype)), rhs
    )
    assert_allclose(namespace, actual, expected, **tol)


def test_dense_solver_skips_guard_for_plain_operator(namespace, tol):
    # A plain Dense operator exposes no primal_block, so the PD guard is skipped
    # and an ordinary (indefinite-but-nonsingular) solve still succeeds.
    op = Dense(array(namespace, [[0.0, 1.0], [1.0, 0.0]]))
    solver = DenseSolver()
    solver.factor(op)
    actual = solver.solve(array(namespace, [2.0, 3.0]))

    assert_allclose(namespace, actual, array(namespace, [3.0, 2.0]), **tol)


def test_dense_solver_rejects_bad_rhs_rank(namespace):
    solver = DenseSolver()
    solver.factor(Dense(array(namespace, [[1.0, 0.0], [0.0, 1.0]])))
    rhs3 = namespace.zeros((2, 1, 1), dtype=array(namespace, [0.0]).dtype)
    with pytest.raises(ValueError, match="vector or matrix"):
        solver.solve(rhs3)


def test_dense_solver_wraps_structured_solve_failure(namespace):
    # A structured solve crashing on unexpected numerics must surface as the
    # controlled LinearSolveError (so the IPM escalates delta_w), not the raw
    # backend exception.
    class _StructuredBoom(Dense):
        def dense_structured_solve(self, rhs):
            raise RuntimeError("backend blew up")

    solver = DenseSolver()
    solver.factor(_StructuredBoom(array(namespace, [[1.0, 0.0], [0.0, 1.0]])))
    with pytest.raises(LinearSolveError, match="structured solve failed"):
        solver.solve(array(namespace, [1.0, 2.0]))


def test_dense_solver_passes_structured_linear_solve_error_through(namespace):
    # An already-classified LinearSolveError is re-raised as-is (no re-wrapping).
    class _StructuredFails(Dense):
        def dense_structured_solve(self, rhs):
            raise LinearSolveError("already classified")

    solver = DenseSolver()
    solver.factor(_StructuredFails(array(namespace, [[1.0, 0.0], [0.0, 1.0]])))
    with pytest.raises(LinearSolveError, match="already classified"):
        solver.solve(array(namespace, [1.0, 2.0]))


def test_dense_solver_wraps_dense_matrix_failure(namespace):
    class _DenseMatrixBoom(Dense):
        def dense_matrix(self, like=None):
            raise RuntimeError("assembly blew up")

    solver = DenseSolver()
    solver.factor(_DenseMatrixBoom(array(namespace, [[1.0, 0.0], [0.0, 1.0]])))
    with pytest.raises(LinearSolveError, match="materialization failed"):
        solver.solve(array(namespace, [1.0, 2.0]))


def test_dense_solver_wraps_matmat_probe_failure(namespace):
    # A duck-typed operator without the optional structured-solve/dense-matrix
    # hooks: the solver skips both and probes matmat, whose failure must also
    # be classified as a LinearSolveError.
    class _MatmatBoom:
        @property
        def shape(self):
            return (2, 2)

        def matvec(self, v):
            return v

        def matmat(self, V):
            raise RuntimeError("probe blew up")

    solver = DenseSolver()
    solver.factor(_MatmatBoom())
    with pytest.raises(LinearSolveError, match="materialization failed"):
        solver.solve(array(namespace, [1.0, 2.0]))


# --- augmented dense route (DenseOptions(kkt_route="augmented")) -----------


def _condensed_with_inequalities(namespace, *, delta_w=1e-6):
    W_dense = array(namespace, [[4.0, 0.5], [0.5, 3.0]])
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    jac = Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]]))
    return build_condensed_operator(
        Dense(W_dense), sigma_x, sigma_s, jac, RegularizationState(delta_w=delta_w)
    )


def test_dense_solver_augmented_route_matches_condensed_solution(namespace, tol):
    op = _condensed_with_inequalities(namespace)
    rhs = array(namespace, [1.0, -2.0])

    augmented_solver = DenseSolver(DenseOptions(kkt_route="augmented"))
    augmented_solver.factor(op)
    actual = augmented_solver.solve(rhs)

    condensed_solver = DenseSolver()
    condensed_solver.factor(op)
    expected = condensed_solver.solve(rhs)

    assert_allclose(namespace, actual, expected, **tol)


def test_dense_solver_augmented_route_reports_inertia_matching_expected(namespace):
    op = _condensed_with_inequalities(namespace)
    solver = DenseSolver(DenseOptions(kkt_route="augmented"))
    solver.factor(op)
    solver.solve(array(namespace, [1.0, -2.0]))

    assert solver.inertia_or_none() == op.expected_inertia()


def test_dense_solver_augmented_route_reuses_factorization_across_solves(namespace):
    op = _condensed_with_inequalities(namespace)

    calls = {"n": 0}
    original = type(op).augmented_dense_matrix

    def counting(self, like=None):
        calls["n"] += 1
        return original(self, like)

    op.augmented_dense_matrix = counting.__get__(op, type(op))

    solver = DenseSolver(DenseOptions(kkt_route="augmented"))
    solver.factor(op)
    solver.solve(array(namespace, [1.0, -2.0]))
    solver.solve(array(namespace, [0.5, 1.5]))

    assert calls["n"] == 1


def test_dense_solver_augmented_route_falls_back_for_lbfgs(namespace, tol):
    # An L-BFGS condensed block's augmented_dense_matrix() raises
    # NotImplementedError, so the solver falls back to the condensed route
    # silently and still solves correctly.
    W = LBFGSOperator(2, LBFGSOptions(memory=5))
    W.update(array(namespace, [1.0, 0.5]), array(namespace, [2.0, 1.0]))
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    jac = Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]]))
    op = build_condensed_operator(
        W, sigma_x, sigma_s, jac, RegularizationState(delta_w=1e-6)
    )
    rhs = array(namespace, [1.0, -2.0])

    solver = DenseSolver(DenseOptions(kkt_route="augmented"))
    solver.factor(op)
    actual = solver.solve(rhs)

    expected = namespace.linalg.solve(op.matmat(namespace.eye(2, dtype=rhs.dtype)), rhs)
    assert_allclose(namespace, actual, expected, **tol)
    assert solver.inertia_or_none() is None


def test_dense_solver_augmented_route_falls_back_for_plain_operator(namespace, tol):
    # A plain Dense operator has no augmented_dense_matrix at all.
    op = Dense(array(namespace, [[2.0, 0.0], [0.0, 3.0]]))
    solver = DenseSolver(DenseOptions(kkt_route="augmented"))
    solver.factor(op)

    actual = solver.solve(array(namespace, [2.0, 3.0]))

    assert_allclose(namespace, actual, array(namespace, [1.0, 1.0]), **tol)
    assert solver.inertia_or_none() is None


def test_dense_solver_augmented_route_uses_eigh_fallback_without_backend_adapter(
    namespace, tol, monkeypatch
):
    import ipax.backend.dense as dense_backend

    monkeypatch.setattr(
        dense_backend, "get_dense_symmetric_indefinite_adapter", lambda xp: None
    )
    op = _condensed_with_inequalities(namespace)
    rhs = array(namespace, [1.0, -2.0])

    solver = DenseSolver(DenseOptions(kkt_route="augmented"))
    solver.factor(op)
    actual = solver.solve(rhs)

    condensed_solver = DenseSolver()
    condensed_solver.factor(op)
    expected = condensed_solver.solve(rhs)

    assert_allclose(namespace, actual, expected, **tol)
    assert solver.inertia_or_none() == op.expected_inertia()


def test_dense_solver_augmented_route_reuses_adapter_across_factor_calls(
    namespace, monkeypatch
):
    # A GPU adapter (e.g. a persistent cuSOLVER handle) must not be re-created
    # every Newton iteration: the lookup happens once per DenseSolver instance
    # and the adapter object itself is reused across factor() calls.
    import ipax.backend.dense as dense_backend

    calls = {"n": 0}
    original_lookup = dense_backend.get_dense_symmetric_indefinite_adapter

    def counting_lookup(xp):
        calls["n"] += 1
        return original_lookup(xp)

    monkeypatch.setattr(
        dense_backend, "get_dense_symmetric_indefinite_adapter", counting_lookup
    )

    solver = DenseSolver(DenseOptions(kkt_route="augmented"))
    for _ in range(3):
        op = _condensed_with_inequalities(namespace)
        solver.factor(op)
        solver.solve(array(namespace, [1.0, -2.0]))

    assert calls["n"] == 1


def test_dense_solver_describe_reflects_augmented_route(namespace):
    op = _condensed_with_inequalities(namespace)
    solver = DenseSolver(DenseOptions(kkt_route="augmented"))
    solver.factor(op)
    solver.solve(array(namespace, [1.0, -2.0]))
    assert "augmented" in solver.describe()

    plain_solver = DenseSolver()
    plain_solver.factor(op)
    plain_solver.solve(array(namespace, [1.0, -2.0]))
    assert "augmented" not in plain_solver.describe()


def test_dense_solver_default_kkt_route_matches_no_arg_construction(namespace, tol):
    op = _condensed_with_inequalities(namespace)
    rhs = array(namespace, [1.0, -2.0])

    default_options_solver = DenseSolver(DenseOptions())
    default_options_solver.factor(op)
    actual = default_options_solver.solve(rhs)

    no_arg_solver = DenseSolver()
    no_arg_solver.factor(op)
    expected = no_arg_solver.solve(rhs)

    assert_allclose(namespace, actual, expected, **tol)


def test_dense_solver_augmented_route_guards_oversized_border(namespace, tol):
    # With m inequality rows the augmented matrix is (n + m)². When that exceeds
    # DenseOptions.augmented_max_size the solver must fall back to the condensed
    # route *without materializing the bordered matrix* (m ≫ n would otherwise
    # allocate gigabytes), so inertia stays unavailable and the solution matches.
    op = _condensed_with_inequalities(namespace)  # n=2, m_ineq=2 → assembled 4

    calls = {"n": 0}
    original = type(op).augmented_dense_matrix

    def counting(self, like=None):
        calls["n"] += 1
        return original(self, like)

    op.augmented_dense_matrix = counting.__get__(op, type(op))

    solver = DenseSolver(DenseOptions(kkt_route="augmented", augmented_max_size=3))
    solver.factor(op)
    rhs = array(namespace, [1.0, -2.0])
    actual = solver.solve(rhs)

    condensed_solver = DenseSolver()
    condensed_solver.factor(op)
    expected = condensed_solver.solve(rhs)

    assert calls["n"] == 0
    assert solver.inertia_or_none() is None
    assert_allclose(namespace, actual, expected, **tol)


def test_condensed_and_saddle_expose_augmented_assembled_size(namespace):
    op = _condensed_with_inequalities(namespace)
    assert op.augmented_assembled_size() == 4  # n=2 + m_ineq=2

    eq_jac = Dense(array(namespace, [[1.0, 1.0]]))
    saddle = build_saddle_operator(op, eq_jac, 1e-8)
    assert saddle.augmented_assembled_size() == 5  # n=2 + m_eq=1 + m_ineq=2
