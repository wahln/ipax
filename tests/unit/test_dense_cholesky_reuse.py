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
from ipax.backend.namespace import array_namespace
from ipax.backend.operators import Dense, Diagonal, LinearOperator
from ipax.ipm.hessian import LBFGSOperator
from ipax.ipm.kkt import build_condensed_operator, build_saddle_operator
from ipax.linalg.dense import DenseSolver
from ipax.linalg.regularize import RegularizationState
from ipax.linalg.solver import LinearSolveError
from ipax.options import DenseOptions, LBFGSOptions
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


def _lbfgs_condensed_with_inequalities(namespace, *, delta_w=1e-6):
    # Powell-damped L-BFGS Hessian + inequality Gram term: no structured solve,
    # and ``primal_block()`` is None, so the PD *guard* never probes it.
    W = LBFGSOperator(2, LBFGSOptions(memory=5))
    W.update(array(namespace, [1.0, 0.5]), array(namespace, [2.0, 1.0]))
    sigma_x = Diagonal(array(namespace, [0.25, 0.75]))
    sigma_s = Diagonal(array(namespace, [2.0, 0.5]))
    jac = Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]]))
    return build_condensed_operator(
        W, sigma_x, sigma_s, jac, RegularizationState(delta_w=delta_w)
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


# --- PD-by-construction (L-BFGS) blocks ---------------------------------------


def test_dense_solver_keeps_cholesky_for_pd_hinted_lbfgs_condensed(
    namespace, tol, monkeypatch
):
    # An L-BFGS block with inequality rows is materialized (no structured
    # solve) but skips the PD guard; it is PD by Powell damping, so the solver
    # should Cholesky-factor it once and back-substitute every RHS instead of
    # paying an LU refactor per solve.
    _require_gap_filler(namespace)
    monkeypatch.setattr(
        DenseSolver,
        "_solve_lu",
        lambda self, matrix, rhs, xp: pytest.fail(
            "LU refactor reached for a PD-by-construction L-BFGS block"
        ),
    )
    op = _lbfgs_condensed_with_inequalities(namespace)
    solver = DenseSolver()
    solver.factor(op)

    rhs1 = array(namespace, [1.0, -2.0])
    rhs2 = array(namespace, [[0.5, 1.0], [1.5, -0.25]])
    x1 = solver.solve(rhs1)
    x2 = solver.solve(rhs2)

    dense = op.matmat(namespace.eye(2, dtype=rhs1.dtype))
    assert_allclose(namespace, namespace.matmul(dense, x1), rhs1, **tol)
    assert_allclose(namespace, namespace.matmul(dense, x2), rhs2, **tol)


class _HintedIndefinite(LinearOperator):
    """Claims PD by construction but materializes an indefinite matrix."""

    def __init__(self, namespace) -> None:
        self._m = array(namespace, [[1.0, 0.0], [0.0, -1.0]])

    @property
    def shape(self) -> tuple[int, int]:
        return 2, 2

    def matvec(self, v):
        return array_namespace(v).matmul(self._m, v)

    def rmatvec(self, v):
        return self.matvec(v)

    def dense_matrix(self, like=None):
        del like
        return self._m

    def primal_block(self):
        return None

    def symmetry_hint(self) -> bool | None:
        return True

    def positive_definite_hint(self) -> bool:
        return True


class _HintedUnknownSymmetry(_HintedIndefinite):
    """PD claim without a symmetry claim: a Cholesky would read one triangle."""

    def __init__(self, namespace) -> None:
        self._m = array(namespace, [[2.0, 0.0], [0.0, 3.0]])

    def symmetry_hint(self) -> bool | None:
        return None


def test_dense_solver_lu_fallback_when_pd_hinted_cholesky_fails(
    namespace, tol, monkeypatch
):
    # The hint only unlocks an optimization: a block that is numerically not
    # PD despite the claim must NOT raise (that is the guard's job, and the
    # guard is deliberately skipped here) — the solver falls back to LU as
    # before, keeping no factor.
    calls = {"n": 0}
    original = DenseSolver._solve_lu

    def counting(self, matrix, rhs, xp):
        calls["n"] += 1
        return original(self, matrix, rhs, xp)

    monkeypatch.setattr(DenseSolver, "_solve_lu", counting)
    op = _HintedIndefinite(namespace)
    rhs = array(namespace, [1.0, -2.0])
    solver = DenseSolver()
    solver.factor(op)

    x = solver.solve(rhs)

    assert calls["n"] == 1
    assert solver._cholesky_factor is None
    assert_allclose(namespace, x, array(namespace, [1.0, 2.0]), **tol)


def test_dense_solver_pd_hint_requires_symmetry_claim(namespace, tol, monkeypatch):
    # ``xp.linalg.cholesky`` reads one triangle, so a PD claim alone would
    # silently solve an asymmetric matrix as its symmetrization: the reuse
    # engages only when the operator also declares ``symmetry_hint() is True``.
    calls = {"n": 0}
    original = DenseSolver._solve_lu

    def counting(self, matrix, rhs, xp):
        calls["n"] += 1
        return original(self, matrix, rhs, xp)

    monkeypatch.setattr(DenseSolver, "_solve_lu", counting)
    rhs = array(namespace, [2.0, -3.0])
    solver = DenseSolver()
    solver.factor(_HintedUnknownSymmetry(namespace))

    x = solver.solve(rhs)

    assert calls["n"] == 1
    assert solver._cholesky_factor is None
    assert_allclose(namespace, x, array(namespace, [1.0, -1.0]), **tol)


def test_dense_solver_pd_hint_keeps_factor_through_mixed_precision(
    namespace, tol, monkeypatch
):
    # gram_dtype="float32" × PD hint: the factor kept while the reduced-
    # precision matrix is engaged is what refinement back-solves against; a
    # refinement rejection rebuilds the exact matrix and must leave the
    # *exact* factor behind (not the stale reduced one, not none).
    _require_gap_filler(namespace)
    if not hasattr(namespace, "float32"):
        pytest.skip("backend has no float32")
    op = _lbfgs_condensed_with_inequalities(namespace)
    rhs = array(namespace, [1.0, -2.0])
    dense = op.matmat(namespace.eye(2, dtype=rhs.dtype))

    engaged = DenseSolver(DenseOptions(gram_dtype="float32"))
    engaged.factor(op)
    x = engaged.solve(rhs)
    assert engaged._mixed_engaged
    assert engaged._cholesky_factor is not None
    assert_allclose(namespace, namespace.matmul(dense, x), rhs, **tol)

    monkeypatch.setattr(DenseSolver, "_refine", lambda self, x, rhs, xp: None)
    monkeypatch.setattr(
        DenseSolver,
        "_solve_lu",
        lambda self, matrix, rhs, xp: pytest.fail("exact rebuild fell back to LU"),
    )
    rejected = DenseSolver(DenseOptions(gram_dtype="float32"))
    rejected.factor(op)
    x = rejected.solve(rhs)
    assert not rejected._mixed_engaged
    assert rejected._cholesky_factor is not None
    assert_allclose(namespace, namespace.matmul(dense, x), rhs, **tol)


def test_dense_solver_pd_hint_needs_gap_filler(namespace, tol, monkeypatch):
    # Without a back-substitution primitive a Cholesky of the hinted block
    # would be pure waste next to the LU: the solver must not keep a factor
    # and must go straight to LU.
    import ipax.backend.dense as dense_backend

    monkeypatch.setattr(dense_backend, "get_dense_cholesky_solve", lambda xp: None)
    calls = {"n": 0}
    original = DenseSolver._solve_lu

    def counting(self, matrix, rhs, xp):
        calls["n"] += 1
        return original(self, matrix, rhs, xp)

    monkeypatch.setattr(DenseSolver, "_solve_lu", counting)
    op = _lbfgs_condensed_with_inequalities(namespace)
    rhs = array(namespace, [1.0, -2.0])
    solver = DenseSolver()
    solver.factor(op)

    x = solver.solve(rhs)

    assert calls["n"] == 1
    assert solver._cholesky_factor is None
    dense = op.matmat(namespace.eye(2, dtype=rhs.dtype))
    assert_allclose(namespace, namespace.matmul(dense, x), rhs, **tol)


# --- breakdown bookkeeping for the PD hint -----------------------------------


class _HintedSwitchable(_HintedIndefinite):
    """PD claim over a matrix the test flips between indefinite and SPD."""

    def __init__(self, namespace, *, pd: bool) -> None:
        self._m = array(namespace, [[2.0, 0.0], [0.0, 3.0 if pd else -1.0]])


def test_pd_hint_failure_limit_must_be_positive():
    assert DenseOptions().pd_hint_failure_limit == 3
    with pytest.raises(ValueError, match="pd_hint_failure_limit"):
        DenseOptions(pd_hint_failure_limit=0)


def test_dense_solver_pd_hint_retries_after_a_single_breakdown(namespace, tol):
    # Conditioning along an IPM run is not monotone, so one numerically
    # non-PD block must not forfeit the O(n²) back-solves for the rest of the
    # run: the very next factorization tries the hinted Cholesky again.
    _require_gap_filler(namespace)
    rhs = array(namespace, [2.0, -3.0])
    solver = DenseSolver()

    solver.factor(_HintedSwitchable(namespace, pd=False))
    solver.solve(rhs)
    assert solver._cholesky_factor is None

    solver.factor(_HintedSwitchable(namespace, pd=True))
    x = solver.solve(rhs)
    assert solver._cholesky_factor is not None
    assert_allclose(namespace, x, array(namespace, [1.0, -1.0]), **tol)


def test_dense_solver_pd_hint_disables_after_consecutive_breakdowns(
    namespace, tol, monkeypatch
):
    # ``pd_hint_failure_limit`` consecutive breakdowns are the signal that
    # the structural claim is not worth its wasted O(n³) attempt: the solver
    # stops trying for the rest of its life and goes straight to LU — even
    # for a block that would have factored.
    _require_gap_filler(namespace)
    attempts = {"n": 0}
    original = DenseSolver._keep_pd_hinted_factor

    def counting(self, matrix, xp, cholesky, *, reduced=False):
        def spy(m):
            attempts["n"] += 1
            return cholesky(m)

        return original(self, matrix, xp, spy, reduced=reduced)

    monkeypatch.setattr(DenseSolver, "_keep_pd_hinted_factor", counting)
    rhs = array(namespace, [2.0, -3.0])
    solver = DenseSolver(DenseOptions(pd_hint_failure_limit=2))

    for _ in range(2):
        solver.factor(_HintedSwitchable(namespace, pd=False))
        solver.solve(rhs)
    assert attempts["n"] == 2
    assert solver._pd_hint_disabled

    solver.factor(_HintedSwitchable(namespace, pd=True))
    x = solver.solve(rhs)
    assert attempts["n"] == 2  # no further Cholesky attempt
    assert solver._cholesky_factor is None
    assert_allclose(namespace, x, array(namespace, [1.0, -1.0]), **tol)


def test_dense_solver_pd_hint_success_resets_the_breakdown_count(namespace, tol):
    # Only *consecutive* breakdowns count (the mixed route's rule): a success
    # in between resets the counter, so alternating hard and easy blocks never
    # trip the kill switch.
    _require_gap_filler(namespace)
    rhs = array(namespace, [2.0, -3.0])
    solver = DenseSolver(DenseOptions(pd_hint_failure_limit=2))

    for pd in (False, True, False, True):
        solver.factor(_HintedSwitchable(namespace, pd=pd))
        solver.solve(rhs)

    assert not solver._pd_hint_disabled
    assert solver._pd_hint_failures == 0
    assert solver._cholesky_factor is not None


def test_dense_solver_describe_marks_a_pd_hint_breakdown(namespace):
    # ``Result.routes`` captures the label once after the run, so the marker
    # is sticky: a block declared PD by construction that was nevertheless
    # solved by LU at least once must not read as a clean ``dense`` run.
    _require_gap_filler(namespace)
    rhs = array(namespace, [2.0, -3.0])
    solver = DenseSolver()
    assert solver.describe() == "dense"

    solver.factor(_HintedSwitchable(namespace, pd=True))
    solver.solve(rhs)
    assert solver.describe() == "dense"

    solver.factor(_HintedSwitchable(namespace, pd=False))
    solver.solve(rhs)
    assert solver.describe() == "dense (pd-hint->lu)"

    solver.factor(_HintedSwitchable(namespace, pd=True))
    solver.solve(rhs)
    assert solver.describe() == "dense (pd-hint->lu)"


class _HintedMixedBreakdown(_HintedSwitchable):
    """SPD exact block whose reduced-precision materialization is not PD."""

    def __init__(self, namespace) -> None:
        super().__init__(namespace, pd=True)
        self._reduced = array(namespace, [[2.0, 0.0], [0.0, -1.0]])

    def dense_matrix_mixed(self, like, gram_dtype, *, hinted_only=False):
        del like, gram_dtype, hinted_only
        return self._reduced


def test_dense_solver_reduced_matrix_breakdown_does_not_retire_the_hint(
    namespace, tol, monkeypatch
):
    # Under the mixed route the hinted Cholesky sees the reduced-precision
    # matrix, so a breakdown there may be precision noise: it is marked in
    # describe() (that factorization did fall back to LU) but must count
    # toward neither kill switch — not the hint's (the exact block may factor
    # fine) and not the mixed route's (the refinement pass is the certificate
    # that judges the reduced matrix, not the Cholesky).
    _require_gap_filler(namespace)
    if not hasattr(namespace, "float32"):
        pytest.skip("backend has no float32")
    monkeypatch.setattr(DenseSolver, "_refine", lambda self, x, rhs, xp: x)
    rhs = array(namespace, [2.0, -3.0])
    solver = DenseSolver(DenseOptions(gram_dtype="float32", pd_hint_failure_limit=1))

    solver.factor(_HintedMixedBreakdown(namespace))
    solver.solve(rhs)

    assert solver._mixed_engaged
    assert solver._mixed_failures == 0 and not solver._mixed_disabled
    assert solver._pd_hint_failures == 0 and not solver._pd_hint_disabled
    assert solver.describe() == "dense (gram=float32, pd-hint->lu)"

    # ...and the exact block still gets its factor on the next factorization.
    native = DenseSolver(DenseOptions(gram_dtype="float32", pd_hint_failure_limit=1))
    native.factor(_HintedMixedBreakdown(namespace))
    native.solve(rhs)
    native.factor(_HintedSwitchable(namespace, pd=True))
    x = native.solve(rhs)
    assert native._cholesky_factor is not None
    assert_allclose(namespace, x, array(namespace, [1.0, -1.0]), **tol)


def test_dense_solver_describe_composes_retired_mixed_and_pd_hint_markers():
    # Both sticky markers survive together after the mixed route retires.
    solver = DenseSolver(DenseOptions(gram_dtype="float32"))
    solver._mixed_disabled = True
    solver._mixed_ever_engaged = True
    solver._mixed_label = "float32"
    solver._pd_hint_ever_failed = True
    assert solver.describe() == "dense (gram=float32->native, pd-hint->lu)"
