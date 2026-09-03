"""Unit tests for the matrix-free Krylov solver.

CG on SPD condensed systems, MINRES on symmetric-indefinite saddles, Jacobi
preconditioning, and the automatic CG→MINRES fallback on indefiniteness. All
parametrized over the backend fixture so each assertion runs on every available
namespace.
"""

from __future__ import annotations

import pytest

from ipax.backend.operators import Dense, Diagonal, MatrixFreeJacobian
from ipax.linalg.krylov import KrylovConvergenceError, KrylovSolver
from ipax.options import KrylovOptions
from tests._helpers import array, assert_allclose, norm_inf


def _spd_system(namespace):
    """A small SPD system with a known solution."""
    A = array(namespace, [[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    x_exact = array(namespace, [1.0, -2.0, 0.5])
    rhs = namespace.matmul(A, x_exact)
    return A, rhs, x_exact


def _indefinite_system(namespace):
    """A symmetric *indefinite* system (mixed-sign spectrum)."""
    A = array(namespace, [[3.0, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, 4.0]])
    x_exact = array(namespace, [0.5, -1.5, 2.0])
    rhs = namespace.matmul(A, x_exact)
    return A, rhs, x_exact


def _solver(**kwargs) -> KrylovSolver:
    return KrylovSolver(KrylovOptions(**kwargs))


def test_cg_solves_spd_system(namespace, tol):
    A, rhs, x_exact = _spd_system(namespace)
    solver = _solver(method="cg", rtol=1e-12)
    solver.factor(Dense(A))
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, **tol)
    assert solver.last_method == "cg"


def test_cg_residual_within_tolerance(namespace):
    A, rhs, _ = _spd_system(namespace)
    solver = _solver(method="cg", rtol=1e-12)
    solver.factor(Dense(A))
    x = solver.solve(rhs)

    residual = Dense(A).matvec(x) - rhs
    assert norm_inf(namespace, residual) <= 1e-9


def test_cg_is_matrix_free(namespace, tol):
    """CG must converge using only ``matvec`` — no adjoint, no materialization."""
    A, rhs, x_exact = _spd_system(namespace)

    def matvec(v):
        return namespace.matmul(A, v)

    # MatrixFreeJacobian exposes matvec only; rmatvec/matmat/diagonal all raise.
    operator = MatrixFreeJacobian((3, 3), matvec)
    solver = _solver(method="cg", rtol=1e-12)
    solver.factor(operator)
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, **tol)


def test_minres_solves_symmetric_indefinite_system(namespace, tol):
    A, rhs, x_exact = _indefinite_system(namespace)
    solver = _solver(method="minres", rtol=1e-12)
    solver.factor(Dense(A))
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, **tol)
    assert solver.last_method == "minres"


def test_cg_falls_back_to_minres_on_indefinite_operator(namespace, tol):
    """The default ``cg`` method detects negative curvature and switches to MINRES."""
    A, rhs, x_exact = _indefinite_system(namespace)
    solver = _solver(method="cg", rtol=1e-12)
    solver.factor(Dense(A))
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, **tol)
    assert solver.last_method == "minres"


def test_jacobi_preconditioner_reduces_iterations(namespace):
    """A diagonal operator with a wide spectrum: Jacobi collapses it to one step."""
    d = array(namespace, [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    operator = Diagonal(d)
    x_exact = array(namespace, [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0])
    rhs = operator.matvec(x_exact)

    unprecond = _solver(method="cg", rtol=1e-12, preconditioner="none")
    unprecond.factor(operator)
    unprecond.solve(rhs)

    jacobi = _solver(method="cg", rtol=1e-12, preconditioner="jacobi")
    jacobi.factor(operator)
    jacobi.solve(rhs)

    assert unprecond.last_iterations > 1
    assert jacobi.last_iterations < unprecond.last_iterations


def test_jacobi_preconditions_the_condensed_operator(namespace):
    """Jacobi now activates on the real condensed Newton operator (matrix-free).

    A bound-regularized, ill-conditioned ``W`` plus an inequality Gram term — the
    operator the IPM actually solves. The diagonal is assembled without forming a
    matrix, and preconditioning must cut the CG iteration count.
    """
    from ipax.ipm.kkt import build_condensed_operator
    from ipax.linalg.regularize import RegularizationState

    d = array(namespace, [1.0, 10.0, 100.0, 1000.0, 5.0, 50.0])
    W = Diagonal(d)
    sigma_x = Diagonal(array(namespace, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    sigma_s = Diagonal(array(namespace, [3.0, 0.25]))
    jac = array(
        namespace,
        [
            [1.0, 0.0, 2.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 3.0, 0.0, 1.0],
        ],
    )
    operator = build_condensed_operator(
        W, sigma_x, sigma_s, Dense(jac), RegularizationState(delta_w=1e-6)
    )
    x_exact = array(namespace, [1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    rhs = operator.matvec(x_exact)

    unprecond = _solver(method="cg", rtol=1e-10, preconditioner="none")
    unprecond.factor(operator)
    x_none = unprecond.solve(rhs)

    jacobi = _solver(method="cg", rtol=1e-10, preconditioner="jacobi")
    jacobi.factor(operator)
    x_jac = jacobi.solve(rhs)

    assert_allclose(namespace, x_none, x_exact, rtol=1e-7, atol=1e-7)
    assert_allclose(namespace, x_jac, x_exact, rtol=1e-7, atol=1e-7)
    assert jacobi.last_iterations < unprecond.last_iterations


def test_minres_jacobi_preconditioner_reduces_saddle_iterations(namespace):
    """SPD block preconditioner for the indefinite equality saddle.

    An ill-conditioned (κ ≈ 500) symmetric quasidefinite saddle — PD primal block
    plus the ``−δ_c`` dual block. The block-diagonal SPD preconditioner (Jacobi on
    the primal block, approximate Schur on the dual) must cut MINRES iterations.
    """
    from ipax.ipm.kkt import build_saddle_operator

    d = array(
        namespace,
        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0],
    )
    c_mat = array(
        namespace,
        [
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ],
    )
    saddle = build_saddle_operator(Diagonal(d), Dense(c_mat), 1e-8)
    x_exact = array(
        namespace,
        [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 0.25, -0.25, 3.0, -3.0],
    )
    rhs = saddle.matvec(x_exact)

    unprecond = _solver(method="minres", rtol=1e-10, preconditioner="none")
    unprecond.factor(saddle)
    x_none = unprecond.solve(rhs)

    jacobi = _solver(method="minres", rtol=1e-10, preconditioner="jacobi")
    jacobi.factor(saddle)
    x_jac = jacobi.solve(rhs)

    assert_allclose(namespace, x_none, x_exact, rtol=1e-6, atol=1e-6)
    assert_allclose(namespace, x_jac, x_exact, rtol=1e-6, atol=1e-6)
    assert jacobi.last_iterations < unprecond.last_iterations


def test_saddle_spd_preconditioner_diagonal_is_positive(namespace, tol):
    """The block preconditioner diagonal is SPD with the expected Schur dual."""
    from ipax.ipm.kkt import build_saddle_operator

    d = array(namespace, [1.0, 2.0, 3.0])
    c_mat = array(namespace, [[1.0, 0.0, 2.0]])
    saddle = build_saddle_operator(Diagonal(d), Dense(c_mat), 1e-8)

    pd = saddle.spd_preconditioner_diagonal()
    assert bool(namespace.all(pd > 0.0))
    # Primal block is diag(N); dual block is δ_c + Σ_k c_k² / d_k.
    assert_allclose(namespace, pd[:3], d, **tol)
    expected_dual = 1e-8 + (1.0 / 1.0 + 4.0 / 3.0)
    assert abs(float(pd[3]) - expected_dual) <= 1e-9


def test_saddle_lbfgs_block_preconditioner_matches_blocks(namespace, tol):
    """diag(N⁻¹, S⁻¹): the Woodbury inverse on the primal block, Schur-diagonal
    reciprocal on the dual block."""
    from ipax.ipm.kkt import build_saddle_operator

    condensed = _condensed_no_inequalities(namespace)  # n = 6, L-BFGS, no ineqs
    n = condensed.shape[0]
    c_mat = array(
        namespace,
        [
            [1.0, 0.0, 2.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 3.0],
        ],
    )
    delta_c = 1e-8
    saddle = build_saddle_operator(condensed, Dense(c_mat), delta_c)
    apply = saddle.lbfgs_block_preconditioner_apply()

    r = array(namespace, [0.5, -1.0, 2.0, -0.25, 1.5, -2.0, 0.7, -0.3])  # n + m = 8
    out = apply(r)

    n_inv = condensed.lbfgs_inverse_apply()
    top = n_inv(r[:n])
    weight = 1.0 / condensed.diagonal()
    dual = delta_c + namespace.sum(
        (c_mat * c_mat) * namespace.reshape(weight, (1, n)), axis=1
    )
    expected = namespace.concat((top, r[n:] / dual))
    assert_allclose(namespace, out, expected, **tol)


def test_saddle_lbfgs_block_preconditioner_requires_lbfgs_structure(namespace):
    """A saddle whose condensed block has no L-BFGS compact form raises."""
    from ipax.ipm.kkt import build_condensed_operator, build_saddle_operator
    from ipax.linalg.regularize import RegularizationState

    n = 3
    dtype = array(namespace, [0.0]).dtype
    condensed = build_condensed_operator(
        Dense(array(namespace, [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])),
        Diagonal(array(namespace, [0.1, 0.2, 0.3])),
        Diagonal(array(namespace, [])),
        Dense(namespace.zeros((0, n), dtype=dtype)),
        RegularizationState(delta_w=1e-6),
    )
    saddle = build_saddle_operator(
        condensed, Dense(array(namespace, [[1.0, 0.0, 1.0]])), 1e-8
    )
    with pytest.raises(NotImplementedError):
        saddle.lbfgs_block_preconditioner_apply()


def test_saddle_lbfgs_block_preconditioner_reduces_iterations(namespace):
    """The block preconditioner diag(N⁻¹, S⁻¹), applied via GMRES, cuts iterations
    on an ill-conditioned equality saddle vs the Jacobi/MINRES route — and solves
    it correctly.
    """
    import random

    from ipax.ipm.hessian import LBFGSOperator
    from ipax.ipm.kkt import build_condensed_operator, build_saddle_operator
    from ipax.linalg.regularize import RegularizationState
    from ipax.options import LBFGSOptions

    rng = random.Random(11)
    n = 30
    w = LBFGSOperator(n, LBFGSOptions(memory=6))
    for _ in range(4):
        delta = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        gamma = [delta[k] + rng.uniform(0.0, 0.5) + 0.5 for k in range(n)]
        w.update(array(namespace, delta), array(namespace, gamma))

    dtype = array(namespace, [0.0]).dtype
    sigma_x = Diagonal(
        array(namespace, [10.0 ** (3.0 * k / (n - 1)) for k in range(n)])
    )
    condensed = build_condensed_operator(
        w,
        sigma_x,
        Diagonal(array(namespace, [])),
        Dense(namespace.zeros((0, n), dtype=dtype)),
        RegularizationState(delta_w=1e-6),
    )
    c_mat = array(
        namespace,
        [
            [1.0 if k % 4 == 0 else 0.0 for k in range(n)],
            [1.0 if k % 7 == 0 else 0.0 for k in range(n)],
            [(-1.0) ** k * 0.5 for k in range(n)],
        ],
    )
    m = 3
    saddle = build_saddle_operator(condensed, Dense(c_mat), 1e-8)
    x_exact = array(namespace, [(-1.0) ** k * (1.0 + k / n) for k in range(n + m)])
    rhs = saddle.matvec(x_exact)

    jac = _solver(method="cg", rtol=1e-8, preconditioner="jacobi")  # → MINRES + Jacobi
    jac.factor(saddle)
    x_j = jac.solve(rhs)

    blk = _solver(method="cg", rtol=1e-8, preconditioner="lbfgs")  # → GMRES + block
    blk.factor(saddle)
    x_b = blk.solve(rhs)

    assert_allclose(namespace, x_j, x_exact, rtol=1e-5, atol=1e-5)
    assert_allclose(namespace, x_b, x_exact, rtol=1e-5, atol=1e-5)
    assert jac.last_method == "minres"
    assert blk.last_method == "gmres"
    assert blk.last_iterations < jac.last_iterations


def test_saddle_preferred_method_skips_cg(namespace, monkeypatch, tol):
    """Saddle operators route directly to MINRES instead of probing with CG."""
    from ipax.ipm.kkt import build_saddle_operator

    saddle = build_saddle_operator(
        Diagonal(array(namespace, [2.0, 3.0])),
        Dense(array(namespace, [[1.0, -1.0]])),
        1e-8,
    )
    x_exact = array(namespace, [0.25, -0.5, 1.5])
    rhs = saddle.matvec(x_exact)
    solver = _solver(method="cg", rtol=1e-12)
    solver.factor(saddle)

    def fail_cg(*args, **kwargs):
        del args, kwargs
        raise AssertionError("CG should not run for equality saddles")

    monkeypatch.setattr(KrylovSolver, "_cg", fail_cg)
    actual = solver.solve(rhs)

    assert solver.last_method == "minres"
    assert_allclose(namespace, actual, x_exact, **tol)


def test_gmres_solves_spd_system(namespace, tol):
    A, rhs, x_exact = _spd_system(namespace)
    solver = _solver(method="gmres", rtol=1e-12)
    solver.factor(Dense(A))
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, **tol)
    assert solver.last_method == "gmres"


def test_gmres_solves_symmetric_indefinite_system(namespace, tol):
    A, rhs, x_exact = _indefinite_system(namespace)
    solver = _solver(method="gmres", rtol=1e-12)
    solver.factor(Dense(A))
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, **tol)
    assert solver.last_method == "gmres"


def test_gmres_solves_nonsymmetric_matrix_free_system(namespace, tol):
    """GMRES covers the nonsymmetric matrix-free path without an adjoint."""
    A = array(namespace, [[3.0, 2.0, 0.0], [-1.0, 4.0, 1.0], [0.5, 0.0, 2.0]])
    x_exact = array(namespace, [1.0, -2.0, 0.5])
    rhs = namespace.matmul(A, x_exact)

    operator = MatrixFreeJacobian((3, 3), lambda v: namespace.matmul(A, v))
    solver = _solver(method="gmres", rtol=1e-12, gmres_restart=3)
    solver.factor(operator)
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, **tol)
    assert solver.last_method == "gmres"


def test_gmres_converges_with_small_restart(namespace, tol):
    """Restarted GMRES(m) with m below the dimension still converges."""
    A, rhs, x_exact = _spd_system(namespace)
    solver = _solver(method="gmres", rtol=1e-12, gmres_restart=2)
    solver.factor(Dense(A))
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, rtol=1e-6, atol=1e-6)


def test_gmres_non_finite_operator_raises_convergence_error(namespace):
    """A NaN in the operator (a bad iterate feeding the saddle GMRES fallback) must
    not crash the Arnoldi with an IndexError from the breakdown path — it raises
    KrylovConvergenceError so the driver escalates δ_w instead of the solve crashing.
    """
    nan = float("nan")
    A = array(namespace, [[nan, 1.0], [1.0, 2.0]])
    rhs = array(namespace, [1.0, 1.0])
    solver = _solver(method="gmres", rtol=1e-10, max_iter=10)
    solver.factor(Dense(A))
    with pytest.raises(KrylovConvergenceError):
        solver.solve(rhs)


def test_gmres_non_convergence_raises(namespace):
    A, rhs, _ = _spd_system(namespace)
    solver = _solver(method="gmres", rtol=1e-14, max_iter=1, gmres_restart=1)
    solver.factor(Dense(A))
    with pytest.raises(KrylovConvergenceError):
        solver.solve(rhs)


def _condensed_no_inequalities(namespace, ineq=None):
    """A condensed operator ``N = B + Σ_x + δ_w I`` with an L-BFGS Hessian ``B``.

    ``ineq=(jac, sigma_s)`` adds an inequality Gram term ``jacᵀ Σ_s jac``.
    """
    from ipax.ipm.hessian import LBFGSOperator
    from ipax.ipm.kkt import build_condensed_operator
    from ipax.linalg.regularize import RegularizationState
    from ipax.options import LBFGSOptions

    n = 6
    w = LBFGSOperator(n, LBFGSOptions(memory=5))
    deltas = [
        [1.0, 0.5, -0.5, 0.2, 0.1, -0.3],
        [0.3, -1.0, 1.0, 0.4, -0.2, 0.5],
        [-0.4, 0.6, 0.2, -1.0, 0.8, 0.1],
    ]
    gammas = [
        [2.0, 1.0, 0.5, 0.3, 0.2, 0.4],
        [1.0, 1.5, 0.5, 0.6, 0.3, 0.2],
        [0.5, 0.4, 1.2, 0.7, 0.6, 0.3],
    ]
    for delta, gamma in zip(deltas, gammas, strict=True):
        w.update(array(namespace, delta), array(namespace, gamma))
    sigma_x = Diagonal(array(namespace, [0.2, 0.5, 1.0, 1.5, 2.0, 0.8]))
    if ineq is not None:
        jac, sigma_s = ineq
    else:
        sigma_s = Diagonal(array(namespace, []))
        jac = Dense(namespace.zeros((0, n), dtype=array(namespace, [0.0]).dtype))
    return build_condensed_operator(
        w, sigma_x, sigma_s, jac, RegularizationState(delta_w=1e-6)
    )


def _condensed_one_inequality(namespace):
    """As :func:`_condensed_no_inequalities` plus one inequality Gram row.

    The Woodbury inverse is then only *approximate* (it drops the Gram
    off-diagonal), so the default/auto modes start on Jacobi — the shape the
    auto-promotion mechanics are exercised on.
    """
    jac = Dense(array(namespace, [[1.0, 0.0, 2.0, 0.0, -1.0, 0.5]]))
    sigma_s = Diagonal(array(namespace, [3.0]))
    return _condensed_no_inequalities(namespace, ineq=(jac, sigma_s))


def test_lbfgs_preconditioner_is_exact_inverse_without_inequalities(namespace):
    """Without inequalities ``N = D̃ − U M⁻¹ Uᵀ`` exactly, so Woodbury inverts it."""
    operator = _condensed_no_inequalities(namespace)
    minv = operator.lbfgs_inverse_apply()

    v = array(namespace, [0.5, -1.0, 2.0, -0.25, 1.5, -2.0])
    recovered = minv(operator.matvec(v))
    assert_allclose(namespace, recovered, v, rtol=1e-8, atol=1e-8)


def test_cg_lbfgs_preconditioner_converges_in_one_step(namespace, tol):
    """The exact Woodbury inverse makes preconditioned CG a single-step solve."""
    operator = _condensed_no_inequalities(namespace)
    x_exact = array(namespace, [1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    rhs = operator.matvec(x_exact)

    solver = _solver(method="cg", rtol=1e-10, preconditioner="lbfgs")
    solver.factor(operator)
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, rtol=1e-7, atol=1e-7)
    assert solver.last_iterations == 1


def test_lbfgs_preconditioner_reduces_iterations_with_inequalities(namespace):
    """With inequalities the Woodbury inverse is approximate but still preconditions.

    A larger, ill-conditioned condensed system (wide ``Σ_x`` spectrum) where
    unpreconditioned CG needs many iterations: folding ``Σ_x`` and the L-BFGS
    low-rank into the preconditioner exactly (only the inequality Gram off-diagonal
    is approximated) collapses the iteration count.
    """
    import random

    from ipax.ipm.hessian import LBFGSOperator
    from ipax.ipm.kkt import build_condensed_operator
    from ipax.linalg.regularize import RegularizationState
    from ipax.options import LBFGSOptions

    rng = random.Random(7)
    n = 40
    w = LBFGSOperator(n, LBFGSOptions(memory=8))
    for _ in range(4):
        delta = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        # γ = δ + small noise keeps δᵀγ > 0 (positive curvature pairs).
        gamma = [delta[k] + rng.uniform(0.0, 0.5) + 0.5 for k in range(n)]
        w.update(array(namespace, delta), array(namespace, gamma))

    # Wide diagonal spectrum (≈ 1e3 conditioning) — hard for plain CG.
    sigma_x = Diagonal(
        array(namespace, [10.0 ** (3.0 * k / (n - 1)) for k in range(n)])
    )
    sigma_s = Diagonal(array(namespace, [1.0, 0.5]))
    jac = Dense(
        array(
            namespace,
            [
                [1.0 if k % 3 == 0 else 0.0 for k in range(n)],
                [1.0 if k % 5 == 0 else 0.0 for k in range(n)],
            ],
        )
    )
    operator = build_condensed_operator(
        w, sigma_x, sigma_s, jac, RegularizationState(delta_w=1e-6)
    )
    x_exact = array(namespace, [(-1.0) ** k * (1.0 + k / n) for k in range(n)])
    rhs = operator.matvec(x_exact)

    none = _solver(method="cg", rtol=1e-9, preconditioner="none")
    none.factor(operator)
    x_none = none.solve(rhs)

    lbfgs = _solver(method="cg", rtol=1e-9, preconditioner="lbfgs")
    lbfgs.factor(operator)
    x_lbfgs = lbfgs.solve(rhs)

    assert_allclose(namespace, x_none, x_exact, rtol=1e-6, atol=1e-6)
    assert_allclose(namespace, x_lbfgs, x_exact, rtol=1e-6, atol=1e-6)
    assert lbfgs.last_iterations < none.last_iterations


def test_lbfgs_preconditioner_falls_back_without_lbfgs_structure(namespace, tol):
    """A plain operator has no L-BFGS compact form ⇒ degrade to Jacobi, still solve."""
    A, rhs, x_exact = _spd_system(namespace)
    solver = _solver(method="cg", rtol=1e-12, preconditioner="lbfgs")
    solver.factor(Dense(A))
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, **tol)


def test_auto_starts_with_jacobi_and_stays_on_easy_solves(namespace, tol):
    """``preconditioner="auto"`` begins as Jacobi and does not promote when the
    solve is easy (converges well inside the iteration budget)."""
    A, rhs, x_exact = _spd_system(namespace)
    solver = _solver(method="cg", rtol=1e-12, preconditioner="auto")
    solver.factor(Dense(A))
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, **tol)
    assert "auto:jacobi" in solver.describe()  # never promoted


def test_auto_promotes_to_lbfgs_on_convergence_failure(namespace, tol, monkeypatch):
    """A jacobi solve that cannot meet the tolerance in the iteration budget
    promotes to the L-BFGS Woodbury preconditioner and retries — succeeding.

    The bound-only block would apply its exact inverse outright in Jacobi
    mode; hide that shortcut so the failure-rescue mechanics are what runs.
    """
    operator = _condensed_no_inequalities(namespace)  # exposes lbfgs_inverse_apply
    monkeypatch.setattr(type(operator), "lbfgs_inverse_is_exact", lambda self: False)
    x_exact = array(namespace, [1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    rhs = operator.matvec(x_exact)

    # max_iter=1 is too tight for plain Jacobi CG here, but the exact Woodbury
    # inverse solves in a single step — so the auto retry converges.
    solver = _solver(method="cg", rtol=1e-10, preconditioner="auto", max_iter=1)
    solver.factor(operator)
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, rtol=1e-7, atol=1e-7)
    assert "auto:lbfgs" in solver.describe()  # promoted by the failure
    assert solver.last_iterations == 1


def test_auto_stays_jacobi_and_raises_without_lbfgs_structure(namespace):
    """A plain operator has no L-BFGS compact form: a failed jacobi solve cannot
    be rescued, so auto re-raises without promoting (no pointless retry loop)."""
    A, rhs, _ = _spd_system(namespace)
    solver = _solver(method="cg", rtol=1e-14, preconditioner="auto", max_iter=1)
    solver.factor(Dense(A))

    with pytest.raises(KrylovConvergenceError):
        solver.solve(rhs)
    assert "auto:jacobi" in solver.describe()  # no L-BFGS ⇒ never promoted


def test_auto_stays_jacobi_when_fast_despite_lbfgs_structure(namespace, tol):
    """A fast Jacobi solve must not promote even when the L-BFGS Woodbury
    inverse *is* available: promotion is reserved for solves that struggle.
    (With an inequality row: a bound-only block applies the exact inverse
    outright — see ``test_default_preconditioner_uses_exact_lbfgs_inverse``.)"""
    operator = _condensed_one_inequality(namespace)
    x_exact = array(namespace, [1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    rhs = operator.matvec(x_exact)

    # Generous budget: the handful of CG iterations stays far below the ratio.
    solver = _solver(method="cg", rtol=1e-10, preconditioner="auto", max_iter=10_000)
    solver.factor(operator)
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, rtol=1e-7, atol=1e-7)
    assert "auto:jacobi" in solver.describe()  # available, but never needed


def test_auto_promotes_after_a_slow_but_successful_solve(namespace, tol):
    """A solve that succeeds but burns more than ``auto_switch_ratio`` of the
    budget promotes to L-BFGS for the *next* solve (sticky across solves)."""
    operator = _condensed_one_inequality(namespace)
    x_exact = array(namespace, [1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    rhs = operator.matvec(x_exact)

    # A tiny ratio: any multi-iteration Jacobi solve counts as "slow".
    solver = _solver(
        method="cg", rtol=1e-10, preconditioner="auto", auto_switch_ratio=1e-6
    )
    solver.factor(operator)

    first = solver.solve(rhs)  # Jacobi, several iterations → triggers promotion
    assert_allclose(namespace, first, x_exact, rtol=1e-7, atol=1e-7)
    assert first is not None
    first_iterations = solver.last_iterations
    assert first_iterations > 1
    assert "auto:lbfgs" in solver.describe()

    second = solver.solve(rhs)  # now the Woodbury inverse (approximate here)
    assert_allclose(namespace, second, x_exact, rtol=1e-7, atol=1e-7)
    assert solver.last_iterations < first_iterations


def test_auto_does_not_slow_promote_the_approximate_saddle_block(namespace, tol):
    """A slow-but-successful *saddle* solve must NOT speculatively promote.

    The saddle block preconditioner ``diag(N⁻¹, S⁻¹)`` uses an *approximate* Schur
    diagonal, so on a rank-deficient/ill-conditioned saddle it can yield worse
    steps than the slow-but-stable Jacobi/MINRES solve (observed on the ACOPP
    power-flow cluster). Speculative slow-promotion is therefore restricted to the
    near-exact condensed Woodbury inverse; a saddle stays on Jacobi unless a solve
    outright fails.
    """
    from ipax.ipm.kkt import build_saddle_operator

    d = array(
        namespace,
        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0],
    )
    c_mat = array(
        namespace,
        [
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ],
    )
    saddle = build_saddle_operator(Diagonal(d), Dense(c_mat), 1e-8)
    x_exact = array(
        namespace,
        [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 0.25, -0.25, 3.0, -3.0],
    )
    rhs = saddle.matvec(x_exact)

    # A tiny ratio would promote any multi-iteration solve — but this saddle
    # exposes only the approximate block preconditioner, not the condensed
    # Woodbury, so slow-promotion must decline it.
    solver = _solver(
        method="cg", rtol=1e-10, preconditioner="auto", auto_switch_ratio=1e-6
    )
    solver.factor(saddle)
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, rtol=1e-6, atol=1e-6)
    assert solver.last_iterations > 1  # genuinely slow
    assert "auto:jacobi" in solver.describe()  # but not promoted


def test_auto_does_not_rescue_a_saddle_to_the_block(namespace):
    """A *failed* saddle solve must NOT promote to the approximate block.

    On an equality saddle the block preconditioner's approximate Schur diagonal
    can *diverge* a solve that plain Jacobi handles (S2MPJ HS109/CLNLBEAM/ACOPP).
    Auto only ever promotes to the near-exact condensed Woodbury inverse, which a
    saddle does not expose — so a failed saddle solve stays Jacobi and re-raises
    rather than switching to the block.
    """
    from ipax.ipm.kkt import build_saddle_operator

    condensed = _condensed_no_inequalities(namespace)  # exposes the block, via lbfgs
    c_mat = array(
        namespace,
        [
            [1.0, 0.0, 2.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 3.0],
        ],
    )
    saddle = build_saddle_operator(condensed, Dense(c_mat), 1e-8)
    x_exact = array(namespace, [1.0, -1.0, 2.0, -0.5, 0.5, -2.0, 0.7, -0.3])
    rhs = saddle.matvec(x_exact)

    # max_iter=1: MINRES and the GMRES fallback both fail, so the solve raises.
    solver = _solver(method="cg", preconditioner="auto", rtol=1e-12, max_iter=1)
    solver.factor(saddle)
    with pytest.raises(KrylovConvergenceError):
        solver.solve(rhs)
    assert "auto:jacobi" in solver.describe()  # never promoted to the block


def _small_saddle(namespace):
    from ipax.ipm.kkt import build_saddle_operator

    saddle = build_saddle_operator(
        Diagonal(array(namespace, [2.0, 3.0])),
        Dense(array(namespace, [[1.0, -1.0]])),
        1e-8,
    )
    x_exact = array(namespace, [0.25, -0.5, 1.5])
    return saddle, x_exact, saddle.matvec(x_exact)


def test_saddle_minres_failure_falls_back_to_gmres(namespace, monkeypatch, tol):
    """When MINRES fails on an equality saddle, the default cg route falls back to
    GMRES (unpreconditioned) instead of surfacing the failure — MINRES is fragile
    on ill-conditioned indefinite saddles where GMRES is robust (S2MPJ Task 7).
    """
    saddle, x_exact, rhs = _small_saddle(namespace)

    def boom(self, *args, **kwargs):
        del self, args, kwargs
        raise KrylovConvergenceError("forced MINRES failure")

    monkeypatch.setattr(KrylovSolver, "_preconditioned_minres", boom)
    solver = _solver(method="cg", preconditioner="jacobi", rtol=1e-10)
    solver.factor(saddle)
    x = solver.solve(rhs)

    assert_allclose(namespace, x, x_exact, **tol)
    assert solver.last_method == "gmres"


def test_explicit_minres_is_not_overridden_by_the_gmres_fallback(
    namespace, monkeypatch
):
    """The GMRES fallback is only for the *default* cg route; an explicit
    ``method="minres"`` is honored — a MINRES failure propagates unchanged.
    """
    saddle, _x_exact, rhs = _small_saddle(namespace)

    def boom(self, *args, **kwargs):
        del self, args, kwargs
        raise KrylovConvergenceError("forced MINRES failure")

    monkeypatch.setattr(KrylovSolver, "_preconditioned_minres", boom)
    solver = _solver(method="minres", preconditioner="jacobi")
    solver.factor(saddle)
    with pytest.raises(KrylovConvergenceError):
        solver.solve(rhs)


def test_effective_rtol_uses_fixed_rtol_without_a_residual_hint(namespace):
    """Before any hint (e.g. a standalone solve, no driver) the fixed rtol holds."""
    solver = _solver(rtol=1e-10, adaptive_tol=True)
    assert solver._effective_rtol() == 1e-10


def test_effective_rtol_is_fixed_when_adaptive_disabled(namespace):
    solver = _solver(rtol=1e-9, adaptive_tol=False)
    solver.set_outer_residual(1e3)
    assert solver._effective_rtol() == 1e-9


def test_effective_rtol_forcing_sequence(namespace):
    """clip(η·‖r‖, rtol, cap): loose (capped) when far, η·‖r‖ mid-range, rtol floor."""
    solver = _solver(
        rtol=1e-10, adaptive_tol=True, adaptive_eta=0.1, adaptive_rtol_max=1e-2
    )

    solver.set_outer_residual(1e3)  # far from optimal → capped loose
    assert solver._effective_rtol() == 1e-2

    solver.set_outer_residual(1e-3)  # mid-range → η·‖r‖
    assert abs(solver._effective_rtol() - 1e-4) <= 1e-18

    solver.set_outer_residual(1e-12)  # near optimal → floored at rtol
    assert solver._effective_rtol() == 1e-10


def test_set_outer_residual_ignores_non_finite_and_nonpositive(namespace):
    solver = _solver(rtol=1e-10, adaptive_tol=True, adaptive_rtol_max=1e-2)
    solver.set_outer_residual(1e-3)  # a valid hint → η·‖r‖ = 1e-4 (within [floor, cap])
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        solver.set_outer_residual(bad)  # ignored → keeps the last valid hint
        assert abs(solver._effective_rtol() - 1e-4) <= 1e-18


def test_adaptive_residual_hint_loosens_the_solve(namespace):
    """A large outer-residual hint loosens the inner tolerance, cutting iterations
    vs the tight fixed rtol on the same ill-conditioned diagonal system."""
    d = array(namespace, [10.0**k for k in range(8)])  # wide spectrum
    operator = Diagonal(d)
    x_exact = array(namespace, [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0])
    rhs = operator.matvec(x_exact)

    tight = _solver(method="cg", preconditioner="none", adaptive_tol=False, rtol=1e-12)
    tight.factor(operator)
    tight.solve(rhs)

    loose = _solver(
        method="cg",
        preconditioner="none",
        adaptive_tol=True,
        rtol=1e-12,
        adaptive_eta=0.1,
        adaptive_rtol_max=1e-1,
    )
    loose.set_outer_residual(1e6)  # far from optimal → loose inner solve
    loose.factor(operator)
    loose.solve(rhs)

    assert loose.last_iterations < tight.last_iterations


def test_non_convergence_raises(namespace):
    A, rhs, _ = _spd_system(namespace)
    solver = _solver(method="cg", rtol=1e-14, max_iter=1)
    solver.factor(Dense(A))
    with pytest.raises(KrylovConvergenceError):
        solver.solve(rhs)


@pytest.mark.parametrize("method", ["cg", "minres"])
def test_minres_breakdown_with_large_true_residual_raises(namespace, method):
    operator = Dense(namespace.zeros((2, 2), dtype=array(namespace, [0.0]).dtype))
    rhs = array(namespace, [1.0, 0.0])
    solver = _solver(method=method, rtol=1e-12, max_iter=10)
    solver.factor(operator)

    with pytest.raises(KrylovConvergenceError):
        solver.solve(rhs)


def test_zero_rhs_returns_zero(namespace, tol):
    A, _, _ = _spd_system(namespace)
    solver = _solver(method="cg")
    solver.factor(Dense(A))
    rhs = namespace.zeros((3,), dtype=A.dtype)
    x = solver.solve(rhs)

    assert_allclose(namespace, x, namespace.zeros((3,), dtype=A.dtype), **tol)
    assert solver.last_iterations == 0


def test_factor_required_before_solve(namespace):
    solver = _solver()
    with pytest.raises(RuntimeError):
        solver.solve(array(namespace, [1.0, 2.0, 3.0]))


def test_cg_orthogonal_preconditioner_breakdown_raises_convergence_error(namespace):
    """<r, M^-1 r> can vanish exactly with a nonzero residual (an underflowed
    or non-SPD approximate preconditioner: MGH09LS under pc=auto hit 0/0 at
    the beta update). CG must surface KrylovConvergenceError so the driver
    escalates delta_w instead of crashing with ZeroDivisionError."""
    xp = namespace
    A = array(xp, [[4.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]])
    rhs = array(xp, [1.0, 0.0, 0.0])
    solver = _solver(method="cg", rtol=1e-12, max_iter=10)
    solver.factor(Dense(A))

    def swap_precond(r):
        # returns a vector exactly orthogonal to r = e1 (IEEE-exact zero inner)
        return xp.stack((r[1], r[0], xp.zeros_like(r[2])))

    with pytest.raises(KrylovConvergenceError):
        solver._cg(Dense(A), rhs, xp, 10, 1e-12, swap_precond)


def test_cg_zero_preconditioner_breakdown_raises_convergence_error(namespace):
    """A preconditioner that annihilates the residual (z = 0) is a breakdown of
    the preconditioned inner product, not evidence of indefiniteness: it must
    raise KrylovConvergenceError, not misdiagnose via the curvature test."""
    xp = namespace
    A, rhs, _ = _spd_system(namespace)
    solver = _solver(method="cg", rtol=1e-12, max_iter=10)
    solver.factor(Dense(A))

    with pytest.raises(KrylovConvergenceError):
        solver._cg(Dense(A), rhs, xp, 10, 1e-12, lambda r: xp.zeros_like(r))


def _bound_only_lbfgs_condensed(namespace, *, with_inequality: bool = False):
    import random

    from ipax.ipm.hessian import LBFGSOperator
    from ipax.ipm.kkt import build_condensed_operator
    from ipax.linalg.regularize import RegularizationState
    from ipax.options import LBFGSOptions

    rng = random.Random(5)
    n = 24
    w = LBFGSOperator(n, LBFGSOptions(memory=6))
    for _ in range(4):
        delta = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        gamma = [delta[k] + rng.uniform(0.0, 0.5) + 0.5 for k in range(n)]
        w.update(array(namespace, delta), array(namespace, gamma))
    dtype = array(namespace, [0.0]).dtype
    sigma_x = Diagonal(
        array(namespace, [10.0 ** (3.0 * k / (n - 1)) for k in range(n)])
    )
    if with_inequality:
        jac = Dense(array(namespace, [[1.0 if k % 3 == 0 else 0.0 for k in range(n)]]))
        sigma_s = Diagonal(array(namespace, [2.0]))
    else:
        jac = Dense(namespace.zeros((0, n), dtype=dtype))
        sigma_s = Diagonal(array(namespace, []))
    return build_condensed_operator(
        w, sigma_x, sigma_s, jac, RegularizationState(delta_w=1e-6)
    )


def test_default_preconditioner_uses_exact_lbfgs_inverse_on_bound_only(namespace, tol):
    # Bound-only L-BFGS (the RT shape): the Woodbury inverse is the exact N⁻¹,
    # so the default (Jacobi) mode must apply it instead — one CG iteration
    # rather than ~30, and no O(n·k²) L-BFGS diagonal.
    K = _bound_only_lbfgs_condensed(namespace)
    n = K.shape[0]
    rhs = array(namespace, [(-1.0) ** k * (1.0 + k / n) for k in range(n)])
    solver = _solver()
    solver.factor(K)

    x = solver.solve(rhs)

    assert solver.last_iterations <= 2
    assert solver.describe() == "krylov (cg, pc=lbfgs-exact)"
    residual = K.matvec(x) - rhs
    assert float(norm_inf(namespace, residual)) <= 1e-8 * float(
        norm_inf(namespace, rhs)
    )


def test_default_preconditioner_stays_jacobi_with_inequality_gram(namespace):
    # With an inequality Gram term the Woodbury inverse is only approximate;
    # the default stays Jacobi (auto-promotion handles struggling solves).
    K = _bound_only_lbfgs_condensed(namespace, with_inequality=True)
    n = K.shape[0]
    rhs = array(namespace, [1.0 + k / n for k in range(n)])
    solver = _solver()
    solver.factor(K)

    solver.solve(rhs)

    assert solver.last_iterations > 2
    assert solver.describe() == "krylov (cg, pc=jacobi)"


def test_exact_lbfgs_inverse_can_be_disabled(namespace):
    # The A/B lever: the plain Jacobi diagonal of the pre-change default.
    K = _bound_only_lbfgs_condensed(namespace)
    n = K.shape[0]
    rhs = array(namespace, [1.0 + k / n for k in range(n)])
    solver = _solver(exact_lbfgs_inverse=False)
    solver.factor(K)

    solver.solve(rhs)

    assert solver.last_iterations > 2
    assert solver.describe() == "krylov (cg, pc=jacobi)"


def test_auto_mode_reports_exact_inverse_with_prefix(namespace):
    K = _bound_only_lbfgs_condensed(namespace)
    n = K.shape[0]
    rhs = array(namespace, [1.0 + k / n for k in range(n)])
    solver = _solver(preconditioner="auto")
    solver.factor(K)

    solver.solve(rhs)

    assert solver.last_iterations <= 2
    assert solver.describe() == "krylov (cg, pc=auto:lbfgs-exact)"


def test_exact_inverse_breakdown_retries_on_jacobi_and_stays_there(namespace):
    # A numerically singular L-BFGS middle matrix can make the "exact" apply
    # return garbage; CG then breaks down (zero preconditioned inner product)
    # and the solve must fall back to Jacobi — for this solve and all later ones.
    K = _bound_only_lbfgs_condensed(namespace)
    n = K.shape[0]
    rhs = array(namespace, [(-1.0) ** k * (1.0 + k / n) for k in range(n)])
    dispatches = []
    original = K.lbfgs_inverse_apply

    def _annihilating():
        return lambda r: 0.0 * r

    K.lbfgs_inverse_apply = _annihilating  # type: ignore[method-assign]
    solver = _solver()
    inner = solver._dispatch

    def _counting(*args, **kwargs):
        try:
            return inner(*args, **kwargs)
        finally:  # the flag is set inside the dispatch; sample it on the way out
            dispatches.append(solver._exact_inverse_active)

    solver._dispatch = _counting  # type: ignore[method-assign]
    solver.factor(K)

    x = solver.solve(rhs)

    assert dispatches == [True, False]
    assert solver.describe() == "krylov (cg, pc=jacobi)"
    residual = K.matvec(x) - rhs
    assert float(norm_inf(namespace, residual)) <= 1e-7 * float(
        norm_inf(namespace, rhs)
    )

    # Sticky: the next solve does not try the exact inverse again even though
    # the operator now offers a working one.
    K.lbfgs_inverse_apply = original  # type: ignore[method-assign]
    dispatches.clear()
    solver.solve(rhs)
    assert dispatches == [False]
    assert solver.describe() == "krylov (cg, pc=jacobi)"


def test_exact_inverse_flag_does_not_leak_across_operators(namespace):
    # ``describe`` reflects the *last* solve: a plain operator solved after a
    # bound-only block must not keep reporting ``lbfgs-exact`` (nor take its
    # retry branch).
    solver = _solver()
    K = _bound_only_lbfgs_condensed(namespace)
    n = K.shape[0]
    solver.factor(K)
    solver.solve(array(namespace, [1.0 + k / n for k in range(n)]))
    assert solver.describe() == "krylov (cg, pc=lbfgs-exact)"

    A, rhs, _ = _spd_system(namespace)
    solver.factor(Dense(A))
    solver.solve(rhs)
    assert solver.describe() == "krylov (cg, pc=jacobi)"


def test_preconditioner_none_ignores_exact_lbfgs_inverse(namespace):
    K = _bound_only_lbfgs_condensed(namespace)
    n = K.shape[0]
    rhs = array(namespace, [1.0 + k / n for k in range(n)])
    solver = _solver(preconditioner="none")
    solver.factor(K)

    solver.solve(rhs)

    assert solver.last_iterations > 2
    assert solver.describe() == "krylov (cg, pc=none)"


def test_exact_inverse_dispatches_directly_not_through_cg(namespace):
    # The exact condensed Woodbury inverse is a *direct* solve; wrapping it in
    # CG paid the loop's extra inner-product host syncs and vector ops just to
    # confirm what holds algebraically. The dispatch must apply it directly.
    K = _bound_only_lbfgs_condensed(namespace)
    n = K.shape[0]
    rhs = array(namespace, [(-1.0) ** k * (1.0 + k / n) for k in range(n)])
    solver = _solver()
    solver.factor(K)

    x = solver.solve(rhs)

    assert solver.last_method == "direct"
    assert solver.last_iterations == 1  # one Woodbury apply, verified
    assert solver.describe() == "krylov (cg, pc=lbfgs-exact)"
    residual = K.matvec(x) - rhs
    assert float(norm_inf(namespace, residual)) <= 1e-8 * float(
        norm_inf(namespace, rhs)
    )


def test_direct_dispatch_refines_a_round_off_limited_apply(namespace):
    # CG around the exact inverse implicitly refined when one apply's residual
    # sat above tolerance (ill-conditioned late-barrier systems); the direct
    # dispatch must keep that robustness via iterative refinement, not fail.
    K = _bound_only_lbfgs_condensed(namespace)
    n = K.shape[0]
    rhs = array(namespace, [(-1.0) ** k * (1.0 + k / n) for k in range(n)])
    original = K.lbfgs_inverse_apply()
    # A slightly-off apply: first residual ~1e-6·‖b‖ (above rtol=1e-10), one
    # refinement round lands it at ~1e-12·‖b‖.
    K.lbfgs_inverse_apply = lambda: (  # type: ignore[method-assign]
        lambda r: (1.0 - 1e-6) * original(r)
    )
    solver = _solver()
    solver.factor(K)

    x = solver.solve(rhs)

    assert solver.last_method == "direct"
    assert solver.last_iterations == 2  # apply + one refinement round
    residual = K.matvec(x) - rhs
    assert float(norm_inf(namespace, residual)) <= 1e-8 * float(
        norm_inf(namespace, rhs)
    )


def test_direct_dispatch_falls_back_to_preconditioned_cg_on_stall(namespace):
    # A finite-but-wrong "exact" apply must not be returned unverified: the
    # refinement stalls and the dispatch falls back to CG preconditioned with
    # the same apply — the pre-direct route, whose per-apply Galerkin
    # optimality recovers cases plain refinement cannot. No sticky Jacobi
    # block: the old route never blocked on this input either.
    K = _bound_only_lbfgs_condensed(namespace)
    n = K.shape[0]
    rhs = array(namespace, [(-1.0) ** k * (1.0 + k / n) for k in range(n)])
    # Identity apply: refinement is Richardson iteration and diverges on this
    # ill-scaled Σ_x, but identity-preconditioned CG converges fine.
    K.lbfgs_inverse_apply = lambda: lambda r: r  # type: ignore[method-assign]
    solver = _solver()
    solver.factor(K)

    x = solver.solve(rhs)

    assert solver.last_method == "cg"  # the fallback ran, and it sufficed
    assert solver.describe() == "krylov (cg, pc=lbfgs-exact)"
    assert not solver._exact_inverse_blocked
    residual = K.matvec(x) - rhs
    assert float(norm_inf(namespace, residual)) <= 1e-7 * float(
        norm_inf(namespace, rhs)
    )


def test_direct_dispatch_converts_backend_errors_to_the_fallback(namespace):
    # An exactly singular L-BFGS middle matrix raises a backend-native error
    # inside the Woodbury apply (e.g. ``numpy.linalg.LinAlgError``); the direct
    # path must convert it to KrylovConvergenceError so the established sticky
    # Jacobi fallback handles it, instead of escaping the driver's δ_w ladder.
    K = _bound_only_lbfgs_condensed(namespace)
    n = K.shape[0]
    rhs = array(namespace, [(-1.0) ** k * (1.0 + k / n) for k in range(n)])

    def _raising():
        def _apply(r):
            raise ValueError("singular middle matrix")

        return _apply

    K.lbfgs_inverse_apply = _raising  # type: ignore[method-assign]
    solver = _solver()
    solver.factor(K)

    x = solver.solve(rhs)

    assert solver.describe() == "krylov (cg, pc=jacobi)"
    assert solver._exact_inverse_blocked
    residual = K.matvec(x) - rhs
    assert float(norm_inf(namespace, residual)) <= 1e-7 * float(
        norm_inf(namespace, rhs)
    )


def test_direct_dispatch_zero_rhs_returns_zero(namespace):
    K = _bound_only_lbfgs_condensed(namespace)
    n = K.shape[0]
    solver = _solver()
    solver.factor(K)

    x = solver.solve(namespace.zeros((n,), dtype=array(namespace, [0.0]).dtype))

    assert bool(namespace.all(x == 0.0))
    assert solver.last_iterations == 0
    assert solver.last_method == "direct"


def test_explicit_gmres_still_uses_the_exact_inverse_as_preconditioner(namespace):
    # The direct dispatch lives on the default CG route only; an explicit
    # ``method="gmres"`` keeps its documented behavior — the exact inverse
    # serves as GMRES's left preconditioner via ``_make_preconditioner``.
    K = _bound_only_lbfgs_condensed(namespace)
    n = K.shape[0]
    rhs = array(namespace, [(-1.0) ** k * (1.0 + k / n) for k in range(n)])
    solver = _solver(method="gmres")
    solver.factor(K)

    x = solver.solve(rhs)

    assert solver.last_method == "gmres"
    assert solver.describe() == "krylov (gmres, pc=lbfgs-exact)"
    residual = K.matvec(x) - rhs
    assert float(norm_inf(namespace, residual)) <= 1e-8 * float(
        norm_inf(namespace, rhs)
    )


def test_auto_promotion_probe_skipped_on_fast_solves(namespace):
    # The auto-mode availability probe (``lbfgs_inverse_apply``) builds the full
    # O(n·k²) Woodbury factors; a fast successful solve must not pay it — the
    # cheap slowness test decides first.
    probes: list[int] = []

    class _ProbeCounting(Dense):
        def lbfgs_inverse_apply(self):
            probes.append(1)
            raise NotImplementedError

    A, rhs, _ = _spd_system(namespace)
    solver = _solver(preconditioner="auto")
    solver.factor(_ProbeCounting(A))

    solver.solve(rhs)

    assert solver.last_iterations <= 5  # well-conditioned: genuinely fast
    assert probes == []


def test_non_convergence_error_carries_the_partial_iterate(namespace, tol):
    """A work-capped solve hands its last iterate to the caller.

    Feasibility restoration reuses it as a truncated-CG trial direction
    (Steihaug 1983) instead of discarding the Krylov work and climbing the LM
    ladder; the KKT driver keeps ignoring it and escalating ``δ_w``.
    """
    A, rhs, _ = _spd_system(namespace)
    solver = _solver(method="cg", preconditioner="none", rtol=1e-14, max_iter=1)
    solver.factor(Dense(A))
    with pytest.raises(KrylovConvergenceError) as info:
        solver.solve(rhs)
    iterate = info.value.iterate
    assert iterate is not None
    # One unpreconditioned CG step from zero is exact-line-search steepest
    # descent along the right-hand side: α = ⟨b, b⟩ / ⟨b, A b⟩.
    xp = namespace
    a_rhs = xp.matmul(A, rhs)
    alpha = float(xp.sum(rhs * rhs)) / float(xp.sum(rhs * a_rhs))
    assert_allclose(namespace, iterate, alpha * rhs, **tol)


@pytest.mark.parametrize("method", ["gmres", "minres"])
def test_non_convergence_error_carries_a_finite_iterate(namespace, method):
    A, rhs, _ = _spd_system(namespace)
    kwargs = {"gmres_restart": 1} if method == "gmres" else {}
    solver = _solver(method=method, rtol=1e-14, max_iter=1, **kwargs)
    solver.factor(Dense(A))
    with pytest.raises(KrylovConvergenceError) as info:
        solver.solve(rhs)
    iterate = info.value.iterate
    assert iterate is not None
    assert iterate.shape == rhs.shape
    assert bool(namespace.all(namespace.isfinite(iterate)))


@pytest.mark.parametrize(
    "failure",
    [
        MemoryError("host allocation failed"),
        type("OutOfMemoryError", (RuntimeError,), {})("CUDA out of memory"),
    ],
    ids=["MemoryError", "cuda-oom"],
)
def test_exact_inverse_apply_resource_failures_propagate(namespace, failure):
    """A resource failure inside the Woodbury apply is not a singular window.

    Relabeling it as ``KrylovConvergenceError`` would sticky-disable the exact
    inverse and retry on Jacobi CG — hiding an out-of-memory condition behind a
    "slow solve" for the rest of the run (torch's CUDA OOM is a RuntimeError,
    not a MemoryError).
    """
    K = _bound_only_lbfgs_condensed(namespace)
    n = K.shape[0]
    rhs = array(namespace, [(-1.0) ** k * (1.0 + k / n) for k in range(n)])

    def _raising():
        def apply(v):
            del v
            raise failure

        return apply

    K.lbfgs_inverse_apply = _raising  # type: ignore[method-assign]
    solver = _solver()
    solver.factor(K)

    with pytest.raises(type(failure)):
        solver.solve(rhs)
    assert not solver._exact_inverse_blocked
