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


def test_gmres_non_convergence_raises(namespace):
    A, rhs, _ = _spd_system(namespace)
    solver = _solver(method="gmres", rtol=1e-14, max_iter=1, gmres_restart=1)
    solver.factor(Dense(A))
    with pytest.raises(KrylovConvergenceError):
        solver.solve(rhs)


def _condensed_no_inequalities(namespace):
    """A condensed operator ``N = B + Σ_x + δ_w I`` with an L-BFGS Hessian ``B``."""
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
    empty_sigma_s = Diagonal(array(namespace, []))
    empty_jac = Dense(namespace.zeros((0, n), dtype=array(namespace, [0.0]).dtype))
    return build_condensed_operator(
        w, sigma_x, empty_sigma_s, empty_jac, RegularizationState(delta_w=1e-6)
    )


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


def test_auto_promotes_to_lbfgs_on_convergence_failure(namespace, tol):
    """A jacobi solve that cannot meet the tolerance in the iteration budget
    promotes to the L-BFGS Woodbury preconditioner and retries — succeeding."""
    operator = _condensed_no_inequalities(namespace)  # exposes lbfgs_inverse_apply
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


def test_auto_promotes_after_a_slow_but_successful_solve(namespace, tol):
    """A solve that succeeds but burns more than ``auto_switch_ratio`` of the
    budget promotes to L-BFGS for the *next* solve (sticky across solves)."""
    operator = _condensed_no_inequalities(namespace)
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
    assert solver.last_iterations > 1
    assert "auto:lbfgs" in solver.describe()

    second = solver.solve(rhs)  # now the exact Woodbury inverse: one step
    assert_allclose(namespace, second, x_exact, rtol=1e-7, atol=1e-7)
    assert solver.last_iterations == 1


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
