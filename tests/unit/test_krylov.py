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
