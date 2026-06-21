"""Integration: solve curated problems through the matrix-free Krylov route.

These mirror the dense-route integration checks but force ``linsolve="krylov"``
so the whole IPM iteration runs without ever materializing a KKT matrix
(Breedveld 2017, eq. 18 condensed system solved by CG; the equality saddle by
MINRES). The returned points must satisfy the same KKT oracles as the dense
reference solver.
"""

from __future__ import annotations

import pytest

from ipax import FunctionProblem, Options, Status, solve
from ipax.backend.operators import LinearOperator, MatrixFreeJacobian
from ipax.linalg.krylov import KrylovSolver
from ipax.options import KrylovOptions
from ipax.problem.base import Problem
from ipax.testing.backends import import_namespace
from ipax.testing.problems import (
    BoundConstrainedQP,
    EqualityConstrainedQP,
    UnconstrainedQuadratic,
    make_rt_like_problem,
)
from tests._helpers import array, assert_allclose


def _krylov_options() -> Options:
    return Options(hessian="exact", linsolve="krylov")


def _assert_optimal(result) -> None:
    assert result.status is Status.OPTIMAL
    assert result.success
    assert result.kkt_error <= 1e-6
    assert result.constraint_violation <= 1e-6


def test_krylov_unconstrained_quadratic(namespace):
    Q = array(namespace, [[4.0, 1.0], [1.0, 3.0]])
    b = array(namespace, [1.0, 2.0])
    problem = UnconstrainedQuadratic(Q, b, namespace)

    result = solve(problem, array(namespace, [0.0, 0.0]), options=_krylov_options())

    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)


def test_krylov_bound_constrained_qp(namespace):
    problem = BoundConstrainedQP(namespace)

    result = solve(problem, array(namespace, [0.25, 0.75]), options=_krylov_options())

    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)


def test_krylov_inequality_qp(namespace):
    class InequalityQP(Problem):
        @property
        def n_vars(self) -> int:
            return 1

        def objective(self, x):
            return 0.5 * (x[0] - 2.0) * (x[0] - 2.0)

        def gradient(self, x):
            return namespace.stack((x[0] - 2.0,))

        def ineq_constraints(self, x):
            return namespace.stack((x[0] - 1.0,))

        def ineq_jacobian(self, x):
            del x
            return array(namespace, [[1.0]])

        def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
            del y_eq, y_ineq
            return sigma * array(namespace, [[1.0 + 0.0 * x[0]]])

    result = solve(InequalityQP(), array(namespace, [0.5]), options=_krylov_options())

    _assert_optimal(result)
    assert_allclose(namespace, result.x, array(namespace, [1.0]), rtol=1e-6, atol=1e-6)


def test_krylov_jacobi_preconditioner_cuts_total_iterations(namespace, monkeypatch):
    """End-to-end scaling win: Jacobi engages mid-solve on the condensed system.

    An ill-conditioned (κ ≈ 1000) bound-constrained QP with a diagonal Hessian
    that *exposes* its diagonal, so the condensed operator builds a Jacobi
    preconditioner matrix-free. Preconditioning must collapse the total CG work.
    """
    n = 24
    diag = array(namespace, [1.0 + (1000.0 - 1.0) * k / (n - 1) for k in range(n)])
    b = array(namespace, [0.5 if k % 2 else -0.5 for k in range(n)])
    hessian = namespace.eye(n, dtype=diag.dtype) * diag

    class IllConditionedBoundedQP(Problem):
        @property
        def n_vars(self) -> int:
            return n

        def bounds(self):
            return namespace.zeros((n,), dtype=diag.dtype), None

        def objective(self, x):
            return 0.5 * namespace.sum(diag * x * x) - namespace.sum(b * x)

        def gradient(self, x):
            return diag * x - b

        def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
            del x, y_eq, y_ineq
            return sigma * hessian

    problem = IllConditionedBoundedQP()
    x0 = namespace.ones((n,), dtype=diag.dtype)
    original_solve = KrylovSolver.solve

    def run(preconditioner: str):
        counts: list[int] = []

        def counting_solve(self, rhs):
            result = original_solve(self, rhs)
            counts.append(self.last_iterations)
            return result

        monkeypatch.setattr(KrylovSolver, "solve", counting_solve)
        result = solve(
            problem,
            x0,
            options=Options(
                hessian="exact",
                linsolve="krylov",
                krylov=KrylovOptions(preconditioner=preconditioner),
            ),
        )
        monkeypatch.undo()
        return result, sum(counts)

    none_result, none_iters = run("none")
    jacobi_result, jacobi_iters = run("jacobi")

    assert none_result.status is Status.OPTIMAL
    assert jacobi_result.status is Status.OPTIMAL
    # The preconditioned solve does an order of magnitude less linear-algebra work.
    assert jacobi_iters * 5 < none_iters


def test_krylov_equality_constrained_qp_uses_minres(namespace):
    """Equalities form the indefinite saddle — the CG path falls back to MINRES."""
    problem = EqualityConstrainedQP(namespace)

    result = solve(problem, array(namespace, [0.9, 0.1]), options=_krylov_options())

    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert_allclose(
        namespace, result.y_eq, problem.known_multiplier(), rtol=1e-6, atol=1e-6
    )


def test_gmres_solves_bound_constrained_qp(namespace):
    """The GMRES route drives the full IPM to the same optimum as CG/MINRES."""
    problem = BoundConstrainedQP(namespace)
    options = Options(
        hessian="exact",
        linsolve="krylov",
        krylov=KrylovOptions(method="gmres"),
    )

    result = solve(problem, array(namespace, [0.25, 0.75]), options=options)

    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)


def test_lbfgs_preconditioner_solves_bound_constrained_qp(namespace):
    """End-to-end L-BFGS Hessian + L-BFGS-aware preconditioner on the Krylov route.

    The problem supplies no analytic Hessian, so ``hessian="lbfgs"`` actually
    builds the compact L-BFGS operator the preconditioner then inverts.
    """
    reference = BoundConstrainedQP(namespace)
    lower, upper = reference.bounds()

    def objective(x):
        diff = x - reference.center
        return 0.5 * namespace.sum(diff * diff)

    def gradient(x):
        return x - reference.center

    problem = FunctionProblem(2, objective, gradient=gradient, bounds=(lower, upper))
    options = Options(
        hessian="lbfgs",
        linsolve="krylov",
        krylov=KrylovOptions(preconditioner="lbfgs"),
    )

    result = solve(problem, array(namespace, [0.25, 0.75]), options=options)

    _assert_optimal(result)
    assert result.derivative_sources.hessian == "lbfgs"
    assert_allclose(
        namespace, result.x, reference.known_solution(), rtol=1e-6, atol=1e-6
    )


def test_krylov_solves_synthetic_rt_like_problem(namespace):
    """A block-structured, matrix-free RT-like NLP solves to KKT."""
    n = 200
    problem = make_rt_like_problem(namespace, n, n_structures=6, density=0.2, seed=3)
    x0 = namespace.full((n,), 0.01, dtype=array(namespace, [0.0]).dtype)

    result = solve(problem, x0, options=_krylov_options())

    _assert_optimal(result)
    # Bounds and dose caps are honored at the returned point.
    assert bool(namespace.all(result.x >= -1e-7))
    assert float(namespace.max(problem.ineq_constraints(result.x))) <= 1e-6


def test_krylov_solve_never_materializes_the_kkt_matrix(namespace, monkeypatch):
    """The headline matrix-free guarantee: no ``n×n`` KKT matrix is ever formed.

    Densifying an operator goes through ``LinearOperator.matmat`` (the dense
    solver materializes ``N`` that way). The Krylov route uses only matvecs, so
    its ``matmat`` count must be exactly zero — in contrast to the dense route.
    """
    n = 150
    problem = make_rt_like_problem(namespace, n, n_structures=5, density=0.2, seed=1)
    x0 = namespace.full((n,), 0.01, dtype=array(namespace, [0.0]).dtype)

    counter = {"matmat": 0}
    original_matmat = LinearOperator.matmat

    def counting_matmat(self, V):
        counter["matmat"] += 1
        return original_matmat(self, V)

    monkeypatch.setattr(LinearOperator, "matmat", counting_matmat)

    krylov_result = solve(
        problem, x0, options=Options(hessian="exact", linsolve="krylov")
    )
    krylov_matmats = counter["matmat"]

    counter["matmat"] = 0
    dense_result = solve(
        problem, x0, options=Options(hessian="exact", linsolve="dense")
    )
    dense_matmats = counter["matmat"]

    assert krylov_result.status is Status.OPTIMAL
    assert dense_result.status is Status.OPTIMAL
    assert krylov_matmats == 0  # fully matrix-free
    assert dense_matmats > 0  # the dense route does densify (contrast)


def test_krylov_operator_errors_propagate(namespace):
    """User/operator callback bugs are not swallowed as numerical failures."""

    class BadHessianQP(Problem):
        @property
        def n_vars(self) -> int:
            return 1

        def objective(self, x):
            return 0.5 * (x[0] - 1.0) * (x[0] - 1.0)

        def gradient(self, x):
            return namespace.stack((x[0] - 1.0,))

        def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
            del x, y_eq, y_ineq, sigma

            def matvec(v):
                del v
                raise ValueError("bad Hessian matvec")

            return MatrixFreeJacobian((1, 1), matvec, matvec)

    with pytest.raises(ValueError, match="bad Hessian matvec"):
        solve(
            BadHessianQP(),
            array(namespace, [0.0]),
            options=Options(hessian="exact", linsolve="krylov"),
        )


@pytest.mark.slow
def test_krylov_solves_10k_rt_like_problem_without_materializing(monkeypatch):
    """Scale smoke: the smallest headline RT-like size stays matrix-free."""
    try:
        namespace = import_namespace("numpy")
    except ImportError:
        pytest.skip("NumPy backend is not installed")

    n = 10_000
    problem = make_rt_like_problem(namespace, n, n_structures=8, density=0.2, seed=11)
    x0 = namespace.full((n,), 0.01, dtype=array(namespace, [0.0]).dtype)

    counter = {"matmat": 0}
    original_matmat = LinearOperator.matmat

    def counting_matmat(self, V):
        counter["matmat"] += 1
        return original_matmat(self, V)

    monkeypatch.setattr(LinearOperator, "matmat", counting_matmat)

    result = solve(problem, x0, options=Options(hessian="exact", linsolve="krylov"))

    _assert_optimal(result)
    assert counter["matmat"] == 0


@pytest.mark.gpu
def test_krylov_gpu_smoke_torch_cuda():
    """Optional GPU smoke for environments with CUDA-enabled PyTorch."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    namespace = import_namespace("torch")

    device = torch.device("cuda")
    dtype = namespace.float64
    q = namespace.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=dtype, device=device)
    b = namespace.asarray([1.0, 2.0], dtype=dtype, device=device)
    x0 = namespace.asarray([0.0, 0.0], dtype=dtype, device=device)
    problem = UnconstrainedQuadratic(q, b, namespace)

    result = solve(problem, x0, options=_krylov_options())

    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
