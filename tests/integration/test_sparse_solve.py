"""End-to-end solves through the sparse-direct route (``linsolve="sparse"``).

The driver hands its condensed/saddle :class:`LinearOperator` to whichever solver
``select_solver`` injected; with ``linsolve="sparse"`` that is the backend-neutral
:class:`~ipax.linalg.sparse.SparseDirectSolver`, which emits the operator's COO
structure and factors it through the per-backend adapter. This covers bound-,
equality- and inequality-constrained problems: bounds and equalities assemble as
pure blocks, while inequalities keep ``∇g`` explicit with a ``−Σ_s⁻¹`` block as
the indefinite augmented border (so the condensed Gram term is never densified).
On a sparse LDLᵀ backend that reports inertia (Feral) the IPM additionally runs
the IPOPT inertia-guided δ_w correction.

The sparse route must (a) reach the known optimum and (b) agree with the dense
reference route to tolerance.
"""

from __future__ import annotations

import pytest

from ipax import FunctionProblem
from ipax.backend.sparse import get_sparse_adapter
from ipax.options import OptimalityConditionOptions, Options
from ipax.result import Status
from ipax.solve import solve
from ipax.testing.problems import (
    HS6,
    HS7,
    HS35,
    HS43,
    BoundConstrainedQP,
    EqualityConstrainedQP,
    UnconstrainedQuadratic,
)
from tests._helpers import array, assert_allclose, float_dtype, norm_inf

pytestmark = pytest.mark.sparse


def _require_sparse(namespace):
    if get_sparse_adapter(namespace) is None:
        pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")


def _cases(namespace):
    Q = array(namespace, [[4.0, 1.0], [1.0, 3.0]])
    b = array(namespace, [1.0, 2.0])
    return [
        (UnconstrainedQuadratic(Q, b, namespace), array(namespace, [0.0, 0.0])),
        (EqualityConstrainedQP(namespace), array(namespace, [0.2, 0.1])),
        (BoundConstrainedQP(namespace), array(namespace, [0.5, 0.5])),
        (HS6(namespace), array(namespace, [-1.2, 1.0])),
        (HS7(namespace), array(namespace, [0.5, 1.5])),
        # Inequalities (indefinite augmented border): bounds + linear inequality,
        # and three quadratic inequalities.
        (HS35(namespace), array(namespace, [0.5, 0.5, 0.5])),
        (HS43(namespace), array(namespace, [0.0, 0.0, 0.0, 0.0])),
    ]


def test_sparse_route_reaches_known_optimum(namespace):
    _require_sparse(namespace)
    for problem, x0 in _cases(namespace):
        result = solve(problem, x0, options=Options(linsolve="sparse"))
        assert result.status is Status.OPTIMAL, type(problem).__name__
        assert_allclose(
            namespace,
            result.x,
            problem.known_solution(),
            rtol=1e-6,
            atol=1e-6,
        )


def test_sparse_route_agrees_with_dense_route(namespace):
    _require_sparse(namespace)
    for problem, x0 in _cases(namespace):
        sparse = solve(problem, x0, options=Options(linsolve="sparse"))
        dense = solve(problem, x0, options=Options(linsolve="dense"))
        assert sparse.status is dense.status is Status.OPTIMAL
        diff = norm_inf(namespace, sparse.x - dense.x)
        assert diff <= 1e-6, f"{type(problem).__name__}: |Δx|∞={diff}"


def _bound_ls_lbfgs(namespace):
    """Bound-constrained least squares with only objective+gradient ⇒ L-BFGS.

    No analytic Hessian is supplied, so the resolver fills it with L-BFGS — the
    matrix-free, dense low-rank Hessian. The sparse-direct route must factor it
    via the IPOPT-style low-rank border, never by densifying.
    """
    a = array(
        namespace,
        [
            [3.0, 1.0, 0.0, 0.5],
            [1.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.5, 0.0, 1.0, 3.0],
            [1.0, 1.0, 0.0, 1.0],
        ],
    )
    b = array(namespace, [1.0, 2.0, 0.5, 1.5, 1.0])
    a_t = namespace.permute_dims(a, (1, 0))

    def objective(x):
        r = namespace.matmul(a, x) - b
        return 0.5 * namespace.sum(r * r)

    def gradient(x):
        return namespace.matmul(a_t, namespace.matmul(a, x) - b)

    lower = namespace.zeros((4,), dtype=float_dtype(namespace))
    problem = FunctionProblem(4, objective, gradient=gradient, bounds=(lower, None))
    return problem, array(namespace, [0.5, 0.5, 0.5, 0.5])


def test_sparse_route_factors_lbfgs_via_low_rank_border(namespace):
    # The configuration that used to crash: matrix-free L-BFGS Hessian + sparse
    # direct solver. IPOPT handles this by bordering the low-rank term; so do we.
    _require_sparse(namespace)
    problem, x0 = _bound_ls_lbfgs(namespace)
    opts = {"hessian": "lbfgs", "max_iter": 200}  # default optimality: 1e-8 KKT
    sparse = solve(problem, x0, options=Options(linsolve="sparse", **opts))
    dense = solve(problem, x0, options=Options(linsolve="dense", **opts))

    assert sparse.status is Status.OPTIMAL
    assert sparse.derivative_sources.hessian == "lbfgs"
    assert dense.status is Status.OPTIMAL
    diff = norm_inf(namespace, sparse.x - dense.x)
    assert diff <= 1e-6, f"sparse vs dense |Δx|∞={diff}"


def _ineq_ls_lbfgs(namespace):
    """Bound + inequality least squares, objective+gradient only ⇒ L-BFGS.

    Exercises both borders at once (the RT pattern): a limited-memory Hessian
    *and* an inequality constraint, stacked onto the same sparse factor.
    """
    problem, x0 = _bound_ls_lbfgs(namespace)
    base_obj = problem.objective
    base_grad = problem.gradient
    lower = namespace.zeros((4,), dtype=float_dtype(namespace))
    g_row = array(namespace, [1.0, 1.0, 1.0, 1.0])

    def ineq(x):  # Σ x ≤ 2
        return namespace.reshape(namespace.sum(g_row * x) - 2.0, (1,))

    def ineq_jac(x):
        return namespace.reshape(g_row, (1, 4))

    full = FunctionProblem(
        4,
        base_obj,
        gradient=base_grad,
        bounds=(lower, None),
        ineq_constraints=ineq,
        ineq_jacobian=ineq_jac,
    )
    return full, x0


def test_sparse_route_stacks_lbfgs_and_inequality_borders(namespace):
    _require_sparse(namespace)
    problem, x0 = _ineq_ls_lbfgs(namespace)
    opts = {"hessian": "lbfgs", "max_iter": 300}  # default optimality: 1e-8 KKT
    sparse = solve(problem, x0, options=Options(linsolve="sparse", **opts))
    dense = solve(problem, x0, options=Options(linsolve="dense", **opts))

    assert sparse.status is Status.OPTIMAL
    assert dense.status is Status.OPTIMAL
    diff = norm_inf(namespace, sparse.x - dense.x)
    assert diff <= 1e-6, f"sparse vs dense |Δx|∞={diff}"


def test_sparse_route_factors_matrix_free_diag_low_rank_hessian(namespace):
    # The synthetic RT problem's Hessian is a matrix-free diag+low-rank operator
    # (neither assemblable nor L-BFGS). Via the generic diagonal_low_rank_form
    # hook it now factors through the sparse route, with inequality caps stacked.
    _require_sparse(namespace)
    from ipax.testing.problems import make_rt_like_problem

    problem = make_rt_like_problem(
        namespace, n_vars=60, n_structures=5, density=0.2, seed=0
    )
    x0 = namespace.ones((60,), dtype=float_dtype(namespace)) * 0.1
    opts = {
        "optimality": OptimalityConditionOptions(
            dual_inf_tol=1e-7, constr_viol_tol=1e-7, compl_inf_tol=1e-7
        ),
        "max_iter": 400,
    }
    sparse = solve(problem, x0, options=Options(linsolve="sparse", **opts))
    dense = solve(problem, x0, options=Options(linsolve="dense", **opts))

    assert sparse.status is Status.OPTIMAL
    assert dense.status is Status.OPTIMAL
    assert sparse.derivative_sources.hessian == "exact"  # matrix-free analytic
    assert norm_inf(namespace, sparse.x - dense.x) <= 1e-6


def _saddle_trap_qp(namespace):
    """Nonconvex bound QP whose only stationary interior point is a saddle.

    ``f(x) = x₀x₁`` has the indefinite Hessian ``[[0, 1], [1, 0]]`` (eigenvalues
    ±1). On the box ``[−1, 1]²`` the minimizers are the vertices ``(1, −1)`` and
    ``(−1, 1)`` (``f = −1``); the origin is a *saddle* KKT point (``∇f = 0``, no
    active bound). A sparse LDLᵀ factor *succeeds* on the indefinite condensed
    block, so without the inertia check the Newton step is not a descent
    direction and the iterates stall at the origin (``f = 0``). The inertia-guided
    δ_w correction restores a PD ``N`` and steers the solve to a true minimizer —
    this test is the regression guard for that.
    """
    hess = array(namespace, [[0.0, 1.0], [1.0, 0.0]])

    def objective(x):
        return x[0] * x[1]

    def gradient(x):
        return namespace.stack((x[1], x[0]))

    def lagrangian_hessian(x, y_eq, y_ineq, sigma=1.0):
        del x, y_eq, y_ineq
        return sigma * hess

    lower = -namespace.ones((2,), dtype=float_dtype(namespace))
    upper = namespace.ones((2,), dtype=float_dtype(namespace))
    problem = FunctionProblem(
        2,
        objective,
        gradient=gradient,
        bounds=(lower, upper),
        lagrangian_hessian=lagrangian_hessian,
    )
    return problem, array(namespace, [0.3, 0.31])


def test_sparse_route_inertia_correction_escapes_saddle(namespace):
    _require_sparse(namespace)
    # Inertia-guided correction needs an inertia-revealing LDLᵀ backend; the
    # SuperLU fallback factors the indefinite system silently and cannot guide it.
    pytest.importorskip("feral")
    problem, x0 = _saddle_trap_qp(namespace)
    result = solve(
        problem, x0, options=Options(linsolve="sparse", hessian="exact", max_iter=300)
    )

    assert result.status is Status.OPTIMAL
    # Reaches a true minimizer (f = −1, a box vertex), not the saddle at f = 0.
    assert abs(float(result.objective) - (-1.0)) <= 1e-5
    assert abs(float(result.x[0] * result.x[1]) - (-1.0)) <= 1e-5


def test_sparse_lbfgs_route_inertia_covers_saddle(namespace):
    """L-BFGS route on the saddle QP: the folded ``In(M)`` target extends the
    inertia guard to the diagonal-plus-low-rank Hessian without regressing it.

    ``expected_inertia`` now returns ``(n + M₊, M₋ + m_I, 0)`` for the L-BFGS
    compact block instead of ``None`` (inertia-route-status gap 1), so the sparse
    LDLᵀ inertia check is active on this route too. The solve must still reach a
    true minimizer (``f = −1``), not stall at the origin saddle.
    """
    _require_sparse(namespace)
    pytest.importorskip("feral")
    problem, x0 = _saddle_trap_qp(namespace)
    result = solve(
        problem, x0, options=Options(linsolve="sparse", hessian="lbfgs", max_iter=500)
    )

    assert result.status is Status.OPTIMAL
    assert abs(float(result.objective) - (-1.0)) <= 1e-5
    assert abs(float(result.x[0] * result.x[1]) - (-1.0)) <= 1e-5


def test_sparse_route_unavailable_backend_errors(namespace):
    # The facade resolves the adapter from the operator's namespace at factor
    # time; a backend without a registered adapter must fail clearly rather than
    # silently producing a wrong step.
    if get_sparse_adapter(namespace) is not None:
        pytest.skip("backend has a sparse adapter")
    problem = EqualityConstrainedQP(namespace)
    with pytest.raises(RuntimeError, match="sparse"):
        solve(problem, array(namespace, [0.2, 0.1]), options=Options(linsolve="sparse"))
