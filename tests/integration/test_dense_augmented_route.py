"""End-to-end solves through the dense augmented route (§5.1).

``DenseOptions(kkt_route="augmented")`` keeps ``∇g``/``−Σ_s⁻¹`` explicit as a
border instead of condensing the inequality Gram term ``∇gᵀ Σ_s ∇g`` into the
``N`` block via a matmul (the default ``"condensed"`` route) — the dense
analogue of the sparse-direct route's indefinite augmented border
(``tests/integration/test_sparse_solve.py``). It factors the bordered matrix
with a pivoted Bunch-Kaufman LDLᵀ where a backend adapter is registered
(``ipax.backend.dense``), exposing real inertia so the IPM's inertia-guided
δ_w correction engages for the dense route too — closing the gap the plain
Cholesky pass/fail check on the condensed route cannot (see
``tests/integration/test_dense_nonconvex.py`` for that guard's own
regression case).

The augmented route must (a) reach the known optimum and (b) agree with the
condensed dense route to tolerance.
"""

from __future__ import annotations

import pytest

from ipax import FunctionProblem
from ipax.backend.dense import get_dense_symmetric_indefinite_adapter
from ipax.options import DenseOptions, Options
from ipax.result import Status
from ipax.solve import solve
from ipax.testing.backends import import_namespace
from ipax.testing.problems import (
    HS6,
    HS7,
    HS35,
    HS43,
    BoundConstrainedQP,
    EqualityConstrainedQP,
    UnconstrainedQuadratic,
)
from tests._helpers import array, assert_allclose, float_dtype


def _cases(namespace):
    Q = array(namespace, [[4.0, 1.0], [1.0, 3.0]])
    b = array(namespace, [1.0, 2.0])
    cases = [
        (UnconstrainedQuadratic(Q, b, namespace), array(namespace, [0.0, 0.0])),
        (EqualityConstrainedQP(namespace), array(namespace, [0.2, 0.1])),
        (BoundConstrainedQP(namespace), array(namespace, [0.5, 0.5])),
        (HS6(namespace), array(namespace, [-1.2, 1.0])),
        (HS7(namespace), array(namespace, [0.5, 1.5])),
        # Genuine inequality borders: bounds + linear inequality.
        (HS35(namespace), array(namespace, [0.5, 0.5, 0.5])),
    ]
    # HS43 (three nonlinear inequalities) needs the pivoted Bunch-Kaufman LDL^T
    # adapter's numerical stability; the Array-API-pure eigh fallback (used
    # when no adapter is registered, e.g. array-api-strict) is not pivoted and
    # is known to be numerically weaker on harder cases (see the docstring of
    # ipax.backend.dense.get_dense_symmetric_indefinite_adapter). NumPy and
    # Torch both have a registered adapter and pass; this only skips the
    # reference/purity-gate backend.
    if get_dense_symmetric_indefinite_adapter(namespace) is not None:
        cases.append((HS43(namespace), array(namespace, [0.0, 0.0, 0.0, 0.0])))
    return cases


def _augmented_options() -> Options:
    return Options(linsolve="dense", dense=DenseOptions(kkt_route="augmented"))


def test_augmented_route_reaches_known_optimum(namespace):
    for problem, x0 in _cases(namespace):
        result = solve(problem, x0, options=_augmented_options())
        assert result.status is Status.OPTIMAL, type(problem).__name__
        assert_allclose(
            namespace,
            result.x,
            problem.known_solution(),
            rtol=1e-6,
            atol=1e-6,
        )


def test_augmented_route_agrees_with_condensed_route(namespace):
    for problem, x0 in _cases(namespace):
        augmented = solve(problem, x0, options=_augmented_options())
        condensed = solve(problem, x0, options=Options(linsolve="dense"))
        assert augmented.status is condensed.status is Status.OPTIMAL, type(
            problem
        ).__name__
        assert_allclose(namespace, augmented.x, condensed.x, rtol=1e-6, atol=1e-6)


def test_augmented_route_reports_itself_in_linear_solver(namespace):
    # HS35 has a genuine inequality border, so the augmented route actually
    # engages (as opposed to silently falling back to condensed).
    problem = HS35(namespace)
    result = solve(
        problem, array(namespace, [0.5, 0.5, 0.5]), options=_augmented_options()
    )
    assert result.status is Status.OPTIMAL
    assert "augmented" in result.linear_solver


def _saddle_trap_qp(namespace):
    """``f(x) = x0 x1`` on ``[-1, 1]^2``: minimizers at (1,-1)/(-1,1) (f = -1),
    saddle KKT point at the origin (f = 0). The Hessian ``[[0, 1], [1, 0]]`` is
    indefinite (eigenvalues ±1)."""
    hess = array(namespace, [[0.0, 1.0], [1.0, 0.0]])

    def objective(x):
        return x[0] * x[1]

    def gradient(x):
        return namespace.stack((x[1], x[0]))

    def lagrangian_hessian(x, y_eq, y_ineq, sigma=1.0):
        del x, y_eq, y_ineq
        return sigma * hess

    problem = FunctionProblem(
        2,
        objective,
        gradient=gradient,
        bounds=(
            -namespace.ones((2,), dtype=float_dtype(namespace)),
            namespace.ones((2,), dtype=float_dtype(namespace)),
        ),
        lagrangian_hessian=lagrangian_hessian,
    )
    return problem, array(namespace, [0.3, 0.31])


def test_augmented_route_escapes_saddle_like_condensed(namespace):
    # Same regression case as test_dense_nonconvex.py's Cholesky-guard test:
    # the augmented route's inertia check must reject the saddle just as
    # reliably, via a different mechanism (real inertia vs. Cholesky failure).
    problem, x0 = _saddle_trap_qp(namespace)
    result = solve(
        problem,
        x0,
        options=Options(
            linsolve="dense",
            hessian="exact",
            max_iter=300,
            dense=DenseOptions(kkt_route="augmented"),
        ),
    )

    assert result.status is Status.OPTIMAL
    assert abs(float(result.objective) - (-1.0)) <= 1e-5
    assert abs(float(result.x[0] * result.x[1]) - (-1.0)) <= 1e-5


@pytest.mark.gpu
def test_augmented_route_gpu_smoke_cupy():
    """Optional GPU smoke: the augmented route's cuSOLVER adapter end-to-end.

    HS35 has a genuine inequality border (bounds + one linear inequality), so
    this exercises the whole pipeline — DenseSolver -> get_dense_symmetric_
    indefinite_adapter -> CuPyLDLFactorization -- not just the raw adapter
    (see tests/unit/test_dense_cupy_ldl.py).
    """
    cupy = pytest.importorskip("cupy")
    try:
        available = cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        available = False
    if not available:
        pytest.skip("CUDA is not available")
    namespace = import_namespace("cupy")

    problem = HS35(namespace)
    x0 = array(namespace, [0.5, 0.5, 0.5])
    result = solve(
        problem,
        x0,
        options=Options(linsolve="dense", dense=DenseOptions(kkt_route="augmented")),
    )

    assert result.status is Status.OPTIMAL
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert "augmented" in result.linear_solver
