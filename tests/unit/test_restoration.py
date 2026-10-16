"""Unit tests for the feasibility restoration phase."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipax import Options, solve
from ipax.backend.namespace import array_namespace
from ipax.backend.operators import Dense, MatrixFreeJacobian, as_operator
from ipax.ipm.restoration import RestorationExit, _dense_gauss_newton, restore
from ipax.linalg.krylov import KrylovSolver
from ipax.linalg.solver import LinearSolveError
from ipax.options import RestorationOptions
from ipax.problem.base import Problem
from ipax.testing.problems import HS6, InfeasibleEqualities
from tests._helpers import array, assert_allclose


def _no_ineq(x):
    raise AssertionError("inequality callbacks should not be used here")


def _krylov_solver(**overrides) -> KrylovSolver:
    """The shipped restoration Krylov preset, optionally tightened or truncated."""
    return KrylovSolver(replace(RestorationOptions().krylov, **overrides))


def _restore_kwargs(xp, x, **over):
    n = int(x.shape[0])
    kwargs = {
        "xp": xp,
        "x": x,
        "s": xp.zeros((0,), dtype=x.dtype),
        "m": 0,
        "m_eq": 0,
        "eq_fn": _no_ineq,
        "eq_jac_fn": _no_ineq,
        "ineq_fn": _no_ineq,
        "ineq_jac_fn": _no_ineq,
        "mask_l": xp.zeros((n,), dtype=xp.bool),
        "mask_u": xp.zeros((n,), dtype=xp.bool),
        "lower_safe": xp.zeros((n,), dtype=x.dtype),
        "upper_safe": xp.zeros((n,), dtype=x.dtype),
        "tol": 1e-8,
    }
    kwargs.update(over)
    return kwargs


def _two_equalities(xp):
    """``x0 + 2 x1 = 1``, ``x2 = -1``: a rank-2 system on three variables."""

    def eq_fn(z):
        return xp.stack((z[0] + 2.0 * z[1] - 1.0, z[2] + 1.0))

    def eq_jac_fn(z):
        del z
        return as_operator(array(xp, [[1.0, 2.0, 0.0], [0.0, 0.0, 1.0]]))

    return eq_fn, eq_jac_fn


def test_restoration_reduces_equality_violation(namespace):
    problem = HS6(namespace)
    # A point off the constraint manifold: 10(x2 - x1^2) = 10(0 - 4) = -40.
    x = array(namespace, [2.0, 0.0])
    s = namespace.zeros((0,), dtype=x.dtype)
    theta0 = float(namespace.max(namespace.abs(problem.eq_constraints(x))))

    x_new, _, exit_reason = restore(
        xp=namespace,
        x=x,
        s=s,
        m=0,
        m_eq=1,
        eq_fn=problem.eq_constraints,
        eq_jac_fn=lambda z: as_operator(problem.eq_jacobian(z)),
        ineq_fn=_no_ineq,
        ineq_jac_fn=_no_ineq,
        mask_l=namespace.zeros((2,), dtype=namespace.bool),
        mask_u=namespace.zeros((2,), dtype=namespace.bool),
        lower_safe=namespace.zeros((2,), dtype=x.dtype),
        upper_safe=namespace.zeros((2,), dtype=x.dtype),
        tol=1e-8,
    )
    theta_new = float(namespace.max(namespace.abs(problem.eq_constraints(x_new))))
    assert exit_reason is RestorationExit.FEASIBLE
    assert theta_new < theta0
    assert theta_new <= 1e-6


def test_matrix_free_restoration_uses_only_jacobian_products(namespace, monkeypatch):
    """The opt-in route must never construct an n-by-n identity or probe matmat."""
    xp = namespace
    n = 3
    x = array(xp, [2.0, 0.0, -1.0])
    eye_shapes: list[int] = []
    original_eye = xp.eye

    def recording_eye(size, *args, **kwargs):
        eye_shapes.append(int(size))
        return original_eye(size, *args, **kwargs)

    monkeypatch.setattr(xp, "eye", recording_eye)
    eq_fn, _ = _two_equalities(xp)

    def eq_jac_fn(z):
        del z

        def matvec(v):
            return xp.stack((v[0] + 2.0 * v[1], v[2]))

        def rmatvec(v):
            return xp.stack((v[0], 2.0 * v[0], v[1]))

        op = MatrixFreeJacobian((2, n), matvec, rmatvec)

        def forbidden_matmat(V):
            del V
            raise AssertionError("matrix-free restoration densified the Jacobian")

        op.matmat = forbidden_matmat  # type: ignore[method-assign]
        return op

    x_new, _, exit_reason = restore(
        **_restore_kwargs(xp, x, m_eq=2, eq_fn=eq_fn, eq_jac_fn=eq_jac_fn),
        linear_solver=_krylov_solver(rtol=1e-10),
    )

    assert exit_reason is RestorationExit.FEASIBLE
    assert float(xp.max(xp.abs(eq_fn(x_new)))) <= 1e-8
    assert n not in eye_shapes


def test_matrix_free_restoration_uses_the_truncated_krylov_iterate(namespace):
    """A work-capped inner solve is a trial direction, not a rejected rung.

    CG iterates minimize the Gauss-Newton model over a growing Krylov space,
    so the truncated iterate is a descent direction (Steihaug 1983). Discarding
    it and climbing the LM ladder would degrade the route to damped steepest
    descent at large λ — and with a one-iteration cap could never certify
    anything but a solve failure.
    """
    xp = namespace
    x = array(xp, [2.0, 0.0, -1.0])
    eq_fn, eq_jac_fn = _two_equalities(xp)

    x_new, _, exit_reason = restore(
        **_restore_kwargs(xp, x, m_eq=2, eq_fn=eq_fn, eq_jac_fn=eq_jac_fn),
        linear_solver=_krylov_solver(max_iter=1, rtol=1e-12),
    )

    assert exit_reason is RestorationExit.FEASIBLE
    assert float(xp.max(xp.abs(eq_fn(x_new)))) <= 1e-8


def test_matrix_free_restoration_tracks_dense_on_an_ill_conditioned_jacobian(
    namespace, monkeypatch
):
    """Where restoration is actually entered — rank-deficient, ill-conditioned
    Jacobians — the Krylov route must make progress comparable to the dense
    Gauss-Newton solve on the same outer budget, not crawl along the LM ladder."""
    import numpy as np

    import ipax.ipm.restoration as restoration_mod

    monkeypatch.setattr(restoration_mod, "_MAX_ITER", 20)
    xp = namespace
    rng = np.random.default_rng(0)
    # Sized so that discarding truncated CG iterates (climbing the LM ladder
    # instead) measurably lags dense: 3.8x here vs 1.4x with the reuse.
    n, m, rank = 200, 150, 130
    u, _ = np.linalg.qr(rng.standard_normal((m, m)))
    v, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sv = np.zeros(m)
    sv[:rank] = np.logspace(0, -5, rank)
    a = (u * sv) @ v[:m, :]
    b = a @ rng.standard_normal(n) + 1e-3 * rng.standard_normal(m)
    a_xp = array(xp, a.tolist())
    b_xp = array(xp, b.tolist())
    x0 = array(xp, rng.standard_normal(n).tolist())

    def eq_fn(z):
        return xp.matmul(a_xp, z) - b_xp

    def eq_jac_fn(z):
        del z
        return as_operator(a_xp)

    kwargs = _restore_kwargs(xp, x0, m_eq=m, eq_fn=eq_fn, eq_jac_fn=eq_jac_fn)
    x_dense, _, _ = restore(**kwargs)
    x_krylov, _, krylov_exit = restore(**kwargs, linear_solver=_krylov_solver())

    theta_dense = float(xp.max(xp.abs(eq_fn(x_dense))))
    theta_krylov = float(xp.max(xp.abs(eq_fn(x_krylov))))
    assert krylov_exit is not RestorationExit.LINEAR_SOLVE_FAILED
    assert theta_krylov <= 2.0 * theta_dense


def test_matrix_free_restoration_does_not_fall_back_on_krylov_failure(namespace):
    """Failed inner solves must climb the LM ladder, never allocate a dense matrix."""
    xp = namespace
    calls = {"factor": 0, "solve": 0}

    class FailingSolver:
        def factor(self, operator):
            calls["factor"] += 1
            assert operator.shape == (1, 1)

        def solve(self, rhs):
            del rhs
            calls["solve"] += 1
            raise LinearSolveError("synthetic Krylov failure")

    def eq_jac_fn(z):
        del z
        return MatrixFreeJacobian((1, 1), lambda v: v, lambda v: v)

    x = array(xp, [0.0])
    _, _, exit_reason = restore(
        **_restore_kwargs(xp, x, m_eq=1, eq_fn=lambda z: z - 1.0, eq_jac_fn=eq_jac_fn),
        linear_solver=FailingSolver(),  # type: ignore[arg-type]
    )

    assert exit_reason is RestorationExit.LINEAR_SOLVE_FAILED
    assert not exit_reason.certifies_infeasibility
    assert calls["factor"] == calls["solve"]
    assert calls["solve"] > 1


def test_matrix_free_and_dense_restoration_match_with_active_bounds_and_inequality(
    namespace, monkeypatch
):
    """The product operator must reproduce the dense reduced normal equations."""
    import ipax.ipm.restoration as restoration_mod

    monkeypatch.setattr(restoration_mod, "_MAX_ITER", 1)
    xp = namespace
    x = xp.zeros((3,), dtype=xp.float64)

    def eq_fn(z):
        return xp.stack((z[0] + z[1] + 1.0,))

    def eq_jac_fn(z):
        del z
        return as_operator(array(xp, [[1.0, 1.0, 0.0]]))

    def ineq_fn(z):
        return xp.stack((z[2] + 0.5,))

    def ineq_jac_fn(z):
        del z
        return as_operator(array(xp, [[0.0, 0.0, 1.0]]))

    kwargs = _restore_kwargs(
        xp,
        x,
        s=xp.ones((1,), dtype=x.dtype),
        m=1,
        m_eq=1,
        eq_fn=eq_fn,
        eq_jac_fn=eq_jac_fn,
        ineq_fn=ineq_fn,
        ineq_jac_fn=ineq_jac_fn,
        mask_l=xp.asarray([True, False, False], dtype=xp.bool),
    )
    x_dense, s_dense, dense_exit = restore(**kwargs)
    x_krylov, s_krylov, krylov_exit = restore(
        **kwargs, linear_solver=_krylov_solver(rtol=1e-10)
    )

    assert dense_exit is krylov_exit
    assert_allclose(xp, x_krylov, x_dense, rtol=1e-8, atol=1e-8)
    assert_allclose(xp, s_krylov, s_dense, rtol=1e-8, atol=1e-8)


def test_matrix_free_restoration_requires_an_adjoint(namespace):
    xp = namespace
    x = array(xp, [0.0])

    with pytest.raises(NotImplementedError, match="adjoint"):
        restore(
            **_restore_kwargs(
                xp,
                x,
                m_eq=1,
                eq_fn=lambda z: z - 1.0,
                eq_jac_fn=lambda z: MatrixFreeJacobian((1, 1), lambda v: v),
            ),
            linear_solver=_krylov_solver(),
        )


def test_solve_rejects_matrix_free_restoration_without_an_adjoint_up_front(
    namespace,
):
    """A matvec-only Jacobian must be refused at entry, not hundreds of
    iterations later when restoration first needs the adjoint."""
    xp = namespace

    class MatvecOnly(Problem):
        @property
        def n_vars(self) -> int:
            return 2

        def objective(self, x):
            return xp.sum(x * x)

        def gradient(self, x):
            return 2.0 * x

        def eq_constraints(self, x):
            return xp.stack((x[0] + x[1] - 1.0,))

        def eq_jacobian(self, x):
            del x
            return MatrixFreeJacobian((1, 2), lambda v: xp.stack((v[0] + v[1],)))

    options = Options(
        hessian="lbfgs", restoration=RestorationOptions(linear_solver="krylov")
    )
    with pytest.raises(ValueError, match="adjoint"):
        solve(MatvecOnly(), array(xp, [0.0, 0.0]), options=options)


def test_restoration_flags_inconsistent_equalities(namespace):
    problem = InfeasibleEqualities(namespace)
    x = array(namespace, [0.5])
    s = namespace.zeros((0,), dtype=x.dtype)

    _, _, exit_reason = restore(
        xp=namespace,
        x=x,
        s=s,
        m=0,
        m_eq=2,
        eq_fn=problem.eq_constraints,
        eq_jac_fn=lambda z: as_operator(problem.eq_jacobian(z)),
        ineq_fn=_no_ineq,
        ineq_jac_fn=_no_ineq,
        mask_l=namespace.zeros((1,), dtype=namespace.bool),
        mask_u=namespace.zeros((1,), dtype=namespace.bool),
        lower_safe=namespace.zeros((1,), dtype=x.dtype),
        upper_safe=namespace.zeros((1,), dtype=x.dtype),
        tol=1e-8,
    )
    # A genuinely infeasible problem must end with a *certificate* — a
    # stationary point of the infeasibility — not a mere stall.
    assert exit_reason.certifies_infeasibility


def test_restoration_keeps_projected_bound_point_strictly_interior(namespace):
    class BoundaryEquality(Problem):
        @property
        def n_vars(self) -> int:
            return 1

        def objective(self, x):
            return namespace.sum(x * x)

        def eq_constraints(self, x):
            return namespace.stack((x[0],))

        def eq_jacobian(self, x):
            del x
            return array(namespace, [[1.0]])

    problem = BoundaryEquality()
    x = array(namespace, [0.5])
    s = namespace.zeros((0,), dtype=x.dtype)

    x_new, _, exit_reason = restore(
        xp=namespace,
        x=x,
        s=s,
        m=0,
        m_eq=1,
        eq_fn=problem.eq_constraints,
        eq_jac_fn=lambda z: as_operator(problem.eq_jacobian(z)),
        ineq_fn=_no_ineq,
        ineq_jac_fn=_no_ineq,
        mask_l=namespace.asarray([True], dtype=namespace.bool),
        mask_u=namespace.asarray([False], dtype=namespace.bool),
        lower_safe=array(namespace, [0.0]),
        upper_safe=array(namespace, [0.0]),
        tol=1e-8,
    )

    assert exit_reason is RestorationExit.FEASIBLE
    assert bool(x_new[0] > 0.0)


def test_restoration_survives_singular_gauss_newton_system(namespace):
    # Regression: an extreme-scale constraint Jacobian makes the Gauss-Newton
    # normal matrix J^T J numerically singular (cf. HS7, whose (1+x1^2)^2
    # Jacobian reaches ~1e201 at a bad restoration iterate). The damped solve
    # must degrade gracefully — previously numpy's solve raised
    # ``LinAlgError: Singular matrix`` and crashed the whole solve.
    xp = namespace
    x = array(xp, [1.0, 1.0])
    s = xp.zeros((0,), dtype=x.dtype)
    # Moderate residual but a hugely ill-scaled Jacobian at this iterate.
    jac = array(xp, [[4.0e100, -1.9e27]])

    def eq_fn(z):
        del z
        return array(xp, [1.0])

    def eq_jac_fn(z):
        del z
        return as_operator(jac)

    x_new, _, exit_reason = restore(
        xp=xp,
        x=x,
        s=s,
        m=0,
        m_eq=1,
        eq_fn=eq_fn,
        eq_jac_fn=eq_jac_fn,
        ineq_fn=_no_ineq,
        ineq_jac_fn=_no_ineq,
        mask_l=xp.zeros((2,), dtype=xp.bool),
        mask_u=xp.zeros((2,), dtype=xp.bool),
        lower_safe=xp.zeros((2,), dtype=x.dtype),
        upper_safe=xp.zeros((2,), dtype=x.dtype),
        tol=1e-8,
    )

    # It cannot reduce the (decoupled) violation, but it returns a finite point
    # and reports a no-descent stationarity certificate instead of raising.
    assert exit_reason.certifies_infeasibility
    assert bool(xp.all(xp.isfinite(x_new)))


@pytest.mark.parametrize(
    "bug",
    [
        TypeError("asarray() got an unexpected keyword argument 'device'"),
        MemoryError("host allocation failed"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_dense_gauss_newton_solve_propagates_non_numerical_errors(
    namespace, monkeypatch, bug
):
    # The dense LM solve treats only a numerical failure
    # (``numerical_failure_types()``) as a rejected trial; a bug or a resource
    # failure must leave ``restore`` as itself instead of being turned into a
    # NO_DESCENT "infeasibility certificate".
    xp = namespace
    x = array(xp, [1.0, 1.0])

    def raising_solve(*args, **kwargs):
        del args, kwargs
        raise bug

    monkeypatch.setattr(xp.linalg, "solve", raising_solve)

    with pytest.raises(type(bug)) as info:
        restore(
            xp=xp,
            x=x,
            s=xp.zeros((0,), dtype=x.dtype),
            m=0,
            m_eq=1,
            eq_fn=lambda z: array(xp, [1.0]),
            eq_jac_fn=lambda z: as_operator(array(xp, [[1.0, 2.0]])),
            ineq_fn=_no_ineq,
            ineq_jac_fn=_no_ineq,
            mask_l=xp.zeros((2,), dtype=xp.bool),
            mask_u=xp.zeros((2,), dtype=xp.bool),
            lower_safe=xp.zeros((2,), dtype=x.dtype),
            upper_safe=xp.zeros((2,), dtype=x.dtype),
            tol=1e-8,
        )
    assert info.value is bug


def test_rejected_lm_trials_reuse_the_jacobian(namespace):
    # Levenberg-Marquardt damping control: a rejected trial step must retry with
    # a larger damping WITHOUT rebuilding the Jacobian/normal matrix — they
    # belong to the unchanged current iterate. arctan(x)=0 from x=2 makes the
    # undamped Gauss-Newton step overshoot (|dx| = |atan(x)|(1+x^2) > |x|), so
    # the first several trials are rejected while the damping grows.
    xp = namespace
    calls = {"jac": 0}

    def eq_fn(z):
        return xp.atan(z)

    def eq_jac_fn(z):
        calls["jac"] += 1
        return as_operator(xp.reshape(1.0 / (1.0 + z * z), (1, 1)))

    x = array(xp, [2.0])
    s = xp.zeros((0,), dtype=x.dtype)
    x_new, _, exit_reason = restore(
        xp=xp,
        x=x,
        s=s,
        m=0,
        m_eq=1,
        eq_fn=eq_fn,
        eq_jac_fn=eq_jac_fn,
        ineq_fn=_no_ineq,
        ineq_jac_fn=_no_ineq,
        mask_l=xp.zeros((1,), dtype=xp.bool),
        mask_u=xp.zeros((1,), dtype=xp.bool),
        lower_safe=xp.zeros((1,), dtype=x.dtype),
        upper_safe=xp.zeros((1,), dtype=x.dtype),
        tol=1e-8,
    )

    assert exit_reason is RestorationExit.FEASIBLE
    assert float(xp.max(xp.abs(xp.atan(x_new)))) <= 1e-8
    # One Jacobian build per accepted iterate (plus the final check), never one
    # per rejected trial: the pre-fix loop rebuilt it ~13 times here.
    assert calls["jac"] <= 9


def test_bound_blocked_infeasibility_is_box_stationary(namespace):
    # c(x) = x - 1 = 0 with the upper bound x <= 0: the descent direction points
    # out of the box, so the projected gradient is zero — a first-order
    # stationary point of the bound-constrained infeasibility. The verdict must
    # come from that test directly instead of grinding the LM damping to its
    # ceiling with one Jacobian rebuild per rejected (projection-swallowed)
    # trial (the MANNE stall anatomy, S2MPJ 2026-07 audit).
    xp = namespace
    calls = {"jac": 0}

    def eq_fn(z):
        return z - 1.0

    def eq_jac_fn(z):
        calls["jac"] += 1
        return as_operator(xp.ones((1, 1), dtype=z.dtype))

    x = array(xp, [-0.5])
    s = xp.zeros((0,), dtype=x.dtype)
    x_new, _, exit_reason = restore(
        xp=xp,
        x=x,
        s=s,
        m=0,
        m_eq=1,
        eq_fn=eq_fn,
        eq_jac_fn=eq_jac_fn,
        ineq_fn=_no_ineq,
        ineq_jac_fn=_no_ineq,
        mask_l=xp.zeros((1,), dtype=xp.bool),
        mask_u=xp.asarray([True], dtype=xp.bool),
        lower_safe=xp.zeros((1,), dtype=x.dtype),
        upper_safe=xp.zeros((1,), dtype=x.dtype),
        tol=1e-8,
    )

    assert exit_reason is RestorationExit.STATIONARY
    assert bool(xp.all(xp.isfinite(x_new)))
    # x rides to the bound in one accepted step; the next iterate detects the
    # blocked gradient. The pre-fix loop burned ~26 Jacobian builds here.
    assert calls["jac"] <= 4


def test_restoration_recovers_inequality_slack_without_filter_residual(namespace):
    class InactiveInequality(Problem):
        @property
        def n_vars(self) -> int:
            return 1

        def objective(self, x):
            return namespace.sum(x * x)

        def ineq_constraints(self, x):
            del x
            return array(namespace, [-1e-4])

        def ineq_jacobian(self, x):
            del x
            return array(namespace, [[0.0]])

    problem = InactiveInequality()
    x = array(namespace, [0.0])
    s = array(namespace, [1.0])

    _, s_new, exit_reason = restore(
        xp=namespace,
        x=x,
        s=s,
        m=1,
        m_eq=0,
        eq_fn=lambda z: namespace.zeros((0,), dtype=z.dtype),
        eq_jac_fn=lambda z: as_operator(namespace.zeros((0, 1), dtype=z.dtype)),
        ineq_fn=problem.ineq_constraints,
        ineq_jac_fn=lambda z: as_operator(problem.ineq_jacobian(z)),
        mask_l=namespace.zeros((1,), dtype=namespace.bool),
        mask_u=namespace.zeros((1,), dtype=namespace.bool),
        lower_safe=namespace.zeros((1,), dtype=x.dtype),
        upper_safe=namespace.zeros((1,), dtype=x.dtype),
        tol=1e-8,
    )

    assert exit_reason is RestorationExit.FEASIBLE
    assert_allclose(
        namespace, problem.ineq_constraints(x) + s_new, array(namespace, [0.0])
    )


def test_blocked_bound_solves_the_reduced_system_on_the_free_set(namespace):
    # DRUGDIS/UBH5 anatomy (S2MPJ 2026-07 probe): with a variable pinned at its
    # bound by a dominant residual, the FULL-space Gauss-Newton step is ruled
    # by the blocked component; projection swallows it, the trial fails to
    # descend, and the LM loop degrades into a microscopic gradient crawl
    # (DRUGDIS: theta 0.19 -> 0.16 in 200 iterations). Solving the normal
    # system reduced to the FREE variables (projected Newton on the binding
    # set; Bertsekas 1999, §2.3) must instead place the free variable at its
    # optimum in a couple of iterations.
    #
    # Residuals: r1 = x1 + x2 - 1 (feasible part), r2 = M (x1 + 1) with x1 >= 0
    # (pins x1 at the lower bound, gradient pointing out of the box). The
    # box-constrained infeasibility minimum is x = (0, 1).
    xp = namespace
    big = 1.0e3
    calls = {"jac": 0}

    def eq_fn(z):
        return xp.stack((z[0] + z[1] - 1.0, big * (z[0] + 1.0)))

    def eq_jac_fn(z):
        calls["jac"] += 1
        one = 1.0 + xp.zeros_like(z[0])
        return as_operator(
            xp.stack((xp.stack((one, one)), xp.stack((big * one, 0.0 * one))))
        )

    x = array(xp, [0.0, 0.0])
    s = xp.zeros((0,), dtype=x.dtype)
    x_new, _, exit_reason = restore(
        xp=xp,
        x=x,
        s=s,
        m=0,
        m_eq=2,
        eq_fn=eq_fn,
        eq_jac_fn=eq_jac_fn,
        ineq_fn=_no_ineq,
        ineq_jac_fn=_no_ineq,
        mask_l=xp.asarray([True, False], dtype=xp.bool),
        mask_u=xp.zeros((2,), dtype=xp.bool),
        lower_safe=array(xp, [0.0, 0.0]),
        upper_safe=array(xp, [0.0, 0.0]),
        tol=1e-8,
    )

    # The reduced solve reaches the box-stationary point (0, 1) and certifies
    # it in a handful of Jacobian builds; the full-space crawl needed dozens
    # and returned x2 far from 1. (The certificate may be NO_DESCENT rather
    # than STATIONARY: f ≈ ½M² dominates, so free-residual improvements below
    # ~1e-5 sit under f's float64 resolution — numerically stationary.)
    assert exit_reason.certifies_infeasibility
    assert float(xp.abs(x_new[1] - 1.0)) <= 1e-3
    assert calls["jac"] <= 6


def test_budget_exit_is_not_an_infeasibility_certificate(namespace, monkeypatch):
    # An exhausted iteration budget says nothing about local infeasibility —
    # HS6 restoration converges given iterations, so a 1-iteration budget must
    # exit as BUDGET (uncertified), never as a stationarity certificate.
    import ipax.ipm.restoration as restoration_mod

    monkeypatch.setattr(restoration_mod, "_MAX_ITER", 1)
    problem = HS6(namespace)
    x = array(namespace, [2.0, 0.0])
    s = namespace.zeros((0,), dtype=x.dtype)

    _, _, exit_reason = restore(
        xp=namespace,
        x=x,
        s=s,
        m=0,
        m_eq=1,
        eq_fn=problem.eq_constraints,
        eq_jac_fn=lambda z: as_operator(problem.eq_jacobian(z)),
        ineq_fn=_no_ineq,
        ineq_jac_fn=_no_ineq,
        mask_l=namespace.zeros((2,), dtype=namespace.bool),
        mask_u=namespace.zeros((2,), dtype=namespace.bool),
        lower_safe=namespace.zeros((2,), dtype=x.dtype),
        upper_safe=namespace.zeros((2,), dtype=x.dtype),
        tol=1e-8,
    )

    assert exit_reason is RestorationExit.BUDGET
    assert not exit_reason.certifies_infeasibility


def test_stall_window_exit_is_not_an_infeasibility_certificate(namespace, monkeypatch):
    # A plateau exit (accepted steps whose improvement is microscopic) is a
    # stall, not a stationarity certificate: here the gradient stays at
    # |2x - 1| ~ 9 throughout, but a pinned huge LM damping makes every
    # accepted step (and hence every f improvement) tiny, so the trailing
    # window fires. That exit must be reported as an uncertified stall.
    import ipax.ipm.restoration as restoration_mod

    monkeypatch.setattr(restoration_mod, "_LM_INIT", 1e14)
    problem = InfeasibleEqualities(namespace)
    x = array(namespace, [5.0])
    s = namespace.zeros((0,), dtype=x.dtype)

    _, _, exit_reason = restore(
        xp=namespace,
        x=x,
        s=s,
        m=0,
        m_eq=2,
        eq_fn=problem.eq_constraints,
        eq_jac_fn=lambda z: as_operator(problem.eq_jacobian(z)),
        ineq_fn=_no_ineq,
        ineq_jac_fn=_no_ineq,
        mask_l=namespace.zeros((1,), dtype=namespace.bool),
        mask_u=namespace.zeros((1,), dtype=namespace.bool),
        lower_safe=namespace.zeros((1,), dtype=x.dtype),
        upper_safe=namespace.zeros((1,), dtype=x.dtype),
        tol=1e-8,
    )

    assert exit_reason is RestorationExit.STALL_WINDOW
    assert not exit_reason.certifies_infeasibility


def _symmetric_trap(xp):
    """``x0 + x1 = 3``, ``x0² + x1² = 5`` from the symmetric point (1.5, 1.5).

    Solutions are (1, 2) and (2, 1). On the diagonal ``x0 = x1`` the Jacobian
    has rank 1 and every Gauss-Newton direction is symmetric, so an exact
    solver keeps the iterates on the diagonal, where ½‖c‖² has a strict local
    minimum with ``c ≠ 0`` — a saddle of the full infeasibility that the
    first-order certificate would report as local infeasibility.
    """

    def eq_fn(z):
        return xp.stack((z[0] + z[1] - 3.0, z[0] * z[0] + z[1] * z[1] - 5.0))

    def eq_jac_fn(z):
        one = xp.ones_like(z[0])
        return as_operator(
            xp.stack((xp.stack((one, one)), xp.stack((2.0 * z[0], 2.0 * z[1]))))
        )

    return eq_fn, eq_jac_fn


@pytest.mark.parametrize("route", ["dense", "krylov"])
def test_restoration_escapes_a_symmetric_saddle_before_certifying(namespace, route):
    """A critical point on a symmetry-invariant subspace is a saddle, not a
    certificate: restoration must probe off the subspace before claiming
    local infeasibility (S2MPJ POWERSUMNE/CYCLOOCT/HADAMARD, 2026-09)."""
    xp = namespace
    x = array(xp, [1.5, 1.5])
    eq_fn, eq_jac_fn = _symmetric_trap(xp)
    solver = None if route == "dense" else _krylov_solver()

    x_new, _, exit_reason = restore(
        **_restore_kwargs(xp, x, m_eq=2, eq_fn=eq_fn, eq_jac_fn=eq_jac_fn),
        linear_solver=solver,
    )

    assert exit_reason is RestorationExit.FEASIBLE
    assert float(xp.max(xp.abs(eq_fn(x_new)))) <= 1e-8
    assert not exit_reason.certifies_infeasibility


def test_restoration_still_certifies_a_genuine_local_minimum(namespace):
    """The escape probe must not turn a true nonzero local minimizer of the
    infeasibility into a stall: ``x² = -1`` has none, and stays certified."""
    xp = namespace
    problem = InfeasibleEqualities(xp)
    x = array(xp, [0.5])
    _, _, exit_reason = restore(
        **_restore_kwargs(
            xp,
            x,
            m_eq=1,
            eq_fn=problem.eq_constraints,
            eq_jac_fn=lambda z: as_operator(problem.eq_jacobian(z)),
        ),
        linear_solver=_krylov_solver(),
    )
    assert exit_reason.certifies_infeasibility


# --------------------------------------------------------------------------- #
# Dense reference route: large Gram-capable blocks are not densified
# --------------------------------------------------------------------------- #
class _SpyDense(Dense):
    """A ``Dense`` Jacobian recording the products the restoration route uses."""

    def __init__(self, A, calls):
        super().__init__(A)
        self.calls = calls

    def matmat(self, V):
        self.calls.append("matmat")
        return super().matmat(V)

    def rmatvec(self, v):
        self.calls.append("rmatvec")
        return super().rmatvec(v)


class _GramSpyDense(_SpyDense):
    """``_SpyDense`` that also exposes the weighted Gram ``Aᵀ diag(w) A``."""

    def gram(self, weights, *, accumulate_dtype=None, hinted_only=False):
        del accumulate_dtype, hinted_only
        self.calls.append("gram")
        A = self.dense_matrix()
        xp = array_namespace(A, weights)
        weighted = A * xp.expand_dims(weights, axis=1)
        return xp.matmul(xp.permute_dims(A, (1, 0)), weighted)


def _tall_blocks(xp):
    """A 7x3 inequality block and a 2x3 equality block with mixed signs."""
    jg = array(
        xp,
        [
            [1.0, -2.0, 0.5],
            [0.0, 3.0, -1.0],
            [2.5, 0.0, 1.0],
            [-1.0, 1.0, 1.0],
            [0.3, -0.7, 2.0],
            [4.0, 1.0, -3.0],
            [0.0, 0.0, 1.5],
        ],
    )
    g = array(xp, [0.4, -0.2, 1.3, 0.0, 2.2, -1.0, 0.7])
    jc = array(xp, [[1.0, 1.0, 0.0], [0.5, -2.0, 3.0]])
    c = array(xp, [0.25, -1.5])
    return jg, g, jc, c


def _old_dense_gauss_newton(xp, n, jc, c, jg, active, gpos):
    """The pre-change dense assembly, verbatim, as the byte-identity oracle."""
    dtype = jg.dtype
    hessian = xp.zeros((n, n), dtype=dtype)
    grad = xp.zeros((n,), dtype=dtype)
    jc = xp.matmul(jc, xp.eye(n, dtype=dtype))
    jc_t = xp.permute_dims(jc, (1, 0))
    hessian = hessian + xp.matmul(jc_t, jc)
    grad = grad + xp.matmul(jc_t, c)
    jg = xp.matmul(jg, xp.eye(n, dtype=dtype))
    jg_w = jg * xp.expand_dims(active, axis=1)
    jg_t = xp.permute_dims(jg, (1, 0))
    hessian = hessian + xp.matmul(jg_t, jg_w)
    grad = grad + xp.matmul(jg_t, gpos)
    return hessian, grad


def _assemble(xp, eq_op, c, ineq_op, g, max_elements):
    dtype = g.dtype
    active = xp.astype(g > 0.0, dtype)
    gpos = xp.maximum(g, xp.zeros_like(g))
    hessian, grad = _dense_gauss_newton(
        xp,
        n=3,
        dtype=dtype,
        eq_jac=eq_op,
        c=c,
        ineq_jac=ineq_op,
        active=active,
        gpos=gpos,
        max_elements=max_elements,
    )
    return hessian, grad, active, gpos


def test_dense_route_forms_a_large_gram_capable_block_through_the_operator(
    namespace,
):
    xp = namespace
    jg, g, jc, c = _tall_blocks(xp)
    eq_calls: list[str] = []
    ineq_calls: list[str] = []
    eq_op = _GramSpyDense(jc, eq_calls)
    ineq_op = _GramSpyDense(jg, ineq_calls)

    # Threshold below both m*n (7*3 = 21, 2*3 = 6): both blocks switch.
    hessian, grad, active, gpos = _assemble(xp, eq_op, c, ineq_op, g, 5)

    assert "matmat" not in ineq_calls
    assert "matmat" not in eq_calls
    assert sorted(ineq_calls) == ["gram", "rmatvec"]
    assert sorted(eq_calls) == ["gram", "rmatvec"]
    ref_h, ref_g = _old_dense_gauss_newton(xp, 3, jc, c, jg, active, gpos)
    assert_allclose(xp, hessian, ref_h, rtol=1e-12, atol=1e-12)
    assert_allclose(xp, grad, ref_g, rtol=1e-12, atol=1e-12)


def test_dense_route_keeps_small_blocks_byte_identical(namespace):
    xp = namespace
    jg, g, jc, c = _tall_blocks(xp)
    eq_calls: list[str] = []
    ineq_calls: list[str] = []
    eq_op = _GramSpyDense(jc, eq_calls)
    ineq_op = _GramSpyDense(jg, ineq_calls)

    # Threshold at the larger m*n (a block switches only strictly above it).
    hessian, grad, active, gpos = _assemble(xp, eq_op, c, ineq_op, g, 21)

    assert ineq_calls == ["matmat"]
    assert eq_calls == ["matmat"]
    ref_h, ref_g = _old_dense_gauss_newton(xp, 3, jc, c, jg, active, gpos)
    assert bool(xp.all(hessian == ref_h))
    assert bool(xp.all(grad == ref_g))


def test_dense_route_decides_per_block_by_size(namespace):
    # 6 < threshold < 21: only the tall inequality block switches.
    xp = namespace
    jg, g, jc, c = _tall_blocks(xp)
    eq_calls: list[str] = []
    ineq_calls: list[str] = []
    hessian, grad, active, gpos = _assemble(
        xp, _GramSpyDense(jc, eq_calls), c, _GramSpyDense(jg, ineq_calls), g, 10
    )
    assert eq_calls == ["matmat"]
    assert sorted(ineq_calls) == ["gram", "rmatvec"]
    ref_h, ref_g = _old_dense_gauss_newton(xp, 3, jc, c, jg, active, gpos)
    assert_allclose(xp, hessian, ref_h, rtol=1e-12, atol=1e-12)
    assert_allclose(xp, grad, ref_g, rtol=1e-12, atol=1e-12)


def test_dense_route_densifies_a_block_that_is_not_gram_capable(namespace):
    xp = namespace
    jg, g, jc, c = _tall_blocks(xp)
    ineq_calls: list[str] = []
    ineq_op = _SpyDense(jg, ineq_calls)  # plain Dense: no Gram override
    assert not ineq_op.gram_capable()

    def matvec(v):
        return xp.matmul(jc, v)

    def rmatvec(v):
        return xp.matmul(xp.permute_dims(jc, (1, 0)), v)

    eq_op = MatrixFreeJacobian((2, 3), matvec, rmatvec)
    assert not eq_op.gram_capable()
    hessian, grad, active, gpos = _assemble(xp, eq_op, c, ineq_op, g, 1)

    assert ineq_calls == ["matmat"]
    ref_h, ref_g = _old_dense_gauss_newton(xp, 3, jc, c, jg, active, gpos)
    assert_allclose(xp, hessian, ref_h, rtol=1e-14, atol=1e-14)
    assert_allclose(xp, grad, ref_g, rtol=1e-14, atol=1e-14)


def test_restore_never_densifies_a_large_gram_capable_block(namespace):
    # End to end: the same restoration on the operator path and on the
    # densified path reaches the same point (round-off aside).
    xp = namespace
    jg, _, _, _ = _tall_blocks(xp)
    rhs = array(xp, [-1.0, 2.0, -0.5, 0.0, -2.0, 1.0, -0.25])
    x0 = array(xp, [1.0, -1.0, 2.0])

    def ineq_fn(z):
        return xp.matmul(jg, z) + rhs

    def run(max_elements):
        calls: list[str] = []

        def ineq_jac_fn(z):
            del z
            return _GramSpyDense(jg, calls)

        out = restore(
            **_restore_kwargs(
                xp,
                x0,
                s=xp.ones((7,), dtype=x0.dtype),
                m=7,
                ineq_fn=ineq_fn,
                ineq_jac_fn=ineq_jac_fn,
            ),
            dense_materialize_max_elements=max_elements,
        )
        return out, calls

    (x_op, s_op, exit_op), op_calls = run(20)
    (x_ref, s_ref, exit_ref), ref_calls = run(21)

    assert "gram" in op_calls
    assert "matmat" not in op_calls
    assert "matmat" in ref_calls
    assert "gram" not in ref_calls
    assert exit_op is exit_ref
    assert exit_op is RestorationExit.FEASIBLE
    assert_allclose(xp, x_op, x_ref, rtol=1e-10, atol=1e-10)
    assert_allclose(xp, s_op, s_ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("wrap", ["coo", "row_scaled", "signed_selection"])
def test_dense_route_real_sparse_gram_operators_match_the_densified_reference(
    namespace, wrap
):
    # The production Gram-capable operators, not a test double: the sparse
    # adapter's Gram, the gradient-scaling wrapper (folds d² into the 0-1
    # weights) and the lowered two-sided linear_ineq selection (folds |S|ᵀ).
    from ipax.backend.operators import COOOperator, _SignedRowSelection
    from ipax.problem.scaling import _RowScaled

    xp = namespace
    jg, _, _, _ = _tall_blocks(xp)
    rows, cols, values, shape = Dense(jg).to_coo()
    op = COOOperator(rows, cols, values, shape)
    if wrap == "row_scaled":
        op = _RowScaled(op, array(xp, [0.5, 2.0, 1.0, 3.0, 0.25, 1.5, 4.0]))
    elif wrap == "signed_selection":
        sel_lower = xp.asarray([True, False, True, False, True, False, True])
        sel_upper = xp.asarray([True, True, False, False, True, True, False])
        op = _SignedRowSelection(op, sel_lower, sel_upper)
    if not op.gram_capable():
        pytest.skip("no sparse Gram adapter for this backend")
    m = op.shape[0]
    g = array(xp, [0.4, -0.2, 1.3, 0.0, 2.2, -1.0, 0.7, 0.9][:m])
    c = xp.zeros((0,), dtype=g.dtype)

    hessian, grad, _, _ = _assemble(xp, None, c, op, g, 1)
    ref_h, ref_g, _, _ = _assemble(xp, None, c, op, g, 10**9)

    assert_allclose(xp, hessian, ref_h, rtol=1e-12, atol=1e-12)
    assert_allclose(xp, grad, ref_g, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "big"])
def test_dense_materialize_max_elements_is_validated(bad):
    with pytest.raises(ValueError, match="dense_materialize_max_elements"):
        RestorationOptions(dense_materialize_max_elements=bad)


def test_dense_materialize_max_elements_default():
    assert RestorationOptions().dense_materialize_max_elements == 10_000_000
    options = RestorationOptions(dense_materialize_max_elements=1)
    assert options.dense_materialize_max_elements == 1
