"""Unit tests for the feasibility restoration phase."""

from __future__ import annotations

from ipax.backend.operators import as_operator
from ipax.ipm.restoration import RestorationExit, restore
from ipax.problem.base import Problem
from ipax.testing.problems import HS6, InfeasibleEqualities
from tests._helpers import array, assert_allclose


def _no_ineq(x):
    raise AssertionError("inequality callbacks should not be used here")


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
