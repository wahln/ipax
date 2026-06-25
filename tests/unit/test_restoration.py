"""Unit tests for the feasibility restoration phase."""

from __future__ import annotations

from ipax.backend.operators import as_operator
from ipax.ipm.restoration import restore
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

    x_new, _, infeasible = restore(
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
    assert not infeasible
    assert theta_new < theta0
    assert theta_new <= 1e-6


def test_restoration_flags_inconsistent_equalities(namespace):
    problem = InfeasibleEqualities(namespace)
    x = array(namespace, [0.5])
    s = namespace.zeros((0,), dtype=x.dtype)

    _, _, infeasible = restore(
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
    assert infeasible


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

    x_new, _, infeasible = restore(
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

    assert not infeasible
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

    x_new, _, infeasible = restore(
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
    # and reports infeasibility instead of raising.
    assert infeasible
    assert bool(xp.all(xp.isfinite(x_new)))


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

    _, s_new, infeasible = restore(
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

    assert not infeasible
    assert_allclose(
        namespace, problem.ineq_constraints(x) + s_new, array(namespace, [0.0])
    )
