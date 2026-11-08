"""The optional-method contract of ``Problem`` and its ready-made subclasses.

API review D7: overriding an optional method declares that the problem
provides it; :meth:`Problem.provides` overrides that inference explicitly,
for wrappers that implement every method but only delegate some. The
Lagrangian Hessian is always called as ``lagrangian_hessian(x, y_eq, y_ineq,
sigma)``, positionally, with ``sigma != 1`` under the default scaling.
"""

from __future__ import annotations

import warnings

import pytest

import ipax
from ipax.backend.operators import LinearOperator, as_operator
from ipax.options import Options
from ipax.problem.base import Problem
from ipax.problem.derivatives import resolve
from tests._helpers import array, assert_allclose, to_float


class _Quadratic(Problem):
    """``sum((x - 1)^2)`` in two variables."""

    def __init__(self, xp):
        self._xp = xp

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        return self._xp.sum((x - 1.0) * (x - 1.0))


def _quiet(**kwargs):
    return Options(enable_autodiff=False, **kwargs)


def test_overriding_a_method_provides_it(namespace):
    class WithGradient(_Quadratic):
        def gradient(self, x):
            return 2.0 * (x - 1.0)

    resolved = resolve(WithGradient(namespace), namespace, _quiet())
    assert resolved.sources.gradient == "analytic"


def test_provides_declares_absence_of_an_overridden_method(namespace):
    class Wrapper(_Quadratic):
        # A delegating wrapper overrides every method but has no gradient.
        def provides(self):
            return frozenset()

        def gradient(self, x):
            raise NotImplementedError

    with pytest.warns(ipax.FiniteDifferenceWarning):
        result = ipax.solve(
            Wrapper(namespace), array(namespace, [0.0, 0.0]), options=_quiet()
        )
    assert result.derivative_sources.gradient == "finite-diff"
    assert result.status is ipax.Status.OPTIMAL


def test_provides_rejects_unknown_method_names(namespace):
    class Typo(_Quadratic):
        def provides(self):
            return frozenset({"jacobian"})

    with pytest.raises(ipax.ProblemDefinitionError, match="jacobian"):
        resolve(Typo(namespace), namespace, _quiet())


def test_function_problem_subclass_override_is_used(namespace):
    calls = []

    class Sub(ipax.FunctionProblem):
        def gradient(self, x):
            calls.append(1)
            return 2.0 * (x - 1.0)

    xp = namespace
    problem = Sub(2, lambda x: xp.sum((x - 1.0) * (x - 1.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ipax.FiniteDifferenceWarning)
        result = ipax.solve(problem, array(xp, [0.0, 0.0]), options=_quiet())
    assert result.derivative_sources.gradient == "analytic"
    assert calls


def test_function_problem_reports_what_it_was_given(namespace):
    xp = namespace
    problem = ipax.FunctionProblem(
        2, lambda x: xp.sum(x * x), gradient=lambda x: 2.0 * x
    )
    assert problem.provides() == frozenset({"gradient"})


def test_private_provided_attribute_still_works_with_a_deprecation_warning(namespace):
    class Legacy(_Quadratic):
        _provided = frozenset({"objective"})

        def gradient(self, x):
            raise NotImplementedError

    with pytest.warns(DeprecationWarning, match="provides"):
        resolved = resolve(Legacy(namespace), namespace, Options())
    assert resolved.sources.gradient != "analytic"


def _large_gradient_start(xp):
    # ‖∇f(x0)‖∞ ≈ 2e4 > max_gradient = 100, so gradient-based scaling is active
    # and the Hessian is requested with sigma = s_f != 1.
    return array(xp, [1.0, 1.0])


def test_quadratic_problem_with_an_operator_hessian_under_scaling(namespace):
    xp = namespace
    Q = as_operator(array(xp, [[1e4, 0.0], [0.0, 2e4]]))
    c = array(xp, [1.0, -1.0])
    result = ipax.solve(ipax.QuadraticProblem(Q, c), _large_gradient_start(xp))
    assert result.status is ipax.Status.OPTIMAL
    assert result.routes.scaling == "gradient-based"
    assert_allclose(xp, result.x, array(xp, [-1e-4, 5e-5]), rtol=1e-6, atol=1e-10)


def test_quadratic_problem_scales_the_operator_hessian(namespace):
    xp = namespace
    Q = as_operator(array(xp, [[2.0, 1.0], [1.0, 3.0]]))
    problem = ipax.QuadraticProblem(Q, array(xp, [0.0, 0.0]))
    empty = array(xp, [])
    W = problem.lagrangian_hessian(array(xp, [0.0, 0.0]), empty, empty, 0.5)
    v = array(xp, [1.0, -2.0])
    assert_allclose(xp, W.matvec(v), 0.5 * Q.matvec(v))
    assert_allclose(xp, W.dense_matrix(), 0.5 * Q.dense_matrix())


def test_linear_problem_hessian_is_a_zero_operator_not_a_dense_matrix(namespace):
    xp = namespace
    n = 5
    problem = ipax.LinearProblem(
        array(xp, [1.0] * n), bounds=(array(xp, [0.0] * n), None)
    )
    empty = array(xp, [])
    W = problem.lagrangian_hessian(array(xp, [0.5] * n), empty, empty, 1.0)
    assert isinstance(W, LinearOperator)
    assert W.shape == (n, n)
    assert to_float(xp.sum(xp.abs(W.matvec(array(xp, [1.0] * n))))) == 0.0


def test_lagrangian_hessian_receives_sigma_positionally_under_scaling(namespace):
    seen = []

    class Renamed(_Quadratic):
        def objective(self, x):
            return 1e4 * self._xp.sum((x - 1.0) * (x - 1.0))

        def gradient(self, x):
            return 2e4 * (x - 1.0)

        # Parameter names differ from the base class: only arity is the contract.
        def lagrangian_hessian(self, x, mult_eq, mult_ineq, obj_factor):
            seen.append(float(obj_factor))
            xp = self._xp
            return obj_factor * 2e4 * xp.eye(2, dtype=x.dtype)

    result = ipax.solve(Renamed(namespace), array(namespace, [-1.0, 3.0]))
    assert result.status is ipax.Status.OPTIMAL
    assert result.routes.scaling == "gradient-based"
    assert seen and all(s != 1.0 for s in seen)


def test_constraints_left_out_of_provides_are_absent(namespace):
    # A wrapper overrides every method; provides() says it has no equalities
    # or inequalities, so the solve must treat it as unconstrained (it used to
    # call the overridden values and crash deriving their Jacobians).
    class Wrapper(_Quadratic):
        def provides(self):
            return frozenset({"gradient"})

        def gradient(self, x):
            return 2.0 * (x - 1.0)

        # The wrapped object would return rows here; unlisted, they are ignored.
        def eq_constraints(self, x):
            return x[0:1] - 5.0

        def ineq_constraints(self, x):
            return x[1:2] + 5.0

    result = ipax.solve(Wrapper(namespace), array(namespace, [0.0, 0.0]))
    assert result.status is ipax.Status.OPTIMAL
    assert_allclose(namespace, result.x, array(namespace, [1.0, 1.0]), atol=1e-6)
    assert result.y_eq is None and result.y_ineq is None
    assert result.derivative_sources.eq_jacobian == "n/a"


def test_scaled_operator_scales_every_explicit_form(namespace):
    from ipax.backend.operators import Diagonal, ScaledOperator

    xp = namespace
    base = Diagonal(array(xp, [2.0, -3.0]))
    op = ScaledOperator(base, -0.5)
    V = array(xp, [[1.0, 0.0], [0.0, 1.0]])
    assert_allclose(xp, op.rmatmat(V), -0.5 * base.rmatmat(V))
    assert_allclose(xp, op.diagonal(), array(xp, [-1.0, 1.5]))
    rows, cols, values, shape = op.to_coo()
    base_rows, base_cols, base_values, base_shape = base.to_coo()
    assert shape == base_shape
    assert_allclose(xp, values, -0.5 * base_values)
    assert bool(xp.all(rows == base_rows)) and bool(xp.all(cols == base_cols))
    assert_allclose(xp, op.coo_values(), -0.5 * base.coo_values())
    assert op.coo_pattern_signature() == base.coo_pattern_signature()
    assert op.symmetry_hint() == base.symmetry_hint()


def test_scaled_operator_keeps_positive_definiteness_only_for_positive_alpha():
    from ipax.backend.operators import LinearOperator, ScaledOperator

    class PD(LinearOperator):
        shape = (1, 1)

        def matvec(self, v):
            return v

        def positive_definite_hint(self):
            return True

    assert ScaledOperator(PD(), 2.0).positive_definite_hint()
    assert not ScaledOperator(PD(), -2.0).positive_definite_hint()
