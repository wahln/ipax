"""The public exception and warning hierarchy (``ipax.errors``).

Every ipax exception derives from :class:`ipax.IpaxError` *and* from the
built-in it replaced, so ``except RuntimeError`` / ``except ValueError``
clauses written against 0.13 keep catching them.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import ipax
from ipax.options import Options
from ipax.problem.base import Problem
from tests._helpers import array


class _ObjectiveOnly(Problem):
    """A problem with no derivatives at all."""

    def __init__(self, xp):
        self._xp = xp

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        return self._xp.sum((x - 1.0) * (x - 1.0))


class _WithEqualities(_ObjectiveOnly):
    """Objective plus a nonlinear equality without a Jacobian."""

    def eq_constraints(self, x):
        return self._xp.stack((x[0] * x[1] - 1.0,))


@pytest.mark.parametrize(
    ("name", "bases"),
    [
        ("IpaxError", (Exception,)),
        ("ConfigurationError", (ipax.IpaxError, ValueError)),
        ("DerivativeUnavailableError", (ipax.IpaxError, RuntimeError)),
        ("ProblemDefinitionError", (ipax.IpaxError, ValueError)),
        ("LinearSolveError", (ipax.IpaxError, RuntimeError)),
        ("IpaxWarning", (UserWarning,)),
        ("FiniteDifferenceWarning", (ipax.IpaxWarning, UserWarning)),
    ],
)
def test_hierarchy_is_exported_and_keeps_the_builtin_bases(name, bases):
    cls = getattr(ipax, name)
    assert name in ipax.__all__
    for base in bases:
        assert issubclass(cls, base)


def test_linear_solve_error_keeps_its_iterate_and_subclasses():
    from ipax.linalg.krylov import KrylovConvergenceError
    from ipax.linalg.solver import LinearSolveError

    assert ipax.LinearSolveError is LinearSolveError
    assert issubclass(KrylovConvergenceError, ipax.IpaxError)
    err = LinearSolveError("stalled", iterate="last")
    assert err.iterate == "last"


def test_no_gradient_source_raises_derivative_unavailable(namespace):
    options = Options(enable_autodiff=False, enable_finite_diff=False)
    with pytest.raises(ipax.DerivativeUnavailableError, match="no gradient source"):
        ipax.solve(
            _ObjectiveOnly(namespace), array(namespace, [0.0, 0.0]), options=options
        )


def test_exact_hessian_without_one_raises_derivative_unavailable(namespace):
    options = Options(hessian="exact")
    with (
        warnings.catch_warnings(),
        pytest.raises(ipax.DerivativeUnavailableError, match="hessian='exact'"),
    ):
        warnings.simplefilter("ignore", ipax.FiniteDifferenceWarning)
        ipax.solve(
            _ObjectiveOnly(namespace), array(namespace, [0.0, 0.0]), options=options
        )


def test_derivative_unavailable_is_still_a_runtime_error(namespace):
    # 0.13 code catches RuntimeError here; it must keep working.
    options = Options(enable_autodiff=False, enable_finite_diff=False)
    with pytest.raises(RuntimeError):
        ipax.solve(
            _ObjectiveOnly(namespace), array(namespace, [0.0, 0.0]), options=options
        )


@pytest.mark.parametrize("problem_cls", [_ObjectiveOnly, _WithEqualities])
def test_finite_difference_warnings_have_their_category_and_point_at_the_caller(
    namespace, problem_cls
):
    options = Options(enable_autodiff=False, max_iter=3)
    with pytest.warns(ipax.FiniteDifferenceWarning) as record:
        ipax.solve(
            problem_cls(namespace), array(namespace, [0.5, 0.5]), options=options
        )
    fd = [w for w in record if issubclass(w.category, ipax.FiniteDifferenceWarning)]
    expected = 2 if problem_cls is _WithEqualities else 1
    assert len(fd) == expected
    for w in fd:
        # samefile: the drive letter's case can differ on Windows.
        assert Path(w.filename).samefile(__file__), w.filename


def test_finite_difference_warnings_can_be_filtered_by_category(namespace):
    options = Options(enable_autodiff=False, max_iter=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", ipax.FiniteDifferenceWarning)
        ipax.solve(
            _ObjectiveOnly(namespace), array(namespace, [0.5, 0.5]), options=options
        )


def test_no_jacobian_source_raises_derivative_unavailable(namespace):
    options = Options(enable_autodiff=False, enable_finite_diff=False)

    class Constrained(_WithEqualities):
        def gradient(self, x):
            return 2.0 * (x - 1.0)

    with pytest.raises(ipax.DerivativeUnavailableError, match="eq_constraints"):
        ipax.solve(
            Constrained(namespace), array(namespace, [0.5, 0.5]), options=options
        )


def test_autodiff_hvp_without_an_autodiff_backend_raises(namespace):
    from ipax.problem.autodiff import get_autodiff_adapter

    if get_autodiff_adapter(namespace) is not None:
        pytest.skip("this backend has autodiff")
    options = Options(hessian="autodiff-hvp")
    with pytest.raises(ipax.DerivativeUnavailableError, match="autodiff-hvp"):
        ipax.solve(
            _ObjectiveOnly(namespace), array(namespace, [0.5, 0.5]), options=options
        )
