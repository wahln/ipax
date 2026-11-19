"""Regression: warnings name the line that called ipax, not ipax's internals.

Through 0.13 the finite-difference warnings used a fixed ``stacklevel`` and
named ``ipax/solve.py``, so per-module ``warnings`` filters and the
once-per-location default did not apply to the user's code. The SciPy front
end's fixed ``stacklevel`` values were right only for a direct
``minimize_ipax`` call: through ``scipy.optimize.minimize(...,
method=minimize_ipax)`` they named SciPy's ``_minimize.py``. Warnings now
walk past ipax's own frames, ``dataclasses.replace`` and ``scipy.optimize``
(``ipax.errors.warn_external``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ipax
from ipax.options import Options
from ipax.problem.base import Problem
from tests._helpers import array


def _named_here(record, category):
    hits = [w for w in record if issubclass(w.category, category)]
    assert hits
    for w in hits:
        # samefile: the drive letter's case can differ on Windows.
        assert Path(w.filename).samefile(__file__), w.filename
    return hits


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


@pytest.mark.parametrize("problem_cls", [_ObjectiveOnly, _WithEqualities])
def test_finite_difference_warnings_name_the_solve_call(namespace, problem_cls):
    options = Options(enable_autodiff=False, max_iter=3)
    with pytest.warns(ipax.FiniteDifferenceWarning) as record:
        ipax.solve(
            problem_cls(namespace), array(namespace, [0.5, 0.5]), options=options
        )
    hits = _named_here(record, ipax.FiniteDifferenceWarning)
    assert len(hits) == (2 if problem_cls is _WithEqualities else 1)


# --- SciPy front end --------------------------------------------------------

TARGET = np.array([1.0, -2.0, 0.5, 3.0])


def _qp(x):
    return float(np.sum((x - TARGET) ** 2))


def _qp_grad(x):
    return 2.0 * (x - TARGET)


def _qp_hess(x):
    del x
    return 2.0 * np.eye(4)


def _circle_without_jac():
    return {"type": "ineq", "fun": lambda x: np.array([1.0 - x[0] ** 2 - x[1] ** 2])}


def _circle_with_jac():
    return {
        "type": "ineq",
        "fun": lambda x: np.array([1.0 - x[0] ** 2 - x[1] ** 2]),
        "jac": lambda x: np.array([[-2.0 * x[0], -2.0 * x[1], 0.0, 0.0]]),
    }


def _adapter_cases():
    optimize = pytest.importorskip("scipy.optimize")
    return {
        "keep_feasible": {
            "constraints": optimize.LinearConstraint(
                np.ones((1, 4)), -np.inf, 1.0, keep_feasible=True
            )
        },
        "hess-strategy": {"hess": optimize.BFGS()},
        "some-jac-missing": {
            "constraints": [_circle_with_jac(), _circle_without_jac()]
        },
        "constraint-hess-missing": {
            "hess": _qp_hess,
            "constraints": [_circle_with_jac()],
        },
        "unknown-option": {"options": {"bogus": 1}},
    }


CASES = [
    "keep_feasible",
    "hess-strategy",
    "some-jac-missing",
    "constraint-hess-missing",
    "unknown-option",
]


@pytest.mark.parametrize("entry", ["direct", "scipy-minimize"])
@pytest.mark.parametrize("case", CASES)
def test_scipy_front_end_warnings_name_the_callers_line(case, entry):
    optimize = pytest.importorskip("scipy.optimize")
    from ipax.interfaces.scipy import minimize_ipax

    kwargs = {"jac": _qp_grad, **_adapter_cases()[case]}
    with pytest.warns(optimize.OptimizeWarning) as record:
        if entry == "direct":
            minimize_ipax(_qp, np.zeros(4), **kwargs)
        else:
            optimize.minimize(_qp, np.zeros(4), method=minimize_ipax, **kwargs)
    _named_here(record, optimize.OptimizeWarning)


def test_core_warnings_through_scipy_minimize_name_the_callers_line():
    optimize = pytest.importorskip("scipy.optimize")
    from ipax.interfaces.scipy import minimize_ipax

    # No jac anywhere: the gradient falls back to ipax's own finite differences.
    with pytest.warns(ipax.FiniteDifferenceWarning) as record:
        optimize.minimize(
            _qp, np.zeros(4), method=minimize_ipax, options={"maxiter": 3}
        )
    _named_here(record, ipax.FiniteDifferenceWarning)
