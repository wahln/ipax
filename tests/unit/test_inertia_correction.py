"""Unit tests for the driver's inertia-guided regularization decision.

``IPMDriver._inertia_acceptable`` is the solver-agnostic seam: it only engages
when the injected solver reports the factor's inertia *and* the KKT operator
knows its target, and otherwise defers to factorization-failure escalation.
"""

from __future__ import annotations

from ipax.ipm.driver import IPMDriver


class _Op:
    def __init__(self, target):
        self._target = target

    def expected_inertia(self):
        return self._target


class _Solver:
    def __init__(self, actual):
        self._actual = actual

    def inertia_or_none(self):
        return self._actual


def _driver_with(solver):
    driver = IPMDriver.__new__(IPMDriver)  # bypass __init__; only _solver is used
    driver._solver = solver
    return driver


def test_matching_inertia_is_accepted():
    driver = _driver_with(_Solver((2, 1, 0)))
    assert driver._inertia_acceptable(_Op((2, 1, 0))) is True


def test_mismatched_inertia_is_rejected():
    driver = _driver_with(_Solver((1, 2, 0)))
    assert driver._inertia_acceptable(_Op((2, 1, 0))) is False


def test_operator_without_target_is_accepted():
    # L-BFGS / low-rank Hessian: no target ⇒ defer to failure-based escalation.
    driver = _driver_with(_Solver((1, 2, 0)))
    assert driver._inertia_acceptable(_Op(None)) is True


def test_solver_without_inertia_is_accepted():
    # Dense / Krylov, or a non-inertia-revealing factorization.
    driver = _driver_with(object())
    assert driver._inertia_acceptable(_Op((2, 1, 0))) is True


def test_solver_reporting_none_is_accepted():
    driver = _driver_with(_Solver(None))
    assert driver._inertia_acceptable(_Op((2, 1, 0))) is True
