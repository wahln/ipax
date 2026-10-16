"""Regression: genuine numerical step-solve failures still drive the δ_w ladder.

Companion to ``test_step_solve_callback_errors_propagate.py``: narrowing the
dense route's exception translation to numerical failures must leave the
numerical path exactly as it was — the backend's own ``LinAlgError`` (NumPy
``numpy.linalg.LinAlgError``, Torch ``torch.linalg.LinAlgError``) or a Python
arithmetic error raised while the condensed system is materialized/factored
is still a :class:`LinearSolveError` to the driver, which escalates δ_w
(Wächter & Biegler 2006, §3.1) and, when the ladder is exhausted, hands the
step to feasibility restoration (§3.3).
"""

from __future__ import annotations

import pytest

import ipax.ipm.driver as driver_module
from ipax import Options, Status, solve
from ipax.ipm.driver import _MAX_REG_ATTEMPTS, IPMDriver
from ipax.ipm.kkt import _CondensedOperator
from ipax.testing.problems import HS35
from tests._helpers import array, backend_linalg_error

_FAILURES = {
    "backend-linalg-error": backend_linalg_error,
    "floating-point-error": lambda xp: FloatingPointError("overflow encountered"),
    "zero-division-error": lambda xp: ZeroDivisionError("float division by zero"),
}


def _fail_materializations(monkeypatch, failure: BaseException, count: int):
    """Make the first ``count`` condensed-matrix materializations raise ``failure``."""
    calls = {"n": 0}
    original = _CondensedOperator.dense_matrix

    def flaky(self, like=None):
        calls["n"] += 1
        if calls["n"] <= count:
            raise failure
        return original(self, like)

    monkeypatch.setattr(_CondensedOperator, "dense_matrix", flaky)
    return calls


def _count(monkeypatch, owner, name) -> dict[str, int]:
    calls = {"n": 0}
    original = getattr(owner, name)

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, counting)
    return calls


def _solve_hs35(xp):
    return solve(
        HS35(xp),
        array(xp, [0.5, 0.5, 0.5]),
        options=Options(hessian="exact", linsolve="dense", max_iter=100),
    )


@pytest.mark.parametrize("kind", sorted(_FAILURES))
def test_numerical_failure_in_first_factorization_escalates_delta_w(
    namespace, monkeypatch, kind
):
    xp = namespace
    baseline_escalations = _count(monkeypatch, driver_module, "escalate_delta_w")
    baseline = _solve_hs35(xp)
    assert baseline.status is Status.OPTIMAL
    clean = baseline_escalations["n"]

    failure = _FAILURES[kind](xp)
    materializations = _fail_materializations(monkeypatch, failure, count=1)
    escalations = _count(monkeypatch, driver_module, "escalate_delta_w")
    restorations = _count(monkeypatch, IPMDriver, "_handle_restoration")

    result = _solve_hs35(xp)

    assert materializations["n"] > 1  # the ladder retried the factorization
    assert escalations["n"] == clean + 1  # exactly one extra δ_w rung
    assert restorations["n"] == 0
    assert result.status is Status.OPTIMAL
    assert abs(float(result.objective) - float(baseline.objective)) <= 1e-9


@pytest.mark.parametrize("kind", sorted(_FAILURES))
def test_exhausted_ladder_hands_the_step_to_restoration(namespace, monkeypatch, kind):
    xp = namespace
    failure = _FAILURES[kind](xp)
    _fail_materializations(monkeypatch, failure, count=_MAX_REG_ATTEMPTS)
    restorations = _count(monkeypatch, IPMDriver, "_handle_restoration")

    result = _solve_hs35(xp)

    assert restorations["n"] >= 1  # W&B 2006 §3.1 → §3.3 step-failure hand-off
    assert result.status is not Status.NUMERICAL_ERROR
    assert result.status is Status.OPTIMAL
    assert abs(float(result.objective) - 1.0 / 9.0) <= 1e-6
