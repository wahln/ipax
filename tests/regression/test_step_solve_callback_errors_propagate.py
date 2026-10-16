"""Regression: a programming error inside a step solve propagates out of ``solve``.

The dense KKT route used to translate *any* exception raised while
materializing/factoring the condensed system into :class:`LinearSolveError`.
The driver answers that error with the δ_w regularization ladder and, once the
ladder is exhausted, hands the iterate to feasibility restoration (Wächter &
Biegler 2006, §3.1 → §3.3). So a plain bug — observed: a benchmark operator's
``gram()`` raising ``TypeError: asarray() got an unexpected keyword argument
'device'`` — was retried 40 times per iteration for several iterations and only
surfaced once restoration happened to call the same Gram.

Only numerical failures (``ipax.linalg.solver.numerical_failure_types()``) may
reach the ladder; everything else must leave ``ipax.solve`` unchanged, on the
first call, without any restoration.
"""

from __future__ import annotations

import pytest

from ipax import Options, solve
from ipax.backend.operators import Dense, LinearOperator
from ipax.ipm.driver import IPMDriver
from ipax.problem.base import Problem
from ipax.testing.problems import HS35
from tests._helpers import array

_BUG = "asarray() got an unexpected keyword argument 'device'"


class _BuggyGram(LinearOperator):
    """A healthy explicit operator whose ``gram`` has a programming error."""

    def __init__(self, A, *, fail_on_keywords_only: bool = False, hint=None) -> None:
        self._inner = Dense(A)
        self._fail_on_keywords_only = fail_on_keywords_only
        self._hint = hint
        self.gram_calls = 0

    @property
    def shape(self):
        return self._inner.shape

    def matvec(self, v):
        return self._inner.matvec(v)

    def rmatvec(self, v):
        return self._inner.rmatvec(v)

    def matmat(self, V):
        return self._inner.matmat(V)

    def rmatmat(self, V):
        return self._inner.rmatmat(V)

    def gram_accumulate_dtype_hint(self):
        return self._hint

    def gram(self, weights, *, accumulate_dtype=None, hinted_only=False):
        self.gram_calls += 1
        if self._fail_on_keywords_only and accumulate_dtype is None:
            return self._inner.gram(weights)
        raise TypeError(_BUG)


class _BuggyDenseMatrix(Dense):
    """An explicit Hessian operator whose dense materialization is buggy."""

    def dense_matrix(self, like=None):
        raise TypeError(_BUG)

    def matmat(self, V):
        raise TypeError(_BUG)


class _LinearIneqQP(Problem):
    """HS35 with its inequality declared as ``linear_ineq`` (an operator)."""

    def __init__(self, xp, operator) -> None:
        self._hs35 = HS35(xp)
        self._operator = operator
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 3

    def bounds(self):
        return self._hs35.bounds()

    def objective(self, x):
        return self._hs35.objective(x)

    def gradient(self, x):
        return self._hs35.gradient(x)

    def linear_ineq(self):
        xp = self.xp
        return self._operator, array(xp, [-float("inf")]), array(xp, [3.0])

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        return self._hs35.lagrangian_hessian(x, y_eq, y_ineq, sigma)


class _OperatorJacobianHS35(HS35):
    """HS35 whose (nonlinear-path) inequality Jacobian is a buggy operator."""

    def __init__(self, xp, operator) -> None:
        super().__init__(xp)
        self._operator = operator

    def ineq_jacobian(self, x):
        del x
        return self._operator


class _BuggyHessianHS35(HS35):
    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del x, y_eq, y_ineq
        return _BuggyDenseMatrix(sigma * self._q)


class _RaisingHessianHS35(HS35):
    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        raise TypeError(_BUG)


def _count_restorations(monkeypatch) -> dict[str, int]:
    calls = {"n": 0}
    original = IPMDriver._handle_restoration

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(IPMDriver, "_handle_restoration", counting)
    return calls


def _jacobian(xp):
    return array(xp, [[1.0, 1.0, 2.0]])


_HESSIANS = ["exact", "lbfgs"]


@pytest.mark.parametrize("hessian", _HESSIANS)
def test_linear_ineq_gram_type_error_propagates(namespace, monkeypatch, hessian):
    xp = namespace
    operator = _BuggyGram(_jacobian(xp))
    restorations = _count_restorations(monkeypatch)

    with pytest.raises(TypeError, match="unexpected keyword argument 'device'"):
        solve(
            _LinearIneqQP(xp, operator),
            array(xp, [0.5, 0.5, 0.5]),
            options=Options(hessian=hessian, linsolve="dense", max_iter=50),
        )

    # Raised by the very first Gram, not after a δ_w ladder of retries.
    assert operator.gram_calls == 1
    assert restorations["n"] == 0


@pytest.mark.parametrize("hessian", _HESSIANS)
def test_ineq_jacobian_gram_type_error_propagates(namespace, monkeypatch, hessian):
    xp = namespace
    operator = _BuggyGram(_jacobian(xp))
    restorations = _count_restorations(monkeypatch)

    with pytest.raises(TypeError, match="unexpected keyword argument 'device'"):
        solve(
            _OperatorJacobianHS35(xp, operator),
            array(xp, [0.5, 0.5, 0.5]),
            options=Options(hessian=hessian, linsolve="dense", max_iter=50),
        )

    assert operator.gram_calls == 1
    assert restorations["n"] == 0


def test_mixed_precision_gram_keyword_bug_is_not_retried_away(namespace, monkeypatch):
    # The pre-0.10 compatibility retry (an operator whose ``gram`` predates the
    # ``accumulate_dtype``/``hinted_only`` keywords) must fire only for that
    # signature mismatch. A Gram that *accepts* the keywords and then crashes
    # is a bug: retrying without them silently degraded the solve to the exact
    # Gram and hid it.
    xp = namespace
    operator = _BuggyGram(_jacobian(xp), fail_on_keywords_only=True, hint="float32")
    restorations = _count_restorations(monkeypatch)

    with pytest.raises(TypeError, match="unexpected keyword argument 'device'"):
        solve(
            _LinearIneqQP(xp, operator),
            array(xp, [0.5, 0.5, 0.5]),
            options=Options(hessian="exact", linsolve="dense", max_iter=50),
        )

    assert operator.gram_calls == 1
    assert restorations["n"] == 0


def test_hessian_operator_materialization_type_error_propagates(namespace, monkeypatch):
    xp = namespace
    restorations = _count_restorations(monkeypatch)

    with pytest.raises(TypeError, match="unexpected keyword argument 'device'"):
        solve(
            _BuggyHessianHS35(xp),
            array(xp, [0.5, 0.5, 0.5]),
            options=Options(hessian="exact", linsolve="dense", max_iter=50),
        )

    assert restorations["n"] == 0


def test_lagrangian_hessian_callback_type_error_propagates(namespace, monkeypatch):
    xp = namespace
    restorations = _count_restorations(monkeypatch)

    with pytest.raises(TypeError, match="unexpected keyword argument 'device'"):
        solve(
            _RaisingHessianHS35(xp),
            array(xp, [0.5, 0.5, 0.5]),
            options=Options(hessian="exact", linsolve="dense", max_iter=50),
        )

    assert restorations["n"] == 0
