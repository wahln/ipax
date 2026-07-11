"""A silent inertia no-op must be surfaced, once.

The IPM's inertia-guided δ_w correction engages only when BOTH the KKT
operator knows its target inertia AND the backend factorization reports the
factor's inertia. When the operator has a target (an assemblable, possibly
nonconvex Hessian) but the backend cannot report inertia (e.g. the SuperLU
fallback), the safety net silently does not run — a symmetric-indefinite
LDLᵀ can then succeed with the *wrong* inertia and hand back a non-descent
step (Wächter & Biegler 2006, §3.1). ``SparseDirectSolver`` must warn once
in that situation. The dense/Krylov routes have their own PD mechanisms
(Cholesky probe / CG breakdown), so no warning belongs there.
"""

from __future__ import annotations

import logging

import pytest

from ipax._logging import LOGGER_NAME
from ipax.backend.operators import Dense, Diagonal
from ipax.ipm.hessian import LBFGSOperator
from ipax.ipm.kkt import build_condensed_operator
from ipax.linalg.regularize import RegularizationState
from ipax.linalg.sparse import SparseDirectSolver
from ipax.options import LBFGSOptions
from tests._helpers import array


class _FakeInner:
    """A backend solver whose factorization reveals no inertia (SuperLU-like)."""

    def __init__(self, inertia=None):
        self._inertia = inertia

    def factor(self, operator):
        del operator

    def solve(self, rhs):
        return rhs

    def inertia_or_none(self):
        return self._inertia


class _FakeAdapter:
    def __init__(self, inertia=None):
        self._inertia = inertia

    def from_coo(self, rows, cols, values, *, shape, symmetric, pattern_signature):
        del rows, cols, values, symmetric, pattern_signature
        return ("fake-matrix", shape)

    def solver(self, *, require_inertia=False):
        del require_inertia
        return _FakeInner(self._inertia)


@pytest.fixture
def fake_sparse_adapter(monkeypatch):
    """Route the facade onto a no-inertia fake backend; returns a setter."""
    import ipax.backend.sparse as sparse_backend

    holder = {"adapter": _FakeAdapter(inertia=None)}
    monkeypatch.setattr(
        sparse_backend, "get_sparse_adapter", lambda xp: holder["adapter"]
    )
    return holder


def _condensed_with_target(namespace):
    """Condensed operator with an assemblable Hessian → expected_inertia known."""
    op = build_condensed_operator(
        Dense(array(namespace, [[4.0, 0.5], [0.5, 3.0]])),
        Diagonal(array(namespace, [0.25, 0.75])),
        Diagonal(array(namespace, [2.0, 0.5])),
        Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]])),
        RegularizationState(delta_w=1e-6),
    )
    assert op.expected_inertia() is not None
    return op


def _condensed_lbfgs(namespace):
    """L-BFGS condensed operator → no inertia target (PD by damping)."""
    W = LBFGSOperator(2, LBFGSOptions(memory=5))
    W.update(array(namespace, [1.0, 0.5]), array(namespace, [2.0, 1.0]))
    op = build_condensed_operator(
        W,
        Diagonal(array(namespace, [0.25, 0.75])),
        Diagonal(array(namespace, [2.0, 0.5])),
        Dense(array(namespace, [[1.0, 2.0], [-1.0, 0.5]])),
        RegularizationState(delta_w=1e-6),
    )
    assert op.expected_inertia() is None
    return op


def _warnings(caplog):
    return [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "inertia" in r.getMessage()
    ]


def test_warns_once_when_target_known_but_inertia_unavailable(
    namespace, fake_sparse_adapter, caplog
):
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    op = _condensed_with_target(namespace)
    solver = SparseDirectSolver()

    solver.factor(op)
    solver.factor(op)  # second iteration: must not warn again

    assert len(_warnings(caplog)) == 1


def test_no_warning_when_backend_reports_inertia(
    namespace, fake_sparse_adapter, caplog
):
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    fake_sparse_adapter["adapter"] = _FakeAdapter(inertia=(2, 2, 0))
    solver = SparseDirectSolver()

    solver.factor(_condensed_with_target(namespace))

    assert _warnings(caplog) == []


def test_no_warning_without_an_inertia_target(namespace, fake_sparse_adapter, caplog):
    # L-BFGS keeps the (1,1) block PD by Powell damping — no target, no gap,
    # no warning.
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    solver = SparseDirectSolver()

    solver.factor(_condensed_lbfgs(namespace))

    assert _warnings(caplog) == []
