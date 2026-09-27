"""Regression: a tall Gram-capable inequality block is never densified by restoration.

TROTS 2026-09 (``Protons_01`` on CuPy, ``hessian="exact"``,
``corrections="mehrotra"``, dense route): restoration's dense reference route
materialized the full inequality Jacobian through an identity ``matmat`` to
form ``Jᵀ diag(g > 0) J`` — a 369 445 × 1101 float64 array (3.2 GB) per
restoration iteration, 1127 s inside ``restore()`` thrashing an 8 GB GPU. The
Gauss-Newton matrix only needs the ``n × n`` weighted Gram, which a sparse
(Gram-capable) Jacobian forms without densifying; above
``RestorationOptions.dense_materialize_max_elements`` the dense route now asks
the operator for it (``gram``/``rmatvec``), and below it keeps the verbatim
densified arithmetic the S2MPJ sweep was tuned on.
"""

from __future__ import annotations

import pytest

from ipax import Options, Status, solve
from ipax.backend.namespace import array_namespace
from ipax.backend.operators import Dense
from ipax.ipm.driver import IPMDriver
from ipax.options import RestorationOptions
from ipax.problem.base import Problem
from tests._helpers import array

_M, _N = 24, 3


class _GramSpyJacobian(Dense):
    """A Gram-capable ``Dense`` that logs identity ``matmat`` probes and Grams."""

    def __init__(self, A, log):
        super().__init__(A)
        self._log = log

    def matmat(self, V):
        if tuple(V.shape) == (self.shape[1], self.shape[1]):
            self._log.append("identity_matmat")
        return super().matmat(V)

    def gram(self, weights, *, accumulate_dtype=None, hinted_only=False):
        del accumulate_dtype, hinted_only
        self._log.append("gram")
        A = self.dense_matrix()
        xp = array_namespace(A, weights)
        weighted = A * xp.expand_dims(weights, axis=1)
        return xp.matmul(xp.permute_dims(A, (1, 0)), weighted)


class _TallLinearIneq(Problem):
    """``min ½‖x − t‖²`` s.t. ``A x ≤ b`` with a tall ``A`` (24 × 3)."""

    def __init__(self, xp, log):
        self._xp = xp
        self._log = log
        rows = []
        for i in range(_M):
            # Deterministic, well-spread directions: rows on a twisted helix.
            a = (i % 5) - 2.0
            b = ((3 * i) % 7) - 3.0
            rows.append([1.0 + 0.1 * i, a, b])
        self._A = array(xp, rows)
        self._b = array(xp, [1.0 + 0.05 * i for i in range(_M)])
        self._t = array(xp, [3.0, 2.0, -1.0])

    @property
    def n_vars(self) -> int:
        return _N

    def objective(self, x):
        d = x - self._t
        return 0.5 * self._xp.sum(d * d)

    def gradient(self, x):
        return x - self._t

    def ineq_constraints(self, x):
        return self._xp.matmul(self._A, x) - self._b

    def ineq_jacobian(self, x):
        del x
        return _GramSpyJacobian(self._A, self._log)

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del y_eq, y_ineq
        return sigma * self._xp.eye(_N, dtype=x.dtype)


def _fail_step_solve_once(monkeypatch):
    """Force the first step solve to fail so the driver enters restoration."""
    original = IPMDriver._solve_step
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        dx, dy_eq, dw, ok = original(self, *args, **kwargs)
        if calls["n"] == 1:
            return dx, dy_eq, dw, False
        return dx, dy_eq, dw, ok

    monkeypatch.setattr(IPMDriver, "_solve_step", flaky)
    return calls


def _restoration_log(monkeypatch):
    """Record which operator calls happen *inside* restoration."""
    import ipax.ipm.driver as driver_mod

    state = {"inside": False, "entries": 0}
    inside: list[str] = []
    original = driver_mod.restore

    def wrapped(**kwargs):
        state["inside"] = True
        state["entries"] += 1
        try:
            return original(**kwargs)
        finally:
            state["inside"] = False

    monkeypatch.setattr(driver_mod, "restore", wrapped)

    class _Log(list):
        def append(self, item):
            if state["inside"]:
                inside.append(item)
            super().append(item)

    return _Log(), inside, state


@pytest.mark.parametrize("threshold", [1, 10_000_000], ids=["operator", "dense"])
def test_tall_gram_block_through_restoration_is_not_densified(
    namespace, monkeypatch, threshold
):
    xp = namespace
    _fail_step_solve_once(monkeypatch)
    log, inside, state = _restoration_log(monkeypatch)
    problem = _TallLinearIneq(xp, log)
    x0 = array(xp, [4.0, 4.0, 4.0])  # far outside A x ≤ b
    assert float(xp.max(problem.ineq_constraints(x0))) > 1.0

    result = solve(
        problem,
        x0,
        options=Options(
            hessian="exact",
            linsolve="dense",
            max_iter=100,
            restoration=RestorationOptions(
                linear_solver="dense", dense_materialize_max_elements=threshold
            ),
        ),
    )

    assert state["entries"] >= 1, "the forced step failure never reached restoration"
    assert result.status is Status.OPTIMAL
    if threshold < _M * _N:
        assert "identity_matmat" not in inside
        assert "gram" in inside
    else:
        assert "identity_matmat" in inside
        assert "gram" not in inside


def test_operator_path_restoration_matches_the_densified_path(namespace, monkeypatch):
    xp = namespace
    calls = _fail_step_solve_once(monkeypatch)
    results = []
    for threshold in (1, 10_000_000):
        calls["n"] = 0  # re-arm the forced failure for each run
        problem = _TallLinearIneq(xp, [])
        results.append(
            solve(
                problem,
                array(xp, [4.0, 4.0, 4.0]),
                options=Options(
                    hessian="exact",
                    linsolve="dense",
                    max_iter=100,
                    restoration=RestorationOptions(
                        linear_solver="dense",
                        dense_materialize_max_elements=threshold,
                    ),
                ),
            )
        )
    r_op, r_dense = results
    assert r_op.status is r_dense.status is Status.OPTIMAL
    assert r_op.n_iter == r_dense.n_iter
    assert abs(float(r_op.objective) - float(r_dense.objective)) <= 1e-9 * max(
        1.0, abs(float(r_dense.objective))
    )
    x_op = r_op.x
    x_dense = r_dense.x
    assert float(xp.max(xp.abs(x_op - x_dense))) <= 1e-7
