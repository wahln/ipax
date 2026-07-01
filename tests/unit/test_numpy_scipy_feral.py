"""Unit coverage for the optional Feral CPU sparse solver bridge."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

# The adapter module imports scipy at load time, so guard it: a conditional
# import skips the whole module when scipy is absent (and dodges E402, which
# exempts try/except-guarded imports across ruff versions).
try:
    from ipax.backend.sparse.numpy_scipy import (
        FeralSparseSolver,
        SciPySparseAdapter,
        SuperLUSparseSolver,
    )
    from ipax.linalg.solver import LinearSolveError
    from ipax.testing.backends import import_namespace
    from tests._helpers import array, assert_allclose
except ImportError:  # pragma: no cover - scipy not installed
    pytest.skip(
        "scipy is required for the Feral sparse adapter", allow_module_level=True
    )

pytestmark = pytest.mark.sparse


class _FakeInertia:
    def __init__(self, n_pos: int, n_neg: int, n_zero: int = 0) -> None:
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.n_zero = n_zero


def _numpy_namespace() -> Any:
    try:
        return import_namespace("numpy")
    except ImportError as exc:
        pytest.skip(f"NumPy namespace unavailable: {exc}")


def _install_fake_feral(
    monkeypatch: pytest.MonkeyPatch,
    *,
    solution: list[float],
    inertia: tuple[int, int, int] = (2, 0, 0),
) -> list[Any]:
    calls: list[Any] = []

    class FakeSolver:
        needs_refinement = False

        def factor(self, matrix: object) -> tuple[int, _FakeInertia]:
            calls.append(("factor", matrix))
            return 0, _FakeInertia(*inertia)

        def solve(self, rhs: np.ndarray) -> np.ndarray:
            calls.append(("solve", rhs.copy()))
            return np.asarray(solution, dtype=np.float64)

    def from_scipy(matrix: object, *, symmetric: str) -> object:
        calls.append(("from_scipy", symmetric, matrix))
        return object()

    fake_feral = SimpleNamespace(
        FactorStatus=SimpleNamespace(SUCCESS=0),
        FeralError=RuntimeError,
        Solver=FakeSolver,
        from_scipy=from_scipy,
    )
    monkeypatch.setitem(sys.modules, "feral", fake_feral)
    return calls


def test_cpu_sparse_default_prefers_feral_for_symmetric_system(monkeypatch):
    xp = _numpy_namespace()
    calls = _install_fake_feral(
        monkeypatch,
        solution=[7.0, 8.0],
        inertia=(1, 1, 0),
    )
    adapter = SciPySparseAdapter()
    rows = xp.asarray([0, 0, 1, 1])
    cols = xp.asarray([0, 1, 0, 1])
    values = array(xp, [2.0, 1.0, 1.0, -3.0])
    K = adapter.from_coo(rows, cols, values, shape=(2, 2))

    solver = adapter.solver(require_inertia=True)
    solver.factor(K)
    actual = solver.solve(array(xp, [1.0, 2.0]))

    assert calls[0][0] == "from_scipy"
    assert calls[0][1] == "full"
    assert solver.inertia == (1, 1, 0)
    assert_allclose(xp, actual, array(xp, [7.0, 8.0]))


def test_feral_captures_inertia_for_free_without_require_inertia(monkeypatch):
    # Feral returns the LDLᵀ inertia on every symmetric factor, so the best-effort
    # accessor surfaces it even when require_inertia was not requested — this is
    # what powers the IPM's inertia-guided δ_w correction on the default path.
    xp = _numpy_namespace()
    _install_fake_feral(monkeypatch, solution=[1.0, 1.0], inertia=(1, 1, 0))
    adapter = SciPySparseAdapter()
    rows = xp.asarray([0, 0, 1, 1])
    cols = xp.asarray([0, 1, 0, 1])
    values = array(xp, [2.0, 1.0, 1.0, -3.0])
    K = adapter.from_coo(rows, cols, values, shape=(2, 2))

    solver = adapter.solver()  # require_inertia=False
    solver.factor(K)

    assert solver.inertia_or_none() == (1, 1, 0)


def test_superlu_inertia_or_none_is_none_without_provider(monkeypatch):
    # The SuperLU fallback is not inertia-revealing; the best-effort accessor must
    # return None (never densify or raise) so the IPM defers to failure-based δ_w.
    xp = _numpy_namespace()
    monkeypatch.setitem(sys.modules, "feral", None)
    adapter = SciPySparseAdapter()
    rows = xp.asarray([0, 0, 1, 1])
    cols = xp.asarray([0, 1, 0, 1])
    values = array(xp, [3.0, 1.0, 1.0, 2.0])
    K = adapter.from_coo(rows, cols, values, shape=(2, 2))

    solver = adapter.solver()
    solver.factor(K)

    assert solver.inertia_or_none() is None


def test_cpu_sparse_default_falls_back_to_superlu_for_nonsymmetric_system(
    monkeypatch,
):
    xp = _numpy_namespace()
    calls = _install_fake_feral(monkeypatch, solution=[99.0, 99.0])
    adapter = SciPySparseAdapter()
    rows = xp.asarray([0, 1, 1])
    cols = xp.asarray([0, 0, 1])
    values = array(xp, [2.0, 1.0, 3.0])
    K = adapter.from_coo(rows, cols, values, shape=(2, 2))

    solver = adapter.solver()
    solver.factor(K)
    actual = solver.solve(array(xp, [2.0, 7.0]))

    assert calls == []
    assert_allclose(xp, actual, array(xp, [1.0, 2.0]))


def test_cpu_sparse_default_falls_back_to_superlu_when_feral_is_missing(monkeypatch):
    xp = _numpy_namespace()
    monkeypatch.setitem(sys.modules, "feral", None)
    adapter = SciPySparseAdapter()
    rows = xp.asarray([0, 0, 1, 1])
    cols = xp.asarray([0, 1, 0, 1])
    values = array(xp, [3.0, 1.0, 1.0, 2.0])
    K = adapter.from_coo(rows, cols, values, shape=(2, 2))

    solver = adapter.solver()
    solver.factor(K)
    actual = solver.solve(array(xp, [5.0, 5.0]))

    assert_allclose(xp, actual, array(xp, [1.0, 2.0]))


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_feral_solver_rejects_non_finite_matrix(monkeypatch, bad):
    # A non-finite KKT entry (upstream derivatives overflowed) must surface as a
    # recoverable LinearSolveError the IPM regularization loop catches — not the
    # backend's raw ValueError, which would abort the whole solve (S2MPJ Task 2
    # solve_error cluster). The guard runs before the Feral import, so no binding
    # is needed to exercise it.
    xp = _numpy_namespace()
    monkeypatch.setitem(sys.modules, "feral", None)
    adapter = SciPySparseAdapter()
    rows = xp.asarray([0, 0, 1, 1])
    cols = xp.asarray([0, 1, 0, 1])
    values = array(xp, [2.0, bad, bad, -3.0])  # symmetric ⇒ Feral route
    K = adapter.from_coo(rows, cols, values, shape=(2, 2), symmetric=True)

    with pytest.raises(LinearSolveError, match="non-finite"):
        FeralSparseSolver().factor(K)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_superlu_solver_rejects_non_finite_matrix(bad):
    # The general (non-symmetric) fallback gets the same guard, so a non-finite
    # matrix fails fast rather than factoring to a non-finite solution.
    xp = _numpy_namespace()
    adapter = SciPySparseAdapter()
    rows = xp.asarray([0, 1, 1])
    cols = xp.asarray([0, 0, 1])
    values = array(xp, [2.0, bad, 3.0])  # non-symmetric ⇒ SuperLU route
    K = adapter.from_coo(rows, cols, values, shape=(2, 2))

    with pytest.raises(LinearSolveError, match="non-finite"):
        SuperLUSparseSolver().factor(K)


def test_feral_solver_rejects_nonsymmetric_operator_before_import(monkeypatch):
    xp = _numpy_namespace()
    monkeypatch.setitem(sys.modules, "feral", None)
    adapter = SciPySparseAdapter()
    rows = xp.asarray([0, 1, 1])
    cols = xp.asarray([0, 0, 1])
    values = array(xp, [2.0, 1.0, 3.0])
    K = adapter.from_coo(rows, cols, values, shape=(2, 2))

    with pytest.raises(ValueError, match="symmetric"):
        FeralSparseSolver().factor(K)


def test_feral_caches_symbolic_and_reanalyzes_on_pattern_change():
    # Symbolic analysis (fill-reducing ordering) is reused across same-pattern
    # factorizations and recomputed only when the pattern changes — the IPOPT
    # structure/values split that makes the per-iteration KKT factor cheap.
    pytest.importorskip("feral")
    xp = _numpy_namespace()
    adapter = SciPySparseAdapter()
    solver = adapter.solver()

    def factor_2x2(vals):
        rows = xp.asarray([0, 0, 1, 1])
        cols = xp.asarray([0, 1, 0, 1])
        solver.factor(adapter.from_coo(rows, cols, array(xp, vals), shape=(2, 2)))

    factor_2x2([2.0, 1.0, 1.0, -3.0])
    feral_solver = solver._inner._solver  # the persistent feral.Solver
    assert feral_solver.symbolic_call_count == 1

    # Same pattern, new values: symbolic reused (no re-analysis), solver persists.
    factor_2x2([5.0, 1.0, 1.0, -2.0])
    assert solver._inner._solver is feral_solver
    assert feral_solver.symbolic_call_count == 1
    actual = solver.solve(array(xp, [1.0, 2.0]))  # A = [[5,1],[1,-2]], det = -11
    assert_allclose(xp, actual, array(xp, [4.0 / 11.0, -9.0 / 11.0]))

    # Larger pattern (mimics the L-BFGS border growing): re-analysis, same solver.
    rows = xp.asarray([0, 0, 1, 1, 2])
    cols = xp.asarray([0, 1, 0, 1, 2])
    solver.factor(
        adapter.from_coo(
            rows, cols, array(xp, [2.0, 1.0, 1.0, -3.0, 4.0]), shape=(3, 3)
        )
    )
    assert solver._inner._solver is feral_solver
    assert feral_solver.symbolic_call_count == 2


def test_facade_persists_inner_solver_across_factors():
    pytest.importorskip("feral")
    from ipax.linalg.sparse import SparseDirectSolver

    xp = _numpy_namespace()
    adapter = SciPySparseAdapter()
    rows = xp.asarray([0, 0, 1, 1])
    cols = xp.asarray([0, 1, 0, 1])

    facade = SparseDirectSolver()
    facade.factor(
        adapter.from_coo(rows, cols, array(xp, [2.0, 1.0, 1.0, -3.0]), shape=(2, 2))
    )
    inner = facade._inner
    facade.factor(
        adapter.from_coo(rows, cols, array(xp, [3.0, 1.0, 1.0, -2.0]), shape=(2, 2))
    )
    assert facade._inner is inner  # the inner solver (and its symbolic cache) persists
