"""Real-cuDSS checks of the perturbed-pivot refinement (GPU-gated).

cuDSS's static pivoting perturbs a zero pivot of a symmetric saddle-point
matrix instead of pairing it off in a 2×2 block; iterative refinement against
the unperturbed matrix recovers the accurate solution (Li & Demmel 2003,
GESP; Higham 2002, ch. 12). The fakes in ``test_cupy_sparse_adapter.py`` pin
the loop's logic; these pin that the real runtime does perturb such a pivot
and that refinement fixes it.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from ipax.linalg.solver import LinearSolveError  # noqa: E402

pytestmark = pytest.mark.gpu


def _cuda_available() -> bool:
    try:
        return bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


if not _cuda_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

import ipax.backend.sparse.cupy as cupy_adapter  # noqa: E402


def _cudss_or_skip() -> None:
    try:
        cupy_adapter._load_cudss()
    except ImportError as exc:
        pytest.skip(f"cuDSS runtime unavailable: {exc}")


# One KKT system of MAKELA1 (S2MPJ, exact Hessian, sparse route) captured
# from the 2026-09 CuPy sweep: rows 0–2 are the primal block, the zero at
# (2, 2) is the minimax epigraph variable's (linear) Hessian diagonal, rows
# 3–4 the two active constraints. Well conditioned (cond ≈ 4e2), yet cuDSS
# perturbs one pivot and, unrefined, returns a solve with relative residual
# ≈ 3e-4 — accepted by the 1e-2 certificate, a poor Newton step.
_MAKELA1_KKT = np.asarray(
    [
        [20.0, 0.0, 0.0, -1.0, -2.0],
        [0.0, 20.0, 0.0, -1.0, -2.0],
        [0.0, 0.0, 0.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0, -1e-3, 0.0],
        [-2.0, -2.0, -1.0, 0.0, -1e-3],
    ]
)
_MAKELA1_RHS = np.asarray([2060.0, 2060.0, 1539.0, 0.0, 0.0])


def _factor(matrix: np.ndarray, **policy: object):
    rows, cols = np.nonzero(matrix)
    adapter = cupy_adapter.CuPySparseAdapter(**policy)
    operator = adapter.from_coo(
        cupy.asarray(rows),
        cupy.asarray(cols),
        cupy.asarray(matrix[rows, cols]),
        shape=matrix.shape,
        symmetric=True,
    )
    solver = adapter.solver()
    solver.factor(operator)
    return solver


def _relative_residual(matrix: np.ndarray, rhs: np.ndarray, x: object) -> float:
    x_host = cupy.asnumpy(x)
    return float(np.max(np.abs(rhs - matrix @ x_host)) / np.max(np.abs(rhs)))


def test_perturbed_saddle_point_is_refined_to_the_reference_solution():
    _cudss_or_skip()
    reference = np.linalg.solve(_MAKELA1_KKT, _MAKELA1_RHS)

    unrefined = _factor(_MAKELA1_KKT, max_refinement_steps=0)
    x0 = unrefined.solve(cupy.asarray(_MAKELA1_RHS))
    premise = (
        "premise: this cuDSS runtime/ordering no longer perturbs the zero pivot "
        "(validated on cuDSS 0.7.1); pick a new perturbed system"
    )
    assert unrefined._inner._npivots > 0, premise
    assert _relative_residual(_MAKELA1_KKT, _MAKELA1_RHS, x0) > 1e-8, premise

    refined = _factor(_MAKELA1_KKT)
    x = refined.solve(cupy.asarray(_MAKELA1_RHS))
    assert refined._inner._npivots > 0, premise
    # Stopped by IPOPT's ratio ≤ 1e-10, whose denominator also holds ‖x‖∞.
    assert _relative_residual(_MAKELA1_KKT, _MAKELA1_RHS, x) <= 1e-9
    # Forward error ≤ ‖A⁻¹‖ ‖r‖: cond ≈ 4e2 × 1e-10 × ‖b‖∞/‖x‖∞ ≈ 1e-7 relative.
    np.testing.assert_allclose(cupy.asnumpy(x), reference, rtol=1e-7, atol=0.0)
    assert 1 <= refined._inner.last_refinement_steps <= 10


def test_inconsistent_singular_system_is_still_rejected_after_refinement():
    _cudss_or_skip()
    solver = _factor(np.ones((2, 2)))
    with pytest.raises(LinearSolveError, match="perturbed"):
        solver.solve(cupy.asarray([1.0, 2.0]))


def test_unperturbed_factorization_is_not_refined(monkeypatch):
    # NPIVOTS = 0: one SOLVE phase per solve and no residual SpMV.
    _cudss_or_skip()
    spd = np.asarray([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    rhs = np.asarray([1.0, 2.0, 3.0])
    solver = _factor(spd)
    calls = {"solve": 0, "certify": 0}
    solve_columns = cupy_adapter.CuDSSSparseSolver._solve_columns
    certify = cupy_adapter.CuDSSSparseSolver._certify_perturbed_solve

    def counting_solve(self, b):
        calls["solve"] += 1
        return solve_columns(self, b)

    def counting_certify(self, b, x):
        calls["certify"] += 1
        return certify(self, b, x)

    monkeypatch.setattr(
        cupy_adapter.CuDSSSparseSolver, "_solve_columns", counting_solve
    )
    monkeypatch.setattr(
        cupy_adapter.CuDSSSparseSolver, "_certify_perturbed_solve", counting_certify
    )
    for _ in range(2):
        x = solver.solve(cupy.asarray(rhs))
    assert solver._inner._npivots == 0
    assert calls == {"solve": 2, "certify": 0}
    np.testing.assert_allclose(cupy.asnumpy(x), np.linalg.solve(spd, rhs), rtol=1e-12)


class _KKTOperator:
    """The captured KKT matrix as a COO-emitting operator on the device."""

    def __init__(self, matrix: np.ndarray) -> None:
        rows, cols = np.nonzero(matrix)
        self._rows = cupy.asarray(rows)
        self._cols = cupy.asarray(cols)
        self._values = cupy.asarray(matrix[rows, cols])
        self.shape = matrix.shape

    def to_coo(self, like: object = None) -> tuple[object, ...]:
        del like
        return self._rows, self._cols, self._values, self.shape

    def coo_values(self, like: object = None) -> object:
        del like
        return self._values

    def coo_pattern_signature(self) -> object:
        return ("makela1-kkt",)

    def symmetry_hint(self) -> bool:
        return True


def _options_solve(sparse_options: object) -> tuple[object, object]:
    """Factor + solve the MAKELA1 system through ``Options`` → ``select_solver``."""
    from ipax.backend.namespace import capabilities
    from ipax.linalg.solver import select_solver
    from ipax.options import Options

    xp = cupy
    solver = select_solver(
        n_vars=_MAKELA1_KKT.shape[0],
        has_equalities=True,
        capabilities=capabilities(xp),
        options=Options(linsolve="sparse", sparse=sparse_options),
    )
    solver.factor(_KKTOperator(_MAKELA1_KKT))  # type: ignore[arg-type]
    x = solver.solve(cupy.asarray(_MAKELA1_RHS))
    return solver, x


def test_sparse_options_policy_reaches_cudss_end_to_end():
    # SparseOptions(perturbed_pivot_refinement_steps=0) switches the
    # refinement off through the public options path (select_solver →
    # SparseDirectSolver facade → adapter hook → cuDSS solver); the default
    # SparseOptions() refines exactly as the adapter defaults do.
    _cudss_or_skip()
    from ipax.options import SparseOptions

    off, x_off = _options_solve(SparseOptions(perturbed_pivot_refinement_steps=0))
    cudss = off._inner._inner  # facade → CuPySparseSolver → CuDSSSparseSolver
    # By name: the facade imports the adapter module afresh when the fake-CuPy
    # fixture of test_cupy_sparse_adapter.py dropped it from sys.modules in
    # this worker, so class identity with ``cupy_adapter`` is not guaranteed.
    assert type(cudss).__name__ == "CuDSSSparseSolver"
    adapter_module = sys.modules[type(cudss).__module__]
    assert cudss._npivots > 0, "premise: cuDSS perturbs this system's zero pivot"
    assert cudss._max_refinement_steps == 0
    assert cudss.last_refinement_steps == 0
    assert _relative_residual(_MAKELA1_KKT, _MAKELA1_RHS, x_off) > 1e-8

    on, x_on = _options_solve(SparseOptions())
    cudss = on._inner._inner
    assert cudss._max_refinement_steps == adapter_module._PERTURBED_MAX_REFINEMENT_STEPS
    assert 1 <= cudss.last_refinement_steps <= 10
    assert _relative_residual(_MAKELA1_KKT, _MAKELA1_RHS, x_on) <= 1e-9
