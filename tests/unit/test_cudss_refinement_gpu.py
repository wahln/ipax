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

from tests._cudss import require_deterministic_cudss

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


def _cudss_or_skip(*, deterministic: bool = False) -> None:
    try:
        cudss = cupy_adapter._load_cudss()
    except ImportError as exc:
        pytest.skip(f"cuDSS runtime unavailable: {exc}")
    if deterministic:
        require_deterministic_cudss(cudss)


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


# --- SparseOptions.deterministic: cuDSS DETERMINISTIC_MODE -------------------


def _synthetic_kkt(n: int = 1500, m: int = 500, seed: int = 0):
    """A seeded sparse saddle point whose default cuDSS solve is not reproducible.

    Random symmetric Hessian (a third of its diagonal zeroed, so static
    pivoting perturbs pivots, as on ROTDISC/CATENARY) bordered by a random
    sparse Jacobian. On cuDSS 0.7.1 five fresh default-mode solvers returned
    five different solutions (max difference 0.24), the run-to-run
    non-determinism the 2026-09-30 study measured on captured KKT systems.
    """
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(seed)
    H = sp.random(n, n, density=4.0 / n, random_state=rng)
    H = H + H.T
    d = rng.uniform(1.0, 10.0, n)
    d[rng.random(n) < 0.3] = 0.0
    J = sp.random(m, n, density=3.0 / n, random_state=rng)
    K = sp.bmat([[H + sp.diags(d), J.T], [J, -1e-8 * sp.eye(m)]]).tocoo()
    rhs = np.random.default_rng(9).standard_normal(n + m)
    return K, rhs


def _factor_coo(K: object, **policy: object):
    adapter = cupy_adapter.CuPySparseAdapter(**policy)
    operator = adapter.from_coo(
        cupy.asarray(K.row),  # type: ignore[attr-defined]
        cupy.asarray(K.col),  # type: ignore[attr-defined]
        cupy.asarray(K.data),  # type: ignore[attr-defined]
        shape=K.shape,  # type: ignore[attr-defined]
        symmetric=True,
    )
    solver = adapter.solver()
    solver.factor(operator)
    return solver


def test_deterministic_mode_makes_fresh_factorizations_bitwise_identical():
    _cudss_or_skip(deterministic=True)
    K, rhs = _synthetic_kkt()
    solutions = []
    for _ in range(3):
        solver = _factor_coo(K, deterministic=True)
        solutions.append(cupy.asnumpy(solver.solve(cupy.asarray(rhs))))
        solver._inner.close()
    # Bitwise, not allclose: reproducibility is the whole point of the mode.
    assert all(x.tobytes() == solutions[0].tobytes() for x in solutions[1:])
    np.testing.assert_allclose(
        K.dot(solutions[0]),  # type: ignore[attr-defined]
        rhs,
        atol=1e-2 * float(np.max(np.abs(rhs))),
    )


def _config_int(solver: object, name: str) -> int:
    import ctypes

    from nvmath.bindings import cudss

    value = ctypes.c_int32()
    written = ctypes.c_size_t()
    cudss.config_get(
        solver._config,  # type: ignore[attr-defined]
        int(getattr(cudss.ConfigParam, name)),
        ctypes.addressof(value),
        ctypes.sizeof(value),
        ctypes.addressof(written),
    )
    return int(value.value)


def test_sparse_options_deterministic_reaches_the_cudss_config():
    # SparseOptions(deterministic=True) → select_solver → facade →
    # configure_deterministic → the cuDSS config, read back from the runtime;
    # the default options leave the runtime's default (0).
    _cudss_or_skip(deterministic=True)
    from ipax.options import SparseOptions

    on, _ = _options_solve(SparseOptions(deterministic=True))
    assert _config_int(on._inner._inner, "DETERMINISTIC_MODE") == 1
    off, _ = _options_solve(SparseOptions())
    assert _config_int(off._inner._inner, "DETERMINISTIC_MODE") == 0


# --- the inertia of a perturbed-pivot factorization ------------------------

# One KKT system of CATENARY (S2MPJ, exact Hessian, sparse route) captured
# from the 2026-09 CuPy sweep, as COO of the full symmetric 19×19 matrix. Five
# of its rows/columns (4, 7, 10, 13, 14) are identically zero, so it is
# singular; a dense Bunch–Kaufman LDLᵀ finds 7 zero pivots, inertia (8, 4, 7).
# cuDSS perturbs 9 pivots to ±1e-13 and counts them by that sign: (15, 4, 0),
# a nonsingular inertia that equals the IPM's target. That reading is kept
# (withholding it was measured and rejected on the corpus, 2026-10-01); this
# test pins it, so a runtime that starts reporting the zeros is noticed.
_CATENARY_ROWS = [0, 0, 1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 6, 6, 7, 8, 8, 9, 9, 9, 10]
_CATENARY_ROWS += [11, 11, 12, 12, 13, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17]
_CATENARY_ROWS += [17, 17, 18, 18, 18, 18]
_CATENARY_COLS = [0, 15, 1, 2, 15, 3, 15, 16, 4, 5, 16, 6, 16, 17, 7, 8, 17, 9, 17]
_CATENARY_COLS += [18, 10, 11, 18, 12, 18, 13, 14, 0, 2, 3, 15, 3, 5, 6, 16, 6, 8, 9]
_CATENARY_COLS += [17, 9, 11, 12, 18]
_CATENARY_VALS = [
    2e15, -1.4142135623730951, 2e15, 2e15, -1.4142135623730951, 0.0,
    2.8284271247461903, -0.37736970136067605, 0.0, 0.0, -1.9640753825897215, 0.0,
    2.3414450839503975, -1.1747801618140798, 0.0, 0.0, -1.6186079115734306, 0.0,
    2.7933880733875105, -1.8336365464521482, 0.0, 0.0, -0.798609426137161,
    2.0888710079901012e16, 2.6322459725893093, 0.0, 0.0, -1.4142135623730951,
    -1.4142135623730951, 2.8284271247461903, -1e-08, -0.37736970136067605,
    -1.9640753825897215, 2.3414450839503975, -1e-08, -1.1747801618140798,
    -1.6186079115734306, 2.7933880733875105, -1e-08, -1.8336365464521482,
    -0.798609426137161, 2.6322459725893093, -1e-08,
]  # fmt: skip
_CATENARY_RHS = [
    0.0, -50.0, 0.0, 0.0, -100.0, 0.0, 0.0, -100.0, 0.0, 0.0, -100.0, 0.0,
    7368421.146619504, -50.0, 0.0, -2.220446049250313e-16, -1.6653345369377348e-15,
    -4.551914400963142e-15, -2.170486013142181e-14,
]  # fmt: skip


def _factor_triplets(rows, cols, vals, n):
    adapter = cupy_adapter.CuPySparseAdapter()
    operator = adapter.from_coo(
        cupy.asarray(rows),
        cupy.asarray(cols),
        cupy.asarray(vals, dtype=cupy.float64),
        shape=(n, n),
        symmetric=True,
    )
    solver = adapter.solver()
    solver.factor(operator)
    return solver


def test_perturbed_factor_of_a_singular_kkt_reports_the_perturbed_count():
    _cudss_or_skip()
    dense = np.zeros((19, 19))
    dense[_CATENARY_ROWS, _CATENARY_COLS] = _CATENARY_VALS
    zero_rows = np.flatnonzero(~dense.any(axis=1))
    assert zero_rows.tolist() == [4, 7, 10, 13, 14]  # singular by construction
    solver = _factor_triplets(_CATENARY_ROWS, _CATENARY_COLS, _CATENARY_VALS, 19)
    # The certificate accepts the solve: relative to ‖b‖∞ ≈ 7e6 the
    # inconsistent rows' residual (≤ 1e2) stays under 1e-2.
    solver.solve(cupy.asarray(_CATENARY_RHS))
    premise = "premise: cuDSS 0.7.1 perturbs this system's zero pivots"
    assert solver._inner._npivots > 0, premise
    # The count of A + E: every perturbed zero pivot counted by its ±ε sign,
    # no zero eigenvalue reported although the matrix is singular.
    assert solver.inertia_or_none() == (15, 4, 0)


def test_unperturbed_factor_reports_its_inertia():
    _cudss_or_skip()
    saddle = np.asarray([[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, -2.0]])
    rows, cols = np.nonzero(saddle)
    solver = _factor_triplets(rows, cols, saddle[rows, cols], 3)
    solver.solve(cupy.asarray([1.0, 2.0, 3.0]))
    assert solver._inner._npivots == 0
    assert solver.inertia_or_none() == (2, 1, 0)
