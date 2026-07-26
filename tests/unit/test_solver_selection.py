"""Unit tests for linear solver strategy selection."""

from __future__ import annotations

import pytest

from ipax.backend.namespace import Capabilities
from ipax.linalg.dense import DenseSolver
from ipax.linalg.krylov import KrylovSolver
from ipax.linalg.solver import select_solver
from ipax.options import DenseOptions, Options
from tests._helpers import implemented


def _caps(*, sparse: bool = False) -> Capabilities:
    return Capabilities(
        name="test",
        has_linalg=True,
        linalg_functions=frozenset({"cholesky", "solve"}),
        has_sparse_adapter=sparse,
        supports_autodiff=False,
        devices=("cpu",),
        default_float="float64",
    )


@pytest.mark.parametrize("mode", ["dense", "krylov"])
def test_select_solver_honors_explicit_linsolve_mode(mode):
    with implemented("solver auto-selection"):
        solver = select_solver(
            n_vars=100,
            has_equalities=False,
            capabilities=_caps(),
            options=Options(linsolve=mode),
        )

    expected_type = DenseSolver if mode == "dense" else KrylovSolver
    assert isinstance(solver, expected_type)


def test_select_solver_auto_switches_to_krylov_at_m4_scale():
    small = select_solver(
        n_vars=9_999,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(linsolve="auto"),
    )
    m4_scale = select_solver(
        n_vars=10_000,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(linsolve="auto"),
    )

    assert isinstance(small, DenseSolver)
    assert isinstance(m4_scale, KrylovSolver)


def test_select_solver_prefers_sparse_when_requested_and_available():
    with implemented("sparse solver selection"):
        solver = select_solver(
            n_vars=10_000,
            has_equalities=True,
            capabilities=_caps(sparse=True),
            options=Options(linsolve="sparse"),
        )

    assert solver.__class__.__name__.lower().startswith("sparse")


@pytest.mark.parametrize("mode", ["dense", "auto"])
def test_select_solver_threads_dense_options_through(mode):
    solver = select_solver(
        n_vars=100,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(
            linsolve=mode,
            dense=DenseOptions(kkt_route="augmented"),  # type: ignore[arg-type]
        ),
    )
    assert isinstance(solver, DenseSolver)
    assert solver._options.kkt_route == "augmented"


def test_select_solver_rejects_unavailable_sparse_backend():
    try:
        select_solver(
            n_vars=10_000,
            has_equalities=True,
            capabilities=_caps(sparse=False),
            options=Options(linsolve="sparse"),
        )
    except NotImplementedError as exc:
        pytest.xfail(f"sparse solver selection: {exc}")
    except RuntimeError as exc:
        assert "sparse" in str(exc).lower()
    else:
        pytest.fail("expected a RuntimeError when sparse solving is unavailable")


# --- n ≪ m (tall, Gram-capable) auto-selection ------------------------------


def test_select_solver_auto_tall_gram_capable_prefers_dense():
    # Above the plain dense cutoff, but with m ≫ n and a Gram-capable inequality
    # Jacobian the condensed normal-equations (dense) route factors an n×n block
    # instead of Krylov matvecs through the huge m×n Jacobian.
    solver = select_solver(
        n_vars=15_000,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(linsolve="auto"),
        m_ineq=300_000,
        ineq_gram_capable=lambda: True,
    )
    assert isinstance(solver, DenseSolver)


def test_select_solver_auto_tall_without_gram_stays_krylov():
    solver = select_solver(
        n_vars=15_000,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(linsolve="auto"),
        m_ineq=300_000,
        ineq_gram_capable=lambda: False,
    )
    assert isinstance(solver, KrylovSolver)


def test_select_solver_auto_tall_needs_row_excess():
    # m comparable to n: no normal-equations advantage, keep Krylov.
    solver = select_solver(
        n_vars=15_000,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(linsolve="auto"),
        m_ineq=30_000,
        ineq_gram_capable=lambda: True,
    )
    assert isinstance(solver, KrylovSolver)


def test_select_solver_auto_tall_sparse_jacobian_stays_krylov():
    # 2026-07 tall-crossover measurement: at ~1% density the condensed dense
    # route loses to Krylov above the plain dense cutoff (n=10k: 75.5 vs
    # 46.5 s/iter) — the tall-dense win comes from the adapter's dense-GEMM
    # Gram, which only engages at >= ~5% density. A sparse tall Jacobian must
    # therefore stay on Krylov.
    solver = select_solver(
        n_vars=15_000,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(linsolve="auto"),
        m_ineq=300_000,
        ineq_gram_capable=lambda: True,
        ineq_density=lambda: 0.01,
    )
    assert isinstance(solver, KrylovSolver)


def test_select_solver_auto_tall_dense_jacobian_prefers_dense():
    # Dense-ish rows (TROTS dose matrices: 11-53%) keep the dense-GEMM Gram
    # fast path — the measured 13-19x TROTS per-iteration wins.
    solver = select_solver(
        n_vars=15_000,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(linsolve="auto"),
        m_ineq=300_000,
        ineq_gram_capable=lambda: True,
        ineq_density=lambda: 0.2,
    )
    assert isinstance(solver, DenseSolver)


def test_select_solver_auto_tall_unknown_density_prefers_dense():
    # A Gram-capable operator without COO structure (matrix-free with a
    # structured Gram) reports no density; keep the previous behavior.
    solver = select_solver(
        n_vars=15_000,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(linsolve="auto"),
        m_ineq=300_000,
        ineq_gram_capable=lambda: True,
        ineq_density=lambda: None,
    )
    assert isinstance(solver, DenseSolver)


def test_select_solver_small_problems_skip_gram_probe():
    # The capability probe may evaluate the Jacobian at x0 — it must only run
    # when the decision actually depends on it.
    def probe() -> bool:
        raise AssertionError("gram probe must not run for small problems")

    solver = select_solver(
        n_vars=100,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(linsolve="auto"),
        m_ineq=10_000,
        ineq_gram_capable=probe,
    )
    assert isinstance(solver, DenseSolver)


# --- tall + sparse Gram: normal-equations auto-selection ---------------------


def _tall_sparse_kwargs(**overrides):
    """Tall, sparse-Jacobian selection scenario (density below the dense-GEMM
    crossover, so the dense tall branch declines and the NE gate decides)."""
    kwargs = {
        "n_vars": 15_000,
        "has_equalities": False,
        "capabilities": _caps(sparse=True),
        "options": Options(linsolve="auto"),
        "m_ineq": 300_000,
        "ineq_gram_capable": lambda: True,
        "ineq_density": lambda: 0.01,
        "ineq_gram_fill": lambda: 0.002,
    }
    kwargs.update(overrides)
    return kwargs


def test_select_solver_auto_tall_sparse_gram_selects_normal_equations():
    # n=20k banded validation (2026-07): sparse NE solved in 62 s where Krylov
    # ran 50+ min unconverged — when the fill probe certifies a sparse Gram,
    # the auto route must take it.
    from ipax.linalg.sparse import SparseDirectSolver

    solver = select_solver(**_tall_sparse_kwargs())
    assert isinstance(solver, SparseDirectSolver)
    assert solver.form == "normal_equations"


def test_select_solver_auto_tall_filled_gram_stays_krylov():
    # Scattered sparsity: the Gram fills in (the reason the route was opt-in) —
    # the probe reports it and the selection keeps Krylov.
    solver = select_solver(**_tall_sparse_kwargs(ineq_gram_fill=lambda: 0.5))
    assert isinstance(solver, KrylovSolver)


def test_select_solver_auto_tall_unknown_gram_fill_stays_krylov():
    solver = select_solver(**_tall_sparse_kwargs(ineq_gram_fill=lambda: None))
    assert isinstance(solver, KrylovSolver)


def test_select_solver_auto_tall_ne_allows_equalities():
    # The NE form keeps ∇c as an explicit border next to the condensed block,
    # so equalities no longer veto the route here. (Whether the equality
    # Jacobian can emit COO structure is the solve()-side probe's concern —
    # it withholds ineq_gram_fill when it cannot.)
    from ipax.linalg.sparse import SparseDirectSolver

    solver = select_solver(**_tall_sparse_kwargs(has_equalities=True))
    assert isinstance(solver, SparseDirectSolver)
    assert solver.form == "normal_equations"


def test_select_solver_auto_tall_ne_requires_sparse_adapter():
    solver = select_solver(**_tall_sparse_kwargs(capabilities=_caps(sparse=False)))
    assert isinstance(solver, KrylovSolver)


def test_select_solver_auto_tall_dense_win_skips_fill_probe():
    # Dense-ish rows take the dense tall route before the (Jacobian-evaluating)
    # fill probe is ever consulted.
    def probe() -> float:
        raise AssertionError("fill probe must not run when the dense route wins")

    solver = select_solver(
        **_tall_sparse_kwargs(ineq_density=lambda: 0.2, ineq_gram_fill=probe)
    )
    assert isinstance(solver, DenseSolver)


# --- explicit sparse mode: kkt_route="auto" ----------------------------------
#
# ``linsolve="sparse"`` used to honor ``SparseOptions.kkt_route`` blindly, so a
# tall problem with a dense Jacobian (TROTS Prostate_BT: n≈70, m≈5.7k, J 100%
# dense) factored an effectively dense (n+m)-sized augmented system through
# sparse LDL^T machinery — ~8x slower end-to-end than condensing to n×n.
# ``kkt_route="auto"`` reuses the same tall gate and thresholds as the
# ``linsolve="auto"`` heuristic to pick the form.


def _sparse_auto_kwargs(**overrides):
    from ipax.options import SparseOptions

    kwargs = {
        "n_vars": 1_000,
        "has_equalities": False,
        "capabilities": _caps(sparse=True),
        "options": Options(linsolve="sparse", sparse=SparseOptions(kkt_route="auto")),
        "m_ineq": 15_000,  # 15x n: past _TALL_ROW_EXCESS = 10
        "ineq_gram_capable": lambda: True,
        "ineq_density": lambda: 0.3,
        "ineq_gram_fill": lambda: 0.001,
    }
    kwargs.update(overrides)
    return kwargs


def test_sparse_auto_tall_sparse_gram_selects_normal_equations():
    # Localized rows: Gram fill below _TALL_SPARSE_NE_MAX_FILL ⇒ n×n NE form.
    from ipax.linalg.sparse import SparseDirectSolver

    solver = select_solver(**_sparse_auto_kwargs())
    assert isinstance(solver, SparseDirectSolver)
    assert solver.form == "normal_equations"


def test_sparse_auto_tall_dense_jacobian_selects_normal_equations():
    # The Gram fills in (0.9 > threshold) but J itself is 30% dense: the
    # augmented factor's border is that dense J, so there is no sparsity for it
    # to exploit either — the tall n×n condensation wins (the same
    # _TALL_DENSE_MIN_DENSITY crossover as the auto-mode dense-GEMM branch).
    from ipax.linalg.sparse import SparseDirectSolver

    solver = select_solver(**_sparse_auto_kwargs(ineq_gram_fill=lambda: 0.9))
    assert isinstance(solver, SparseDirectSolver)
    assert solver.form == "normal_equations"


def test_sparse_auto_filled_gram_sparse_jacobian_stays_augmented():
    # Non-localized rows on a genuinely sparse J: NE would densify n², the
    # augmented factor stays as sparse as J — keep it.
    from ipax.linalg.sparse import SparseDirectSolver

    solver = select_solver(
        **_sparse_auto_kwargs(ineq_gram_fill=lambda: 0.9, ineq_density=lambda: 0.01)
    )
    assert isinstance(solver, SparseDirectSolver)
    assert solver.form == "augmented"


def test_sparse_auto_without_fill_probe_stays_augmented():
    # The caller withholds the fill probe when the NE form is unusable (e.g.
    # analytic Hessian, matrix-free ∇c): auto must not gamble.
    from ipax.linalg.sparse import SparseDirectSolver

    solver = select_solver(**_sparse_auto_kwargs(ineq_gram_fill=None))
    assert isinstance(solver, SparseDirectSolver)
    assert solver.form == "augmented"


def test_sparse_auto_non_tall_stays_augmented_without_probing():
    # Not tall ⇒ augmented, and the (Jacobian-evaluating) probes never run.
    from ipax.linalg.sparse import SparseDirectSolver

    def probe() -> float:
        raise AssertionError("probes must not run for non-tall problems")

    solver = select_solver(
        **_sparse_auto_kwargs(m_ineq=5_000, ineq_density=probe, ineq_gram_fill=probe)
    )
    assert isinstance(solver, SparseDirectSolver)
    assert solver.form == "augmented"


def test_sparse_auto_beyond_tall_bound_stays_augmented():
    # Past _TALL_DENSE_MAX_VARS the n×n condensed factor is no longer the
    # obvious win; keep the sparse route on its documented default.
    from ipax.linalg.sparse import SparseDirectSolver

    solver = select_solver(**_sparse_auto_kwargs(n_vars=25_000, m_ineq=300_000))
    assert isinstance(solver, SparseDirectSolver)
    assert solver.form == "augmented"


def test_sparse_explicit_routes_bypass_the_auto_gate():
    # Explicit "augmented"/"normal_equations" requests are honored verbatim.
    from ipax.linalg.sparse import SparseDirectSolver
    from ipax.options import SparseOptions

    for route in ("augmented", "normal_equations"):
        solver = select_solver(
            **_sparse_auto_kwargs(
                options=Options(
                    linsolve="sparse", sparse=SparseOptions(kkt_route=route)
                ),
                ineq_gram_fill=None,
            )
        )
        assert isinstance(solver, SparseDirectSolver)
        assert solver.form == route
