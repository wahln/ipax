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
