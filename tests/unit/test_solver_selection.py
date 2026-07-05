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
