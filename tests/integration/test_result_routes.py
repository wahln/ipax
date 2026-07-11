"""``Result.routes`` — which route each auto-selectable mechanism actually took.

The solver auto-selects a lot (linear solver, KKT form, Hessian source, …) and
until now only fragments were visible (``Result.linear_solver``, the tier-1
setup log line). ``Result.routes`` records requested → resolved for each
category in one place, so a user can read off *why* a solve behaved the way it
did without re-running at higher verbosity.
"""

from __future__ import annotations

import pytest

from ipax import COOOperator, Options, Problem, solve
from ipax.backend.sparse import get_sparse_adapter
from ipax.options import DenseOptions, SparseOptions


class _BoundQP(Problem):
    """min ½‖x − 1.5‖² s.t. x ≥ 0 — no analytic Hessian (resolves to L-BFGS)."""

    def __init__(self, namespace) -> None:
        self._xp = namespace

    @property
    def n_vars(self) -> int:
        return 4

    def objective(self, x):
        d = x - 1.5
        return 0.5 * self._xp.sum(d * d)

    def gradient(self, x):
        return x - 1.5

    def bounds(self):
        return self._xp.zeros((4,), dtype=self._xp.float64), None


class _IneqQP(_BoundQP):
    """The bound QP plus two banded inequality rows and an analytic Hessian."""

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del x, y_eq, y_ineq
        return sigma * self._xp.eye(4, dtype=self._xp.float64)

    def ineq_constraints(self, x):
        return self._jac().matvec(x) - 2.2

    def ineq_jacobian(self, x):
        del x
        return self._jac()

    def _jac(self):
        xp = self._xp
        return COOOperator(
            xp.asarray([0, 0, 1, 1]),
            xp.asarray([0, 1, 2, 3]),
            xp.ones((4,), dtype=xp.float64),
            (2, 4),
            pattern_key="A",
        )


def test_routes_record_the_default_auto_selection(namespace):
    result = solve(_BoundQP(namespace), namespace.full((4,), 0.4))

    routes = result.routes
    assert routes is not None
    assert routes.linsolve_requested == "auto"
    assert "dense" in routes.linear_solver
    assert routes.linear_solver == result.linear_solver
    assert routes.kkt_form == "condensed"
    assert routes.hessian_requested == "auto"
    assert routes.hessian == "lbfgs"
    assert routes.globalization == "filter"
    assert routes.mu_schedule == "monotone"
    assert routes.scaling == "gradient-based"  # the IPOPT-style default
    assert routes.corrections == "none"


def test_routes_reflect_forced_krylov(namespace):
    result = solve(
        _BoundQP(namespace),
        namespace.full((4,), 0.4),
        options=Options(linsolve="krylov"),
    )

    routes = result.routes
    assert routes is not None
    assert routes.linsolve_requested == "krylov"
    assert "krylov" in routes.linear_solver
    assert routes.kkt_form == "condensed"


def test_routes_reflect_the_sparse_normal_equations_form(namespace):
    if get_sparse_adapter(namespace) is None:
        pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")
    result = solve(
        _IneqQP(namespace),
        namespace.full((4,), 0.4),
        options=Options(
            linsolve="sparse",
            sparse=SparseOptions(kkt_route="normal_equations"),
            hessian="exact",
        ),
    )

    routes = result.routes
    assert routes is not None
    assert routes.kkt_form == "normal_equations"
    assert routes.linear_solver.startswith("sparse-NE")
    assert routes.hessian == "exact"
    assert routes.hessian_requested == "exact"


def test_routes_reflect_the_dense_augmented_form(namespace):
    result = solve(
        _IneqQP(namespace),
        namespace.full((4,), 0.4),
        options=Options(
            linsolve="dense",
            dense=DenseOptions(kkt_route="augmented"),
            hessian="exact",
        ),
    )

    routes = result.routes
    assert routes is not None
    assert routes.kkt_form == "augmented"


def test_routes_echo_configured_algorithm_pieces(namespace):
    result = solve(
        _IneqQP(namespace),
        namespace.full((4,), 0.4),
        options=Options(scaling="none", corrections="mehrotra"),
    )

    routes = result.routes
    assert routes is not None
    assert routes.scaling == "none"
    assert routes.corrections == "mehrotra"


def test_routes_absent_on_pre_solver_exits(namespace):
    class _InfeasibleBounds(_BoundQP):
        def bounds(self):
            xp = self._xp
            return xp.ones((4,), dtype=xp.float64), xp.zeros((4,), dtype=xp.float64)

    result = solve(_InfeasibleBounds(namespace), namespace.full((4,), 0.4))
    assert not result.success
    assert result.routes is None


def test_format_result_reports_routes(namespace):
    from ipax._logging import format_result

    result = solve(_BoundQP(namespace), namespace.full((4,), 0.4))
    text = format_result(result)
    assert "routes" in text
    assert "auto->" in text  # requested -> resolved rendering

    bare = solve(_BoundQP(namespace), namespace.full((4,), 0.4))
    bare = __import__("dataclasses").replace(bare, routes=None)
    assert "routes" not in format_result(bare)


def test_routes_is_exported():
    import ipax

    assert hasattr(ipax, "Routes")
    assert "Routes" in ipax.__all__
