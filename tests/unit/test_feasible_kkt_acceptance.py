"""Driver wiring for the feasible-point KKT-progress line-search rescue.

The filter line search accepts a first trial that fails Armijo at a feasible
iterate (θ0 = 0) when the scaled KKT error decreases sufficiently
(``LineSearchOptions.feasible_kkt_progress``). The driver must hand the
certifying closure to the search exactly at feasible iterates — and not at
all when the option is disabled.
"""

from __future__ import annotations

from ipax import Options, Status, solve
from ipax.ipm.filter_ls import FilterLineSearch
from ipax.options import LineSearchOptions
from ipax.testing.problems import HS35
from tests._helpers import array


def _record_kkt_progress_kwargs(monkeypatch):
    seen: list[object] = []
    original = FilterLineSearch.search

    def recording(self, **kwargs):
        seen.append(kwargs.get("kkt_progress"))
        return original(self, **kwargs)

    monkeypatch.setattr(FilterLineSearch, "search", recording)
    return seen


def test_driver_passes_certifier_at_feasible_iterates(namespace, monkeypatch):
    # HS35 from a strictly feasible interior point: the linear inequality keeps
    # θ = 0 along the whole trajectory, so every line search must receive the
    # KKT-progress certifier.
    seen = _record_kkt_progress_kwargs(monkeypatch)
    problem = HS35(namespace)
    result = solve(
        problem,
        array(namespace, [0.5, 0.5, 0.5]),
        options=Options(hessian="exact", linsolve="dense"),
    )

    assert result.status is Status.OPTIMAL
    assert len(seen) > 0
    assert all(fn is not None for fn in seen)


def test_certifier_disabled_by_option(namespace, monkeypatch):
    seen = _record_kkt_progress_kwargs(monkeypatch)
    problem = HS35(namespace)
    result = solve(
        problem,
        array(namespace, [0.5, 0.5, 0.5]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            line_search=LineSearchOptions(feasible_kkt_progress=None),
        ),
    )

    assert result.status is Status.OPTIMAL
    assert len(seen) > 0
    assert all(fn is None for fn in seen)


def test_certifier_absent_at_infeasible_iterates(namespace, monkeypatch):
    # From an infeasible start (g = +3) the early iterations have θ0 > 0: the
    # rescue is a θ0 = 0 mechanism only, so those searches get no certifier.
    seen = _record_kkt_progress_kwargs(monkeypatch)
    problem = HS35(namespace)
    result = solve(
        problem,
        array(namespace, [2.0, 2.0, 2.0]),
        options=Options(hessian="exact", linsolve="dense"),
    )

    assert result.status is Status.OPTIMAL
    assert len(seen) > 0
    assert seen[0] is None  # first iteration is infeasible
