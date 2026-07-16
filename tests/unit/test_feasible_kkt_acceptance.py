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
    # θ = 0 along the whole trajectory, so with the rescue enabled every line
    # search must receive the KKT-progress certifier.
    seen = _record_kkt_progress_kwargs(monkeypatch)
    problem = HS35(namespace)
    result = solve(
        problem,
        array(namespace, [0.5, 0.5, 0.5]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            line_search=LineSearchOptions(feasible_kkt_progress=0.1),
        ),
    )

    assert result.status is Status.OPTIMAL
    assert len(seen) > 0
    assert all(fn is not None for fn in seen)


def test_certifier_disabled_by_default(namespace, monkeypatch):
    # OPT-IN since the S2MPJ v14 sweep: as a default the rescue walked
    # unconstrained/bounds-only runs (θ ≡ 0 — its whole domain) into worse
    # stationary points (48 corpus flips attributed). The default must not
    # wire the certifier at all.
    assert LineSearchOptions().feasible_kkt_progress is None
    seen = _record_kkt_progress_kwargs(monkeypatch)
    problem = HS35(namespace)
    result = solve(
        problem,
        array(namespace, [0.5, 0.5, 0.5]),
        options=Options(hessian="exact", linsolve="dense"),
    )

    assert result.status is Status.OPTIMAL
    assert len(seen) > 0
    assert all(fn is None for fn in seen)


def _certifier_verdicts(namespace, monkeypatch, gamma, alpha):
    """Verdicts of the driver's real certifier at ``alpha``, over a full solve."""
    verdicts: list[bool] = []
    original = FilterLineSearch.search

    def recording(self, **kwargs):
        certify = kwargs.get("kkt_progress")
        if certify is not None:
            verdicts.append(certify(alpha))
        return original(self, **kwargs)

    monkeypatch.setattr(FilterLineSearch, "search", recording)
    result = solve(
        HS35(namespace),
        array(namespace, [0.5, 0.5, 0.5]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            line_search=LineSearchOptions(feasible_kkt_progress=gamma),
        ),
    )
    assert result.status is Status.OPTIMAL
    assert verdicts, "the certifier was never handed to the line search"
    return verdicts


def test_certifier_computes_a_real_kkt_decrease_verdict(namespace, monkeypatch):
    # The certifier is a closure over live solver state: it re-evaluates the
    # gradient/Jacobian at the trial point and rescores the scaled KKT error.
    # Exercise the arithmetic against a real solve rather than a stub — on this
    # convex QP the Newton step genuinely drives the KKT error down, so a 10%
    # decrease must be certified at least once.
    verdicts = _certifier_verdicts(namespace, monkeypatch, gamma=0.1, alpha=1.0)

    assert all(isinstance(v, bool) for v in verdicts)
    assert any(verdicts)


def test_certifier_threshold_is_honoured(namespace, monkeypatch):
    # The verdict is the option's decrease fraction applied to a real error
    # ratio: demanding a 99.9999% drop per step is unattainable here, so the
    # certificate must refuse every trial — proving the threshold is actually
    # consulted rather than the rescue rubber-stamping the step.
    verdicts = _certifier_verdicts(namespace, monkeypatch, gamma=0.999999, alpha=1.0)

    assert not any(verdicts)


def test_certifier_advances_duals_even_for_a_null_primal_step(namespace, monkeypatch):
    # Design note worth pinning: the certificate scores the state the driver
    # would *adopt* — the trial primal plus the iteration's fraction-to-boundary
    # dual step — so the duals move even at α = 0. A null primal step is
    # therefore not automatically "no progress"; the dual step alone can reduce
    # the KKT error. (The line search only ever consults it at α > 0; this pins
    # the closure's semantics against an accidental "use α for the duals" edit.)
    verdicts = _certifier_verdicts(namespace, monkeypatch, gamma=0.1, alpha=0.0)

    assert all(isinstance(v, bool) for v in verdicts)
    assert any(verdicts)


def test_certifier_absent_at_infeasible_iterates(namespace, monkeypatch):
    # From an infeasible start (g = +3) the early iterations have θ0 > 0: the
    # rescue is a θ0 = 0 mechanism only, so even with the option enabled those
    # searches get no certifier.
    seen = _record_kkt_progress_kwargs(monkeypatch)
    problem = HS35(namespace)
    result = solve(
        problem,
        array(namespace, [2.0, 2.0, 2.0]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            line_search=LineSearchOptions(feasible_kkt_progress=0.1),
        ),
    )

    assert result.status is Status.OPTIMAL
    assert len(seen) > 0
    assert seen[0] is None  # first iteration is infeasible
