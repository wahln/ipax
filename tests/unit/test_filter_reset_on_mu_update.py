"""The W&B filter is re-initialized whenever the barrier parameter changes.

The filter's entries are ``(θ, φ_μ)`` pairs; the φ coordinate is specific to
the μ it was recorded at. Wächter & Biegler's implementation re-initializes
the filter at every barrier-parameter update (IPOPT: ``linesearch_->Reset()``
→ ``FilterLSAcceptor::Reset()`` → ``filter_.Clear()`` in
``IpMonotoneMuUpdate``; NWW 2005/2009 §5 for free-mode oracles: "the history
in the filter [is] reset at every free iteration because the barrier problem
itself changes"). Before the fix, ipax created one ``Filter()`` per solve and
never cleared it, so stale old-μ φ entries were compared against new-μ trial
values.
"""

from __future__ import annotations

from ipax import Options, Status, solve
from ipax.ipm.filter_ls import FilterLineSearch
from ipax.testing.problems import HS7
from tests._helpers import array


def test_filter_is_reset_when_mu_decreases(namespace, monkeypatch):
    # HS7's nonlinear equality keeps early iterates infeasible, so accepted
    # θ-type steps augment the filter while μ steps down several times on the
    # way to optimality. Entries can only vanish again through a reset —
    # ``augment`` always appends — so an empty filter observed *after* it held
    # entries is exactly the μ-update re-initialization.
    entry_counts: list[int] = []
    original = FilterLineSearch.search

    def recording(self, **kwargs):
        entry_counts.append(len(kwargs["entries"]))
        return original(self, **kwargs)

    monkeypatch.setattr(FilterLineSearch, "search", recording)

    problem = HS7(namespace)
    result = solve(
        problem,
        array(namespace, [2.0, 2.0]),
        options=Options(hessian="exact", linsolve="dense"),
    )

    assert result.status is Status.OPTIMAL
    # Premises: the run actually changed μ and actually augmented the filter.
    assert len({record.mu for record in result.history}) > 1
    first_augmented = next(
        (i for i, count in enumerate(entry_counts) if count > 0), None
    )
    assert first_augmented is not None
    # The re-initialization: some later search starts from an empty filter.
    assert any(count == 0 for count in entry_counts[first_augmented + 1 :])
