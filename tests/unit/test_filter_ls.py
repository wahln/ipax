"""Unit tests for filter accept/reject behavior."""

from __future__ import annotations

from ipax.ipm.filter_ls import Filter, FilterLineSearch
from ipax.options import LineSearchOptions
from tests._helpers import implemented


def test_empty_filter_accepts_candidate():
    filt = Filter()
    with implemented("filter"):
        assert filt.is_acceptable(theta=1.0, phi=10.0)


def test_filter_rejects_candidate_dominated_in_both_measures():
    filt = Filter(entries=[(1.0, 10.0)])
    with implemented("filter"):
        assert not filt.is_acceptable(theta=1.1, phi=10.5)


def test_filter_accepts_candidate_that_improves_one_measure():
    filt = Filter(entries=[(1.0, 10.0)])
    with implemented("filter"):
        assert filt.is_acceptable(theta=0.5, phi=11.0)
        assert filt.is_acceptable(theta=1.1, phi=9.0)


def test_filter_augment_removes_entries_dominated_by_new_pair():
    filt = Filter(entries=[(1.0, 10.0), (0.25, 20.0)])
    with implemented("filter"):
        filt.augment(theta=0.5, phi=8.0)

    assert (1.0, 10.0) not in filt.entries
    assert (0.25, 20.0) in filt.entries
    assert (0.5, 8.0) in filt.entries


def test_switching_condition_survives_overflowing_directional_derivative():
    # Regression (INDEF): a badly-scaled iterate yields an enormous |dphi|; the
    # switching condition's ``(-dphi) ** s_phi`` must not raise OverflowError.
    line_search = FilterLineSearch(LineSearchOptions())

    result = line_search.search(
        alpha_max=1.0,
        theta0=1.0,
        phi0=1.0,
        dphi=-1e308,  # would overflow float ** s_phi before the fix
        eval_point=lambda alpha: (0.5, 0.5),
        entries=[],
        soc=None,
    )

    # No crash; the f-type Armijo test cannot be met for such a step, so the
    # search exhausts α and hands off to restoration.
    assert result.restoration
    assert not result.accepted


def test_line_search_reports_accepted_soc_trial():
    line_search = FilterLineSearch(LineSearchOptions())

    result = line_search.search(
        alpha_max=1.0,
        theta0=1.0,
        phi0=1.0,
        dphi=1.0,
        eval_point=lambda alpha: (2.0, 2.0),
        entries=[],
        soc=lambda alpha: (0.1, 2.0),
    )

    assert result.accepted
    assert result.used_soc
    assert not result.restoration
