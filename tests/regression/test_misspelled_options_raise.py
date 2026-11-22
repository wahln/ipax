"""Regression: a misspelled option value raises instead of changing the solve.

Through 0.13 the top-level selectors were never validated: a typo in
``hessian``, ``linsolve`` or ``globalization`` ran as the default, and a
misspelled ``mu_schedule`` failed mid-solve with a bare ``AssertionError``
(none under ``python -O``). ``BarrierOptions(fallback="kkt_error")``
(underscore) silently switched off the Nocedal–Wächter–Waltz (2009) §5.1
safeguard, and ``free_mode_acceptance="obj_constr_filter"`` ran
``"rigorous"``. All raise :class:`ipax.ConfigurationError` from 0.14.
"""

from __future__ import annotations

import pytest

import ipax


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hessian": "bogus"},
        {"linsolve": "krylv"},
        {"globalization": "bogus"},
        {"mu_schedule": "bogus"},
        {"scaling": "gradient_based"},
        {"corrections": "Mehrotra"},
    ],
    ids=lambda kw: next(iter(kw)),
)
def test_misspelled_top_level_selectors_are_rejected(kwargs):
    with pytest.raises(ipax.ConfigurationError):
        ipax.Options(**kwargs)


def test_underscore_spelling_of_the_barrier_fallback_is_rejected_not_ignored():
    with pytest.raises(ipax.ConfigurationError, match="'kkt-error'"):
        ipax.BarrierOptions(fallback="kkt_error")
    with pytest.raises(ipax.ConfigurationError, match="'obj-constr-filter'"):
        ipax.LineSearchOptions(free_mode_acceptance="obj_constr_filter")
