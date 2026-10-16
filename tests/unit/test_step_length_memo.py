"""Unit tests for the corrector pass's fraction-to-boundary memo.

``ipax.ipm.driver._memoized_step_length`` caches a step-length rule per
``(direction object, τ)`` for one corrector pass, keeping only the two most
recent entries; an identity check guards against a recycled ``id``.
"""

from __future__ import annotations

from ipax.ipm import driver as driver_mod
from ipax.ipm.driver import _STEP_LENGTH_MEMO_SIZE, _memoized_step_length


class _Direction:
    """Stand-in for a ``NewtonStep``: the memo only uses its identity."""


def _counting_rule():
    calls: list[tuple[object, float]] = []

    def rule(step, *, tau):
        calls.append((step, tau))
        return 0.5

    return rule, calls


def test_memo_hits_repeat_and_keys_on_tau():
    rule, calls = _counting_rule()
    memo = _memoized_step_length(rule)
    a = _Direction()
    assert memo(a, tau=1.0) == 0.5
    assert memo(a, tau=1.0) == 0.5
    assert len(calls) == 1
    memo(a, tau=0.99)  # a different τ is a different key
    assert len(calls) == 2


def test_memo_evicts_the_oldest_entry_beyond_its_size():
    assert _STEP_LENGTH_MEMO_SIZE == 2
    rule, calls = _counting_rule()
    memo = _memoized_step_length(rule)
    a, b, c = _Direction(), _Direction(), _Direction()
    memo(a, tau=1.0)
    memo(b, tau=1.0)
    memo(c, tau=1.0)  # evicts a (oldest insertion)
    assert len(calls) == 3
    memo(b, tau=1.0)
    memo(c, tau=1.0)
    assert len(calls) == 3  # b and c still cached
    memo(a, tau=1.0)
    assert len(calls) == 4  # a was evicted: recomputed
    assert calls[-1][0] is a


def test_memo_misses_on_a_recycled_id(monkeypatch):
    # Force every direction to share one ``id`` (as a collected step's id can
    # be recycled for a new one): the cached entry belongs to a different
    # object, so the identity check must turn the key hit into a miss.
    monkeypatch.setattr(driver_mod, "id", lambda obj: 42, raising=False)
    rule, calls = _counting_rule()
    memo = _memoized_step_length(rule)
    first, second = _Direction(), _Direction()
    memo(first, tau=1.0)
    memo(second, tau=1.0)
    assert [step for step, _ in calls] == [first, second]
    memo(second, tau=1.0)  # the entry now holds ``second``: a genuine hit
    assert len(calls) == 2
