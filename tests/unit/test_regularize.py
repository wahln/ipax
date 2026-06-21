"""Unit tests for primal-dual regularization state updates."""

from __future__ import annotations

from ipax.linalg.regularize import RegularizationState, escalate_delta_w, shrink_delta_w
from ipax.options import RegularizationOptions
from tests._helpers import implemented


def test_escalate_delta_w_starts_at_configured_initial_value():
    state = RegularizationState()
    options = RegularizationOptions(delta_w_init=1e-6, delta_w_factor=8.0)

    with implemented("regularization loop"):
        delta = escalate_delta_w(state, options)

    assert delta == options.delta_w_init
    assert state.delta_w == delta
    assert state.last_delta_w == delta


def test_escalate_delta_w_multiplies_existing_value():
    state = RegularizationState(delta_w=1e-6)
    options = RegularizationOptions(delta_w_init=1e-6, delta_w_factor=8.0)

    with implemented("regularization loop"):
        delta = escalate_delta_w(state, options)

    assert delta == 8e-6
    assert state.delta_w == delta


def test_shrink_delta_w_reduces_without_going_negative():
    state = RegularizationState(delta_w=1e-4, last_delta_w=1e-4)
    options = RegularizationOptions(delta_w_init=1e-6, delta_w_factor=8.0)

    with implemented("regularization loop"):
        delta = shrink_delta_w(state, options)

    assert 0.0 <= delta < 1e-4
    assert state.delta_w == delta
