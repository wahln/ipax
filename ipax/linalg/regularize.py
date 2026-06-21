# Copyright 2026 Niklas Wahl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Primal-dual regularization helpers for the condensed route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipax.options import RegularizationOptions


@dataclass(slots=True)
class RegularizationState:
    """Carry regularization parameters across solver iterations."""

    delta_w: float = 0.0
    delta_c: float = 0.0
    last_delta_w: float = 0.0


def escalate_delta_w(
    state: RegularizationState,
    options: RegularizationOptions,
) -> float:
    """Bump ``delta_w`` after a failed dense factorization."""
    if state.delta_w <= 0.0:
        delta_w = options.delta_w_init
    else:
        delta_w = state.delta_w * options.delta_w_factor
    delta_w = min(delta_w, options.delta_w_max)
    state.delta_w = delta_w
    state.delta_c = options.delta_c
    state.last_delta_w = delta_w
    return delta_w


def shrink_delta_w(
    state: RegularizationState,
    options: RegularizationOptions,
) -> float:
    """Reduce ``delta_w`` after a successful factorization."""
    if state.delta_w <= 0.0:
        delta_w = 0.0
    else:
        delta_w = state.delta_w / options.delta_w_factor
        if delta_w <= options.delta_w_init:
            delta_w = 0.0
    state.delta_w = delta_w
    state.delta_c = options.delta_c
    state.last_delta_w = delta_w
    return delta_w


__all__ = ["RegularizationState", "escalate_delta_w", "shrink_delta_w"]
