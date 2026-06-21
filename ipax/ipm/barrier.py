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

"""Barrier parameter schedule and fraction-to-boundary helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipax.options import BarrierOptions
    from ipax.typing import Array


def update_mu(mu: float, options: BarrierOptions, tol: float) -> float:
    """Monotone Fiacco-McCormick barrier reduction."""
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    return float(
        max(options.mu_min, tol / 10.0, options.kappa_mu * (mu**options.theta_mu))
    )


def fraction_to_boundary(v: Array, dv: Array, tau: float) -> float:
    """Largest ``alpha`` in ``[0, 1]`` preserving ``v + alpha * dv >= (1 - tau) * v``."""
    if len(v.shape) != 1 or len(dv.shape) != 1:
        raise ValueError("fraction-to-boundary expects rank-1 vectors")
    if v.shape != dv.shape:
        raise ValueError("v and dv must have the same shape")
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")

    alpha = 1.0
    for idx in range(int(v.shape[0])):
        step = float(dv[idx])
        if step < 0.0:
            alpha = min(alpha, tau * float(v[idx]) / (-step))
    return max(0.0, min(1.0, alpha))


__all__ = ["fraction_to_boundary", "update_mu"]
