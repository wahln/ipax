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

"""Measured per-problem routing hints — the curated "known win" registry.

Several ipax opt-ins are corpus-neutral as defaults but decisive on a
*signature*: an adaptive μ oracle where the monotone schedule leaves the
central path, a scale-aware slack floor on badly-scaled constraints, the
L-BFGS update/seed levers on misaligned curvature. This registry records the
problems where such a lever has a **measured** win over the default
configuration, with the metrics and provenance, so that

- the sweep report can print the known win next to a default-configuration
  row that missed (``format_markdown``'s *Routing hints* section — omitted
  entirely when nothing applies), and
- the docs routing-hints page has a single source of truth.

Every entry must be measured, not guessed: record the run's budget alongside
the outcome, and prune entries a default change makes redundant (the sweep
report only prints a hint next to a row the default *missed*, so a stale
entry is invisible rather than wrong — but prune it anyway).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HINTS", "RoutingHint", "hints_for"]


@dataclass(frozen=True)
class RoutingHint:
    """One measured lever for one problem.

    ``options`` is the recipe as the user would type it; ``win`` the measured
    outcome including metrics and the default-configuration result it beats;
    ``signature`` the failure pattern this lever addresses (what to look for
    on problems *outside* this registry).
    """

    options: str
    win: str
    signature: str


# Keyed by the bare problem name (report rows use "s2mpj/<name>").
# Budgets: entries below were measured at max_iter=1000 / max_time=60 s
# (the IPOPT-triage budget) on lbfgs/dense unless stated otherwise.
HINTS: dict[str, tuple[RoutingHint, ...]] = {
    "AGG": (
        RoutingHint(
            options='mu_schedule="quality"',
            win="optimal in 42 iterations at the true optimum -3.5992e7 "
            "(default: restoration_failed at violation 19.9, obj -3.111e7)",
            signature="stall or restoration failure with a large constraint "
            "violation on an LP-like problem — the monotone μ schedule left "
            "the central path",
        ),
    ),
    "CRESC50": (
        RoutingHint(
            options='mu_schedule="quality"',
            win="objective 0.598 at max_iter — past the IPOPT reference 0.786 "
            "but uncertified (default: stalled at 15.3)",
            signature="stall far above a reference objective on a nonconvex "
            "packing/geometry problem",
        ),
    ),
    "GASOIL": (
        RoutingHint(
            options='LBFGSOptions(seed_formula="scalar1")',
            win="optimal in 25 iterations at 5.289e-3, IPOPT parity, dense "
            "and sparse routes (default: stalled after 508 iterations at 2.14)",
            signature="an L-BFGS run creeping at a KKT plateau while the "
            "exact-Hessian routes solve the problem — the direct ξ seed "
            "inflated by δ–γ misalignment",
        ),
    ),
    "HS59": (
        RoutingHint(
            options="BarrierOptions(slack_init_scale=0.1)",
            win="optimal in 18 iterations at the documented -7.8028 "
            "(default: converges to the worse local minimum -6.7495)",
            signature="clean convergence to a worse basin on a problem whose "
            "constraints are badly scaled at x0",
        ),
    ),
    "HS98": (
        RoutingHint(
            options="BarrierOptions(slack_init_scale=0.1)",
            win="optimal in 32 iterations at the documented 3.1358 "
            "(default: stalled after 668 iterations at 5.129)",
            signature="clean convergence to a worse basin on a problem whose "
            "constraints are badly scaled at x0",
        ),
    ),
    "ORTHRGDS": (
        RoutingHint(
            options="LBFGSOptions(damping_skip_ratio=1.0)",
            win="optimal in 19 iterations at IPOPT's 6.2102 "
            "(default: max_iter at 22.89)",
            signature="an L-BFGS run grinding at a worse objective on a "
            "nonconvex problem — Powell damping fabricating curvature from "
            "strongly-contradicted pairs",
        ),
    ),
    "PALMER1E": (
        RoutingHint(
            options='mu_schedule="quality"',
            win="optimal in 733 iterations at the documented 8.353e-4 "
            "(default: acceptable at the worse basin 0.1135)",
            signature="certified convergence to a visibly worse least-squares "
            "objective than a reference",
        ),
    ),
    "SINROSNB": (
        RoutingHint(
            options="BarrierOptions(slack_init_scale=0.1)",
            win="acceptable at 1.07e-5 (documented optimum 0; default: "
            "max_iter at 1.419)",
            signature="clean convergence to a worse basin on a problem whose "
            "constraints are badly scaled at x0",
        ),
    ),
}


def hints_for(problem: str) -> tuple[RoutingHint, ...]:
    """Hints for a report row's problem name (``"s2mpj/AGG"`` or ``"AGG"``)."""
    return HINTS.get(problem.rsplit("/", 1)[-1], ())
