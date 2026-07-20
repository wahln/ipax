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

"""A/B two sweep reports — the pre-release gate's diff tool.

The full-corpus sweep is the mandatory gate for any default-behaviour change,
and the question it answers is always "what moved, and is it attributable?".
This compares two ``--out`` reports keyed by ``(config, problem)``.

Beyond the correctness delta it reports two things learned the hard way:

**Objective drift.** ``correct`` is scored against the dataset's documented
optimum, and many problems do not have one — for those the metric credits any
certified convergence, so a change that converges to a *different, much worse*
local optimum scores unchanged. That is not hypothetical: the sparse
normal-equations default moved ``OET7`` from ``4.45e-05`` to ``0.0872`` (~2000x
worse) while the correct-count showed nothing. Drift on such problems is
reported separately and flagged ``unscored``, because the headline ± number
structurally cannot see it.

**Route changes.** Which problems changed ``linear_solver``. A config with zero
route changes is a built-in control when A/B-ing linear-algebra work: anything
that moves there is unrelated to the change (timing noise, nondeterminism).

Usage::

    python -m benchmarks.runners.compare base.json new.json
    python -m benchmarks.runners.compare base.json new.json --config lbfgs/sparse
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# Relative objective change below which two runs are "the same answer": the
# routes differ in rounding, not in the optimum they find.
_OBJECTIVE_TOL = 1e-6


@dataclass(frozen=True, slots=True)
class Flip:
    """A problem whose ``correct`` verdict changed between the two reports."""

    config: str
    problem: str
    base_status: str
    new_status: str


@dataclass(frozen=True, slots=True)
class Drift:
    """A problem whose *objective* moved materially between the two reports.

    ``unscored`` marks the dangerous case: no dataset reference objective, so
    the ``correct`` count cannot see this drift at all.
    """

    config: str
    problem: str
    base_objective: float
    new_objective: float
    relative_change: float
    worse: bool  # minimization: a higher objective is worse
    unscored: bool


@dataclass(frozen=True, slots=True)
class Comparison:
    """Everything the release gate needs from an A/B of two sweeps."""

    base_path: str
    new_path: str
    base_correct: int
    new_correct: int
    common: int
    base_only: int
    new_only: int
    fixed: list[Flip] = field(default_factory=list)
    broken: list[Flip] = field(default_factory=list)
    objective_drift: list[Drift] = field(default_factory=list)
    per_config: dict[str, tuple[int, int]] = field(
        default_factory=dict
    )  # (fixed, broken)
    route_changes: dict[str, int] = field(default_factory=dict)

    @property
    def delta(self) -> int:
        """Net change in the corpus correct-count over the common rows."""
        return self.new_correct - self.base_correct


def _load(path: str) -> dict[tuple[str, str], dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return {(r["config"], r["problem"]): r for r in payload["results"]}


def _relative_change(base: float, new: float) -> float:
    """Signed relative change, guarded for a (near-)zero baseline."""
    if not (math.isfinite(base) and math.isfinite(new)):
        return math.inf
    return (new - base) / max(abs(base), 1e-12)


def compare_reports(base_path: str, new_path: str) -> Comparison:
    """A/B two sweep reports keyed by ``(config, problem)``."""
    base, new = _load(base_path), _load(new_path)
    common = sorted(base.keys() & new.keys())

    fixed: list[Flip] = []
    broken: list[Flip] = []
    drift: list[Drift] = []
    per_config: dict[str, list[int]] = {}
    routes: Counter[str] = Counter()

    for key in common:
        cfg, problem = key
        b, n = base[key], new[key]
        per_config.setdefault(cfg, [0, 0])
        routes.setdefault(cfg, 0)

        if b.get("linear_solver") != n.get("linear_solver"):
            routes[cfg] += 1

        if n["correct"] and not b["correct"]:
            fixed.append(Flip(cfg, problem, b["status"], n["status"]))
            per_config[cfg][0] += 1
        elif b["correct"] and not n["correct"]:
            broken.append(Flip(cfg, problem, b["status"], n["status"]))
            per_config[cfg][1] += 1

        rel = _relative_change(b["objective"], n["objective"])
        if abs(rel) > _OBJECTIVE_TOL:
            drift.append(
                Drift(
                    config=cfg,
                    problem=problem,
                    base_objective=b["objective"],
                    new_objective=n["objective"],
                    relative_change=rel,
                    worse=rel > 0.0,
                    unscored=b.get("expected_objective") is None,
                )
            )

    return Comparison(
        base_path=base_path,
        new_path=new_path,
        base_correct=sum(1 for k in common if base[k]["correct"]),
        new_correct=sum(1 for k in common if new[k]["correct"]),
        common=len(common),
        base_only=len(base.keys() - new.keys()),
        new_only=len(new.keys() - base.keys()),
        fixed=fixed,
        broken=broken,
        objective_drift=drift,
        per_config={c: (v[0], v[1]) for c, v in sorted(per_config.items())},
        route_changes=dict(routes),
    )


def format_comparison(cmp: Comparison, *, config: str | None = None) -> str:
    """Render a comparison as the text block the sweep write-ups quote."""
    out: list[str] = []
    out.append(f"base = {cmp.base_path}")
    out.append(f"new  = {cmp.new_path}")
    out.append(
        f"common rows: {cmp.common}  (base-only {cmp.base_only}, new-only {cmp.new_only})"
    )
    out.append("")
    out.append(
        f"CORRECT  base={cmp.base_correct}  new={cmp.new_correct}  delta={cmp.delta:+d}"
    )
    out.append(f"  fixed (incorrect -> correct): {len(cmp.fixed)}")
    out.append(f"  broken (correct -> incorrect): {len(cmp.broken)}")
    out.append("")

    out.append(
        f"{'config':16s} {'+fixed':>7s} {'-broken':>8s} {'net':>6s} {'routes':>7s}"
    )
    for cfg, (f, b) in cmp.per_config.items():
        if config and cfg != config:
            continue
        out.append(
            f"{cfg:16s} {f:7d} {b:8d} {f - b:+6d} {cmp.route_changes.get(cfg, 0):7d}"
        )

    for title, flips in (("BROKEN", cmp.broken), ("FIXED", cmp.fixed)):
        rows = [f for f in flips if not config or f.config == config]
        if not rows:
            continue
        out.append("")
        out.append(f"--- {title} (status base -> new) ---")
        for f in sorted(rows, key=lambda x: (x.config, x.problem)):
            out.append(
                f"  {f.config:14s} {f.problem:28s} {f.base_status:18s} -> {f.new_status}"
            )

    rows = [d for d in cmp.objective_drift if not config or d.config == config]
    if rows:
        blind = [d for d in rows if d.unscored]
        out.append("")
        out.append(
            f"--- OBJECTIVE drift ({len(rows)}; {len(blind)} unscored, i.e. invisible "
            f"to the correct-count) ---"
        )
        for d in sorted(rows, key=lambda x: (not x.unscored, x.config, x.problem)):
            tag = "WORSE" if d.worse else "better"
            mark = "  [unscored]" if d.unscored else ""
            out.append(
                f"  {d.config:14s} {d.problem:28s} {d.base_objective:14.6g} -> "
                f"{d.new_objective:14.6g}  {tag:6s} ({d.relative_change:+.2e}){mark}"
            )
        if blind:
            out.append("")
            out.append(
                "  NOTE: unscored rows have no dataset reference objective, so the "
                "correct-count above cannot see these; review them before "
                "accepting the delta."
            )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A/B two benchmark sweep reports (the pre-release gate diff)."
    )
    parser.add_argument("base", help="baseline report .json")
    parser.add_argument("new", help="candidate report .json")
    parser.add_argument(
        "--config", default=None, help="restrict the detail sections to one config"
    )
    args = parser.parse_args(argv)

    print(format_comparison(compare_reports(args.base, args.new), config=args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
