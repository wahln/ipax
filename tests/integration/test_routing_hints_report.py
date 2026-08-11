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

"""The sweep report's *Routing hints* section.

For a problem the curated registry (``benchmarks.routing_hints``) records a
measured win for, the Markdown report prints that win next to the
default-configuration row — but only when the row actually *missed* (a
correct row needs no hint). When no hinted problem missed, the section is
omitted entirely.
"""

from __future__ import annotations

from dataclasses import replace

from benchmarks.harness import CaseResult, format_markdown
from benchmarks.routing_hints import HINTS, hints_for


def _row(problem: str, *, correct: bool, status: str = "stalled") -> CaseResult:
    return CaseResult(
        problem=problem,
        kind="NLP",
        backend="numpy",
        config="lbfgs/dense",
        status=status,
        success=correct,
        correct=correct,
        converged=correct,
        n_iter=100,
        kkt_error=1e-4,
        dual_infeasibility=1e-4,
        primal_infeasibility=0.0,
        complementarity=0.0,
        constraint_violation=0.0,
        error_vs_optimum=None,
        objective=1.0,
        expected_objective=None,
        expected_infeasible=False,
        pbclass=None,
        solve_time=1.0,
        linear_solver="dense",
        gradient_source="analytic",
        hessian_source="lbfgs",
        error=None,
    )


def test_registry_lookup_strips_the_corpus_prefix():
    assert hints_for("s2mpj/AGG") == HINTS["AGG"]
    assert hints_for("AGG") == HINTS["AGG"]
    assert hints_for("s2mpj/NOSUCHPROBLEM") == ()


def test_a_missed_hinted_problem_prints_its_win():
    rows = [_row("s2mpj/AGG", correct=False)]

    markdown = format_markdown(rows, {})

    assert "## Routing hints" in markdown
    assert 'mu_schedule="quality"' in markdown
    assert "optimal in 42 iterations" in markdown  # the measured metrics
    assert "stalled" in markdown  # next to this run's default result


def test_a_correct_hinted_problem_stays_silent():
    rows = [_row("s2mpj/AGG", correct=True, status="optimal")]

    markdown = format_markdown(rows, {})

    assert "## Routing hints" not in markdown


def test_a_missed_unhinted_problem_stays_silent():
    rows = [_row("s2mpj/NOSUCHPROBLEM", correct=False)]

    markdown = format_markdown(rows, {})

    assert "## Routing hints" not in markdown


def test_one_hint_row_per_problem_config_and_lever():
    # Two configs miss the same hinted problem: each config gets its own row
    # so the reader sees the default metrics the hint is beating.
    rows = [
        _row("s2mpj/HS98", correct=False),
        replace(_row("s2mpj/HS98", correct=False), config="lbfgs/sparse"),
    ]

    markdown = format_markdown(rows, {})

    section = markdown.split("## Routing hints", 1)[1]
    assert section.count("slack_init_scale=0.1") == 2


def test_a_lever_that_cannot_act_in_this_config_is_not_offered():
    # GASOIL's win is an LBFGSOptions seed; under hessian="exact" that setting
    # is inert, so recommending it next to an exact/* row would advertise a
    # no-op AND quote a win measured on a different route.
    lbfgs_row = _row("s2mpj/GASOIL", correct=False)
    exact_row = replace(lbfgs_row, config="exact/dense", hessian_source="exact")

    lbfgs_md = format_markdown([lbfgs_row], {})
    exact_md = format_markdown([exact_row], {})

    assert "seed_formula" in lbfgs_md
    assert "## Routing hints" not in exact_md


def test_a_hessian_agnostic_lever_is_offered_to_every_config():
    # mu_schedule is an Options-level lever: it acts under either Hessian mode,
    # so restricting the LBFGS-only hints must not suppress these too.
    lbfgs_row = _row("s2mpj/AGG", correct=False)
    exact_row = replace(lbfgs_row, config="exact/sparse", hessian_source="exact")

    for md in (format_markdown([lbfgs_row], {}), format_markdown([exact_row], {})):
        assert "## Routing hints" in md
        assert "mu_schedule" in md


def test_every_lbfgs_only_hint_declares_its_hessian():
    # A new LBFGSOptions entry added without `hessian="lbfgs"` would silently
    # be offered to exact rows again; pin the invariant at the registry.
    for name, hints in HINTS.items():
        for hint in hints:
            if "LBFGSOptions" in hint.options:
                assert hint.hessian == "lbfgs", (
                    f"{name}: an LBFGSOptions lever is inert under exact "
                    "Hessians and must declare hessian='lbfgs'"
                )
