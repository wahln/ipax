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
