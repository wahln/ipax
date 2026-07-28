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

"""The full-corpus driver behind the ipax-vs-IPOPT comparison.

The comparison *semantics* (Jacobian translation, ``BaselineUnsupported``,
verdict categories) are covered by ``test_baseline_ipyopt.py``, which needs the
``ipyopt`` binding. This module covers the **driver** — problem selection,
per-problem skip/error isolation, resume, and the report — with the reference
solver stubbed out, so it runs in the ordinary per-PR suite without either
``ipyopt`` or an S2MPJ checkout.

The isolation tests are the load-bearing ones: a full-corpus run is unattended
over ~1100 problems, so a build that raises, a problem too large for the route,
or a reference that blows up must each become a recorded row or a counted skip —
never an exception that ends the sweep.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

import ipax
from benchmarks.baselines import BaselineUnsupported, ReferenceResult
from benchmarks.runners import s2mpj_baselines as runner
from ipax.testing.problems import UnconstrainedQuadratic

# Minimizer of 0.5 xᵀx − bᵀx is x* = b, with f* = −0.5·bᵀb.
_B = (1.0, 2.0, 3.0)
_OPTIMUM = -0.5 * sum(v * v for v in _B)


class _StubCase:
    """Stands in for a :class:`BenchmarkProblem` from the S2MPJ corpus."""

    def __init__(self, *, n: int = 3, expected: float | None = _OPTIMUM, error=None):
        self.n = n
        self.expected = expected
        self.error = error

    def build(self, xp):
        if self.error is not None:
            raise self.error
        b = np.asarray(_B[: self.n] + (0.0,) * (self.n - len(_B)), dtype=float)
        problem = UnconstrainedQuadratic(np.eye(self.n), b, np)
        problem.expected_objective = self.expected  # type: ignore[attr-defined]
        return problem, np.zeros(self.n)


class _StubBaseline:
    """A reference solver whose outcome the test dictates."""

    name = "stub"
    result: ReferenceResult | None = None
    error: Exception | None = None

    def __init__(self, **_kwargs) -> None:
        pass

    def solve(self, problem, x0) -> ReferenceResult:
        if _StubBaseline.error is not None:
            raise _StubBaseline.error
        assert _StubBaseline.result is not None
        return _StubBaseline.result


class _StalledResult:
    """An ipax result that reached the right objective without converging."""

    def __init__(self, objective: float) -> None:
        self.objective = objective
        self.x = np.zeros(3)
        self.n_iter = 999
        self.status = type("S", (), {"value": "stalled"})()


def _reference(objective: float, *, success: bool = True) -> ReferenceResult:
    return ReferenceResult(
        name="stub",
        x=np.zeros(3),
        objective=objective,
        success=success,
        n_iter=7,
        solve_time=0.01,
    )


@pytest.fixture
def stubbed(monkeypatch):
    """Route the runner at a stub corpus and a stub reference solver."""
    _StubBaseline.result = _reference(_OPTIMUM)
    _StubBaseline.error = None
    monkeypatch.setattr(runner, "IpyoptBaseline", _StubBaseline)
    return _StubBaseline


def _options() -> ipax.Options:
    return ipax.Options(hessian="lbfgs", linsolve="dense", max_iter=200, max_time=30.0)


# -- problem selection --------------------------------------------------------


def _args(**over) -> argparse.Namespace:
    base: dict[str, object] = {
        "problems": [],
        "names": None,
        "names_file": None,
        "all": False,
        "limit": 0,
        "sample": 0,
    }
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture
def corpus(monkeypatch):
    """A deterministic stand-in for the S2MPJ checkout listing."""
    names = [f"P{i:02d}" for i in range(10)]
    monkeypatch.setattr(runner, "list_s2mpj_problems", lambda root=None: list(names))
    return names


def test_names_file_takes_precedence_over_every_other_selector(tmp_path, corpus):
    path = tmp_path / "names.txt"
    path.write_text("AGG  # the known gap\n\n# a comment\nOET7\n")
    args = _args(names_file=str(path), problems=["HS35"], all=True)
    assert runner._select_names(args, None) == ("AGG", "OET7")


def test_positional_names_and_the_names_flag_combine(corpus):
    args = _args(problems=["AGG"], names="OET7, HS35")
    assert runner._select_names(args, None) == ("AGG", "OET7", "HS35")


def test_all_sweeps_the_checkout_and_honours_limit(corpus):
    assert runner._select_names(_args(all=True), None) == tuple(corpus)
    assert runner._select_names(_args(all=True, limit=3), None) == tuple(corpus[:3])


def test_sample_takes_a_corpus_prefix(corpus):
    assert runner._select_names(_args(sample=2), None) == ("P00", "P01")


def test_no_selector_selects_nothing(corpus):
    assert runner._select_names(_args(), None) == ()


# -- per-problem isolation ----------------------------------------------------


def test_objective_free_problem_is_skipped_not_raised(stubbed, monkeypatch):
    case = _StubCase(error=NotImplementedError("no objective"))
    monkeypatch.setattr(runner, "_build_case", lambda root, name: case)
    row, skip = runner._compare_problem(None, "FEAS", _options(), 0, 3000, 60.0)
    assert row is None
    assert skip == "no_objective"


def test_oversized_problem_is_skipped_before_the_solve(stubbed, monkeypatch):
    monkeypatch.setattr(runner, "_build_case", lambda root, name: _StubCase(n=5))
    row, skip = runner._compare_problem(None, "BIG", _options(), 4, 3000, 60.0)
    assert row is None
    assert skip == "too_large"


def test_build_error_is_recorded_as_a_skip(stubbed, monkeypatch):
    case = _StubCase(error=RuntimeError("bad SIF translation"))
    monkeypatch.setattr(runner, "_build_case", lambda root, name: case)
    row, skip = runner._compare_problem(None, "BROKEN", _options(), 0, 3000, 60.0)
    assert row is None
    assert skip == "build_error"


def test_missing_problem_is_recorded_as_a_skip(stubbed, monkeypatch):
    def _missing(root, name):
        raise RuntimeError(f"{name}: not available in this S2MPJ checkout")

    monkeypatch.setattr(runner, "_build_case", _missing)
    row, skip = runner._compare_problem(None, "NOPE", _options(), 0, 3000, 60.0)
    assert row is None
    assert skip == "build_error"


def test_both_solvers_correct_is_an_agreement(stubbed, monkeypatch):
    monkeypatch.setattr(runner, "_build_case", lambda root, name: _StubCase())
    row, skip = runner._compare_problem(None, "QUAD", _options(), 0, 3000, 60.0)
    assert skip is None
    assert row is not None
    assert row.verdict == "agree"
    assert row.ipax_correct and row.ref_correct
    assert row.ref_iters == 7


def test_reference_solving_what_ipax_misses_is_an_ipax_gap(stubbed, monkeypatch):
    # Document an optimum neither ipax's start nor its route reaches, but which
    # the reference reports: exactly the AGG anatomy the triage exists to find.
    monkeypatch.setattr(
        runner, "_build_case", lambda root, name: _StubCase(expected=-99.0)
    )
    stubbed.result = _reference(-99.0)
    row, _ = runner._compare_problem(None, "GAP", _options(), 0, 3000, 60.0)
    assert row is not None
    assert row.verdict == "ipax-gap"


def test_a_failed_solver_is_not_credited_for_the_right_objective(stubbed, monkeypatch):
    # The accuracy sweep requires `result.success` before calling a case
    # correct; this runner must use the same criterion, or a stall that happens
    # to sit at the documented optimum scores as a success on both reports.
    monkeypatch.setattr(runner, "_build_case", lambda root, name: _StubCase())
    monkeypatch.setattr(
        runner.ipax,
        "solve",
        lambda *a, **k: _StalledResult(_OPTIMUM),
    )
    row, _ = runner._compare_problem(None, "QUAD", _options(), 0, 3000, 60.0)
    assert row is not None
    assert row.ipax_correct is False  # right objective, but it did not converge
    assert row.verdict == "ipax-gap"  # the reference did converge there


def test_two_failed_solvers_do_not_score_as_agreement(stubbed, monkeypatch):
    # Unscored problems fall back to comparing objectives; two solvers that
    # both *failed* at the same objective agreed about nothing.
    monkeypatch.setattr(
        runner, "_build_case", lambda root, name: _StubCase(expected=None)
    )
    monkeypatch.setattr(runner.ipax, "solve", lambda *a, **k: _StalledResult(_OPTIMUM))
    stubbed.result = _reference(_OPTIMUM, success=False)
    row, _ = runner._compare_problem(None, "QUAD", _options(), 0, 3000, 60.0)
    assert row is not None
    assert row.verdict == "both-hard*"


def test_reference_failure_does_not_lose_the_ipax_row(stubbed, monkeypatch):
    monkeypatch.setattr(runner, "_build_case", lambda root, name: _StubCase())
    stubbed.error = RuntimeError("ipyopt exploded")
    row, skip = runner._compare_problem(None, "QUAD", _options(), 0, 3000, 60.0)
    assert skip is None
    assert row is not None
    assert row.ref_error is not None and "ipyopt exploded" in row.ref_error
    assert row.ipax_correct is True  # ipax's own result survives the ref failure


def test_unsupported_reference_is_reported_as_unsupported(stubbed, monkeypatch):
    monkeypatch.setattr(runner, "_build_case", lambda root, name: _StubCase())
    stubbed.error = BaselineUnsupported("matrix-free Jacobian")
    row, _ = runner._compare_problem(None, "QUAD", _options(), 0, 3000, 60.0)
    assert row is not None
    assert row.ref_error is not None and row.ref_error.startswith("unsupported:")


# -- report + resume ----------------------------------------------------------


def _row(problem: str, verdict: str, **over) -> runner.Row:
    fields: dict[str, object] = {
        "problem": problem,
        "n_vars": 3,
        "expected_objective": -7.0,
        "ipax_status": "optimal",
        "ipax_iters": 11,
        "ipax_objective": -7.0,
        "ipax_correct": True,
        "ref_name": "ipyopt",
        "ref_success": True,
        "ref_iters": 9,
        "ref_objective": -7.0,
        "ref_correct": True,
        "ref_error": None,
        "verdict": verdict,
        "ipax_time": 0.5,
        "ref_time": 0.1,
    }
    fields.update(over)
    return runner.Row(**fields)  # type: ignore[arg-type]


def test_payload_round_trips_through_json(tmp_path):
    rows = [_row("AGG", "ipax-gap"), _row("HS35", "agree")]
    payload = runner.to_payload(rows, {"ipax": "0.8.0"}, "lbfgs/dense", {})
    restored = json.loads(json.dumps(payload))
    back = [runner._row_from_dict(r) for r in restored["rows"]]
    assert [r.problem for r in back] == ["AGG", "HS35"]
    assert back[0].verdict == "ipax-gap"


def test_row_from_dict_tolerates_an_older_schema(tmp_path):
    payload = {
        "problem": "AGG",
        "verdict": "ipax-gap",
        "a_field_from_a_future_schema": 1,
    }
    row = runner._row_from_dict(payload)
    assert row.problem == "AGG"
    assert row.verdict == "ipax-gap"
    assert row.ipax_status == ""  # defaulted, not an error


# -- feasibility of the compared points ---------------------------------------


class _ConstrainedStub:
    """`x0 + x1 = 1`, `x0 ≤ 2`, `0 ≤ x ≤ 3` — every constraint block at once."""

    n_vars = 2

    def bounds(self):
        return np.zeros(2), np.full(2, 3.0)

    def objective(self, x):
        return float(np.sum(x))

    def eq_constraints(self, x):
        return np.array([x[0] + x[1] - 1.0])

    def ineq_constraints(self, x):
        return np.array([x[0] - 2.0])

    def linear_eq(self):
        return None

    def linear_ineq(self):
        return None


def test_infeasibility_is_zero_at_a_feasible_point():
    assert runner._infeasibility(_ConstrainedStub(), np.array([0.5, 0.5])) == 0.0


def test_infeasibility_sees_each_constraint_block():
    p = _ConstrainedStub()
    # equality residual 1.0
    assert runner._infeasibility(p, np.array([1.0, 1.0])) == pytest.approx(1.0)
    # inequality x0 <= 2 violated by 1.0 (equality violated by 1.0 too)
    assert runner._infeasibility(p, np.array([3.0, -1.0])) == pytest.approx(1.0)
    # bound violation below zero
    assert runner._infeasibility(p, np.array([2.0, -1.0])) == pytest.approx(1.0)


def test_a_lower_objective_at_an_infeasible_point_is_recorded(stubbed, monkeypatch):
    # The interpretive trap this column exists to close: ipax reporting a
    # *lower* objective than the reference is only a win if its point is also
    # feasible, so both points' violations must be on the row.
    monkeypatch.setattr(runner, "_build_case", lambda root, name: _StubCase())
    stubbed.result = _reference(_OPTIMUM)
    row, _ = runner._compare_problem(None, "QUAD", _options(), 0, 3000, 60.0)
    assert row is not None
    assert row.ipax_infeasibility == pytest.approx(0.0)
    assert row.ref_infeasibility == pytest.approx(0.0)


# -- naming the crash culprit -------------------------------------------------


def test_worker_marks_the_problem_it_is_running(stubbed, monkeypatch, tmp_path):
    # A native crash kills the worker mid-solve, so the only way to learn which
    # problem did it is a marker the worker itself wrote. The parent's pending
    # -futures list cannot do this job: every unstarted problem is in it too.
    prefix = tmp_path / "run.inflight"
    seen: dict[str, str] = {}

    def _peek(root, name):
        marker = next(tmp_path.glob("run.inflight.*"))
        seen["path"] = marker.name
        seen["content"] = marker.read_text(encoding="utf-8")
        return _StubCase()

    monkeypatch.setattr(runner, "_build_case", _peek)
    runner._compare_problem(
        None, "CRASHER", _options(), 0, 3000, 60.0, inflight_prefix=str(prefix)
    )
    assert seen["content"] == "CRASHER"
    # Cleared on the way out, so only a hard crash leaves one behind.
    assert list(tmp_path.glob("run.inflight.*")) == []


def test_worker_marker_survives_a_skip(stubbed, monkeypatch, tmp_path):
    prefix = tmp_path / "run.inflight"
    case = _StubCase(error=RuntimeError("boom"))
    monkeypatch.setattr(runner, "_build_case", lambda root, name: case)
    _row, skip = runner._compare_problem(
        None, "BROKEN", _options(), 0, 3000, 60.0, inflight_prefix=str(prefix)
    )
    assert skip == "build_error"
    assert list(tmp_path.glob("run.inflight.*")) == []  # no marker leaked


# -- crash-safe report writing ------------------------------------------------


def test_atomic_write_replaces_the_previous_report(tmp_path):
    path = tmp_path / "report.json"
    runner._atomic_write(path, "first")
    runner._atomic_write(path, "second")
    assert path.read_text(encoding="utf-8") == "second"
    assert list(tmp_path.iterdir()) == [path]  # no temp file left behind


def test_atomic_write_retries_a_transient_sharing_violation(tmp_path, monkeypatch):
    # On Windows os.replace raises PermissionError (WinError 5) whenever the
    # target is momentarily open — an antivirus scan, an indexer, or simply
    # someone reading the report while the sweep flushes. It is transient, so
    # it must be retried, not propagated: this killed a 964-row run.
    path = tmp_path / "report.json"
    runner._atomic_write(path, "old")
    real = runner.Path.replace
    calls = {"n": 0}

    def _flaky(self, target):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(5, "Access is denied")
        return real(self, target)

    monkeypatch.setattr(runner.Path, "replace", _flaky)
    runner._atomic_write(path, "new", retry_delay=0.0)

    assert path.read_text(encoding="utf-8") == "new"
    assert calls["n"] == 3
    assert list(tmp_path.iterdir()) == [path]


def test_report_writing_never_kills_the_run(tmp_path, monkeypatch, capsys):
    # Flushing exists so an unattended run survives a crash; a flush that
    # cannot complete must therefore degrade to a warning, not end the sweep.
    def _always_denied(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(runner.Path, "replace", _always_denied)
    runner._write_reports(
        tmp_path / "r.json", tmp_path / "r.md", "{}", "# report", retry_delay=0.0
    )
    assert "could not write" in capsys.readouterr().out.lower()


def test_a_failed_flush_leaves_the_previous_report_intact(tmp_path, monkeypatch):
    # Flush-per-problem is the crash-survival mechanism of an unattended run,
    # so the flush itself must not be the thing that destroys the report: a
    # plain write truncates first, and a process killed in that window leaves
    # an empty file where hours of rows used to be.
    path = tmp_path / "report.json"
    runner._atomic_write(path, "hours of rows")

    def _boom(self, target):
        raise OSError("killed mid-flush")

    monkeypatch.setattr(runner.Path, "replace", _boom)
    with pytest.raises(OSError):
        runner._atomic_write(path, "the next flush")

    assert path.read_text(encoding="utf-8") == "hours of rows"


# -- reference budget symmetry ------------------------------------------------


def _budget_args(**over) -> argparse.Namespace:
    base: dict[str, object] = {
        "max_iter": 1000,
        "max_time": 60.0,
        "ref_max_iter": None,
        "ref_max_time": None,
        "ref_option": [],
    }
    base.update(over)
    return argparse.Namespace(**base)


def test_reference_inherits_ipax_budget_by_default():
    # A reference handed a bigger budget than ipax manufactures `ipax-gap`
    # rows, so symmetry is the default and asymmetry must be asked for.
    max_iter, max_time, opts = runner._ref_settings(_budget_args())
    assert (max_iter, max_time, opts) == (1000, 60.0, ())


def test_reference_budget_can_be_set_asymmetrically_on_purpose():
    max_iter, max_time, _ = runner._ref_settings(
        _budget_args(ref_max_iter=3000, ref_max_time=120.0)
    )
    assert (max_iter, max_time) == (3000, 120.0)


def test_reference_option_overrides_are_typed():
    _, _, opts = runner._ref_settings(
        _budget_args(
            ref_option=[
                "mu_strategy=monotone",
                "limited_memory_max_history=10",
                "tol=1e-6",
            ]
        )
    )
    assert dict(opts) == {
        "mu_strategy": "monotone",
        "limited_memory_max_history": 10,
        "tol": 1e-6,
    }


def test_malformed_reference_option_is_rejected():
    with pytest.raises(ValueError):
        runner._ref_settings(_budget_args(ref_option=["mu_strategy"]))


def test_report_records_both_solvers_parameters():
    # A report that does not state the two parameter sets can be misread as a
    # like-for-like comparison; the knobs that do NOT match must be named.
    params = runner._parameters(_options(), 1000, 60.0, (("mu_strategy", "monotone"),))
    text = runner.format_report(
        [_row("HS35", "agree")], {"ipax": "0.8.0"}, "lbfgs/dense", {}, params
    )
    assert "mu_strategy" in text
    assert "monotone" in text
    # The un-matched knobs are called out, not left for the reader to discover.
    assert "limited-memory" in text or "L-BFGS history" in text


def test_report_tallies_verdicts_and_names_the_gaps():
    rows = [
        _row("AGG", "ipax-gap"),
        _row("PDE1", "ipax-gap*"),
        _row("HS35", "agree"),
        _row("OET7", "both-hard"),
    ]
    text = runner.format_report(
        rows, {"ipax": "0.8.0", "timestamp": "now"}, "lbfgs/dense", {"too_large": 2}
    )
    assert "ipax-gap" in text
    assert "AGG" in text and "PDE1" in text
    assert "too_large" in text  # skips are visible, not silently dropped
    # The actionable list is the point of the report: it names the gaps, and
    # only the gaps (both the scored and the unscored `*` variant).
    section = text.split("## ipax-gap")[1].split("\n## ")[0]
    assert "AGG" in section and "PDE1" in section
    assert "HS35" not in section and "OET7" not in section
