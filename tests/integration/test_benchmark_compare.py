"""The A/B report comparison, including the objective-drift blind spot.

Motivating incident (S2MPJ ``OET7``, 2026-07-20): a default change moved a
problem from objective ``4.45e-05`` to ``0.0872`` — a ~2000x worse local
optimum — and the sweep's ``correct`` count scored it unchanged, because
``OET7`` carries no dataset reference objective and the metric credits any
certified convergence. The correctness delta therefore *understates* basin
damage on exactly the problems that cannot be scored against a reference.
``compare_reports`` surfaces those separately.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.runners.compare import compare_reports, format_comparison


def _row(problem, config="lbfgs/sparse", **over):
    row = {
        "problem": f"s2mpj/{problem}",
        "kind": "NLP",
        "backend": "numpy",
        "config": config,
        "status": "optimal",
        "success": True,
        "correct": True,
        "converged": True,
        "n_iter": 10,
        "kkt_error": 1e-9,
        "dual_infeasibility": 1e-9,
        "primal_infeasibility": 1e-9,
        "complementarity": 1e-9,
        "constraint_violation": 0.0,
        "error_vs_optimum": None,
        "objective": 1.0,
        "expected_objective": None,
        "expected_infeasible": False,
        "pbclass": "C-CLLR2-AN-3-4",
        "solve_time": 1.0,
        "linear_solver": "sparse [Feral LDL^T (CPU)]",
        "gradient_source": "analytic",
        "hessian_source": "lbfgs",
        "error": None,
    }
    row.update(over)
    return row


def _write(tmp_path, name, rows):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"environment": {}, "results": rows}))
    return str(path)


def test_reports_correctness_flips_per_config(tmp_path):
    base = _write(tmp_path, "base", [_row("A"), _row("B", correct=False)])
    new = _write(tmp_path, "new", [_row("A", correct=False), _row("B")])

    cmp = compare_reports(base, new)

    assert cmp.base_correct == 1 and cmp.new_correct == 1
    assert [f.problem for f in cmp.broken] == ["s2mpj/A"]
    assert [f.problem for f in cmp.fixed] == ["s2mpj/B"]
    assert cmp.per_config["lbfgs/sparse"] == (1, 1)


def test_objective_drift_is_reported_when_correctness_cannot_see_it(tmp_path):
    # The OET7 case: no reference objective, `correct` True in both runs, but
    # the objective moved by orders of magnitude. This must be surfaced.
    base = _write(tmp_path, "base", [_row("OET7", objective=4.4474e-05)])
    new = _write(tmp_path, "new", [_row("OET7", objective=0.0871596)])

    cmp = compare_reports(base, new)

    assert not cmp.broken and not cmp.fixed  # invisible to the correct-count
    assert len(cmp.objective_drift) == 1
    drift = cmp.objective_drift[0]
    assert drift.problem == "s2mpj/OET7"
    assert drift.worse is True  # minimization: higher is worse
    assert drift.unscored is True  # no reference objective to catch it
    assert drift.base_objective == pytest.approx(4.4474e-05)
    assert "OET7" in format_comparison(cmp)
    assert "drift" in format_comparison(cmp).lower()


def test_improvement_is_distinguished_from_regression(tmp_path):
    # CRESC50's case: the objective *improved*. Reported, but not as damage.
    base = _write(tmp_path, "base", [_row("CRESC50", objective=1.0613)])
    new = _write(tmp_path, "new", [_row("CRESC50", objective=0.599455)])

    drift = compare_reports(base, new).objective_drift
    assert len(drift) == 1 and drift[0].worse is False


def test_numerically_identical_objectives_are_not_drift(tmp_path):
    # Rounding-level differences (the 19 unchanged problems) are not drift.
    base = _write(tmp_path, "base", [_row("DUALC1", objective=6155.25)])
    new = _write(tmp_path, "new", [_row("DUALC1", objective=6155.2500000001)])

    assert compare_reports(base, new).objective_drift == []


def test_drift_on_a_scored_problem_is_flagged_but_not_unscored(tmp_path):
    # ELATTAR's case: it *has* a reference objective, so the correct-count
    # already catches it; the drift entry records that it is not a blind spot.
    base = _write(
        tmp_path, "base", [_row("ELATTAR", objective=0.1427, expected_objective=0.1427)]
    )
    new = _write(
        tmp_path,
        "new",
        [_row("ELATTAR", objective=74.2, expected_objective=0.1427, correct=False)],
    )

    cmp = compare_reports(base, new)

    assert [f.problem for f in cmp.broken] == ["s2mpj/ELATTAR"]
    assert len(cmp.objective_drift) == 1
    assert cmp.objective_drift[0].unscored is False


def test_route_changes_are_summarized(tmp_path):
    # Attribution aid: which problems changed linear-solver route at all. A
    # config with zero route changes is a built-in control for route work.
    base = _write(tmp_path, "base", [_row("A"), _row("B", config="exact/sparse")])
    new = _write(
        tmp_path,
        "new",
        [
            _row("A", linear_solver="sparse-NE [Feral LDL^T (CPU)]"),
            _row("B", config="exact/sparse"),
        ],
    )

    cmp = compare_reports(base, new)

    assert cmp.route_changes["lbfgs/sparse"] == 1
    assert cmp.route_changes.get("exact/sparse", 0) == 0


def test_missing_and_extra_rows_are_reported_not_silently_dropped(tmp_path):
    base = _write(tmp_path, "base", [_row("A"), _row("OnlyInBase")])
    new = _write(tmp_path, "new", [_row("A"), _row("OnlyInNew")])

    cmp = compare_reports(base, new)

    assert cmp.base_only == 1 and cmp.new_only == 1
    assert "OnlyInBase" not in [f.problem for f in cmp.broken]
