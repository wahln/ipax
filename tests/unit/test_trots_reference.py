"""Breedveld et al. 2017's TROTS timing supplement (ESM 2) as reference data.

The loader (:mod:`benchmarks.corpus.trots_reference`) reads the committed JSON
conversion of the paper's supplement; the TROTS runner shows the paper's
Mehrotra iterations and seconds next to ipax's. No dataset is needed here (the
dataset-gated cross-checks live in ``tests/integration/test_trots_reference_dataset.py``).
"""

from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path

import pytest

ref = pytest.importorskip("benchmarks.corpus.trots_reference")

_SITE_COUNTS = {
    "Prostate CK": 30,
    "Prostate VMAT": 30,
    "Head-and-Neck": 15,
    "Head-and-Neck Alt": 15,
    "Protons": 20,
    "Liver": 10,
}


def test_supplement_covers_the_papers_120_problems():
    refs = ref.breedveld2017_references()
    assert len(refs) == 120
    assert sorted(r.index for r in refs.values()) == list(range(120))
    counts: dict[str, int] = {}
    for r in refs.values():
        counts[r.site] = counts.get(r.site, 0) + 1
    assert counts == _SITE_COUNTS
    # The paper does not solve the brachytherapy cases.
    assert not any(case.startswith("Prostate_BT") for case in refs)


@pytest.mark.parametrize(
    ("label", "case"),
    [
        ("Prostate CK 01", "Prostate_CK_01"),
        ("Prostate VMAT 101", "Prostate_VMAT_101"),
        ("Head-and-Neck 01", "Head-and-Neck_01"),
        ("Head-and-Neck Alt 03", "Head-and-Neck-Alt_03"),
        ("Protons 20", "Protons_20"),
        ("Liver 10", "Liver_10"),
    ],
)
def test_paper_labels_map_to_trots_case_names(label, case):
    r = ref.breedveld2017_reference(case)
    assert r is not None
    assert r.label == label
    assert r.case == case


def test_lookup_accepts_corpus_names_and_misses_cleanly():
    assert ref.breedveld2017_reference("trots/Protons_01") == (
        ref.breedveld2017_reference("Protons_01")
    )
    assert ref.breedveld2017_reference("Prostate_BT_01") is None
    assert ref.breedveld2017_reference("nonexistent") is None


def test_protons01_values_are_the_workbooks():
    # Spot values straight from the xlsx (TROTS Overview row 96, Problem
    # Complexity row 94): the conversion does not round.
    r = ref.breedveld2017_reference("Protons_01")
    assert r.site == "Protons"
    assert (r.n, r.m, r.n_final, r.m_final) == (1080, 87637, 1101, 369445)
    assert r.default == ref.BreedveldRun(120, 71.840895175900002)
    assert r.mehrotra == ref.BreedveldRun(79, 55.050395965600003)
    assert r.gondzio == ref.BreedveldRun(56, 52.781903982199999)
    assert r.variants["no_tile_no_perm_no_mpa"] == ref.BreedveldRun(88, 117.430926085)


def test_duplicated_cells_agree_with_the_mehrotra_run():
    # The workbook's O/T, Z/AJ, AY/BC and BS cells are formula references to
    # the Mehrotra columns D/H; the naive 16-thread cells AE/AO reference the
    # no-tile/no-perm/no-MPA variant K/P.
    for r in ref.breedveld2017_references().values():
        assert r.variants["tile_perm_mpa"] == r.mehrotra
        assert r.threads["optimal"][16] == r.mehrotra
        assert r.init_ratio["init_ratio"] == r.mehrotra
        assert r.breakdown_s["total_runtime"] == r.mehrotra.seconds
        assert r.threads["naive"][16] == r.variants["no_tile_no_perm_no_mpa"]
        assert set(r.threads["optimal"]) == {1, 2, 4, 8, 16}
        assert r.n_final >= r.n
        assert r.m_final >= r.m


def test_site_timings_are_the_sums_of_the_per_case_breakdowns():
    sites = ref.breedveld2017_site_timings()
    assert set(sites) == set(_SITE_COUNTS)
    modules = (
        "permute_tile",
        "initialise",
        "cholesky",
        "backsolves",
        "functions_and_gradients",
        "make_tqd",
        "steplength",
        "make_n",
    )
    refs = ref.breedveld2017_references().values()
    for site, per_module in sites.items():
        assert tuple(per_module) == modules
        for module, seconds in per_module.items():
            total = math.fsum(r.breakdown_s[module] for r in refs if r.site == site)
            assert seconds == pytest.approx(total, rel=1e-12)
    # A fresh copy: mutating it does not leak into the next call.
    sites["Liver"]["make_n"] = -1.0
    assert ref.breedveld2017_site_timings()["Liver"]["make_n"] > 0


def test_json_documents_source_license_and_units():
    path = Path(ref.__file__).parent / "data" / "breedveld2017" / "trots_reference.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["source"]["doi"] == "10.1007/s10589-017-9919-4"
    assert "creativecommons.org/licenses/by/4.0" in doc["source"]["license"]
    assert {"iterations", "time_s", "breakdown_s"} <= set(doc["units"])
    assert (path.parent / "LICENSE").is_file()
    assert (path.parent / "README.md").is_file()


# --------------------------------------------------------------------------- #
# Runner columns (no solve)
# --------------------------------------------------------------------------- #
def _case_result(problem: str, **overrides):
    from benchmarks.harness import CaseResult

    values = {
        "problem": problem,
        "kind": "RT",
        "backend": "numpy",
        "config": "exact/dense",
        "status": "optimal",
        "success": True,
        "correct": True,
        "converged": True,
        "n_iter": 129,
        "kkt_error": 1e-9,
        "dual_infeasibility": 1e-9,
        "primal_infeasibility": 1e-12,
        "complementarity": 1e-9,
        "constraint_violation": 1e-12,
        "error_vs_optimum": None,
        "objective": 32.65,
        "expected_objective": 32.69,
        "expected_infeasible": False,
        "pbclass": None,
        "solve_time": 38.8,
        "linear_solver": "dense",
        "gradient_source": "analytic",
        "hessian_source": "analytic",
        "error": None,
    }
    values.update(overrides)
    assert set(values) >= {
        f.name for f in fields(CaseResult) if f.name != "prepare_time"
    }
    return CaseResult(**values)


def test_solve_row_adds_the_papers_mehrotra_run():
    from benchmarks.runners import trots as runner

    row = runner._solve_row(_case_result("trots/Protons_01"))
    assert row["ref_iters"] == 79
    assert row["ref_time_s"] == 55.050395965600003
    # The harness columns are all kept.
    assert row["n_iter"] == 129
    assert row["solve_time"] == 38.8
    assert row["problem"] == "trots/Protons_01"
    json.dumps(row)  # the JSON report serializes it as-is


def test_solve_row_without_a_paper_reference():
    from benchmarks.runners import trots as runner

    row = runner._solve_row(_case_result("trots/Prostate_BT_01"))
    assert row["ref_iters"] is None
    assert row["ref_time_s"] is None
    assert "paper n/a" in runner._solve_line("Prostate_BT_01", "exact/dense", row)


def test_console_line_shows_ipax_and_paper_side_by_side():
    from benchmarks.runners import trots as runner

    row = runner._solve_row(_case_result("trots/Protons_01"))
    line = runner._solve_line("Protons_01", "exact/dense", row)
    assert "iters= 129" in line
    assert "t=38.8s" in line
    assert "paper iters=79 t=55.1s" in line


def test_markdown_report_has_paper_columns(tmp_path):
    from benchmarks.runners import trots as runner

    rows = [
        runner._solve_row(_case_result("trots/Protons_01")),
        runner._solve_row(_case_result("trots/Prostate_BT_01")),
    ]
    path = tmp_path / "report.md"
    runner._write_markdown(path, [], rows, {})
    text = path.read_text(encoding="utf-8")
    assert "| paper iters | paper time (s) |" in text
    assert "| 38.8 | 79 | 55.1 |" in text
    assert "| 38.8 |  |  |" in text
    assert "CC BY 4.0" in text
