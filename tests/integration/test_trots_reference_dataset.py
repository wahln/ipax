"""Breedveld et al. 2017's timing supplement against the local TROTS dataset.

Gated on ``IPAX_TROTS_DIR`` (the dataset is not vendored); no ``h5py`` needed —
only the file listing and the ``Results/*_result.txt`` headers are read.

* Every problem of the paper's supplement maps to a ``.mat`` case of the
  dataset, and every dataset case outside the supplement is a
  ``Prostate_BT_*`` brachytherapy case (the paper does not solve those).
* The dataset's reference results come from the paper's Mehrotra runs: for
  all 120 cases the ``Results`` file's "Number of Iterations" and "Solve Time"
  equal the supplement's Mehrotra iterations and seconds (no exceptions as of
  the 2026-09-27 conversion).
"""

from __future__ import annotations

import pytest

trots = pytest.importorskip("benchmarks.corpus.trots")
ref = pytest.importorskip("benchmarks.corpus.trots_reference")

_ROOT = trots.trots_dir()
if _ROOT is None:
    pytest.skip("no TROTS dataset (set IPAX_TROTS_DIR)", allow_module_level=True)

_REFS = ref.breedveld2017_references()


def test_every_supplement_problem_maps_to_a_dataset_case():
    cases = set(trots.list_trots_cases())
    missing = sorted(set(_REFS) - cases)
    if missing and len(cases) < 145:
        pytest.skip(f"partial TROTS copy ({len(cases)} cases)")
    assert missing == []
    assert all(c.startswith("Prostate_BT_") for c in cases - set(_REFS))


@pytest.mark.parametrize("case", sorted(_REFS))
def test_results_file_is_the_papers_mehrotra_run(case):
    result = trots.reference_for(case)
    if result is None:
        pytest.skip(f"no Results/{case}_result.txt in this TROTS copy")
    paper = _REFS[case].mehrotra
    assert result.n_iter == paper.iterations
    assert result.header["Solve Time"] == pytest.approx(paper.seconds, rel=1e-9)
