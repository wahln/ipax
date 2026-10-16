"""Reject invalid LSQR tolerances before preparing a TROTS warm start."""

from __future__ import annotations

import pytest

from benchmarks.corpus.trots import TROTSProblem
from benchmarks.runners import trots as runner
from tests.unit.test_trots_breedveld_start import _instance


@pytest.mark.parametrize("tol", [float("nan"), float("inf"), -float("inf"), 0.0, -1e-6])
def test_breedveld_start_rejects_invalid_lsqr_tolerance(namespace, tol):
    problem = TROTSProblem(_instance(), namespace)
    x0 = namespace.ones((problem.n_vars,), dtype=namespace.float64)

    with pytest.raises(ValueError, match="lsqr_tol"):
        problem.breedveld_warm_start(x0, lsqr_tol=tol)


@pytest.mark.parametrize("tol", ["nan", "inf", "-inf", "0", "-1e-6"])
def test_runner_rejects_invalid_lsqr_tolerance_before_loading_data(capsys, tol):
    with pytest.raises(SystemExit) as excinfo:
        runner._parse_args(runner._build_parser(), [f"--lsqr-tol={tol}"])

    assert excinfo.value.code == 2
    assert "--lsqr-tol must be positive and finite" in capsys.readouterr().err
