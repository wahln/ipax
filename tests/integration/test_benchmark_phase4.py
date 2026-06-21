"""Phase-4 benchmark extras: plots, OSQP baseline, external corpus loaders."""

from __future__ import annotations

import pytest

import ipax
from benchmarks.corpus import default_corpus
from benchmarks.corpus.external import cutest_problems, maros_meszaros_problems
from ipax.testing.backends import import_namespace


def _case(name: str):
    return next(c for c in default_corpus() if c.name == name)


# -- external loaders are gracefully empty when their deps are absent --------


def test_external_loaders_return_empty_without_deps():
    # pycutest needs a system CUTEst install; the data set is download-gated.
    assert cutest_problems() == []
    assert maros_meszaros_problems() == []


# -- matplotlib plots (skipped if matplotlib is not installed) ---------------


def test_plot_scaling_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from benchmarks.harness import ScalingPoint
    from benchmarks.harness.plots import plot_scaling

    points = [
        ScalingPoint("krylov", 100, 5, True, 0.1, 0.5, 1e-9),
        ScalingPoint("krylov", 200, 6, True, 0.2, 0.8, 1e-9),
        ScalingPoint("dense", 100, 5, True, 0.3, 4.0, 1e-9),
        ScalingPoint("dense", 200, 6, True, 0.9, 16.0, 1e-9),
    ]
    out = plot_scaling(points, tmp_path / "scaling.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_qc_iterations_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from benchmarks.harness import run_case
    from benchmarks.harness.plots import plot_qc_iterations

    xp = import_namespace("numpy")
    results = [
        run_case(
            _case("bound_qp"),
            config=label,
            options=opts,
            xp=xp,
            backend="numpy",
        )
        for label, opts in (
            ("exact/dense", ipax.Options(hessian="exact", linsolve="dense")),
            ("lbfgs/dense", ipax.Options(hessian="lbfgs", linsolve="dense")),
        )
    ]
    out = plot_qc_iterations(results, tmp_path / "iters.png")
    assert out.exists() and out.stat().st_size > 0


# -- OSQP baseline (skipped if osqp is not installed) ------------------------


def test_osqp_available_and_solves_qp():
    pytest.importorskip("osqp")
    from benchmarks.baselines import OsqpBaseline, available_baselines

    assert "osqp" in [b.name for b in available_baselines()]

    xp = import_namespace("numpy")
    problem, x0 = _case("hs35").build(xp)
    result = OsqpBaseline().solve(problem, x0)
    assert result.success
    # HS35 optimum (4/3, 7/9, 4/9).
    expected = xp.asarray([4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0])
    assert float(xp.max(xp.abs(xp.asarray(result.x) - expected))) <= 1e-5


def test_osqp_rejects_nonlinear_problem():
    pytest.importorskip("osqp")
    from benchmarks.baselines import BaselineUnsupported, OsqpBaseline

    xp = import_namespace("numpy")
    problem, x0 = _case("hs7").build(xp)  # nonlinear equality + objective
    with pytest.raises(BaselineUnsupported):
        OsqpBaseline().solve(problem, x0)
