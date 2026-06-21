"""Smoke test for the reference cross-check harness (scipy baseline).

Keeps ``benchmarks/baselines`` + the cross-check path importable and correct.
Skipped entirely when SciPy is not installed.
"""

from __future__ import annotations

import pytest

scipy = pytest.importorskip("scipy")

import ipax  # noqa: E402
from benchmarks.baselines import (  # noqa: E402
    BaselineUnsupported,
    ScipyBaseline,
    available_baselines,
)
from benchmarks.corpus import default_corpus  # noqa: E402
from benchmarks.harness import (  # noqa: E402
    capture_environment,
    cross_check,
    format_crosscheck,
)
from ipax.testing.backends import import_namespace  # noqa: E402


def _case(name: str):
    return next(c for c in default_corpus() if c.name == name)


def test_available_baselines_includes_scipy():
    assert "scipy-trust-constr" in [b.name for b in available_baselines()]


def test_crosscheck_agrees_with_scipy_on_equality_qp():
    xp = import_namespace("numpy")
    result = cross_check(
        _case("equality_qp"),
        ipax_options=ipax.Options(hessian="exact", linsolve="dense"),
        baseline=ScipyBaseline(),
        xp=xp,
    )
    assert result.status == "ok"
    assert result.agree
    assert result.x_gap is not None and result.x_gap <= 1e-4


def test_crosscheck_skips_matrix_free_problem():
    xp = import_namespace("numpy")
    result = cross_check(
        _case("rt_like_300"),
        ipax_options=ipax.Options(hessian="exact", linsolve="krylov"),
        baseline=ScipyBaseline(),
        xp=xp,
    )
    assert result.status == "skipped"  # matrix-free Jacobian -> BaselineUnsupported


def test_scipy_baseline_rejects_matrix_free_directly():
    xp = import_namespace("numpy")
    problem, x0 = _case("rt_like_300").build(xp)
    with pytest.raises(BaselineUnsupported):
        ScipyBaseline().solve(problem, x0)


def test_format_crosscheck_renders_markdown():
    xp = import_namespace("numpy")
    results = [
        cross_check(
            _case("hs35"),
            ipax_options=ipax.Options(hessian="exact", linsolve="dense"),
            baseline=ScipyBaseline(),
            xp=xp,
        )
    ]
    markdown = format_crosscheck(results, capture_environment())
    assert "reference cross-check" in markdown
    assert "hs35" in markdown
