"""End-to-end tests of second-order corrections (filter and Breedveld).

``BreedveldOptions.max_soc`` wires the driver's second-order correction
(Wächter & Biegler 2006, §2.4) into the Breedveld step controller's Markov
acceptance (Breedveld 2017, eqs. (34)–(35)); ``LineSearchOptions.max_soc``
(default on) is the same correction in the filter line search itself. The
Hock–Schittkowski subset never rejects a full step that worsens feasibility
under the (loose) Breedveld test, so the end-to-end exercise uses the
textbook Maratos-effect problem (Nocedal & Wright 2006, Example 15.4), whose
Newton steps from a far start increase both the objective and the constraint
violation and trigger a correction under either globalization.

Acceptance/attempt counts come from the public per-iteration record
(``IterationRecord.used_soc`` / ``.soc_attempted`` on ``Result.history``),
not from spying on the controller.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from ipax import Options, Status, solve
from ipax.ipm.driver import IPMDriver
from ipax.options import BreedveldOptions, LineSearchOptions
from ipax.testing.problems import HS43, HS71, Maratos
from tests._helpers import array, assert_allclose, assert_scalar_close

# A far start (radius 0.5, angle 2.5 rad) whose later full steps are rejected
# by the Markov test while worsening θ — measured to trigger accepted
# corrections on every backend.
_START = (0.5 * math.cos(2.5), 0.5 * math.sin(2.5))


def _options(**breedveld: object) -> Options:
    return Options(
        hessian="exact",
        linsolve="dense",
        globalization="breedveld",
        breedveld=BreedveldOptions(**breedveld),  # type: ignore[arg-type]
    )


def _soc_counts(result) -> tuple[int, int]:
    """(attempted, accepted) rows in ``result.history`` (see module docstring)."""
    attempted = sum(1 for r in result.history if r.soc_attempted)
    accepted = sum(1 for r in result.history if r.used_soc)
    return attempted, accepted


def _solve_maratos(namespace, options):
    return solve(Maratos(namespace), array(namespace, list(_START)), options=options)


def test_soc_reaches_the_same_optimum_as_backtracking(namespace):
    plain = _solve_maratos(namespace, _options())
    assert _soc_counts(plain) == (0, 0)
    assert all(not r.used_soc for r in plain.history)

    corrected = _solve_maratos(namespace, _options(max_soc=1))
    assert _soc_counts(corrected)[1] >= 1
    assert any(r.used_soc for r in corrected.history)
    # An accepted correction is judged at the trial's own (full) length.
    assert all(r.alpha_primal == 1.0 for r in corrected.history if r.used_soc)
    # At least one attempt is tried and rejected before restoration takes
    # over (the row convention: an attempt may precede a restoration jump).
    assert any(r.restored and r.soc_attempted for r in corrected.history)

    for result in (plain, corrected):
        assert result.status is Status.OPTIMAL
        assert_allclose(namespace, result.x, array(namespace, [1.0, 0.0]), atol=1e-6)
        assert_scalar_close(result.objective, -1.0, rtol=1e-8, atol=1e-8)


def test_filter_soc_recorded_on_iteration_history(namespace):
    # LineSearchOptions.max_soc defaults to 4 (on): the filter line search's
    # own W&B 2006 §2.4 correction, exercised the same way as Breedveld's.
    off = _solve_maratos(
        namespace,
        Options(
            hessian="exact",
            linsolve="dense",
            line_search=LineSearchOptions(max_soc=0),
        ),
    )
    assert _soc_counts(off) == (0, 0)
    assert all(not r.used_soc for r in off.history)

    on = _solve_maratos(namespace, Options(hessian="exact", linsolve="dense"))
    assert _soc_counts(on)[1] >= 1
    assert any(r.used_soc for r in on.history)
    assert on.status is Status.OPTIMAL


def test_soc_composes_with_common_step(namespace):
    # An accepted correction is taken at the full length the Markov test
    # judged; under ``common_step`` that same α moves the duals.
    result = _solve_maratos(namespace, _options(max_soc=1, common_step=True))
    assert _soc_counts(result)[1] >= 1
    assert result.status is Status.OPTIMAL
    steps = [r for r in result.history[1:] if not r.restored]
    assert steps
    assert all(r.alpha_primal == r.alpha_dual for r in steps)
    assert_allclose(namespace, result.x, array(namespace, [1.0, 0.0]), atol=1e-6)


def test_breedveld_soc_rounds_come_from_breedveld_options(namespace, monkeypatch):
    # The driver threads BreedveldOptions.max_soc into the correction loop; the
    # filter line search's LineSearchOptions.max_soc is not read on this path.
    # The round count itself is internal to ``soc()`` (not surfaced on
    # IterationRecord), so this still spies on the factorization-reuse call —
    # that is what actually tests "the round count comes from BreedveldOptions".
    resolves = {"n": 0}
    original = IPMDriver._resolve_reused_factorization

    def spy(self, *args, **kwargs):
        resolves["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(IPMDriver, "_resolve_reused_factorization", spy)

    silent = dataclasses.replace(
        _options(max_soc=0), line_search=LineSearchOptions(max_soc=4)
    )
    silent_result = _solve_maratos(namespace, silent)
    assert _soc_counts(silent_result) == (0, 0)
    assert resolves["n"] == 0

    corrected = dataclasses.replace(
        _options(max_soc=2), line_search=LineSearchOptions(max_soc=0)
    )
    result = _solve_maratos(namespace, corrected)
    assert result.status is Status.OPTIMAL
    attempts, accepted = _soc_counts(result)
    assert accepted >= 1
    # Up to two rounds per attempt — the count is BreedveldOptions.max_soc, not
    # the filter's 0. (A round may take the direct route's fresh δ_w = 0 solve
    # instead of the re-solve, so the bound is not always attained.)
    assert attempts < resolves["n"] <= 2 * attempts


@pytest.mark.parametrize(
    ("cls", "start"),
    [(HS71, [1.0, 5.0, 5.0, 1.0]), (HS43, [0.0, 0.0, 0.0, 0.0])],
    ids=["HS71", "HS43"],
)
def test_soc_leaves_hock_schittkowski_optima_unchanged(namespace, cls, start):
    # The Markov test never rejects a θ-worsening full step here, so the
    # option is inert: same optimum, no correction attempted.
    problem, x0 = cls(namespace), array(namespace, start)
    plain = solve(problem, x0, options=_options())
    corrected = solve(problem, x0, options=_options(max_soc=1))
    assert _soc_counts(corrected)[0] == 0
    assert corrected.status is Status.OPTIMAL
    assert_scalar_close(corrected.objective, plain.objective, rtol=1e-8, atol=1e-8)
    assert_allclose(namespace, corrected.x, plain.x, rtol=1e-6, atol=1e-6)
