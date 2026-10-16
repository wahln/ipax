"""Regression: the step computation's tall-Jacobian passes and O(m) sweeps.

TROTS 2026-09 (``Protons_01``: n = 1101, m = 369 445 linear inequality rows,
``hessian="exact"``, dense route, ``corrections="mehrotra"``): outside the
condensed Gram, an iteration's cost is Jacobian passes (``J·v``/``Jᵀ·v``, a
sweep over all 369k rows each) and O(m) vector sweeps, so every repeated one
shows in the per-iteration wall. Pinned here, per step:

- one Gram per factorization (the corrector re-solves the retained factor);
- ``Jᵀ·v`` only for the loop-top ``∇L`` and one per condensed right-hand
  side — the μ-reduction loop's KKT tests (W&B 2006, eq. (7)) reuse the
  loop-top ``∇L`` instead of re-forming ``∇gᵀλ`` per test;
- ``J·v`` only for the per-solve back-substitution ``Δs = −(g+s) − JΔx``;
- the corrector pass measures each direction's fraction-to-boundary length
  once (the probing oracle and the Mehrotra keep-fraction test share the
  affine direction's), and the barrier directional derivative ``∇φ_μᵀd`` is
  evaluated once per step (the corrector's descent check hands it to the line
  search — the Breedveld controller and the filter line search alike);
- Gondzio's loop measures each candidate once and re-measures the direction
  it just accepted through the per-pass memo;
- under the filter line search, each second-order correction (W&B 2006,
  §2.4, eq. (26)) re-solves the retained factor and costs exactly one
  ``Jᵀ·v`` (its right-hand side) and one ``J·v`` (its slack update).

The problem evaluates ``g(x) = A x − b`` directly, so every operator call
counted below comes from the step computation.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ipax import Options, solve
from ipax.backend.namespace import array_namespace
from ipax.backend.operators import Dense
from ipax.ipm import driver as driver_mod
from ipax.ipm.driver import IPMDriver
from ipax.linalg.dense import DenseSolver
from ipax.problem.base import Problem
from tests._helpers import array

_M, _N = 24, 3


class _SpyJacobian(Dense):
    """A Gram-capable ``Dense`` that counts its tall passes and Grams."""

    def __init__(self, A, counts):
        super().__init__(A)
        self._counts = counts

    def matvec(self, v):
        self._counts["matvec"] += 1
        return super().matvec(v)

    def rmatvec(self, v):
        self._counts["rmatvec"] += 1
        return super().rmatvec(v)

    def gram(self, weights, *, accumulate_dtype=None, hinted_only=False):
        del accumulate_dtype, hinted_only
        self._counts["gram"] += 1
        A = self.dense_matrix()
        xp = array_namespace(A, weights)
        weighted = A * xp.expand_dims(weights, axis=1)
        return xp.matmul(xp.permute_dims(A, (1, 0)), weighted)


class _TallQP(Problem):
    """``min ½‖x − t‖²`` s.t. ``A x ≤ b`` with a tall ``A`` (24 × 3)."""

    def __init__(self, xp, counts):
        self._xp = xp
        self._counts = counts
        rows = []
        for i in range(_M):
            a = (i % 5) - 2.0
            b = ((3 * i) % 7) - 3.0
            rows.append([1.0 + 0.1 * i, a, b])
        self._A = array(xp, rows)
        self._b = array(xp, [1.0 + 0.05 * i for i in range(_M)])
        self._t = array(xp, [3.0, 2.0, -1.0])

    @property
    def n_vars(self) -> int:
        return _N

    def objective(self, x):
        d = x - self._t
        return 0.5 * self._xp.sum(d * d)

    def gradient(self, x):
        return x - self._t

    def ineq_constraints(self, x):
        return self._xp.matmul(self._A, x) - self._b

    def ineq_jacobian(self, x):
        del x
        return _SpyJacobian(self._A, self._counts)

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del y_eq, y_ineq
        return sigma * self._xp.eye(_N, dtype=x.dtype)


def _count_calls(monkeypatch, counts, owner, name, key=None, *, when=None):
    original = getattr(owner, name)

    def wrapped(*args, **kwargs):
        if when is None or when(*args, **kwargs):
            counts[key or name] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapped)


def _instrument(monkeypatch) -> Counter:
    counts: Counter = Counter()
    for name in (
        "_condensed_rhs",
        "_corrected_step",
        "_dphi",
        "_resolve_reused_factorization",
    ):
        _count_calls(monkeypatch, counts, IPMDriver, name)
    # The loop top evaluates E_0 (μ = 0) once per iteration; the monotone
    # μ-reduction loop evaluates E_μ (μ > 0) at the same iterate.
    _count_calls(
        monkeypatch,
        counts,
        IPMDriver,
        "kkt_error",
        "kkt_top",
        when=lambda *a, **k: k.get("mu") == 0.0,
    )
    _count_calls(
        monkeypatch,
        counts,
        IPMDriver,
        "kkt_error",
        "kkt_mu",
        when=lambda *a, **k: k.get("mu", 0.0) > 0.0,
    )
    for name in ("_alpha_primal", "_alpha_dual"):
        # τ = 1 is the corrector's maximal-step measurement; the driver's own
        # boundary rule for the accepted step uses τ < 1.
        _count_calls(
            monkeypatch,
            counts,
            IPMDriver,
            name,
            f"{name}@1",
            when=lambda *a, **k: k.get("tau") == 1.0,
        )
    _count_calls(monkeypatch, counts, DenseSolver, "factor")
    _count_calls(monkeypatch, counts, DenseSolver, "solve")
    _count_calls(monkeypatch, counts, driver_mod, "recover_eliminated", "recover")
    return counts


def _run(xp, counts, *, globalization="breedveld", **options):
    problem = _TallQP(xp, counts)
    return solve(
        problem,
        array(xp, [0.0, 0.0, 0.0]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            globalization=globalization,
            max_iter=12,
            **options,
        ),
    )


def _soc_solves(counts: Counter) -> int:
    """Re-solves of the retained factor made outside the corrector pass.

    The corrector's re-solves are every condensed RHS but the affine one (one
    per corrected step); anything else on the retained factor is the filter
    line search's second-order correction (W&B 2006, §2.4).
    """
    corrector = counts["_condensed_rhs"] - counts["_corrected_step"]
    return counts["_resolve_reused_factorization"] - corrector


def _assert_tall_passes(counts: Counter, soc: int = 0) -> None:
    # One Gram per factorization; every further solve reuses the factor.
    assert counts["gram"] == counts["factor"]
    # Every solve is a condensed direction or a second-order correction.
    assert counts["solve"] == counts["_condensed_rhs"] + soc
    # Jᵀ·v: loop-top ∇L + one per condensed RHS + one per SOC RHS — nothing
    # from the KKT tests.
    assert counts["rmatvec"] == counts["kkt_top"] + counts["_condensed_rhs"] + soc
    # J·v: one per recovered direction + one per SOC slack update.
    assert counts["matvec"] == counts["recover"] + soc


def test_no_corrector_monotone_passes(namespace, monkeypatch):
    counts = _instrument(monkeypatch)
    result = _run(namespace, counts, corrections="none", mu_schedule="monotone")
    assert result.n_iter >= 3
    # Meaningful only if the μ-reduction loop evaluated its own KKT tests.
    assert counts["kkt_mu"] > 0
    assert counts["factor"] == counts["solve"] == counts["_condensed_rhs"]
    _assert_tall_passes(counts)


@pytest.mark.parametrize("globalization", ["filter", "breedveld"])
@pytest.mark.parametrize("mu_schedule", ["monotone", "probing"])
def test_mehrotra_corrector_passes(namespace, monkeypatch, mu_schedule, globalization):
    counts = _instrument(monkeypatch)
    result = _run(
        namespace,
        counts,
        globalization=globalization,
        corrections="mehrotra",
        mu_schedule=mu_schedule,
    )
    steps = counts["_corrected_step"]
    assert result.n_iter >= 3
    assert steps >= 3
    # Precondition: plain predictor + corrector every step (no keep-fraction or
    # descent fallback re-solves, no δ_w retries) so the counts are exact.
    assert counts["factor"] == steps
    if mu_schedule == "monotone":
        assert counts["kkt_mu"] > 0
    assert counts["_condensed_rhs"] == counts["recover"] == 2 * steps
    soc = _soc_solves(counts)
    if globalization == "breedveld":
        assert soc == 0  # no second-order corrections outside the filter
    else:
        assert soc > 0  # precondition: the SOC accounting is exercised
    _assert_tall_passes(counts, soc)
    # Each direction's maximal step measured once: affine + corrected.
    assert counts["_alpha_primal@1"] == counts["_alpha_dual@1"] == 2 * steps
    # ∇φ_μᵀd once per step: the descent check's value feeds the line search.
    assert counts["_dphi"] == steps


@pytest.mark.parametrize("globalization", ["filter", "breedveld"])
def test_gondzio_corrector_passes(namespace, monkeypatch, globalization):
    counts = _instrument(monkeypatch)
    result = _run(
        namespace,
        counts,
        globalization=globalization,
        corrections="gondzio",
        mu_schedule="probing",
    )
    steps = counts["_corrected_step"]
    assert result.n_iter >= 3
    assert steps >= 3
    assert counts["factor"] == steps
    # Precondition: at least one Gondzio candidate beyond affine + Mehrotra.
    assert counts["recover"] > 2 * steps
    assert counts["_condensed_rhs"] == counts["recover"]
    soc = _soc_solves(counts)
    if globalization == "breedveld":
        assert soc == 0
    else:
        assert soc > 0
    _assert_tall_passes(counts, soc)
    # Each recovered direction's maximal step measured exactly once: the
    # accepted candidate's re-measure at the next Gondzio round is a memo hit.
    assert counts["_alpha_primal@1"] == counts["_alpha_dual@1"] == counts["recover"]
    assert counts["_dphi"] == steps
