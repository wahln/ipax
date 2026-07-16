"""Regression: free-mode μ oracles must not run away on an ill-scaled problem.

Found via ``examples/radiotherapy_sparse.py`` (2026-07): on a non-negative
least-squares problem with gradients of order 1e4–1e5 and an L-BFGS Hessian,
every non-monotone μ oracle stagnated at a huge *constant* μ (1e2–1e5) while
the filter line search polished the wrong barrier problem to machine precision.
Three stacked defects produced the runaway:

1. ``probing_mu`` trusted the Mehrotra σ-rule with a *quasi-Newton* affine
   probe. A cold L-BFGS direction inflates the dual products (μ_aff ≫ μ), and
   σ = (μ_aff/μ)³ was unguarded above 1 — σ ≈ 3300 on the first iteration.
2. With correctors active, the second-order −ΔΔ targets built from that same
   low-quality affine direction inflated the complementarity state, and the
   adaptive/breedveld oracles (μ ∝ average complementarity) followed it up.
3. Once μ was inflated, the monotone reducer could never bring it down:
   ``update_mu`` computed κ_μ·μ^θ_μ — which *increases* for μ ≥ 25 — instead
   of Wächter & Biegler 2006 eq. (7)'s min(κ_μ·μ, μ^θ_μ). The NWW §5.1
   fallback therefore re-entered monotone mode at 0.8·(inflated gap) and
   locked there forever.

The deterministic problem below reproduces all of it at 200×40 scale.
"""

from __future__ import annotations

import pytest

from ipax import FunctionProblem, Options, Status, solve
from tests._helpers import array

_N_ROWS = 200
_N_COLS = 40


def _ls_problem(xp):
    """Non-negative least squares with badly scaled columns (gradient ~1e4)."""
    rows = [
        [
            ((i * (j + 3)) % 11) / 11.0 * (1.0 + 9.0 * (j + 1) / _N_COLS)
            for j in range(_N_COLS)
        ]
        for i in range(_N_ROWS)
    ]
    d_mat = array(xp, rows)
    x_true = array(xp, [0.0 if j % 3 == 0 else 0.7 for j in range(_N_COLS)])
    d_pres = xp.matmul(d_mat, x_true)
    d_t = xp.permute_dims(d_mat, (1, 0))

    def objective(x):
        r = xp.matmul(d_mat, x) - d_pres
        return 0.5 * xp.sum(r * r)

    def gradient(x):
        return xp.matmul(d_t, xp.matmul(d_mat, x) - d_pres)

    problem = FunctionProblem(
        _N_COLS,
        objective,
        gradient=gradient,
        bounds=(xp.zeros((_N_COLS,), dtype=d_pres.dtype), None),
    )
    return problem, xp.ones((_N_COLS,), dtype=d_pres.dtype)


@pytest.mark.parametrize(
    ("schedule", "corrections"),
    [
        ("probing", "none"),
        ("probing", "gondzio"),
        ("adaptive", "gondzio"),
        ("breedveld", "mehrotra"),
        # The quality function may raise μ (σ > 1) by design, but it is
        # closed-loop — a large μ scores badly through its own predicted
        # complementarity — so it must not reproduce the open-loop Mehrotra
        # runaway either.
        ("quality", "none"),
        ("quality", "gondzio"),
    ],
)
def test_free_mode_oracles_do_not_run_away_on_ill_scaled_ls(
    namespace, schedule, corrections
):
    problem, x0 = _ls_problem(namespace)
    result = solve(
        problem,
        x0,
        options=Options(
            hessian="lbfgs",
            linsolve="dense",
            mu_schedule=schedule,
            corrections=corrections,
            max_iter=200,
        ),
    )

    # Pre-fix these runs hit MAX_ITER at f ~ 1e6-1e9 with μ locked at 1e2-1e5;
    # the realizable prescription makes the true optimum f* = 0.
    assert result.status is Status.OPTIMAL, (
        f"{schedule} x {corrections}: {result.status} "
        f"(f={float(result.objective):.3e}, last mu={result.history[-1].mu:.3e})"
    )
    assert float(result.objective) <= 1e-3
