"""Regression: the IPM loop's device->host sync count per iteration is bounded.

Every ``float()``/``bool()`` of a device scalar is a host sync on a GPU
backend; the driver's merit, feasibility, KKT-residual and fraction-to-boundary
helpers used to reduce block by block — ~15 syncs per iteration on an
unconstrained problem, ~60 on one with equalities, inequalities and bounds
(HS71) — where one reduction of the stacked block scalars does the job. This
pins the per-iteration budget on a backend whose array type can be patched
(Torch); on NumPy counting is unavailable and the test skips.
"""

from __future__ import annotations

import pytest

from benchmarks.harness import _ScalarSyncCounter
from ipax import Options, Status, solve
from ipax.testing.problems import HS43, HS71, BoundConstrainedQP
from tests._helpers import array
from tests.regression.test_callback_evaluation_counts import _Rosenbrock


def _bulk_transfer_available(namespace) -> bool:
    """Whether ``read_scalars`` can batch on this backend (see backend/scalars)."""
    try:
        from array_api_compat import to_device

        to_device(namespace.zeros(2), "cpu")
        return True
    except Exception:
        return False


@pytest.mark.parametrize(
    ("make", "x0", "budget", "options"),
    [
        # Budgets tightened 2026-09-02 with the fused decision reads
        # (``backend/scalars.py``: one transfer for the KKT-residual block,
        # the θ/φ merit pair, and the L-BFGS curvature scalars; the L-BFGS
        # apply guard selects device-side). Measured on torch:
        # 10.1 / 10.5 / 18.7 / 15.2 — pre-fusion these ran 12.6 / 19.x /
        # 29.x / 24.2, so the budgets below fail if any fusion regresses.
        (lambda xp: _Rosenbrock(xp, 20), [-1.2, 1.0] * 10, 11.0, Options()),
        (HS43, [0.0, 0.0, 0.0, 0.0], 12.0, Options()),
        (HS71, [1.0, 5.0, 5.0, 1.0], 20.0, Options()),
        # Bound-only L-BFGS on the Krylov route: also pins that the direct
        # exact-inverse dispatch stays dispatched — re-wrapping it in CG
        # adds the loop's inner-product reads and busts the budget.
        (
            BoundConstrainedQP,
            [0.25, 0.75],
            16.0,
            Options(hessian="lbfgs", linsolve="krylov"),
        ),
    ],
    ids=[
        "rosenbrock-unconstrained",
        "HS43-ineq",
        "HS71-eq-ineq-bounds",
        "bound-only-lbfgs-krylov-direct",
    ],
)
def test_host_syncs_per_iteration_bounded(namespace, make, x0, budget, options):
    if not _bulk_transfer_available(namespace):
        pytest.skip(
            "no bulk device->host transfer: read_scalars falls back to exact "
            "per-element reads (k+1 counts), so the fused budgets do not apply"
        )
    problem = make(namespace)
    x = array(namespace, x0)
    with _ScalarSyncCounter(type(x)) as counter:
        result = solve(problem, x, options=options)
    if not counter.available:
        pytest.skip(f"scalar-sync counting unavailable for {type(x)!r}")
    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    per_iter = counter.count / max(1, result.n_iter)
    assert per_iter <= budget, f"{per_iter:.1f} host syncs/iter (budget {budget})"
