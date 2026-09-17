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


@pytest.mark.parametrize(
    ("make", "x0", "budget", "options"),
    [
        (lambda xp: _Rosenbrock(xp, 20), [-1.2, 1.0] * 10, 13.0, Options()),
        (HS43, [0.0, 0.0, 0.0, 0.0], 21.0, Options()),
        (HS71, [1.0, 5.0, 5.0, 1.0], 31.0, Options()),
        # Bound-only L-BFGS on the Krylov route: the budget sits *between* the
        # direct exact-inverse dispatch (~22.8 syncs/iter) and the same solve
        # re-wrapped in CG (~25.5, both with the device-side apply guard), so
        # it pins that the direct route stays dispatched — the change's
        # device-efficiency property, not just its iteration counts.
        (
            BoundConstrainedQP,
            [0.25, 0.75],
            24.0,
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
    problem = make(namespace)
    x = array(namespace, x0)
    with _ScalarSyncCounter(type(x)) as counter:
        result = solve(problem, x, options=options)
    if not counter.available:
        pytest.skip(f"scalar-sync counting unavailable for {type(x)!r}")
    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    per_iter = counter.count / max(1, result.n_iter)
    assert per_iter <= budget, f"{per_iter:.1f} host syncs/iter (budget {budget})"
