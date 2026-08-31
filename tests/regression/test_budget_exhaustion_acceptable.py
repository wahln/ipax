"""Regression: budget exhaustion at a near-optimal iterate reports ACCEPTABLE.

S2MPJ budget-cluster audit (2026-07, Task 5): DIAMON2DLS oscillates at the
acceptable KKT level (best iterate 6.7e-7, acceptable 1e-6) without ever
holding it for the acceptable-iter window, then reported MAX_TIME — while a
*stall* at the same quality reports ACCEPTABLE through the relaxed-tolerance
salvage. The driver already returns the best accepted iterate on failure
statuses; when that returned iterate satisfies the relaxed KKT tolerances
(1e2 × the optimality tolerances, IPOPT ``acceptable_tol``), running out of
iterations or clock must report the same ACCEPTABLE the other salvage paths
do.

The test self-calibrates: a reference 3-iteration run reads the best
iterate's dual infeasibility, then reruns with ``dual_inf_tol`` placed so
that value sits between the exact and the relaxed tolerance.
"""

from __future__ import annotations

from ipax import Options, Status, solve
from ipax.options import LineSearchOptions, OptimalityConditionOptions
from ipax.testing.problems import BoundConstrainedQP
from tests._helpers import array


def _reference_best_dual(namespace) -> float:
    problem = BoundConstrainedQP(namespace)
    result = solve(
        problem,
        array(namespace, [0.25, 0.75]),
        # Plain halving pins the calibration trajectory: under the
        # interpolating backtrack this QP's best iterate reaches an exactly
        # zero dual infeasibility within the 3-iteration budget on some
        # backends, and the scenario (best iterate near but not at the
        # tolerance) needs a nonzero value to place the tolerances around.
        # The salvage logic under test is orthogonal to the backtracking rule.
        options=Options(
            hessian="exact",
            linsolve="dense",
            max_iter=3,
            line_search=LineSearchOptions(backtrack_interpolation=False),
        ),
    )
    assert result.status is Status.MAX_ITER
    best = min(result.history, key=lambda r: r.kkt_error)
    assert best.dual_infeasibility > 0.0
    return float(best.dual_infeasibility)


def _solve_with_dual_tol(namespace, dual_tol: float):
    # Loose viol/compl keep the decision on the dual component alone; the
    # trajectory is tolerance-independent, so the reference calibration holds.
    return solve(
        BoundConstrainedQP(namespace),
        array(namespace, [0.25, 0.75]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            max_iter=3,
            line_search=LineSearchOptions(backtrack_interpolation=False),
            optimality=OptimalityConditionOptions(
                dual_inf_tol=dual_tol,
                constr_viol_tol=1e3,
                compl_inf_tol=1e3,
            ),
        ),
    )


def test_max_iter_within_relaxed_tolerance_reports_acceptable(namespace):
    best_dual = _reference_best_dual(namespace)
    result = _solve_with_dual_tol(namespace, best_dual / 10.0)

    assert result.status is Status.ACCEPTABLE
    assert result.success
    assert "acceptable" in result.message
    assert result.dual_infeasibility <= 1e2 * (best_dual / 10.0)


def test_max_iter_far_from_tolerance_stays_max_iter(namespace):
    best_dual = _reference_best_dual(namespace)
    result = _solve_with_dual_tol(namespace, best_dual / 1e5)

    assert result.status is Status.MAX_ITER
