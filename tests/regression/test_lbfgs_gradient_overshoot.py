"""Regression: the L-BFGS route recovers from a step that overshoots into a
region where the objective gradient is non-finite.

S2MPJ sweep Task 2 — residual ``numerical_error`` (RAT42LS). A quasi-Newton step
can land at a point where the objective and constraints (θ, φ) are still finite
but the *gradient* overflows to inf/NaN — e.g. RAT42LS's logistic
``p1/(1+exp(p2-p3·t))``, whose residual stays finite while its derivative
``exp/(1+exp)²`` becomes ``inf/inf = NaN``. The filter line search evaluates only
θ/φ, so it accepted such a step and the next KKT solve was poisoned, ending in a
``numerical_error``.

The fix threads a ``grad_finite(α)`` check into the line search on the L-BFGS
route: a trial whose gradient is non-finite is treated as unacceptable, so the
search backtracks to a damped step that stays in the finite region (and hands off
to restoration only if the whole ray is bad). This test reproduces the overshoot
with a steep objective whose gradient overflows past a threshold, and checks the
solver navigates around it to the optimum instead of failing.
"""

from __future__ import annotations

from ipax import Options, Status, solve
from ipax.problem.base import Problem
from tests._helpers import array, assert_scalar_close


class _SteepOvershoot(Problem):
    """``min ½K·x²`` (optimum ``x=0``) whose gradient overflows to ``+inf`` for
    ``x < -15``. From ``x0 = 10`` the steep identity-seed L-BFGS step overshoots
    far past the optimum into the non-finite region, so the line search must
    backtrack to make progress."""

    def __init__(self, xp) -> None:
        self.xp = xp
        self.K = 200.0

    @property
    def n_vars(self) -> int:
        return 1

    def objective(self, x):
        return 0.5 * self.K * x[0] * x[0]

    def gradient(self, x):
        xp = self.xp
        return xp.where(x < -15.0, xp.full_like(x, float("inf")), self.K * x)


def test_lbfgs_recovers_from_non_finite_gradient_overshoot(namespace):
    result = solve(
        _SteepOvershoot(namespace),
        array(namespace, [10.0]),
        options=Options(hessian="lbfgs", linsolve="dense", max_iter=100),
    )

    # Without the grad-finite backtracking the full step lands where the gradient
    # is +inf and the next KKT solve fails with a numerical_error; with it the
    # search damps the step and converges to the optimum at x = 0.
    assert result.status is Status.OPTIMAL, f"unexpected status {result.status}"
    assert_scalar_close(float(result.x[0]), 0.0, atol=1e-6)
