# Copyright 2026 Niklas Wahl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Terminal KKT certificate at the returned iterate (IPOPT-triage class B).

The v4 ipax-vs-IPOPT triage's class B — "reached the answer, won't certify" —
is runs that end STALLED/MAX_ITER *at* a point satisfying the acceptable KKT
test. The per-iteration test misses it because the tested iterate carries
multipliers that drifted away from the ones the point can justify: a
rank-deficient ``∇c`` under-determines ``y`` (S2MPJ ``NONSCOMPNE`` reports
KKT 6.8e-5 at a point whose least-squares multipliers give dual infeasibility
exactly 0), and initialized bound duals never re-converge once the line
search freezes (``WEEDS`` returns a best iterate at KKT 3.8e-7 — inside the
acceptable band — yet reports STALLED off the frozen 6.2e-6 tail iterate).

The fix is terminal-only: when a run ends STALLED / MAX_ITER / MAX_TIME, the
returned best iterate is re-judged with *candidate* multipliers — equality
duals from the least-squares estimate, inequality/bound duals dropped to the
zero the ``E_0`` complementarity test justifies at an interior point. If the
full residual set passes the acceptable-stopping tolerances the run reports
ACCEPTABLE and returns the certified multipliers. Exhibited duals can only
*upper-bound* the achievable dual infeasibility, so the certificate can
under-certify but never falsely certify — the negative controls below pin
that direction.
"""

from __future__ import annotations

import pytest

from ipax import FunctionProblem, Options, Status, WarmStart, solve
from ipax.ipm.filter_ls import FilterLineSearch, LineSearchResult
from ipax.options import AcceptableStoppingOptions
from tests._helpers import array


def _rank_deficient_feasibility(xp):
    """``min 0`` s.t. a duplicated equality — the NONSCOMPNE shape in miniature.

    The duplicate row makes ``∇c`` exactly rank 1 of 2, so the dual block is
    under-determined: any ``y`` with the same ``y_0 + y_1`` leaves the same
    stationarity residual, and nothing in the step system pulls a poisoned
    pair back toward the ``y* = 0`` that the zero objective forces.
    """

    def eq(x):
        r = x[0] ** 2 + x[1] - 2.0
        return xp.stack((r, r))

    def eq_jac(x):
        row = xp.stack((2.0 * x[0], 1.0 + 0.0 * x[0]))
        return xp.stack((row, row))

    return FunctionProblem(
        2,
        lambda x: 0.0 * x[0],
        gradient=lambda x: 0.0 * x,
        eq_constraints=eq,
        eq_jacobian=eq_jac,
    )


def _asymmetric_box_problem(xp):
    """``min ½‖x‖²`` on ``[-1, 10]²`` — the WEEDS shape in miniature.

    The asymmetric box keeps the μ-complementarity bound duals from
    cancelling in the stationarity residual, so a frozen iterate *near* the
    unconstrained optimum still reports a large dual infeasibility with the
    initialized ``z`` even though the gradient itself is inside the
    acceptable band.
    """
    return FunctionProblem(
        2,
        lambda x: 0.5 * xp.sum(x * x),
        gradient=lambda x: 1.0 * x,
        bounds=(array(xp, [-1.0, -1.0]), array(xp, [10.0, 10.0])),
        lagrangian_hessian=lambda x, y_eq, y_ineq, sigma=1.0: (
            sigma * xp.eye(2, dtype=x.dtype)
        ),
    )


def _poisoned(namespace) -> WarmStart:
    # In the duplicated-row null direction the pair is unobservable to the
    # step system; the sum 1e6 puts the stationarity residual at ~2e6.
    return WarmStart(y_eq=array(namespace, [5e5, 5e5]))


def _always_fail_line_search(monkeypatch):
    def always_fail(self, **kwargs):
        return LineSearchResult(self._o.alpha_min_frac, False, False, True)

    monkeypatch.setattr(FilterLineSearch, "search", always_fail)


def test_budget_exit_with_repairable_duals_reports_acceptable(namespace):
    # Feasible start (c(x0) = 0) carrying poisoned equality duals; the
    # one-iteration budget ends the run before any step could fix them.
    result = solve(
        _rank_deficient_feasibility(namespace),
        array(namespace, [1.0, 1.0]),
        options=Options(hessian="lbfgs", linsolve="dense", max_iter=1),
        warm_start=_poisoned(namespace),
    )

    assert result.status is Status.ACCEPTABLE, result.message
    assert result.success
    assert result.dual_infeasibility <= 1e-6
    assert result.primal_infeasibility <= 1e-6
    # The certificate's multipliers are the ones returned, not the poison.
    assert result.y_eq is not None
    y_max = max(abs(float(result.y_eq[0])), abs(float(result.y_eq[1])))
    assert y_max <= 1e-6


def test_certificate_declines_at_an_infeasible_point(namespace):
    # Same poisoned duals, but c(x0) = 3: the certificate's feasibility
    # component fails, so the honest budget status must survive.
    result = solve(
        _rank_deficient_feasibility(namespace),
        array(namespace, [2.0, 1.0]),
        options=Options(hessian="lbfgs", linsolve="dense", max_iter=1),
        warm_start=_poisoned(namespace),
    )

    assert result.status is Status.MAX_ITER


def test_stall_with_certifiable_gradient_reports_acceptable(namespace, monkeypatch):
    # Frozen at gradient 5e-7 — inside acceptable (1e-6), outside optimal
    # (1e-8) — while the initialized bound duals keep the *recorded* dual
    # infeasibility at ~9e-2. Pre-fix this reported STALLED (the in-loop
    # relaxed-tolerance check sees only the recorded multipliers).
    _always_fail_line_search(monkeypatch)
    result = solve(
        _asymmetric_box_problem(namespace),
        array(namespace, [5e-7, 5e-7]),
        options=Options(
            hessian="exact", linsolve="dense", max_iter=100, max_stall_iter=10
        ),
    )

    assert result.status is Status.ACCEPTABLE, result.message
    assert result.dual_infeasibility <= 1e-6
    assert result.n_iter < 50  # the stall detector ended the run, not the budget


def test_stall_at_a_non_stationary_point_stays_stalled(namespace, monkeypatch):
    # Gradient 0.5 at the frozen point: no multiplier choice can certify it.
    _always_fail_line_search(monkeypatch)
    result = solve(
        _asymmetric_box_problem(namespace),
        array(namespace, [0.5, 0.5]),
        options=Options(
            hessian="exact", linsolve="dense", max_iter=100, max_stall_iter=10
        ),
    )

    assert result.status is Status.STALLED


def test_stall_at_an_active_bound_stays_stalled(namespace, monkeypatch):
    # min Σ(x−2)² on [-1, 1]²: the optimum sits ON the bound with a genuine
    # z_U = 2. Dropping the bound duals to zero leaves the raw gradient ~2 in
    # the residual, so the certificate must decline — the zero-dual candidate
    # under-certifies rather than inventing a multiplier it cannot justify.
    xp = namespace
    problem = FunctionProblem(
        2,
        lambda x: xp.sum((x - 2.0) * (x - 2.0)),
        gradient=lambda x: 2.0 * (x - 2.0),
        bounds=(array(xp, [-1.0, -1.0]), array(xp, [1.0, 1.0])),
        lagrangian_hessian=lambda x, y_eq, y_ineq, sigma=1.0: (
            2.0 * sigma * xp.eye(2, dtype=x.dtype)
        ),
    )
    _always_fail_line_search(monkeypatch)
    result = solve(
        problem,
        array(namespace, [0.999, 0.999]),
        options=Options(
            hessian="exact", linsolve="dense", max_iter=100, max_stall_iter=10
        ),
    )

    assert result.status is Status.STALLED


def test_disabled_acceptable_stopping_disables_the_certificate(namespace):
    # "Acceptable" is what the caller is willing to accept; all-None means
    # they accept nothing short of optimal, and the certificate must respect
    # that instead of introducing a tolerance of its own.
    result = solve(
        _rank_deficient_feasibility(namespace),
        array(namespace, [1.0, 1.0]),
        options=Options(
            hessian="lbfgs",
            linsolve="dense",
            max_iter=1,
            acceptable=AcceptableStoppingOptions(
                dual_inf_tol=None, constr_viol_tol=None, compl_inf_tol=None
            ),
        ),
        warm_start=_poisoned(namespace),
    )

    assert result.status is Status.MAX_ITER


@pytest.mark.parametrize(
    "acceptable",
    [
        # With the candidate multipliers the complementarity residual is zero
        # BY CONSTRUCTION, and on a bound-only problem so is the primal one —
        # neither carries any information about stationarity. A legal
        # partial-None configuration enabling only an uninformative component
        # must therefore never certify (invariant-audit blocker: pre-fix,
        # both of these upgraded a point with raw gradient 5e-1 to
        # ACCEPTABLE/success).
        AcceptableStoppingOptions(
            dual_inf_tol=None, constr_viol_tol=None, compl_inf_tol=1e-6
        ),
        AcceptableStoppingOptions(
            dual_inf_tol=None, constr_viol_tol=1e-6, compl_inf_tol=None
        ),
    ],
)
def test_certificate_requires_the_dual_tolerance_enabled(
    namespace, monkeypatch, acceptable
):
    _always_fail_line_search(monkeypatch)
    result = solve(
        _asymmetric_box_problem(namespace),
        array(namespace, [0.5, 0.5]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            max_iter=100,
            max_stall_iter=10,
            acceptable=acceptable,
        ),
    )

    assert result.status is Status.STALLED


def _constant_objective_feasibility(xp):
    """The rank-deficient fixture with ``f ≡ 1`` — for the ``f_tol`` gate."""

    def eq(x):
        r = x[0] ** 2 + x[1] - 2.0
        return xp.stack((r, r))

    def eq_jac(x):
        row = xp.stack((2.0 * x[0], 1.0 + 0.0 * x[0]))
        return xp.stack((row, row))

    return FunctionProblem(
        2,
        lambda x: 1.0 + 0.0 * x[0],
        gradient=lambda x: 0.0 * x,
        eq_constraints=eq,
        eq_jacobian=eq_jac,
    )


def test_an_enabled_f_tol_is_honored_terminally(namespace):
    # The in-loop acceptable checker requires |f| <= f_tol when enabled; the
    # certificate must apply the same test instead of silently dropping it.
    # Here every residual passes (grad = 0 at a feasible point) but |f| = 1.
    def _solve(f_tol):
        return solve(
            _constant_objective_feasibility(namespace),
            array(namespace, [1.0, 1.0]),
            options=Options(
                hessian="lbfgs",
                linsolve="dense",
                max_iter=1,
                acceptable=AcceptableStoppingOptions(f_tol=f_tol),
            ),
            warm_start=_poisoned(namespace),
        )

    assert _solve(1e-30).status is Status.MAX_ITER
    assert _solve(2.0).status is Status.ACCEPTABLE


def test_max_time_exit_is_certified_too(namespace):
    # The docstring claims all three failure exits; MAX_TIME is the one the
    # other tests never produce.
    result = solve(
        _rank_deficient_feasibility(namespace),
        array(namespace, [1.0, 1.0]),
        options=Options(hessian="lbfgs", linsolve="dense", max_time=1e-9),
        warm_start=_poisoned(namespace),
    )

    assert result.status is Status.ACCEPTABLE, result.message
    assert "certificate" in result.message


def test_certificate_survives_a_raising_dual_estimator(namespace, monkeypatch):
    # A certificate must never be the thing that fails: an estimator error
    # declines it and the honest failure status survives.
    def raising(eq_jac, grad, *, xp, m_eq=None):
        raise RuntimeError("synthetic estimator failure")

    monkeypatch.setattr("ipax.ipm.driver.least_squares_duals", raising)
    result = solve(
        _rank_deficient_feasibility(namespace),
        array(namespace, [1.0, 1.0]),
        options=Options(hessian="lbfgs", linsolve="dense", max_iter=1),
        warm_start=_poisoned(namespace),
    )

    assert result.status is Status.MAX_ITER


def test_certificate_declines_a_non_finite_dual_estimate(namespace, monkeypatch):
    def non_finite(eq_jac, grad, *, xp, m_eq=None):
        return xp.full((2,), float("nan"), dtype=grad.dtype)

    monkeypatch.setattr("ipax.ipm.driver.least_squares_duals", non_finite)
    result = solve(
        _rank_deficient_feasibility(namespace),
        array(namespace, [1.0, 1.0]),
        options=Options(hessian="lbfgs", linsolve="dense", max_iter=1),
        warm_start=_poisoned(namespace),
    )

    assert result.status is Status.MAX_ITER


def test_stall_returning_a_best_iterate_within_relaxed_tol(namespace, monkeypatch):
    # The WEEDS shape with honest multipliers: one accepted step moves the
    # iterate from gradient 5e-7 (inside the relaxed band) to ~1e-5 (outside),
    # then the line search freezes. The in-loop stall check sees only the
    # frozen tail; the salvage swaps the best iterate back in, and the
    # relaxed-tolerance re-judge — not the certificate — must certify it.
    xp = namespace
    problem = FunctionProblem(
        1,
        lambda x: 5e-7 * x[0] + (3.8e7 / 3.0) * x[0] ** 3,
        gradient=lambda x: 5e-7 + 3.8e7 * x * x,
        lagrangian_hessian=lambda x, y_eq, y_ineq, sigma=1.0: (
            sigma * xp.eye(1, dtype=x.dtype)
        ),
    )
    calls = {"n": 0}

    def scripted(self, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return LineSearchResult(
                alpha=1.0, accepted=True, augment=False, restoration=False
            )
        return LineSearchResult(
            alpha=self._o.alpha_min_frac,
            accepted=False,
            augment=False,
            restoration=True,
        )

    monkeypatch.setattr(FilterLineSearch, "search", scripted)
    result = solve(
        problem,
        array(namespace, [0.0]),
        options=Options(
            hessian="exact", linsolve="dense", max_iter=100, max_stall_iter=10
        ),
    )

    assert result.status is Status.ACCEPTABLE, result.message
    assert "relaxed KKT tolerance" in result.message
    assert "certificate" not in result.message
    assert result.kkt_error <= 1e-6
    assert abs(float(result.x[0])) < 1e-12  # the best iterate is x0, not the tail


def test_enabled_relative_change_tolerance_declines_the_certificate(
    namespace, monkeypatch
):
    # ``f_rel_change_tol`` compares CONSECUTIVE iterates; it has no one-shot
    # terminal analogue, so a caller who enabled it has asked for a test the
    # certificate cannot evaluate. Declining is the honest answer — quietly
    # dropping the component would certify against a weaker acceptance rule
    # than the one the caller configured.
    _always_fail_line_search(monkeypatch)
    result = solve(
        _asymmetric_box_problem(namespace),
        array(namespace, [0.5, 0.5]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            max_iter=100,
            max_stall_iter=10,
            acceptable=AcceptableStoppingOptions(
                dual_inf_tol=1e6,  # loose enough that the point WOULD certify
                constr_viol_tol=None,
                compl_inf_tol=None,
                f_rel_change_tol=1e-8,
            ),
        ),
    )

    assert result.status is Status.STALLED


def test_a_failing_residual_declines_the_certificate_instead_of_raising(
    namespace, monkeypatch
):
    # The certificate's contract is that it can only ever *decline* — never
    # make a run worse. kkt_error() applies the Jacobians (rmatvec on a
    # possibly lazy or user-supplied operator), so it can fail for the same
    # reasons the gradient/Jacobian evaluation can. Outside a guard that would
    # turn an ordinary stall into an exception.
    import inspect

    from ipax.ipm.driver import IPMDriver

    _always_fail_line_search(monkeypatch)
    original = IPMDriver.kkt_error
    state = {"terminal": False}

    def failing(self, **kwargs):
        # Fail *only* the certificate's own residual. Several call sites pass
        # mu=0.0 (the in-loop convergence check among them), and those must
        # keep working or nothing would converge — so discriminate on the
        # caller, not the arguments.
        if any(f.function == "_terminal_certificate" for f in inspect.stack()[:4]):
            state["terminal"] = True
            raise RuntimeError("operator apply failed")
        return original(self, **kwargs)

    monkeypatch.setattr(IPMDriver, "kkt_error", failing)

    result = solve(
        _asymmetric_box_problem(namespace),
        array(namespace, [5e-7, 5e-7]),
        options=Options(
            hessian="exact", linsolve="dense", max_iter=100, max_stall_iter=10
        ),
    )

    assert state["terminal"], "the terminal certificate path must have been taken"
    # Declined, not crashed: the run still returns its ordinary stalled verdict.
    assert result.status is Status.STALLED, result.message
