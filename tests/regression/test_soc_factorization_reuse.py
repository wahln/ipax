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

"""Regression: SOC reuses the step's factorization at unregularized steps.

At a ``δ_w = 0`` step the retained matrix is exactly Wächter & Biegler
2006's SOC choice (§2.4, eq. (26): "the same matrix as in (13)... to avoid
additional matrix factorizations"), yet ipax's SOC closure re-entered
``_solve_step`` — rebuilding the operator and re-running the δ_w ladder, a
fresh factorization per round. This pins the re-solve fast path on a
problem whose SOC iterations are all unregularized (HS71): no
``_solve_step`` re-entry during the line search, and the reuse helper
actually running there. (At ``δ_w > 0`` steps SOC deliberately keeps the
fresh δ_w = 0 solve on *direct* routes — reusing the inflated matrix
rerouted ZAMB2/ACOPP30/TWIRIMD1 onto restoration-heavy trajectories, measured
2026-09-02, and a fresh factorization per rung is cheap there — so the
in-search ladder stays legitimate on such iterations; HS71 has none. Iterative
routes reuse the step's system regardless: a fresh solve is a full Krylov
ladder per round, which timed DRUGDIS/DALLASS/NET1 out in the v29 sweep. The
driver asks the solver which it is through the optional ``is_direct`` hook.)
"""

from __future__ import annotations

import pytest

from ipax import Options, Status, solve
from ipax.backend.operators import Dense
from ipax.ipm.driver import IPMDriver
from ipax.ipm.filter_ls import FilterLineSearch
from ipax.linalg.dense import DenseSolver
from ipax.linalg.solver import LinearSolveError
from ipax.options import RegularizationOptions
from ipax.problem.base import Problem
from ipax.testing.problems import HS71
from tests._helpers import array


@pytest.mark.parametrize("linsolve", ["dense", "krylov"])
def test_soc_solves_reuse_the_step_factorization(namespace, linsolve, monkeypatch):
    in_search = {"active": False}
    soc_invoked = {"n": 0}
    reused = {"n": 0}

    orig_reuse = IPMDriver._resolve_reused_factorization

    def counting_reuse(self, *args, **kwargs):
        if in_search["active"]:
            reused["n"] += 1
        return orig_reuse(self, *args, **kwargs)

    orig_search = FilterLineSearch.search

    def marked_search(self, *args, **kwargs):
        # Count SOC *invocations* (not acceptances): the guard below is
        # non-vacuous as long as the soc callback ran at all, even if a
        # future tuning change made its corrected points get rejected.
        inner_soc = kwargs.get("soc")
        if inner_soc is not None:

            def counting_soc(alpha, _inner=inner_soc):
                soc_invoked["n"] += 1
                return _inner(alpha)

            kwargs["soc"] = counting_soc
        in_search["active"] = True
        try:
            return orig_search(self, *args, **kwargs)
        finally:
            in_search["active"] = False

    orig_step = IPMDriver._solve_step

    def guarded_step(self, *args, **kwargs):
        # HS71's SOC iterations are all unregularized (δ_w = 0), so every SOC
        # solve must be a re-solve on the retained factorization — never a
        # rebuild through the δ_w ladder. (On a problem with δ_w > 0 SOC
        # iterations the ladder is legitimate; see the module docstring.)
        assert not in_search["active"], (
            "SOC re-entered _solve_step (fresh operator + δ_w ladder) at an "
            "unregularized step instead of reusing the step's factorization "
            "(W&B 2006 §2.4, eq. (26))"
        )
        return orig_step(self, *args, **kwargs)

    monkeypatch.setattr(FilterLineSearch, "search", marked_search)
    monkeypatch.setattr(IPMDriver, "_solve_step", guarded_step)
    monkeypatch.setattr(IPMDriver, "_resolve_reused_factorization", counting_reuse)

    result = solve(
        HS71(namespace),
        array(namespace, [1.0, 5.0, 5.0, 1.0]),
        options=Options(linsolve=linsolve),
    )

    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    # Non-vacuous: HS71 under defaults invokes SOC several times, so the
    # guard above actually saw SOC solves — and they took the reuse path.
    assert soc_invoked["n"] > 0
    assert reused["n"] > 0


@pytest.mark.parametrize("linsolve", ["dense", "krylov"])
def test_soc_gate_follows_a_regularized_fallback_inside_the_soc_loop(
    namespace, monkeypatch, linsolve
):
    """After round 1's re-solve fails and its fallback ladder ends at δ_w > 0,
    the later rounds reuse *that* factorization instead of re-climbing the
    ladder per round.

    The ladder established that δ_w = 0 does not solve this rhs family, so a
    fresh ladder per round only repeats the same rungs. The dense arm is the
    one that isolates this clause; on the iterative arm a failed re-solve
    must *skip* the correction — no ``_solve_step`` ladder inside SOC at all
    (v29 sweep: DRUGDIS's 233 in-SOC ladders cost 120 s). The step's own
    regularized matrix on a direct route is a different matter —
    ``test_direct_soc_never_reuses_the_steps_regularized_factorization``.
    """
    state = {"in_search": False, "done": False, "force_floor": False}
    events: list[str] = []  # KKT-solve events inside the first SOC search
    orig_reuse = IPMDriver._resolve_reused_factorization

    def failing_first_reuse(self, *args, **kwargs):
        if state["in_search"] and not state["done"]:
            events.append("reuse")
            if len(events) == 1:
                state["force_floor"] = True
                return None  # simulate a failed re-solve on the SOC rhs
        return orig_reuse(self, *args, **kwargs)

    orig_step = IPMDriver._solve_step

    def regularizing_step(self, *args, **kwargs):
        if state["force_floor"]:
            # The fallback ladder ends at δ_w = 1e-4: the retained factorization
            # is no longer the unregularized matrix.
            state["force_floor"] = False
            args = (*args[:9], 1e-4)
        out = orig_step(self, *args, **kwargs)
        if state["in_search"] and not state["done"]:
            events.append(f"step:{out[2]:g}")
        return out

    orig_search = FilterLineSearch.search

    def marked_search(self, *args, **kwargs):
        if state["done"]:
            return orig_search(self, *args, **kwargs)
        state["in_search"] = True
        try:
            return orig_search(self, *args, **kwargs)
        finally:
            state["in_search"] = False
            if events:
                state["done"] = True

    monkeypatch.setattr(IPMDriver, "_resolve_reused_factorization", failing_first_reuse)
    monkeypatch.setattr(IPMDriver, "_solve_step", regularizing_step)
    monkeypatch.setattr(FilterLineSearch, "search", marked_search)

    result = solve(
        HS71(namespace),
        array(namespace, [1.0, 5.0, 5.0, 1.0]),
        options=Options(linsolve=linsolve),
    )

    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    if linsolve == "dense":
        # Round 1: the re-solve failed and the fallback ended at δ_w = 1e-4;
        # the remaining rounds re-solve that own factorization (max_soc = 4).
        assert events[:5] == ["reuse", "step:0.0001", "reuse", "reuse", "reuse"]
    else:
        # The failed round-1 re-solve ends this correction; later SOC
        # invocations in the same search re-solve the step's system again.
        assert events[0] == "reuse"
        assert not any(e.startswith("step:") for e in events)


class _EqualityOnly(Problem):
    """``min ‖x‖²`` s.t. ``x0 + x1 = 1`` — a saddle with a one-row Jacobian."""

    def __init__(self, xp):
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        return self.xp.sum(x * x)

    def gradient(self, x):
        return 2.0 * x

    def eq_constraints(self, x):
        return self.xp.stack((x[0] + x[1] - 1.0,))

    def eq_jacobian(self, x):
        del x
        return array(self.xp, [[1.0, 1.0]])


class _FailsUntilDualPhase:
    """Raise on every saddle whose δ_c is still the iteration's base value.

    Drives ``_solve_step`` through phase 1 (δ_w alone) into phase 2, where
    δ_w is reset to 0 and δ_c escalated — the retained factorization is then
    regularized even though the returned δ_w reads 0.
    """

    def __init__(self, base_delta_c: float) -> None:
        self._base = base_delta_c
        self._inner = DenseSolver()

    def factor(self, operator):
        if operator._delta_c <= self._base:
            raise LinearSolveError("synthetic saddle failure")
        self._inner.factor(operator)

    def solve(self, rhs):
        return self._inner.solve(rhs)


def _equality_driver(xp, solver, **regularization):
    driver = IPMDriver(
        _EqualityOnly(xp),
        xp=xp,
        solver=solver,
        options=Options(
            hessian="lbfgs",
            linsolve="dense",
            regularization=RegularizationOptions(**regularization),
        ),
        lower=None,
        upper=None,
        has_ineq=False,
        has_eq=True,
    )
    driver._has_lower = driver._has_upper = False  # set by run(); no bounds here
    return driver


def _saddle_operands(xp):
    dtype = xp.float64
    return {
        "w": Dense(xp.eye(2, dtype=dtype)),
        "sigma_x": xp.zeros((2,), dtype=dtype),
        "sigma_s": xp.zeros((0,), dtype=dtype),
        "ineq_jac": Dense(xp.zeros((0, 2), dtype=dtype)),
        "rhs_x": xp.ones((2,), dtype=dtype),
        "eq_jac": Dense(array(xp, [[1.0, 1.0]])),
        "m_eq": 1,
        "r_y": xp.zeros((1,), dtype=dtype),
        "delta_c": 1e-8,
    }


def test_step_solve_records_whether_the_retained_factorization_is_unregularized(
    namespace,
):
    xp = namespace
    ops = _saddle_operands(xp)

    driver = _equality_driver(xp, DenseSolver())
    _, _, delta_w, ok = driver._solve_step(**ops)
    assert ok and delta_w == 0.0
    assert driver._factor_unregularized

    _, _, delta_w, ok = driver._solve_step(**ops, delta_w_floor=1e-4)
    assert ok and delta_w == 1e-4
    assert not driver._factor_unregularized


class _AlwaysFails:
    def factor(self, operator):
        raise LinearSolveError("synthetic: every rung fails")

    def solve(self, rhs):  # pragma: no cover - factor always raises first
        raise LinearSolveError("unreachable")

    def is_direct(self) -> bool:
        return False


def test_step_solve_records_a_failed_ladder(namespace):
    """A ladder that ends without a usable factorization is recorded as
    *failed*, distinct from merely regularized: the iterative-route SOC reuse
    must not re-solve a system whose every rung just failed. A later success
    clears it."""
    xp = namespace
    ops = _saddle_operands(xp)
    driver = _equality_driver(xp, _AlwaysFails())
    assert not driver._solver_is_direct

    _, _, _, ok = driver._solve_step(**ops)
    assert not ok
    assert driver._factor_failed
    assert not driver._factor_unregularized

    driver = _equality_driver(xp, DenseSolver())
    _, _, _, ok = driver._solve_step(**ops)
    assert ok
    assert not driver._factor_failed


def test_a_delta_c_escalated_solve_is_not_an_unregularized_factorization(namespace):
    """δ_w = 0 is not proof the retained matrix is the step's unregularized
    saddle: phase 2 of the ladder resets δ_w while escalating δ_c."""
    xp = namespace
    ops = _saddle_operands(xp)
    # δ_w reaches the dual-phase trigger after one rung (δ_w_init = 1e-6).
    driver = _equality_driver(
        xp, _FailsUntilDualPhase(ops["delta_c"]), delta_c_trigger=1e-6
    )

    _, _, delta_w, ok = driver._solve_step(**ops)

    assert ok
    assert delta_w == 0.0  # the trap: the returned δ_w alone looks unregularized
    assert not driver._factor_unregularized


def _regularized_step_soc_events(namespace, monkeypatch, linsolve: str) -> list[str]:
    """KKT-solve events inside the first SOC search when every *step* solve
    is forced to end at δ_w = 1e-4 (the retained factorization is regularized)."""
    state = {"in_search": False, "done": False}
    events: list[str] = []
    orig_reuse = IPMDriver._resolve_reused_factorization
    orig_step = IPMDriver._solve_step
    orig_search = FilterLineSearch.search

    def recording_reuse(self, *args, **kwargs):
        if state["in_search"] and not state["done"]:
            events.append("reuse")
        return orig_reuse(self, *args, **kwargs)

    def regularized_main_step(self, *args, **kwargs):
        if not state["in_search"]:
            args = (*args[:9], 1e-4)  # every *step* solve ends at δ_w = 1e-4
        out = orig_step(self, *args, **kwargs)
        if state["in_search"] and not state["done"]:
            events.append(f"step:{out[2]:g}")
        return out

    def marked_search(self, *args, **kwargs):
        if state["done"]:
            return orig_search(self, *args, **kwargs)
        state["in_search"] = True
        try:
            return orig_search(self, *args, **kwargs)
        finally:
            state["in_search"] = False
            if events:
                state["done"] = True

    monkeypatch.setattr(IPMDriver, "_resolve_reused_factorization", recording_reuse)
    monkeypatch.setattr(IPMDriver, "_solve_step", regularized_main_step)
    monkeypatch.setattr(FilterLineSearch, "search", marked_search)

    result = solve(
        HS71(namespace),
        array(namespace, [1.0, 5.0, 5.0, 1.0]),
        options=Options(linsolve=linsolve),
    )

    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    assert events, "the first search never ran a second-order correction"
    return events


def test_direct_soc_never_reuses_the_steps_regularized_factorization(
    namespace, monkeypatch
):
    """On a direct route a δ_w > 0 *step* factorization is not reused for the
    first correction (the measured ZAMB2/ACOPP30 reroute; a fresh factorization
    per rung is cheap); the fresh round-1 solve then leaves an unregularized
    factorization that the later rounds reuse."""
    events = _regularized_step_soc_events(namespace, monkeypatch, "dense")
    assert events[0] == "step:0"  # round 1 is fresh, not a reuse of δ_w = 1e-4
    assert all(e == "reuse" for e in events[1:4])


def test_iterative_soc_reuses_the_steps_regularized_factorization(
    namespace, monkeypatch
):
    """On an iterative route every correction re-solves the step's retained
    system even at δ_w > 0 (Wächter & Biegler 2006 eq. (26) verbatim): a
    fresh δ_w = 0 solve there is a full Krylov ladder per SOC round — v29
    sweep, DRUGDIS 21 s (reuse) vs `max_time` (fresh), DALLASS/NET1 alike."""
    events = _regularized_step_soc_events(namespace, monkeypatch, "krylov")
    assert events[:4] == ["reuse"] * 4


class _NoKindSolver:
    """A third-party solver that predates ``is_direct``: treated as direct."""

    def __init__(self) -> None:
        self._inner = DenseSolver()

    def factor(self, operator):
        self._inner.factor(operator)

    def solve(self, rhs):
        return self._inner.solve(rhs)


def test_builtin_solvers_report_their_kind():
    from ipax.linalg.krylov import KrylovSolver
    from ipax.linalg.sparse import SparseDirectSolver
    from ipax.options import KrylovOptions

    assert DenseSolver().is_direct() is True
    assert SparseDirectSolver().is_direct() is True
    assert KrylovSolver(KrylovOptions()).is_direct() is False


def test_driver_defaults_a_solver_without_is_direct_to_the_direct_policy(namespace):
    assert _equality_driver(namespace, DenseSolver())._solver_is_direct is True
    assert _equality_driver(namespace, _NoKindSolver())._solver_is_direct is True
