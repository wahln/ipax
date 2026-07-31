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

"""A globalization failure at a feasible point must never enter restoration.

Restoration cannot move an already-feasible iterate — it exits immediately at
the same ``x`` — so the driver repairs the barrier state instead. That guard was
written for HS101, which has inequality constraints and no equalities, and its
condition therefore required ``m > 0``: re-centering means re-flooring *slacks*,
and without inequalities there are none.

But the harm being prevented is entering a **no-op restoration**, which does not
depend on slacks existing. On an equality-only problem the guard never fired, so
the driver did exactly what its own comment forbids: entered restoration, got
the identical point back, resumed with stale barrier state, re-derived the same
rejected direction, and repeated until the stall detector ended the run.

Measured on S2MPJ before the fix (2026-07-29): ``METHANL8``, ``DRCAVTY1`` and
``CYCLOOCF`` each entered restoration **26 times, every one exiting FEASIBLE**,
with ``x`` unchanged in 25 of the 26 — reported as ``stalled`` at a violation
IPOPT drives to ~1e-11 in a handful of iterations. Every such stopping point had
``‖Aᵀc‖ / ‖c‖`` between 1 and 56, so feasibility descent was still available:
these were not local minima of the infeasibility.
"""

from __future__ import annotations

import numpy as np
import pytest

import ipax.ipm.driver as driver_module
from ipax.ipm.driver import IPMDriver, _RestorationState
from ipax.ipm.filter_ls import Filter
from ipax.ipm.restoration import RestorationExit, feasible_theta_tol
from ipax.linalg.dense import DenseSolver
from ipax.options import Options
from ipax.problem.base import Problem
from ipax.result import IterationRecord


class _WithInequality(Problem):
    """The HS101 shape: an inequality alongside the equality, so ``m > 0``."""

    n_vars = 2

    def objective(self, x):
        return 0.0 * x[0]

    def gradient(self, x):
        return 0.0 * x

    def eq_constraints(self, x):
        return np.asarray([x[0] ** 2 + x[1] ** 2 - 1.0])

    def eq_jacobian(self, x):
        return np.asarray([[2.0 * x[0], 2.0 * x[1]]])

    def ineq_constraints(self, x):
        return np.asarray([-x[0], -x[1], x[0] - 10.0])

    def ineq_jacobian(self, x):
        return np.asarray([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]])


class _EqualityOnly(Problem):
    """``min 0`` subject to one equality — no inequality constraints at all."""

    n_vars = 2

    def objective(self, x):
        return 0.0 * x[0]

    def gradient(self, x):
        return 0.0 * x

    def eq_constraints(self, x):
        return np.asarray([x[0] ** 2 + x[1] ** 2 - 1.0])

    def eq_jacobian(self, x):
        return np.asarray([[2.0 * x[0], 2.0 * x[1]]])


def _driver(has_ineq: bool) -> IPMDriver:
    return IPMDriver(
        _WithInequality() if has_ineq else _EqualityOnly(),
        xp=np,
        solver=DenseSolver(),
        options=Options(hessian="lbfgs", linsolve="dense"),
        lower=None,
        upper=None,
        has_ineq=has_ineq,
        has_eq=True,
    )


def _failure_at_a_feasible_point(driver: IPMDriver, *, m: int):
    """Drive one globalization failure at a feasible, far-from-optimal iterate."""
    theta0 = 0.0  # feasible: restoration provably cannot move this point
    assert theta0 <= feasible_theta_tol(1e-8)
    record = IterationRecord(
        iteration=7,
        objective=0.0,
        mu=0.1,
        theta=theta0,
        # Far from optimal, so the "stalled within relaxed tolerance" branch
        # ahead of the guard does not claim this failure first.
        kkt_error=1e4,
        alpha_primal=0.0,
        alpha_dual=1.0,
        regularization=0.0,
        dual_infeasibility=1e4,
        primal_infeasibility=0.0,
        complementarity=0.0,
    )
    x = np.asarray([1.0, 0.0])
    empty = np.zeros(m)
    # Slacks that make g(x) + s = 0 at this x, so the point is feasible for the
    # inequality block too (the guard measures ‖(c, g+s)‖_∞).
    slacks = -np.asarray(driver._problem.ineq_constraints(x)) if m else empty
    return driver._handle_restoration(
        x=x,
        s=slacks,
        y_ineq=empty,
        y_eq=np.zeros(1),
        g=empty,
        mu=0.1,
        m=m,
        m_eq=1,
        mask_l=np.zeros(2, dtype=bool),
        mask_u=np.zeros(2, dtype=bool),
        lower_safe=np.full(2, -1e20),
        upper_safe=np.full(2, 1e20),
        theta0=theta0,
        theta_inf=theta0,
        phi0=0.0,
        record=record,
        filt=Filter(),
        theta_best=theta0,
        x_restore_anchor=x,
        rstate=_RestorationState(),
        it=7,
    )


def test_feasible_point_failure_skips_restoration_without_inequalities(monkeypatch):
    # The regression: with m == 0 the guard used to fall through to restoration,
    # which returns the identical point and livelocks the driver.
    called: list[object] = []

    def _spy(*args, **kwargs):
        called.append(kwargs or args)
        raise AssertionError("restoration must not run at a feasible point")

    monkeypatch.setattr(driver_module, "restore", _spy)
    outcome = _failure_at_a_feasible_point(_driver(has_ineq=False), m=0)

    assert called == []
    assert outcome.resume is True  # repaired in place, not restored


def test_feasible_point_failure_still_skips_restoration_with_inequalities(monkeypatch):
    # The HS101-class path the guard was originally written for must be
    # unchanged: the fix widens the guard, it does not move it.
    monkeypatch.setattr(
        driver_module,
        "restore",
        lambda *a, **k: pytest.fail("restoration must not run at a feasible point"),
    )
    outcome = _failure_at_a_feasible_point(_driver(has_ineq=True), m=3)

    assert outcome.resume is True


def test_an_infeasible_failure_still_restores(monkeypatch):
    # The guard must stay narrow: a failure at an *infeasible* point is exactly
    # what restoration is for, inequalities or not.
    seen: list[str] = []

    def _spy(*args, **kwargs):
        seen.append("restored")
        return np.asarray([1.0, 0.0]), np.zeros(0), RestorationExit.STATIONARY

    monkeypatch.setattr(driver_module, "restore", _spy)
    driver = _driver(has_ineq=False)
    record = IterationRecord(
        iteration=3,
        objective=0.0,
        mu=0.1,
        theta=5.0,
        kkt_error=1e4,
        alpha_primal=0.0,
        alpha_dual=1.0,
        regularization=0.0,
        dual_infeasibility=1e4,
        primal_infeasibility=5.0,
        complementarity=0.0,
    )
    x = np.asarray([3.0, 4.0])
    empty = np.zeros(0)
    driver._handle_restoration(
        x=x,
        s=empty,
        y_ineq=empty,
        y_eq=np.zeros(1),
        g=empty,
        mu=0.1,
        m=0,
        m_eq=1,
        mask_l=np.zeros(2, dtype=bool),
        mask_u=np.zeros(2, dtype=bool),
        lower_safe=np.full(2, -1e20),
        upper_safe=np.full(2, 1e20),
        theta0=5.0,  # infeasible
        theta_inf=5.0,
        phi0=0.0,
        record=record,
        filt=Filter(),
        theta_best=5.0,
        x_restore_anchor=x,
        rstate=_RestorationState(),
        it=3,
    )
    assert seen == ["restored"]


class _BoundsOnly(Problem):
    """No constraints at all — only bounds, so theta is identically zero."""

    n_vars = 2

    def objective(self, x):
        return float(x[0] ** 2 + x[1] ** 2)

    def gradient(self, x):
        return 2.0 * np.asarray(x)

    def bounds(self):
        return np.full(2, -5.0), np.full(2, 5.0)


def test_a_problem_with_no_constraints_still_goes_to_restoration(monkeypatch):
    # theta is identically zero without constraints, so an unguarded
    # feasible-point test fires on every globalization failure — and there is
    # nothing to re-center, no slacks and no equality duals. The branch would
    # only pick between two no-ops, and picking the new one cost S2MPJ
    # HADAMALS its convergence (optimal -> stalled) on two routes.
    seen: list[str] = []

    def _spy(*args, **kwargs):
        seen.append("restored")
        return np.asarray([0.5, 0.5]), np.zeros(0), RestorationExit.STATIONARY

    monkeypatch.setattr(driver_module, "restore", _spy)
    driver = IPMDriver(
        _BoundsOnly(),
        xp=np,
        solver=DenseSolver(),
        options=Options(hessian="lbfgs", linsolve="dense"),
        lower=np.full(2, -5.0),
        upper=np.full(2, 5.0),
        has_ineq=False,
        has_eq=False,
    )
    record = IterationRecord(
        iteration=5,
        objective=1.0,
        mu=0.1,
        theta=0.0,
        kkt_error=1e4,
        alpha_primal=0.0,
        alpha_dual=1.0,
        regularization=0.0,
        dual_infeasibility=1e4,
        primal_infeasibility=0.0,
        complementarity=0.0,
    )
    empty = np.zeros(0)
    driver._handle_restoration(
        x=np.asarray([1.0, 1.0]),
        s=empty,
        y_ineq=empty,
        y_eq=empty,
        g=empty,
        mu=0.1,
        m=0,
        m_eq=0,
        mask_l=np.ones(2, dtype=bool),
        mask_u=np.ones(2, dtype=bool),
        lower_safe=np.full(2, -5.0),
        upper_safe=np.full(2, 5.0),
        theta0=0.0,
        theta_inf=0.0,
        phi0=1.0,
        record=record,
        filt=Filter(),
        theta_best=0.0,
        x_restore_anchor=np.asarray([1.0, 1.0]),
        rstate=_RestorationState(),
        it=5,
    )
    assert seen == ["restored"]
