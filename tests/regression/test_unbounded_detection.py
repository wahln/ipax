"""Regression: diverging iterates on an unbounded problem report UNBOUNDED.

S2MPJ sweep — INDEF (a deliberately indefinite, unbounded-below objective) drove
the objective monotonically to ``-1.3e155`` while the KKT residual never fell, then
reported ``NUMERICAL_ERROR`` only once the runaway iterate overflowed to non-finite.
That is a misleading verdict: the solve *correctly* found no (nonexistent) minimum.
``Status.UNBOUNDED`` existed in the enum but was never emitted — there was no
divergence test. This adds an IPOPT-style ``diverging_iterates_tol`` check and
verifies it fires on an unbounded problem without misfiring on a normal one.
"""

from __future__ import annotations

from ipax import Options, Status, solve
from ipax.problem.base import Problem
from tests._helpers import array


class _ConcaveUnbounded(Problem):
    """``min -½‖x‖²`` with free variables — unbounded below, no finite optimum.

    L-BFGS keeps a PD Hessian (``ξI``), so the step grows the iterate geometrically
    and ``‖x‖`` diverges past any threshold within a handful of iterations.
    """

    def __init__(self, xp) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        return -0.5 * self.xp.sum(x * x)

    def gradient(self, x):
        return -x


class _BoundedQuadratic(Problem):
    """``min ½‖x-1‖²`` — a well-posed convex problem with optimum ``x=1``."""

    def __init__(self, xp) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        d = x - 1.0
        return 0.5 * self.xp.sum(d * d)

    def gradient(self, x):
        return x - 1.0


def test_unbounded_problem_reports_unbounded(namespace):
    result = solve(
        _ConcaveUnbounded(namespace),
        array(namespace, [1.0, -1.0]),
        options=Options(hessian="lbfgs", linsolve="dense", max_iter=2000),
    )
    assert result.status is Status.UNBOUNDED, f"got {result.status}"
    assert not result.success


def test_well_posed_problem_is_not_flagged_unbounded(namespace):
    result = solve(
        _BoundedQuadratic(namespace),
        array(namespace, [5.0, -3.0]),
        options=Options(hessian="lbfgs", linsolve="dense"),
    )
    assert result.status is Status.OPTIMAL


class _FarOptimum(Problem):
    """``min ½((x − 2e20)/1e10)²`` — bounded below with an astronomical optimum.

    The iterate legitimately crosses any ‖x‖ divergence threshold on its way
    to ``x* = 2e20`` while the objective stays bounded (positive, → 0). The
    S2MPJ KOEBHELB anatomy in miniature: iterate-norm-only divergence
    detection misreported such runs as UNBOUNDED (v5–v10 sweeps,
    exact/dense+sparse) even though every other route proved a finite optimum
    exists.
    """

    def __init__(self, xp) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 1

    def objective(self, x):
        d = (x - 2.0e20) / 1.0e10
        return 0.5 * self.xp.sum(d * d)

    def gradient(self, x):
        return (x - 2.0e20) / 1.0e20


def test_far_optimum_is_not_misreported_unbounded(namespace):
    # Bounded-below objective, iterates past the ‖x‖ threshold: the two-signal
    # test (norm AND diverged objective) must keep solving instead of claiming
    # UNBOUNDED from a bounded problem.
    result = solve(
        _FarOptimum(namespace),
        array(namespace, [0.0]),
        options=Options(hessian="lbfgs", linsolve="dense", max_iter=500),
    )
    assert result.status is not Status.UNBOUNDED
    # The solve genuinely travelled past the divergence threshold ...
    assert float(result.x[0]) > 1e20
    # ... and minimized the bounded objective there.
    assert float(result.objective) < 1e12  # ½(d/1e10)² scale: far below f(x0)=2e20


def test_diverging_iterates_tol_none_disables_detection(namespace):
    # With detection off, the unbounded solve runs out its iteration budget
    # (or fails numerically) instead of reporting UNBOUNDED.
    result = solve(
        _ConcaveUnbounded(namespace),
        array(namespace, [1.0, -1.0]),
        options=Options(
            hessian="lbfgs",
            linsolve="dense",
            max_iter=200,
            diverging_iterates_tol=None,
        ),
    )
    assert result.status is not Status.UNBOUNDED
