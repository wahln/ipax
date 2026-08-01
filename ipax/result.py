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

"""Solver result, status codes, iteration history, and derivative-source log."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipax.typing import Array


class Status(Enum):
    """Termination status."""

    OPTIMAL = "optimal"
    ACCEPTABLE = "acceptable"  # IPOPT-style acceptable-level termination
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    MAX_ITER = "max_iter"
    STALLED = "stalled"  # frozen iterates: no accepted step, KKT error constant
    RESTORATION_FAILED = "restoration_failed"
    NUMERICAL_ERROR = "numerical_error"
    STOPPED = "stopped"  # user iteration callback requested termination
    MAX_TIME = "max_time"

    @property
    def is_success(self) -> bool:
        """Whether this status represents a usable converged point."""
        return self in (
            Status.OPTIMAL,
            Status.ACCEPTABLE,
        )


@dataclass(frozen=True, slots=True)
class KKTResiduals:
    """Scaled components whose maximum is the solver's KKT error."""

    dual_infeasibility: float
    primal_infeasibility: float
    complementarity: float

    @property
    def error(self) -> float:
        """Return the aggregate scaled KKT infinity norm."""
        components = (
            self.dual_infeasibility,
            self.primal_infeasibility,
            self.complementarity,
        )
        if not all(math.isfinite(value) for value in components):
            return float("inf")
        return max(components)


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """One row of the iteration log.

    ``objective`` uses the original problem's units. ``mu``, ``theta``, and
    ``kkt_error`` and its three component residuals remain scaled-space
    algorithm diagnostics when gradient-based problem scaling is enabled.
    """

    iteration: int
    objective: float
    mu: float
    theta: float  # constraint violation ‖(c, g+s)‖₁
    kkt_error: float  # scaled KKT ∞-norm
    alpha_primal: float
    alpha_dual: float
    regularization: float  # δ_w applied this step
    problem_time: float = 0.0  # seconds in problem callbacks since prior row
    step_solve_time: float = 0.0  # seconds in step solves since prior row
    dual_infeasibility: float = float("inf")
    primal_infeasibility: float = float("inf")
    complementarity: float = float("inf")
    line_search_iters: int = 0  # backtracking trials in the search reaching this row
    restored: bool = False  # this iterate is the result of a restoration jump


@dataclass(frozen=True, slots=True)
class IterationInfo:
    """Read-only snapshot handed to a user iteration callback each iteration.

    ``record`` is the same :class:`IterationRecord` appended to
    ``Result.history``; the remaining fields expose the current primal/dual
    iterate so callbacks can plot progress or compute a custom stopping rule.
    Optional fields are ``None`` when the corresponding block is absent (no
    slacks/inequalities, no equalities, no bounds). Values use the original
    problem's units and array namespace. Treat arrays as read-only and copy
    before mutating.

    A callback returning a truthy value requests early termination; the solve
    then finishes with :attr:`Status.STOPPED`.
    """

    record: IterationRecord
    x: Array
    s: Array | None = None
    y_eq: Array | None = None
    y_ineq: Array | None = None
    z_lower: Array | None = None
    z_upper: Array | None = None


# A user hook invoked once per iteration; return ``True`` to request a stop.
IterationCallback = Callable[["IterationInfo"], "bool | None"]


@dataclass(frozen=True, slots=True)
class DerivativeSources:
    """How each derivative was obtained (§3.3) — surfaced for transparency."""

    gradient: str = "unknown"  # analytic | autodiff | finite-diff
    eq_jacobian: str = "n/a"
    ineq_jacobian: str = "n/a"
    hessian: str = "lbfgs"  # exact | autodiff-hvp | lbfgs


@dataclass(frozen=True, slots=True)
class Routes:
    """Which route each auto-selectable mechanism actually took.

    The solver resolves several choices at setup ("auto" linear-solver
    selection, the KKT assembly form, the Hessian source) and simply *runs*
    others as configured; this records them all in one place so a finished
    :class:`Result` explains itself. The ``*_requested`` fields echo the
    option as configured (``"auto"`` means the solver chose); the plain
    fields record what actually ran. Complements :class:`DerivativeSources`,
    which logs the gradient/Jacobian resolution in detail — ``hessian`` is
    mirrored here because it doubles as the solver-route gate (e.g. the
    sparse normal-equations fold).
    """

    # Resolved linear solver, including the dispatched backend — the same
    # string as ``Result.linear_solver`` (e.g. "sparse-NE [Feral LDL^T (CPU)]").
    linear_solver: str = ""
    linsolve_requested: str = ""  # Options.linsolve as configured
    # KKT assembly actually factored/iterated: "condensed" (normal-equations
    # block, dense or matrix-free; equality saddles included), "augmented"
    # (indefinite bordered form), or "normal_equations" (sparsely condensed
    # Gram). Reflects runtime fallbacks (e.g. the dense augmented route
    # falling back to condensed).
    kkt_form: str = ""
    hessian: str = ""  # resolved source: exact | autodiff-hvp | lbfgs
    hessian_requested: str = ""  # Options.hessian as configured
    globalization: str = ""  # filter | breedveld
    mu_schedule: str = ""  # the μ oracle as configured
    scaling: str = ""  # none | gradient-based
    corrections: str = ""  # none | mehrotra | gondzio


@dataclass(frozen=True, slots=True)
class Result:
    """Solution and diagnostics returned by :func:`ipax.solve`.

    ``success`` is true for strict :attr:`Status.OPTIMAL` and the explicitly
    enabled :attr:`Status.ACCEPTABLE` exit. The status itself records the stopping
    condition; component KKT residuals provide the corresponding diagnostics.

    When an :attr:`Status.ACCEPTABLE` exit came from the terminal KKT
    certificate (a failing run salvaged at its returned best iterate with
    repaired multipliers — ``message`` says so), ``kkt_error`` and the
    component residuals describe that certificate and therefore need not
    match any row of ``history``, which keeps the residuals as the loop
    measured them.
    """

    status: Status
    x: Array
    objective: float

    # Multipliers (Array-API arrays in the problem's namespace).
    y_eq: Array | None = None
    y_ineq: Array | None = None
    z_lower: Array | None = None
    z_upper: Array | None = None

    n_iter: int = 0
    kkt_error: float = float("inf")
    constraint_violation: float = float("inf")
    solve_time: float = 0.0  # total wall-clock seconds for the solve
    linear_solver: str = ""  # the linear solver used internally (e.g. "dense")
    device: str = ""  # device the solve ran on (e.g. "cpu", "<CUDA Device 0>")

    derivative_sources: DerivativeSources = field(default_factory=DerivativeSources)
    history: tuple[IterationRecord, ...] = ()
    message: str = ""
    dual_infeasibility: float = float("inf")
    primal_infeasibility: float = float("inf")
    complementarity: float = float("inf")
    # The resolved auto-selection routes, or ``None`` when the solve exited
    # before any machinery was selected (e.g. infeasible bounds at x0).
    # Appended last so positional construction of older Results stays valid.
    routes: Routes | None = None

    @property
    def success(self) -> bool:
        return self.status.is_success


@dataclass(frozen=True, slots=True)
class WarmStart:
    """Initial slacks/multipliers seeding a solve.

    Pair with the starting ``x0`` passed to :func:`ipax.solve`: the primal point
    comes from ``x0`` while these dual quantities (and optionally the inequality
    slacks) seed the interior-point iterate instead of the default
    μ-complementarity start. Any field left ``None`` falls back to the standard
    initialization, so a partial warm start is fine. The slacks and the bound /
    inequality multipliers are floored to stay strictly interior; equality
    multipliers (free sign) pass through unchanged.

    :meth:`from_result` reuses a previous :class:`Result` directly — the typical
    case when re-solving a perturbed problem (e.g. an RT re-plan). Slacks are not
    stored on :class:`Result`, so they are recomputed from feasibility unless
    supplied explicitly. Values are in the original problem's units; when
    gradient-based scaling is enabled the solver rescales them internally.
    """

    s: Array | None = None
    y_eq: Array | None = None
    y_ineq: Array | None = None
    z_lower: Array | None = None
    z_upper: Array | None = None

    @classmethod
    def from_result(cls, result: Result, *, s: Array | None = None) -> WarmStart:
        """Seed a solve from a prior :class:`Result`'s multipliers."""
        return cls(
            s=s,
            y_eq=result.y_eq,
            y_ineq=result.y_ineq,
            z_lower=result.z_lower,
            z_upper=result.z_upper,
        )


__all__ = [
    "DerivativeSources",
    "IterationCallback",
    "IterationInfo",
    "IterationRecord",
    "KKTResiduals",
    "Result",
    "Routes",
    "Status",
    "WarmStart",
]
