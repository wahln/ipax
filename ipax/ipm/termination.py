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

"""Explicit termination policy for the interior-point iteration.

The optimality and acceptable families share one mechanism: a
:class:`ConditionChecker` fires once its enabled conditions (a relative
objective change plus the scaled KKT-residual components) have held for a
required number of *consecutive* iterations. Optimality requires a single
iteration and reports :attr:`Status.OPTIMAL`; acceptable requires ``n_iter`` and
reports :attr:`Status.ACCEPTABLE`. The ``max_iter`` and ``max_time`` limits are
handled directly by the driver.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ipax.result import Status

if TYPE_CHECKING:
    from ipax.options import AcceptableStoppingOptions, OptimalityConditionOptions
    from ipax.result import IterationRecord


@dataclass(frozen=True, slots=True)
class TerminationDecision:
    """One termination decision returned to the driver."""

    status: Status
    message: str


# (threshold, value, label) for one component condition; ``threshold`` ``None``
# disables it. Compared as ``value <= threshold`` so a NaN never qualifies.
_Level = tuple[float | None, float, str]


def _levels_within(levels: Iterable[_Level]) -> bool:
    """Return whether every enabled level holds (positive form: NaN fails)."""
    return all(
        threshold is None or value <= threshold for threshold, value, _ in levels
    )


class ConditionChecker:
    """Fire when the shared termination conditions hold for enough iterations.

    The conditions are the scaled KKT-residual components (dual infeasibility,
    constraint violation, complementarity), the absolute objective magnitude
    ``f_tol`` (``|f_k| <= f_tol``), and the relative objective change
    ``f_rel_change_tol``. All *enabled* conditions must hold together;
    ``f_rel_change_tol`` additionally needs a previous iterate, so it never
    holds on iteration 0.
    """

    def __init__(
        self,
        *,
        f_tol: float | None,
        f_rel_change_tol: float | None,
        dual_inf_tol: float | None,
        constr_viol_tol: float | None,
        compl_inf_tol: float | None,
        n_iter: int,
        status: Status,
    ) -> None:
        self._f_tol = f_tol
        self._f_rel_change_tol = f_rel_change_tol
        self._dual_inf_tol = dual_inf_tol
        self._constr_viol_tol = constr_viol_tol
        self._compl_inf_tol = compl_inf_tol
        self._n_iter = n_iter
        self._status = status
        self._count = 0
        self._previous_objective: float | None = None
        self._enabled = any(
            threshold is not None
            for threshold in (
                f_tol,
                f_rel_change_tol,
                dual_inf_tol,
                constr_viol_tol,
                compl_inf_tol,
            )
        )

    @classmethod
    def for_optimality(cls, options: OptimalityConditionOptions) -> ConditionChecker:
        return cls(
            f_tol=options.f_tol,
            f_rel_change_tol=options.f_rel_change_tol,
            dual_inf_tol=options.dual_inf_tol,
            constr_viol_tol=options.constr_viol_tol,
            compl_inf_tol=options.compl_inf_tol,
            n_iter=1,
            status=Status.OPTIMAL,
        )

    @classmethod
    def for_acceptable(cls, options: AcceptableStoppingOptions) -> ConditionChecker:
        return cls(
            f_tol=options.f_tol,
            f_rel_change_tol=options.f_rel_change_tol,
            dual_inf_tol=options.dual_inf_tol,
            constr_viol_tol=options.constr_viol_tol,
            compl_inf_tol=options.compl_inf_tol,
            n_iter=options.n_iter,
            status=Status.ACCEPTABLE,
        )

    def observe(self, record: IterationRecord) -> TerminationDecision | None:
        """Observe one accepted iterate and return an optional exit decision."""
        if not self._enabled:
            return None
        hold = self._conditions_hold(record)
        self._previous_objective = record.objective
        self._count = self._count + 1 if hold else 0
        if self._count >= self._n_iter:
            return TerminationDecision(status=self._status, message=self._message())
        return None

    def _conditions_hold(self, record: IterationRecord) -> bool:
        levels: tuple[_Level, ...] = (
            (self._dual_inf_tol, record.dual_infeasibility, "dual infeasibility"),
            (
                self._constr_viol_tol,
                record.primal_infeasibility,
                "constraint violation",
            ),
            (self._compl_inf_tol, record.complementarity, "complementarity"),
        )
        if not _levels_within(levels):
            return False
        # Positive form (``not value <= tol``) so a NaN objective never holds.
        if self._f_tol is not None and not abs(record.objective) <= self._f_tol:
            return False
        if self._f_rel_change_tol is not None and not self._objective_change_is_small(
            record.objective
        ):
            return False
        return True

    def _objective_change_is_small(self, objective: float) -> bool:
        assert self._f_rel_change_tol is not None  # guarded by the caller
        previous = self._previous_objective
        if previous is None:
            return False
        scale = max(1.0, abs(objective), abs(previous))
        return abs(objective - previous) <= self._f_rel_change_tol * scale

    def _message(self) -> str:
        labels = [
            label
            for threshold, label in (
                (self._dual_inf_tol, "dual infeasibility"),
                (self._constr_viol_tol, "constraint violation"),
                (self._compl_inf_tol, "complementarity"),
                (self._f_tol, "objective value"),
                (self._f_rel_change_tol, "objective change"),
            )
            if threshold is not None
        ]
        criteria = ", ".join(labels)
        if self._status is Status.OPTIMAL:
            return f"converged: {criteria} within tolerance"
        return (
            "converged to an acceptable point: "
            f"{criteria} within tolerance for {self._n_iter} consecutive iterations"
        )


__all__ = ["ConditionChecker", "TerminationDecision"]
