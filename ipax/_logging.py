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

"""Layered diagnostics for the solver.

``ipax`` logs through a package logger carrying a :class:`logging.NullHandler`,
so importing the package never emits output on its own. ``Options.verbose`` opts
in to a console handler via :func:`configure_verbosity`; applications that
configure their own handlers on the ``"ipax"`` logger keep full control.

The verbosity ladder is expressed as **custom numeric logging levels**, one per
content tier, so a *single* handler/logger threshold selects what is shown and
downstream handlers (or :func:`caplog`) still receive every record regardless of
``verbose``:

====== ===================================== ==================
level  content                               numeric log level
====== ===================================== ==================
0      (silent — only warnings/errors)       —
1      result summary                         ``RESULT`` (25)
2      + per-iteration table & timing split   ``ITERATION`` (22)
3      + problem structure                    ``PROBLEM`` (19)
4      + resolved solver setup                ``SOLVER`` (16)
5      + every sub-option                     ``OPTIONS`` (13)
≥6     + debug diagnostics                    ``logging.DEBUG`` (10)
====== ===================================== ==================

Higher verbosity → lower threshold → more tiers pass. Emitting code guards the
formatting cost with ``logger.isEnabledFor(LEVEL)``.

This module holds no solver state (invariant #5) — the logger is a process-wide
sink, not mutable algorithm state.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipax.options import Options
    from ipax.result import IterationRecord, Result

LOGGER_NAME = "ipax"

# Content tiers as custom levels, ordered so lower ``verbose`` shows only the
# headline tiers (a higher level passes a higher threshold). Spaced by 3 to stay
# clear of the stdlib levels (DEBUG=10, INFO=20, WARNING=30).
RESULT = 25
ITERATION = 22
PROBLEM = 19
SOLVER = 16
OPTIONS = 13

for _level, _name in (
    (RESULT, "RESULT"),
    (ITERATION, "ITER"),
    (PROBLEM, "PROBLEM"),
    (SOLVER, "SOLVER"),
    (OPTIONS, "OPTIONS"),
):
    logging.addLevelName(_level, _name)

logger = logging.getLogger(LOGGER_NAME)
logger.addHandler(logging.NullHandler())

# Marks the handler this module owns so repeated ``configure_verbosity`` calls
# (e.g. nested ``solve`` invocations) reuse it instead of stacking duplicates.
_VERBOSE_HANDLER_ATTR = "_ipax_verbose_handler"

# Reprint the column header every this many iteration rows so the table stays
# readable on long runs that scroll past the original header.
HEADER_REPEAT_INTERVAL = 10

_HEADER = (
    f"{'iter':>4} {'objective':>15} {'infeas':>10} {'kkt':>10} "
    f"{'mu':>10} {'alpha_pr':>9} {'alpha_du':>9} {'reg':>9} "
    f"{'prob_s':>9} {'step_s':>9}"
)


def verbosity_threshold(verbose: int) -> int:
    """Map ``Options.verbose`` to the logger/handler threshold level.

    ``verbose <= 0`` silences ipax's own output (warnings/errors still pass);
    ``1..5`` select the content tiers ``RESULT..OPTIONS``; ``>= 6`` drops to
    ``DEBUG`` so the scattered diagnostic traces appear.
    """
    if verbose <= 0:
        return RESULT + 1  # above every content tier; warnings (30) still pass
    if verbose >= 6:
        return logging.DEBUG
    return RESULT - 3 * (verbose - 1)  # 1→25, 2→22, 3→19, 4→16, 5→13


def configure_verbosity(verbose: int) -> None:
    """Attach (or update) a console handler driven by ``Options.verbose``.

    ``verbose`` 0 leaves logging untouched (silent unless the application has
    configured its own handlers). Higher values lower the threshold so more
    content tiers reach the console. Idempotent: the module owns a single tagged
    handler, so repeated calls only adjust its level rather than duplicating
    output, and the logger level is only ever lowered (never raised) so an
    application's own configuration is respected.
    """
    if verbose <= 0:
        return
    threshold = verbosity_threshold(verbose)
    # Reuse the handler this module owns; if the application has attached its own
    # handler to the ``"ipax"`` logger, defer to it entirely rather than adding a
    # second console handler — that duplicate is what prints every record twice.
    # Propagation stays on so ancestor handlers (and ``caplog``) keep receiving
    # records regardless of ``verbose``.
    owned: logging.Handler | None = None
    app_configured = False
    for handler in logger.handlers:
        if getattr(handler, _VERBOSE_HANDLER_ATTR, False):
            owned = handler
        elif not isinstance(handler, logging.NullHandler):
            app_configured = True
    if owned is None and not app_configured:
        owned = logging.StreamHandler()
        owned.setFormatter(logging.Formatter("%(message)s"))
        setattr(owned, _VERBOSE_HANDLER_ATTR, True)
        logger.addHandler(owned)
    if owned is not None:
        owned.setLevel(threshold)
    if logger.level == logging.NOTSET or logger.level > threshold:
        logger.setLevel(threshold)


def format_header() -> str:
    """Column header for the per-iteration table (IPOPT-style)."""
    return _HEADER


def format_record(record: IterationRecord, *, acceptable: bool = False) -> str:
    """One iteration-table row matching :func:`format_header`.

    When ``acceptable`` is true the row is tagged with a trailing ``*`` to mark
    an iterate that already satisfies every enabled acceptable-stopping
    criterion, even though the required consecutive-iteration count has not yet
    been reached.
    """
    row = (
        f"{record.iteration:>4d} {record.objective:>15.7e} "
        f"{record.theta:>10.3e} {record.kkt_error:>10.3e} {record.mu:>10.3e} "
        f"{record.alpha_primal:>9.2e} {record.alpha_dual:>9.2e} "
        f"{record.regularization:>9.2e} "
        f"{record.problem_time:>9.2e} {record.step_solve_time:>9.2e}"
    )
    return f"{row} *" if acceptable else row


def format_problem(
    *,
    n_vars: int,
    n_ineq: int,
    n_eq_nonlinear: int,
    n_eq_linear: int,
    n_lower: int,
    n_upper: int,
) -> str:
    """Problem structure block (verbosity tier 3)."""
    return (
        "problem structure:\n"
        f"  variables    = {n_vars}\n"
        f"  inequalities = {n_ineq}\n"
        f"  equalities   = {n_eq_nonlinear} nonlinear + {n_eq_linear} linear\n"
        f"  bounded vars = {n_lower} lower, {n_upper} upper"
    )


def format_solver(opts: Options, solver_name: str) -> str:
    """Resolved solver setup block (verbosity tier 4)."""
    scaling = opts.scaling
    method = getattr(scaling, "method", scaling)
    corrections = getattr(opts.corrections, "method", opts.corrections)
    lines = [
        "solver setup:",
        f"  hessian       = {opts.hessian}",
        f"  linear solver = {opts.linsolve} ({solver_name})",
        f"  globalization = {opts.globalization}",
        f"  mu schedule   = {opts.mu_schedule}",
        f"  scaling       = {method}",
        f"  corrections   = {corrections}",
    ]
    if opts.linsolve in ("krylov", "auto"):
        lines.append(f"  krylov method = {opts.krylov.method}")
    return "\n".join(lines)


def format_options(opts: Options) -> str:
    """Full option dump including every sub-group (verbosity tier 5)."""
    lines = ["options:"]
    for f in dataclasses.fields(opts):
        value = getattr(opts, f.name)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            lines.append(f"  {f.name}:")
            for sub in dataclasses.fields(value):
                lines.append(f"    {sub.name} = {getattr(value, sub.name)}")
        else:
            lines.append(f"  {f.name} = {value}")
    return "\n".join(lines)


def format_result(result: Result) -> str:
    """Final result summary block (verbosity tier 1)."""
    src = result.derivative_sources
    return (
        f"result: {result.status.value} - {result.message}\n"
        f"  objective     = {result.objective:.8e}\n"
        f"  iterations    = {result.n_iter}\n"
        f"  kkt error     = {result.kkt_error:.3e}\n"
        f"  kkt components= dual:{result.dual_infeasibility:.3e} "
        f"primal:{result.primal_infeasibility:.3e} "
        f"compl:{result.complementarity:.3e}\n"
        f"  infeasibility = {result.constraint_violation:.3e}\n"
        f"  solve time    = {result.solve_time:.3e}s\n"
        f"  linear solver = {result.linear_solver}\n"
        f"  device        = {result.device}\n"
        f"  derivatives   = grad:{src.gradient} eq_jac:{src.eq_jacobian} "
        f"ineq_jac:{src.ineq_jacobian} hess:{src.hessian}"
    )


def format_timing(history: tuple[IterationRecord, ...]) -> str:
    """Aggregate problem-callback vs inner-solve time split (verbosity tier 2)."""
    problem_total = sum(record.problem_time for record in history)
    step_total = sum(record.step_solve_time for record in history)
    return (
        f"timing: problem-callbacks = {problem_total:.3e}s, "
        f"inner-solve = {step_total:.3e}s"
    )


__all__ = [
    "HEADER_REPEAT_INTERVAL",
    "ITERATION",
    "LOGGER_NAME",
    "OPTIONS",
    "PROBLEM",
    "RESULT",
    "SOLVER",
    "configure_verbosity",
    "format_header",
    "format_options",
    "format_problem",
    "format_record",
    "format_result",
    "format_solver",
    "format_timing",
    "logger",
    "verbosity_threshold",
]
