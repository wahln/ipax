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

"""Exceptions and warnings raised by ipax.

Every exception derives from :class:`IpaxError` *and* from the built-in type
ipax raised for the same condition before 0.14, so existing ``except
ValueError`` / ``except RuntimeError`` clauses keep working. Numerical trouble
inside a solve is not raised: it is reported through
:class:`~ipax.result.Status`. A linear-solve failure the solver cannot recover
from surfaces as :class:`~ipax.linalg.solver.LinearSolveError`, also a subclass
of :class:`IpaxError`.
"""

from __future__ import annotations

import os
import sys
import warnings

_IPAX_DIR = os.path.normcase(os.path.dirname(os.path.abspath(__file__))) + os.sep
# ``dataclasses.replace`` calls the class from inside the stdlib; a warning
# raised in a frozen-dataclass __init__ should name the code that called
# ``replace``, not ``dataclasses.py``.
_DATACLASSES_FILE = os.path.normcase(
    os.path.abspath(sys.modules["dataclasses"].__file__ or "")
)


class IpaxError(Exception):
    """Base class of every exception ipax raises on purpose."""


class ConfigurationError(IpaxError, ValueError):
    """An option value, or a combination of options, is invalid."""


class ProblemDefinitionError(IpaxError, ValueError):
    """A :class:`~ipax.problem.base.Problem`, or the arguments of a solve, are inconsistent.

    For example a start point whose length is not ``n_vars``, bounds of the
    wrong shape, or :meth:`~ipax.problem.base.Problem.provides` naming a
    method that does not exist.
    """


class DerivativeUnavailableError(IpaxError, RuntimeError):
    """A derivative the solve needs cannot be obtained.

    Raised before the first iteration, e.g. when the problem defines no
    gradient and both autodiff and finite differences are disabled, or when
    ``hessian="exact"`` is requested for a problem without
    ``lagrangian_hessian``.
    """


class IpaxWarning(UserWarning):
    """Base class of the warnings ipax emits."""


class FiniteDifferenceWarning(IpaxWarning):
    """A derivative is being approximated by finite differences.

    Finite differences cost one or two extra function evaluations per variable
    and are less accurate than an analytic or autodiff derivative. Silence it
    with ``warnings.simplefilter("ignore", ipax.FiniteDifferenceWarning)``.
    """


def _is_internal(filename: str) -> bool:
    path = os.path.normcase(os.path.abspath(filename))
    return path.startswith(_IPAX_DIR) or path == _DATACLASSES_FILE


def warn_external(message: str, category: type[Warning]) -> None:
    """Warn, attributing the warning to the first frame outside ipax.

    A fixed ``stacklevel`` points at the right line only for one call path;
    the same warning can be reached from :func:`ipax.solve`, the SciPy front
    end or a test. Walking past ipax's own frames (and ``dataclasses.replace``)
    always names the user's line, so ``warnings`` filters by module and the
    once-per-location default work as users expect.
    """
    frame = sys._getframe(1)
    level = 2  # stacklevel 2 = the caller of warn_external
    while frame.f_back is not None and _is_internal(frame.f_code.co_filename):
        frame = frame.f_back
        level += 1
    warnings.warn(message, category, stacklevel=level)


__all__ = [
    "ConfigurationError",
    "DerivativeUnavailableError",
    "FiniteDifferenceWarning",
    "IpaxError",
    "IpaxWarning",
    "ProblemDefinitionError",
]
