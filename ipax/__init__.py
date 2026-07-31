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

"""ipax — Array-API interior-point solver for nonlinear constrained optimization.

Public API. The package never imports a concrete array library at the top level
(invariant #1); the backend is inferred from the arrays a :class:`Problem`
returns.
"""

from __future__ import annotations

from ipax.backend.operators import (
    COOOperator,
    CSCOperator,
    CSROperator,
)
from ipax.options import (
    AcceptableStoppingOptions,
    CorrectionsOptions,
    OptimalityConditionOptions,
    Options,
    ScalingOptions,
)
from ipax.problem.base import Problem
from ipax.problem.function import FunctionProblem, LinearProblem, QuadraticProblem
from ipax.result import (
    DerivativeSources,
    IterationCallback,
    IterationInfo,
    IterationRecord,
    KKTResiduals,
    Result,
    Routes,
    Status,
    WarmStart,
)
from ipax.solve import solve

__all__ = [
    "AcceptableStoppingOptions",
    "COOOperator",
    "CSCOperator",
    "CSROperator",
    "CorrectionsOptions",
    "DerivativeSources",
    "FunctionProblem",
    "IterationCallback",
    "IterationInfo",
    "IterationRecord",
    "KKTResiduals",
    "LinearProblem",
    "OptimalityConditionOptions",
    "Options",
    "Problem",
    "QuadraticProblem",
    "Result",
    "Routes",
    "ScalingOptions",
    "Status",
    "WarmStart",
    "solve",
]

__version__ = "0.9.0"
