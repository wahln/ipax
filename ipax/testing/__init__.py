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

"""Shared analytic oracle problems and backend helpers (§8.2).

Importable by both the test suite and the benchmark corpus, so correctness
oracles and performance problems stay in sync.
"""

from __future__ import annotations

from ipax.testing.problems import (
    BoundConstrainedQP,
    EqualityConstrainedQP,
    UnconstrainedQuadratic,
)

__all__ = [
    "BoundConstrainedQP",
    "EqualityConstrainedQP",
    "UnconstrainedQuadratic",
]
