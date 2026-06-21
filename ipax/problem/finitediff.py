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

"""Finite-difference gradient/Jacobian fallback (pure Array API).

Last resort in the derivative-precedence chain — emits a warning, since FD
Jacobians are expensive and only appropriate for small problems / debugging.
Implemented entirely with the array namespace; no concrete library.

Central differences are used throughout (second-order accurate). The default
per-coordinate step is ``eps**(1/3) * max(|x_i|, 1)`` — the standard optimal
central-difference scaling, read from the input dtype so float32 and float64
backends each get an appropriate ``h``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipax.backend.namespace import array_namespace

if TYPE_CHECKING:
    from collections.abc import Callable

    from ipax.typing import Array, Namespace, Scalar


def _steps(xp: Namespace, x: Array, rel_step: float | None) -> tuple[Array, float]:
    """Per-coordinate central-difference steps ``h_i`` and the effective rel_step."""
    if rel_step is None:
        # eps**(1/3) minimizes truncation + round-off for central differences.
        rel_step = float(xp.finfo(x.dtype).eps) ** (1.0 / 3.0)
    one = xp.ones_like(x)
    scale = xp.where(xp.abs(x) > one, xp.abs(x), one)
    return rel_step * scale, rel_step


def gradient_fd(
    f: Callable[[Array], Scalar],
    x: Array,
    *,
    rel_step: float | None = None,
) -> Array:
    """Central-difference gradient of a scalar function."""
    xp = array_namespace(x)
    n = int(x.shape[0])
    steps, _ = _steps(xp, x, rel_step)
    eye = xp.eye(n, dtype=x.dtype)
    components = []
    for i in range(n):
        h = steps[i]
        e_i = eye[:, i]
        f_plus = xp.asarray(f(x + h * e_i))
        f_minus = xp.asarray(f(x - h * e_i))
        components.append((f_plus - f_minus) / (2.0 * h))
    return xp.stack(components)


def jacobian_fd(
    f: Callable[[Array], Array],
    x: Array,
    *,
    rel_step: float | None = None,
) -> Array:
    """Central-difference Jacobian of a vector function (dense; small problems).

    Column ``i`` of the returned ``m × n`` matrix is ``∂f/∂x_i`` evaluated by a
    central difference, so the result matches ``∇f(x)`` row-by-constraint.
    """
    xp = array_namespace(x)
    n = int(x.shape[0])
    steps, _ = _steps(xp, x, rel_step)
    eye = xp.eye(n, dtype=x.dtype)
    columns = []
    for i in range(n):
        h = steps[i]
        e_i = eye[:, i]
        f_plus = xp.asarray(f(x + h * e_i))
        f_minus = xp.asarray(f(x - h * e_i))
        columns.append((f_plus - f_minus) / (2.0 * h))
    return xp.stack(columns, axis=1)


__all__ = ["gradient_fd", "jacobian_fd"]
