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

"""JAX autodiff adapter (ADAPTER — concrete-library import allowed here).

Provides grad, dense Jacobian, and ``hvp`` (via ``jax.jvp`` of ``jax.grad``) for
problems whose arrays are JAX arrays. Used only when the resolved backend is
JAX; the core dispatches here through :mod:`ipax.problem.autodiff`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Allowed import boundary (invariant #1): adapters under problem/autodiff/.
import jax

if TYPE_CHECKING:
    from collections.abc import Callable

    from ipax.typing import Array, Scalar


def grad(f: Callable[[Array], Scalar], x: Array) -> Array:
    """Reverse-mode gradient ``∇f(x)``."""
    return jax.grad(f)(x)


def jacobian(f: Callable[[Array], Array], x: Array) -> Array:
    """Dense Jacobian ``∇f(x)`` (``m × n``) of a vector function."""
    return jax.jacobian(f)(x)


def hvp(f: Callable[[Array], Scalar], x: Array, v: Array) -> Array:
    """Hessian–vector product ``∇²f(x) · v`` as a forward-over-reverse jvp."""
    return jax.jvp(jax.grad(f), (x,), (v,))[1]


__all__ = ["grad", "hvp", "jacobian"]
