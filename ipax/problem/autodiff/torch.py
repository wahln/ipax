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

"""PyTorch autodiff adapter (ADAPTER — concrete-library import allowed here).

Provides grad, dense Jacobian, and double-backprop Hessian-vector products for
problems whose arrays are ``torch.Tensor``. Used only when the resolved backend
is PyTorch; the core dispatches here through :mod:`ipax.problem.autodiff`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Allowed import boundary (invariant #1): adapters under problem/autodiff/.
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from ipax.typing import Array, Scalar


def grad(f: Callable[[Array], Scalar], x: Array) -> Array:
    """Reverse-mode gradient ``∇f(x)``."""
    x_req = x.detach().clone().requires_grad_(True)
    value = f(x_req)
    (gradient,) = torch.autograd.grad(value, x_req)
    return gradient.detach()


def jacobian(f: Callable[[Array], Array], x: Array) -> Array:
    """Dense Jacobian ``∇f(x)`` (``m × n``) of a vector function."""
    # Bind through Any: torch.autograd.functional.jacobian is untyped when
    # torch stubs are present and absent (Any) when they are not, so calling it
    # directly would need an env-dependent ``type: ignore`` (mypy strict's
    # warn_unused_ignores then fails in whichever env the stubs disagree).
    jacobian_fn: Any = torch.autograd.functional.jacobian
    return jacobian_fn(f, x.detach()).detach()


def hvp(f: Callable[[Array], Scalar], x: Array, v: Array) -> Array:
    """Hessian–vector product ``∇²f(x) · v`` via double backprop."""
    x_req = x.detach().clone().requires_grad_(True)
    value = f(x_req)
    (gradient,) = torch.autograd.grad(value, x_req, create_graph=True)
    (product,) = torch.autograd.grad(torch.sum(gradient * v), x_req)
    return product.detach()


__all__ = ["grad", "hvp", "jacobian"]
