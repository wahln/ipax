"""Regression: an iteration's Hessian must not outlive the iteration.

The line search's second-order-correction closure (``soc`` in
``IPMDriver.run``) binds the iteration's Hessian operator ``w`` and inequality
Jacobian as default arguments. The closure stayed bound until the *next*
iteration redefined it, after that iteration had already assembled and
factored its KKT matrix, so the previous Hessian — and every memo it holds —
lived through the next assembly. On TROTS Head-and-Neck_01 (n = 9984, CuPy)
that kept two stale n×n float64 arrays (the loader Hessian's dense memo and
the nonlinear rows' Gram memo, 0.76 GB each) alive at the assembly's
peak, on an 8 GB card that the case already overfilled.
"""

from __future__ import annotations

import gc
import weakref
from typing import Any

import pytest

from ipax import Options, Status, solve
from ipax.backend.operators import Dense
from ipax.testing.problems import HS71
from tests._helpers import array


class _TrackedHessian(Dense):
    """A dense Hessian that counts the live Hessians whenever it is formed."""

    def __init__(self, a: Any, owner: _TrackedHessianHS71) -> None:
        super().__init__(a)
        self._owner = owner

    def dense_matrix(self, like: Any = None) -> Any:
        # Materialized by the dense solver after it factored this iteration's
        # KKT operator: by then the loop has moved past every earlier Hessian.
        gc.collect()
        alive = sum(1 for ref in self._owner.refs if ref() is not None)
        self._owner.peak = max(self._owner.peak, alive)
        return super().dense_matrix(like)


class _TrackedHessianHS71(HS71):
    """HS71 whose Lagrangian Hessian is a fresh tracked operator per call."""

    def __init__(self, xp: Any) -> None:
        super().__init__(xp)
        self.refs: list[weakref.ref[Any]] = []
        self.peak = 0

    def lagrangian_hessian(
        self, x: Any, y_eq: Any, y_ineq: Any, sigma: Any = 1.0
    ) -> Any:
        op = _TrackedHessian(super().lagrangian_hessian(x, y_eq, y_ineq, sigma), self)
        self.refs.append(weakref.ref(op))
        return op


@pytest.mark.parametrize("globalization", ["filter", "breedveld"])
def test_previous_hessian_is_released_before_the_next_assembly(
    namespace, globalization
):
    problem = _TrackedHessianHS71(namespace)
    x0 = array(namespace, [1.0, 5.0, 5.0, 1.0])
    options = Options(hessian="exact", linsolve="dense", globalization=globalization)

    result = solve(problem, x0, options=options)

    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    assert len(problem.refs) >= 5
    # Only the Hessian being formed: never the previous iteration's as well.
    assert problem.peak == 1, problem.peak
