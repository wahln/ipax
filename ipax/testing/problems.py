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

"""Analytic oracle problems with closed-form optima and KKT points.

Also hosts the **synthetic RT-like generator** (:func:`make_rt_like_problem`):
a backend-agnostic, fully matrix-free convex NLP whose block-structured
Hessian/Jacobian emulate dose-influence coupling *without* the real dose kernels
(out of scope). It is shared by the cross-backend tests and the scaling
benchmarks.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

from ipax.backend.namespace import array_namespace
from ipax.backend.operators import Dense, LinearOperator
from ipax.problem.base import Problem

if TYPE_CHECKING:
    from ipax.typing import Array, Namespace, Scalar


def _float_dtype(xp: Namespace) -> Any:
    return getattr(xp, "float64", getattr(xp, "float32", None))


def _array(xp: Namespace, value: object) -> Array:
    dtype = _float_dtype(xp)
    if dtype is None:
        return xp.asarray(value)
    return xp.asarray(value, dtype=dtype)


def _transpose(xp: Namespace, value: Array) -> Array:
    return xp.permute_dims(value, (1, 0))


def _mat(xp: Namespace, rows: tuple[tuple[Array, ...], ...]) -> Array:
    """Build a 2-D array from 0-d scalar arrays, dtype-stable across backends."""
    return xp.stack(tuple(xp.stack(row) for row in rows))


class UnconstrainedQuadratic(Problem):
    """Minimize ``0.5 * x.T @ Q @ x - b.T @ x``."""

    def __init__(self, Q: Array, b: Array, xp: Namespace) -> None:
        self.Q = Q
        self.b = b
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return int(self.b.shape[0])

    def objective(self, x: Array) -> Scalar:
        Qx = self.xp.matmul(self.Q, x)
        return 0.5 * self.xp.sum(x * Qx) - self.xp.sum(self.b * x)

    def gradient(self, x: Array) -> Array:
        return self.xp.matmul(self.Q, x) - self.b

    def lagrangian_hessian(
        self,
        x: Array,
        y_eq: Array,
        y_ineq: Array,
        sigma: Scalar = 1.0,
    ) -> Array:
        del x, y_eq, y_ineq
        return sigma * self.Q

    def known_solution(self) -> Array:
        """Closed-form minimizer ``solve(Q, b)``."""
        return self.xp.linalg.solve(self.Q, self.b)


class BoundConstrainedQP(Problem):
    """Two-variable bound QP with both lower and upper active bounds."""

    def __init__(self, xp: Namespace) -> None:
        self.xp = xp
        self.center = _array(xp, [-1.0, 2.0])
        self.lower = _array(xp, [0.0, 0.0])
        self.upper = _array(xp, [float("inf"), 1.0])

    @property
    def n_vars(self) -> int:
        return 2

    def bounds(self) -> tuple[Array, Array]:
        return self.lower, self.upper

    def objective(self, x: Array) -> Scalar:
        diff = x - self.center
        return 0.5 * self.xp.sum(diff * diff)

    def gradient(self, x: Array) -> Array:
        return x - self.center

    def lagrangian_hessian(
        self,
        x: Array,
        y_eq: Array,
        y_ineq: Array,
        sigma: Scalar = 1.0,
    ) -> Array:
        del y_eq, y_ineq
        return sigma * self.xp.eye(2, dtype=x.dtype)

    def known_solution(self) -> Array:
        return _array(self.xp, [0.0, 1.0])

    def known_bound_multipliers(self) -> tuple[Array, Array]:
        return _array(self.xp, [1.0, 0.0]), _array(self.xp, [0.0, 1.0])


class EqualityConstrainedQP(Problem):
    """Minimize ``0.5 * ||x||^2`` subject to ``sum(x) = 1``."""

    def __init__(self, xp: Namespace) -> None:
        self.xp = xp
        self.A = _array(xp, [[1.0, 1.0]])
        self.b = _array(xp, [1.0])

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x: Array) -> Scalar:
        return 0.5 * self.xp.sum(x * x)

    def gradient(self, x: Array) -> Array:
        return x

    def linear_eq(self) -> tuple[Array, Array]:
        return self.A, self.b

    def lagrangian_hessian(
        self,
        x: Array,
        y_eq: Array,
        y_ineq: Array,
        sigma: Scalar = 1.0,
    ) -> Array:
        del y_eq, y_ineq
        return sigma * self.xp.eye(2, dtype=x.dtype)

    def known_solution(self) -> Array:
        return _array(self.xp, [0.5, 0.5])

    def known_multiplier(self) -> Array:
        return _array(self.xp, [-0.5])

    def stationarity(self, x: Array, y: Array) -> Array:
        return self.gradient(x) + self.xp.matmul(_transpose(self.xp, self.A), y)


class HS6(Problem):
    """HS6: ``min (1-x1)²`` s.t. ``10(x2 - x1²) = 0``. Optimum ``(1, 1)``, f=0."""

    def __init__(self, xp: Namespace) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x: Array) -> Scalar:
        return (1.0 - x[0]) ** 2

    def gradient(self, x: Array) -> Array:
        xp = self.xp
        return xp.stack((-2.0 * (1.0 - x[0]), xp.zeros_like(x[0])))

    def eq_constraints(self, x: Array) -> Array:
        return self.xp.stack((10.0 * (x[1] - x[0] * x[0]),))

    def eq_jacobian(self, x: Array) -> Array:
        xp = self.xp
        ten = 10.0 + xp.zeros_like(x[0])
        return _mat(xp, ((-20.0 * x[0], ten),))

    def lagrangian_hessian(
        self, x: Array, y_eq: Array, y_ineq: Array, sigma: Scalar = 1.0
    ) -> Array:
        del y_ineq
        xp = self.xp
        zero = xp.zeros_like(x[0])
        h00 = 2.0 * sigma - 20.0 * y_eq[0]
        return _mat(xp, ((h00, zero), (zero, zero)))

    def known_solution(self) -> Array:
        return _array(self.xp, [1.0, 1.0])


class HS7(Problem):
    """HS7: ``min ln(1+x1²) - x2`` s.t. ``(1+x1²)² + x2² - 4 = 0``.

    Optimum ``(0, √3)``, ``f* = -√3``.
    """

    def __init__(self, xp: Namespace) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x: Array) -> Scalar:
        xp = self.xp
        return xp.log(1.0 + x[0] * x[0]) - x[1]

    def gradient(self, x: Array) -> Array:
        xp = self.xp
        return xp.stack((2.0 * x[0] / (1.0 + x[0] * x[0]), -1.0 + xp.zeros_like(x[1])))

    def eq_constraints(self, x: Array) -> Array:
        t = 1.0 + x[0] * x[0]
        return self.xp.stack((t * t + x[1] * x[1] - 4.0,))

    def eq_jacobian(self, x: Array) -> Array:
        xp = self.xp
        d0 = 4.0 * x[0] * (1.0 + x[0] * x[0])  # d/dx1 (1+x1²)²
        return _mat(xp, ((d0, 2.0 * x[1]),))

    def lagrangian_hessian(
        self, x: Array, y_eq: Array, y_ineq: Array, sigma: Scalar = 1.0
    ) -> Array:
        del y_ineq
        xp = self.xp
        zero = xp.zeros_like(x[0])
        t = 1.0 + x[0] * x[0]
        # ∇²f: d/dx1 [2x1/(1+x1²)] = 2(1-x1²)/(1+x1²)²
        f00 = sigma * 2.0 * (1.0 - x[0] * x[0]) / (t * t)
        # ∇²c1: d²/dx1² (1+x1²)² = 12x1² + 4 ; (2,2) entry = 2
        c00 = 12.0 * x[0] * x[0] + 4.0
        h00 = f00 + y_eq[0] * c00
        h11 = y_eq[0] * 2.0 + zero
        return _mat(xp, ((h00, zero), (zero, h11)))

    def known_solution(self) -> Array:
        xp = self.xp
        three = _array(xp, [3.0])
        return xp.concat((_array(xp, [0.0]), xp.sqrt(three)))


class HS8(Problem):
    """HS8: constant objective ``-1`` s.t. two quadratic equalities.

    ``x1²+x2²-25=0``, ``x1 x2 - 9 = 0``; feasible value ``f* = -1``.
    """

    def __init__(self, xp: Namespace) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x: Array) -> Scalar:
        return -1.0 + self.xp.sum(x) * 0.0

    def gradient(self, x: Array) -> Array:
        return self.xp.zeros_like(x)

    def eq_constraints(self, x: Array) -> Array:
        return self.xp.stack((x[0] * x[0] + x[1] * x[1] - 25.0, x[0] * x[1] - 9.0))

    def eq_jacobian(self, x: Array) -> Array:
        xp = self.xp
        return _mat(
            xp,
            ((2.0 * x[0], 2.0 * x[1]), (x[1], x[0])),
        )

    def lagrangian_hessian(
        self, x: Array, y_eq: Array, y_ineq: Array, sigma: Scalar = 1.0
    ) -> Array:
        del x, y_ineq, sigma
        xp = self.xp
        two = 2.0 * y_eq[0]
        return _mat(xp, ((two, y_eq[1]), (y_eq[1], two)))

    def known_solution(self) -> Array:
        xp = self.xp
        return xp.concat(
            (
                (xp.sqrt(_array(xp, [43.0])) + xp.sqrt(_array(xp, [7.0]))) / 2.0,
                (xp.sqrt(_array(xp, [43.0])) - xp.sqrt(_array(xp, [7.0]))) / 2.0,
            )
        )


class HS35(Problem):
    """HS35 (Beale): convex QP with one linear inequality and ``x ≥ 0``.

    ``min 9 - 8x1 - 6x2 - 4x3 + 2x1² + 2x2² + x3² + 2x1x2 + 2x1x3``
    s.t. ``x1 + x2 + 2x3 ≤ 3``, ``x ≥ 0``. Optimum ``(4/3, 7/9, 4/9)``, ``f*=1/9``.
    """

    def __init__(self, xp: Namespace) -> None:
        self.xp = xp
        self._q = _array(xp, [[4.0, 2.0, 2.0], [2.0, 4.0, 0.0], [2.0, 0.0, 2.0]])
        self._c = _array(xp, [-8.0, -6.0, -4.0])

    @property
    def n_vars(self) -> int:
        return 3

    def bounds(self) -> tuple[Array, Array | None]:
        return _array(self.xp, [0.0, 0.0, 0.0]), None

    def objective(self, x: Array) -> Scalar:
        xp = self.xp
        return 9.0 + xp.sum(self._c * x) + 0.5 * xp.sum(x * xp.matmul(self._q, x))

    def gradient(self, x: Array) -> Array:
        return self.xp.matmul(self._q, x) + self._c

    def ineq_constraints(self, x: Array) -> Array:
        return self.xp.stack((x[0] + x[1] + 2.0 * x[2] - 3.0,))

    def ineq_jacobian(self, x: Array) -> Array:
        xp = self.xp
        one = 1.0 + xp.zeros_like(x[0])
        return _mat(xp, ((one, one, 2.0 * one),))

    def lagrangian_hessian(
        self, x: Array, y_eq: Array, y_ineq: Array, sigma: Scalar = 1.0
    ) -> Array:
        del x, y_eq, y_ineq
        return sigma * self._q

    def known_solution(self) -> Array:
        return _array(self.xp, [4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0])


class HS43(Problem):
    """HS43 (Rosen–Suzuki): convex objective, three quadratic inequalities.

    Optimum ``(0, 1, 2, -1)``, ``f* = -44``.
    """

    def __init__(self, xp: Namespace) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 4

    def objective(self, x: Array) -> Scalar:
        return (
            x[0] * x[0]
            + x[1] * x[1]
            + 2.0 * x[2] * x[2]
            + x[3] * x[3]
            - 5.0 * x[0]
            - 5.0 * x[1]
            - 21.0 * x[2]
            + 7.0 * x[3]
        )

    def gradient(self, x: Array) -> Array:
        return self.xp.stack(
            (
                2.0 * x[0] - 5.0,
                2.0 * x[1] - 5.0,
                4.0 * x[2] - 21.0,
                2.0 * x[3] + 7.0,
            )
        )

    def ineq_constraints(self, x: Array) -> Array:
        return self.xp.stack(
            (
                x[0] * x[0]
                + x[1] * x[1]
                + x[2] * x[2]
                + x[3] * x[3]
                + x[0]
                - x[1]
                + x[2]
                - x[3]
                - 8.0,
                x[0] * x[0]
                + 2.0 * x[1] * x[1]
                + x[2] * x[2]
                + 2.0 * x[3] * x[3]
                - x[0]
                - x[3]
                - 10.0,
                2.0 * x[0] * x[0]
                + x[1] * x[1]
                + x[2] * x[2]
                + 2.0 * x[0]
                - x[1]
                - x[3]
                - 5.0,
            )
        )

    def ineq_jacobian(self, x: Array) -> Array:
        xp = self.xp
        return _mat(
            xp,
            (
                (
                    2.0 * x[0] + 1.0,
                    2.0 * x[1] - 1.0,
                    2.0 * x[2] + 1.0,
                    2.0 * x[3] - 1.0,
                ),
                (2.0 * x[0] - 1.0, 4.0 * x[1], 2.0 * x[2], 4.0 * x[3] - 1.0),
                (
                    4.0 * x[0] + 2.0,
                    2.0 * x[1] - 1.0,
                    2.0 * x[2],
                    -1.0 + xp.zeros_like(x[3]),
                ),
            ),
        )

    def lagrangian_hessian(
        self, x: Array, y_eq: Array, y_ineq: Array, sigma: Scalar = 1.0
    ) -> Array:
        del x, y_eq
        xp = self.xp
        d0 = 2.0 * sigma + 2.0 * y_ineq[0] + 2.0 * y_ineq[1] + 4.0 * y_ineq[2]
        d1 = 2.0 * sigma + 2.0 * y_ineq[0] + 4.0 * y_ineq[1] + 2.0 * y_ineq[2]
        d2 = 4.0 * sigma + 2.0 * y_ineq[0] + 2.0 * y_ineq[1] + 2.0 * y_ineq[2]
        d3 = 2.0 * sigma + 2.0 * y_ineq[0] + 4.0 * y_ineq[1]
        zero = xp.zeros_like(d0)
        return _mat(
            xp,
            (
                (d0, zero, zero, zero),
                (zero, d1, zero, zero),
                (zero, zero, d2, zero),
                (zero, zero, zero, d3),
            ),
        )

    def known_solution(self) -> Array:
        return _array(self.xp, [0.0, 1.0, 2.0, -1.0])


class HS21(Problem):
    """HS21: bound-constrained QP with one (inactive) linear inequality.

    ``min 0.01 x1² + x2² - 100`` s.t. ``10 x1 - x2 ≥ 10``, ``2 ≤ x1 ≤ 50``,
    ``-50 ≤ x2 ≤ 50``. Optimum ``(2, 0)`` — the lower bound on ``x1`` is active
    while the inequality stays slack — ``f* = -99.96``. Exercises an active bound
    multiplier (``z_L``) alongside a two-sided ``linear_ineq`` block.
    """

    def __init__(self, xp: Namespace) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 2

    def bounds(self) -> tuple[Array, Array]:
        return _array(self.xp, [2.0, -50.0]), _array(self.xp, [50.0, 50.0])

    def objective(self, x: Array) -> Scalar:
        return 0.01 * x[0] * x[0] + x[1] * x[1] - 100.0

    def gradient(self, x: Array) -> Array:
        return self.xp.stack((0.02 * x[0], 2.0 * x[1]))

    def linear_ineq(self) -> tuple[Array, Array, Array]:
        # 10 x1 - x2 ≥ 10  ⇒  l ≤ A x ≤ u with l = 10, u = +∞.
        xp = self.xp
        a = _array(xp, [[10.0, -1.0]])
        return a, _array(xp, [10.0]), _array(xp, [float("inf")])

    def lagrangian_hessian(
        self, x: Array, y_eq: Array, y_ineq: Array, sigma: Scalar = 1.0
    ) -> Array:
        del y_eq, y_ineq  # objective Hessian is constant; constraint is linear
        xp = self.xp
        zero = xp.zeros_like(x[0])
        return _mat(xp, ((0.02 * sigma + zero, zero), (zero, 2.0 * sigma + zero)))

    def known_solution(self) -> Array:
        return _array(self.xp, [2.0, 0.0])

    def known_bound_multipliers(self) -> tuple[Array, Array]:
        # ∇f(x*) = (0.04, 0); only the lower bound on x1 is active.
        return _array(self.xp, [0.04, 0.0]), _array(self.xp, [0.0, 0.0])


class HS28(Problem):
    """HS28: convex equality-constrained QP in three variables.

    ``min (x1+x2)² + (x2+x3)²`` s.t. ``x1 + 2x2 + 3x3 = 1``. Optimum
    ``(0.5, -0.5, 0.5)``, ``f* = 0``. The objective gradient vanishes at the
    optimum, so the equality multiplier is ``0`` — a useful degenerate-dual case.
    """

    def __init__(self, xp: Namespace) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 3

    def objective(self, x: Array) -> Scalar:
        a = x[0] + x[1]
        b = x[1] + x[2]
        return a * a + b * b

    def gradient(self, x: Array) -> Array:
        xp = self.xp
        a = x[0] + x[1]
        b = x[1] + x[2]
        return xp.stack((2.0 * a, 2.0 * a + 2.0 * b, 2.0 * b))

    def linear_eq(self) -> tuple[Array, Array]:
        xp = self.xp
        return _array(xp, [[1.0, 2.0, 3.0]]), _array(xp, [1.0])

    def lagrangian_hessian(
        self, x: Array, y_eq: Array, y_ineq: Array, sigma: Scalar = 1.0
    ) -> Array:
        del x, y_eq, y_ineq  # constant ∇²f; constraint is linear
        xp = self.xp
        return sigma * _array(xp, [[2.0, 2.0, 0.0], [2.0, 4.0, 2.0], [0.0, 2.0, 2.0]])

    def known_solution(self) -> Array:
        return _array(self.xp, [0.5, -0.5, 0.5])

    def known_multiplier(self) -> Array:
        return _array(self.xp, [0.0])


class HS9(Problem):
    """HS9: equality-constrained problem with a non-unique (periodic) optimum.

    ``min sin(π x1 / 12) · cos(π x2 / 16)`` s.t. ``4 x1 - 3 x2 = 0``. The
    optimum is non-unique — every ``(12k-3, 16k-4)`` attains ``f* = -0.5`` — so
    callers assert ``f*`` and the KKT conditions rather than a specific ``x*``.
    Exercises a trigonometric (non-quadratic) objective with a dense Hessian.
    """

    _A = math.pi / 12.0
    _B = math.pi / 16.0

    def __init__(self, xp: Namespace) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x: Array) -> Scalar:
        xp = self.xp
        return xp.sin(self._A * x[0]) * xp.cos(self._B * x[1])

    def gradient(self, x: Array) -> Array:
        xp = self.xp
        a, b = self._A, self._B
        return xp.stack(
            (
                a * xp.cos(a * x[0]) * xp.cos(b * x[1]),
                -b * xp.sin(a * x[0]) * xp.sin(b * x[1]),
            )
        )

    def linear_eq(self) -> tuple[Array, Array]:
        xp = self.xp
        return _array(xp, [[4.0, -3.0]]), _array(xp, [0.0])

    def lagrangian_hessian(
        self, x: Array, y_eq: Array, y_ineq: Array, sigma: Scalar = 1.0
    ) -> Array:
        del y_eq, y_ineq  # constraint is linear ⇒ no constraint-Hessian term
        xp = self.xp
        a, b = self._A, self._B
        s0, c0 = xp.sin(a * x[0]), xp.cos(a * x[0])
        s1, c1 = xp.sin(b * x[1]), xp.cos(b * x[1])
        h00 = sigma * (-a * a * s0 * c1)
        h01 = sigma * (-a * b * c0 * s1)
        h11 = sigma * (-b * b * s0 * c1)
        return _mat(xp, ((h00, h01), (h01, h11)))


class HS71(Problem):
    """HS71: the canonical IPOPT test NLP — equality, inequality, and bounds.

    ``min x1 x4 (x1+x2+x3) + x3`` s.t. ``x1 x2 x3 x4 ≥ 25``,
    ``x1²+x2²+x3²+x4² = 40``, ``1 ≤ xi ≤ 5``. Optimum
    ``(1, 4.743…, 3.821…, 1.379…)`` with the lower bound on ``x1`` active,
    ``f* ≈ 17.014``. Exercises every constraint class at once with a fully
    nonlinear objective, equality, and inequality.
    """

    def __init__(self, xp: Namespace) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 4

    def bounds(self) -> tuple[Array, Array]:
        return _array(self.xp, [1.0] * 4), _array(self.xp, [5.0] * 4)

    def objective(self, x: Array) -> Scalar:
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def gradient(self, x: Array) -> Array:
        xp = self.xp
        s = x[0] + x[1] + x[2]
        one = 1.0 + xp.zeros_like(x[0])
        return xp.stack(
            (
                x[3] * (s + x[0]),
                x[0] * x[3],
                x[0] * x[3] + one,
                x[0] * s,
            )
        )

    def eq_constraints(self, x: Array) -> Array:
        return self.xp.stack(
            (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40.0,)
        )

    def eq_jacobian(self, x: Array) -> Array:
        xp = self.xp
        return _mat(xp, ((2.0 * x[0], 2.0 * x[1], 2.0 * x[2], 2.0 * x[3]),))

    def ineq_constraints(self, x: Array) -> Array:
        # x1 x2 x3 x4 ≥ 25  ⇒  g = 25 - x1 x2 x3 x4 ≤ 0.
        return self.xp.stack((25.0 - x[0] * x[1] * x[2] * x[3],))

    def ineq_jacobian(self, x: Array) -> Array:
        xp = self.xp
        return _mat(
            xp,
            (
                (
                    -x[1] * x[2] * x[3],
                    -x[0] * x[2] * x[3],
                    -x[0] * x[1] * x[3],
                    -x[0] * x[1] * x[2],
                ),
            ),
        )

    def lagrangian_hessian(
        self, x: Array, y_eq: Array, y_ineq: Array, sigma: Scalar = 1.0
    ) -> Array:
        xp = self.xp
        ye, yi = y_eq[0], y_ineq[0]
        zero = xp.zeros_like(x[0])
        # ∇²f (only the listed entries are nonzero).
        f00 = 2.0 * x[3]
        f01 = x[3]
        f02 = x[3]
        f03 = 2.0 * x[0] + x[1] + x[2]
        f13 = x[0]
        f23 = x[0]
        # ∇²g = -∂²(x1 x2 x3 x4): off-diagonal products of the other two vars.
        g01 = -x[2] * x[3]
        g02 = -x[1] * x[3]
        g03 = -x[1] * x[2]
        g12 = -x[0] * x[3]
        g13 = -x[0] * x[2]
        g23 = -x[0] * x[1]
        # ∇²c = 2·I (equality is a sphere); contributes only on the diagonal.
        diag = 2.0 * ye + zero
        h00 = sigma * f00 + diag
        h01 = sigma * f01 + yi * g01
        h02 = sigma * f02 + yi * g02
        h03 = sigma * f03 + yi * g03
        h12 = yi * g12
        h13 = sigma * f13 + yi * g13
        h23 = sigma * f23 + yi * g23
        return _mat(
            xp,
            (
                (h00, h01, h02, h03),
                (h01, diag, h12, h13),
                (h02, h12, diag, h23),
                (h03, h13, h23, diag),
            ),
        )

    def known_solution(self) -> Array:
        return _array(self.xp, [1.0, 4.74299963, 3.82114998, 1.37940829])


class InfeasibleEqualities(Problem):
    """Inconsistent equalities ``x = 0`` and ``x = 1`` (no feasible point)."""

    def __init__(self, xp: Namespace) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 1

    def objective(self, x: Array) -> Scalar:
        return self.xp.sum(x * x)

    def gradient(self, x: Array) -> Array:
        return 2.0 * x

    def eq_constraints(self, x: Array) -> Array:
        return self.xp.stack((x[0], x[0] - 1.0))

    def eq_jacobian(self, x: Array) -> Array:
        xp = self.xp
        one = 1.0 + xp.zeros_like(x[0])
        return _mat(xp, ((one,), (one,)))

    def lagrangian_hessian(
        self, x: Array, y_eq: Array, y_ineq: Array, sigma: Scalar = 1.0
    ) -> Array:
        del y_eq, y_ineq
        xp = self.xp
        return _mat(xp, ((2.0 * sigma + xp.zeros_like(x[0]),),))


class _DiagPlusLowRank(LinearOperator):
    """Matrix-free SPD Hessian ``W = σ·(diag(h) + C Cᵀ)`` (RT-like coupling).

    ``diag(h)`` is the per-beamlet curvature and each column of ``C`` couples the
    beamlets of one "structure" (a rank term), emulating intra-structure
    dose coupling. ``matvec`` is ``O(n·B)`` with ``B`` structures — no ``n×n``
    matrix — and the diagonal is available in closed form, so the condensed
    Jacobi preconditioner engages.
    """

    def __init__(self, h: Array, C: Array, sigma: float = 1.0) -> None:
        self._h = h
        self._C = C
        self._sigma = sigma
        self._n = int(h.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return self._n, self._n

    def matvec(self, v: Array) -> Array:
        xp = array_namespace(v)
        ctv = xp.matmul(xp.permute_dims(self._C, (1, 0)), v)
        low = xp.matmul(self._C, ctv)
        return self._sigma * (self._h * v + low)

    def rmatvec(self, v: Array) -> Array:
        return self.matvec(v)  # symmetric

    def diagonal(self, like: Array | None = None) -> Array:
        del like
        xp = array_namespace(self._h)
        return self._sigma * (self._h + xp.sum(self._C * self._C, axis=1))

    def diagonal_low_rank_form(self) -> tuple[Array, Array, Array]:
        """Return ``(d, U, M)`` with ``W == diag(d) − U M⁻¹ Uᵀ`` (sparse-assemblable).

        Here ``W = σ(diag(h) + C Cᵀ)``, so ``d = σ h``, ``U = √σ·C`` and ``M = −I``
        (``−U M⁻¹ Uᵀ = +σ C Cᵀ``). This lets the matrix-free RT-like Hessian factor
        through the sparse-direct route as a low-rank border (§4.3) — the same hook
        the L-BFGS compact Hessian implements.
        """
        xp = array_namespace(self._h)
        scale = self._sigma**0.5
        d = self._sigma * self._h
        u = scale * self._C
        m = -xp.eye(int(self._C.shape[1]), dtype=self._h.dtype)
        return d, u, m


class SyntheticRTProblem(Problem):
    """Backend-agnostic, fully matrix-free RT-like convex NLP (§9.1).

    Minimize ``0.5 xᵀ W x − qᵀ x`` over fluences ``x ≥ 0`` subject to per-structure
    mean-dose caps ``A_struct x ≤ caps`` (linear inequalities declared through the
    nonlinear interface so their constant Jacobian is reused each iteration). ``W``
    is the matrix-free :class:`_DiagPlusLowRank` operator, so the whole solve runs
    without materializing an ``n×n`` matrix. No closed-form optimum — verified via
    the KKT conditions at the returned point.
    """

    def __init__(
        self,
        xp: Namespace,
        h: Array,
        C: Array,
        q: Array,
        a_struct: Array,
        caps: Array,
    ) -> None:
        self.xp = xp
        self._h = h
        self._C = C
        self._q = q
        self._a_struct = a_struct
        self._caps = caps
        self._n = int(h.shape[0])

    @property
    def n_vars(self) -> int:
        return self._n

    def bounds(self) -> tuple[Array, None]:
        return self.xp.zeros((self._n,), dtype=self._h.dtype), None

    def _hessian_operator(self, sigma: float) -> _DiagPlusLowRank:
        return _DiagPlusLowRank(self._h, self._C, sigma)

    def objective(self, x: Array) -> Scalar:
        wx = self._hessian_operator(1.0).matvec(x)
        return 0.5 * self.xp.sum(x * wx) - self.xp.sum(self._q * x)

    def gradient(self, x: Array) -> Array:
        return self._hessian_operator(1.0).matvec(x) - self._q

    def ineq_constraints(self, x: Array) -> Array:
        return self.xp.matmul(self._a_struct, x) - self._caps

    def ineq_jacobian(self, x: Array) -> LinearOperator:
        del x
        return Dense(self._a_struct)

    def lagrangian_hessian(
        self, x: Array, y_eq: Array, y_ineq: Array, sigma: Scalar = 1.0
    ) -> LinearOperator:
        # Constraints are linear ⇒ zero Hessian contribution; W is the objective's.
        del x, y_eq, y_ineq
        return self._hessian_operator(float(sigma))


def make_rt_like_problem(
    xp: Namespace,
    n_vars: int,
    *,
    n_structures: int = 8,
    density: float = 0.2,
    seed: int = 0,
) -> SyntheticRTProblem:
    """Build a synthetic, matrix-free, block-structured RT-like NLP (§9.1).

    Deterministic across backends: the random data is drawn with a seeded Python
    RNG into plain lists, then cast through ``xp`` so every namespace sees the
    identical problem. ``density`` controls how many structures each beamlet
    couples to (Breedveld Table 1's 5–50% range); ``n_structures`` is the number
    of rank/constraint blocks.
    """
    if n_vars <= 0:
        raise ValueError("n_vars must be positive")
    if n_structures <= 0:
        raise ValueError("n_structures must be positive")
    rng = random.Random(seed)
    n = n_vars
    blocks = n_structures

    # Per-beamlet curvature h > 0 and structure membership (round-robin blocks).
    h = [0.5 + rng.random() for _ in range(n)]
    member = [j % blocks for j in range(n)]

    # Low-rank coupling C (n×blocks): always the primary structure, plus extra
    # cross-structure coupling with probability `density`.
    C = [[0.0] * blocks for _ in range(n)]
    for j in range(n):
        for b in range(blocks):
            if b == member[j] or rng.random() < density:
                C[j][b] = 0.3 + 0.7 * rng.random()

    # Linear term q > 0 (pulls fluences positive, so the caps/bounds matter).
    q = [0.5 + rng.random() for _ in range(n)]

    # Mean-dose-per-structure rows and caps. Caps are positive (so x = 0 is
    # strictly feasible) and loose enough that the convex solve converges
    # reliably across sizes/backends, while still exercising the full inequality
    # machinery (slacks, multipliers, fraction-to-boundary) each iteration.
    counts = [max(1, member.count(b)) for b in range(blocks)]
    a_struct = [
        [1.0 / counts[b] if member[j] == b else 0.0 for j in range(n)]
        for b in range(blocks)
    ]
    caps = [0.5 for _ in range(blocks)]

    return SyntheticRTProblem(
        xp,
        _array(xp, h),
        _array(xp, C),
        _array(xp, q),
        _array(xp, a_struct),
        _array(xp, caps),
    )


__all__ = [
    "HS6",
    "HS7",
    "HS8",
    "HS9",
    "HS21",
    "HS28",
    "HS35",
    "HS43",
    "HS71",
    "BoundConstrainedQP",
    "EqualityConstrainedQP",
    "InfeasibleEqualities",
    "SyntheticRTProblem",
    "UnconstrainedQuadratic",
    "make_rt_like_problem",
]
