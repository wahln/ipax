"""Minimal example: the fully matrix-free Krylov solve route.

The Lagrangian Hessian is handed to the solver as a ``LinearOperator`` that only
knows how to apply ``H @ v`` — no matrix is ever stored or factored. With
``linsolve="krylov"`` the whole interior-point iteration solves its condensed
Newton systems with conjugate gradients on matvecs alone, so the same code would
scale from this toy problem to the 1e4–1e5-variable RT-like systems the
matrix-free route targets.
"""

from __future__ import annotations

import numpy as np

import ipax
from ipax.backend.operators import MatrixFreeJacobian


class DiagonalBoundedQP(ipax.Problem):
    """Minimize ``0.5 xᵀ diag(d) x − bᵀ x`` s.t. ``x ≥ 0``.

    Separable, so the optimum is ``x_i = max(b_i / d_i, 0)``. The Hessian
    ``diag(d)`` is supplied purely as a matvec (``v ↦ d * v``).
    """

    def __init__(self, d: np.ndarray, b: np.ndarray) -> None:
        self._d = d
        self._b = b

    @property
    def n_vars(self) -> int:
        return int(self._d.shape[0])

    def bounds(self):
        zero = np.zeros_like(self._d)
        return zero, None

    def objective(self, x):
        return 0.5 * float(np.sum(self._d * x * x)) - float(np.sum(self._b * x))

    def gradient(self, x):
        return self._d * x - self._b

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del x, y_eq, y_ineq
        d = self._d
        n = self.n_vars
        # Matrix-free Hessian operator: applies diag(d) without forming it.
        return MatrixFreeJacobian(
            (n, n), lambda v: sigma * d * v, lambda v: sigma * d * v
        )


def main() -> None:
    rng = np.random.default_rng(0)
    n = 50
    d = 1.0 + rng.uniform(0.0, 10.0, size=n)
    b = rng.uniform(-1.0, 1.0, size=n)
    expected_x = np.maximum(b / d, 0.0)

    result = ipax.solve(
        DiagonalBoundedQP(d, b),
        np.full(n, 0.5, dtype=np.float64),
        options=ipax.Options(hessian="exact", linsolve="krylov", verbose=1),
    )

    print(f"status: {result.status.value}")
    print(f"n_vars: {n}")
    print(f"max |x - x*|: {np.max(np.abs(result.x - expected_x)):.3e}")
    print(f"objective: {result.objective:.12g}")
    print(f"kkt_error: {result.kkt_error:.3e}")
    print(f"iterations: {result.n_iter}")


if __name__ == "__main__":
    main()
