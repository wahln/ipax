"""Minimal equality-constrained solve via the regularized saddle route."""

from __future__ import annotations

import numpy as np

import ipax


class EqualityQP(ipax.Problem):
    """Minimize ``0.5 * ||x||^2`` subject to ``x1 + x2 = 1``. Optimum (0.5, 0.5)."""

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        return 0.5 * float(np.sum(x * x))

    def gradient(self, x):
        return x

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del x, y_eq, y_ineq
        return sigma * np.eye(2, dtype=np.float64)

    def linear_eq(self):
        return np.asarray([[1.0, 1.0]], dtype=np.float64), np.asarray(
            [1.0], dtype=np.float64
        )


def main() -> None:
    expected_x = np.asarray([0.5, 0.5], dtype=np.float64)
    expected_y_eq = np.asarray([-0.5], dtype=np.float64)
    expected_objective = 0.25

    result = ipax.solve(
        EqualityQP(),
        np.asarray([0.9, 0.1], dtype=np.float64),
        options=ipax.Options(hessian="exact", linsolve="dense", verbose=1),
    )

    print(f"status: {result.status.value}")
    print(f"x: {result.x}")
    print(f"expected_x: {expected_x}")
    print(f"y_eq: {result.y_eq}")
    print(f"expected_y_eq: {expected_y_eq}")
    print(f"objective: {result.objective:.12g}")
    print(f"expected_objective: {expected_objective:.12g}")
    print(f"kkt_error: {result.kkt_error:.3e}")


if __name__ == "__main__":
    main()
