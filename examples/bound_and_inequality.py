"""Minimal custom problem with an active inequality constraint."""

from __future__ import annotations

import numpy as np

import ipax


class OneDimensionalInequality(ipax.Problem):
    """Minimize ``0.5 * (x - 2)^2`` subject to ``x <= 1``."""

    @property
    def n_vars(self) -> int:
        return 1

    def objective(self, x):
        return 0.5 * (x[0] - 2.0) * (x[0] - 2.0)

    def gradient(self, x):
        return np.stack((x[0] - 2.0,))

    def ineq_constraints(self, x):
        return np.stack((x[0] - 1.0,))

    def ineq_jacobian(self, x):
        del x
        return np.asarray([[1.0]], dtype=np.float64)

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del x, y_eq, y_ineq
        return sigma * np.asarray([[1.0]], dtype=np.float64)


def main() -> None:
    expected_x = np.asarray([1.0], dtype=np.float64)
    expected_y_ineq = np.asarray([1.0], dtype=np.float64)
    expected_objective = 0.5

    result = ipax.solve(
        OneDimensionalInequality(),
        np.asarray([0.5], dtype=np.float64),
        options=ipax.Options(hessian="exact", linsolve="dense", verbose=1),
    )

    print(f"status: {result.status.value}")
    print(f"x: {result.x}")
    print(f"expected_x: {expected_x}")
    print(f"y_ineq: {result.y_ineq}")
    print(f"expected_y_ineq: {expected_y_ineq}")
    print(f"objective: {result.objective:.12g}")
    print(f"expected_objective: {expected_objective:.12g}")
    print(f"kkt_error: {result.kkt_error:.3e}")


if __name__ == "__main__":
    main()
