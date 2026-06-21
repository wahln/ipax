"""Minimal dense solve for an unconstrained quadratic."""

from __future__ import annotations

import numpy as np

import ipax


def main() -> None:
    q_matrix = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    linear = np.asarray([-1.0, -2.0], dtype=np.float64)
    problem = ipax.QuadraticProblem(q_matrix, linear)
    expected_x = np.asarray([1.0 / 11.0, 7.0 / 11.0], dtype=np.float64)
    expected_objective = -15.0 / 22.0

    result = ipax.solve(
        problem,
        np.asarray([0.0, 0.0], dtype=np.float64),
        options=ipax.Options(hessian="exact", linsolve="dense", verbose=1),
    )

    print(f"status: {result.status.value}")
    print(f"x: {result.x}")
    print(f"expected_x: {expected_x}")
    print(f"objective: {result.objective:.12g}")
    print(f"expected_objective: {expected_objective:.12g}")
    print(f"kkt_error: {result.kkt_error:.3e}")


if __name__ == "__main__":
    main()
