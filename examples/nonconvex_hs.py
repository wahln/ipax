"""Minimal nonconvex equality solve: Hock-Schittkowski problem 6.

HS6: ``min (1 - x1)^2`` s.t. ``10 (x2 - x1^2) = 0``; optimum (1, 1), f = 0.
The filter line-search and the exact (multiplier-aware) Hessian handle the
nonconvex equality constraint.
"""

from __future__ import annotations

import numpy as np

import ipax
from ipax.testing.problems import HS6


def main() -> None:
    problem = HS6(np)
    expected_x = problem.known_solution()
    expected_y_eq = np.asarray([0.0], dtype=np.float64)
    expected_objective = 0.0
    expected_constraint_violation = 0.0

    result = ipax.solve(
        problem,
        np.asarray([-1.2, 1.0], dtype=np.float64),
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
    print(f"constraint_violation: {result.constraint_violation:.3e}")
    print(f"expected_constraint_violation: {expected_constraint_violation:.3e}")


if __name__ == "__main__":
    main()
