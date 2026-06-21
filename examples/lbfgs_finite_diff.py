"""Minimal example: derivative resolution + L-BFGS, NumPy backend.

Demonstrates the capability-graded problem interface on a backend with
**no autodiff**:

- the objective is supplied as a plain callable, with **no** gradient and **no**
  Hessian. NumPy is not an autodiff backend, so the resolver fills the gradient
  by **finite differences** and the Lagrangian Hessian by Powell-damped L-BFGS;
- ``result.derivative_sources`` reports exactly how each derivative was obtained
  (here: ``gradient='finite-diff'``, ``hessian='lbfgs'``).

For a true forward/reverse-mode autodiff gradient, see ``autodiff.py``.

HS1 Rosenbrock: ``min 100 (x2 - x1^2)^2 + (1 - x1)^2``; optimum (1, 1), f = 0.
"""

from __future__ import annotations

import numpy as np

import ipax


def rosenbrock(x: np.ndarray) -> np.ndarray:
    return 100.0 * (x[1] - x[0] * x[0]) ** 2 + (1.0 - x[0]) ** 2


def main() -> None:
    # Only the objective and the dimension are provided; everything else is
    # resolved by precedence (analytic -> autodiff -> finite-difference for the
    # gradient; analytic -> autodiff-HVP -> L-BFGS for the Hessian). On NumPy,
    # autodiff is unavailable, so the gradient resolves to finite differences.
    problem = ipax.FunctionProblem(2, rosenbrock)

    result = ipax.solve(
        problem,
        np.asarray([-1.2, 1.0], dtype=np.float64),
        options=ipax.Options(hessian="lbfgs", linsolve="dense", verbose=1),
    )

    print(f"status: {result.status.value}")
    print(f"x: {result.x}")
    print("expected_x: [1. 1.]")
    print(f"objective: {result.objective:.12g}")
    print("expected_objective: 0")
    print(f"kkt_error: {result.kkt_error:.3e}")
    print(
        f"gradient source: {result.derivative_sources.gradient}  (expected: finite-diff)"
    )
    print(f"hessian source: {result.derivative_sources.hessian}  (expected: lbfgs)")
    print(f"iterations: {result.n_iter}")


if __name__ == "__main__":
    main()
