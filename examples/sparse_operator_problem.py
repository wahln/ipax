"""Supplying a sparse Jacobian/Hessian via the public sparse operators.

``ipax.COOOperator`` (and its ``CSROperator``/``CSCOperator`` siblings) let a
``Problem`` hand the solver a sparse Jacobian or Hessian as plain Array-API
index/value vectors — no concrete sparse library in the model, so the same
problem runs on any backend. The solver emits the operator's COO structure and
factorizes it through the sparse-direct route (``linsolve="sparse"``).

This is a tiny equality-constrained QP::

    min ½ xᵀQ x + bᵀx   s.t.   C x = d

with the SPD Hessian ``Q`` and the constraint Jacobian ``C`` both declared as
sparse operators. ``pattern_key`` marks each pattern fixed across the solve, so
the sparse backend reuses its symbolic analysis between iterations.
"""

from __future__ import annotations

import numpy as np

import ipax


def main() -> None:
    # Q = [[4,1,0],[1,3,1],[0,1,2]] (SPD, tridiagonal), b = [1,2,3].
    q_rows = np.asarray([0, 0, 1, 1, 1, 2, 2])
    q_cols = np.asarray([0, 1, 0, 1, 2, 1, 2])
    q_vals = np.asarray([4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0])
    Q = np.asarray([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b = np.asarray([1.0, 2.0, 3.0])

    # C = [[1, 1, 1]], d = [1]: the variables must sum to one.
    C = np.asarray([[1.0, 1.0, 1.0]])
    d = np.asarray([1.0])

    class EqualityQP(ipax.Problem):
        @property
        def n_vars(self) -> int:
            return 3

        def objective(self, x):
            return 0.5 * float(x @ Q @ x) + float(b @ x)

        def gradient(self, x):
            return Q @ x + b

        def eq_constraints(self, x):
            return C @ x - d

        def eq_jacobian(self, x):
            # Sparse constraint Jacobian as COO triplets (1 x 3).
            return ipax.COOOperator(
                np.asarray([0, 0, 0]),
                np.asarray([0, 1, 2]),
                np.asarray([1.0, 1.0, 1.0]),
                (1, 3),
                pattern_key="C",
            )

        def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
            # Sparse, symmetric Hessian of the Lagrangian (here just sigma * Q,
            # since the constraint is linear). pattern_key enables symbolic reuse.
            return ipax.COOOperator(
                q_rows, q_cols, sigma * q_vals, (3, 3), symmetric=True, pattern_key="Q"
            )

    result = ipax.solve(
        EqualityQP(),
        np.asarray([0.2, 0.1, 0.3]),
        options=ipax.Options(linsolve="sparse", hessian="exact"),
    )

    print(f"status     = {result.status.name}")
    print(f"x*         = {np.asarray(result.x)}")
    print(f"sum(x*)    = {float(np.sum(result.x)):.6f}  (constraint target: 1.0)")
    print(f"objective  = {result.objective:.6f}")
    print(f"linsolver  = {result.linear_solver}")


if __name__ == "__main__":
    main()
