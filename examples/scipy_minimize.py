"""Run a ``scipy.optimize.minimize`` model on ipax through ``minimize_ipax``.

Hock–Schittkowski problem 71 as a SciPy user would write it: ``Bounds`` plus
two ``NonlinearConstraint`` objects with exact Jacobians and Hessians. The call
differs from a ``trust-constr`` one only in ``method=minimize_ipax``; the result
differs slightly (see "Differences from trust-constr" in docs/guide/scipy.md).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from ipax.interfaces.scipy import minimize_ipax


def objective(x):
    return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]


def gradient(x):
    return np.array(
        [
            x[3] * (2 * x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
        ]
    )


def hessian(x):
    s = 2 * x[0] + x[1] + x[2]
    return np.array(
        [
            [2 * x[3], x[3], x[3], s],
            [x[3], 0.0, 0.0, x[0]],
            [x[3], 0.0, 0.0, x[0]],
            [s, x[0], x[0], 0.0],
        ]
    )


def product_hessian(x, v):
    H = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i != j:
                H[i, j] = np.prod([x[k] for k in range(4) if k not in (i, j)])
    return v[0] * H


def main() -> None:
    constraints = [
        NonlinearConstraint(  # x0·x1·x2·x3 ≥ 25
            lambda x: np.array([np.prod(x)]),
            25.0,
            np.inf,
            jac=lambda x: (np.prod(x) / x)[None, :],
            hess=product_hessian,
        ),
        NonlinearConstraint(  # Σ xᵢ² = 40
            lambda x: np.array([x @ x]),
            40.0,
            40.0,
            jac=lambda x: (2.0 * x)[None, :],
            hess=lambda x, v: 2.0 * v[0] * np.eye(4),
        ),
    ]
    res = minimize(
        objective,
        np.array([1.0, 5.0, 5.0, 1.0]),
        jac=gradient,
        hess=hessian,
        bounds=Bounds(np.ones(4), 5.0 * np.ones(4)),
        constraints=constraints,
        method=minimize_ipax,
    )

    print(f"success: {res.success} ({res.ipax_status}), nit = {res.nit}")
    print(f"x: {res.x}")
    print(f"fun: {res.fun:.10f}   (published optimum 17.0140173)")
    print(f"multipliers v: {[float(v[0]) for v in res.v]}")
    print(f"Hessian source: {res.ipax_result.routes.hessian}")


if __name__ == "__main__":
    main()
