"""Synthetic RT-like problem generators.

Parametric builders emitting **block-structured sparse** Jacobians that emulate
dose-influence structure — controllable ``n ∈ [1e3, 1e5]``, density 5–50% (à la
Breedveld Table 1), number of "structures"/blocks, linear smoothing constraints,
and smoothed nonconvex DVH-like constraints. This exercises dimensionality and
sparsity **without** the real dose kernels (out of scope, §1.2).
"""

from __future__ import annotations

import numpy as np

from ipax.testing.problems import SyntheticRTProblem
from ipax.testing.problems import make_rt_like_problem as _make_rt_like_problem


def make_rt_like_problem(
    n_vars: int,
    *,
    n_structures: int = 8,
    density: float = 0.2,
    seed: int = 0,
) -> SyntheticRTProblem:
    """Build a synthetic, NumPy-backed RT-like NLP with block structure.

    Thin application-edge wrapper around the backend-agnostic builder in
    ``ipax.testing.problems`` (shared with the cross-backend tests, §8.2); the
    benchmark layer pins it to the NumPy backend. The block-structured
    diagonal-plus-low-rank Hessian and per-structure dose caps are fully
    matrix-free, so the matrix-free Krylov route never densifies an ``n×n``
    matrix even at ``n = 1e5``.
    """
    return _make_rt_like_problem(
        np, n_vars, n_structures=n_structures, density=density, seed=seed
    )


def initial_point(n_vars: int) -> np.ndarray:
    """A strictly feasible cold start (every dose cap is positive)."""
    return np.full(n_vars, 0.01, dtype=np.float64)


__all__ = ["initial_point", "make_rt_like_problem"]
