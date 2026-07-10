"""Synthetic RT-like problem generators.

Parametric builders emitting **block-structured sparse** Jacobians that emulate
dose-influence structure — controllable ``n ∈ [1e3, 1e5]``, density 5–50% (à la
Breedveld Table 1), number of "structures"/blocks, linear smoothing constraints,
and smoothed nonconvex DVH-like constraints. This exercises dimensionality and
sparsity **without** the real dose kernels (out of scope, §1.2).
"""

from __future__ import annotations

import numpy as np

from ipax import Problem
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


class TallSparseQP(Problem):
    """Tall (``m ≫ n``) bound-constrained QP with a sparse inequality Jacobian.

    ``min ½‖x − t‖²  s.t.  A x ≤ b,  x ≥ 0`` with a seeded random sparse
    ``A ≥ 0`` (RT-like row structure: a fixed number of entries per row)
    supplied through a Gram-capable :class:`ipax.CSROperator` Jacobian
    (``pattern_key`` set → symbolic-reuse-ready). Declared through the
    nonlinear inequality API: the linear-block lowering requires a rank-2
    array matrix, and the whole point here is the *operator* Jacobian. The
    pull target ``t`` sits *above* the caps so a seeded slice of the rows
    is active at the solution — the shape that exercises the tall/condensed
    normal-equations selection heuristics (``_TALL_DENSE_MAX_VARS`` /
    ``_TALL_ROW_EXCESS`` in ``linalg/solver.py``).
    """

    def __init__(
        self,
        n_vars: int,
        m_ineq: int,
        *,
        density: float = 0.01,
        seed: int = 0,
        bandwidth: float | None = None,
    ) -> None:
        import scipy.sparse

        rng = np.random.default_rng(seed)
        nnz_per_row = max(1, round(density * n_vars))
        rows = np.repeat(np.arange(m_ineq), nnz_per_row)
        if bandwidth is None:
            # Global random columns: the Gram AᵀΣA fills toward dense — the
            # regime for the dense-condensed / Krylov routes.
            cols = rng.integers(0, n_vars, size=rows.size)
        else:
            # Localized rows: row i hits columns within a window of
            # ``bandwidth·n`` around its position, so AᵀΣA stays banded —
            # the sparse normal-equations regime (dose-influence-like).
            half = max(1, round(0.5 * bandwidth * n_vars))
            centers = (rows * n_vars) // m_ineq
            cols = np.clip(
                centers + rng.integers(-half, half + 1, size=rows.size),
                0,
                n_vars - 1,
            )
        vals = rng.uniform(0.1, 1.0, size=rows.size)
        A = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(m_ineq, n_vars)
        ).tocsr()
        A.sum_duplicates()
        self._A = A
        x_feas = np.full(n_vars, 0.5)
        # Caps between the feasible reference and the pull target: a seeded
        # slice of the rows ends up active at the solution.
        self._b = A @ x_feas + rng.uniform(0.05, 0.5, size=m_ineq) * (
            A @ np.ones(n_vars) * 0.5
        )
        self._target = np.full(n_vars, 1.5)
        self._n = n_vars
        self._m = m_ineq
        self._jac = None

    @property
    def n_vars(self) -> int:
        return self._n

    def objective(self, x):
        d = x - self._target
        return 0.5 * (d @ d)

    def gradient(self, x):
        return x - self._target

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        from ipax.backend.operators import Diagonal

        del x, y_eq, y_ineq
        return Diagonal(np.full(self._n, float(sigma)))

    def ineq_constraints(self, x):
        return self._A @ x - self._b

    def ineq_jacobian(self, x):
        from ipax import CSROperator

        del x
        if self._jac is None:
            self._jac = CSROperator(
                np.asarray(self._A.indptr),
                np.asarray(self._A.indices),
                np.asarray(self._A.data),
                (self._m, self._n),
                pattern_key="tall-A",
            )
        return self._jac

    def bounds(self):
        return np.zeros(self._n), None


def make_tall_sparse_problem(
    n_vars: int,
    m_ineq: int,
    *,
    density: float = 0.01,
    seed: int = 0,
    bandwidth: float | None = None,
) -> TallSparseQP:
    """Build the tall sparse QP (NumPy application edge; see :class:`TallSparseQP`)."""
    return TallSparseQP(n_vars, m_ineq, density=density, seed=seed, bandwidth=bandwidth)


__all__ = [
    "TallSparseQP",
    "initial_point",
    "make_rt_like_problem",
    "make_tall_sparse_problem",
]
