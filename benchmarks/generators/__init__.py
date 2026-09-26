"""Synthetic RT-like problem generators.

Parametric builders emitting **block-structured sparse** Jacobians that emulate
dose-influence structure — controllable ``n ∈ [1e3, 1e5]``, density 5–50% (à la
Breedveld Table 1), number of "structures"/blocks, linear smoothing constraints,
and smoothed nonconvex DVH-like constraints. This exercises dimensionality and
sparsity **without** the real dose kernels (out of scope, §1.2).
"""

from __future__ import annotations

from typing import Literal

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


TallDeclaration = Literal["operator", "linear_ineq"]
_TALL_DECLARATIONS: tuple[str, ...] = ("operator", "linear_ineq")


class TallSparseQP(Problem):
    """Tall (``m ≫ n``) bound-constrained QP with a sparse inequality matrix.

    ``min ½‖x − t‖²  s.t.  l ≤ A x ≤ b,  x ≥ 0`` with a seeded random sparse
    ``A ≥ 0`` (RT-like row structure: a fixed number of entries per row). The
    pull target ``t`` sits *above* the caps ``b`` so a seeded slice of the
    rows is active at the solution — the shape that exercises the
    tall/condensed selection heuristics (``_TALL_DENSE_MAX_VARS`` /
    ``_TALL_ROW_EXCESS`` / the sparse-NE fill gate in ``linalg/solver.py``).

    ``l`` is ``−∞`` except on a seeded ``two_sided`` fraction of the rows,
    which get a finite lower bound strictly below the reference point
    ``x = ½·e`` (so it stays strictly feasible). The pull target keeps these
    lower sides inactive at the solution; they exist to exercise the
    range-row lowering, which turns each such row into two one-sided rows.

    ``declaration`` picks how the rows reach the solver. Both describe the
    *same* feasible set with the same one-sided row layout:

    * ``"operator"``: through the nonlinear inequality API
      (``ineq_constraints``/``ineq_jacobian``) with a constant Gram-capable
      :class:`ipax.CSROperator` Jacobian (``pattern_key`` set, so
      symbolic-reuse-ready). The generator lowers range rows itself, in the
      core's ``linear_ineq`` lowering order: every finite lower row
      ``−A_i x + l_i ≤ 0`` in source order, then every cap row
      ``A_i x − b_i ≤ 0``.
    * ``"linear_ineq"``: through :meth:`linear_ineq` as
      ``(CSROperator, l, b)``; the core lowers the operator at the triplet
      level without densifying it, and the problem declares no nonlinear
      inequalities (``ineq_constraints`` raises ``NotImplementedError``).
    """

    def __init__(
        self,
        n_vars: int,
        m_ineq: int,
        *,
        density: float = 0.01,
        seed: int = 0,
        bandwidth: float | None = None,
        declaration: TallDeclaration = "operator",
        two_sided: float = 0.0,
    ) -> None:
        import scipy.sparse

        if declaration not in _TALL_DECLARATIONS:
            raise ValueError(
                f"declaration must be one of {_TALL_DECLARATIONS}, got {declaration!r}"
            )
        if not 0.0 <= two_sided <= 1.0:
            raise ValueError("two_sided must be a fraction in [0, 1]")
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
        ax_feas = A @ x_feas
        half_row_sums = A @ np.ones(n_vars) * 0.5
        # Caps between the feasible reference and the pull target: a seeded
        # slice of the rows ends up active at the solution.
        self._b = ax_feas + rng.uniform(0.05, 0.5, size=m_ineq) * half_row_sums
        # Range rows, drawn after the caps so ``two_sided=0`` reproduces the
        # one-sided problem draw for draw. The lower bound mirrors the cap's
        # margin below the reference point, which therefore stays strictly
        # feasible.
        self._lower = np.full(m_ineq, -np.inf)
        if two_sided > 0.0:
            n_two = round(two_sided * m_ineq)
            picked = np.sort(rng.choice(m_ineq, size=n_two, replace=False))
            margin = rng.uniform(0.05, 0.5, size=n_two) * half_row_sums[picked]
            self._lower[picked] = ax_feas[picked] - margin
        self._two_rows = np.flatnonzero(np.isfinite(self._lower))
        self._target = np.full(n_vars, 1.5)
        self._n = n_vars
        self._m = m_ineq
        self._declaration: TallDeclaration = declaration
        self._G = None
        self._h = None
        if declaration == "operator":
            # Hand-lowered one-sided rows ``G x − h ≤ 0``, core lowering order.
            G = scipy.sparse.vstack([-A[self._two_rows], A], format="csr")
            G.sort_indices()
            self._G = G
            self._h = np.concatenate((-self._lower[self._two_rows], self._b))
        self._jac = None

    @property
    def declaration(self) -> TallDeclaration:
        """How the rows are declared (``"operator"`` or ``"linear_ineq"``)."""
        return self._declaration

    @property
    def m_lowered(self) -> int:
        """One-sided rows the solver sees: ``m`` plus one per range row."""
        return self._m + int(self._two_rows.size)

    @property
    def matrix(self):
        """The source matrix ``A`` (``scipy.sparse`` CSR)."""
        return self._A

    @property
    def row_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """``(l, b)``, with ``l = −∞`` on the one-sided rows."""
        return self._lower, self._b

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
        if self._G is None:
            raise NotImplementedError  # rows are declared through linear_ineq
        return self._G @ x - self._h

    def ineq_jacobian(self, x):
        del x
        if self._G is None:
            raise NotImplementedError  # rows are declared through linear_ineq
        if self._jac is None:
            self._jac = _csr_operator(self._G, "tall-G")
        return self._jac

    def linear_ineq(self):
        if self._G is not None:
            return None
        # Built fresh per call: the core lowers it once and keeps only the
        # lowered block.
        return _csr_operator(self._A, "tall-A"), self._lower, self._b

    def bounds(self):
        return np.zeros(self._n), None


def _csr_operator(M, pattern_key: str):
    """Wrap a ``scipy.sparse`` CSR matrix as a pattern-keyed ``CSROperator``.

    ``pattern_key`` names the matrix (``A`` vs the hand-lowered ``G``): the key
    promises a fixed sparsity pattern, so two different matrices must never
    share one.
    """
    from ipax import CSROperator

    return CSROperator(
        np.asarray(M.indptr),
        np.asarray(M.indices),
        np.asarray(M.data),
        M.shape,
        pattern_key=pattern_key,
    )


def make_tall_sparse_problem(
    n_vars: int,
    m_ineq: int,
    *,
    density: float = 0.01,
    seed: int = 0,
    bandwidth: float | None = None,
    declaration: TallDeclaration = "operator",
    two_sided: float = 0.0,
) -> TallSparseQP:
    """Build the tall sparse QP (NumPy application edge; see :class:`TallSparseQP`)."""
    return TallSparseQP(
        n_vars,
        m_ineq,
        density=density,
        seed=seed,
        bandwidth=bandwidth,
        declaration=declaration,
        two_sided=two_sided,
    )


__all__ = [
    "TallDeclaration",
    "TallSparseQP",
    "initial_point",
    "make_rt_like_problem",
    "make_tall_sparse_problem",
]
