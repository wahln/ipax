"""Integration tests for two-sided linear inequalities (``l ≤ A x ≤ u``).

The solver lowers a constant-data ``linear_ineq`` block into the standard
one-sided inequality machinery, so these exercise the full IPM path (slacks,
multipliers, fraction-to-boundary, filter line search) across every backend.
"""

from __future__ import annotations

import pytest

from ipax import Options, Status, solve
from ipax.problem.base import Problem
from ipax.testing.problems import HS21
from tests._helpers import array, assert_allclose, assert_scalar_close

_EXACT_DENSE = Options(hessian="exact", linsolve="dense")


def _assert_optimal(result) -> None:
    assert result.status is Status.OPTIMAL
    assert result.kkt_error <= 1e-6
    assert result.constraint_violation <= 1e-6


class _LowerBoundedQP(Problem):
    """``min 0.5‖x - c‖²`` s.t. ``A x ≥ b`` (one-sided lower) via ``linear_ineq``.

    With ``c = (0, 0)``, ``A = [[1, 1]]`` and ``b = 2`` the unconstrained
    minimizer ``(0, 0)`` is infeasible; the optimum is the projection onto the
    line ``x1 + x2 = 2`` ⇒ ``(1, 1)``, ``f* = 1``.
    """

    def __init__(self, xp) -> None:
        self.xp = xp

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        return 0.5 * self.xp.sum(x * x)

    def gradient(self, x):
        return x

    def linear_ineq(self):
        xp = self.xp
        a = array(xp, [[1.0, 1.0]])
        return a, array(xp, [2.0]), array(xp, [float("inf")])

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del y_eq, y_ineq
        return sigma * self.xp.eye(2, dtype=x.dtype)

    def known_solution(self):
        return array(self.xp, [1.0, 1.0])


class _RangeBoxQP(Problem):
    """``min 0.5‖x - c‖²`` s.t. ``l ≤ A x ≤ u`` with both sides finite.

    ``c = (3, 0)``, ``A = [[1, 0]]`` (reads ``x1``), ``1 ≤ x1 ≤ 2``. The
    unconstrained min wants ``x1 = 3``; the active upper row drives ``x1 = 2``,
    ``x2 = 0``, ``f* = 0.5``.
    """

    def __init__(self, xp) -> None:
        self.xp = xp
        self.center = array(xp, [3.0, 0.0])

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        diff = x - self.center
        return 0.5 * self.xp.sum(diff * diff)

    def gradient(self, x):
        return x - self.center

    def linear_ineq(self):
        xp = self.xp
        a = array(xp, [[1.0, 0.0]])
        return a, array(xp, [1.0]), array(xp, [2.0])

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del y_eq, y_ineq
        return sigma * self.xp.eye(2, dtype=x.dtype)

    def known_solution(self):
        return array(self.xp, [2.0, 0.0])


class _MixedIneqQP(Problem):
    """A nonlinear inequality *and* a two-sided linear inequality together.

    ``min 0.5‖x - (3, 3)‖²`` s.t. ``‖x‖² ≤ 2`` (nonlinear) and ``x1 ≤ 5``
    (linear, ``-inf ≤ x1 ≤ 5``, deliberately slack). The disc constraint binds;
    the optimum lies on ``‖x‖² = 2`` nearest ``(3, 3)`` ⇒ ``(1, 1)`` (a single,
    non-degenerate active constraint), ``f* = 0.5·((1-3)² + (1-3)²) = 4``.
    """

    def __init__(self, xp) -> None:
        self.xp = xp
        self.center = array(xp, [3.0, 3.0])

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        diff = x - self.center
        return 0.5 * self.xp.sum(diff * diff)

    def gradient(self, x):
        return x - self.center

    def ineq_constraints(self, x):
        return self.xp.stack((x[0] * x[0] + x[1] * x[1] - 2.0,))

    def ineq_jacobian(self, x):
        xp = self.xp
        return xp.stack((xp.stack((2.0 * x[0], 2.0 * x[1])),))

    def linear_ineq(self):
        xp = self.xp
        a = array(xp, [[1.0, 0.0]])
        return a, array(xp, [float("-inf")]), array(xp, [5.0])

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del y_eq
        xp = self.xp
        # Only the nonlinear inequality contributes curvature (2·y·I); the linear
        # block must not be indexed here, so y_ineq is the nonlinear part only.
        assert int(y_ineq.shape[0]) == 1
        h = sigma + 2.0 * y_ineq[0]
        zero = xp.zeros_like(x[0])
        return xp.stack((xp.stack((h, zero)), xp.stack((zero, h))))

    def known_solution(self):
        return array(self.xp, [1.0, 1.0])


def test_lower_bounded_linear_inequality(namespace):
    problem = _LowerBoundedQP(namespace)
    result = solve(problem, array(namespace, [0.0, 0.0]), options=_EXACT_DENSE)
    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert_scalar_close(result.objective, 1.0, atol=1e-6)
    # One active lower row ⇒ a single positive inequality multiplier.
    assert result.y_ineq is not None
    assert int(result.y_ineq.shape[0]) == 1
    assert float(result.y_ineq[0]) > 0.0


def test_two_sided_range_upper_active(namespace):
    problem = _RangeBoxQP(namespace)
    result = solve(problem, array(namespace, [1.5, 0.0]), options=_EXACT_DENSE)
    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert_scalar_close(result.objective, 0.5, atol=1e-6)
    # Lowered to two rows (lower then upper); only the upper row is active.
    assert result.y_ineq is not None
    assert int(result.y_ineq.shape[0]) == 2


def test_mixed_nonlinear_and_linear_inequalities(namespace):
    problem = _MixedIneqQP(namespace)
    result = solve(problem, array(namespace, [0.5, 0.5]), options=_EXACT_DENSE)
    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert_scalar_close(result.objective, 4.0, atol=1e-6)
    # One nonlinear inequality + one lowered linear inequality.
    assert result.y_ineq is not None
    assert int(result.y_ineq.shape[0]) == 2


def test_linear_inequality_with_gradient_scaling(namespace):
    problem = _LowerBoundedQP(namespace)
    result = solve(
        problem,
        array(namespace, [0.0, 0.0]),
        options=Options(hessian="exact", linsolve="dense", scaling="gradient-based"),
    )
    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)


def test_hs21_via_linear_ineq_matches_known_optimum(namespace):
    problem = HS21(namespace)
    result = solve(problem, array(namespace, [3.0, 1.0]), options=_EXACT_DENSE)
    _assert_optimal(result)
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)
    assert_scalar_close(result.objective, -99.96, atol=1e-6)
    z_lower, z_upper = problem.known_bound_multipliers()
    assert_allclose(namespace, result.z_lower, z_lower, rtol=1e-6, atol=1e-6)
    assert_allclose(namespace, result.z_upper, z_upper, rtol=1e-6, atol=1e-6)


def test_matrix_free_linear_ineq_is_lowered_and_solved(namespace):
    # Formerly rejected (NotImplementedError): an operator-valued matrix is now
    # lowered through a signed row selection. ``0.5 ≤ x ≤ 1`` with the
    # objective pulling x to 0 ⇒ the lower rows bind at ``(0.5, 0.5)``.
    from ipax.backend.operators import Identity

    class _MatrixFreeLinearIneq(Problem):
        @property
        def n_vars(self) -> int:
            return 2

        def objective(self, x):
            return namespace.sum(x * x)

        def gradient(self, x):
            return 2.0 * x

        def linear_ineq(self):
            return (
                Identity(2),
                array(namespace, [0.5, 0.5]),
                array(namespace, [1.0, 1.0]),
            )

    result = solve(_MatrixFreeLinearIneq(), array(namespace, [0.75, 0.75]))
    _assert_optimal(result)
    assert_allclose(namespace, result.x, array(namespace, [0.5, 0.5]), atol=1e-6)
    assert_scalar_close(result.objective, 0.5, atol=1e-6)


# --------------------------------------------------------------------------- #
# Tall sparse ``linear_ineq`` (m ≫ n) declared as an operator.
#
# The same problem is declared twice — through ``linear_ineq`` with a
# ``CSROperator`` (lowered by the core at the triplet level) and through
# ``ineq_constraints``/``ineq_jacobian`` with the hand-lowered one-sided rows
# in the same order — and must solve identically on every linear-solver route,
# as well as satisfy the KKT conditions checked directly.
# --------------------------------------------------------------------------- #

_TALL_N, _TALL_M = 60, 3000


def _tall_data(seed: int = 17):
    """Sparse rows (3 nonzeros each) and a lower/upper/range/free bound mix.

    ``x = 0`` is strictly feasible (every finite lower bound < 0 < every finite
    upper bound) while the objective's centre lies outside the polytope, so a
    handful of rows are active at the optimum. (A centre much farther out
    activates dozens of rows; the matrix-free Krylov route then needs hundreds
    of iterations on this degenerate tall problem — a property of that route,
    identical for both declarations, and too slow for the per-PR suite.)
    """
    import random

    rng = random.Random(seed)
    rows, cols, vals = [], [], []
    for i in range(_TALL_M):
        for j in sorted(rng.sample(range(_TALL_N), 3)):
            rows.append(i)
            cols.append(j)
            vals.append(rng.uniform(-1.0, 1.0))
    lo, hi = [], []
    for i in range(_TALL_M):
        kind = ("upper", "lower", "range", "upper", "lower", "free")[i % 6]
        lo.append(-rng.uniform(0.5, 1.5) if kind in ("lower", "range") else -_INF)
        hi.append(rng.uniform(0.5, 1.5) if kind in ("upper", "range") else _INF)
    center = [rng.uniform(-0.5, 0.5) for _ in range(_TALL_N)]
    return rows, cols, vals, lo, hi, center


_INF = float("inf")


def _csr_from_triplets(xp, rows, cols, vals, m, n, **kwargs):
    from ipax.backend.operators import CSROperator

    indptr = [0] * (m + 1)
    for r in rows:
        indptr[r + 1] += 1
    for i in range(m):
        indptr[i + 1] += indptr[i]
    return CSROperator(
        xp.asarray(indptr, dtype=xp.int64),
        xp.asarray(cols, dtype=xp.int64),
        array(xp, vals),
        (m, n),
        **kwargs,
    )


class _TallQP(Problem):
    """``min ½‖x − c‖²`` subject to a tall sparse two-sided block."""

    def __init__(self, xp, data) -> None:
        self.xp = xp
        self._data = data
        self.center = array(xp, data[5])

    @property
    def n_vars(self) -> int:
        return _TALL_N

    def objective(self, x):
        diff = x - self.center
        return 0.5 * self.xp.sum(diff * diff)

    def gradient(self, x):
        return x - self.center

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del y_eq, y_ineq
        return sigma * self.xp.eye(_TALL_N, dtype=x.dtype)


class _TallViaLinearIneq(_TallQP):
    def __init__(self, xp, data) -> None:
        super().__init__(xp, data)
        rows, cols, vals, lo, hi, _ = data
        self._a = _csr_from_triplets(
            xp, rows, cols, vals, _TALL_M, _TALL_N, pattern_key="A"
        )
        self._lo = array(xp, lo)
        self._hi = array(xp, hi)

    def linear_ineq(self):
        return self._a, self._lo, self._hi


class _TallViaCallbacks(_TallQP):
    """The same rows hand-lowered: finite-lower rows negated, then upper rows."""

    def __init__(self, xp, data) -> None:
        super().__init__(xp, data)
        rows, cols, vals, lo, hi, _ = data
        by_row: list[list[tuple[int, float]]] = [[] for _ in range(_TALL_M)]
        for r, c, v in zip(rows, cols, vals, strict=True):
            by_row[r].append((c, v))
        g_rows, g_cols, g_vals, h = [], [], [], []
        k = 0
        for sign, bounds in ((-1.0, lo), (1.0, hi)):
            for i in range(_TALL_M):
                if bounds[i] in (_INF, -_INF):
                    continue
                for c, v in by_row[i]:
                    g_rows.append(k)
                    g_cols.append(c)
                    g_vals.append(sign * v)
                h.append(bounds[i] if sign < 0 else -bounds[i])
                k += 1
        self.jac = _csr_from_triplets(
            xp, g_rows, g_cols, g_vals, k, _TALL_N, pattern_key="G"
        )
        self._h = array(xp, h)

    def ineq_constraints(self, x):
        return self.jac.matvec(x) + self._h

    def ineq_jacobian(self, x):
        return self.jac


def _require_sparse_adapter(namespace):
    from ipax.backend.sparse import get_sparse_adapter

    if get_sparse_adapter(namespace) is None:
        pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")


_TALL_ROUTES = ("dense", "krylov", "sparse", "auto")


@pytest.mark.sparse
@pytest.mark.parametrize("linsolve", _TALL_ROUTES)
def test_tall_sparse_linear_ineq_matches_callback_declaration(namespace, linsolve):
    _require_sparse_adapter(namespace)
    xp = namespace
    data = _tall_data()
    options = Options(hessian="exact", linsolve=linsolve)
    x0 = xp.zeros((_TALL_N,), dtype=array(xp, [0.0]).dtype)

    via_linear = solve(_TallViaLinearIneq(xp, data), x0, options=options)
    reference_problem = _TallViaCallbacks(xp, data)
    via_callbacks = solve(reference_problem, x0, options=options)

    _assert_optimal(via_linear)
    assert via_linear.routes == via_callbacks.routes
    assert via_linear.n_iter == via_callbacks.n_iter
    assert_scalar_close(
        via_linear.objective, float(via_callbacks.objective), rtol=1e-10, atol=1e-10
    )
    assert abs(via_linear.kkt_error - via_callbacks.kkt_error) <= 1e-10
    assert_allclose(xp, via_linear.x, via_callbacks.x, rtol=1e-8, atol=1e-8)

    # KKT oracle, independent of the reference solve: stationarity
    # x − c + Gᵀy = 0, primal feasibility G x + h ≤ 0, dual feasibility y ≥ 0,
    # and complementarity y ∘ (G x + h) = 0 (bounds are absent).
    y = via_linear.y_ineq
    assert y is not None
    g = reference_problem.ineq_constraints(via_linear.x)
    assert int(y.shape[0]) == int(g.shape[0])
    residual = (
        via_linear.x - reference_problem.center + reference_problem.jac.rmatvec(y)
    )
    assert float(xp.max(xp.abs(residual))) <= 1e-6
    assert float(xp.max(g)) <= 1e-6
    assert float(xp.min(y)) >= -1e-8
    assert float(xp.max(xp.abs(y * g))) <= 1e-6
    # The problem is non-trivial: a meaningful set of rows is active.
    assert int(xp.sum(xp.astype(y > 1e-6, xp.int64))) >= 5


@pytest.mark.sparse
@pytest.mark.parametrize("linsolve", ("dense", "krylov", "sparse"))
def test_tall_sparse_linear_ineq_is_never_densified(namespace, linsolve, monkeypatch):
    # The dense route consumes the sparse block through ``gram`` (an n×n Gram,
    # never the m×n Jacobian), Krylov through matvecs and ``gram_diagonal``,
    # and the sparse route through ``to_coo``/``gram_coo``. None may fall back
    # to ``dense_matrix`` on the lowered block.
    if namespace.__name__.split(".")[-1] != "numpy":
        pytest.skip("the SciPy adapter is the one with a sparse Gram on every route")
    _require_sparse_adapter(namespace)
    from ipax.backend.operators import _SparseStructured

    def _forbidden(self, like=None):
        raise AssertionError("the lowered sparse linear_ineq block was densified")

    monkeypatch.setattr(_SparseStructured, "dense_matrix", _forbidden)
    xp = namespace
    x0 = xp.zeros((_TALL_N,), dtype=xp.float64)

    result = solve(
        _TallViaLinearIneq(xp, _tall_data()),
        x0,
        options=Options(hessian="exact", linsolve=linsolve),
    )

    _assert_optimal(result)
