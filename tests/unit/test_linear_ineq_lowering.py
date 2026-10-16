"""Unit tests for two-sided linear-inequality lowering.

These check the algebra of :func:`ipax.problem.linear_ineq.lower_linear_inequalities`
directly (one-sided rows, signs, offsets, row selection) independent of a solve.
"""

from __future__ import annotations

import math
import random

import pytest

from ipax.backend.namespace import array_namespace
from ipax.backend.operators import (
    COOOperator,
    CSCOperator,
    CSROperator,
    Dense,
    LinearOperator,
    MatrixFreeJacobian,
    VStack,
)
from ipax.backend.sparse import get_sparse_adapter
from ipax.problem.base import Problem
from ipax.problem.linear_ineq import (
    LoweredLinearIneqProblem,
    _lower_two_sided,
    lower_linear_inequalities,
)
from tests._helpers import array, assert_allclose, transpose


class _LinearIneqOnly(Problem):
    """Minimal problem carrying only a two-sided ``linear_ineq`` block."""

    def __init__(self, xp, a, lower, upper) -> None:
        self.xp = xp
        self._a = a
        self._lower = lower
        self._upper = upper

    @property
    def n_vars(self) -> int:
        return int(self._a.shape[1])

    def objective(self, x):
        return self.xp.sum(x * x)

    def gradient(self, x):
        return 2.0 * x

    def linear_ineq(self):
        return self._a, self._lower, self._upper


def _lowered_g(namespace, a_rows, lower, upper, x_values):
    a = array(namespace, a_rows)
    problem = _LinearIneqOnly(
        namespace, a, array(namespace, lower), array(namespace, upper)
    )
    lowered = lower_linear_inequalities(
        problem, array(namespace, [0.0] * a.shape[1]), namespace
    )
    assert lowered.linear_ineq() is None  # lowered away
    x = array(namespace, x_values)
    return lowered.ineq_constraints(x)


def test_lower_only_row_is_negated(namespace):
    # 10 x1 - x2 ≥ 10  ⇒  g = 10 - 10 x1 + x2 ≤ 0.
    g = _lowered_g(namespace, [[10.0, -1.0]], [10.0], [float("inf")], [1.0, 2.0])
    assert int(g.shape[0]) == 1
    assert_allclose(namespace, g, array(namespace, [10.0 - 10.0 + 2.0]), atol=1e-9)


def test_upper_only_row_keeps_sign(namespace):
    # x1 ≤ 3  ⇒  g = x1 - 3 ≤ 0.
    g = _lowered_g(namespace, [[1.0, 0.0]], [float("-inf")], [3.0], [5.0, 0.0])
    assert int(g.shape[0]) == 1
    assert_allclose(namespace, g, array(namespace, [2.0]), atol=1e-9)


def test_two_sided_row_yields_lower_then_upper(namespace):
    # 1 ≤ x1 ≤ 4 at x1 = 2 ⇒ lower g = 1 - 2 = -1, upper g = 2 - 4 = -2.
    g = _lowered_g(namespace, [[1.0, 0.0]], [1.0], [4.0], [2.0, 0.0])
    assert int(g.shape[0]) == 2
    assert_allclose(namespace, g, array(namespace, [-1.0, -2.0]), atol=1e-9)


def test_free_row_is_dropped(namespace):
    # Row 0 has both bounds infinite ⇒ contributes nothing; row 1 (x2 ≤ 1) stays.
    g = _lowered_g(
        namespace,
        [[1.0, 0.0], [0.0, 1.0]],
        [float("-inf"), float("-inf")],
        [float("inf"), 1.0],
        [9.0, 3.0],
    )
    assert int(g.shape[0]) == 1
    assert_allclose(namespace, g, array(namespace, [2.0]), atol=1e-9)


def test_inverted_range_raises(namespace):
    problem = _LinearIneqOnly(
        namespace,
        array(namespace, [[1.0, 0.0]]),
        array(namespace, [5.0]),
        array(namespace, [1.0]),
    )
    with pytest.raises(ValueError, match="lower bound exceeds"):
        lower_linear_inequalities(problem, array(namespace, [0.0, 0.0]), namespace)


def test_bound_shape_mismatch_raises(namespace):
    problem = _LinearIneqOnly(
        namespace,
        array(namespace, [[1.0, 0.0], [0.0, 1.0]]),
        array(namespace, [0.0]),
        array(namespace, [1.0]),
    )
    with pytest.raises(ValueError, match="match the matrix row count"):
        lower_linear_inequalities(problem, array(namespace, [0.0, 0.0]), namespace)


# --------------------------------------------------------------------------- #
# Operator-valued ``linear_ineq`` matrices (no densification).
#
# Every matrix kind is checked against the *reference* lowering computed here
# in plain Python: per block (one block unless the matrix is a ``VStack``), the
# finite-lower rows negated, then the finite-upper rows, with offsets ``l_i``
# and ``−u_i``. Sparse kinds are lowered at the triplet level and any other
# operator through a signed row-selection wrapper; both must agree with the
# reference on every capability the solver routes consume.
# --------------------------------------------------------------------------- #

_INF = float("inf")
_M, _N = 12, 5
_SPLIT = 7  # VStack kind: rows [0, 7) float32-sourced, [7, 12) float64
_TOL = {"rtol": 1e-12, "atol": 1e-12}


def _random_block(seed: int = 3):
    """Sparse-ish 12×5 rows and a mix of lower/upper/range/free row bounds."""
    rng = random.Random(seed)
    rows = [
        [
            round(rng.uniform(-2.0, 2.0), 3) if rng.random() < 0.55 else 0.0
            for _ in range(_N)
        ]
        for _ in range(_M)
    ]
    rows[5] = [0.0] * _N  # an all-zero row must survive lowering as a zero row
    kinds = ("lower", "upper", "range", "free")
    lo, hi = [], []
    for i in range(_M):
        kind = kinds[i % 4]
        a, b = sorted(
            (round(rng.uniform(-3.0, 3.0), 3), round(rng.uniform(-3.0, 3.0), 3))
        )
        lo.append(a if kind in ("lower", "range") else -_INF)
        hi.append(b if kind in ("upper", "range") else _INF)
    return rows, lo, hi


def _reference(rows, lo, hi, block_sizes):
    jac, off = [], []
    start = 0
    for size in block_sizes:
        span = range(start, start + size)
        for i in span:
            if math.isfinite(lo[i]):
                jac.append([-v for v in rows[i]])
                off.append(lo[i])
        for i in span:
            if math.isfinite(hi[i]):
                jac.append(list(rows[i]))
                off.append(-hi[i])
        start += size
    return jac, off


def _triplets(xp, rows):
    r, c, v = [], [], []
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            if value != 0.0:
                r.append(i)
                c.append(j)
                v.append(value)
    return (
        xp.asarray(r, dtype=xp.int64),
        xp.asarray(c, dtype=xp.int64),
        array(xp, v),
    )


def _csr(xp, rows, **kwargs):
    _, c, v = _triplets(xp, rows)
    indptr = [0]
    for row in rows:
        indptr.append(indptr[-1] + sum(1 for value in row if value != 0.0))
    return CSROperator(
        xp.asarray(indptr, dtype=xp.int64), c, v, (len(rows), _N), **kwargs
    )


def _csc(xp, rows):
    r, v = [], []
    indptr = [0]
    for j in range(_N):
        for i, row in enumerate(rows):
            if row[j] != 0.0:
                r.append(i)
                v.append(row[j])
        indptr.append(len(r))
    return CSCOperator(
        xp.asarray(indptr, dtype=xp.int64),
        xp.asarray(r, dtype=xp.int64),
        array(xp, v),
        (len(rows), _N),
    )


class _Opaque(LinearOperator):
    """A non-``Dense`` explicit operator exposing the full Gram surface.

    Stands in for any user operator the triplet path cannot see into: the
    lowering must wrap it (signed row selection) and fold the selection into
    the Gram weights rather than densify.
    """

    def __init__(self, a) -> None:
        self._a = a

    @property
    def shape(self):
        return int(self._a.shape[0]), int(self._a.shape[1])

    def matvec(self, v):
        return Dense(self._a).matvec(v)

    def rmatvec(self, v):
        return Dense(self._a).rmatvec(v)

    def dense_matrix(self, like=None):
        return self._a

    def gram(self, weights, *, accumulate_dtype=None, hinted_only=False):
        del accumulate_dtype, hinted_only
        xp = array_namespace(self._a)
        weighted = xp.expand_dims(weights, axis=1) * self._a
        return xp.matmul(transpose(xp, self._a), weighted)

    def gram_diagonal(self, weights):
        return Dense(self._a).gram_diagonal(weights)

    def row_gram_diagonal(self, weights):
        return Dense(self._a).row_gram_diagonal(weights)

    def row_inf_norms(self, like=None):
        return Dense(self._a).row_inf_norms(like)


_SPARSE_KINDS = ("coo", "csr", "csc", "vstack")
_ALL_KINDS = ("array", "dense", *_SPARSE_KINDS, "opaque")


def _make(xp, kind, rows):
    """``(matrix, block_sizes)`` for the given kind over the same data."""
    if kind == "array":
        return array(xp, rows), [_M]
    if kind == "dense":
        return Dense(array(xp, rows)), [_M]
    if kind == "coo":
        r, c, v = _triplets(xp, rows)
        return COOOperator(r, c, v, (_M, _N), pattern_key="A"), [_M]
    if kind == "csr":
        return _csr(xp, rows, pattern_key="A"), [_M]
    if kind == "csc":
        return _csc(xp, rows), [_M]
    if kind == "vstack":
        top = _csr(xp, rows[:_SPLIT], values_dtype_hint="float32", pattern_key="t")
        bottom = _csr(xp, rows[_SPLIT:])
        return VStack((top, bottom)), [_SPLIT, _M - _SPLIT]
    if kind == "opaque":
        return _Opaque(array(xp, rows)), [_M]
    raise AssertionError(kind)


def _needs_adapter(namespace, kind):
    if kind in _SPARSE_KINDS and get_sparse_adapter(namespace) is None:
        pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")


def _coo_dense(xp, r, c, v, shape):
    """Host-side densify of COO triplets (duplicates summed) — test oracle only."""
    out = [[0.0] * shape[1] for _ in range(shape[0])]
    for k in range(int(v.shape[0])):
        out[int(r[k])][int(c[k])] += float(v[k])
    return array(xp, out)


def _weights(xp, m, seed=11):
    rng = random.Random(seed)
    return array(xp, [rng.uniform(0.25, 4.0) for _ in range(m)])


def _lower(xp, matrix, lo, hi):
    return _lower_two_sided(matrix, array(xp, lo), array(xp, hi), xp)


def _with_block(xp, rows, matrix, lo, hi):
    """A ``_LinearIneqOnly`` problem whose ``linear_ineq`` returns ``matrix``."""
    problem = _LinearIneqOnly(xp, array(xp, rows), array(xp, lo), array(xp, hi))
    problem._a = matrix
    return problem


@pytest.mark.parametrize("kind", _ALL_KINDS)
def test_operator_lowering_matches_reference(namespace, kind):
    _needs_adapter(namespace, kind)
    xp = namespace
    rows, lo, hi = _random_block()
    matrix, sizes = _make(xp, kind, rows)
    jac_ref, off_ref = _reference(rows, lo, hi, sizes)
    J = array(xp, jac_ref)
    Jt = transpose(xp, J)
    k = len(jac_ref)

    op, offset = _lower(xp, matrix, lo, hi)

    assert op.shape == (k, _N)
    assert_allclose(xp, offset, array(xp, off_ref), **_TOL)
    x = array(xp, [0.3, -1.2, 2.0, 0.7, -0.4])
    v = _weights(xp, k, seed=5) - 2.0
    X = xp.stack((x, 2.0 * x - 1.0), axis=1)
    V = xp.stack((v, 1.0 - v), axis=1)
    assert_allclose(xp, op.matvec(x), xp.matmul(J, x), **_TOL)
    assert_allclose(xp, op.rmatvec(v), xp.matmul(Jt, v), **_TOL)
    assert_allclose(xp, op.matmat(X), xp.matmul(J, X), **_TOL)
    assert_allclose(xp, op.rmatmat(V), xp.matmul(Jt, V), **_TOL)
    assert_allclose(xp, op.dense_matrix(), J, **_TOL)
    assert_allclose(xp, op.row_inf_norms(x), xp.max(xp.abs(J), axis=1), **_TOL)

    w = _weights(xp, k)
    col_energy = xp.sum(xp.expand_dims(w, axis=1) * J * J, axis=0)
    assert_allclose(xp, op.gram_diagonal(w), col_energy, **_TOL)
    wn = _weights(xp, _N, seed=7)
    row_energy = xp.sum(J * J * xp.expand_dims(wn, axis=0), axis=1)
    assert_allclose(xp, op.row_gram_diagonal(wn), row_energy, **_TOL)

    # Capability probes forward the source's (no densify fallback smuggled in).
    source = matrix if isinstance(matrix, LinearOperator) else Dense(matrix)
    gram_ref = xp.matmul(Jt, xp.expand_dims(w, axis=1) * J)
    assert op.gram_capable() == source.gram_capable()
    if op.gram_capable():
        assert_allclose(xp, op.gram(w), gram_ref, **_TOL)
        reduced = op.gram(w, accumulate_dtype="float32", hinted_only=True)
        assert_allclose(xp, reduced, gram_ref, rtol=1e-5, atol=1e-5)
    assert op.gram_coo_capable() == source.gram_coo_capable()
    if op.gram_coo_capable():
        r, c, g, shape = op.gram_coo(w)
        assert shape == (_N, _N)
        assert_allclose(xp, _coo_dense(xp, r, c, g, shape), gram_ref, **_TOL)


@pytest.mark.parametrize("kind", _SPARSE_KINDS)
def test_sparse_lowering_is_triplet_level_and_keeps_the_family(namespace, kind):
    # Pure Array API on the index/value vectors: runs on every backend, with or
    # without a sparse adapter (the adapter is needed only for the algebra).
    xp = namespace
    rows, lo, hi = _random_block()
    matrix, sizes = _make(xp, kind, rows)
    jac_ref, _ = _reference(rows, lo, hi, sizes)

    op, _ = _lower(xp, matrix, lo, hi)

    r, c, v, shape = op.to_coo()
    assert shape == (len(jac_ref), _N)
    dense = _coo_dense(xp, r, c, v, shape)
    assert_allclose(xp, dense, array(xp, jac_ref), atol=0, rtol=0)
    assert_allclose(xp, op.coo_values(), v, atol=0, rtol=0)
    expected = {"coo": COOOperator, "csr": CSROperator, "csc": CSCOperator}
    if kind == "vstack":
        assert isinstance(op, VStack)
        assert all(type(block) is CSROperator for block in op.blocks)
    else:
        assert type(op) is expected[kind]
    # No structural entries are invented: nnz is exactly the selected rows'.
    source_nnz = [sum(1 for value in row if value != 0.0) for row in rows]
    n_expected = sum(
        source_nnz[i] * (int(math.isfinite(lo[i])) + int(math.isfinite(hi[i])))
        for i in range(_M)
    )
    assert int(v.shape[0]) == n_expected


def test_vstack_lowering_keeps_per_block_precision_hints(namespace):
    # gram_dtype="auto" reduces the Gram accumulate block by block, so a
    # float32-sourced block must stay its own block after lowering.
    xp = namespace
    rows, lo, hi = _random_block()
    matrix, _ = _make(xp, "vstack", rows)

    op, _ = _lower(xp, matrix, lo, hi)

    top, bottom = op.blocks
    assert top.gram_accumulate_dtype_hint() == "float32"
    assert bottom.gram_accumulate_dtype_hint() is None
    assert op.gram_accumulate_dtype_hint() == "float32"


def test_sparse_lowering_keeps_a_declared_values_hint(namespace):
    xp = namespace
    rows, lo, hi = _random_block()

    op, _ = _lower(xp, _csr(xp, rows, values_dtype_hint="float32"), lo, hi)

    assert op.gram_accumulate_dtype_hint() == "float32"


def test_sparse_lowering_derives_a_stable_pattern_key(namespace):
    xp = namespace
    rows, lo, hi = _random_block()
    keyed = _csr(xp, rows, pattern_key="A")

    op, _ = _lower(xp, keyed, lo, hi)
    plain, _ = _lower(xp, _csr(xp, rows), lo, hi)

    signature = op.coo_pattern_signature()
    assert signature is not None
    assert signature == op.coo_pattern_signature()  # stable across calls
    assert signature != keyed.coo_pattern_signature()  # a different pattern
    # A second lowering is a different operator: it must not alias the first.
    again, _ = _lower(xp, keyed, lo, hi)
    assert again.coo_pattern_signature() != signature
    # No source key ⇒ no promise: the lowering stays conservative.
    assert plain.coo_pattern_signature() is None


@pytest.mark.parametrize("kind", ("csr", "opaque", "vstack"))
def test_operator_lowering_rejects_inverted_ranges(namespace, kind):
    xp = namespace
    rows, lo, hi = _random_block()
    lo[2], hi[2] = 1.0, -1.0
    matrix, _ = _make(xp, kind, rows)
    with pytest.raises(ValueError, match="lower bound exceeds"):
        _lower(xp, matrix, lo, hi)


@pytest.mark.parametrize("kind", ("csr", "opaque", "vstack"))
def test_operator_lowering_rejects_bound_length_mismatch(namespace, kind):
    xp = namespace
    rows, lo, hi = _random_block()
    matrix, _ = _make(xp, kind, rows)
    with pytest.raises(ValueError, match="match the matrix row count"):
        _lower(xp, matrix, lo[:-1], hi[:-1])


@pytest.mark.parametrize("kind", ("csr", "opaque", "vstack"))
def test_all_unbounded_operator_block_lowers_to_nothing(namespace, kind):
    _needs_adapter(namespace, kind)
    xp = namespace
    rows, _, _ = _random_block()
    matrix, _ = _make(xp, kind, rows)
    problem = _with_block(xp, rows, matrix, [-_INF] * _M, [_INF] * _M)

    lowered = lower_linear_inequalities(problem, array(xp, [0.0] * _N), xp)

    x = array(xp, [1.0] * _N)
    assert int(lowered.ineq_constraints(x).shape[0]) == 0
    assert lowered.ineq_jacobian(x).shape == (0, _N)


@pytest.mark.parametrize("kind", ("csr", "vstack", "opaque"))
def test_operator_block_without_nonlinear_inequalities(namespace, kind):
    _needs_adapter(namespace, kind)
    xp = namespace
    rows, lo, hi = _random_block()
    matrix, sizes = _make(xp, kind, rows)
    jac_ref, off_ref = _reference(rows, lo, hi, sizes)
    problem = _with_block(xp, rows, matrix, lo, hi)

    lowered = lower_linear_inequalities(problem, array(xp, [0.0] * _N), xp)

    x = array(xp, [0.5, 0.25, -1.0, 2.0, 0.0])
    expected = xp.matmul(array(xp, jac_ref), x) + array(xp, off_ref)
    assert_allclose(xp, lowered.ineq_constraints(x), expected, **_TOL)
    jac = lowered.ineq_jacobian(x)
    assert_allclose(xp, jac.matvec(x), xp.matmul(array(xp, jac_ref), x), **_TOL)


class _NonlinearPlusLinear(Problem):
    """One nonlinear inequality ``‖x‖² − 4 ≤ 0`` plus an operator ``linear_ineq``."""

    def __init__(self, xp, matrix, lo, hi) -> None:
        self.xp = xp
        self._matrix = matrix
        self._lo = lo
        self._hi = hi
        self.seen_y_ineq = None

    @property
    def n_vars(self) -> int:
        return _N

    def objective(self, x):
        return self.xp.sum(x * x)

    def gradient(self, x):
        return 2.0 * x

    def ineq_constraints(self, x):
        return self.xp.stack((self.xp.sum(x * x) - 4.0,))

    def ineq_jacobian(self, x):
        return self.xp.stack((2.0 * x,))

    def linear_ineq(self):
        return self._matrix, self._lo, self._hi

    def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
        del y_eq
        self.seen_y_ineq = y_ineq
        return (sigma + 2.0 * y_ineq[0]) * self.xp.eye(_N, dtype=x.dtype)


@pytest.mark.parametrize(
    ("kind", "n_blocks"), [("csr", 1), ("vstack", 2), ("opaque", 1)]
)
def test_operator_block_multipliers_are_sliced_off(namespace, kind, n_blocks):
    _needs_adapter(namespace, kind)
    xp = namespace
    rows, lo, hi = _random_block()
    matrix, sizes = _make(xp, kind, rows)
    k = len(_reference(rows, lo, hi, sizes)[0])
    inner = _NonlinearPlusLinear(xp, matrix, array(xp, lo), array(xp, hi))

    lowered = lower_linear_inequalities(inner, array(xp, [0.0] * _N), xp)
    assert isinstance(lowered, LoweredLinearIneqProblem)

    x = array(xp, [0.5, 0.25, -1.0, 2.0, 0.0])
    g = lowered.ineq_constraints(x)
    assert int(g.shape[0]) == 1 + k
    assert_allclose(xp, g[:1], array(xp, [float(xp.sum(x * x)) - 4.0]), **_TOL)
    jac = lowered.ineq_jacobian(x)
    assert jac.shape == (1 + k, _N)
    # Nonlinear rows lead; a stacked lowered block is flattened into the stack
    # so each block keeps its own precision hint at the top level.
    assert isinstance(jac, VStack)
    assert len(jac.blocks) == 1 + n_blocks
    y = array(xp, [3.0] + [9.0] * k)
    lowered.lagrangian_hessian(x, array(xp, [0.0])[:0], y, 1.0)
    assert int(inner.seen_y_ineq.shape[0]) == 1
    assert float(inner.seen_y_ineq[0]) == 3.0


def test_matrix_free_block_lowers_through_the_wrapper(namespace):
    # A callback-only operator: matvec/rmatvec work, while the explicit
    # capabilities it lacks stay unavailable (never a silent identity probe).
    xp = namespace
    rows, lo, hi = _random_block()
    a = array(xp, rows)
    matrix = MatrixFreeJacobian(
        (_M, _N),
        matvec=lambda v: xp.matmul(a, v),
        rmatvec=lambda v: xp.matmul(transpose(xp, a), v),
    )
    jac_ref, _ = _reference(rows, lo, hi, [_M])
    J = array(xp, jac_ref)

    op, _ = _lower(xp, matrix, lo, hi)

    x = array(xp, [0.3, -1.2, 2.0, 0.7, -0.4])
    v = _weights(xp, len(jac_ref))
    assert_allclose(xp, op.matvec(x), xp.matmul(J, x), **_TOL)
    assert_allclose(xp, op.rmatvec(v), xp.matmul(transpose(xp, J), v), **_TOL)
    assert op.gram_capable() is False
    assert op.gram_coo_capable() is False
    assert op.gram_fill_estimate() is None
    assert op.gram_accumulate_dtype_hint() is None
    assert op.coo_pattern_signature() is None
    with pytest.raises(NotImplementedError):
        op.dense_matrix()
    with pytest.raises(NotImplementedError):
        op.to_coo()
    with pytest.raises(NotImplementedError):
        op.gram(v)


def test_wrapper_to_coo_remaps_a_structured_inner_operator(namespace):
    # A subclass of a sparse operator is wrapped (its overrides must survive),
    # but its triplets are still exposed — remapped — for the sparse route.
    xp = namespace
    rows, lo, hi = _random_block()

    class _Custom(COOOperator):
        pass

    r, c, v = _triplets(xp, rows)
    matrix = _Custom(r, c, v, (_M, _N), pattern_key="custom")
    jac_ref, _ = _reference(rows, lo, hi, [_M])

    op, _ = _lower(xp, matrix, lo, hi)

    assert not isinstance(op, COOOperator)
    rr, cc, vv, shape = op.to_coo()
    dense = _coo_dense(xp, rr, cc, vv, shape)
    assert_allclose(xp, dense, array(xp, jac_ref), atol=0, rtol=0)
    assert_allclose(xp, op.coo_values(), vv, atol=0, rtol=0)
    signature = op.coo_pattern_signature()
    assert signature is not None
    assert signature == op.coo_pattern_signature()


def test_dense_operator_lowering_is_bitwise_the_array_lowering(namespace):
    xp = namespace
    rows, lo, hi = _random_block()
    a = array(xp, rows)

    from_array, off_a = _lower(xp, a, lo, hi)
    from_dense, off_d = _lower(xp, Dense(a), lo, hi)

    assert isinstance(from_dense, Dense)
    assert bool(xp.all(from_array.dense_matrix() == from_dense.dense_matrix()))
    assert bool(xp.all(off_a == off_d))


def test_csr_linear_ineq_no_longer_raises(namespace):
    # Regression: an operator-valued ``linear_ineq`` matrix raised
    # NotImplementedError ("require an explicit rank-2 array"), forcing large
    # sparse inequality sets through hand-lowered ineq_constraints callbacks.
    xp = namespace
    rows, lo, hi = _random_block()
    problem = _with_block(xp, rows, _csr(xp, rows), lo, hi)

    lowered = lower_linear_inequalities(problem, array(xp, [0.0] * _N), xp)

    assert isinstance(lowered, LoweredLinearIneqProblem)
    assert lowered.linear_ineq() is None


def test_dense_subclass_is_wrapped_so_its_overrides_survive(namespace):
    # Only the exact ``Dense`` type is unwrapped to its array; a subclass may
    # override the algebra (here a Gram counter) and must keep being consulted.
    xp = namespace
    rows, lo, hi = _random_block()
    calls = []

    class _CountingDense(Dense):
        def gram(self, weights, *, accumulate_dtype=None, hinted_only=False):
            calls.append(1)
            a = self.dense_matrix()
            return xp.matmul(transpose(xp, a), xp.expand_dims(weights, axis=1) * a)

        def gram_capable(self):
            return True

    matrix = _CountingDense(array(xp, rows))
    jac_ref, _ = _reference(rows, lo, hi, [_M])
    J = array(xp, jac_ref)

    op, _ = _lower(xp, matrix, lo, hi)

    assert not isinstance(op, Dense)
    w = _weights(xp, len(jac_ref))
    assert op.gram_capable()
    gram = op.gram(w)
    assert calls
    assert_allclose(
        xp, gram, xp.matmul(transpose(xp, J), xp.expand_dims(w, 1) * J), **_TOL
    )


def test_wrapper_coo_values_reuse_the_cached_selection(namespace, monkeypatch):
    # The sparse route refreshes values every iteration; with a structural
    # inner pattern the wrapper must not re-derive the selection (cumulative
    # sums + data-dependent nonzero syncs) on each refresh.
    import ipax.backend.operators as ops

    xp = namespace
    rows, lo, hi = _random_block()

    class _Custom(COOOperator):
        pass

    r, c, v = _triplets(xp, rows)
    op, _ = _lower(xp, _Custom(r, c, v, (_M, _N), pattern_key="custom"), lo, hi)
    first = op.to_coo()

    plans = []
    original = ops._signed_row_select_plan

    def _spy(*args, **kwargs):
        plans.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ops, "_signed_row_select_plan", _spy)
    again = op.to_coo()
    values = op.coo_values()
    values_again = op.coo_values()

    assert plans == []
    for a, b in zip(first[:3], again[:3], strict=True):
        assert bool(xp.all(a == b))
    assert bool(xp.all(values == first[2]))
    assert bool(xp.all(values_again == first[2]))


def test_wrapper_without_a_pattern_key_rederives_its_coo(namespace):
    # No structural promise from the inner operator ⇒ no cached plan: the
    # values are re-derived with the pattern each time (always correct).
    xp = namespace
    rows, lo, hi = _random_block()

    class _Custom(COOOperator):
        pass

    r, c, v = _triplets(xp, rows)
    op, _ = _lower(xp, _Custom(r, c, v, (_M, _N)), lo, hi)
    jac_ref, _ = _reference(rows, lo, hi, [_M])

    rr, cc, vv, shape = op.to_coo()
    assert op._coo_plan is None
    assert_allclose(xp, _coo_dense(xp, rr, cc, vv, shape), array(xp, jac_ref), **_TOL)
    assert bool(xp.all(op.coo_values() == vv))


def test_vstack_multipliers_are_ordered_block_by_block(namespace):
    # Documented order: a VStack lowers block by block (each block's lower rows,
    # then its upper rows); a single operator lowers all lower rows, then all
    # upper rows. Same set, different layout of y_ineq.
    xp = namespace
    rows, lo, hi = _random_block()
    stacked, sizes = _make(xp, "opaque", rows)
    top = _Opaque(array(xp, rows[:_SPLIT]))
    bottom = _Opaque(array(xp, rows[_SPLIT:]))

    single, _ = _lower(xp, stacked, lo, hi)
    split, _ = _lower(xp, VStack((top, bottom)), lo, hi)

    x = array(xp, [0.3, -1.2, 2.0, 0.7, -0.4])
    j_single = _reference(rows, lo, hi, sizes)[0]
    j_split = _reference(rows, lo, hi, [_SPLIT, _M - _SPLIT])[0]
    assert_allclose(xp, single.matvec(x), xp.matmul(array(xp, j_single), x), **_TOL)
    assert_allclose(xp, split.matvec(x), xp.matmul(array(xp, j_split), x), **_TOL)
    assert sorted(map(tuple, j_single)) == sorted(map(tuple, j_split))


def test_vstack_subclass_is_wrapped_so_its_overrides_survive(namespace):
    # Regression: an ``isinstance`` dispatch decomposed a ``VStack`` subclass
    # into its blocks, silently dropping overridden algebra (the residuals of a
    # reproduction moved from [0, 2] to [-1, -1]). Only the exact ``VStack`` is
    # lowered block by block; a subclass takes the signed-row-selection path.
    xp = namespace
    rows, lo, hi = _random_block()
    calls = []

    class _DoubledVStack(VStack):
        def matvec(self, v):
            return 2.0 * super().matvec(v)

        def rmatvec(self, v):
            return 2.0 * super().rmatvec(v)

        def gram(self, weights, *, accumulate_dtype=None, hinted_only=False):
            calls.append(1)
            return 4.0 * super().gram(
                weights, accumulate_dtype=accumulate_dtype, hinted_only=hinted_only
            )

    top = _Opaque(array(xp, rows[:_SPLIT]))
    bottom = _Opaque(array(xp, rows[_SPLIT:]))
    matrix = _DoubledVStack((top, bottom))
    jac_ref, _ = _reference(rows, lo, hi, [_M])  # one operator: no per-block split
    J = 2.0 * array(xp, jac_ref)

    op, _ = _lower(xp, matrix, lo, hi)

    assert not isinstance(op, VStack)
    x = array(xp, [0.3, -1.2, 2.0, 0.7, -0.4])
    assert_allclose(xp, op.matvec(x), xp.matmul(J, x), **_TOL)
    w = _weights(xp, len(jac_ref))
    assert op.gram_capable()
    gram = op.gram(w)
    assert calls
    assert_allclose(
        xp, gram, xp.matmul(transpose(xp, J), xp.expand_dims(w, 1) * J), **_TOL
    )


def test_compressed_lowering_does_not_read_the_array_device_attribute():
    # Regression: NumPy < 2.0 (still supported: ``numpy>=1.26``) has no
    # ``ndarray.device``; CSR/CSC construction and lowering read it directly and
    # raised AttributeError. The compat ``device()`` helper must be used instead.
    np = pytest.importorskip("numpy")
    xp = pytest.importorskip("array_api_compat.numpy")

    class _NoDevice(np.ndarray):
        """An ndarray that behaves like NumPy 1.26: no ``.device`` attribute."""

        @property
        def device(self):
            raise AttributeError("'numpy.ndarray' object has no attribute 'device'")

    def hide(a):
        return np.asarray(a).view(_NoDevice)

    rows, lo, hi = _random_block()
    _, c, v = _triplets(xp, rows)
    indptr_r = [0]
    for row in rows:
        indptr_r.append(indptr_r[-1] + sum(1 for value in row if value != 0.0))
    csr = CSROperator(hide(indptr_r), hide(c), hide(v), (_M, _N))
    r_c, v_c, indptr_c = [], [], [0]
    for j in range(_N):
        for i, row in enumerate(rows):
            if row[j] != 0.0:
                r_c.append(i)
                v_c.append(row[j])
        indptr_c.append(len(r_c))
    csc = CSCOperator(hide(indptr_c), hide(r_c), hide(v_c), (_M, _N))

    jac_ref, off_ref = _reference(rows, lo, hi, [_M])
    x = xp.asarray([0.3, -1.2, 2.0, 0.7, -0.4])
    for matrix in (csr, csc):
        op, offset = _lower_two_sided(matrix, hide(lo), hide(hi), xp)
        assert_allclose(xp, op.matvec(x), xp.matmul(xp.asarray(jac_ref), x), **_TOL)
        assert_allclose(xp, offset, xp.asarray(off_ref), **_TOL)


# --------------------------------------------------------------------------- #
# Edge cases of the lowering and its signed-row-selection wrapper.
# --------------------------------------------------------------------------- #
def test_all_unbounded_array_block_lowers_to_an_empty_dense(namespace):
    xp = namespace
    rows, _, _ = _random_block()
    op, offset = _lower(xp, array(xp, rows), [-_INF] * _M, [_INF] * _M)
    assert isinstance(op, Dense) and op.shape == (0, _N)
    assert int(offset.shape[0]) == 0


def test_vstack_with_one_bounded_block_is_not_restacked(namespace):
    # Only the first block has finite bounds: the lowered result is that
    # block's own operator, not a one-element VStack.
    xp = namespace
    rows, lo, hi = _random_block()
    top = _Opaque(array(xp, rows[:_SPLIT]))
    bottom = _Opaque(array(xp, rows[_SPLIT:]))
    lo = lo[:_SPLIT] + [-_INF] * (_M - _SPLIT)
    hi = hi[:_SPLIT] + [_INF] * (_M - _SPLIT)

    op, offset = _lower(xp, VStack((top, bottom)), lo, hi)

    jac_ref, off_ref = _reference(rows[:_SPLIT], lo[:_SPLIT], hi[:_SPLIT], [_SPLIT])
    assert not isinstance(op, VStack)
    x = array(xp, [0.3, -1.2, 2.0, 0.7, -0.4])
    assert_allclose(xp, op.matvec(x), xp.matmul(array(xp, jac_ref), x), **_TOL)
    assert_allclose(xp, offset, array(xp, off_ref), **_TOL)


def test_row_selection_rejects_masks_of_the_wrong_length(namespace):
    from ipax.backend.operators import _SignedRowSelection

    xp = namespace
    a = Dense(array(xp, [[1.0, 2.0], [3.0, 4.0]]))
    ok = array(xp, [1.0, 0.0]) > 0.5
    short = array(xp, [1.0]) > 0.5
    with pytest.raises(ValueError, match="row-selection masks"):
        _SignedRowSelection(a, short, ok)
    with pytest.raises(ValueError, match="row-selection masks"):
        _SignedRowSelection(a, ok, short)


def test_row_selection_without_rows_adjoint_is_zero(namespace):
    # k = 0: Sᵀ v is an empty-source expansion, so the adjoint products are
    # zero for rank-1 and rank-2 inputs alike.
    from ipax.backend.operators import _SignedRowSelection

    xp = namespace
    a = Dense(array(xp, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    none = array(xp, [0.0, 0.0, 0.0]) > 0.5
    op = _SignedRowSelection(a, none, none)
    assert op.shape == (0, 2)
    empty = array(xp, [])
    assert_allclose(xp, op.rmatvec(empty), array(xp, [0.0, 0.0]), **_TOL)
    empty2 = xp.reshape(array(xp, []), (0, 3))
    assert_allclose(xp, op.rmatmat(empty2), array(xp, [[0.0] * 3, [0.0] * 3]), **_TOL)


def test_wrapper_forwards_gram_coo_with_folded_weights(namespace):
    xp = namespace
    rows, lo, hi = _random_block()

    class _OpaqueCoo(_Opaque):
        def gram_coo(self, weights):
            g = self.gram(weights)
            r, c = [], []
            for i in range(_N):
                for j in range(_N):
                    r.append(i)
                    c.append(j)
            flat = xp.reshape(g, (-1,))
            return (
                xp.asarray(r, dtype=xp.int64),
                xp.asarray(c, dtype=xp.int64),
                flat,
                (_N, _N),
            )

    op, _ = _lower(xp, _OpaqueCoo(array(xp, rows)), lo, hi)
    jac_ref, _ = _reference(rows, lo, hi, [_M])
    J = array(xp, jac_ref)
    w = _weights(xp, len(jac_ref))
    assert op.gram_coo_capable()
    r, c, g, shape = op.gram_coo(w)
    assert_allclose(
        xp,
        _coo_dense(xp, r, c, g, shape),
        xp.matmul(transpose(xp, J), xp.expand_dims(w, 1) * J),
        **_TOL,
    )


def test_lowered_problem_delegates_equality_constraints(namespace):
    xp = namespace

    class _WithEquality(_LinearIneqOnly):
        def eq_constraints(self, x):
            return xp.sum(x, keepdims=True) - 1.0

        def eq_jacobian(self, x):
            return xp.ones((1, int(x.shape[0])), dtype=x.dtype)

    problem = _WithEquality(
        xp,
        array(xp, [[1.0, 0.0], [0.0, 1.0]]),
        array(xp, [0.0, 0.0]),
        array(xp, [1.0, 1.0]),
    )
    lowered = lower_linear_inequalities(problem, array(xp, [0.0, 0.0]), xp)
    x = array(xp, [0.25, 0.5])
    assert_allclose(xp, lowered.eq_constraints(x), array(xp, [-0.25]), **_TOL)
    assert_allclose(xp, lowered.eq_jacobian(x), array(xp, [[1.0, 1.0]]), **_TOL)


def test_array_block_must_be_rank_two(namespace):
    xp = namespace
    with pytest.raises(ValueError, match="rank-2"):
        _lower(xp, array(xp, [1.0, 2.0]), [0.0, 0.0], [1.0, 1.0])
