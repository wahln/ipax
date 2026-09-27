"""The TROTS loader's tiled per-structure Gram (``TiledGramCSROperator``).

An application-edge operator (host SciPy/BLAS), so the operator-level tests are
NumPy-only; the core-level parity test runs it through ``ipax.solve``. The
reference everywhere is the dense ``Aᵀ diag(w) A`` or the plain
``CSROperator`` path the operator replaces (Breedveld et al. 2017, eq. 48).
"""

from __future__ import annotations

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")
trots = pytest.importorskip("benchmarks.corpus.trots")

from ipax.backend.operators import CSROperator  # noqa: E402


def _random_sparse(m, n, density, seed):
    rng = np.random.default_rng(seed)
    a = sp.random(m, n, density=density, format="csr", random_state=rng)
    a.data = rng.uniform(0.5, 2.0, a.nnz)
    return a


def _banded(m, n, width, seed):
    # Voxel-like locality: row i touches a window of columns around i·n/m.
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i in rng.permutation(m):  # scrambled order: the RCM plan must undo it
        centre = int(i * n / m)
        for j in range(max(0, centre - width), min(n, centre + width + 1)):
            if rng.random() < 0.7:
                rows.append(i)
                cols.append(j)
    vals = rng.uniform(0.1, 3.0, len(rows))
    return sp.csr_matrix((vals, (rows, cols)), shape=(m, n))


def _dense_gram(a, w):
    d = np.asarray(a.todense())
    return d.T @ (w[:, None] * d)


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _op(a, **cfg):
    cfg.setdefault("flop_ratio_threshold", 1.0)  # tile regardless of the gate
    return trots.TiledGramCSROperator(a, np, config=trots.TiledGramConfig(**cfg))


PATTERNS = {
    "random": lambda: _random_sparse(700, 40, 0.08, 0),
    "banded": lambda: _banded(900, 60, 4, 1),
}


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
@pytest.mark.parametrize("block_rows", [64, 128, 5000])
@pytest.mark.parametrize("threads", [1, 3])
def test_tiled_gram_matches_dense_reference(pattern, block_rows, threads):
    # Small problems would run on one lane by the work heuristic; lower it so
    # the per-lane accumulators are exercised.
    a = PATTERNS[pattern]()
    w = np.random.default_rng(2).uniform(0.0, 5.0, a.shape[0])
    op = _op(a, block_rows=block_rows, threads=threads, lane_min_flops=1.0)

    g = np.asarray(op.gram(w))

    assert op.tiled
    assert g.dtype == np.float64
    assert _rel(g, _dense_gram(a, w)) < 1e-12
    np.testing.assert_array_equal(g, g.T)  # exact symmetry (mirrored once)


def test_rcm_plan_compacts_a_scrambled_banded_matrix():
    # The permutation is what makes tiles narrow: a scrambled banded matrix
    # touches every column per tile in row order, a handful after RCM.
    a = _banded(2000, 400, 3, 3)
    op = _op(a, block_rows=128)
    widths = [t.cols.shape[0] for p in op._plans for t in p.tiles]
    assert max(widths) < 0.25 * a.shape[1]
    assert op.flop_ratio < 0.1


def test_zero_weights_and_empty_rows():
    a = _random_sparse(500, 30, 0.1, 4).tolil()
    a[10:60, :] = 0.0  # an all-zero stretch (possibly whole empty tiles)
    a = a.tocsr()
    w = np.random.default_rng(5).uniform(0.0, 2.0, a.shape[0])
    w[::3] = 0.0

    g = np.asarray(_op(a, block_rows=32).gram(w))

    assert _rel(g, _dense_gram(a, w)) < 1e-12


@pytest.mark.parametrize("sparse_ratio", [0.0, 1.0])
def test_negative_weights_take_the_signed_fallback(sparse_ratio):
    # Signed weights (exact-Hessian nonconvex case) cannot take √w: dense tiles
    # fall back to Dᵀ(w∘D), sparse tiles are sign-agnostic. Result still exact.
    a = _random_sparse(600, 25, 0.1, 6)
    w = np.random.default_rng(7).uniform(-1.0, 3.0, a.shape[0])
    op = _op(a, block_rows=100, sparse_tile_flop_ratio=sparse_ratio)

    g = np.asarray(op.gram(w))

    assert _rel(g, _dense_gram(a, w)) < 1e-12
    np.testing.assert_array_equal(g, g.T)


def test_sparse_tile_kernel_matches_dense_reference():
    a = _banded(800, 50, 2, 8)
    w = np.random.default_rng(9).uniform(0.1, 4.0, a.shape[0])
    op = _op(a, block_rows=100, sparse_tile_flop_ratio=1.0)  # every tile sparse

    assert all(t.sparse for p in op._plans for t in p.tiles)
    assert _rel(np.asarray(op.gram(w)), _dense_gram(a, w)) < 1e-12


@pytest.mark.parametrize("threads", [1, 3])
def test_float32_accumulate_agrees_and_returns_float64(threads):
    a = _random_sparse(800, 40, 0.1, 10)
    a.data = a.data.astype(np.float32).astype(np.float64)  # exact float32 data
    w = np.random.default_rng(11).uniform(0.1, 5.0, a.shape[0])
    op = trots.TiledGramCSROperator(
        a,
        np,
        values_dtype_hint="float32",
        config=trots.TiledGramConfig(
            block_rows=128,
            flop_ratio_threshold=1.0,
            threads=threads,
            lane_min_flops=1.0,
        ),
    )

    reduced = np.asarray(op.gram(w, accumulate_dtype="float32", hinted_only=True))
    exact = np.asarray(op.gram(w))

    assert reduced.dtype == np.float64
    assert _rel(reduced, _dense_gram(a, w)) < 1e-6
    assert _rel(exact, _dense_gram(a, w)) < 1e-12
    assert not np.array_equal(reduced, exact)  # the reduction really engaged
    np.testing.assert_array_equal(reduced, reduced.T)


def test_float32_hint_on_inexact_values_keeps_exact_requests_exact():
    # The float32 tile storage also serves exact (float64) requests, so it is
    # only used when the values are exact float32 widenings (checked once per
    # structure); 1/3-multiples are not, and must stay exact to 1e-12.
    a = _random_sparse(800, 40, 0.1, 50)
    a.data = np.random.default_rng(51).integers(1, 9, a.nnz) / 3.0
    assert not np.array_equal(a.data.astype(np.float32).astype(np.float64), a.data)
    w = np.random.default_rng(52).uniform(0.1, 5.0, a.shape[0])
    cfg = trots.TiledGramConfig(block_rows=128, flop_ratio_threshold=1.0)
    op = trots.TiledGramCSROperator(a, np, values_dtype_hint="float32", config=cfg)
    assert op.tiled
    assert all(t.csr.dtype == np.float64 for p in op._plans for t in p.tiles)
    ref = _dense_gram(a, w)
    assert _rel(np.asarray(op.gram(w)), ref) < 1e-12
    # The hint still drives the reduced-accumulate path (rounded at product time).
    reduced = np.asarray(op.gram(w, accumulate_dtype="float32", hinted_only=True))
    assert _rel(reduced, ref) < 1e-6
    # Positive control: float32-exact values are stored narrow.
    b = a.copy()
    b.data = b.data.astype(np.float32).astype(np.float64)
    op32 = trots.TiledGramCSROperator(b, np, values_dtype_hint="float32", config=cfg)
    assert all(t.csr.dtype == np.float32 for p in op32._plans for t in p.tiles)
    assert _rel(np.asarray(op32.gram(w)), _dense_gram(b, w)) < 1e-12


def test_hinted_only_keeps_unhinted_data_exact():
    a = _random_sparse(400, 30, 0.1, 12)
    w = np.random.default_rng(13).uniform(0.1, 5.0, a.shape[0])
    op = _op(a, block_rows=64)  # no float32 hint

    g = np.asarray(op.gram(w, accumulate_dtype="float32", hinted_only=True))

    np.testing.assert_array_equal(g, np.asarray(op.gram(w)))


def test_gate_above_threshold_keeps_the_plain_path():
    # A dense-ish matrix barely compacts: the gate keeps the plain adapter Gram
    # (bitwise the plain CSROperator's) and drops the tile copies.
    a = _random_sparse(600, 30, 0.6, 14)
    w = np.random.default_rng(15).uniform(0.1, 5.0, a.shape[0])
    op = trots.TiledGramCSROperator(
        a, np, config=trots.TiledGramConfig(block_rows=64, flop_ratio_threshold=0.5)
    )
    plain = CSROperator(
        np.asarray(a.indptr, dtype=np.int64),
        np.asarray(a.indices, dtype=np.int64),
        a.data,
        a.shape,
    )

    assert op.flop_ratio > 0.5
    assert not op.tiled
    assert op._plans == []
    np.testing.assert_array_equal(np.asarray(op.gram(w)), np.asarray(plain.gram(w)))


def test_dense_plans_are_rejected_before_any_permutation_work(monkeypatch):
    # Σ nnz_row² bounds the tiled arithmetic from below: a dense matrix fails
    # the gate on that bound alone, so no RCM/tile plan is ever built.
    def boom(*_args, **_kwargs):
        raise AssertionError("planned a matrix the pre-gate should reject")

    monkeypatch.setattr(trots, "_plan_structure", boom)
    a = _random_sparse(600, 30, 0.9, 30)
    op = trots.TiledGramCSROperator(a, np)

    assert not op.tiled
    assert op.flop_ratio > op.config.flop_ratio_threshold


def test_small_work_runs_on_one_lane(monkeypatch):
    # A tiny tiled Gram is not worth a thread pool (its per-lane n² accumulators
    # and dispatch cost more than the BLAS work).
    import concurrent.futures

    class NoPool:
        def __init__(self, *args, **kwargs):
            raise AssertionError("thread pool used for trivial work")

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", NoPool)
    a = _random_sparse(300, 20, 0.1, 31)
    w = np.random.default_rng(32).uniform(0.1, 2.0, a.shape[0])
    op = _op(a, block_rows=32, threads=8)

    assert _rel(np.asarray(op.gram(w)), _dense_gram(a, w)) < 1e-12


def test_operator_presents_the_original_row_order():
    # The RCM permutation is private to the Gram plan: matvec/rmatvec/COO are
    # the plain CSR operator's, row for row.
    a = _banded(700, 45, 3, 16)
    op = _op(a, block_rows=64)
    rng = np.random.default_rng(17)
    x, y = rng.standard_normal(a.shape[1]), rng.standard_normal(a.shape[0])

    np.testing.assert_allclose(np.asarray(op.matvec(x)), a @ x, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(op.rmatvec(y)), a.T @ y, rtol=1e-13)
    rows, cols, vals, shape = op.to_coo()
    rebuilt = sp.coo_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=shape
    )
    assert (rebuilt.tocsr() != a).nnz == 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_non_finite_weights_agree_with_the_plain_verdict():
    # Non-finite merged weights take the gemm fallback; the NaN footprint may
    # differ from the plain path (compaction confines 0·NaN to a tile's active
    # columns) but the finiteness verdict the solver acts on must not.
    a = _random_sparse(400, 20, 0.1, 34)
    plain = CSROperator(
        np.asarray(a.indptr, dtype=np.int64),
        np.asarray(a.indices, dtype=np.int64),
        a.data,
        a.shape,
    )
    op = _op(a, block_rows=64)
    for bad in (np.inf, np.nan):
        w = np.random.default_rng(35).uniform(0.1, 2.0, a.shape[0])
        w[7] = bad
        g, g_plain = np.asarray(op.gram(w)), np.asarray(plain.gram(w))
        assert bool(np.isfinite(g).all()) == bool(np.isfinite(g_plain).all())


def test_memo_keeps_one_slot_per_effective_precision():
    a = _random_sparse(300, 20, 0.1, 36)
    a.data = a.data.astype(np.float32).astype(np.float64)
    op = trots.TiledGramCSROperator(
        a,
        np,
        values_dtype_hint="float32",
        config=trots.TiledGramConfig(block_rows=64, flop_ratio_threshold=1.0),
    )
    w = np.random.default_rng(37).uniform(0.1, 5.0, a.shape[0])

    op.gram(w)
    op.gram(w, accumulate_dtype="float64")  # exact: same slot, a memo hit
    op.gram(w, accumulate_dtype="float16")  # no syrk: exact fallback, same slot
    assert op.gram_compute_count == 1
    op.gram(w, accumulate_dtype="float32")  # reduced: its own slot
    op.gram(w, accumulate_dtype="float32")
    assert op.gram_compute_count == 2
    assert sorted(op._memo, key=str) == [None, "float32"]


def test_repeated_weights_are_memoized():
    a = _random_sparse(300, 20, 0.1, 18)
    op = _op(a, block_rows=64)
    w = np.random.default_rng(19).uniform(0.1, 5.0, a.shape[0])

    first = np.asarray(op.gram(w))
    again = np.asarray(op.gram(w.copy()))
    other = np.asarray(op.gram(w * 2.0))

    assert op.gram_compute_count == 2
    np.testing.assert_array_equal(first, again)
    np.testing.assert_allclose(other, 2.0 * first, rtol=1e-13)


def test_shared_structure_segments_merge_weights_and_add_aux_borders():
    # Rows [A; A; -A | +1 aux]: a min/max constraint pair on one structure plus
    # a maximise-minimax link on it. The plan multiplies A once with the merged
    # weights (eq. 48) and adds the aux column's borders; the result must equal
    # the Gram of the explicitly stacked matrix.
    a = _banded(500, 30, 3, 20)
    m, nx = a.shape
    n = nx + 2  # two aux columns; the second stays untouched
    aux = sp.csr_matrix(
        (np.ones(m), (np.arange(m), np.zeros(m, dtype=int))), shape=(m, 2)
    )
    zero = sp.csr_matrix((m, 2))
    stacked = sp.vstack(
        [sp.hstack([a, zero]), sp.hstack([a, zero]), sp.hstack([-a, aux])], format="csr"
    )
    segments = [
        trots.GramSegment(0, m, a, key="A"),
        trots.GramSegment(m, 2 * m, a, key="A"),
        trots.GramSegment(
            2 * m, 3 * m, a, key="A", sign=-1.0, aux_col=nx, aux_coef=1.0
        ),
    ]
    op = trots.TiledGramCSROperator(
        stacked,
        np,
        segments=segments,
        config=trots.TiledGramConfig(block_rows=64, flop_ratio_threshold=1.0),
    )
    w = np.random.default_rng(21).uniform(0.0, 3.0, 3 * m)

    g = np.asarray(op.gram(w))

    assert len(op._plans) == 1  # one structure, one plan
    assert g.shape == (n, n)
    assert _rel(g, _dense_gram(stacked, w)) < 1e-12
    np.testing.assert_array_equal(g, g.T)


def test_segments_must_match_their_structure():
    a = _random_sparse(100, 10, 0.2, 22)
    with pytest.raises(ValueError, match="segment rows"):
        trots.TiledGramCSROperator(
            a, np, segments=[trots.GramSegment(0, 50, a, key="A")]
        )


def _two_halves():
    a = _random_sparse(200, 10, 0.2, 38)
    top, bottom = a[:100], a[100:]
    return a, top, bottom


@pytest.mark.parametrize(
    ("segments", "match"),
    [
        # a gap: rows 100..199 described by nothing
        (lambda a, t, b: [trots.GramSegment(0, 100, t, key="t")], "tile"),
        # an overlap
        (
            lambda a, t, b: [
                trots.GramSegment(0, 100, t, key="t"),
                trots.GramSegment(50, 150, b, key="b"),
                trots.GramSegment(100, 200, b, key="b"),
            ],
            "tile",
        ),
        # out of range
        (lambda a, t, b: [trots.GramSegment(150, 250, t, key="t")], "outside"),
        # one key, two different matrices: would multiply the wrong one
        (
            lambda a, t, b: [
                trots.GramSegment(0, 100, t, key="same"),
                trots.GramSegment(100, 200, b, key="same"),
            ],
            "share the structure",
        ),
        # a scaled copy: merged weights assume |sign| = 1 (eq. 48)
        (
            lambda a, t, b: [
                trots.GramSegment(0, 100, t, key="t", sign=2.0),
                trots.GramSegment(100, 200, b, key="b"),
            ],
            "sign must be",
        ),
        # aux column overlapping the structure's own columns
        (
            lambda a, t, b: [
                trots.GramSegment(0, 100, t, key="t", aux_col=3, aux_coef=1.0),
                trots.GramSegment(100, 200, b, key="b"),
            ],
            "aux column",
        ),
    ],
)
def test_malformed_segments_are_rejected(segments, match):
    a, top, bottom = _two_halves()
    with pytest.raises(ValueError, match=match):
        trots.TiledGramCSROperator(a, np, segments=segments(a, top, bottom))


def test_segments_must_describe_the_operator_entries():
    # Row ranges that tile the operator but describe another matrix are
    # caught by the stored-entry count.
    a, top, _bottom = _two_halves()
    other = _random_sparse(100, 10, 0.5, 39)
    with pytest.raises(ValueError, match="stored entries"):
        trots.TiledGramCSROperator(
            a,
            np,
            segments=[
                trots.GramSegment(0, 100, top, key="t"),
                trots.GramSegment(100, 200, other, key="o"),
            ],
        )


def test_planning_never_mutates_the_callers_matrix():
    # The loader hands the instance's cached structure matrices to the plan.
    a = _random_sparse(300, 20, 0.2, 40)
    rng = np.random.default_rng(41)
    for i in range(a.shape[0]):  # scramble column order inside every row
        lo, hi = a.indptr[i], a.indptr[i + 1]
        order = rng.permutation(hi - lo)
        a.indices[lo:hi] = a.indices[lo:hi][order]
        a.data[lo:hi] = a.data[lo:hi][order]
    a.has_sorted_indices = False
    before = (a.indptr.copy(), a.indices.copy(), a.data.copy())

    trots.TiledGramPlan.build(
        a.shape,
        [trots.GramSegment(0, a.shape[0], a, key="A")],
        config=trots.TiledGramConfig(block_rows=64, flop_ratio_threshold=1.0),
    )

    for kept, now in zip(before, (a.indptr, a.indices, a.data), strict=True):
        np.testing.assert_array_equal(kept, now)


# --------------------------------------------------------------------------- #
# Through the core: linear_ineq lowering + solve
# --------------------------------------------------------------------------- #
class _LinIneqQP:
    """min ½‖x − c‖² s.t. lo ≤ A x ≤ hi, declared through ``linear_ineq``."""

    def __init__(self, a, lo, hi, c, make_op):
        from ipax.problem.base import Problem

        class _P(Problem):
            @property
            def n_vars(self_inner):
                return a.shape[1]

            def objective(self_inner, x):
                return 0.5 * np.sum((x - c) ** 2)

            def gradient(self_inner, x):
                return x - c

            def lagrangian_hessian(self_inner, x, y_eq, y_ineq, sigma=1.0):
                return sigma * np.eye(a.shape[1])

            def linear_ineq(self_inner):
                return make_op(), lo, hi

        self.problem = _P()


def _qp_data():
    a = _banded(3000, 60, 3, 23)
    rng = np.random.default_rng(24)
    x_feas = rng.uniform(0.0, 1.0, a.shape[1])
    ax = a @ x_feas
    lo = np.where(rng.random(a.shape[0]) < 0.5, ax - 0.5, -np.inf)
    hi = np.where(
        np.isinf(lo), ax + 0.2, np.where(rng.random(a.shape[0]) < 0.3, ax + 1.0, np.inf)
    )
    c = x_feas + rng.standard_normal(a.shape[1])
    return a, lo, hi, c


def test_tiled_operator_solves_like_the_plain_csr_operator():
    from ipax import Options, solve

    a, lo, hi, c = _qp_data()

    def plain():
        return CSROperator(
            np.asarray(a.indptr, dtype=np.int64),
            np.asarray(a.indices, dtype=np.int64),
            a.data,
            a.shape,
        )

    def tiled():
        op = _op(a, block_rows=256)
        assert op.tiled
        return op

    opts = Options(linsolve="dense", hessian="exact", max_iter=200)
    x0 = np.full(a.shape[1], 0.5)
    r_plain = solve(_LinIneqQP(a, lo, hi, c, plain).problem, x0, options=opts)
    r_tiled = solve(_LinIneqQP(a, lo, hi, c, tiled).problem, x0, options=opts)

    assert r_tiled.status == r_plain.status
    assert r_tiled.n_iter == r_plain.n_iter
    assert r_tiled.routes == r_plain.routes
    assert abs(r_tiled.objective - r_plain.objective) <= 1e-10 * max(
        1.0, abs(r_plain.objective)
    )
    np.testing.assert_allclose(np.asarray(r_tiled.x), np.asarray(r_plain.x), atol=1e-8)


# --------------------------------------------------------------------------- #
# Loader integration (synthetic instance, no HDF5)
# --------------------------------------------------------------------------- #
def _structured_instance():
    """Two structures: one float64 used by a min/max constraint pair AND a
    maximise-minimax objective, one float32 used by an upper constraint."""
    big = _banded(1500, 40, 3, 25)
    small = _random_sparse(600, 40, 0.05, 26)
    small.data = small.data.astype(np.float32).astype(np.float64)
    cache = {
        1: trots.TROTSMatrix("big", 0, big, np.zeros(0), 0.0, "float64"),
        2: trots.TROTSMatrix("small", 0, small, np.zeros(0), 0.0, "float32"),
    }

    def entry(name, is_con, minimise, data_id, bound):
        return trots.TROTSEntry(
            name=name,
            is_constraint=is_con,
            minimise=minimise,
            ctype=1,
            data_id=data_id,
            params=None,
            bound=bound,
            weight=1.0,
            priority=1,
            active=True,
        )

    entries = [
        entry("big max", True, True, 1, 50.0),
        entry("small max", True, True, 2, 30.0),
        entry("big min", True, False, 1, 1.0),
        entry("big minimax", False, False, 1, 0.0),
    ]
    inst = trots.TROTSInstance(
        identifier="synthetic", n=40, real=40, entries=entries, solution=None
    )
    inst._matrix_cache = cache
    return inst


def _split_problem(xp, inst=None, **kwargs):
    """The structured instance with its float32 group split off (share gate
    off): its 5.02% float32 share sits right at the default 5% gate."""
    p = trots.TROTSProblem(
        inst if inst is not None else _structured_instance(),
        xp,
        float32_group_min_share=0.0,
        **kwargs,
    )
    assert [h for _c, h in p._lin_groups] == ["float32", None]
    assert not p.float32_group_folded
    return p


def _lowered_jac(problem):
    from ipax.problem.linear_ineq import lower_linear_inequalities

    return lower_linear_inequalities(problem, np.zeros(problem.n_vars), np)._jac


def test_loader_tiled_block_matches_the_plain_block():
    inst = _structured_instance()
    cfg = trots.TiledGramConfig(block_rows=128)
    tiled = _split_problem(np, inst, tiled_gram=cfg)
    plain = _split_problem(np, inst, tiled_gram=False)

    op_t, lo_t, hi_t = tiled.linear_ineq()
    op_p, lo_p, hi_p = plain.linear_ineq()
    kinds = [type(b) for b in op_t.blocks]
    assert trots.TiledGramCSROperator in kinds
    assert all(type(b) is CSROperator for b in op_p.blocks)
    # Same rows, same order, same bounds — only the Gram kernel differs.
    np.testing.assert_array_equal(np.asarray(lo_t), np.asarray(lo_p))
    np.testing.assert_array_equal(np.asarray(hi_t), np.asarray(hi_p))
    np.testing.assert_array_equal(
        np.asarray(op_t.dense_matrix()), np.asarray(op_p.dense_matrix())
    )
    for bt, bp in zip(op_t.blocks, op_p.blocks, strict=True):
        assert bt.gram_accumulate_dtype_hint() == bp.gram_accumulate_dtype_hint()

    jt, jp = _lowered_jac(tiled), _lowered_jac(plain)
    assert jt.shape == jp.shape
    w = np.random.default_rng(27).uniform(0.1, 5.0, jt.shape[0])
    assert _rel(np.asarray(jt.gram(w)), np.asarray(jp.gram(w))) < 1e-12
    # The structure shared by three row sets is planned (and multiplied) once.
    f64 = next(
        b
        for b in op_t.blocks
        if isinstance(b, trots.TiledGramCSROperator)
        and b.gram_accumulate_dtype_hint() is None
    )
    assert len(f64._plans) == 1
    assert len(f64._plans[0].uses) == 3


def test_loader_gates_dense_blocks_back_to_plain_csr(monkeypatch):
    # A group whose plan fails the gate goes to the core as a plain
    # CSROperator (triplet lowering, exactly the untiled path) — and the
    # loader consults the plan first, so no tiled operator (with its own
    # Array-API copies of the block) is ever built for it.
    def boom(*_args, **_kwargs):
        raise AssertionError("built a tiled operator for a gated-off group")

    monkeypatch.setattr(trots.TiledGramCSROperator, "__init__", boom)
    problem = _split_problem(
        np, tiled_gram=trots.TiledGramConfig(flop_ratio_threshold=0.0)
    )
    op = problem.linear_ineq()[0]
    assert all(type(b) is CSROperator for b in op.blocks)


def test_loader_falls_back_to_plain_csr_on_a_segment_mismatch(caplog):
    # A bookkeeping mismatch between the segments and the assembled rows
    # must not abort the load: the plain operator is always correct.
    import dataclasses
    import logging

    inst = _structured_instance()
    problem = trots.TROTSProblem(
        inst, np, tiled_gram=trots.TiledGramConfig(block_rows=128)
    )
    g = next(
        i
        for i, segs in enumerate(problem._lin_segments)
        if any(s.structure.shape[0] == 1500 for s in segs)
    )
    segs = problem._lin_segments[g]
    other = _banded(1500, 40, 1, 99)  # same rows, different stored entries
    assert other.nnz != segs[0].structure.nnz
    segs[0] = dataclasses.replace(segs[0], structure=other, key="other")
    with caplog.at_level(logging.WARNING, logger=trots.__name__):
        op = problem.linear_ineq()[0]
    blocks = op.blocks if hasattr(op, "blocks") else (op,)
    assert type(blocks[g]) is CSROperator
    assert "tiled Gram disabled" in caplog.text


def test_plan_must_match_the_operator_shape():
    a = _random_sparse(100, 10, 0.2, 33)
    plan = trots.TiledGramPlan.build(
        (50, 10), [trots.GramSegment(0, 50, a[:50], key="A")]
    )
    with pytest.raises(ValueError, match="different shape"):
        trots.TiledGramCSROperator(a, np, plan=plan)


def test_structured_instance_share_gate_splits_at_the_threshold():
    # A share equal to the threshold splits; the next float above it folds
    # (the instance's float32 share, ~5.02%, sits right at the default gate).
    inst = _structured_instance()
    share = trots.TROTSProblem(inst, np).float32_nnz_share
    assert 0.0 < share < 1.0

    at = trots.TROTSProblem(inst, np, float32_group_min_share=share)
    assert [h for _c, h in at._lin_groups] == ["float32", None]
    assert not at.float32_group_folded

    above = trots.TROTSProblem(
        inst, np, float32_group_min_share=float(np.nextafter(share, 1.0))
    )
    assert [h for _c, h in above._lin_groups] == [None]
    assert above.float32_group_folded


def test_non_numpy_namespaces_get_the_plain_block():
    # The tiled kernel is host BLAS; any other namespace keeps the plain
    # CSROperator (lowered at the triplet level by the core).
    xp = pytest.importorskip("array_api_strict")
    problem = _split_problem(xp, tiled_gram=trots.TiledGramConfig(block_rows=128))
    op = problem.linear_ineq()[0]
    assert all(type(b) is CSROperator for b in op.blocks)


def test_loader_tiled_problem_solves_like_the_plain_one():
    from ipax import Options, solve

    inst = _structured_instance()
    cfg = trots.TiledGramConfig(block_rows=128)
    opts = Options(linsolve="dense", max_iter=150)
    results = []
    for tiled in (cfg, False):
        problem = trots.TROTSProblem(inst, np, tiled_gram=tiled)
        results.append(
            solve(problem, problem.initial_point(warm_start=False), options=opts)
        )
    r_t, r_p = results
    assert r_t.status == r_p.status
    assert r_t.n_iter == r_p.n_iter
    assert r_t.routes == r_p.routes
    assert abs(r_t.objective - r_p.objective) <= 1e-8 * max(1.0, abs(r_p.objective))


# --------------------------------------------------------------------------- #
# Device namespaces (CuPy / Torch): the same plan, run in the namespace
# --------------------------------------------------------------------------- #
def _host(x):
    return trots._to_numpy_nd(x)


def _device_namespace(namespace):
    """Skip unless ``namespace`` runs the device kernel (CuPy / Torch)."""
    kind = trots._device_kind(namespace)
    if kind is None or trots._is_host_numpy(namespace):
        pytest.skip("device tiled Gram runs on CuPy / Torch namespaces only")
    return kind


def _weights(kind, m, seed):
    rng = np.random.default_rng(seed)
    if kind == "positive":
        return rng.uniform(0.0, 5.0, m)
    if kind == "zeros":
        w = rng.uniform(0.0, 2.0, m)
        w[::3] = 0.0
        return w
    return rng.uniform(-1.0, 3.0, m)  # signed: the gemm fallback


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
@pytest.mark.parametrize("sparse_ratio", [0.0, 0.01, 1.0])
@pytest.mark.parametrize("weights", ["positive", "zeros", "signed"])
def test_device_tiled_gram_matches_the_numpy_kernel(
    namespace, pattern, sparse_ratio, weights
):
    xp = namespace
    _device_namespace(xp)
    a = PATTERNS[pattern]()
    cfg = trots.TiledGramConfig(
        block_rows=64, flop_ratio_threshold=1.0, sparse_tile_flop_ratio=sparse_ratio
    )
    host = trots.TiledGramCSROperator(a, np, config=cfg)
    dev = trots.TiledGramCSROperator(a, xp, config=cfg)
    w = _weights(weights, a.shape[0], 40)

    g = dev.gram(xp.asarray(w))

    assert dev.tiled and dev._device is not None
    assert type(g) is type(xp.asarray(0.0))  # the solve namespace's array
    g = _host(g)
    assert g.dtype == np.float64
    assert _rel(g, np.asarray(host.gram(w))) < 1e-12
    assert _rel(g, _dense_gram(a, w)) < 1e-12
    np.testing.assert_array_equal(g, g.T)


@pytest.mark.parametrize("device_sparse_tiles", [True, False])
def test_device_sparse_tiles_match_either_kernel(namespace, device_sparse_tiles):
    # CuPy multiplies sparse tiles by cuSPARSE unless told to densify them;
    # Torch always densifies. Every tile is sparse here.
    xp = namespace
    kind = _device_namespace(xp)
    a = _banded(800, 50, 2, 8)
    cfg = trots.TiledGramConfig(
        block_rows=100,
        flop_ratio_threshold=1.0,
        sparse_tile_flop_ratio=1.0,
        device_sparse_tiles=device_sparse_tiles,
    )
    op = trots.TiledGramCSROperator(a, xp, config=cfg)
    kernels = {t.sparse for ds in op.device_structures() for t in ds.tiles}
    assert kernels == {kind == "cupy" and device_sparse_tiles}
    for weights in ("positive", "signed"):
        w = _weights(weights, a.shape[0], 9)
        assert _rel(_host(op.gram(xp.asarray(w))), _dense_gram(a, w)) < 1e-12
        # Reduced-precision requests densify sparse tiles on every namespace.
        reduced = op.gram(xp.asarray(w), accumulate_dtype="float32")
        assert _rel(_host(reduced), _dense_gram(a, w)) < 1e-6


def test_device_float32_accumulate_agrees_and_returns_float64(namespace):
    xp = namespace
    _device_namespace(xp)
    a = _random_sparse(800, 40, 0.1, 10)
    a.data = a.data.astype(np.float32).astype(np.float64)  # exact float32 data
    w = np.random.default_rng(11).uniform(0.1, 5.0, a.shape[0])
    cfg = trots.TiledGramConfig(block_rows=128, flop_ratio_threshold=1.0)
    dev = trots.TiledGramCSROperator(a, xp, values_dtype_hint="float32", config=cfg)
    host = trots.TiledGramCSROperator(a, np, values_dtype_hint="float32", config=cfg)
    assert all(ds.data.dtype == xp.float32 for ds in dev.device_structures())

    reduced = _host(
        dev.gram(xp.asarray(w), accumulate_dtype="float32", hinted_only=True)
    )
    exact = _host(dev.gram(xp.asarray(w)))

    assert reduced.dtype == np.float64
    ref = _dense_gram(a, w)
    assert _rel(reduced, ref) < 1e-6
    assert _rel(exact, ref) < 1e-12
    assert _rel(exact, np.asarray(host.gram(w))) < 1e-12
    assert not np.array_equal(reduced, exact)  # the reduction really engaged
    np.testing.assert_array_equal(reduced, reduced.T)
    # Unhinted data stays exact under hinted_only (as on the host).
    plain = trots.TiledGramCSROperator(a, xp, config=cfg)
    np.testing.assert_array_equal(
        _host(plain.gram(xp.asarray(w), accumulate_dtype="float32", hinted_only=True)),
        _host(plain.gram(xp.asarray(w))),
    )


def test_device_memo_keeps_one_slot_per_effective_precision(namespace):
    xp = namespace
    _device_namespace(xp)
    a = _random_sparse(300, 20, 0.1, 36)
    a.data = a.data.astype(np.float32).astype(np.float64)
    op = trots.TiledGramCSROperator(
        a,
        xp,
        values_dtype_hint="float32",
        config=trots.TiledGramConfig(block_rows=64, flop_ratio_threshold=1.0),
    )
    w = xp.asarray(np.random.default_rng(37).uniform(0.1, 5.0, a.shape[0]))

    first = _host(op.gram(w))
    op.gram(xp.asarray(_host(w)))  # equal values, a fresh array: memo hit
    op.gram(w, accumulate_dtype="float16")  # no kernel for it: the exact slot
    assert op.gram_compute_count == 1
    op.gram(w, accumulate_dtype="float32")
    op.gram(w, accumulate_dtype="float32")
    assert op.gram_compute_count == 2
    other = _host(op.gram(w * 2.0))
    assert op.gram_compute_count == 3
    np.testing.assert_allclose(other, 2.0 * first, rtol=1e-13)


def test_device_operator_presents_the_original_row_order(namespace):
    xp = namespace
    _device_namespace(xp)
    a = _banded(700, 45, 3, 16)
    op = trots.TiledGramCSROperator(
        a, xp, config=trots.TiledGramConfig(block_rows=64, flop_ratio_threshold=1.0)
    )
    rng = np.random.default_rng(17)
    x, y = rng.standard_normal(a.shape[1]), rng.standard_normal(a.shape[0])

    np.testing.assert_allclose(_host(op.matvec(xp.asarray(x))), a @ x, rtol=1e-13)
    np.testing.assert_allclose(_host(op.rmatvec(xp.asarray(y))), a.T @ y, rtol=1e-12)
    rows, cols, vals, shape = op.to_coo()
    rebuilt = sp.coo_matrix((_host(vals), (_host(rows), _host(cols))), shape=shape)
    assert (rebuilt.tocsr() != a).nnz == 0


def test_device_gate_keeps_the_plain_path(namespace):
    xp = namespace
    _device_namespace(xp)
    a = _random_sparse(600, 30, 0.6, 14)
    w = np.random.default_rng(15).uniform(0.1, 5.0, a.shape[0])
    op = trots.TiledGramCSROperator(
        a, xp, config=trots.TiledGramConfig(block_rows=64, flop_ratio_threshold=0.5)
    )
    plain = CSROperator(
        xp.asarray(np.asarray(a.indptr, dtype=np.int64)),
        xp.asarray(np.asarray(a.indices, dtype=np.int64)),
        xp.asarray(a.data),
        a.shape,
    )
    assert not op.tiled
    assert op.device_structures() == []
    np.testing.assert_array_equal(
        _host(op.gram(xp.asarray(w))), _host(plain.gram(xp.asarray(w)))
    )


def test_device_shared_segments_merge_weights_and_add_aux_borders(namespace):
    # The host test's [A; A; -A | +1 aux] stack, on the device: one structure,
    # merged weights, the aux border by one masked transpose product.
    xp = namespace
    _device_namespace(xp)
    a = _banded(500, 30, 3, 20)
    m, nx = a.shape
    aux = sp.csr_matrix(
        (np.ones(m), (np.arange(m), np.zeros(m, dtype=int))), shape=(m, 2)
    )
    zero = sp.csr_matrix((m, 2))
    stacked = sp.vstack(
        [sp.hstack([a, zero]), sp.hstack([a, zero]), sp.hstack([-a, aux])], format="csr"
    )
    segments = [
        trots.GramSegment(0, m, a, key="A"),
        trots.GramSegment(m, 2 * m, a, key="A"),
        trots.GramSegment(
            2 * m, 3 * m, a, key="A", sign=-1.0, aux_col=nx, aux_coef=1.0
        ),
    ]
    cfg = trots.TiledGramConfig(block_rows=64, flop_ratio_threshold=1.0)
    dev = trots.TiledGramCSROperator(stacked, xp, segments=segments, config=cfg)
    host = trots.TiledGramCSROperator(stacked, np, segments=segments, config=cfg)
    w = np.random.default_rng(21).uniform(0.0, 3.0, 3 * m)

    g = _host(dev.gram(xp.asarray(w)))

    assert len(dev.device_structures()) == 1
    assert _rel(g, np.asarray(host.gram(w))) < 1e-12
    assert _rel(g, _dense_gram(stacked, w)) < 1e-12
    np.testing.assert_array_equal(g, g.T)


def test_loader_device_block_matches_the_numpy_block(namespace):
    # Loader integration on the device namespaces: CuPy / Torch get the tiled
    # operator (device kernel), tiled_gram=False the plain one; the lowered
    # Grams agree with the NumPy tiled one. (NumPy and array_api_strict are
    # covered by the host tests above.)
    from ipax.problem.linear_ineq import lower_linear_inequalities

    xp = namespace
    _device_namespace(xp)
    inst = _structured_instance()
    cfg = trots.TiledGramConfig(block_rows=128)
    host = _split_problem(np, inst, tiled_gram=cfg)
    tiled = _split_problem(xp, inst, tiled_gram=cfg)
    plain = _split_problem(xp, inst, tiled_gram=False)

    assert trots.TiledGramCSROperator in {
        type(b) for b in tiled.linear_ineq()[0].blocks
    }
    assert all(type(b) is CSROperator for b in plain.linear_ineq()[0].blocks)

    z0 = xp.zeros(tiled.n_vars, dtype=xp.float64)
    jt = lower_linear_inequalities(tiled, z0, xp)._jac
    jp = lower_linear_inequalities(plain, z0, xp)._jac
    jh = _lowered_jac(host)
    w = np.random.default_rng(27).uniform(0.1, 5.0, jh.shape[0])
    ref = np.asarray(jh.gram(w))
    g_t = _host(jt.gram(xp.asarray(w)))
    assert _rel(g_t, ref) < 1e-12
    np.testing.assert_array_equal(g_t, g_t.T)
    assert _rel(_host(jp.gram(xp.asarray(w))), ref) < 1e-12
    reduced = _host(
        jt.gram(xp.asarray(w), accumulate_dtype="float32", hinted_only=True)
    )
    assert _rel(reduced, ref) < 1e-6
    # One device copy per structure, shared by every operator the problem
    # builds (linear_ineq() builds its operator fresh per call).
    tiles_before = dict(tiled._device_tiles)
    again = tiled.linear_ineq()[0]
    z1 = lower_linear_inequalities(tiled, z0, xp)._jac
    z1.gram(xp.asarray(w * 2.0))
    assert tiled._device_tiles.keys() == tiles_before.keys()
    planned = sum(
        len(b._plans) for b in again.blocks if isinstance(b, trots.TiledGramCSROperator)
    )
    assert len(tiled._device_tiles) == planned


def test_device_cache_is_keyed_by_the_kernel_selection(namespace):
    # Two contexts that would materialize a structure differently (here: the
    # densify budget) never share a cache entry; equal ones do.
    xp = namespace
    kind = _device_namespace(xp)
    a = _banded(800, 50, 2, 8)
    cfg = trots.TiledGramConfig(
        block_rows=100, flop_ratio_threshold=1.0, sparse_tile_flop_ratio=1.0
    )
    op = trots.TiledGramCSROperator(a, xp, config=cfg)
    plan = op._plans[0]
    ctx = op._device
    cache = {}

    def context(budget):
        return trots._DeviceGramContext(
            xp, kind, ctx.device, cache, densify_budget_bytes=budget
        )

    rows = plan.structure.shape[0]
    first = context(None).structure(plan.perm, plan.tiles, rows, cfg.block_rows)
    again = context(None).structure(plan.perm, plan.tiles, rows, cfg.block_rows)
    capped = context(1).structure(plan.perm, plan.tiles, rows, cfg.block_rows)
    assert again is first
    assert capped is not first
    assert len(cache) == 2
    assert {key[0] for key in cache} == {id(plan.tiles)}
    assert not any(t.capped for t in first.tiles)
    assert all(t.capped for t in capped.tiles)
    assert all(t.sparse == (kind == "cupy") for t in capped.tiles)


@pytest.mark.parametrize("device_sparse_tiles", [True, False])
def test_device_densify_budget_caps_wide_sparse_tiles(
    namespace, device_sparse_tiles, monkeypatch
):
    # A wide, nearly empty tile: the gate prices it Σ nnz_row², but densified
    # it is a b×c temporary and b·c² flops. Over the budget it is never
    # densified — CuPy: SpGEMM at every precision (whatever
    # device_sparse_tiles says); Torch: the host SciPy product.
    xp = namespace
    kind = _device_namespace(xp)
    rng = np.random.default_rng(70)
    m, n = 600, 3000
    rows = np.repeat(np.arange(m), 2)
    a = sp.csr_matrix(
        (rng.uniform(0.5, 2.0, 2 * m), (rows, rng.integers(0, n, 2 * m))),
        shape=(m, n),
    )
    budget = 1 << 16  # 64 KiB: every 200-row tile (c ≥ 200) densifies above it
    cfg = {
        "block_rows": 200,
        "flop_ratio_threshold": 1.0,
        "device_sparse_tiles": device_sparse_tiles,
    }
    op = trots.TiledGramCSROperator(
        a, xp, config=trots.TiledGramConfig(**cfg, device_densify_budget_bytes=budget)
    )
    assert all(t.sparse for p in op._plans for t in p.tiles)  # planned sparse
    tiles = [t for ds in op.device_structures() for t in ds.tiles]
    assert tiles and all(t.capped for t in tiles)
    assert all(t.width * (t.stop - t.start) * 8 > budget for t in tiles)
    assert all(t.sparse == (kind == "cupy") for t in tiles)
    assert all((t.host is not None) == (kind != "cupy") for t in tiles)

    def no_densify(*_args, **_kwargs):
        raise AssertionError("densified an over-budget sparse tile")

    monkeypatch.setattr(trots._DeviceGramContext, "gram_block", no_densify)
    for weights in ("positive", "signed"):
        w = _weights(weights, m, 71)
        ref = _dense_gram(a, w)
        g = _host(op.gram(xp.asarray(w)))
        assert _rel(g, ref) < 1e-12
        np.testing.assert_array_equal(g, g.T)
        reduced = _host(op.gram(xp.asarray(w), accumulate_dtype="float32"))
        assert _rel(reduced, ref) < 1e-12  # never densified: computed exactly

    # The default budget densifies the same tiles (Torch; CuPy float32).
    monkeypatch.undo()
    roomy = trots.TiledGramCSROperator(a, xp, config=trots.TiledGramConfig(**cfg))
    assert not any(t.capped for ds in roomy.device_structures() for t in ds.tiles)


def test_device_gram_reads_the_host_once_per_call(namespace, monkeypatch):
    # The memo comparison and the per-tile weight flags come back in one
    # host read, whether the call hits the memo or computes.
    xp = namespace
    _device_namespace(xp)
    a = _banded(900, 60, 4, 1)
    op = trots.TiledGramCSROperator(
        a, xp, config=trots.TiledGramConfig(block_rows=64, flop_ratio_threshold=1.0)
    )
    reads = []
    real = trots._DeviceGramContext.to_host

    def spy(self, x):
        reads.append(x.shape)
        return real(self, x)

    monkeypatch.setattr(trots._DeviceGramContext, "to_host", spy)
    w = xp.asarray(np.random.default_rng(72).uniform(0.1, 2.0, a.shape[0]))
    op.gram(w)  # empty memo slot: flags only
    op.gram(w)  # memo hit
    op.gram(w * 2.0)  # memo miss
    assert op.gram_compute_count == 2
    assert len(reads) == 3


def test_loader_device_tiled_problem_solves_like_the_plain_one(namespace):
    from ipax import Options, solve

    xp = namespace
    _device_namespace(xp)
    inst = _structured_instance()
    opts = Options(linsolve="dense", max_iter=150)
    results = []
    for tiled in (trots.TiledGramConfig(block_rows=128), False):
        problem = trots.TROTSProblem(inst, xp, tiled_gram=tiled)
        results.append(
            solve(problem, problem.initial_point(warm_start=False), options=opts)
        )
    r_t, r_p = results
    assert r_t.status == r_p.status
    assert r_t.n_iter == r_p.n_iter
    assert r_t.routes == r_p.routes
    assert abs(r_t.objective - r_p.objective) <= 1e-8 * max(1.0, abs(r_p.objective))


def test_device_context_normalizes_a_raw_torch_module():
    # The device kernels pass ``device=`` keywords; the context normalizes the
    # namespace it is given to the Array API compat wrapper (CPU-runnable).
    torch = pytest.importorskip("torch")
    from ipax.backend.namespace import array_namespace

    ctx = trots._DeviceGramContext(torch, "torch", torch.device("cpu"))

    assert ctx.xp is array_namespace(torch.asarray(0.0))
    assert ctx.xp is not torch


def _raw_cupy():
    """The raw ``cupy`` module (not the compat wrapper), or skip without a GPU."""
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.runtime.getDeviceCount()
        cupy.asarray(0.0)
    except Exception:
        pytest.skip("no usable CUDA device")
    return cupy


def test_device_tiled_gram_accepts_raw_cupy_namespace():
    # Regression (synthetic twin of the Protons_01 integration test): a problem
    # built on the raw ``cupy`` module failed at its first lowered-block Gram
    # because raw ``cupy.asarray`` takes no ``device=`` keyword.
    cupy = _raw_cupy()
    from ipax.problem.linear_ineq import lower_linear_inequalities
    from ipax.testing.backends import import_namespace

    xp = import_namespace("cupy")
    inst = _structured_instance()
    cfg = trots.TiledGramConfig(block_rows=128)
    raw = _split_problem(cupy, inst, tiled_gram=cfg)
    ref = _split_problem(xp, inst, tiled_gram=cfg)
    assert any(
        isinstance(b, trots.TiledGramCSROperator) for b in raw.linear_ineq()[0].blocks
    )

    def lowered(problem, ns):
        z0 = ns.zeros(problem.n_vars, dtype=ns.float64)
        return lower_linear_inequalities(problem, z0, ns)._jac

    j_raw, j_ref = lowered(raw, cupy), lowered(ref, xp)
    w = np.random.default_rng(11).uniform(1e-3, 1e3, j_raw.shape[0])
    g_raw = _host(j_raw.gram(cupy.asarray(w)))
    g_ref = _host(j_ref.gram(xp.asarray(w)))
    # Two device problems scatter their tiles in their own order: agreement to
    # reduction-order round-off, not bit-for-bit.
    np.testing.assert_allclose(g_raw, g_ref, rtol=1e-12, atol=0.0)
