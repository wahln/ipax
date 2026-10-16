"""The TROTS loader's tiled per-structure Gram (``TiledGramCSROperator``).

An application-edge operator (host SciPy/BLAS), so the operator-level tests are
NumPy-only; the core-level parity test runs it through ``ipax.solve``. The
reference everywhere is the dense ``Aᵀ diag(w) A`` or the plain
``CSROperator`` path the operator replaces (Breedveld et al. 2017, eq. 48).
"""

from __future__ import annotations

import gc
import itertools
import math
import os
import weakref

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")
trots = pytest.importorskip("benchmarks.corpus.trots")

from benchmarks.corpus import _trots_kernels as kernels  # noqa: E402
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


@pytest.mark.parametrize("accumulate", [None, "float32"])
@pytest.mark.parametrize("signed", [False, True])
def test_wide_sparse_tiles_are_densified_on_the_host(signed, accumulate, monkeypatch):
    # 100 entries per row over 2000 columns: Σ nnz_row² is 0.25% of b·c²
    # (sparse by the 1% rule) but 5× the tile's b·c entries, so the host
    # densifies it (host_sparse_entry_ratio = 4); the plan's flag and flop
    # count -- the gate statistic, the lane deal, the device kernels -- stay.
    rng = np.random.default_rng(58)
    a = sp.random(256, 2000, density=0.05, format="csr", random_state=rng)
    a.data = rng.uniform(0.5, 2.0, a.nnz)
    w = rng.uniform(-1.0 if signed else 0.1, 2.0, a.shape[0])
    cfg = {"block_rows": 64, "threads": 1}
    new, old = _op(a, **cfg), _op(a, host_sparse_entry_ratio=None, **cfg)
    tiles = [t for p in new._plans for t in p.tiles]
    assert tiles and all(t.sparse and t.host_sparse is False for t in tiles)
    assert [t.flops for t in tiles] == [t.flops for p in old._plans for t in p.tiles]
    assert all(t.sparse and t.host_sparse is None for p in old._plans for t in p.tiles)

    kernels = []
    real = trots._densify

    def spy(csr, acc_dtype, scale=None, pool=None):
        kernels.append(csr.shape)
        return real(csr, acc_dtype, scale, pool)

    monkeypatch.setattr(trots, "_densify", spy)
    g_new = np.asarray(new.gram(w, accumulate_dtype=accumulate))
    assert len(kernels) == len(tiles)  # every tile took the dense kernel
    kernels.clear()
    g_old = np.asarray(old.gram(w, accumulate_dtype=accumulate))
    assert kernels == []  # ... and none of them without the width rule
    ref = _dense_gram(a, w)
    # float32 requests round the densified tile (ssyrk), as for any dense
    # tile; the old SpGEMM stayed float64 for these tiles.
    tol = 1e-12 if accumulate is None else 1e-5
    assert g_new.dtype == np.float64
    assert _rel(g_new, ref) < tol and _rel(g_old, ref) < 1e-12
    np.testing.assert_array_equal(g_new, g_new.T)


def test_multi_lane_grams_keep_spgemm_for_wide_sparse_tiles(monkeypatch):
    # On lanes the SpGEMMs run in parallel and the deal follows the plan's
    # flop counts: the width rule applies to single-lane Grams only.
    rng = np.random.default_rng(59)
    a = sp.random(256, 2000, density=0.05, format="csr", random_state=rng)
    a.data = rng.uniform(0.5, 2.0, a.nnz)
    w = rng.uniform(0.1, 2.0, a.shape[0])
    op = _op(a, block_rows=64, threads=2, lane_min_flops=1.0, lane_budget_bytes=1 << 30)
    assert all(t.host_sparse is False for p in op._plans for t in p.tiles)
    kernels = []
    real = trots._densify
    monkeypatch.setattr(
        trots,
        "_densify",
        lambda *args, **kw: kernels.append(1) or real(*args, **kw),
    )
    g = np.asarray(op.gram(w))
    assert kernels == []
    assert _rel(g, _dense_gram(a, w)) < 1e-12


def test_narrow_sparse_tiles_stay_sparse_on_the_host():
    a = _banded(800, 50, 2, 8)  # ~4 entries per row over ≤ 50 columns
    op = _op(a, block_rows=100, sparse_tile_flop_ratio=1.0)
    assert all(t.host_sparse is True for p in op._plans for t in p.tiles)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_host_sparse_entry_ratio_is_validated(bad):
    with pytest.raises(ValueError, match="host_sparse_entry_ratio"):
        trots.TiledGramConfig(host_sparse_entry_ratio=bad)


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


def _gram_with_lane_budget(a, w, budget, **cfg):
    cfg.setdefault("block_rows", 64)
    cfg.setdefault("flop_ratio_threshold", 1.0)
    cfg.setdefault("lane_min_flops", 1.0)  # lane count tracks the budget, not flops
    op = trots.TiledGramCSROperator(
        a, np, config=trots.TiledGramConfig(lane_budget_bytes=budget, **cfg)
    )
    return np.asarray(op.gram(w))


def test_lane_budget_bytes_changes_the_gram_only_by_round_off():
    # TiledGramConfig's docstring is explicit: the lane count fixes the
    # summation order, and it is a function of lane_budget_bytes (and the
    # data) alone -- so changing the budget in a way that changes the lane
    # count may change the Gram's last bits (float addition is not
    # associative), which is why the budget has no machine-dependent default.
    # This pins the size of that difference, not its presence. This is the verification the 2026-09-28 profile's
    # lane_budget_bytes=4<<30 lever asked for: budgets are scaled down so a
    # tiny matrix (not a multi-GB one) already exercises 1 vs 2 vs 5 vs 8
    # lanes, via the same `lane_budget_bytes // (8*nx*nx)` arithmetic the
    # config uses at n≈1e4 in production (see docs/benchmarks/trots.md's
    # Head-and-Neck_01 profile, which confirms the same non-identity on the
    # real linear block: max|diff| ~ 1e-16 absolute).
    a = _random_sparse(2000, 60, 0.2, 40)
    w = np.random.default_rng(41).uniform(0.0, 5.0, a.shape[0])
    per_lane = 8 * a.shape[1] * a.shape[1]  # one lane's n×n float64 accumulator

    one_lane = _gram_with_lane_budget(a, w, per_lane)
    two_lanes = _gram_with_lane_budget(a, w, 2 * per_lane)
    five_lanes = _gram_with_lane_budget(a, w, 5 * per_lane)

    # Any difference is round-off, not a bug: tiny relative to the matrix,
    # and every candidate agrees with the dense reference to 1e-12.
    assert _rel(one_lane, two_lanes) < 1e-9
    assert _rel(one_lane, _dense_gram(a, w)) < 1e-12
    assert _rel(two_lanes, _dense_gram(a, w)) < 1e-12
    assert _rel(five_lanes, _dense_gram(a, w)) < 1e-12


def test_lane_budget_bytes_is_identical_once_the_lane_count_saturates():
    # A budget change that does NOT change the actual lane count (both above
    # the thread/flops cap) must not change a single bit: the byte budget
    # only enters through the lane *count*.
    a = _random_sparse(2000, 60, 0.2, 40)
    w = np.random.default_rng(41).uniform(0.0, 5.0, a.shape[0])
    per_lane = 8 * a.shape[1] * a.shape[1]

    capped = _gram_with_lane_budget(a, w, 8 * per_lane, threads=8)
    also_capped = _gram_with_lane_budget(a, w, 100 * per_lane, threads=8)

    np.testing.assert_array_equal(capped, also_capped)


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


def _mirror_lower_row_loop_reference(out):
    """The original ``_mirror_lower`` (a per-row Python loop), kept verbatim
    as the reference for the blocked replacement's bit-identity test."""
    out = np.ascontiguousarray(out)
    n = out.shape[0]
    for j in range(n):
        out[j, j + 1 :] = out[j + 1 :, j]
    return out


@pytest.mark.parametrize("n", [0, 1, 2, 255, 256, 257, 511, 600, 1200])
@pytest.mark.parametrize("block", [64, 256, 512, 1024])
def test_mirror_lower_matches_row_loop_reference(n, block, monkeypatch):
    # The blocked mirror (cache-friendly tiles) is a pure data copy, so it
    # must reproduce the row-loop's array bit for bit at every size relative
    # to the block (below it, exactly at it, and straddling it) and at every
    # block size, including a signed zero (a copy must preserve -0.0, not
    # normalize it through arithmetic).
    monkeypatch.setattr(trots, "_HOST_MIRROR_BLOCK", block)
    rng = np.random.default_rng(1000 + n)
    a = rng.standard_normal((n, n))  # upper triangle: garbage, fully overwritten
    tril_rows, tril_cols = np.tril_indices(n)
    a[tril_rows, tril_cols] = rng.standard_normal(tril_rows.shape[0])
    if n > 1:
        a[1, 0] = -0.0  # an off-diagonal entry that gets mirrored (plain
        # assignment, not arithmetic, so it cannot be normalized to +0.0)

    got = trots._mirror_lower(a.copy())
    want = _mirror_lower_row_loop_reference(a.copy())

    np.testing.assert_array_equal(got, want)
    if n > 1:
        # -0.0 survives a pure copy (mirrored from [1, 0] into [0, 1]).
        assert np.signbit(got[0, 1])


def test_mirror_lower_default_block_divides_unevenly_at_hn_scale():
    # A regression pin for the ragged last panel at n not a multiple of the
    # default block (Head-and-Neck_01 is n=9984, not tested directly here to
    # keep the unit suite fast).
    n = 1000  # not a multiple of trots._HOST_MIRROR_BLOCK (512)
    assert n % trots._HOST_MIRROR_BLOCK != 0
    rng = np.random.default_rng(2024)
    lower = np.tril(rng.standard_normal((n, n)))
    a = lower + np.triu(rng.standard_normal((n, n)), 1)

    got = trots._mirror_lower(a.copy())
    want = _mirror_lower_row_loop_reference(a.copy())

    np.testing.assert_array_equal(got, want)


def _full_block_scatter(acc, cols, g, pool=None, kernel=None):
    # The scatter before 2026-10-02: the whole c×c block at ix_(cols, cols),
    # on the calling thread.
    del pool, kernel
    acc[np.ix_(cols, cols)] += g


@pytest.mark.parametrize("weights", ["positive", "signed"])
@pytest.mark.parametrize("scatter_threads", [1, 3, 8])
@pytest.mark.parametrize("lanes", [1, 2])
def test_row_chunked_scatter_is_bitwise_the_full_block_scatter(
    weights, scatter_threads, lanes, monkeypatch
):
    # Every lower-triangle entry receives the same single addition in both
    # schemes and the mirror overwrites the upper one, so the Gram is the
    # same to the bit -- for every thread count, on the multi-lane path
    # (private accumulators, chunked without a pool), and on the gemm
    # (signed) path whose products carry a nonzero upper triangle.
    monkeypatch.setattr(trots, "_SCATTER_POOL_MIN_ENTRIES", 1)  # pool every tile
    monkeypatch.setattr(trots, "_SCATTER_CHUNK_MIN_ENTRIES", 1)  # chunk every tile
    a = _random_sparse(900, 70, 0.15, 50)
    rng = np.random.default_rng(51)
    low = 0.1 if weights == "positive" else -1.0
    w = rng.uniform(low, 2.0, a.shape[0])
    cfg = {
        "block_rows": 64,
        "threads": lanes,
        "lane_min_flops": 1.0,
        "sparse_tile_flop_ratio": 0.0,
        # The NumPy row-chunked scatter, Numba installed or not (the
        # compiled kernel has its own sha256 tests below).
        "compiled_scatter": False,
    }

    new = np.asarray(_op(a, scatter_threads=scatter_threads, **cfg).gram(w))
    monkeypatch.setattr(trots, "_scatter_lower", _full_block_scatter)
    old = np.asarray(_op(a, **cfg).gram(w))

    np.testing.assert_array_equal(new, old)
    assert _rel(new, _dense_gram(a, w)) < 1e-12


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs a 2-worker pool")
def test_the_pool_starts_threads_only_for_wide_tiles(monkeypatch):
    import concurrent.futures

    made = []
    real_pool = concurrent.futures.ThreadPoolExecutor

    class CountingPool(real_pool):
        def __init__(self, *args, **kwargs):
            made.append(kwargs.get("max_workers"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", CountingPool)
    a = _random_sparse(600, 50, 0.2, 52)
    w = np.random.default_rng(53).uniform(0.1, 2.0, a.shape[0])
    cfg = {"block_rows": 64, "threads": 1, "sparse_tile_flop_ratio": 0.0}

    ref = np.asarray(_op(a, scatter_threads=3, **cfg).gram(w))  # c² < threshold
    assert made == []

    monkeypatch.setattr(trots, "_SCATTER_POOL_MIN_ENTRIES", 1)
    pooled = np.asarray(_op(a, scatter_threads=3, **cfg).gram(w))
    assert made == [min(3, os.cpu_count() or 1)]  # one pool per Gram, lazily
    np.testing.assert_array_equal(pooled, ref)

    made.clear()
    single = np.asarray(_op(a, scatter_threads=1, **cfg).gram(w))
    assert made == []  # one worker: no pool at all
    np.testing.assert_array_equal(single, ref)


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs a 2-worker pool")
@pytest.mark.parametrize("acc", [np.float64, np.float32])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("scaled", [False, True])
def test_densify_on_the_pool_is_bitwise_the_whole_tile_toarray(
    dtype, scaled, acc, monkeypatch
):
    # Row-chunked windows, densified into one output and scaled in place:
    # the same values and products as toarray() and *= on the whole tile.
    # Mismatched dtypes are densified per window and cast into their rows
    # (the per-element astype cast), also on the pool.
    monkeypatch.setattr(trots, "_SCATTER_POOL_MIN_ENTRIES", 1)
    csr = _random_sparse(203, 37, 0.3, 54).astype(dtype)
    # Values that pin the cast: an overflow to inf and a round-to-even tie.
    csr.data[0] = 1e39 if dtype == np.float64 else csr.data[0]
    csr.data[1] = 1.0 + 2.0**-24 if dtype == np.float64 else csr.data[1]
    # Non-canonical storage: unsorted indices and duplicate entries (which
    # csr_todense adds in storage order, window or whole tile alike).
    rng = np.random.default_rng(57)
    for r in range(csr.shape[0]):
        lo, hi = csr.indptr[r], csr.indptr[r + 1]
        order = rng.permutation(hi - lo)
        csr.indices[lo:hi] = csr.indices[lo:hi][order]
        csr.data[lo:hi] = csr.data[lo:hi][order]
    if csr.indptr[-1] > 1:
        csr.indices[1] = csr.indices[0] if csr.indptr[1] > 1 else csr.indices[1]
    csr.has_sorted_indices = False
    # The kernel's own reference: whole-tile toarray, astype, then *= by
    # the accumulate-dtype scale (_tile_work casts √w to it).
    scale = np.random.default_rng(55).uniform(0.1, 2.0, 203).astype(acc)
    with np.errstate(over="ignore"):
        want = csr.toarray().astype(acc)
        if scaled:
            want *= scale[:, None]
    calls = []
    with trots._host_pool(trots.TiledGramConfig(scatter_threads=4)) as pool:
        real_map = pool.map
        pool.map = lambda fn, bounds: calls.append(1) or real_map(fn, bounds)
        with np.errstate(over="ignore"):
            got = trots._densify(csr, acc, scale if scaled else None, pool)

    assert calls  # the pooled path, whatever the dtypes
    assert got.dtype == acc
    np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8))


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs a 2-worker pool")
@pytest.mark.parametrize("n", [600, 1000, 1031])
def test_mirror_lower_on_the_pool_matches_the_row_loop(n, monkeypatch):
    monkeypatch.setattr(trots, "_HOST_MIRROR_BLOCK", 64)  # many panels
    monkeypatch.setattr(trots, "_SCATTER_POOL_MIN_ENTRIES", 1)
    rng = np.random.default_rng(56)
    a = rng.standard_normal((n, n))
    a[1, 0] = -0.0
    calls = []
    with trots._host_pool(trots.TiledGramConfig(scatter_threads=4)) as pool:
        real_map = pool.map
        pool.map = lambda fn, bounds: calls.append(len(bounds)) or real_map(fn, bounds)
        got = trots._mirror_lower(a.copy(), pool)
    assert calls  # the panels ran on the pool

    np.testing.assert_array_equal(got, _mirror_lower_row_loop_reference(a.copy()))
    assert np.signbit(got[0, 1])


@pytest.mark.parametrize("c", [1, 2, 7, 100, 9684])
@pytest.mark.parametrize("chunks", [1, 4, 32, 200])
def test_lower_row_bounds_cover_the_rows_in_equal_area_chunks(c, chunks):
    bounds = trots._lower_row_bounds(c, chunks)

    assert bounds[0] == 0 and bounds[-1] == c
    assert all(r0 < r1 for r0, r1 in itertools.pairwise(bounds))
    assert len(bounds) - 1 <= min(chunks, c)
    if c >= 100 * chunks:
        # Row r holds r + 1 lower entries: each chunk ~ c²/(2·chunks) of them.
        areas = [
            (r1 * (r1 + 1) - r0 * (r0 + 1)) / 2 for r0, r1 in itertools.pairwise(bounds)
        ]
        assert max(areas) <= 1.1 * min(areas)


def test_scatter_threads_must_be_positive():
    with pytest.raises(ValueError, match="scatter_threads"):
        trots.TiledGramConfig(scatter_threads=0)


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


def test_superseded_memo_is_released_before_the_next_product(monkeypatch):
    # The memo serves repeats of the *same* weights (δ_w retries); a new
    # weight vector supersedes it, so it must not stay alive (an n×n array)
    # while the next Gram is accumulated beside it.
    a = _random_sparse(300, 20, 0.1, 18)
    op = _op(a, block_rows=64)
    w = np.random.default_rng(19).uniform(0.1, 5.0, a.shape[0])
    held = []
    memos = []
    real = trots._accumulate_tiles

    def spy(*args, **kwargs):
        gc.collect()
        held.append([ref() is not None for ref in memos])
        return real(*args, **kwargs)

    monkeypatch.setattr(trots, "_accumulate_tiles", spy)
    op.gram(w)
    memos.append(weakref.ref(op._memo[None][1]))
    op.gram(w * 2.0)

    assert held == [[], [False]]
    assert op.gram_compute_count == 2
    assert None in op._memo  # the new product is memoized


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
# Restoration active-set Grams: only the rows with a nonzero weight
# --------------------------------------------------------------------------- #
def _active_weights(m, k, seed, signed=False):
    """``k`` nonzero weights among ``m`` (restoration's 1[g > 0], or signed)."""
    rng = np.random.default_rng(seed)
    w = np.zeros(m)
    idx = rng.choice(m, k, replace=False)
    w[idx] = rng.uniform(-1.0, 2.0, k) if signed else 1.0
    return w


def _plain_csr(a):
    return CSROperator(
        np.asarray(a.indptr, dtype=np.int64),
        np.asarray(a.indices, dtype=np.int64),
        a.data,
        a.shape,
    )


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("block_rows", [16, 5000])
def test_active_row_gram_matches_the_dense_reference(signed, block_rows):
    a = _random_sparse(800, 30, 0.7, 60)  # dense-ish: what the gate keeps off
    op = trots.ActiveRowGramCSROperator(
        a, np, config=trots.TiledGramConfig(block_rows=block_rows)
    )
    w = _active_weights(a.shape[0], 40, 61, signed)

    g = np.asarray(op.gram(w))

    assert op.active_rows.count == 1
    assert _rel(g, _dense_gram(a, w)) < 1e-13
    np.testing.assert_array_equal(g, g.T)
    # No active row at all: the zero matrix, no product.
    z = np.asarray(op.gram(np.zeros(a.shape[0])))
    np.testing.assert_array_equal(z, np.zeros((a.shape[1], a.shape[1])))


def test_active_row_gram_keeps_the_plain_product_for_interior_point_weights(
    monkeypatch,
):
    # Σ_s > 0 on every row (every IPM iteration): the adapter's product,
    # bit for bit, and the compacted kernel is never reached.
    a = _random_sparse(600, 25, 0.6, 62)
    op = trots.ActiveRowGramCSROperator(a, np)
    plain = _plain_csr(a)
    w = np.random.default_rng(63).uniform(0.1, 5.0, a.shape[0])

    def boom(*_args, **_kwargs):
        raise AssertionError("compacted a positive weight vector")

    monkeypatch.setattr(trots._ActiveRowGram, "_product", boom)
    np.testing.assert_array_equal(np.asarray(op.gram(w)), np.asarray(plain.gram(w)))
    for acc in ("float32", None):
        np.testing.assert_array_equal(
            np.asarray(op.gram(w, accumulate_dtype=acc)),
            np.asarray(plain.gram(w, accumulate_dtype=acc)),
        )
    # One row above the threshold share: still the plain product.
    m = a.shape[0]
    k = int(op.active_rows.max_fraction * m)
    over = np.zeros(m)
    over[: k + 1] = 1.0
    np.testing.assert_array_equal(
        np.asarray(op.gram(over)), np.asarray(plain.gram(over))
    )
    # Exactly at the threshold share: compacted (same matrix, own round-off).
    monkeypatch.undo()
    at = np.zeros(m)
    at[:k] = 1.0
    assert _rel(np.asarray(op.gram(at)), np.asarray(plain.gram(at))) < 1e-13
    assert op.active_rows.count == 1


def test_active_row_gram_non_finite_weights_agree_with_the_plain_verdict():
    # Non-finite weights stay in the active set and take the gemm branch;
    # the finiteness pattern and the finite entries match the plain product.
    a = _random_sparse(500, 20, 0.6, 73)
    op = trots.ActiveRowGramCSROperator(a, np)
    plain = _plain_csr(a)
    for bad in (np.inf, np.nan):
        w = _active_weights(a.shape[0], 25, 74, signed=True)
        w[np.flatnonzero(w)[3]] = bad
        g, g_plain = np.asarray(op.gram(w)), np.asarray(plain.gram(w))
        np.testing.assert_array_equal(np.isfinite(g), np.isfinite(g_plain))
        ok = np.isfinite(g_plain)
        if ok.any():  # 0·inf spreads NaN over a dense-ish row's whole block
            assert _rel(g[ok], g_plain[ok]) < 1e-13
    assert op.active_rows.count == 2


def test_active_row_gram_float32_request_and_memo():
    a = _random_sparse(500, 20, 0.6, 64)
    a.data = a.data.astype(np.float32).astype(np.float64)
    op = trots.ActiveRowGramCSROperator(a, np, values_dtype_hint="float32")
    w = _active_weights(a.shape[0], 30, 65)
    exact = np.asarray(op.gram(w))
    again = np.asarray(op.gram(w.copy()))
    reduced = np.asarray(op.gram(w, accumulate_dtype="float32", hinted_only=True))

    assert op.active_rows.count == 2  # one slot per effective precision
    np.testing.assert_array_equal(exact, again)
    assert reduced.dtype == np.float64
    assert _rel(reduced, exact) < 1e-6
    # An unhinted operator keeps a hinted-only float32 request exact.
    plain64 = trots.ActiveRowGramCSROperator(a, np)
    np.testing.assert_array_equal(
        np.asarray(plain64.gram(w, accumulate_dtype="float32", hinted_only=True)),
        exact,
    )


def test_active_row_rmatvec_matches_the_plain_product():
    a = _random_sparse(700, 30, 0.6, 70)
    op = trots.ActiveRowGramCSROperator(a, np)
    plain = _plain_csr(a)
    rng = np.random.default_rng(71)
    sparse_v = _active_weights(a.shape[0], 20, 72, signed=True)  # compacted
    dense_v = rng.standard_normal(a.shape[0])  # the plain product
    for v in (sparse_v, dense_v, np.zeros(a.shape[0])):
        np.testing.assert_array_equal(
            np.asarray(op.rmatvec(v)), np.asarray(plain.rmatvec(v))
        )
    assert op.active_rows.rmatvec_count == 2  # the sparse and the zero vector
    # A column (rank-2) keeps the plain product and its shape.
    col = np.asarray(op.rmatvec(sparse_v[:, None]))
    np.testing.assert_array_equal(col, np.asarray(plain.rmatvec(sparse_v[:, None])))
    assert op.active_rows.rmatvec_count == 2
    V = rng.standard_normal((a.shape[0], 2))
    np.testing.assert_array_equal(
        np.asarray(op.rmatmat(V)), np.asarray(plain.rmatmat(V))
    )


def test_tiled_operator_serves_active_sets_from_the_active_rows(monkeypatch):
    a = _banded(900, 60, 4, 66)
    op = _op(a, block_rows=64)
    assert op.tiled
    w = _active_weights(a.shape[0], 25, 67)

    def boom(*_args, **_kwargs):
        raise AssertionError("ran every tile for an active-set Gram")

    monkeypatch.setattr(trots.TiledGramCSROperator, "_tiled_gram", boom)
    g = np.asarray(op.gram(w))

    assert op.active_rows.count == 1
    assert op.gram_compute_count == 0
    assert _rel(g, _dense_gram(a, w)) < 1e-13
    # A hinted-only float32 request on unhinted data stays exact (memo hit);
    # on float32-hinted data it accumulates reduced, in its own memo slot.
    np.testing.assert_array_equal(
        np.asarray(op.gram(w, accumulate_dtype="float32", hinted_only=True)), g
    )
    assert op.active_rows.count == 1
    b = a.copy()
    b.data = b.data.astype(np.float32).astype(np.float64)
    op32 = trots.TiledGramCSROperator(
        b,
        np,
        values_dtype_hint="float32",
        config=trots.TiledGramConfig(block_rows=64, flop_ratio_threshold=1.0),
    )
    exact = np.asarray(op32.gram(w))
    reduced = np.asarray(op32.gram(w, accumulate_dtype="float32", hinted_only=True))
    assert op32.active_rows.count == 2
    assert reduced.dtype == np.float64
    assert 0.0 < _rel(reduced, exact) < 1e-6


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


def test_loader_gates_dense_blocks_back_to_the_plain_product(monkeypatch):
    # A group whose plan fails the gate goes to the core as the plain product
    # (an ActiveRowGramCSROperator on NumPy: the adapter's Gram except for
    # restoration's active sets) — and the loader consults the plan first,
    # so no tiled operator is ever built for it.
    def boom(*_args, **_kwargs):
        raise AssertionError("built a tiled operator for a gated-off group")

    monkeypatch.setattr(trots.TiledGramCSROperator, "__init__", boom)
    problem = _split_problem(
        np, tiled_gram=trots.TiledGramConfig(flop_ratio_threshold=0.0)
    )
    op = problem.linear_ineq()[0]
    assert all(type(b) is trots.ActiveRowGramCSROperator for b in op.blocks)


def test_gated_off_groups_lower_to_the_plain_products():
    # The subclass is lowered as a signed row selection (no triplet rewrite);
    # the lowered block is the plain problem's, product for product. The
    # upper-only float32 group keeps the triplet-lowered row order, so its
    # interior-point Gram and products are bit-identical; the float64 group
    # has min rows (lower bounds first after lowering): round-off only.
    from ipax.backend.operators import _SignedRowSelection

    inst = _structured_instance()
    gated = _split_problem(
        np, inst, tiled_gram=trots.TiledGramConfig(flop_ratio_threshold=0.0)
    )
    plain = _split_problem(np, inst, tiled_gram=False)
    jg, jp = _lowered_jac(gated), _lowered_jac(plain)
    assert jg.shape == jp.shape
    assert [type(b) for b in jg.blocks] == [_SignedRowSelection] * 2
    assert [type(b) for b in jp.blocks] == [CSROperator] * 2
    rng = np.random.default_rng(68)
    w = rng.uniform(0.1, 5.0, jg.shape[0])
    v, u = rng.standard_normal(jg.shape[1]), rng.standard_normal(jg.shape[0])
    k32 = jg.blocks[0].shape[0]  # the float32 group: upper rows only
    b32g, b32p = jg.blocks[0], jp.blocks[0]
    for acc in (None, "float32"):
        np.testing.assert_array_equal(
            np.asarray(b32g.gram(w[:k32], accumulate_dtype=acc, hinted_only=True)),
            np.asarray(b32p.gram(w[:k32], accumulate_dtype=acc, hinted_only=True)),
        )
    np.testing.assert_array_equal(
        np.asarray(b32g.matvec(v)), np.asarray(b32p.matvec(v))
    )
    np.testing.assert_array_equal(
        np.asarray(b32g.rmatvec(u[:k32])), np.asarray(b32p.rmatvec(u[:k32]))
    )
    assert _rel(np.asarray(jg.gram(w)), np.asarray(jp.gram(w))) < 1e-13
    assert _rel(np.asarray(jg.rmatvec(u)), np.asarray(jp.rmatvec(u))) < 1e-13
    np.testing.assert_array_equal(np.asarray(jg.matvec(v)), np.asarray(jp.matvec(v)))
    # A restoration active set: the compacted product, same matrix.
    act = _active_weights(jg.shape[0], 12, 69)
    assert _rel(np.asarray(jg.gram(act)), np.asarray(jp.gram(act))) < 1e-13
    assert all(b._op.active_rows.count == 1 for b in jg.blocks)


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


def test_non_numpy_namespaces_keep_gated_off_groups_plain(namespace):
    # The active-row Gram is host BLAS: on any other namespace (CuPy / Torch
    # device kernel, array_api_strict) a gated-off group stays the plain
    # CSROperator, lowered at the triplet level by the core, as before.
    if trots._is_host_numpy(namespace):
        pytest.skip("the host-NumPy namespace takes ActiveRowGramCSROperator")
    problem = _split_problem(
        namespace, tiled_gram=trots.TiledGramConfig(flop_ratio_threshold=0.0)
    )
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
        device_sparse_tile_flop_ratio=1.0,  # every sparse tile stays sparse
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


def test_device_superseded_memo_is_released_before_the_next_product(
    namespace, monkeypatch
):
    # Device twin of the host test: on Head-and-Neck_01 the stale n×n memo
    # (0.76 GB) sat beside the new Gram's result and tile buffers at the
    # Gram's memory peak.
    xp = namespace
    _device_namespace(xp)
    a = _random_sparse(300, 20, 0.1, 36)
    op = trots.TiledGramCSROperator(
        a, xp, config=trots.TiledGramConfig(block_rows=64, flop_ratio_threshold=1.0)
    )
    w = np.random.default_rng(37).uniform(0.1, 5.0, a.shape[0])
    held = []
    memos = []
    real = trots._device_run

    def spy(*args, **kwargs):
        gc.collect()
        held.append([ref() is not None for ref in memos])
        return real(*args, **kwargs)

    monkeypatch.setattr(trots, "_device_run", spy)
    op.gram(xp.asarray(w))
    memos.append(weakref.ref(op._memo[None][1]))
    op.gram(xp.asarray(w))  # a memo hit: no product
    op.gram(xp.asarray(w * 2.0))

    assert held == [[], [False]]
    assert op.gram_compute_count == 2
    assert None in op._memo


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


def _cupy_namespace(namespace):
    """Skip unless ``namespace`` is a CuPy namespace (the device CSR path)."""
    if _device_namespace(namespace) != "cupy":
        pytest.skip("the single-copy device CSR is the CuPy adapter's")
    import cupy

    return cupy


@pytest.mark.parametrize("gated", [False, True])
def test_device_operator_holds_the_block_once(namespace, gated):
    # Regression (Head-and-Neck_01 on an 8 GB card): the CuPy operator kept
    # the block's COO triplets (int64 rows and columns plus the values) next
    # to the sparse adapter's CSR copy, 36 bytes per nonzero where the CSR
    # alone is 12, and building that CSR from the triplets peaked at ~3.5x
    # the block. The CSR is now the only device copy: its values are the
    # operator's COO values (one buffer), built straight from the host CSR.
    cupy = _cupy_namespace(namespace)
    xp = namespace
    a = _banded(6000, 400, 30, 41)
    cfg = trots.TiledGramConfig(
        block_rows=256, flop_ratio_threshold=0.0 if gated else 1.0
    )
    rng = np.random.default_rng(42)
    x = xp.asarray(rng.standard_normal(a.shape[1]))
    pool = cupy.get_default_memory_pool()
    cupy.cuda.Device().synchronize()
    before = pool.used_bytes()

    op = trots.TiledGramCSROperator(a, xp, config=cfg)
    y = op.matvec(x)  # builds the adapter operator

    cupy.cuda.Device().synchronize()
    held = pool.used_bytes() - before
    assert op.tiled is not gated
    # The CSR (8-byte values + 4-byte column indices per nonzero) plus
    # O(rows) index pointers and the product vector.
    assert held <= 12 * a.nnz + 16 * (a.shape[0] + 1) + 8 * a.shape[0] + 2**16
    csr = op._adapter_op().cupy_matrix
    assert csr.data.data.ptr == op.coo_values().data.ptr  # one values buffer
    np.testing.assert_allclose(_host(y), a @ _host(x), rtol=1e-12, atol=1e-12)


def test_device_operator_matches_the_plain_operator_bitwise(namespace):
    # The single-copy CSR is the same canonical CSR the adapter built from the
    # triplets before (same indices, values and index dtypes), so the adapter
    # products are bit-identical to the plain CSROperator's, and the COO
    # triplets it rebuilds on demand are the plain operator's. The transpose
    # products (rmatvec, rmatmat, the column energies of gram_diagonal) are
    # the exception: cuSPARSE accumulates them with atomics, so
    # they differ in the last bits between two calls on the same matrix.
    _cupy_namespace(namespace)
    xp = namespace
    a = _banded(900, 60, 4, 43)
    cfg = trots.TiledGramConfig(block_rows=64, flop_ratio_threshold=1.0)
    op = trots.TiledGramCSROperator(a, xp, config=cfg)
    plain = CSROperator(
        xp.asarray(np.asarray(a.indptr, dtype=np.int64)),
        xp.asarray(np.asarray(a.indices, dtype=np.int64)),
        xp.asarray(a.data),
        a.shape,
        symmetric=False,
    )
    rng = np.random.default_rng(44)
    x = xp.asarray(rng.standard_normal(a.shape[1]))
    y = xp.asarray(rng.standard_normal(a.shape[0]))
    X = xp.asarray(rng.standard_normal((a.shape[1], 3)))
    Y = xp.asarray(rng.standard_normal((a.shape[0], 3)))
    w = xp.asarray(rng.uniform(0.1, 5.0, a.shape[0]))
    v = xp.asarray(rng.uniform(0.1, 5.0, a.shape[1]))
    pairs = [
        (op.matvec(x), plain.matvec(x)),
        (op.matmat(X), plain.matmat(X)),
        (op.row_inf_norms(), plain.row_inf_norms()),
        (op.row_gram_diagonal(v), plain.row_gram_diagonal(v)),
        (op.coo_values(), plain.coo_values()),
        *zip(op.to_coo()[:3], plain.to_coo()[:3], strict=True),
    ]
    for got, ref in pairs:
        assert got.dtype == ref.dtype
        np.testing.assert_array_equal(_host(got), _host(ref))
    assert op.to_coo()[3] == plain.to_coo()[3]
    for got, ref in (
        (op.diagonal(), plain.diagonal()),
        (op.dense_matrix(), plain.dense_matrix()),
        *zip(op.gram_coo(w)[:3], plain.gram_coo(w)[:3], strict=True),
    ):
        assert got.dtype == ref.dtype
        np.testing.assert_array_equal(_host(got), _host(ref))
    for probe in (
        "shape",
        "gram_accumulate_dtype_hint",
        "gram_capable",
        "gram_coo_capable",
        "gram_fill_estimate",
        "coo_pattern_signature",
        "symmetry_hint",
    ):
        got, ref = getattr(op, probe), getattr(plain, probe)
        assert (got if probe == "shape" else got()) == (
            ref if probe == "shape" else ref()
        ), probe
    for got, ref in (
        (op.rmatvec(y), plain.rmatvec(y)),
        (op.rmatmat(Y), plain.rmatmat(Y)),
        (op.gram_diagonal(w), plain.gram_diagonal(w)),  # (A∘A)ᵀ w
    ):
        np.testing.assert_allclose(_host(got), _host(ref), rtol=1e-13, atol=0)


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
    # (The cache also holds the device's sparse-tile calibration, one entry.)
    tile_entries = [k for k in tiled._device_tiles if k[0] != trots._CALIBRATION_TAG]
    assert len(tile_entries) == planned


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
            xp, kind, ctx.device, cache, densify_budget_bytes=budget, sparse_ratio=1.0
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


# --------------------------------------------------------------------------- #
# Size-aware device sparse-tile choice (the device break-even ratio)
# --------------------------------------------------------------------------- #
# Head-and-Neck_01's eight host-sparse tiles (rows b, columns c, Σ nnz_row²,
# seconds by cuSPARSE SpGEMM, seconds densified + cuBLAS syrk), each
# multiplied alone, float64, min of 3 (2026-09-28, RTX 4070 Laptop, CuPy 14).
_HN01_SPARSE_TILES = (
    (2048, 2059, 46766, 5.81e-04, 3.96e-02),
    (2048, 2080, 46854, 5.28e-04, 3.97e-02),
    (2048, 2082, 46974, 5.52e-04, 3.97e-02),
    (2048, 2072, 47058, 6.42e-04, 3.96e-02),
    (1785, 1797, 40825, 5.67e-04, 2.78e-02),
    (2048, 3609, 107043178, 2.35e00, 1.05e-01),
    (2048, 9684, 789611950, 2.48e01, 1.87e00),
    (1250, 8158, 332929587, 1.06e01, 1.10e00),
)


def test_device_break_even_ratio_is_the_throughput_ratio():
    # 1e6 products in 10 ms (1e8/s) against 1e9 flops in 4 ms (2.5e11/s).
    ratio = trots._device_break_even_ratio(1e6, 0.01, 1e9, 0.004)
    assert ratio == pytest.approx(4e-4)
    at = ratio * 100 * 30 * 30  # a tile exactly at the break-even
    assert trots._prefer_device_sparse(at, 100, 30, ratio)
    assert not trots._prefer_device_sparse(at * 1.01, 100, 30, ratio)


@pytest.mark.parametrize(
    "timings",
    [
        (1e6, 0.0, 1e9, 0.004),
        (1e6, 0.01, 1e9, 0.0),
        (1e6, float("nan"), 1e9, 0.004),
        (1e6, 0.01, 1e9, float("inf")),
        (0.0, 0.01, 1e9, 0.004),
        (1e6, -1.0, 1e9, 0.004),
    ],
)
def test_degenerate_calibration_densifies(timings):
    assert trots._device_break_even_ratio(*timings) == 0.0


def test_calibrated_choice_picks_the_faster_kernel_on_head_and_neck_tiles():
    # The context calibrates on the candidate with the least dense work; on
    # the measured Head-and-Neck_01 tiles that ratio (~3.5e-4, against the
    # host's 1e-2) picks the faster kernel on every tile: 37.7 s of sparse
    # tiles by SpGEMM become 3.1 s.
    b, c, s2, t_sparse, t_dense = min(
        _HN01_SPARSE_TILES, key=lambda t: t[0] * t[1] ** 2
    )
    ratio = trots._device_break_even_ratio(s2, t_sparse, float(b) * c * c, t_dense)
    assert 1e-4 < ratio < 1e-3
    for b, c, s2, t_sparse, t_dense in _HN01_SPARSE_TILES:
        assert trots._prefer_device_sparse(s2, b, c, ratio) == (t_sparse < t_dense)


def _mixed_matrix():
    """A structure whose tiles span the break-even: a smoothing-like band
    (two entries per row over the whole column range) above a dense-ish
    block."""
    rng = np.random.default_rng(80)
    rows = np.repeat(np.arange(400), 2)
    band = sp.csr_matrix(
        (rng.uniform(0.5, 2.0, 800), (rows, rng.integers(0, 400, 800))),
        shape=(400, 400),
    )
    return sp.vstack([band, _random_sparse(400, 400, 0.3, 81)]).tocsr()


def _mixed_sparse_plan():
    """The mixed structure planned with every tile sparse (host ratio 1), so
    each tile is a device candidate."""
    a = _mixed_matrix()
    cfg = trots.TiledGramConfig(block_rows=100, sparse_tile_flop_ratio=1.0)
    perm, tiles = trots._plan_structure(a, cfg, np.float64)
    assert all(t.sparse for t in tiles)
    return perm, tiles, a.shape[0], cfg.block_rows


def _stub_attach(monkeypatch):
    """Record the tiles given a device CSR instead of building one (cupyx):
    the kernel choice itself needs no GPU."""
    attached = []

    def attach(self, t, dt, data_dev):
        attached.append((t.start, t.stop))

    monkeypatch.setattr(trots._DeviceGramContext, "_attach_sparse", attach)
    return attached


def _host_context_posing_as_cupy(cache=None, **kwargs):
    """The device context on the NumPy compat namespace with ``kind="cupy"``:
    materialization and the kernel choice run on the host (neither launches
    a kernel)."""
    return trots._DeviceGramContext(np, "cupy", "cpu", cache, **kwargs)


def _ratios(tiles):
    return [t.flops / ((t.stop - t.start) * float(len(t.cols)) ** 2) for t in tiles]


def _split_ratio(tiles):
    """A break-even halfway between two neighbouring tile statistics (no tile
    sits at the threshold, where rounding could decide it)."""
    r = sorted(set(_ratios(tiles)))
    k = len(r) // 2
    return 0.5 * (r[k - 1] + r[k])


def test_pinned_ratio_selects_tiles_without_measuring(monkeypatch):
    perm, tiles, m, rows = _mixed_sparse_plan()
    attached = _stub_attach(monkeypatch)

    def no_measure(*_args, **_kwargs):
        raise AssertionError("a pinned ratio must not be calibrated")

    monkeypatch.setattr(trots._DeviceGramContext, "_measure_break_even", no_measure)
    ratio = _split_ratio(tiles)
    ctx = _host_context_posing_as_cupy(sparse_ratio=ratio)
    ds = ctx.structure(perm, tiles, m, rows)

    chosen = [t.sparse for t in ds.tiles]
    assert chosen == [r <= ratio for r in _ratios(tiles)]
    assert any(chosen) and not all(chosen)  # the plan spans the break-even
    assert attached == [(t.start, t.stop) for t in ds.tiles if t.sparse]
    assert ctx.sparse_break_even() == ratio
    # The extremes: 1.0 keeps every sparse tile sparse (the choice before the
    # calibration existed), 0.0 densifies them all.
    keep = _host_context_posing_as_cupy(sparse_ratio=1.0)
    assert all(t.sparse for t in keep.structure(perm, tiles, m, rows).tiles)
    none = _host_context_posing_as_cupy(sparse_ratio=0.0)
    assert not any(t.sparse for t in none.structure(perm, tiles, m, rows).tiles)


def test_calibration_runs_once_per_cache_on_the_cheapest_candidate(monkeypatch):
    perm, tiles, m, rows = _mixed_sparse_plan()
    _stub_attach(monkeypatch)
    ratio = _split_ratio(tiles)
    probes = []

    def measure(self, t, dt, flat_dev, data_dev):
        probes.append((t.start, t.stop))
        return ratio

    monkeypatch.setattr(trots._DeviceGramContext, "_measure_break_even", measure)
    cache = {}
    first = _host_context_posing_as_cupy(cache)
    assert first.sparse_break_even() is None
    ds = first.structure(perm, tiles, m, rows)
    cheapest = min(tiles, key=lambda t: (t.stop - t.start) * len(t.cols) ** 2)
    assert probes == [(cheapest.start, cheapest.stop)]
    assert [t.sparse for t in ds.tiles] == [r <= ratio for r in _ratios(tiles)]
    # A second context on the same cache and device (the exact Hessian's,
    # say) reuses the calibration, also for a structure it plans itself.
    perm2, tiles2, m2, _rows = _mixed_sparse_plan()
    second = _host_context_posing_as_cupy(cache)
    assert second.sparse_break_even() == ratio
    second.structure(perm2, tiles2, m2, rows)
    assert len(probes) == 1
    # A pinned context never shares its tiles with a calibrated one.
    pinned = _host_context_posing_as_cupy(cache, sparse_ratio=ratio)
    assert pinned.signature != first.signature
    assert pinned.structure(perm, tiles, m, rows) is not ds


def test_over_budget_and_disabled_sparse_tiles_skip_the_choice(monkeypatch):
    perm, tiles, m, rows = _mixed_sparse_plan()
    _stub_attach(monkeypatch)

    def no_measure(*_args, **_kwargs):
        raise AssertionError("no candidate tile, nothing to calibrate")

    monkeypatch.setattr(trots._DeviceGramContext, "_measure_break_even", no_measure)
    # Over the densify budget: SpGEMM whatever the ratio would say.
    capped = _host_context_posing_as_cupy(densify_budget_bytes=1)
    ds = capped.structure(perm, tiles, m, rows)
    assert all(t.capped and t.sparse for t in ds.tiles)
    # device_sparse_tiles=False: every tile densified, nothing calibrated.
    off = _host_context_posing_as_cupy(sparse_tiles=False)
    assert not any(t.sparse for t in off.structure(perm, tiles, m, rows).tiles)


@pytest.mark.parametrize("bad", [float("nan"), -1e-4, float("inf")])
def test_device_ratio_must_be_finite_and_nonnegative(bad):
    with pytest.raises(ValueError, match="device_sparse_tile_flop_ratio"):
        trots.TiledGramConfig(device_sparse_tile_flop_ratio=bad)
    assert trots.TiledGramConfig(device_sparse_tile_flop_ratio=0.0)


def test_host_kernel_ignores_the_device_ratio():
    # NumPy runs never consult the device break-even: bitwise the same Gram.
    a = _banded(900, 60, 4, 1)
    w = np.random.default_rng(82).uniform(0.1, 2.0, a.shape[0])
    base = _op(a, block_rows=64, sparse_tile_flop_ratio=0.3)
    pinned = _op(
        a, block_rows=64, sparse_tile_flop_ratio=0.3, device_sparse_tile_flop_ratio=0.0
    )
    assert any(t.sparse for p in base._plans for t in p.tiles)
    np.testing.assert_array_equal(np.asarray(pinned.gram(w)), np.asarray(base.gram(w)))


def test_device_calibration_measures_a_finite_ratio(namespace):
    # On CuPy the first materialization with sparse candidates times one tile
    # both ways. Whichever kernels the measured ratio picks, the Gram agrees
    # with the dense reference, and with the all-sparse choice (pinned 1.0)
    # to reduction-order round-off.
    xp = namespace
    kind = _device_namespace(xp)
    if kind != "cupy":
        pytest.skip("only CuPy has a device sparse kernel to calibrate")
    a = _mixed_matrix()
    w = np.random.default_rng(85).uniform(0.1, 2.0, a.shape[0])
    ref = _dense_gram(a, w)
    grams = {}
    for label, ratio in (("calibrated", None), ("all-sparse", 1.0)):
        cfg = trots.TiledGramConfig(
            block_rows=100,
            flop_ratio_threshold=1.0,
            sparse_tile_flop_ratio=1.0,
            device_sparse_tile_flop_ratio=ratio,
        )
        op = trots.TiledGramCSROperator(a, xp, config=cfg)
        tiles = [t for ds in op.device_structures() for t in ds.tiles]
        measured = op._device.sparse_break_even()
        assert measured is not None and math.isfinite(measured) and measured >= 0.0
        if ratio is not None:
            assert all(t.sparse for t in tiles)
        grams[label] = _host(op.gram(xp.asarray(w)))
        assert _rel(grams[label], ref) < 1e-12
        np.testing.assert_array_equal(grams[label], grams[label].T)
    assert _rel(grams["calibrated"], grams["all-sparse"]) < 1e-13


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_cupy_block_scatter_matches_add_at_on_the_lower_triangle(
    namespace, layout, dtype
):
    # The CuPy scatter kernel replaces cupy.add.at with broadcast ix_ indices
    # (a c x c int64 index temporary): the lower triangle -- all the device
    # Gram keeps before its mirror -- must be bitwise the same, whatever the
    # block's memory order (syrk may return F-ordered blocks) and dtype
    # (float32 accumulate requests scatter float32 blocks into float64).
    xp = namespace
    if _device_namespace(xp) != "cupy":
        pytest.skip("the scatter kernel is CuPy's")
    import cupy

    rng = np.random.default_rng(90)
    n, c = 37, 23
    cols = cupy.asarray(np.sort(rng.choice(n, c, replace=False)).astype(np.int64))
    base = rng.standard_normal((n, n))
    block = cupy.asarray(rng.standard_normal((c, c)).astype(dtype), order=layout)
    ctx = trots._DeviceGramContext(xp, "cupy", cupy.cuda.Device())
    got = cupy.asarray(base)
    ctx.scatter_block(got, cols, block)
    ref = cupy.asarray(base)
    cupy.add.at(ref, (cols[:, None], cols[None, :]), block)
    low = np.tril(np.ones((n, n), dtype=bool))
    np.testing.assert_array_equal(cupy.asnumpy(got)[low], cupy.asnumpy(ref)[low])
    # Entries outside the tile's columns are untouched.
    touched = np.zeros((n, n), dtype=bool)
    idx = cupy.asnumpy(cols)
    touched[np.ix_(idx, idx)] = True
    np.testing.assert_array_equal(cupy.asnumpy(got)[~touched], base[~touched])


@pytest.mark.parametrize("accumulate", [None, "float32"])
@pytest.mark.parametrize("weights", ["positive", "one-signed-tile"])
def test_cupy_dense_workspace_is_bitwise_the_allocating_path(
    namespace, monkeypatch, accumulate, weights
):
    # The per-call workspace (one tile buffer, one C-ordered syrk result
    # reused by every densified tile) changes where the products land, not
    # what they are: bitwise the Gram of the allocate-per-tile path.
    xp = namespace
    if _device_namespace(xp) != "cupy":
        pytest.skip("the dense workspace is CuPy's")
    a = _random_sparse(1500, 300, 0.1, 91)
    a.data = a.data.astype(np.float32).astype(np.float64)
    w = np.random.default_rng(92).uniform(0.1, 3.0, a.shape[0])
    if weights == "one-signed-tile":
        # One tile takes the gemm fallback (its own product), the others the
        # workspace syrk: the workspace must not leak between the two.
        w[:10] = -0.5
    cfg = trots.TiledGramConfig(
        block_rows=256, flop_ratio_threshold=1.0, sparse_tile_flop_ratio=0.0
    )
    op = trots.TiledGramCSROperator(a, xp, config=cfg)
    used = trots._DenseWorkspace.for_tiles(
        op._device,
        [(ds, None, None) for ds in op.device_structures()],
        "float64",
        xp.float64,
    )
    assert used is not None
    with_work = _host(op.gram(xp.asarray(w), accumulate_dtype=accumulate))
    monkeypatch.setattr(
        trots._DenseWorkspace, "for_tiles", classmethod(lambda cls, *a, **k: None)
    )
    fresh = trots.TiledGramCSROperator(a, xp, config=cfg)
    alloc = _host(fresh.gram(xp.asarray(w), accumulate_dtype=accumulate))
    np.testing.assert_array_equal(with_work, alloc)


# --------------------------------------------------------------------------- #
# The compiled compacted scatter (benchmarks/corpus/_trots_kernels.py)
# --------------------------------------------------------------------------- #
# The kernel is plain Python compiled by Numba when it is installed. CI has no
# Numba: there the plain-Python kernel pins the semantics on small blocks and
# the NumPy fallback is what runs; the compiled kernel's tests skip.


def _numba_kernel():
    pytest.importorskip("numba")
    trots._compiled_scatter.cache_clear()
    kernel = trots._compiled_scatter()
    assert kernel is not None
    return kernel


def _tile_product(rng, c, layout, dtype):
    # The two shapes the loader produces: a syrk product (F-ordered, lower
    # triangle only, upper exactly zero) and a gemm product (C-ordered, full).
    d = rng.standard_normal((5, c)).astype(dtype)
    if layout == "syrk":
        from scipy.linalg.blas import get_blas_funcs

        g = get_blas_funcs("syrk", dtype=dtype)(1.0, d.T, trans=0, lower=1)
        assert g.flags.f_contiguous
        return g
    w = rng.uniform(-1.0, 1.0, 5).astype(dtype)
    return np.ascontiguousarray(d.T @ (w[:, None] * d))


def _accumulator(rng, n, view):
    # A random accumulator, either C-contiguous or a leading sub-block view
    # of a wider array (the Gram's out[:nx, :nx] with auxiliary columns).
    base = rng.standard_normal((n + 3, n + 3) if view else (n, n))
    return (base[:n, :n] if view else base), base


def _special_values(acc, cols, g):
    # Signed zeros and a NaN in the product, onto signed-zero accumulator
    # entries: -0 + -0 = -0, -0 + +0 = +0, x + NaN = NaN (its payload).
    c = int(cols.shape[0])
    if c < 2:
        return
    g[1, 0], g[c - 1, c - 1], g[1, 1] = -0.0, 0.0, np.nan
    acc[cols[1], cols[0]] = -0.0
    acc[cols[c - 1], cols[c - 1]] = -0.0


def _lower_bits(a):
    return np.tril(np.asarray(a)).view(np.uint8)


@pytest.mark.parametrize("c", [1, 2, 9, 37])
@pytest.mark.parametrize("layout", ["syrk", "gemm"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("view", [False, True])
def test_scatter_kernel_is_bitwise_the_numpy_scatter_on_the_lower_triangle(
    c, layout, dtype, view
):
    # The plain-Python kernel (what Numba compiles): every lower-triangle
    # entry receives the NumPy scatter's single addition, a float32 product
    # widened exactly; nothing outside the block's lower triangle is
    # touched; row ranges split anywhere, run in any order, give the same
    # bits.
    rng = np.random.default_rng(100 + c)
    n = 50
    cols = np.sort(rng.choice(n, size=c, replace=False)).astype(np.int64)
    g = _tile_product(rng, c, layout, dtype)
    acc, base = _accumulator(rng, n, view)
    _special_values(acc, cols, g)
    ref = acc.copy()
    trots._scatter_lower(ref, cols, g)  # the NumPy scatter (no kernel)
    before, base_before = acc.copy(), base.copy()
    cuts = sorted({0, c, *rng.integers(0, c + 1, size=3).tolist()})
    for r0, r1 in reversed(list(itertools.pairwise(cuts))):
        kernels.scatter_lower_rows(acc, cols, g, r0, r1)

    np.testing.assert_array_equal(_lower_bits(acc), _lower_bits(ref))
    np.testing.assert_array_equal(np.triu(acc, 1), np.triu(before, 1))
    np.testing.assert_array_equal(base[n:], base_before[n:])
    np.testing.assert_array_equal(base[:, n:], base_before[:, n:])


@pytest.mark.parametrize("c", [40, 300, 1000])
@pytest.mark.parametrize("layout", ["syrk", "gemm"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("pooled", [False, True])
def test_compiled_scatter_is_bitwise_the_numpy_scatter(
    c, layout, dtype, pooled, monkeypatch
):
    kernel = _numba_kernel()
    monkeypatch.setattr(trots, "_SCATTER_POOL_MIN_ENTRIES", 1)
    rng = np.random.default_rng(200 + c)
    n = 1200
    cols = np.sort(rng.choice(n, size=c, replace=False)).astype(np.int64)
    g = _tile_product(rng, c, layout, dtype)
    acc, _ = _accumulator(rng, n, view=True)
    _special_values(acc, cols, g)
    ref = acc.copy()
    with trots._host_pool(trots.TiledGramConfig(scatter_threads=4)) as pool:
        use = pool if pooled else None
        trots._scatter_lower(ref, cols, g, use)
        trots._scatter_lower(acc, cols, g, use, kernel)

    np.testing.assert_array_equal(_lower_bits(acc), _lower_bits(ref))


def test_the_numpy_scatter_is_the_fallback_without_numba(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "numba", None)  # `import numba` fails
    trots._compiled_scatter.cache_clear()
    try:
        assert kernels.compiled_scatter_lower_rows() is None
        assert trots._compiled_scatter() is None
        assert trots._scatter_kernel(trots.TiledGramConfig()) is None
        seen = []
        real = trots._scatter_lower

        def spy(acc, cols, g, pool=None, kernel=None):
            seen.append(kernel)
            real(acc, cols, g, pool, kernel)

        monkeypatch.setattr(trots, "_scatter_lower", spy)
        a = _random_sparse(600, 50, 0.2, 60)
        w = np.random.default_rng(61).uniform(0.1, 2.0, a.shape[0])
        g = np.asarray(_op(a, block_rows=64, sparse_tile_flop_ratio=0.0).gram(w))
    finally:
        trots._compiled_scatter.cache_clear()  # later tests see the real import

    assert seen
    assert all(k is None for k in seen)
    assert _rel(g, _dense_gram(a, w)) < 1e-12


def test_a_numba_that_fails_to_compile_falls_back_with_a_warning(monkeypatch, caplog):
    import sys
    import types

    def njit(**_kw):
        def build(_fn):
            def broken(*_a):
                raise RuntimeError("llvmlite exploded")

            return broken

        return build

    monkeypatch.setitem(sys.modules, "numba", types.SimpleNamespace(njit=njit))
    with caplog.at_level("WARNING", logger=kernels.__name__):
        assert kernels.compiled_scatter_lower_rows() is None
    assert "NumPy" in caplog.text


def test_compiled_scatter_false_keeps_the_numpy_scatter(monkeypatch):
    def unwanted():
        raise AssertionError("compiled_scatter=False must not load the kernel")

    monkeypatch.setattr(trots, "_compiled_scatter", unwanted)
    a = _random_sparse(600, 50, 0.2, 62)
    w = np.random.default_rng(63).uniform(0.1, 2.0, a.shape[0])
    op = _op(a, block_rows=64, sparse_tile_flop_ratio=0.0, compiled_scatter=False)
    assert _rel(np.asarray(op.gram(w)), _dense_gram(a, w)) < 1e-12


@pytest.mark.parametrize("source", ["python", "numba"])
@pytest.mark.parametrize("weights", ["positive", "signed"])
@pytest.mark.parametrize("lanes", [1, 2])
@pytest.mark.parametrize("accumulate", [None, "float32"])
def test_gram_sha256_is_the_same_with_and_without_the_scatter_kernel(
    source, weights, lanes, accumulate, monkeypatch
):
    import hashlib

    kernel = _numba_kernel() if source == "numba" else kernels.scatter_lower_rows
    monkeypatch.setattr(trots, "_SCATTER_POOL_MIN_ENTRIES", 1)  # pool every tile
    a = _random_sparse(900, 70, 0.15, 64)
    a.data = a.data.astype(np.float32).astype(np.float64)
    rng = np.random.default_rng(65)
    w = rng.uniform(0.1 if weights == "positive" else -1.0, 2.0, a.shape[0])
    cfg = {
        "block_rows": 64,
        "threads": lanes,
        "lane_min_flops": 1.0,
        "sparse_tile_flop_ratio": 0.0,
        "scatter_threads": 3,
    }

    def digest(compiled):
        op = _op(a, compiled_scatter=compiled, **cfg)
        g = np.asarray(op.gram(w, accumulate_dtype=accumulate))
        return hashlib.sha256(np.ascontiguousarray(g).tobytes()).hexdigest()

    old = digest(False)
    monkeypatch.setattr(trots, "_compiled_scatter", lambda: kernel)
    assert digest(True) == old


@pytest.mark.parametrize("scatter_threads", [1, 4])
def test_the_scatter_kernel_runs_on_the_scatter_threads_pool(
    scatter_threads, monkeypatch
):
    # One thread knob: the kernel runs in the pool's equal-area row chunks
    # (TiledGramConfig.scatter_threads workers, 4 chunks each) off the
    # calling thread, or as a single [0, c) call on it without a pool.
    # Numba's own threading layer is never used.
    import threading

    if (os.cpu_count() or 1) < scatter_threads:
        pytest.skip("needs a pool of scatter_threads workers")
    calls = []

    def spy(acc, cols, g, r0, r1):
        calls.append((threading.get_ident(), int(cols.shape[0]), r0, r1))
        kernels.scatter_lower_rows(acc, cols, g, r0, r1)

    monkeypatch.setattr(trots, "_SCATTER_POOL_MIN_ENTRIES", 1)
    monkeypatch.setattr(trots, "_compiled_scatter", lambda: spy)
    a = _random_sparse(600, 50, 0.2, 66)
    w = np.random.default_rng(67).uniform(0.1, 2.0, a.shape[0])
    op = _op(
        a,
        block_rows=64,
        threads=1,
        sparse_tile_flop_ratio=0.0,
        scatter_threads=scatter_threads,
    )
    g = np.asarray(op.gram(w))

    assert _rel(g, _dense_gram(a, w)) < 1e-12
    main = threading.get_ident()
    tiles = [t for p in op._plans for t in p.tiles]
    i = 0
    for t in tiles:  # single lane: the tiles in order, each one's chunks
        c = int(t.cols.shape[0])
        if scatter_threads > 1:
            want = trots._lower_row_bounds(c, scatter_threads * 4)
        else:
            want = [0, c]
        part = calls[i : i + len(want) - 1]
        i += len(want) - 1
        assert sorted((r0, r1) for _t, _c, r0, r1 in part) == list(
            itertools.pairwise(want)
        )
        assert all(cc == c for _t, cc, _r0, _r1 in part)
        assert all((tid == main) == (scatter_threads == 1) for tid, *_ in part)
    assert i == len(calls) > 0


@pytest.mark.parametrize("compiled", [True, False])
def test_host_operator_warms_the_kernel_up_at_construction(compiled, monkeypatch):
    # The Numba import and compilation (or cache load) happen when the
    # operator is built, not inside the first Gram of a timed iteration.
    made = []
    monkeypatch.setattr(trots, "_compiled_scatter", lambda: made.append(1))
    _op(_random_sparse(600, 50, 0.2, 68), block_rows=64, compiled_scatter=compiled)
    assert made == ([1] if compiled else [])
