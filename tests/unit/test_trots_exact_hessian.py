"""The TROTS loader's grouped exact Lagrangian Hessian (``TROTSExactHessian``).

Every TROTS cost function is, in dose space, a diagonal plus at most a rank-1
term (van Haveren & Breedveld 2019, eqs. (8)–(9)), so the beamlet-space
Hessian is one product per unique structure with the diagonals of all its
terms summed (Breedveld et al. 2017, eq. (48)). The reference everywhere is
the original per-term path (``grouped_hessian=False``: one SciPy SpGEMM per
term). Also covers the cost functions' singular-point guards.
"""

from __future__ import annotations

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")
trots = pytest.importorskip("benchmarks.corpus.trots")


def _banded(m, n, width, seed):
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i in rng.permutation(m):
        centre = int(i * n / m)
        for j in range(max(0, centre - width), min(n, centre + width + 1)):
            if rng.random() < 0.7:
                rows.append(i)
                cols.append(j)
    vals = rng.uniform(0.1, 1.0, len(rows))
    return sp.csc_matrix((vals, (rows, cols)), shape=(m, n))


def _dense_ish(m, n, seed):
    rng = np.random.default_rng(seed)
    a = sp.random(m, n, density=0.9, format="csr", random_state=rng)
    a.data = rng.uniform(0.05, 0.5, a.nnz).astype(np.float32).astype(np.float64)
    return a


N = 40


def _instance():
    """Structures shared between objective, nonlinear constraints and the
    linear block; every curved cost type; a beamlet-space quadratic."""
    ptv = _banded(900, N, 3, 1)  # tiled (sparse, local)
    oar = _banded(700, N, 4, 2)  # tiled; also a linear constraint
    body = _dense_ish(300, N, 3)  # gated off: dense row chunks
    rng = np.random.default_rng(4)
    q = sp.random(N, N, density=0.1, format="csr", random_state=rng)
    q = (q @ q.T + sp.identity(N)).tocsr()
    cache = {
        1: trots.TROTSMatrix("ptv", 0, ptv, np.zeros(0), 0.0, "float64"),
        2: trots.TROTSMatrix("oar", 0, oar, np.zeros(0), 0.0, "float64"),
        3: trots.TROTSMatrix("body", 0, body, np.zeros(0), 0.0, "float32"),
        4: trots.TROTSMatrix("smooth", 2, q, np.zeros(0), 0.0, "float64"),
    }

    def entry(name, is_con, minimise, ctype, data_id, params, bound, weight=1.0):
        return trots.TROTSEntry(
            name=name,
            is_constraint=is_con,
            minimise=minimise,
            ctype=ctype,
            data_id=data_id,
            params=None if params is None else np.asarray(params, dtype=float),
            bound=bound,
            weight=weight,
            priority=1,
            active=True,
        )

    entries = [
        entry("ptv ltcp", False, True, 4, 1, [2.0, 0.8], 0.0, 0.5),
        entry("ptv geud", False, False, 3, 1, [-8.0], 0.0, 0.3),  # maximise, a<0
        entry("oar geud", False, True, 3, 2, [6.0], 0.0, 0.2),  # rank-1, a>1
        entry("oar mean", False, True, 3, 2, [1.0], 0.0, 0.4),  # a=1: no curvature
        entry("body dvh", False, True, 5, 3, [1.5, 6.0], 0.0, 0.1),
        entry("smooth", False, True, 2, 4, None, 0.0, 1e-3),
        entry("oar dvh con", True, True, 5, 2, [1.0, 5.0], 0.8),
        entry("ptv geud con", True, True, 3, 1, [4.0], 5.0),
        entry("body ltcp con", True, False, 4, 3, [1.0, 0.5], 0.2),
        entry("smooth con", True, True, 2, 4, None, 50.0),
        entry("oar max", True, True, 1, 2, None, 6.0),  # linear block
    ]
    inst = trots.TROTSInstance(
        identifier="synthetic-exact", n=N, real=N, entries=entries, solution=None
    )
    inst._matrix_cache = cache
    return inst


CFG = trots.TiledGramConfig(block_rows=128)


def _problems(xp=np, cfg=CFG):
    inst = _instance()
    grouped = trots.TROTSExactProblem(inst, xp, tiled_gram=cfg)
    per_term = trots.TROTSExactProblem(inst, xp, tiled_gram=cfg, grouped_hessian=False)
    return grouped, per_term


def _point(problem, seed=5):
    rng = np.random.default_rng(seed)
    z = np.concatenate([rng.uniform(0.2, 2.0, N), np.zeros(problem._n_aux)])
    y = rng.uniform(-1.0, 3.0, problem._n_nl_ineq)
    return z, y


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def test_grouped_dense_matrix_matches_the_per_term_hessian():
    grouped, per_term = _problems()
    z, y = _point(grouped)
    h_g = grouped.lagrangian_hessian(z, np.zeros(0), y, 0.7)
    h_p = per_term.lagrangian_hessian(z, np.zeros(0), y, 0.7)
    assert isinstance(h_g, trots.TROTSExactHessian)
    dg, dp = np.asarray(h_g.dense_matrix()), np.asarray(h_p.dense_matrix())
    assert dg.shape == dp.shape == (grouped.n_vars, grouped.n_vars)
    assert _rel(dg, dp) < 1e-12
    np.testing.assert_array_equal(dg, dg.T)  # exactly symmetric
    # One product per unique curved structure (3), not one per term (7).
    assert len(h_g.terms) == 3
    tiled = {cs.mat.name: cs.tiles is not None for cs, _w in h_g.terms}
    assert tiled == {"ptv": True, "oar": True, "body": False}


def test_matvec_matmat_diagonal_and_coo_agree_with_the_dense_matrix():
    grouped, _ = _problems()
    z, y = _point(grouped, 6)
    h = grouped.lagrangian_hessian(z, np.zeros(0), y, 1.0)
    dense = np.asarray(h.dense_matrix())
    rng = np.random.default_rng(7)
    v = rng.standard_normal(grouped.n_vars)
    V = rng.standard_normal((grouped.n_vars, 3))
    np.testing.assert_allclose(
        np.asarray(h.matvec(v)), dense @ v, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(h.rmatvec(v)), dense @ v, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(h.matmat(V)), dense @ V, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(h.diagonal()), np.diag(dense), rtol=1e-12, atol=1e-12
    )
    sq = [cs.sq_t for cs, _w in h.terms]  # (A∘A)ᵀ cached per structure
    assert all(s is not None for s in sq)
    h.diagonal()
    assert all(cs.sq_t is s for (cs, _w), s in zip(h.terms, sq, strict=True))
    r, c, vals, shape = h.to_coo()
    coo = sp.coo_matrix(
        (np.asarray(vals), (np.asarray(r), np.asarray(c))), shape=shape
    ).toarray()
    assert _rel(coo, dense) < 1e-12
    assert h.symmetry_hint() is True


def test_rank_one_part_holds_exactly_the_curved_geud_terms():
    grouped, _ = _problems()
    z, y = _point(grouped, 8)
    y[1] = 0.0  # "ptv geud con" inactive: its rank-1 term drops out
    h = grouped.lagrangian_hessian(z, np.zeros(0), y, 1.0)
    # ptv geud (a=-8) + oar geud (a=6); the a=1 mean has c2 = 0.
    assert h.low_rank is not None
    assert h.low_rank.shape == (N, N)
    assert h.low_rank_factors[0].shape == h.low_rank_factors[1].shape == (N, 2)
    # A maximised gEUD with a>1 would flip sign; a=-8 maximised: c2 > 0 · (−w).
    assert np.linalg.matrix_rank(np.asarray(h.low_rank.dense_matrix())) == 2


def test_negative_merged_weights_take_the_signed_fallback(monkeypatch):
    calls = []
    real = trots._tile_gram_into

    def spy(acc, tile, wp, swp, bad, acc_dtype, syrk):
        calls.append(bad is not None and bool(bad[tile.start : tile.stop].any()))
        return real(acc, tile, wp, swp, bad, acc_dtype, syrk)

    monkeypatch.setattr(trots, "_tile_gram_into", spy)
    grouped, per_term = _problems()
    z, y = _point(grouped, 9)
    h = grouped.lagrangian_hessian(z, np.zeros(0), y, 1.0)
    w = {cs.mat.name: wk for cs, wk in h.terms}
    assert np.any(w["body"] < 0) or np.any(w["oar"] < 0)  # DVH is indefinite
    dense = np.asarray(h.dense_matrix())
    assert any(calls), "no tile took the signed (gemm) path"
    ref = np.asarray(per_term.lagrangian_hessian(z, np.zeros(0), y, 1.0).dense_matrix())
    assert _rel(dense, ref) < 1e-12


def test_structures_shared_with_the_linear_block_hold_one_tile_copy():
    grouped, _ = _problems()
    op = grouped.linear_ineq()[0]
    blocks = op.blocks if hasattr(op, "blocks") else (op,)
    lin = [b for b in blocks if isinstance(b, trots.TiledGramCSROperator)]
    assert lin, "the oar linear block should be tiled"
    oar = next(cs for cs in grouped._curved.values() if cs.mat.name == "oar")
    lin_tiles = [p.tiles for b in lin for p in b._plans]
    assert any(t is oar.tiles for t in lin_tiles)


def test_dense_matrix_is_memoized_per_operator():
    grouped, _ = _problems()
    z, y = _point(grouped, 10)
    h = grouped.lagrangian_hessian(z, np.zeros(0), y, 1.0)
    h.dense_matrix()
    h.dense_matrix()
    assert grouped.hessian_stats["dense_calls"] == 1


def test_untiled_config_uses_dense_chunks_everywhere():
    inst = _instance()
    plain = trots.TROTSExactProblem(inst, np, tiled_gram=False)
    ref = trots.TROTSExactProblem(inst, np, grouped_hessian=False)
    z, y = _point(plain, 11)
    h = plain.lagrangian_hessian(z, np.zeros(0), y, 1.0)
    assert all(cs.tiles is None for cs, _w in h.terms)
    dense = np.asarray(h.dense_matrix())
    assert (
        _rel(
            dense,
            np.asarray(ref.lagrangian_hessian(z, np.zeros(0), y, 1.0).dense_matrix()),
        )
        < 1e-12
    )


def test_grouped_hessian_on_the_namespace_backends(namespace):
    # NumPy: the host-resident grouped operator itself. CuPy / Torch: the
    # device operator — tiled structures through the device tiled kernel,
    # sharing the problem's device tiles. Any other namespace with a sparse
    # adapter (JAX): the grouped Hessian crossed once as the namespace's
    # CSCOperator. No adapter (array_api_strict): the host operator. The NumPy
    # per-term Hessian is the reference either way.
    from ipax.backend.operators import CSCOperator
    from ipax.backend.sparse import get_sparse_adapter

    grouped, _ = _problems(namespace)
    _, per_term = _problems()
    z, y = _point(grouped, 12)
    zero = namespace.asarray(np.zeros(0))
    h = grouped.lagrangian_hessian(
        namespace.asarray(z), zero, namespace.asarray(y), 1.0
    )
    host = isinstance(namespace.asarray(0.0), np.ndarray) or (
        get_sparse_adapter(namespace) is None
    )
    device = not host and trots._device_kind(namespace) is not None
    if host:
        assert isinstance(h, trots.TROTSExactHessian)
    elif device:
        assert isinstance(h, trots.TROTSDeviceExactHessian)
        tiled = {cs.mat.name: cs.tiles is not None for cs, _w in h.terms}
        assert tiled == {"ptv": True, "oar": True, "body": False}
        assert len(grouped._device_tiles) == 0  # materialized on first use
    else:
        assert isinstance(h, CSCOperator)
        assert not grouped._plan_registry  # no host tile plans for the Hessian
        assert all(cs.tiles is None for cs in grouped._curved.values())
    ref = per_term.lagrangian_hessian(z, np.zeros(0), y, 1.0)
    # The COO triplets are Array-API on every namespace (array_api_strict has
    # no sparse adapter for dense_matrix()).
    r, c, vals, shape = h.to_coo()
    r, c, vals = (trots._to_numpy_nd(t) for t in (r, c, vals))
    dense = sp.coo_matrix((vals, (r, c)), shape=shape).toarray()
    assert _rel(dense, np.asarray(ref.dense_matrix())) < 1e-12
    if host:
        assert _rel(np.asarray(h.dense_matrix()), dense) < 1e-12
    if device:
        d = trots._to_numpy_nd(h.dense_matrix())
        assert type(h.dense_matrix()) is type(namespace.asarray(0.0))
        assert _rel(d, dense) < 1e-12
        np.testing.assert_array_equal(d, d.T)  # exactly symmetric
        assert len(grouped._device_tiles) == 2  # ptv + oar, once each
        np.testing.assert_allclose(
            trots._to_numpy_nd(h.diagonal()), np.diag(dense), rtol=1e-12, atol=1e-12
        )
        V = np.random.default_rng(14).standard_normal((grouped.n_vars, 3))
        np.testing.assert_allclose(
            trots._to_numpy_nd(h.matmat(namespace.asarray(V))),
            dense @ V,
            rtol=1e-12,
            atol=1e-12,
        )
    v = np.random.default_rng(13).standard_normal(grouped.n_vars)
    hv = h.matvec(namespace.asarray(v))
    np.testing.assert_allclose(
        trots._to_numpy_nd(hv), dense @ v, rtol=1e-12, atol=1e-12
    )


def test_device_hessian_shares_the_linear_block_tiles(namespace):
    # "oar" is both curved and a linear constraint: one plan (registry) and
    # one device copy (device tile cache) serve the linear block's Gram and
    # the Hessian's product.
    kind = trots._device_kind(namespace)
    if kind is None or trots._is_host_numpy(namespace):
        pytest.skip("device Hessian runs on CuPy / Torch namespaces only")
    from ipax.problem.linear_ineq import lower_linear_inequalities

    grouped, _ = _problems(namespace)
    z, y = _point(grouped, 15)
    jac = lower_linear_inequalities(grouped, namespace.asarray(z), namespace)._jac
    w = np.random.default_rng(16).uniform(0.1, 2.0, jac.shape[0])
    jac.gram(namespace.asarray(w))
    after_gram = {key[0] for key in grouped._device_tiles}  # id(host tiles)
    h = grouped.lagrangian_hessian(
        namespace.asarray(z), namespace.asarray(np.zeros(0)), namespace.asarray(y)
    )
    h.dense_matrix()
    oar = next(cs for cs in grouped._curved.values() if cs.mat.name == "oar")
    assert id(oar.tiles) in after_gram  # the linear block materialized it
    assert {key[0] for key in grouped._device_tiles} == after_gram | {
        id(cs.tiles) for cs, _w in h.terms if cs.tiles is not None
    }
    assert len(grouped._device_tiles) == 2  # ptv (Hessian) + oar (shared)


def test_untiled_device_hessian_keeps_the_csc_crossing(namespace):
    # tiled_gram=False on a device namespace: no plans, no device tiles —
    # the grouped Hessian crosses as a CSCOperator, exactly the old path.
    from ipax.backend.operators import CSCOperator

    kind = trots._device_kind(namespace)
    if kind is None or trots._is_host_numpy(namespace):
        pytest.skip("device Hessian runs on CuPy / Torch namespaces only")
    plain = trots.TROTSExactProblem(_instance(), namespace, tiled_gram=False)
    _, per_term = _problems()
    z, y = _point(plain, 17)
    h = plain.lagrangian_hessian(
        namespace.asarray(z), namespace.asarray(np.zeros(0)), namespace.asarray(y)
    )
    assert isinstance(h, CSCOperator)
    assert all(cs.tiles is None for cs in plain._curved.values())
    ref = np.asarray(per_term.lagrangian_hessian(z, np.zeros(0), y).dense_matrix())
    assert _rel(trots._to_numpy_nd(h.dense_matrix()), ref) < 1e-12


def test_device_hessian_solves_like_the_csc_crossing(namespace):
    from ipax import Options, solve

    kind = trots._device_kind(namespace)
    if kind is None or trots._is_host_numpy(namespace):
        pytest.skip("device Hessian runs on CuPy / Torch namespaces only")
    opts = Options(hessian="exact", linsolve="dense", max_iter=200)
    results = []
    for tiled in (CFG, False):
        problem = trots.TROTSExactProblem(_instance(), namespace, tiled_gram=tiled)
        results.append(
            solve(problem, problem.initial_point(warm_start=False), options=opts)
        )
    r_t, r_p = results
    assert r_t.status == r_p.status
    assert r_t.routes == r_p.routes
    assert abs(r_t.n_iter - r_p.n_iter) <= 2  # round-off level only
    assert abs(r_t.objective - r_p.objective) <= 1e-8 * max(1.0, abs(r_p.objective))


@pytest.mark.parametrize("linsolve", ["dense", "krylov"])
def test_grouped_and_per_term_hessians_solve_alike(linsolve):
    from ipax import Options, solve

    grouped, per_term = _problems()
    opts = Options(hessian="exact", linsolve=linsolve, max_iter=200)
    results = [
        solve(p, p.initial_point(warm_start=False), options=opts)
        for p in (grouped, per_term)
    ]
    r_g, r_p = results
    assert r_g.status == r_p.status
    assert r_g.routes == r_p.routes
    assert abs(r_g.n_iter - r_p.n_iter) <= 2  # round-off level only
    assert abs(r_g.objective - r_p.objective) <= 1e-8 * max(1.0, abs(r_p.objective))


# --------------------------------------------------------------------------- #
# Singular-point guards of the cost functions
# --------------------------------------------------------------------------- #
def _unguarded_geud_hess(d, a):
    m = d.size
    s = float(np.mean(np.abs(d) ** a))
    diag = s ** (1 / a - 1) * (a - 1) * np.abs(d) ** (a - 2) / m
    u = np.abs(d) ** (a - 1) * np.sign(d) / m
    return diag, u, (1 / a - 1) * a * s ** (1 / a - 2)


@pytest.mark.parametrize("a", [-30.0, -8.0, 1.0, 1.5, 4.0])
def test_geud_guards_leave_normal_doses_bitwise_unchanged(a):
    d = np.random.default_rng(14).uniform(0.1, 70.0, 500)
    par = np.array([a])
    diag, u, coef = trots._cost_hess_d(3, d, par)
    ref = _unguarded_geud_hess(d, a)
    np.testing.assert_array_equal(diag, ref[0])
    np.testing.assert_array_equal(u, ref[1])
    assert coef == ref[2]
    s = np.mean(d**a)
    np.testing.assert_array_equal(
        trots._cost_grad_d(3, d, par), s ** (1 / a - 1) * d ** (a - 1) / d.size
    )


@pytest.mark.parametrize("a", [-30.0, -8.0, 1.0, 1.5])
def test_geud_hessian_is_finite_at_zero_dose(a):
    d = np.array([0.0, 1.0, 2.0, 0.0])
    with np.errstate(all="raise"):
        diag, u, coef = trots._cost_hess_d(3, d, np.array([a]))
        g = trots._cost_grad_d(3, d, np.array([a]))
        v = trots._cost_value(3, d, np.array([a]))
    assert np.all(np.isfinite(diag)) and np.all(np.isfinite(u))
    assert np.isfinite(coef) and np.all(np.isfinite(g)) and np.isfinite(v)
    if a == 1.0:  # the mean has no curvature anywhere, including d = 0
        np.testing.assert_array_equal(diag, 0.0)
        assert coef == 0.0
    if a < 0.0:  # a cold voxel: S = inf, gEUD 0, the closed forms vanish
        assert v == 0.0
        np.testing.assert_array_equal(g, 0.0)
        np.testing.assert_array_equal(diag, 0.0)
        np.testing.assert_array_equal(u, 0.0)
        assert coef == 0.0


@pytest.mark.parametrize(("a", "d0"), [(-24.0, 5e-13), (-30.0, 3e-11), (-60.0, 1e-3)])
def test_geud_negative_exponent_floor_scales_with_the_exponent(a, d0):
    # 1e-12^(a-2) overflows float64 for a < ≈ -23.7. (-24, 5e-13): S is finite
    # but the voxel sits below the floor, whose fixed-1e-12 power was inf.
    d = np.array([d0, 1.0, 2.0, 1e-3])
    diag, u, coef = trots._cost_hess_d(3, d, np.array([a]))
    g = trots._cost_grad_d(3, d, np.array([a]))
    assert np.all(np.isfinite(diag)) and np.all(np.isfinite(u))
    assert np.isfinite(coef) and np.all(np.isfinite(g))


def test_geud_all_zero_structure_has_zero_gradient_and_curvature():
    d = np.zeros(6)
    par = np.array([4.0])
    g = trots._cost_grad_d(3, d, par)  # was: ZeroDivisionError (0.0 ** -0.75)
    diag, u, coef = trots._cost_hess_d(3, d, par)
    np.testing.assert_array_equal(g, 0.0)
    np.testing.assert_array_equal(diag, 0.0)
    np.testing.assert_array_equal(u, 0.0)
    assert coef == 0.0
    assert trots._cost_value(3, d, par) == 0.0


def test_ltcp_exponent_is_clipped_without_touching_normal_values():
    par = np.array([60.0, 0.8])
    d = np.random.default_rng(15).uniform(0.0, 80.0, 300)
    ref = np.exp(-0.8 * (d - 60.0))
    assert trots._cost_value(4, d, par) == float(np.mean(ref))
    np.testing.assert_array_equal(trots._cost_grad_d(4, d, par), (-0.8 / d.size) * ref)
    np.testing.assert_array_equal(
        trots._cost_hess_d(4, d, par)[0], (0.8 * 0.8 / d.size) * ref
    )
    far = np.array([-2000.0, 0.0])  # exponent 1648: overflowed to inf before
    with np.errstate(over="raise"):
        v = trots._cost_value(4, far, par)
        h = trots._cost_hess_d(4, far, par)[0]
    assert np.isfinite(v) and np.all(np.isfinite(h))


# --------------------------------------------------------------------------- #
# Share-gate fold: one tile set for a folded float32 structure
# --------------------------------------------------------------------------- #
def _folded_instance():
    """A float32 structure that is curved (LTCP objective) *and* a linear
    constraint, holding ~2% of the linear block's nonzeros: the share gate
    folds it into the float64 group."""
    big = _banded(6000, N, 3, 21)
    small = _banded(300, N, 2, 22)
    small.data = small.data.astype(np.float32).astype(np.float64)
    cache = {
        1: trots.TROTSMatrix("big", 0, big, np.zeros(0), 0.0, "float64"),
        2: trots.TROTSMatrix("small", 0, small, np.zeros(0), 0.0, "float32"),
    }

    def entry(name, is_con, ctype, data_id, params, bound):
        return trots.TROTSEntry(
            name=name,
            is_constraint=is_con,
            minimise=True,
            ctype=ctype,
            data_id=data_id,
            params=None if params is None else np.asarray(params, dtype=float),
            bound=bound,
            weight=1.0,
            priority=1,
            active=True,
        )

    entries = [
        entry("big max", True, 1, 1, None, 50.0),
        entry("small max", True, 1, 2, None, 30.0),
        entry("small ltcp", False, 4, 2, [2.0, 0.8], 0.0),
    ]
    inst = trots.TROTSInstance(
        identifier="synthetic-fold", n=N, real=N, entries=entries, solution=None
    )
    inst._matrix_cache = cache
    return inst


def test_folded_float32_structure_shares_one_tile_set(namespace):
    # Folded, the linear block plans "small" without a hint (key <f8); the
    # Hessian must plan it the same way, or the registry holds two tile sets
    # (and a device namespace two device copies) of one structure.
    xp = namespace
    if not trots._is_host_numpy(xp) and trots._device_kind(xp) is None:
        pytest.skip("tile plans run on NumPy / CuPy / Torch namespaces only")
    cfg = trots.TiledGramConfig(block_rows=128, flop_ratio_threshold=1.0)
    problem = trots.TROTSExactProblem(_folded_instance(), xp, tiled_gram=cfg)
    assert problem.float32_group_folded
    assert 0.0 < problem.float32_nnz_share < 0.05
    assert [h for _c, h in problem._lin_groups] == [None]
    small = next(cs for cs in problem._curved.values() if cs.mat.name == "small")
    assert small.tiles is not None

    op = problem.linear_ineq()[0]
    assert isinstance(op, trots.TiledGramCSROperator)
    keys = [k for k in problem._plan_registry if k[0] == id(small.mat)]
    assert len(keys) == 1
    assert any(p.tiles is small.tiles for p in op._plans)
    if trots._device_kind(xp) is not None and not trots._is_host_numpy(xp):
        from ipax.problem.linear_ineq import lower_linear_inequalities

        rng = np.random.default_rng(23)
        z = xp.asarray(rng.uniform(0.2, 2.0, problem.n_vars))
        jac = lower_linear_inequalities(problem, z, xp)._jac
        jac.gram(xp.asarray(rng.uniform(0.1, 2.0, jac.shape[0])))
        h = problem.lagrangian_hessian(
            z, xp.asarray(np.zeros(0)), xp.asarray(np.zeros(0))
        )
        assert isinstance(h, trots.TROTSDeviceExactHessian)
        h.dense_matrix()
        assert len(problem._device_tiles) == 2  # big + small, once each


def test_device_hessian_host_bridges_go_through_the_context(namespace, monkeypatch):
    # The Torch matvec is a host round trip: it must download through the
    # device context (which handles CUDA tensors), never the NumPy-only
    # bridge; diagonal is the host form, uploaded once.
    xp = namespace
    if trots._device_kind(xp) is None or trots._is_host_numpy(xp):
        pytest.skip("device Hessian runs on CuPy / Torch namespaces only")
    grouped, per_term = _problems(xp)[0], _problems()[1]
    z, y = _point(grouped, 18)
    h = grouped.lagrangian_hessian(
        xp.asarray(z), xp.asarray(np.zeros(0)), xp.asarray(y)
    )
    assert isinstance(h, trots.TROTSDeviceExactHessian)

    def boom(*_args, **_kwargs):
        raise AssertionError("NumPy-only host bridge used on a device array")

    monkeypatch.setattr(trots, "_to_numpy_nd", boom)
    v = np.random.default_rng(19).standard_normal(grouped.n_vars)
    hv, diag = h.matvec(xp.asarray(v)), h.diagonal()
    monkeypatch.undo()
    assert type(hv) is type(xp.asarray(0.0))
    ref = np.asarray(per_term.lagrangian_hessian(z, np.zeros(0), y).dense_matrix())
    np.testing.assert_allclose(trots._to_numpy_nd(hv), ref @ v, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        trots._to_numpy_nd(diag), np.diag(ref), rtol=1e-12, atol=1e-12
    )
