"""The TROTS loader's fused exact Hessian + linear-block Gram (one product).

The exact Hessian forms ``A_sᵀ diag(h_s) A_s`` per curved structure (van
Haveren & Breedveld 2019, eqs. (8)–(9)) and the linear block's tiled Gram
``A_sᵀ diag(σ_s) A_s`` per constrained structure; a structure in both was
multiplied twice per iteration. Through the core's optional
``LinearOperator.dense_matrix_with_gram`` hook the Hessian now multiplies it
once with ``h_s + σ_s`` (Breedveld et al. 2017, eq. (48) and §4.2). The
reference is the two-pass block ``H + Jᵀ diag(w) J`` with the merge off
(``TiledGramConfig.merge_hessian_gram=False``).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")
trots = pytest.importorskip("benchmarks.corpus.trots")

from ipax.problem.linear_ineq import lower_linear_inequalities  # noqa: E402
from tests.unit.test_trots_exact_hessian import (  # noqa: E402
    CFG,
    _instance,
    _point,
    _rel,
)

NO_MERGE = dataclasses.replace(CFG, merge_hessian_gram=False)


def _setup(xp, cfg=CFG, seed=31, sparse_weights=False):
    """Problem, its Hessian and the solver's inequality Jacobian at a point."""
    problem = trots.TROTSExactProblem(_instance(), xp, tiled_gram=cfg)
    z, y = _point(problem, seed)
    zx = xp.asarray(z)
    lowered = lower_linear_inequalities(problem, zx, xp)
    jac = lowered.ineq_jacobian(zx)
    h = problem.lagrangian_hessian(zx, xp.asarray(np.zeros(0)), xp.asarray(y), 0.7)
    rng = np.random.default_rng(seed + 1)
    w = rng.uniform(0.1, 3.0, jac.shape[0])
    if sparse_weights:  # restoration-like: a few active rows
        w[rng.random(jac.shape[0]) < 0.9] = 0.0
    return problem, h, jac, xp.asarray(w)


def _two_pass(xp, w_host, seed=31):
    _, h, jac, _ = _setup(xp, NO_MERGE, seed)
    w = xp.asarray(w_host)
    return trots._to_numpy_nd(h.dense_matrix()) + trots._to_numpy_nd(jac.gram(w))


def _host_or_device(xp):
    if not trots._is_host_numpy(xp) and trots._device_kind(xp) is None:
        pytest.skip("tiled plans run on NumPy / CuPy / Torch namespaces only")


@pytest.mark.parametrize("sparse_weights", [False, True])
def test_fused_block_matches_the_two_pass_block(namespace, sparse_weights):
    xp = namespace
    _host_or_device(xp)
    problem, h, jac, w = _setup(xp, sparse_weights=sparse_weights)
    fused = h.dense_matrix_with_gram(jac, w)
    assert type(fused) is type(xp.asarray(0.0))
    got = trots._to_numpy_nd(fused)
    ref = _two_pass(xp, trots._to_numpy_nd(w))
    assert _rel(got, ref) < 1e-12
    np.testing.assert_array_equal(got, got.T)  # exactly symmetric
    # "oar" is curved and in the linear block: multiplied once, merged.
    assert [cs.mat.name for cs in h.fused_structures] == ["oar"]
    assert problem.hessian_stats["fused_calls"] == 1


def test_merge_off_or_nothing_shared_declines(namespace):
    # Declining (NotImplementedError) is what keeps the core's two-pass
    # assembly — and every case without a shared structure — bit for bit.
    xp = namespace
    _host_or_device(xp)
    _, h, jac, w = _setup(xp, NO_MERGE)
    with pytest.raises(NotImplementedError):
        h.dense_matrix_with_gram(jac, w)
    # No structure shared: "oar max" (the only linear row group) removed.
    inst = _instance()
    inst.entries = [e for e in inst.entries if e.name != "oar max"]
    problem = trots.TROTSExactProblem(inst, xp, tiled_gram=CFG)
    z, y = _point(problem, 5)
    zx = xp.asarray(z)
    jac2 = lower_linear_inequalities(problem, zx, xp).ineq_jacobian(zx)
    h2 = problem.lagrangian_hessian(zx, xp.asarray(np.zeros(0)), xp.asarray(y))
    with pytest.raises(NotImplementedError):
        h2.dense_matrix_with_gram(jac2, xp.asarray(np.ones(jac2.shape[0])))


def test_linear_block_gram_is_unchanged_after_a_fused_call(namespace):
    # The merged weights must never reach the linear block's memo: a plain
    # Gram request afterwards is the pure Jᵀ diag(w) J, bit for bit.
    xp = namespace
    _host_or_device(xp)
    _, h, jac, w = _setup(xp)
    _, _, jac_ref, _ = _setup(xp, NO_MERGE)
    h.dense_matrix_with_gram(jac, w)
    after = trots._to_numpy_nd(jac.gram(w))
    np.testing.assert_array_equal(after, trots._to_numpy_nd(jac_ref.gram(w)))


def test_fused_block_is_memoized_per_weights(namespace, monkeypatch):
    # δ_w retries rebuild the condensed operator around the same Hessian and
    # Σ_s: the fused block is formed once per weight vector.
    xp = namespace
    _host_or_device(xp)
    problem, h, jac, w = _setup(xp)
    first = h.dense_matrix_with_gram(jac, w)
    second = h.dense_matrix_with_gram(jac, xp.asarray(trots._to_numpy_nd(w)))
    assert second is first
    assert problem.hessian_stats["fused_calls"] == 1
    w2 = xp.asarray(trots._to_numpy_nd(w) * 1.5)
    third = h.dense_matrix_with_gram(jac, w2)
    assert problem.hessian_stats["fused_calls"] == 2
    ref = _two_pass(xp, trots._to_numpy_nd(w2))
    assert _rel(trots._to_numpy_nd(third), ref) < 1e-12


@pytest.mark.parametrize("hinted_only", [False, True])
def test_reduced_requests_keep_the_hessian_in_float64(namespace, hinted_only):
    # The linear block here is float64-sourced (no float32 hint). An explicit
    # float32 request reduces it, so nothing may merge into it (the Hessian
    # part would be rounded): the hook declines. Under "auto" (hinted_only)
    # the block stays exact and merges.
    xp = namespace
    _host_or_device(xp)
    _, h, jac, w = _setup(xp)
    if not hinted_only:
        with pytest.raises(NotImplementedError):
            h.dense_matrix_with_gram(
                jac, w, accumulate_dtype="float32", hinted_only=False
            )
        return
    fused = h.dense_matrix_with_gram(
        jac, w, accumulate_dtype="float32", hinted_only=True
    )
    ref = _two_pass(xp, trots._to_numpy_nd(w))
    assert _rel(trots._to_numpy_nd(fused), ref) < 1e-12
    assert [cs.mat.name for cs in h.fused_structures] == ["oar"]


def test_float32_hinted_block_does_not_merge_under_a_reduced_request():
    # A float32-sourced, float32-exact linear group reduces under "auto":
    # its shared structure stays in the Hessian's own float64 product.
    inst = _instance()
    oar = inst._matrix_cache[2]
    oar.matrix.data = oar.matrix.data.astype(np.float32).astype(np.float64)
    inst._matrix_cache[2] = dataclasses.replace(oar, source_dtype="float32")
    problem = trots.TROTSExactProblem(inst, np, tiled_gram=CFG)
    z, y = _point(problem, 33)
    jac = lower_linear_inequalities(problem, z, np).ineq_jacobian(z)
    assert jac.gram_accumulate_dtype_hint() == "float32"
    h = problem.lagrangian_hessian(z, np.zeros(0), y)
    w = np.random.default_rng(34).uniform(0.1, 3.0, jac.shape[0])
    with pytest.raises(NotImplementedError):
        h.dense_matrix_with_gram(jac, w, accumulate_dtype="float32", hinted_only=True)
    fused = h.dense_matrix_with_gram(jac, w)  # an exact request merges
    ref = np.asarray(h.dense_matrix()) + np.asarray(jac.gram(w))
    assert _rel(np.asarray(fused), ref) < 1e-12


@pytest.mark.parametrize("linsolve", ["dense", "krylov"])
def test_merged_and_two_pass_solves_agree(linsolve):
    from ipax import Options, solve

    opts = Options(hessian="exact", linsolve=linsolve, max_iter=200)
    results = []
    for cfg in (CFG, NO_MERGE):
        problem = trots.TROTSExactProblem(_instance(), np, tiled_gram=cfg)
        results.append(
            (
                solve(problem, problem.initial_point(warm_start=False), options=opts),
                problem.hessian_stats["fused_calls"],
            )
        )
    (r_m, fused_calls), (r_p, plain_calls) = results
    assert plain_calls == 0
    if linsolve == "dense":
        assert fused_calls > 0  # the dense route went through the hook
    else:
        assert fused_calls == 0  # matrix-free: no dense block, no hook
    assert r_m.status == r_p.status
    assert r_m.routes == r_p.routes
    assert abs(r_m.n_iter - r_p.n_iter) <= 2  # round-off level only
    assert abs(r_m.objective - r_p.objective) <= 1e-8 * max(1.0, abs(r_p.objective))


def test_an_error_inside_the_offer_closes_it(monkeypatch):
    # The offer is open only for the one Gram call: if that call raises, a
    # later plain Gram must not pick up the Hessian's weights.
    problem, h, jac, w = _setup(np)
    _, _, jac_ref, _ = _setup(np, NO_MERGE)

    def boom(*_args, **_kwargs):
        raise RuntimeError("gram failed")

    monkeypatch.setattr(trots.TiledGramCSROperator, "_tiled_gram", boom)
    with pytest.raises(RuntimeError):
        h.dense_matrix_with_gram(jac, w)
    monkeypatch.undo()
    assert problem._gram_merge._offer is None
    np.testing.assert_array_equal(np.asarray(jac.gram(w)), np.asarray(jac_ref.gram(w)))


def test_offered_structure_nobody_takes_declines():
    # "oar" is offered, but the Jacobian handed over holds only the
    # nonlinear rows (no tiled operator to take it): the hook declines and
    # the core's two-pass block runs.
    problem, h, _, _ = _setup(np)
    z, _y = _point(problem, 31)
    nonlinear = problem.ineq_jacobian(z)
    with pytest.raises(NotImplementedError):
        h.dense_matrix_with_gram(nonlinear, np.ones(nonlinear.shape[0]))


def _shared_body(xp, cfg, seed):
    from tests.unit.test_trots_exact_hessian import _shared_body_instance

    problem = trots.TROTSExactProblem(_shared_body_instance(), xp, tiled_gram=cfg)
    z, y = _point(problem, seed)
    zx = xp.asarray(z)
    jac = lower_linear_inequalities(problem, zx, xp).ineq_jacobian(zx)
    h = problem.lagrangian_hessian(zx, xp.asarray(np.zeros(0)), xp.asarray(y))
    w = xp.asarray(np.random.default_rng(seed + 1).uniform(0.1, 3.0, jac.shape[0]))
    return h, jac, w


SHARED_CFG = trots.TiledGramConfig(block_rows=128, flop_ratio_threshold=0.5)


def test_partial_take_keeps_the_untaken_structures_in_the_hessian():
    # body and oar are curved and both in the linear block, but on the host
    # only oar's tile set is shared (the pre-gate keeps body's Hessian on
    # dense chunks): oar merges, body stays a Hessian term.
    h, jac, w = _shared_body(np, SHARED_CFG, 41)
    fused = np.asarray(h.dense_matrix_with_gram(jac, w))
    body = next(cs for cs, _w in h.terms if cs.mat.name == "body")
    assert body.tiles is None
    assert [cs.mat.name for cs in h.fused_structures] == ["oar"]
    h2, jac2, w2 = _shared_body(
        np, dataclasses.replace(SHARED_CFG, merge_hessian_gram=False), 41
    )
    ref = np.asarray(h2.dense_matrix()) + np.asarray(jac2.gram(w2))
    assert _rel(fused, ref) < 1e-12


def test_device_gated_off_structure_merges_through_its_device_plan(namespace):
    # On a device the Hessian multiplies a structure the gate keeps off
    # through its device plan, which the linear block shares: it merges.
    xp = namespace
    if trots._device_kind(xp) is None or trots._is_host_numpy(xp):
        pytest.skip("device Hessian runs on CuPy / Torch namespaces only")
    h, jac, w = _shared_body(xp, SHARED_CFG, 43)
    fused = trots._to_numpy_nd(h.dense_matrix_with_gram(jac, w))
    body = next(cs for cs, _w in h.terms if cs.mat.name == "body")
    assert body.tiles is None and body.device_tiles is not None
    assert {cs.mat.name for cs in h.fused_structures} == {"oar", "body"}
    h2, jac2, w2 = _shared_body(
        xp, dataclasses.replace(SHARED_CFG, merge_hessian_gram=False), 43
    )
    ref = trots._to_numpy_nd(h2.dense_matrix()) + trots._to_numpy_nd(jac2.gram(w2))
    assert _rel(fused, ref) < 1e-12
