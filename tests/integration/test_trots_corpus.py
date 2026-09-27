"""Gated tests for the TROTS corpus loader (host-bridged radiotherapy problems).

These require a local copy of the TROTS ``.mat`` dataset pointed to by
``IPAX_TROTS_DIR`` (too large / licensed to vendor) *and* ``h5py`` to read the
MATLAB v7.3 (HDF5) files; the whole module skips when either is absent, so the
per-PR suite is unaffected. With the data present they validate that:

* the scalarised objective evaluated at the stored reference plan ``solutionX``
  reproduces the ``Results/*.txt`` ``Objective Function Value`` (the loader's
  ground-truth oracle across the linear / quadratic / gEUD / LTCP / DVH cost
  functions), and
* the assembled :class:`TROTSProblem` is self-consistent — its objective matches
  :func:`objective_at`, its gradient matches finite differences, its exact
  Lagrangian Hessian matches finite differences of the gradient, and a small
  non-convex case solves to a feasible KKT point on the sparse-direct route.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")
trots = pytest.importorskip("benchmarks.corpus.trots")

from ipax import Options, Status, solve  # noqa: E402

_ROOT = trots.trots_dir()
if _ROOT is None:
    pytest.skip("no TROTS dataset (set IPAX_TROTS_DIR)", allow_module_level=True)


def _padded_solution(problem, instance):
    """The reference ``solutionX`` extended with the minimax aux variables."""
    x = instance.solution
    z = np.concatenate([x, np.zeros(problem._n_aux)])
    for k, (mat, _w, minimise) in enumerate(problem._minimax):
        d = mat.matrix @ x
        z[instance.n + k] = np.max(d) if minimise else np.min(d)
    return z


# Cases with a reference result, kept to the smaller-loading groups so the gated
# run stays bounded (Head-and-Neck files are ~0.5–1.3 GB and are left out).
_REFERENCE_CASES = ("Protons_01", "Prostate_CK_01", "Prostate_VMAT_101", "Liver_01")


@pytest.mark.parametrize("case", _REFERENCE_CASES)
def test_objective_at_solution_matches_reference(case):
    if case not in trots.list_trots_cases():
        pytest.skip(f"{case} not present in this TROTS copy")
    instance = trots.load_trots_file(f"{_ROOT}/{case}.mat")
    ref = trots.reference_for(case)
    assert ref is not None, "reference result file missing"

    obj = trots.objective_at(instance, instance.solution)
    # Match to the significant figures the reference solve itself reports.
    rel_tol = 10.0 ** (-(ref.significant_figures - 1))
    rel_err = abs(obj - ref.objective) / max(1.0, abs(ref.objective))
    assert rel_err <= rel_tol, (
        f"{case}: objective_at={obj!r} vs reference {ref.objective!r} "
        f"(rel_err={rel_err:.2e} > {rel_tol:.2e})"
    )


def test_problem_objective_matches_objective_at():
    instance = trots.load_trots_file(f"{_ROOT}/Prostate_BT_01.mat")
    problem = trots.TROTSProblem(instance, np, sparse=True)
    z = _padded_solution(problem, instance)
    assert float(problem.objective(z)) == pytest.approx(
        trots.objective_at(instance, instance.solution), rel=1e-9, abs=1e-12
    )


def test_gradient_matches_finite_differences():
    instance = trots.load_trots_file(f"{_ROOT}/Prostate_BT_01.mat")
    problem = trots.TROTSProblem(instance, np, sparse=True)
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 2.0, size=instance.n)
    z = np.concatenate([x, np.zeros(problem._n_aux)])
    for k, (mat, _w, mini) in enumerate(problem._minimax):
        d = mat.matrix @ x
        z[instance.n + k] = np.max(d) if mini else np.min(d)

    g = np.asarray(problem.gradient(z))
    eps = 1e-6
    for i in rng.choice(problem.n_vars, size=min(10, problem.n_vars), replace=False):
        zp, zm = z.copy(), z.copy()
        zp[i] += eps
        zm[i] -= eps
        fd = (float(problem.objective(zp)) - float(problem.objective(zm))) / (2 * eps)
        assert fd == pytest.approx(float(g[i]), rel=1e-5, abs=1e-7)


def test_exact_hessian_matches_finite_differences():
    instance = trots.load_trots_file(f"{_ROOT}/Prostate_BT_01.mat")
    problem = trots.TROTSExactProblem(instance, np, sparse=True)
    rng = np.random.default_rng(1)
    x = rng.uniform(0.5, 2.0, size=instance.n)
    z = np.concatenate([x, np.zeros(problem._n_aux)])
    for k, (mat, _w, mini) in enumerate(problem._minimax):
        d = mat.matrix @ x
        z[instance.n + k] = np.max(d) if mini else np.min(d)

    # The solver slices the (curvature-free) lowered linear_ineq multipliers
    # off, so lagrangian_hessian receives the nonlinear rows' multipliers only.
    n_ineq = problem._n_nl_ineq
    H = problem.lagrangian_hessian(z, np.zeros(0), np.zeros(n_ineq), 1.0)
    Hd = np.asarray(H.dense_matrix())
    eps = 1e-6
    for i in rng.choice(instance.n, size=6, replace=False):
        zp, zm = z.copy(), z.copy()
        zp[i] += eps
        zm[i] -= eps
        fd_col = (
            np.asarray(problem.gradient(zp)) - np.asarray(problem.gradient(zm))
        ) / (2 * eps)
        np.testing.assert_allclose(fd_col, Hd[:, i], rtol=1e-4, atol=1e-6)


def test_reference_solution_is_feasible():
    instance = trots.load_trots_file(f"{_ROOT}/Liver_01.mat")
    problem = trots.TROTSExactProblem(instance, np, sparse=True)
    z = _padded_solution(problem, instance)
    # Evaluate as the solver does: the nonlinear rows plus the lowered
    # linear_ineq block (dose constraints and minimax links).
    from ipax.problem.linear_ineq import lower_linear_inequalities

    lowered = lower_linear_inequalities(problem, z, np)
    g = np.asarray(lowered.ineq_constraints(z))
    assert g.shape[0] > problem._n_nl_ineq
    # The reference plan satisfies every constraint up to the DVH smoothing slack.
    assert float(np.max(g)) <= 1e-3


@pytest.mark.parametrize("hessian", ["lbfgs", "exact"])
def test_small_case_solves_to_feasible_kkt_point(hessian):
    instance = trots.load_trots_file(f"{_ROOT}/Prostate_BT_01.mat")
    cls = trots.TROTSExactProblem if hessian == "exact" else trots.TROTSProblem
    problem = cls(instance, np, sparse=True)
    # Uniform start: a deterministic solver-route smoke test. (The least-squares
    # warm start, covered by its own tests, is tuned for the large convex cases and
    # can stall this small *non-convex* brachytherapy case — expected sensitivity.)
    result = solve(
        problem,
        problem.initial_point(warm_start=False),
        options=Options(hessian=hessian, linsolve="sparse", max_iter=100),
    )
    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    assert result.constraint_violation <= 1e-6


def test_least_squares_warm_start_delivers_target_dose():
    # The dataset's least-squares start (misc.Initialise*) drives the tumour
    # matrices toward their reference dose; the delivered mean dose should sit in a
    # sensible band around the target and the fluence must be non-negative.
    instance = trots.load_trots_file(f"{_ROOT}/Protons_01.mat")
    problem = trots.TROTSProblem(instance, np, sparse=True)
    x = problem.least_squares_fluence()
    assert x.shape == (instance.n,)
    assert float(np.min(x)) >= 0.0
    ids = np.asarray(instance.init_matrix_ids).ravel()
    dose = np.asarray(instance.init_reference_dose).ravel()
    assert ids.size > 0
    for k, mid in enumerate(ids):
        d = instance.matrix(int(mid)).matrix @ x
        # Within a factor of ~2 of the reference dose (regularisation pulls it in).
        assert 0.4 * dose[k] <= float(np.mean(d)) <= 2.0 * dose[k]


def test_initial_point_warm_start_beats_uniform_objective():
    # The warm start should land much closer to the reference objective than a
    # uniform fluence — the whole point of using the initialisation matrices.
    instance = trots.load_trots_file(f"{_ROOT}/Protons_01.mat")
    problem = trots.TROTSProblem(instance, np, sparse=True)
    ref = trots.reference_for("Protons_01")
    assert ref is not None
    warm = np.asarray(problem.initial_point(warm_start=True))[: instance.n]
    uniform = np.asarray(problem.initial_point(warm_start=False))[: instance.n]
    warm_err = abs(trots.objective_at(instance, warm) - ref.objective)
    uniform_err = abs(trots.objective_at(instance, uniform) - ref.objective)
    assert warm_err < uniform_err


def test_linear_block_declares_float32_source():
    # The dose matrices are float32 in the TROTS files: the assembled linear
    # inequality block must declare that source precision so the dense
    # route's gram_dtype="auto" default engages the reduced accumulate.
    instance = trots.load_trots_file(f"{_ROOT}/Prostate_BT_01.mat")
    problem = trots.TROTSProblem(instance, np, sparse=True)
    op = problem.linear_ineq()[0]
    assert op.gram_accumulate_dtype_hint() == "float32"


def test_device_evaluation_matches_host():
    # GPU-gated: with a CuPy namespace the problem callbacks must evaluate
    # device-side (no per-call host round-trip of the dose products) and agree
    # with the host NumPy/SciPy path to reduction-order rounding.
    pytest.importorskip("cupy")
    import cupy

    try:
        cupy.cuda.runtime.getDeviceCount()
        cupy.asarray(0.0)
    except Exception:
        pytest.skip("no usable CUDA device")
    from ipax.testing.backends import import_namespace

    xp = import_namespace("cupy")
    instance = trots.load_trots_file(f"{_ROOT}/Prostate_BT_01.mat")
    host = trots.TROTSProblem(instance, np, sparse=True)
    dev = trots.TROTSProblem(instance, xp, sparse=True)
    assert host._eval_device is False
    assert dev._eval_device is True

    rng = np.random.default_rng(7)
    x = rng.uniform(0.5, 2.0, size=instance.n)
    z = np.concatenate([x, np.zeros(host._n_aux)])
    for k, (mat, _w, mini) in enumerate(host._minimax):
        d = mat.matrix @ x
        z[instance.n + k] = np.max(d) if mini else np.min(d)
    z_dev = xp.asarray(z)

    # Objective / gradient / constraints agree and stay on the device.
    assert float(dev.objective(z_dev)) == pytest.approx(
        float(host.objective(z)), rel=1e-10, abs=1e-12
    )
    g_dev = dev.gradient(z_dev)
    assert hasattr(g_dev, "__cuda_array_interface__")
    np.testing.assert_allclose(
        cupy.asnumpy(g_dev), np.asarray(host.gradient(z)), rtol=1e-9, atol=1e-11
    )
    c_dev = dev.ineq_constraints(z_dev)
    assert hasattr(c_dev, "__cuda_array_interface__")
    np.testing.assert_allclose(
        cupy.asnumpy(c_dev),
        np.asarray(host.ineq_constraints(z)),
        rtol=1e-9,
        atol=1e-9,
    )

    # Jacobian parity through matvec/rmatvec probes (the operator itself must
    # be device-resident, so no dense m×n materialisation here).
    J_host = host.ineq_jacobian(z)
    J_dev = dev.ineq_jacobian(z_dev)
    m = J_host.shape[0]
    v = rng.standard_normal(host.n_vars)
    u = rng.standard_normal(m)
    np.testing.assert_allclose(
        cupy.asnumpy(J_dev.matvec(xp.asarray(v))),
        np.asarray(J_host.matvec(v)),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        cupy.asnumpy(J_dev.rmatvec(xp.asarray(u))),
        np.asarray(J_host.rmatvec(u)),
        rtol=1e-9,
        atol=1e-9,
    )

    # The linear_ineq block is declared in the solve namespace (device-resident
    # operator and bounds) and agrees with the host declaration.
    A_host, lo_host, hi_host = host.linear_ineq()
    A_dev, lo_dev, hi_dev = dev.linear_ineq()
    assert hasattr(lo_dev, "__cuda_array_interface__")
    np.testing.assert_array_equal(cupy.asnumpy(lo_dev), np.asarray(lo_host))
    np.testing.assert_array_equal(cupy.asnumpy(hi_dev), np.asarray(hi_host))
    np.testing.assert_allclose(
        cupy.asnumpy(A_dev.matvec(xp.asarray(z))),
        np.asarray(A_host.matvec(z)),
        rtol=1e-9,
        atol=1e-9,
    )


def test_list_and_reference_parsing():
    cases = trots.list_trots_cases()
    assert cases == sorted(cases)
    ref = trots.reference_for("Liver_01")
    if ref is not None:
        assert ref.significant_figures >= 1
        assert ref.x.size == trots.load_trots_file(f"{_ROOT}/Liver_01.mat").real


def test_all_zero_sparse_matrix_case_loads_and_validates():
    # Regression (2026-07-19): Head-and-Neck_05's 'Brainstem' matrix is an
    # all-zero MATLAB sparse (v7.3 writes only ``jc`` at nnz = 0), which
    # KeyError'd the reader and made the whole case unloadable. The synthetic
    # layout is pinned in tests/unit/test_trots_loader.py; this anchors the real
    # file (the one smaller HN case, ~135MB — the rest of the family stays out
    # of the gated run for load size).
    case = "Head-and-Neck_05"
    if case not in trots.list_trots_cases():
        pytest.skip(f"{case} not present in this TROTS copy")
    instance = trots.load_trots_file(f"{_ROOT}/{case}.mat")

    empty = [
        instance.matrix(e.data_id)
        for e in instance.entries
        if instance.matrix(e.data_id).matrix.nnz == 0
    ]
    assert empty, "expected at least one all-zero dose matrix in HN_05"
    assert all(m.matrix.shape[0] > 0 for m in empty)

    ref = trots.reference_for(case)
    if ref is not None and instance.solution is not None:
        obj = trots.objective_at(instance, instance.solution)
        rel_tol = 10.0 ** (-(ref.significant_figures - 1))
        rel_err = abs(obj - ref.objective) / max(1.0, abs(ref.objective))
        assert rel_err <= rel_tol


# --------------------------------------------------------------------------- #
# Tiled per-structure Gram (TiledGramCSROperator) vs the plain CSR block
# --------------------------------------------------------------------------- #
def _lowered_linear_jacobian(problem):
    from ipax.problem.linear_ineq import lower_linear_inequalities

    return lower_linear_inequalities(problem, np.zeros(problem.n_vars), np)._jac


def _linear_blocks(problem):
    op = problem.linear_ineq()[0]
    return op.blocks if hasattr(op, "blocks") else (op,)


def test_tiled_gram_gate_keeps_prostate_bt02_plain():
    # The dense 70-beamlet brachytherapy block cannot compact: the gate keeps
    # every group on the plain operator (so its solves are the plain ones).
    case = "Prostate_BT_02"
    if case not in trots.list_trots_cases():
        pytest.skip(f"{case} not present in this TROTS copy")
    instance = trots.load_trots_file(f"{_ROOT}/{case}.mat")
    problem = trots.TROTSProblem(instance, np)
    assert not any(
        isinstance(b, trots.TiledGramCSROperator) for b in _linear_blocks(problem)
    )


def test_tiled_gram_is_solve_neutral_on_liver01():
    # A case whose float64 group really tiles (Gram 0.74 -> 0.20 s): a short
    # solve on the RT recipe takes the same path with and without the kernel.
    from ipax.options import BarrierOptions

    case = "Liver_01"
    if case not in trots.list_trots_cases():
        pytest.skip(f"{case} not present in this TROTS copy")
    instance = trots.load_trots_file(f"{_ROOT}/{case}.mat")
    opts = Options(
        hessian="lbfgs",
        linsolve="dense",
        max_iter=10,
        barrier=BarrierOptions(slack_init_scale=0.1),
    )
    results = []
    for tiled in (True, False):
        problem = trots.TROTSProblem(instance, np, tiled_gram=tiled)
        kinds = [
            isinstance(b, trots.TiledGramCSROperator) for b in _linear_blocks(problem)
        ]
        assert any(kinds) is tiled
        results.append(solve(problem, problem.initial_point(), options=opts))
    r_t, r_p = results
    assert r_t.status == r_p.status
    assert r_t.n_iter == r_p.n_iter
    assert r_t.routes == r_p.routes
    assert abs(r_t.objective - r_p.objective) <= 1e-8 * max(1.0, abs(r_p.objective))


def test_tiled_gram_matches_plain_on_protons01_lowered_block():
    if "Protons_01" not in trots.list_trots_cases():
        pytest.skip("Protons_01 not present in this TROTS copy")
    instance = trots.load_trots_file(f"{_ROOT}/Protons_01.mat")
    tiled = trots.TROTSProblem(instance, np)
    plain = trots.TROTSProblem(instance, np, tiled_gram=False)
    blocks = _linear_blocks(tiled)  # a VStack of groups, or one group's operator
    assert any(isinstance(b, trots.TiledGramCSROperator) for b in blocks)

    j_t, j_p = _lowered_linear_jacobian(tiled), _lowered_linear_jacobian(plain)
    assert j_t.shape == j_p.shape
    rng = np.random.default_rng(2026)
    for _ in range(2):  # fresh weights each time (both paths memoize)
        w = rng.uniform(1e-3, 1e3, j_t.shape[0])
        g_t, g_p = np.asarray(j_t.gram(w)), np.asarray(j_p.gram(w))
        assert np.max(np.abs(g_t - g_p)) <= 1e-12 * np.max(np.abs(g_p))
        np.testing.assert_array_equal(g_t, g_t.T)


# --------------------------------------------------------------------------- #
# Grouped exact Hessian (TROTSExactHessian) vs the per-term SpGEMM path
# --------------------------------------------------------------------------- #
def _exact_pair(case):
    if case not in trots.list_trots_cases():
        pytest.skip(f"{case} not present in this TROTS copy")
    instance = trots.load_trots_file(f"{_ROOT}/{case}.mat")
    return (
        trots.TROTSExactProblem(instance, np),
        trots.TROTSExactProblem(instance, np, grouped_hessian=False),
    )


def test_grouped_exact_hessian_solves_like_per_term_on_prostate_bt02():
    # LTCP + DVH objectives and DVH constraints on shared structures, some of
    # them also linear-block structures.
    grouped, per_term = _exact_pair("Prostate_BT_02")
    opts = Options(hessian="exact", linsolve="dense")
    r_g, r_p = (solve(p, p.initial_point(), options=opts) for p in (grouped, per_term))
    assert r_g.status == r_p.status
    assert r_g.routes == r_p.routes
    assert abs(r_g.n_iter - r_p.n_iter) <= 2, (r_g.n_iter, r_p.n_iter)
    assert abs(r_g.objective - r_p.objective) <= 1e-8 * max(1.0, abs(r_p.objective))


@pytest.mark.parametrize("case", ["Protons_01", "Liver_01", "Prostate_CK_01"])
def test_grouped_exact_hessian_matches_per_term_at_the_initial_point(case):
    # Protons_01 has no curved term at all (linear dose block + minimax only):
    # its exact Hessian is identically zero on both paths. Liver_01 (DVH
    # constraints + smoothing quadratics) and Prostate_CK_01 (LTCP on a
    # dense-ish PTV shared by two terms) carry the real curvature.
    grouped, per_term = _exact_pair(case)
    z = np.asarray(grouped.initial_point())
    rng = np.random.default_rng(2026)
    y = rng.uniform(0.0, 2.0, grouped._n_nl_ineq)
    h_g = grouped.lagrangian_hessian(z, np.zeros(0), y, 1.0).dense_matrix()
    h_p = per_term.lagrangian_hessian(z, np.zeros(0), y, 1.0).dense_matrix()
    h_g, h_p = np.asarray(h_g), np.asarray(h_p)
    np.testing.assert_array_equal(h_g, h_g.T)
    scale = float(np.max(np.abs(h_p)))
    if scale == 0.0:
        assert case == "Protons_01"
        assert not np.any(h_g)
        return
    assert float(np.max(np.abs(h_g - h_p))) <= 1e-10 * scale


# --------------------------------------------------------------------------- #
# Device tiled Gram (CuPy): the NumPy plan run on the GPU
# --------------------------------------------------------------------------- #
def _cupy_namespace():
    pytest.importorskip("cupy")
    import cupy

    try:
        cupy.cuda.runtime.getDeviceCount()
        cupy.asarray(0.0)
    except Exception:
        pytest.skip("no usable CUDA device")
    from ipax.testing.backends import import_namespace

    return import_namespace("cupy")


def _lowered_on(problem, xp):
    from ipax.problem.linear_ineq import lower_linear_inequalities

    z0 = xp.zeros(problem.n_vars, dtype=xp.float64)
    return lower_linear_inequalities(problem, z0, xp)._jac


def test_device_tiled_gram_matches_plain_on_protons01_cupy():
    xp = _cupy_namespace()
    if "Protons_01" not in trots.list_trots_cases():
        pytest.skip("Protons_01 not present in this TROTS copy")
    instance = trots.load_trots_file(f"{_ROOT}/Protons_01.mat")
    tiled = trots.TROTSProblem(instance, xp)
    plain = trots.TROTSProblem(instance, xp, tiled_gram=False)
    assert any(isinstance(b, trots.TiledGramCSROperator) for b in _linear_blocks(tiled))
    j_t, j_p = _lowered_on(tiled, xp), _lowered_on(plain, xp)
    assert j_t.shape == j_p.shape
    rng = np.random.default_rng(2026)
    for _ in range(2):  # fresh weights each time (both paths memoize)
        w = xp.asarray(rng.uniform(1e-3, 1e3, j_t.shape[0]))
        g_t = trots._to_numpy_nd(j_t.gram(w))
        g_p = trots._to_numpy_nd(j_p.gram(w))
        assert np.max(np.abs(g_t - g_p)) <= 1e-10 * np.max(np.abs(g_p))
        np.testing.assert_array_equal(g_t, g_t.T)


def test_device_tiled_gram_accepts_raw_cupy_namespace():
    # Regression: the device kernels use ``device=`` keywords that the raw
    # ``cupy`` module (as opposed to the Array API compat wrapper the fixtures
    # hand out) does not accept, so a problem built on plain ``cupy`` failed at
    # its first lowered-block Gram with ``asarray() got an unexpected keyword
    # argument 'device'`` -- in the solver only once restoration asked for it.
    xp = _cupy_namespace()
    import cupy

    if "Protons_01" not in trots.list_trots_cases():
        pytest.skip("Protons_01 not present in this TROTS copy")
    instance = trots.load_trots_file(f"{_ROOT}/Protons_01.mat")
    raw = trots.TROTSProblem(instance, cupy)
    ref = trots.TROTSProblem(instance, xp)
    assert any(isinstance(b, trots.TiledGramCSROperator) for b in _linear_blocks(raw))
    j_raw, j_ref = _lowered_on(raw, cupy), _lowered_on(ref, xp)
    w = np.random.default_rng(11).uniform(1e-3, 1e3, j_raw.shape[0])
    g_raw = trots._to_numpy_nd(j_raw.gram(cupy.asarray(w)))
    g_ref = trots._to_numpy_nd(j_ref.gram(xp.asarray(w)))
    # Two device problems scatter their tiles in their own order, so the two
    # Grams agree to reduction-order round-off rather than bit-for-bit.
    np.testing.assert_allclose(g_raw, g_ref, rtol=1e-12, atol=0.0)


def test_device_tiled_gram_is_solve_neutral_on_protons01_cupy():
    from ipax.options import BarrierOptions

    xp = _cupy_namespace()
    if "Protons_01" not in trots.list_trots_cases():
        pytest.skip("Protons_01 not present in this TROTS copy")
    instance = trots.load_trots_file(f"{_ROOT}/Protons_01.mat")
    opts = Options(
        hessian="lbfgs",
        linsolve="dense",
        max_iter=10,
        barrier=BarrierOptions(slack_init_scale=0.1),
    )
    results = []
    for tiled in (True, False):
        problem = trots.TROTSProblem(instance, xp, tiled_gram=tiled)
        results.append(solve(problem, problem.initial_point(), options=opts))
    r_t, r_p = results
    assert r_t.status == r_p.status
    assert r_t.n_iter == r_p.n_iter
    assert r_t.routes == r_p.routes
    assert abs(r_t.objective - r_p.objective) <= 1e-8 * max(1.0, abs(r_p.objective))


def test_device_exact_hessian_solves_like_numpy_on_liver01_cupy(monkeypatch):
    # The device exact Hessian through ipax.solve: Liver_01's curved
    # structures (DVH constraints) tile, so CuPy gets a TROTSDeviceExactHessian
    # whose dense_matrix (device tiled kernel) and matvec (device products)
    # the solve drives. Parity with the NumPy exact run over 10 iterations.
    from ipax.options import BarrierOptions

    xp = _cupy_namespace()
    if "Liver_01" not in trots.list_trots_cases():
        pytest.skip("Liver_01 not present in this TROTS copy")
    instance = trots.load_trots_file(f"{_ROOT}/Liver_01.mat")
    calls = {"dense_matrix": 0, "matvec": 0}
    for name in calls:
        real = getattr(trots.TROTSDeviceExactHessian, name)

        def spy(self, *args, _real=real, _name=name, **kwargs):
            calls[_name] += 1
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(trots.TROTSDeviceExactHessian, name, spy)
    opts = Options(
        hessian="exact",
        linsolve="dense",
        max_iter=10,
        barrier=BarrierOptions(slack_init_scale=0.1),
    )
    results = []
    for ns in (np, xp):
        problem = trots.TROTSExactProblem(instance, ns)
        results.append(solve(problem, problem.initial_point(), options=opts))
    r_np, r_cp = results
    assert calls["dense_matrix"] > 0
    assert calls["matvec"] > 0
    assert r_cp.status == r_np.status
    assert r_cp.n_iter == r_np.n_iter
    assert r_cp.routes == r_np.routes
    assert abs(r_cp.objective - r_np.objective) <= 1e-8 * max(1.0, abs(r_np.objective))


# --------------------------------------------------------------------------- #
# Restoration's dense route on a tall linear block: Gram, not densification
# --------------------------------------------------------------------------- #
def test_restoration_forms_the_tall_linear_block_by_gram_on_liver01(monkeypatch):
    # TROTS 2026-09: the dense restoration route densified the whole lowered
    # dose block (Protons_01: 369 445 x 1101, 3.2 GB per restoration
    # iteration). Liver_01's block (121 975 x 1124, 1.1 GB) is above the
    # default RestorationOptions.dense_materialize_max_elements, so restore()
    # must form J^T diag(g > 0) J through the operators' Gram instead.
    import ipax.ipm.restoration as restoration_mod
    from ipax.backend.operators import _SparseStructured
    from ipax.options import RestorationOptions
    from ipax.problem.linear_ineq import lower_linear_inequalities

    case = "Liver_01"
    if case not in trots.list_trots_cases():
        pytest.skip(f"{case} not present in this TROTS copy")
    instance = trots.load_trots_file(f"{_ROOT}/{case}.mat")
    problem = trots.TROTSProblem(instance, np)
    z0 = np.asarray(problem.initial_point())
    lowered = lower_linear_inequalities(problem, z0, np)
    jac = lowered.ineq_jacobian(z0)
    m, n = jac.shape
    assert m * n > RestorationOptions().dense_materialize_max_elements
    g0 = np.asarray(lowered.ineq_constraints(z0))
    assert float(np.max(g0)) > 1.0  # the RT start is well outside the dose limits

    # Two outer iterations bound the test's wall; each is one Gram.
    monkeypatch.setattr(restoration_mod, "_MAX_ITER", 2)
    events: list[str] = []
    real_dense = restoration_mod._dense

    def spy_dense(op, xp, dtype):
        events.append(f"dense{op.shape}")
        return real_dense(op, xp, dtype)

    monkeypatch.setattr(restoration_mod, "_dense", spy_dense)
    for cls in (_SparseStructured, trots.TiledGramCSROperator):
        real_gram = cls.gram

        def spy_gram(self, *args, _real=real_gram, **kwargs):
            events.append("gram")
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(cls, "gram", spy_gram)
    real_matmat = _SparseStructured.matmat

    def spy_matmat(self, V):
        if tuple(V.shape) == (n, n):
            events.append("identity_matmat")
        return real_matmat(self, V)

    monkeypatch.setattr(_SparseStructured, "matmat", spy_matmat)

    lower, upper = problem.bounds()
    lower = np.full(n, -np.inf) if lower is None else np.asarray(lower)
    upper = np.full(n, np.inf) if upper is None else np.asarray(upper)
    mask_l, mask_u = np.isfinite(lower), np.isfinite(upper)
    x, s, _exit = restoration_mod.restore(
        xp=np,
        x=z0,
        s=np.ones(m),
        m=m,
        m_eq=0,
        eq_fn=lambda z: np.zeros(0),
        eq_jac_fn=lambda z: None,
        ineq_fn=lowered.ineq_constraints,
        ineq_jac_fn=lowered.ineq_jacobian,
        mask_l=mask_l,
        mask_u=mask_u,
        lower_safe=np.where(mask_l, lower, 0.0),
        upper_safe=np.where(mask_u, upper, 0.0),
        tol=1e-8,
    )

    assert "gram" in events
    assert not any(e.startswith("dense") for e in events), events
    assert "identity_matmat" not in events
    # ...and the Gram-formed Gauss-Newton step made progress on the violation.
    g1 = np.asarray(lowered.ineq_constraints(x))
    assert float(np.sum(np.maximum(g1, 0.0) ** 2)) < float(
        np.sum(np.maximum(g0, 0.0) ** 2)
    )
    assert s.shape == (m,)
