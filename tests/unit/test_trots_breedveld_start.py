# Copyright 2026 Niklas Wahl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Breedveld et al. 2017 §3.1 initialisation of the TROTS loader + runner wiring.

The primal start is the dataset's least-squares fluence (eq. (23)) rescaled
towards its mean (eq. (24)) and floored at 5; the dual start solves
``∇g(x0)ᵀ y0 = −∇f(x0)`` by LSQR (eq. (26)) on the problem *as the solver sees
it* (``linear_ineq`` lowered by ipax) and floors ``y0``, ``z0`` (eq. (27)) and
the slacks ``w0`` (eq. (28)) at 5; ``μ0`` is the resulting average
complementarity (eq. (10)). No dataset is needed: the instance is synthetic.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")
spla = pytest.importorskip("scipy.sparse.linalg")
trots = pytest.importorskip("benchmarks.corpus.trots")

import ipax  # noqa: E402
from ipax.options import BarrierOptions  # noqa: E402
from ipax.problem.linear_ineq import lower_linear_inequalities  # noqa: E402
from ipax.result import WarmStart  # noqa: E402

N = 12


def _instance(ref_dose=6.0):
    """Init (LS-fluence) data, curved objectives/constraints, a minimax
    objective (one aux variable) and a two-sided linear block. ``ref_dose``
    is the LS target dose: at 6 every rescaled fluence entry lands on the
    floor of 5, at 50 about half of them clear it."""
    rng = np.random.default_rng(0)
    ptv = sp.csr_matrix(rng.uniform(0.5, 1.5, (20, N)))
    oar = sp.random(15, N, density=0.5, format="csr", random_state=rng)
    oar.data = rng.uniform(0.1, 1.0, oar.nnz)
    cache = {
        1: trots.TROTSMatrix("ptv", 0, ptv, np.zeros(0), 0.0, "float64"),
        2: trots.TROTSMatrix("oar", 0, oar, np.zeros(0), 0.0, "float32"),
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

    # Objective weights large enough that some LSQR multipliers (and some
    # stationarity residuals) clear the floor of 5 while others do not.
    entries = [
        entry("ptv ltcp", False, True, 4, 1, [6.0, 0.8], 0.0, 5000.0),
        entry("oar geud", False, True, 3, 2, [6.0], 0.0, 2000.0),
        entry("oar max", False, True, 1, 2, None, 0.0, 50.0),  # minimax aux
        entry("oar geud con", True, True, 3, 2, [4.0], 40.0),
        entry("ptv min", True, False, 1, 1, None, 4.0),  # A x ≥ 4 (lower rows)
        entry("oar max con", True, True, 1, 2, None, 60.0),  # A x ≤ 60
    ]
    inst = trots.TROTSInstance(
        identifier="synthetic-breedveld",
        n=N,
        real=N,
        entries=entries,
        solution=None,
        init_matrix_ids=np.asarray([1]),
        init_reference_dose=np.asarray([ref_dose]),
    )
    inst._matrix_cache = cache
    return inst


def _host(x):
    return trots._to_numpy_nd(x)


def _lowered_host(inst, x0):
    """The reference: a NumPy problem lowered exactly as ``ipax.solve`` does."""
    return lower_linear_inequalities(trots.TROTSProblem(inst, np), x0, np)


def _lsqr_reference(low, x0):
    jac = low.ineq_jacobian(x0)
    m, nv = jac.shape
    jt = spla.LinearOperator(
        (nv, m),
        matvec=lambda y: np.asarray(jac.rmatvec(np.ravel(y))),
        rmatvec=lambda v: np.asarray(jac.matvec(np.ravel(v))),
        dtype=float,
    )
    grad = np.asarray(low.gradient(x0))
    tol, iters = trots._BREEDVELD_LSQR_TOL, trots._BREEDVELD_LSQR_ITER
    return spla.lsqr(jt, -grad, atol=tol, btol=tol, iter_lim=iters)[0]


def _sparse_capable(namespace):
    """Whether the namespace can apply the TROTS CSR linear block itself."""
    from ipax.backend.sparse import get_sparse_adapter

    return isinstance(namespace.asarray(0.0), np.ndarray) or (
        get_sparse_adapter(namespace) is not None
    )


# --------------------------------------------------------------------------- #
# Primal start: eq. (24) + floor 5
# --------------------------------------------------------------------------- #
def test_breedveld_initial_point_rescales_and_floors_the_ls_fluence():
    problem = trots.TROTSProblem(_instance(), np)
    ls = problem.least_squares_fluence()
    expected_x = np.maximum(0.5 * ls + 0.5 * np.mean(ls), 5.0)  # eq. (24)

    z0 = _host(problem.initial_point(strategy="breedveld"))

    assert z0.shape == (problem.n_vars,)
    np.testing.assert_array_equal(z0[:N], expected_x)
    np.testing.assert_array_equal(z0, problem._pad_aux(expected_x))
    assert problem.n_vars == N + 1  # one minimax aux, consistent with x
    # The floor bites on this instance (reference dose 6 → fluence ~0.3).
    assert np.all(z0[:N] == 5.0)


def test_breedveld_initial_point_keeps_eq_24_above_the_floor():
    # A higher target dose lifts part of the rescaled fluence above 5: those
    # entries are exactly eq. (24), the rest sit on the floor.
    problem = trots.TROTSProblem(_instance(ref_dose=50.0), np)
    ls = problem.least_squares_fluence()
    rescaled = 0.5 * ls + 0.5 * np.mean(ls)  # eq. (24)
    above = rescaled > 5.0
    assert 0 < int(above.sum()) < N  # both branches are exercised

    z0 = _host(problem.initial_point(strategy="breedveld"))

    np.testing.assert_array_equal(z0[:N][above], rescaled[above])
    assert np.all(z0[:N][~above] == 5.0)
    np.testing.assert_array_equal(z0, problem._pad_aux(z0[:N]))


def test_ipax_strategy_is_the_default_and_unchanged():
    problem = trots.TROTSProblem(_instance(), np)
    default = _host(problem.initial_point())
    np.testing.assert_array_equal(
        _host(problem.initial_point(strategy="ipax")), default
    )
    ls = problem.least_squares_fluence()
    x = np.maximum(ls, 1e-6 * float(np.max(ls)))
    np.testing.assert_array_equal(default, problem._pad_aux(x))


def test_unknown_initial_point_strategy_is_rejected():
    with pytest.raises(ValueError, match="strategy"):
        trots.TROTSProblem(_instance(), np).initial_point(strategy="nope")


# --------------------------------------------------------------------------- #
# The least-squares fluence itself: eq. (23) normal equations
# --------------------------------------------------------------------------- #
def _ls_instance(case):
    """``_instance()`` with its initialisation data replaced per ``case``.

    ``"blocks+reg"``: two dense-ish init blocks (one with a dose offset ``b``,
    two reference doses) plus a non-symmetric regularisation matrix — the
    symmetrised ``R`` branch. ``"indefinite-reg"``: an ``R`` that makes the
    normal matrix indefinite — the LU fallback after a failed Cholesky.
    ``"rank-deficient"``: one wide block (rank 5 <
    N) and no regularisation — only the ridge makes the solve well-posed.
    ``"sparse-block"``: adds a block below the adapter's dense-Gram density
    gate (SpGEMM fallback) next to a dense one.
    """
    rng = np.random.default_rng(7)
    inst = _instance()
    cache = inst._matrix_cache
    dense = sp.csr_matrix(rng.uniform(0.5, 1.5, (20, N)))
    offset = sp.random(30, N, density=0.4, format="csr", random_state=rng)
    offset.data = rng.uniform(0.2, 1.0, offset.nnz)
    cache[11] = trots.TROTSMatrix("init a", 0, dense, np.zeros(0), 0.0)
    cache[12] = trots.TROTSMatrix(
        "init b", 0, offset.tocsc(), rng.uniform(0.0, 0.5, 30), 0.0
    )
    if case == "blocks+reg":
        reg = sp.random(N, N, density=0.3, format="csc", random_state=rng) * 0.05
        reg = (reg + 0.2 * sp.eye(N)).tocsc()  # non-symmetric on purpose
        cache[13] = trots.TROTSMatrix("reg", 2, reg, np.zeros(0), 0.0)
        inst.init_matrix_ids = np.asarray([11, 12])
        inst.init_reference_dose = np.asarray([6.0, 3.0])
        inst.init_reg_matrix_id = 13
    elif case == "indefinite-reg":
        # R with one strongly negative entry: the normal matrix is indefinite
        # (nonsingular), Cholesky fails and the general solve must take over.
        reg = sp.diags(np.r_[-1e3, np.zeros(N - 1)]).tocsc()
        cache[13] = trots.TROTSMatrix("reg", 2, reg, np.zeros(0), 0.0)
        inst.init_matrix_ids = np.asarray([11])
        inst.init_reference_dose = np.asarray([6.0])
        inst.init_reg_matrix_id = 13
    elif case == "rank-deficient":
        wide = sp.csr_matrix(rng.uniform(0.5, 1.5, (5, N)))
        cache[14] = trots.TROTSMatrix("wide", 0, wide, np.zeros(0), 0.0)
        inst.init_matrix_ids = np.asarray([14])
        inst.init_reference_dose = np.asarray([6.0])
        inst.init_reg_matrix_id = 0
    else:
        thin = sp.random(400, N, density=0.02, format="csr", random_state=rng)
        thin.data = rng.uniform(0.2, 1.0, thin.nnz)
        cache[15] = trots.TROTSMatrix("thin", 0, thin, np.zeros(0), 0.0)
        inst.init_matrix_ids = np.asarray([11, 15])
        inst.init_reference_dose = np.asarray([6.0, 2.0])
        inst.init_reg_matrix_id = 0
    return inst


def _ls_reference(inst, n, ridge):
    """The straightforward SciPy eq. (23) solve the loader must reproduce."""
    ids = np.asarray(inst.init_matrix_ids).ravel()
    dose = np.asarray(inst.init_reference_dose, dtype=float).ravel()
    blocks, rhs = [], []
    for k, mid in enumerate(ids):
        mat = inst.matrix(int(mid))
        a = sp.csr_matrix(mat.matrix)
        b = mat.b if mat.b.size == a.shape[0] else 0.0
        blocks.append(a)
        rhs.append(np.full(a.shape[0], dose[k]) - b)
    a_init = sp.vstack(blocks, format="csr")
    normal = (a_init.T @ a_init).toarray()
    if inst.init_reg_matrix_id:
        r = inst.matrix(inst.init_reg_matrix_id).matrix.toarray()
        normal = normal + 0.5 * (r + r.T)
    normal = normal + ridge * np.trace(normal) / n * np.eye(n)
    x = np.linalg.solve(normal, a_init.T @ np.concatenate(rhs))
    return np.maximum(x, 0.0)


@pytest.mark.parametrize(
    ("case", "ridge"),
    [
        ("blocks+reg", 1e-8),
        ("indefinite-reg", 1e-8),
        ("rank-deficient", 1e-4),
        ("sparse-block", 1e-8),
    ],
)
def test_least_squares_fluence_matches_the_scipy_normal_equations(case, ridge):
    # 2026-09-28: the normal matrix is formed by the dense-BLAS Gram and
    # solved by Cholesky instead of SpGEMM + todense + LU — the same x to
    # round-off (the ridge case keeps its conditioning test-friendly).
    inst = _ls_instance(case)
    problem = trots.TROTSProblem(inst, np)

    x = problem.least_squares_fluence(ridge=ridge)
    ref = _ls_reference(inst, N, ridge)

    assert isinstance(x, np.ndarray) and x.shape == (N,)
    assert float(np.min(x)) >= 0.0
    assert int(np.sum(ref > 0.0)) > 0  # a non-trivial start
    assert float(np.max(np.abs(x - ref))) <= 1e-10 * float(np.max(np.abs(ref)))


def test_least_squares_fluence_forms_no_sparse_gram(monkeypatch):
    # The SpGEMM ``(AᵀA).todense()`` cost 1–35 s on the TROTS cases; dense-ish
    # init blocks must reach the Gram through dense BLAS instead.
    compressed = pytest.importorskip("scipy.sparse._compressed")
    base = pytest.importorskip("scipy.sparse._base")
    if not hasattr(compressed._cs_matrix, "_matmul_sparse"):
        pytest.skip("SciPy internals changed: no _matmul_sparse to spy on")
    inst = _ls_instance("blocks+reg")
    problem = trots.TROTSProblem(inst, np)
    calls: list[str] = []
    real_matmul = compressed._cs_matrix._matmul_sparse
    real_todense = base._spbase.todense

    def spy_matmul(self, other):
        calls.append("spgemm")
        return real_matmul(self, other)

    def spy_todense(self, *args, **kwargs):
        calls.append("todense")
        return real_todense(self, *args, **kwargs)

    monkeypatch.setattr(compressed._cs_matrix, "_matmul_sparse", spy_matmul)
    monkeypatch.setattr(base._spbase, "todense", spy_todense)

    problem.least_squares_fluence()
    assert calls == []

    a = sp.csr_matrix(np.ones((3, 2)))
    (a.T @ a).todense()  # the spies do see the old formulation
    assert calls == ["spgemm", "todense"]


def test_breedveld_initial_point_on_the_namespace(namespace):
    host = _host(
        trots.TROTSProblem(_instance(), np).initial_point(strategy="breedveld")
    )
    z0 = trots.TROTSProblem(_instance(), namespace).initial_point(strategy="breedveld")
    assert type(z0) is type(namespace.asarray(0.0))
    np.testing.assert_array_equal(_host(z0), host)


# --------------------------------------------------------------------------- #
# Dual start: eqs. (26)–(28) and μ0 by eq. (10)
# --------------------------------------------------------------------------- #
def _check_warm_start(inst, x0_host, warm, mu0, s_f=1.0, *, device_lsqr=False):
    """Check eqs. (26)–(28) and (10); ``s_f`` is the solver's objective factor
    the returned μ0 carries (1 for ``scaling="none"``). ``device_lsqr`` marks
    a start whose LSQR ran on the GPU (looser multiplier tolerance)."""
    low = _lowered_host(inst, x0_host)
    m = int(np.asarray(low.ineq_constraints(x0_host)).shape[0])
    s, y, zl = _host(warm.s), _host(warm.y_ineq), _host(warm.z_lower)
    assert warm.y_eq is None and warm.z_upper is None
    assert s.shape == (m,) and y.shape == (m,) and zl.shape == (low.n_vars,)

    # eq. (26) + floor: y0 = max(LSQR(∇gᵀ y = −∇f), 5).
    y_ls = _lsqr_reference(low, x0_host)
    # rtol 1e-8 only on the device path: the CuPy start runs LSQR on the GPU,
    # whose reduction-order round-off LSQR amplifies (7e-10 here); the host
    # path keeps the tight 1e-10.
    rtol_y = 1e-8 if device_lsqr else 1e-10
    np.testing.assert_allclose(y, np.maximum(y_ls, 5.0), rtol=rtol_y, atol=1e-10)
    # eq. (28): w0 = max(−g(x0), 5); ipax slacks satisfy g + s = 0.
    g = np.asarray(low.ineq_constraints(x0_host))
    np.testing.assert_allclose(s, np.maximum(-g, 5.0), rtol=1e-12, atol=0)
    # eq. (27) on the bounded fluence block; aux entries are free (0).
    lo = _host(low.bounds()[0])
    bounded = np.isfinite(lo)
    assert bounded.sum() == N
    raw_z = np.asarray(low.gradient(x0_host)) + np.asarray(
        low.ineq_jacobian(x0_host).rmatvec(y)
    )
    np.testing.assert_allclose(
        zl[bounded], np.maximum(raw_z[bounded], 5.0), rtol=1e-10, atol=1e-10
    )
    assert np.all(zl[~bounded] == 0.0)
    assert np.all(s >= 5.0) and np.all(y >= 5.0) and np.all(zl[bounded] >= 5.0)
    # Where the floor did not bite, z0 is exactly the stationarity residual.
    free = bounded & (raw_z > 5.0)
    np.testing.assert_allclose(zl[free], raw_z[free], rtol=1e-12, atol=0)
    # eq. (10): average complementarity over slacks and bounded variables.
    expected_mu = (float(s @ y) + float(zl[bounded] @ (x0_host - lo)[bounded])) / (
        m + int(bounded.sum())
    )
    assert mu0 == pytest.approx(s_f * expected_mu, rel=1e-12)
    return y_ls, raw_z, bounded


def test_breedveld_warm_start_matches_eqs_26_to_28_and_10():
    inst = _instance()
    problem = trots.TROTSProblem(inst, np)
    x0 = problem.initial_point(strategy="breedveld")

    warm, mu0 = problem.breedveld_warm_start(x0, scaling="none")

    assert isinstance(warm, WarmStart)
    y_ls, raw_z, bounded = _check_warm_start(inst, _host(x0), warm, mu0)
    # The instance exercises both sides of every floor.
    assert np.any(y_ls > 5.0) and np.any(y_ls < 5.0)
    assert np.any(raw_z[bounded] > 5.0) and np.any(raw_z[bounded] < 5.0)


def test_breedveld_warm_start_on_the_exact_problem_matches_the_base():
    # The exact-Hessian subclass lowers the same rows in the same order.
    inst = _instance()
    base = trots.TROTSProblem(inst, np)
    exact = trots.TROTSExactProblem(inst, np)
    x0 = base.initial_point(strategy="breedveld")
    wb, mb = base.breedveld_warm_start(x0)
    we, me = exact.breedveld_warm_start(x0)
    for a, b in ((wb.s, we.s), (wb.y_ineq, we.y_ineq), (wb.z_lower, we.z_lower)):
        np.testing.assert_allclose(_host(a), _host(b), rtol=1e-12, atol=1e-12)
    assert mb == pytest.approx(me, rel=1e-12)


def test_breedveld_warm_start_on_the_namespace(namespace):
    # Host-side computation off CuPy (a NumPy twin of the problem supplies the
    # lowered Jacobian), on the GPU on CuPy; results come back in the
    # namespace, on x0's device, and agree with the host reference.
    from ipax.backend.namespace import device

    inst = _instance()
    problem = trots.TROTSProblem(inst, namespace)
    x0 = problem.initial_point(strategy="breedveld")

    warm, mu0 = problem.breedveld_warm_start(x0, scaling="none")

    kind = type(namespace.asarray(0.0))
    assert all(type(a) is kind for a in (warm.s, warm.y_ineq, warm.z_lower))
    assert all(device(a) == device(x0) for a in (warm.s, warm.y_ineq, warm.z_lower))
    assert isinstance(mu0, float)
    _check_warm_start(inst, _host(x0), warm, mu0, device_lsqr=problem._eval_device)
    if _sparse_capable(namespace):
        # The solver-order claim, directly: the slacks line up with the
        # namespace problem's own lowered rows (as ipax.solve lowers them).
        g_ns = lower_linear_inequalities(problem, x0, namespace).ineq_constraints(x0)
        np.testing.assert_allclose(
            _host(warm.s), np.maximum(-_host(g_ns), 5.0), rtol=1e-12, atol=1e-12
        )


def test_host_twin_keeps_the_multiplier_order_of_a_folded_float32_group(namespace):
    # The float32 share gate decides the lowered row order; the host twin
    # (built off NumPy) must be built with the same threshold as the problem.
    if isinstance(namespace.asarray(0.0), np.ndarray):
        pytest.skip("on NumPy the problem is its own host twin")
    if not _sparse_capable(namespace):
        pytest.skip("the TROTS linear block needs a sparse adapter to be applied")
    inst = _instance()
    folded = trots.TROTSProblem(inst, namespace, float32_group_min_share=1.0)
    split = trots.TROTSProblem(inst, np, float32_group_min_share=0.0)
    assert folded.float32_group_folded and not split.float32_group_folded
    twin = folded._host_twin()
    assert twin is not folded
    assert twin.float32_group_folded
    x0_ns = folded.initial_point(strategy="breedveld")
    x0 = _host(x0_ns)

    g_ns = _host(
        lower_linear_inequalities(folded, x0_ns, namespace).ineq_constraints(x0_ns)
    )
    g_twin = np.asarray(lower_linear_inequalities(twin, x0, np).ineq_constraints(x0))
    g_split = np.asarray(lower_linear_inequalities(split, x0, np).ineq_constraints(x0))

    np.testing.assert_allclose(g_twin, g_ns, rtol=1e-12, atol=1e-12)
    # The two layouts order the rows differently, so the check can tell them
    # apart (same rows, other order).
    np.testing.assert_allclose(np.sort(g_split), np.sort(g_ns), rtol=1e-12)
    assert not np.allclose(g_split, g_ns)


def test_mu0_is_in_the_solver_scaled_space(namespace):
    # ipax.solve scales every warm-start complementarity pair by the objective
    # factor s_f = min(1, 100/‖∇f(x0)‖∞) but takes mu_init as given, so μ0 is
    # returned in that space; the warm-start arrays stay in original units.
    from ipax.options import ScalingOptions

    inst = _instance()
    problem = trots.TROTSProblem(inst, namespace)
    x0 = problem.initial_point(strategy="breedveld")
    grad_norm = float(np.max(np.abs(_host(problem.gradient(x0)))))
    assert grad_norm > 100.0  # the scaling is active on this instance
    s_f = min(1.0, 100.0 / grad_norm)

    # CuPy runs LSQR on the device (see _check_warm_start's device_lsqr).
    device = problem._eval_device
    warm_none, mu_none = problem.breedveld_warm_start(x0, scaling="none")
    # unscaled eq. (10)
    _check_warm_start(inst, _host(x0), warm_none, mu_none, device_lsqr=device)
    for scaling in (None, "gradient-based", ScalingOptions()):
        warm, mu0 = problem.breedveld_warm_start(x0, scaling=scaling)
        assert mu0 == pytest.approx(s_f * mu_none, rel=1e-12)
        _check_warm_start(inst, _host(x0), warm, mu0, s_f=s_f, device_lsqr=device)
        np.testing.assert_array_equal(_host(warm.y_ineq), _host(warm_none.y_ineq))
    _, mu_loose = problem.breedveld_warm_start(
        x0, scaling=ScalingOptions(max_gradient=1e6)
    )
    assert mu_loose == pytest.approx(mu_none, rel=1e-12)


def test_mu0_scale_factor_matches_the_core_scaling():
    # The loader mirrors ipax's gradient-based objective factor; check it
    # against the core's own computation on the lowered problem.
    from ipax.problem.scaling import compute_scaling

    problem = trots.TROTSProblem(_instance(), np)
    x0 = problem.initial_point(strategy="breedveld")
    low = lower_linear_inequalities(problem, x0, np)
    core = compute_scaling(
        low, x0, np, has_eq=False, has_ineq=True, max_gradient=100.0
    ).obj
    _, mu_none = problem.breedveld_warm_start(x0, scaling="none")
    _, mu0 = problem.breedveld_warm_start(x0)
    assert core < 1.0
    assert mu0 == pytest.approx(core * mu_none, rel=1e-12)


def test_lsqr_iteration_limit_is_logged(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(trots, "_BREEDVELD_LSQR_ITER", 1)
    problem = trots.TROTSProblem(_instance(), np)
    x0 = problem.initial_point(strategy="breedveld")
    with caplog.at_level(logging.WARNING, logger=trots.__name__):
        problem.breedveld_warm_start(x0)
    assert "iteration limit" in caplog.text


def test_truncating_lsqr_is_not_safe_in_general(monkeypatch):
    # 2026-09-28 profile follow-up: on the measured TROTS corpus, every LSQR
    # multiplier ends up below the floor of 5 (see docs/benchmarks/trots.md's
    # Head-and-Neck_01 profile), so the floored dual start does not depend on
    # the LSQR iterate at all on those cases -- but that is a corpus fact, not
    # a property of the code. This instance (``_instance``'s docstring:
    # "exercises both sides of every floor") is the counterexample: an early
    # exit before LSQR converges changes the floored y0/z0/mu0 outright,
    # which is why no such shortcut is shipped as a default (a bound on the
    # *current* LSQR iterate's infinity norm cannot certify that later
    # iterations will not cross the floor -- LSQR's ‖y_k‖ is monotonically
    # non-decreasing, so a small early iterate proves nothing about later
    # ones; only running to convergence, or to the iteration cap, does).
    inst = _instance()
    problem = trots.TROTSProblem(inst, np)
    x0 = problem.initial_point(strategy="breedveld")

    warm_full, mu0_full = problem.breedveld_warm_start(x0, scaling="none")

    monkeypatch.setattr(trots, "_BREEDVELD_LSQR_ITER", 1)  # an early exit
    warm_early, mu0_early = problem.breedveld_warm_start(x0, scaling="none")

    assert not np.array_equal(
        np.asarray(warm_full.y_ineq), np.asarray(warm_early.y_ineq)
    )
    assert mu0_early != pytest.approx(mu0_full, rel=1e-6)


def test_lsqr_iter_is_an_opt_in_parameter():
    # 2026-09-28: the iteration cap of the counterexample above is now a
    # keyword parameter, not just a monkeypatchable module constant.
    # ``lsqr_iter=5`` reproduces the module constant's early exit (the 7 of 11
    # corpus-sweep cases hitting the 200 cap all bottom out well below 5) and
    # must change the floored start on this counterexample instance; passing
    # the current default explicitly must reproduce the implicit default
    # bit-for-bit.
    inst = _instance()
    problem = trots.TROTSProblem(inst, np)
    x0 = problem.initial_point(strategy="breedveld")

    warm_default, mu0_default = problem.breedveld_warm_start(x0, scaling="none")
    warm_explicit, mu0_explicit = problem.breedveld_warm_start(
        x0, scaling="none", lsqr_iter=trots._BREEDVELD_LSQR_ITER
    )
    np.testing.assert_array_equal(warm_default.y_ineq, warm_explicit.y_ineq)
    assert mu0_default == mu0_explicit

    warm_capped, mu0_capped = problem.breedveld_warm_start(
        x0, scaling="none", lsqr_iter=5
    )
    assert not np.array_equal(
        np.asarray(warm_default.y_ineq), np.asarray(warm_capped.y_ineq)
    )
    assert mu0_capped != pytest.approx(mu0_default, rel=1e-6)


def test_lsqr_tol_is_an_opt_in_parameter():
    inst = _instance()
    problem = trots.TROTSProblem(inst, np)
    x0 = problem.initial_point(strategy="breedveld")

    warm_default, mu0_default = problem.breedveld_warm_start(x0, scaling="none")
    warm_explicit, mu0_explicit = problem.breedveld_warm_start(
        x0, scaling="none", lsqr_tol=trots._BREEDVELD_LSQR_TOL
    )
    np.testing.assert_array_equal(warm_default.y_ineq, warm_explicit.y_ineq)
    assert mu0_default == mu0_explicit

    # A much looser tolerance stops LSQR earlier, same as a small iter cap.
    warm_loose, mu0_loose = problem.breedveld_warm_start(
        x0, scaling="none", lsqr_tol=1e-1
    )
    assert not np.array_equal(
        np.asarray(warm_default.y_ineq), np.asarray(warm_loose.y_ineq)
    )
    assert mu0_loose != pytest.approx(mu0_default, rel=1e-6)


def test_lsqr_iter_and_tol_are_validated():
    problem = trots.TROTSProblem(_instance(), np)
    x0 = problem.initial_point(strategy="breedveld")
    with pytest.raises(ValueError, match="lsqr_iter must be >= 1"):
        problem.breedveld_warm_start(x0, lsqr_iter=0)
    with pytest.raises(ValueError, match="lsqr_tol must be > 0"):
        problem.breedveld_warm_start(x0, lsqr_tol=0.0)
    with pytest.raises(ValueError, match="lsqr_tol must be > 0"):
        problem.breedveld_warm_start(x0, lsqr_tol=-1e-6)


def test_lsqr_port_matches_scipy():
    # trots._lsqr (the device LSQR) transcribes SciPy's Paige–Saunders
    # iteration: on the host it reproduces SciPy's iterate, stop code and
    # iteration count on a well-conditioned rectangular system, through the
    # matvec / rmatvec pair only.
    rng = np.random.default_rng(7)
    a = rng.standard_normal((40, 90))  # A: 40×90, b in R^40 (eq. (26)'s shape)
    b = rng.standard_normal(40)
    for tol, iters in ((1e-6, 200), (0.0, 7)):
        ref_x, ref_stop, ref_itn = spla.lsqr(
            a, b, atol=tol, btol=tol, conlim=1e8, iter_lim=iters
        )[:3]
        x, istop, itn = trots._lsqr(
            lambda v: a @ v, lambda u: a.T @ u, b, 90, np,
            atol=tol, btol=tol, iter_lim=iters,
        )  # fmt: skip
        assert (istop, itn) == (ref_stop, ref_itn)
        np.testing.assert_allclose(x, ref_x, rtol=1e-10, atol=1e-13)
    assert ref_stop == 7  # the second case exercises the iteration limit
    # b = 0: the exact solution x = 0 (SciPy's istop 0, no iterations).
    x, istop, itn = trots._lsqr(
        lambda v: a @ v, lambda u: a.T @ u, np.zeros(40), 90, np,
        atol=1e-6, btol=1e-6, iter_lim=10,
    )  # fmt: skip
    assert (istop, itn) == (0, 0)
    assert not np.any(x)


def test_lsqr_mode_is_validated():
    problem = trots.TROTSProblem(_instance(), np)
    x0 = problem.initial_point(strategy="breedveld")
    with pytest.raises(ValueError, match="CuPy"):
        problem.breedveld_warm_start(x0, lsqr="device")
    with pytest.raises(ValueError, match="unknown lsqr"):
        problem.breedveld_warm_start(x0, lsqr="gpu")
    # "auto" is the host path off CuPy.
    wa, ma = problem.breedveld_warm_start(x0)
    wh, mh = problem.breedveld_warm_start(x0, lsqr="host")
    np.testing.assert_array_equal(wa.y_ineq, wh.y_ineq)
    assert ma == mh


def test_device_lsqr_start_matches_the_host_start_on_cupy():
    # On a CuPy namespace the start runs on the GPU (the device LSQR on the
    # problem's own lowered operator, no host twin) and matches the host
    # SciPy path to round-off, before and after the floors.
    pytest.importorskip("cupy")
    import cupy

    try:
        cupy.cuda.runtime.getDeviceCount()
        cupy.asarray(0.0)
    except Exception:
        pytest.skip("no usable CUDA device")
    from ipax.testing.backends import import_namespace

    xp = import_namespace("cupy")
    inst = _instance()
    problem = trots.TROTSProblem(inst, xp)
    x0 = problem.initial_point(strategy="breedveld")

    dev = problem._breedveld_least_squares(x0, "auto")
    host = problem._breedveld_least_squares(x0, "host")
    assert dev.on_device and not host.on_device
    assert isinstance(dev.y, cupy.ndarray)
    np.testing.assert_allclose(cupy.asnumpy(dev.y), host.y, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(
        host.y, _lsqr_reference(_lowered_host(inst, _host(x0)), _host(x0)), rtol=1e-12
    )

    wd, md = problem.breedveld_warm_start(x0, scaling="none")
    wh, mh = problem.breedveld_warm_start(x0, scaling="none", lsqr="host")
    for a, b in ((wd.s, wh.s), (wd.y_ineq, wh.y_ineq), (wd.z_lower, wh.z_lower)):
        assert isinstance(a, cupy.ndarray) and a.device == x0.device
        np.testing.assert_allclose(cupy.asnumpy(a), _host(b), rtol=1e-8, atol=1e-12)
    assert md == pytest.approx(mh, rel=1e-8)
    _check_warm_start(inst, _host(x0), wd, md, device_lsqr=True)


def test_undefined_mu0_raises(monkeypatch):
    # No inequalities and no finite lower bounds: eq. (10) has no terms.
    inst = _instance()
    inst.entries = [e for e in inst.entries if e.name == "ptv ltcp"]
    problem = trots.TROTSProblem(inst, np)
    free = np.full((problem.n_vars,), -np.inf)
    monkeypatch.setattr(problem, "bounds", lambda: (free, -free))
    x0 = problem.initial_point(strategy="breedveld")
    with pytest.raises(ValueError, match="undefined"):
        problem.breedveld_warm_start(x0)


def test_tiled_linear_block_lowers_in_the_host_twin_row_order():
    # ipax lowers a tiled-Gram block (a CSROperator subclass) through a signed
    # row selection, the plain block at the triplet level; the NumPy problem
    # uses itself (tiled) and every other namespace a plain host twin, so the
    # two lowerings must order the rows — and hence y_ineq — identically.
    from ipax.backend.operators import VStack

    forced = trots.TiledGramConfig(block_rows=4, flop_ratio_threshold=1e9, threads=1)
    inst = _instance()
    tiled = trots.TROTSProblem(inst, np, tiled_gram=forced)
    plain = trots.TROTSProblem(inst, np, tiled_gram=False)
    op = tiled.linear_ineq()[0]
    blocks = op.blocks if isinstance(op, VStack) else (op,)
    assert all(isinstance(b, trots.TiledGramCSROperator) for b in blocks)
    x0 = tiled.initial_point(strategy="breedveld")

    wt, mt = tiled.breedveld_warm_start(x0)
    wp, mp = plain.breedveld_warm_start(x0)

    for a, b in ((wt.s, wp.s), (wt.y_ineq, wp.y_ineq), (wt.z_lower, wp.z_lower)):
        np.testing.assert_allclose(_host(a), _host(b), rtol=1e-12, atol=1e-12)
    assert mt == pytest.approx(mp, rel=1e-12)


def test_solve_accepts_the_breedveld_start(namespace):
    from ipax.backend.sparse import get_sparse_adapter

    if not isinstance(namespace.asarray(0.0), np.ndarray) and (
        get_sparse_adapter(namespace) is None
    ):
        pytest.skip("the TROTS linear block needs a sparse adapter to be applied")
    problem = trots.TROTSExactProblem(_instance(), namespace)
    x0 = problem.initial_point(strategy="breedveld")
    warm, mu0 = problem.breedveld_warm_start(x0)
    opts = ipax.Options(
        hessian="exact",
        linsolve="dense",
        barrier=BarrierOptions(mu_init=mu0),
        max_iter=3,
    )

    result = ipax.solve(problem, x0, warm_start=warm, options=opts)

    assert result.n_iter >= 1
    assert np.all(np.isfinite(_host(result.x)))


# --------------------------------------------------------------------------- #
# Runner: Breedveld's setup is the default; the old recipe stays reachable
# --------------------------------------------------------------------------- #
def test_runner_defaults_are_breedveld_setup():
    # Breedveld et al. 2017's setup with the Mehrotra probing μ and the
    # centrality floor off (the 2026-09-27 TROTS sweep's winner on every case).
    from benchmarks.runners import trots as runner

    args = runner._build_parser().parse_args([])
    assert args.config == "exact/dense"
    assert args.corrections == "mehrotra"
    assert args.globalization == "breedveld"
    assert args.mu_schedule == "probing"
    assert args.kappa_centrality == 0.0
    assert args.init == "breedveld"
    assert args.tol == 1e-8

    opts = runner._solve_options(args, "exact/dense")
    assert opts.hessian == "exact" and opts.linsolve == "dense"
    assert opts.corrections.method == "mehrotra"
    assert opts.globalization == "breedveld"
    assert opts.mu_schedule == "probing"
    assert opts.barrier.kappa_centrality == 0.0
    assert opts.optimality.dual_inf_tol == 1e-8
    assert opts.optimality.constr_viol_tol == 1e-8
    assert opts.optimality.compl_inf_tol == 1e-8


def test_runner_tol_sets_the_three_kkt_tolerances():
    from benchmarks.runners import trots as runner

    args = runner._build_parser().parse_args(["--tol", "1e-6"])
    opt = runner._solve_options(args, "exact/dense").optimality
    assert (opt.dual_inf_tol, opt.constr_viol_tol, opt.compl_inf_tol) == (1e-6,) * 3


def test_runner_reproduces_the_previous_default_options():
    # The pre-Breedveld default: both condensed dense configs, the loader's
    # start, monotone μ, the filter line search, no corrector, ipax's own
    # centrality floor — which the old runner built as
    # ``Options(max_iter, max_time, **config)`` alone.
    from benchmarks.runners import trots as runner

    parser = runner._build_parser()
    args = runner._parse_args(
        parser,
        [
            "--config",
            "lbfgs/dense,exact/dense",
            "--init",
            "ipax",
            "--mu-schedule",
            "monotone",
            "--globalization",
            "filter",
            "--corrections",
            "none",
            "--kappa-centrality",
            "1e-2",
        ],
    )
    labels = args.config.split(",")
    assert labels == ["lbfgs/dense", "exact/dense"]
    for label in labels:
        previous = ipax.Options(max_iter=300, max_time=120.0, **runner._CONFIGS[label])
        assert runner._solve_options(args, label) == previous


def test_runner_slack_init_scale_is_an_ipax_init_lever():
    from benchmarks.runners import trots as runner

    parser = runner._build_parser()
    args = runner._parse_args(
        parser,
        ["--init", "ipax", "--slack-init-scale", "0.1", "--kappa-centrality", "1e-4"],
    )
    barrier = runner._solve_options(args, "lbfgs/dense").barrier
    assert barrier.slack_init_scale == 0.1
    assert barrier.kappa_centrality == 1e-4


def test_runner_rejects_slack_init_scale_under_breedveld_init(capsys):
    # The Breedveld start sets the slacks itself (eq. (28)), so the flag would
    # be ignored: an argument error, raised before any dataset lookup.
    from benchmarks.runners import trots as runner

    with pytest.raises(SystemExit) as excinfo:
        runner.main(["--slack-init-scale", "0.1"])
    assert excinfo.value.code == 2
    assert "--slack-init-scale" in capsys.readouterr().err


def test_runner_lsqr_iter_and_tol_are_breedveld_init_levers():
    from benchmarks.runners import trots as runner

    parser = runner._build_parser()
    args = runner._parse_args(
        parser, ["--init", "breedveld", "--lsqr-iter", "5", "--lsqr-tol", "1e-3"]
    )
    assert (args.lsqr_iter, args.lsqr_tol) == (5, 1e-3)


@pytest.mark.parametrize("flag", ["--lsqr-iter", "--lsqr-tol"])
def test_runner_rejects_lsqr_overrides_under_ipax_init(capsys, flag):
    # These parameters only affect the Breedveld dual start's LSQR solve
    # (eq. (26)); the ipax start never runs it, so the flag would be ignored.
    from benchmarks.runners import trots as runner

    value = "5" if flag == "--lsqr-iter" else "1e-3"
    with pytest.raises(SystemExit) as excinfo:
        runner.main(["--init", "ipax", flag, value])
    assert excinfo.value.code == 2
    assert flag in capsys.readouterr().err


def test_runner_wires_lsqr_iter_and_tol_into_the_breedveld_start(monkeypatch):
    from benchmarks.runners import trots as runner

    inst = _instance()
    monkeypatch.setattr(runner, "load_trots_file", lambda path: inst)
    monkeypatch.setattr(runner, "reference_for", lambda case, root: None)
    captured = {}

    def fake_run_case(bench, **kwargs):
        captured["bench"] = bench
        return "row"

    monkeypatch.setattr(runner, "run_case", fake_run_case)
    args = runner._build_parser().parse_args(
        ["--init", "breedveld", "--lsqr-iter", "5"]
    )
    opts = runner._solve_options(args, "exact/dense")

    runner._solve_case(
        "root",
        "Synthetic",
        "exact/dense",
        opts,
        np,
        init="breedveld",
        lsqr_iter=args.lsqr_iter,
        lsqr_tol=args.lsqr_tol,
    )
    bench = captured["bench"]
    problem, placeholder = bench.build(np)
    x0, warm, _ = bench.prepare_solve(problem, placeholder, opts)
    capped_warm, capped_mu0 = problem.breedveld_warm_start(
        x0, scaling=opts.scaling, lsqr_iter=5
    )
    default_warm, default_mu0 = problem.breedveld_warm_start(x0, scaling=opts.scaling)
    np.testing.assert_array_equal(_host(warm.y_ineq), _host(capped_warm.y_ineq))
    assert not np.array_equal(_host(warm.y_ineq), _host(default_warm.y_ineq))
    assert capped_mu0 != pytest.approx(default_mu0, rel=1e-6)


@pytest.mark.parametrize("init", ["breedveld", "ipax"])
def test_runner_case_wires_the_initialisation(monkeypatch, init):
    from benchmarks.runners import trots as runner

    inst = _instance()
    monkeypatch.setattr(runner, "load_trots_file", lambda path: inst)
    monkeypatch.setattr(runner, "reference_for", lambda case, root: None)
    captured = {}

    def fake_run_case(bench, **kwargs):
        captured["bench"] = bench
        captured.update(kwargs)
        return "row"

    monkeypatch.setattr(runner, "run_case", fake_run_case)
    args = runner._build_parser().parse_args(["--init", init])
    opts = runner._solve_options(args, "exact/dense")

    assert runner._solve_case("root", "Synthetic", "exact/dense", opts, np, init=init)
    bench = captured["bench"]
    problem, placeholder = bench.build(np)
    assert isinstance(problem, trots.TROTSExactProblem)
    # ``build`` returns the loader's uniform placeholder; the start itself is
    # initialisation and comes out of the timed hook.
    np.testing.assert_array_equal(
        _host(placeholder), _host(problem.initial_point(warm_start=False))
    )
    x0, warm, new_opts = bench.prepare_solve(problem, placeholder, opts)
    np.testing.assert_array_equal(
        _host(x0), _host(problem.initial_point(strategy=init))
    )
    if init == "ipax":
        assert warm is None and new_opts is opts
        return
    ref_warm, mu0 = problem.breedveld_warm_start(x0, scaling=opts.scaling)
    np.testing.assert_allclose(_host(warm.y_ineq), _host(ref_warm.y_ineq))
    assert new_opts == dataclasses.replace(
        opts, barrier=dataclasses.replace(opts.barrier, mu_init=mu0)
    )
    # The hook forwards the solve's own scaling (μ0 lives in solver space).
    unscaled = dataclasses.replace(opts, scaling="none")
    _, _, none_opts = bench.prepare_solve(problem, placeholder, unscaled)
    mu_none = problem.breedveld_warm_start(x0, scaling="none")[1]
    assert none_opts.barrier.mu_init == mu_none != mu0


@pytest.mark.parametrize("init", ["breedveld", "ipax"])
def test_runner_times_the_least_squares_start_as_initialisation(monkeypatch, init):
    # Review finding (2026-09-27): the least-squares fluence solve of eq. (23)
    # ran inside ``build``, outside both timers, so a slow primal start left
    # the reported solve time unchanged. It now runs in the timed hook: a
    # 0.2 s solve shows up in ``prepare_time`` and hence in ``solve_time``.
    import time

    from benchmarks.harness import run_case
    from benchmarks.runners import trots as runner

    inst = _instance()
    monkeypatch.setattr(runner, "load_trots_file", lambda path: inst)
    monkeypatch.setattr(runner, "reference_for", lambda case, root: None)
    calls: list[str] = []
    phase = ["build"]
    real_ls = trots.TROTSProblem.least_squares_fluence

    def slow_ls(self, *args, **kwargs):
        calls.append(phase[0])
        time.sleep(0.2)
        return real_ls(self, *args, **kwargs)

    monkeypatch.setattr(trots.TROTSProblem, "least_squares_fluence", slow_ls)
    captured = {}
    monkeypatch.setattr(
        runner, "run_case", lambda bench, **kw: captured.setdefault("bench", bench)
    )
    args = runner._build_parser().parse_args(["--init", init, "--max-iter", "2"])
    opts = runner._solve_options(args, "exact/dense")
    runner._solve_case("root", "Synthetic", "exact/dense", opts, np, init=init)
    bench = captured["bench"]

    bench.build(np)
    assert calls == []  # the untimed build does not solve for the start

    phase[0] = "prepare"
    result = run_case(bench, config="exact/dense", options=opts, xp=np, backend="numpy")

    assert result.error is None, result.error
    assert calls == ["prepare"]  # one least-squares solve, inside the timed hook
    assert result.prepare_time >= 0.2
    assert result.solve_time >= result.prepare_time


# --------------------------------------------------------------------------- #
# Units-aware dual floor (Head-and-Neck_01 diagnosis, 2026-09-28) and the
# runner's --dual-floor / --common-step levers
# --------------------------------------------------------------------------- #
def test_dual_floor_default_is_the_papers_floor():
    inst = _instance()
    problem = trots.TROTSProblem(inst, np)
    x0 = problem.initial_point(strategy="breedveld")
    implicit, mu_implicit = problem.breedveld_warm_start(x0, scaling="none")
    explicit, mu_explicit = problem.breedveld_warm_start(
        x0, scaling="none", dual_floor=5.0
    )
    for a, b in (
        (implicit.s, explicit.s),
        (implicit.y_ineq, explicit.y_ineq),
        (implicit.z_lower, explicit.z_lower),
    ):
        np.testing.assert_array_equal(_host(a), _host(b))
    assert mu_implicit == mu_explicit


def test_dual_floor_floors_the_multipliers_and_keeps_the_slack_floor():
    # eqs. (26)–(27) with the floor ``dual_floor`` on y0 and z0; the slacks of
    # eq. (28) keep the paper's floor of 5 (the diagnosis lowered only the
    # multipliers), and μ0 is eq. (10) of the resulting start.
    inst = _instance()
    problem = trots.TROTSProblem(inst, np)
    x0 = _host(problem.initial_point(strategy="breedveld"))
    floor = 1e-6

    warm, mu0 = problem.breedveld_warm_start(x0, scaling="none", dual_floor=floor)

    low = _lowered_host(inst, x0)
    s, y, zl = _host(warm.s), _host(warm.y_ineq), _host(warm.z_lower)
    y_ls = _lsqr_reference(low, x0)
    np.testing.assert_allclose(y, np.maximum(y_ls, floor), rtol=1e-10, atol=1e-12)
    assert np.any(y_ls < 5.0) and np.all(y >= floor)
    g = np.asarray(low.ineq_constraints(x0))
    np.testing.assert_allclose(s, np.maximum(-g, 5.0), rtol=1e-12, atol=0)
    lo = _host(low.bounds()[0])
    bounded = np.isfinite(lo)
    raw_z = np.asarray(low.gradient(x0)) + np.asarray(low.ineq_jacobian(x0).rmatvec(y))
    np.testing.assert_allclose(
        zl[bounded], np.maximum(raw_z[bounded], floor), rtol=1e-10, atol=1e-12
    )
    assert np.all(zl[~bounded] == 0.0)
    m = s.shape[0]
    expected_mu = (float(s @ y) + float(zl[bounded] @ (x0 - lo)[bounded])) / (
        m + int(bounded.sum())
    )
    assert mu0 == pytest.approx(expected_mu, rel=1e-12)
    # A lower floor gives a smaller μ0 than the paper's floor of 5.
    assert mu0 < problem.breedveld_warm_start(x0, scaling="none")[1]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_dual_floor_must_be_positive_and_finite(bad):
    problem = trots.TROTSProblem(_instance(), np)
    x0 = problem.initial_point(strategy="breedveld")
    with pytest.raises(ValueError, match="dual_floor"):
        problem.breedveld_warm_start(x0, dual_floor=bad)


def test_runner_common_step_and_dual_floor_default_off():
    from benchmarks.runners import trots as runner

    args = runner._build_parser().parse_args([])
    assert args.common_step is False
    assert args.dual_floor == 5.0
    opts = runner._solve_options(args, "exact/dense")
    assert opts.breedveld.common_step is False


def test_runner_common_step_sets_the_breedveld_option():
    from benchmarks.runners import trots as runner

    args = runner._parse_args(runner._build_parser(), ["--common-step"])
    opts = runner._solve_options(args, "exact/dense")
    assert opts.breedveld.common_step is True
    # Only the flag differs from the default options.
    default = runner._solve_options(
        runner._build_parser().parse_args([]), "exact/dense"
    )
    assert (
        dataclasses.replace(
            opts, breedveld=dataclasses.replace(opts.breedveld, common_step=False)
        )
        == default
    )


def test_runner_gram_dtype_defaults_to_auto_and_sets_the_dense_option():
    from benchmarks.runners import trots as runner

    default = runner._solve_options(
        runner._build_parser().parse_args([]), "exact/dense"
    )
    assert default.dense.gram_dtype == "auto"
    assert default.dense == ipax.Options().dense  # the ipax default: a no-op
    args = runner._parse_args(runner._build_parser(), ["--gram-dtype", "float32"])
    opts = runner._solve_options(args, "exact/dense")
    assert opts.dense.gram_dtype == "float32"
    assert dataclasses.replace(opts, dense=default.dense) == default


def test_runner_rejects_an_unknown_gram_dtype(capsys):
    from benchmarks.runners import trots as runner

    with pytest.raises(SystemExit):
        runner._parse_args(runner._build_parser(), ["--gram-dtype", "float16"])
    assert "--gram-dtype" in capsys.readouterr().err


def test_runner_breedveld_soc_defaults_off_and_sets_the_option():
    from benchmarks.runners import trots as runner

    default_args = runner._build_parser().parse_args([])
    assert default_args.breedveld_soc == 0
    default = runner._solve_options(default_args, "exact/dense")
    assert default.breedveld.max_soc == 0

    args = runner._parse_args(runner._build_parser(), ["--breedveld-soc", "2"])
    opts = runner._solve_options(args, "exact/dense")
    assert opts.breedveld.max_soc == 2
    # Only the flag differs from the default options.
    assert (
        dataclasses.replace(
            opts, breedveld=dataclasses.replace(opts.breedveld, max_soc=0)
        )
        == default
    )
    # 0 is the default and stays valid under the filter line search.
    runner._parse_args(
        runner._build_parser(), ["--globalization", "filter", "--breedveld-soc", "0"]
    )


def test_runner_solve_line_shows_soc_counts_only_when_attempted():
    # The counts themselves come from benchmarks.harness.run_case, which sums
    # the public IterationRecord.soc_attempted/.used_soc off Result.history
    # (see tests/integration/test_benchmark_harness.py); this only checks the
    # console line's own formatting.
    from benchmarks.runners import trots as runner

    row = {
        "status": "optimal",
        "n_iter": 10,
        "kkt_error": 1e-9,
        "constraint_violation": 0.0,
        "solve_time": 1.0,
        "ref_iters": None,
        "ref_time_s": None,
        "soc_attempted": 0,
        "soc_accepted": 0,
    }
    assert "soc=" not in runner._solve_line("case", "exact/dense", row)
    row.update(soc_attempted=5, soc_accepted=3)
    assert "soc=3/5" in runner._solve_line("case", "exact/dense", row)


@pytest.mark.parametrize(
    ("argv", "flag"),
    [
        (["--init", "ipax", "--dual-floor", "1e-6"], "--dual-floor"),
        (["--globalization", "filter", "--common-step"], "--common-step"),
        (["--dual-floor", "0"], "--dual-floor"),
        (["--globalization", "filter", "--breedveld-soc", "1"], "--breedveld-soc"),
        (["--breedveld-soc", "-1"], "--breedveld-soc"),
    ],
)
def test_runner_rejects_ignored_or_invalid_levers(capsys, argv, flag):
    # A dual floor only exists for the Breedveld start, the common step only
    # for the Breedveld controller: an argument error, not a silent no-op.
    from benchmarks.runners import trots as runner

    with pytest.raises(SystemExit) as excinfo:
        runner._parse_args(runner._build_parser(), argv)
    assert excinfo.value.code == 2
    assert flag in capsys.readouterr().err


def test_runner_case_wires_the_dual_floor(monkeypatch):
    from benchmarks.runners import trots as runner

    inst = _instance()
    monkeypatch.setattr(runner, "load_trots_file", lambda path: inst)
    monkeypatch.setattr(runner, "reference_for", lambda case, root: None)
    captured = {}
    monkeypatch.setattr(
        runner, "run_case", lambda bench, **kw: captured.setdefault("bench", bench)
    )
    args = runner._parse_args(runner._build_parser(), ["--dual-floor", "1e-6"])
    opts = runner._solve_options(args, "exact/dense")
    runner._solve_case(
        "root", "Synthetic", "exact/dense", opts, np, dual_floor=args.dual_floor
    )
    bench = captured["bench"]
    problem, placeholder = bench.build(np)
    x0, warm, new_opts = bench.prepare_solve(problem, placeholder, opts)

    ref_warm, mu0 = problem.breedveld_warm_start(
        x0, scaling=opts.scaling, dual_floor=1e-6
    )
    np.testing.assert_array_equal(_host(warm.y_ineq), _host(ref_warm.y_ineq))
    np.testing.assert_array_equal(_host(warm.z_lower), _host(ref_warm.z_lower))
    assert new_opts.barrier.mu_init == mu0
