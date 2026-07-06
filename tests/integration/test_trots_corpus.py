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

    n_ineq = problem._n_nl_ineq + problem._n_lin_ineq
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
    g = np.asarray(problem.ineq_constraints(z))
    # The reference plan satisfies every constraint up to the DVH smoothing slack.
    assert float(np.max(g)) <= 1e-3


@pytest.mark.parametrize("hessian", ["lbfgs", "exact"])
def test_small_case_solves_to_feasible_kkt_point(hessian):
    instance = trots.load_trots_file(f"{_ROOT}/Prostate_BT_01.mat")
    cls = trots.TROTSExactProblem if hessian == "exact" else trots.TROTSProblem
    problem = cls(instance, np, sparse=True)
    result = solve(
        problem,
        problem.initial_point(),
        options=Options(hessian=hessian, linsolve="sparse", max_iter=100),
    )
    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    assert result.constraint_violation <= 1e-6


def test_list_and_reference_parsing():
    cases = trots.list_trots_cases()
    assert cases == sorted(cases)
    ref = trots.reference_for("Liver_01")
    if ref is not None:
        assert ref.significant_figures >= 1
        assert ref.x.size == trots.load_trots_file(f"{_ROOT}/Liver_01.mat").real
