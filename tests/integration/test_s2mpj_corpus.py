"""Gated tests for the S2MPJ corpus adapter (host-bridged, multi-backend).

These require a local S2MPJ checkout (no license to vendor) pointed to by
``IPAX_S2MPJ_DIR``; the whole module skips when it is absent, so the per-PR suite
is unaffected. With a checkout present they validate the NumPy→backend bridge:
the lowered constraint signs (via finite differences) and that representative
Hock-Schittkowski problems reach their known optima on every CPU backend.
"""

from __future__ import annotations

import pytest

from benchmarks.harness import run_case
from ipax import Options, Status, solve
from ipax.problem.finitediff import gradient_fd, jacobian_fd
from tests._helpers import array, assert_allclose

s2mpj = pytest.importorskip("benchmarks.corpus.s2mpj")

_ROOT = s2mpj.s2mpj_dir()
if _ROOT is None:
    pytest.skip("no S2MPJ checkout (set IPAX_S2MPJ_DIR)", allow_module_level=True)

# The bridge evaluates in NumPy then converts; only CPU host backends apply.
_BRIDGE_BACKENDS = {"numpy", "torch"}

# Reliable known optima under the default L-BFGS path from the S2MPJ start point.
_KNOWN_F = {
    "s2mpj/HS21": -99.96,
    "s2mpj/HS35": 1.0 / 9.0,
    "s2mpj/HS71": 17.0140173,
    "s2mpj/HS6": 0.0,
    "s2mpj/HS8": -1.0,
    "s2mpj/HS28": 0.0,
}
_NAMES = ("HS21", "HS35", "HS71", "HS6", "HS8", "HS28")


@pytest.fixture
def bridge_namespace(backend_name, namespace):
    if backend_name not in _BRIDGE_BACKENDS:
        pytest.skip(f"S2MPJ host bridge does not target {backend_name!r}")
    return namespace


@pytest.mark.parametrize("name", _NAMES)
def test_s2mpj_problem_reaches_known_optimum(bridge_namespace, name):
    xp = bridge_namespace
    (case,) = s2mpj.s2mpj_problems((name,), backends=(xp.__name__.split(".")[-1],))
    problem, x0 = case.build(xp)
    result = solve(problem, x0, options=Options(hessian="lbfgs", linsolve="dense"))

    assert result.status is Status.OPTIMAL
    assert result.kkt_error <= 1e-6
    assert result.constraint_violation <= 1e-6
    assert abs(float(result.objective) - _KNOWN_F[case.name]) <= 1e-4


def test_list_s2mpj_problems_enumerates_the_checkout():
    names = s2mpj.list_s2mpj_problems()
    assert names == sorted(names)
    assert "s2mpjlib" not in names
    # The curated set is drawn from the full listing.
    assert set(_NAMES).issubset(set(names))


def test_unconstrained_problem_solves_when_available(bridge_namespace):
    xp = bridge_namespace
    if "ROSENBR" not in s2mpj.list_s2mpj_problems():
        pytest.skip("ROSENBR not in this S2MPJ checkout")
    (case,) = s2mpj.s2mpj_problems(("ROSENBR",), backends=(xp.__name__.split(".")[-1],))
    problem, x0 = case.build(xp)
    # Unconstrained: no clower/cupper attrs on the S2MPJ instance.
    result = solve(problem, x0, options=Options(hessian="lbfgs", linsolve="dense"))
    assert result.status is Status.OPTIMAL
    assert abs(float(result.objective)) <= 1e-6


def test_objective_free_problem_is_rejected_at_build(bridge_namespace):
    xp = bridge_namespace
    if "ARGLALE" not in s2mpj.list_s2mpj_problems():
        pytest.skip("ARGLALE not in this S2MPJ checkout")
    (case,) = s2mpj.s2mpj_problems(("ARGLALE",), backends=(xp.__name__.split(".")[-1],))
    with pytest.raises(NotImplementedError, match="no objective"):
        case.build(xp)


@pytest.mark.parametrize("name", ("HS71", "HS35", "HS6"))
def test_exact_hessian_route_reaches_known_optimum(bridge_namespace, name):
    # The exact Lagrangian Hessian S2MPJ supplies must drive the dense exact-Newton
    # route to the same optima the L-BFGS path finds — validating the multiplier
    # sign-mapping and σ handling end-to-end (scaling on, the solver default).
    xp = bridge_namespace
    (case,) = s2mpj.s2mpj_problems(
        (name,), backends=(xp.__name__.split(".")[-1],), hessian="exact"
    )
    problem, x0 = case.build(xp)
    result = solve(problem, x0, options=Options(hessian="exact", linsolve="dense"))

    assert result.status is Status.OPTIMAL
    assert result.kkt_error <= 1e-6
    assert result.derivative_sources.hessian == "exact"
    assert abs(float(result.objective) - _KNOWN_F[case.name]) <= 1e-4


@pytest.mark.parametrize("name", ("HS71", "HS35"))
def test_exact_sparse_route_matches_exact_dense(bridge_namespace, name):
    # The sparse-direct route factors the COO Jacobians/Hessian; it must agree with
    # the dense exact route to solver tolerance on the same problem.
    xp = bridge_namespace
    backend = xp.__name__.split(".")[-1]
    (dense_case,) = s2mpj.s2mpj_problems(
        (name,), backends=(backend,), hessian="exact", sparse=False
    )
    (sparse_case,) = s2mpj.s2mpj_problems(
        (name,), backends=(backend,), hessian="exact", sparse=True
    )
    dense_problem, x0 = dense_case.build(xp)
    sparse_problem, _ = sparse_case.build(xp)

    dense = solve(dense_problem, x0, options=Options(hessian="exact", linsolve="dense"))
    sparse = solve(
        sparse_problem, x0, options=Options(hessian="exact", linsolve="sparse")
    )

    assert dense.status is Status.OPTIMAL
    assert sparse.status is Status.OPTIMAL
    assert_allclose(xp, sparse.x, dense.x, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", ("BT1", "DISC2"))
def test_nonconvex_equality_exact_sparse_converges(bridge_namespace, name):
    # Regression: well-conditioned nonconvex *equality*-constrained problems on the
    # exact/sparse (inertia) route. Escalating δ_c in lockstep with δ_w on an
    # inertia-mismatch failure changes the very inertia the check tests against, so
    # δ_w ran away and the solve diverged (BT1 optimal→max_time). δ_c must escalate
    # only on a factorization failure, not an inertia mismatch. These reach the
    # optimum cleanly when the gating is correct.
    xp = bridge_namespace
    if name not in s2mpj.list_s2mpj_problems():
        pytest.skip(f"{name} not in this S2MPJ checkout")
    backend = xp.__name__.split(".")[-1]
    (case,) = s2mpj.s2mpj_problems(
        (name,), backends=(backend,), hessian="exact", sparse=True
    )
    problem, x0 = case.build(xp)
    result = solve(problem, x0, options=Options(hessian="exact", linsolve="sparse"))

    assert result.status is Status.OPTIMAL, f"{name}: {result.status}"
    assert result.kkt_error <= 1e-6


def test_documented_expected_outcome_is_attached(bridge_namespace):
    # The loader threads the dataset's documented outcome onto the built problem.
    xp = bridge_namespace
    backend = xp.__name__.split(".")[-1]
    (hs71,) = s2mpj.s2mpj_problems(("HS71",), backends=(backend,))
    problem, _ = hs71.build(xp)
    assert abs(problem.expected_objective - 17.0140173) <= 1e-7
    assert problem.expected_infeasible is False
    assert problem.pbclass and problem.pbclass.startswith("C-")

    if "BURKEHAN" in s2mpj.list_s2mpj_problems():
        (burke,) = s2mpj.s2mpj_problems(("BURKEHAN",), backends=(backend,))
        bproblem, _ = burke.build(xp)
        assert bproblem.expected_infeasible is True


def test_expected_infeasible_problem_scores_correct(bridge_namespace):
    # BURKEHAN is documented infeasible — detecting infeasibility is the correct
    # outcome, so the harness must score it correct (not flag it as a failure).
    xp = bridge_namespace
    backend = xp.__name__.split(".")[-1]
    if "BURKEHAN" not in s2mpj.list_s2mpj_problems():
        pytest.skip("BURKEHAN not in this S2MPJ checkout")
    (case,) = s2mpj.s2mpj_problems(("BURKEHAN",), backends=(backend,))
    result = run_case(
        case,
        config="lbfgs/dense",
        options=Options(hessian="lbfgs", linsolve="dense"),
        xp=xp,
        backend=backend,
    )
    assert result.status == "infeasible"
    assert result.expected_infeasible is True
    assert result.correct is True


def test_objective_free_problem_runs_as_feasibility(bridge_namespace):
    xp = bridge_namespace
    backend = xp.__name__.split(".")[-1]
    if "ARGLALE" not in s2mpj.list_s2mpj_problems():
        pytest.skip("ARGLALE not in this S2MPJ checkout")
    # Default: still rejected at build.
    (skipped,) = s2mpj.s2mpj_problems(("ARGLALE",), backends=(backend,))
    with pytest.raises(NotImplementedError, match="no objective"):
        skipped.build(xp)
    # With feasibility=True it builds and solves (min 0 s.t. the constraints).
    (case,) = s2mpj.s2mpj_problems(("ARGLALE",), backends=(backend,), feasibility=True)
    problem, x0 = case.build(xp)
    assert float(problem.objective(x0)) == 0.0
    result = solve(problem, x0, options=Options(hessian="lbfgs", linsolve="dense"))
    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE, Status.INFEASIBLE)


def test_sized_instantiation_scales_and_falls_back(bridge_namespace):
    # ``size`` requests a scalable problem's dimension (the scaling-sweep lever); a
    # fixed-size problem ignores it and keeps its SIF default.
    xp = bridge_namespace
    backend = xp.__name__.split(".")[-1]
    if "ARWHEAD" not in s2mpj.list_s2mpj_problems():
        pytest.skip("ARWHEAD not in this S2MPJ checkout")

    (scaled,) = s2mpj.s2mpj_problems(("ARWHEAD",), backends=(backend,), size=500)
    problem, _x0 = scaled.build(xp)
    assert problem.n_vars == 500  # scalable: honored the requested size

    (fixed,) = s2mpj.s2mpj_problems(("HS71",), backends=(backend,), size=500)
    fixed_problem, _ = fixed.build(xp)
    assert fixed_problem.n_vars == 4  # not size-parametrized: fell back to default


def test_s2mpj_bridge_derivatives_match_finite_differences(bridge_namespace):
    # HS71 exercises objective gradient + a nonlinear inequality and equality, so
    # FD agreement confirms the bridge and the constraint-lowering signs.
    xp = bridge_namespace
    (case,) = s2mpj.s2mpj_problems(("HS71",), backends=(xp.__name__.split(".")[-1],))
    problem, _x0 = case.build(xp)
    x = array(xp, [1.5, 4.0, 3.5, 1.5])

    assert_allclose(
        xp, problem.gradient(x), gradient_fd(problem.objective, x), rtol=1e-5, atol=1e-5
    )
    assert_allclose(
        xp,
        problem.eq_jacobian(x),
        jacobian_fd(problem.eq_constraints, x),
        rtol=1e-5,
        atol=1e-5,
    )
    assert_allclose(
        xp,
        problem.ineq_jacobian(x),
        jacobian_fd(problem.ineq_constraints, x),
        rtol=1e-5,
        atol=1e-5,
    )
