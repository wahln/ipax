"""Gated tests for the S2MPJ corpus adapter (host-bridged, multi-backend).

These require a local S2MPJ checkout (no license to vendor) pointed to by
``IPAX_S2MPJ_DIR``; the whole module skips when it is absent, so the per-PR suite
is unaffected. With a checkout present they validate the NumPy→backend bridge:
the lowered constraint signs (via finite differences) and that representative
Hock-Schittkowski problems reach their known optima on every CPU backend.
"""

from __future__ import annotations

import pytest

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
