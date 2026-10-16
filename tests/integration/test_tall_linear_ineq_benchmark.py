"""Smoke + equivalence tests for the tall ``linear_ineq`` benchmark suite.

``benchmarks/runners/bench_tall_linear_ineq.py`` tracks the tall condensed path
(``m ≫ n``) over commits. These tests keep it honest at tiny sizes:

* the two declarations of :class:`benchmarks.generators.TallSparseQP`
  (hand-lowered operator Jacobian vs public ``linear_ineq``) are the same
  problem — same objective, KKT tolerance, routes, and lowered row set;
* range rows (``two_sided``) are honoured;
* the asv class's ``setup``/``time_*``/``track_*`` methods run for one cell;
* the route-code mapping covers every route the built-in solvers can emit, so
  a new route string fails here loudly instead of silently in a nightly run.

The generator is a NumPy/SciPy application-edge object, so these run on NumPy
only (the core paths they exercise are covered multi-backend elsewhere).
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import numpy as np
import pytest

pytest.importorskip("scipy")

import ipax
import ipax.linalg
from benchmarks.generators import make_tall_sparse_problem
from benchmarks.runners import bench_tall_linear_ineq as bench
from ipax.backend.sparse import get_sparse_adapter
from ipax.options import DenseOptions, SparseOptions

N, M = 40, 600


def _solve(problem, **options):
    options.setdefault("hessian", "lbfgs")
    opts = ipax.Options(**options)
    return ipax.solve(problem, np.full(problem.n_vars, 0.5), options=opts)


@pytest.mark.parametrize("bandwidth", [None, 0.05])
@pytest.mark.parametrize("two_sided", [0.0, 0.3])
def test_declarations_describe_the_same_problem(bandwidth, two_sided):
    results = {}
    for declaration in ("operator", "linear_ineq"):
        problem = make_tall_sparse_problem(
            N,
            M,
            density=0.05,
            bandwidth=bandwidth,
            declaration=declaration,
            two_sided=two_sided,
        )
        results[declaration] = _solve(problem)
    op, lin = results["operator"], results["linear_ineq"]
    for result in (op, lin):
        assert result.status is ipax.Status.OPTIMAL
        assert result.kkt_error <= 1e-8
    assert abs(op.objective - lin.objective) <= 1e-8 * max(1.0, abs(op.objective))
    assert op.routes is not None and lin.routes is not None
    assert op.routes == lin.routes
    # Same one-sided row layout ⇒ the multipliers line up row for row.
    assert op.y_ineq.shape == lin.y_ineq.shape
    np.testing.assert_allclose(op.x, lin.x, rtol=1e-6, atol=1e-7)


def test_two_sided_rows_are_honoured_and_lowered_identically():
    op = make_tall_sparse_problem(N, M, density=0.05, two_sided=0.3)
    lin = make_tall_sparse_problem(
        N, M, density=0.05, two_sided=0.3, declaration="linear_ineq"
    )
    lower, upper = lin.row_bounds
    two = np.isfinite(lower)
    assert int(two.sum()) == round(0.3 * M)
    assert op.m_lowered == lin.m_lowered == M + int(two.sum())
    # Same data under both declarations, and the reference point is strictly
    # inside every range row.
    np.testing.assert_array_equal(op.row_bounds[0], lower)
    np.testing.assert_array_equal(op.row_bounds[1], upper)
    x_ref = np.full(N, 0.5)
    ax_ref = lin.matrix @ x_ref
    assert np.all(lower[two] < ax_ref[two])
    assert np.all(ax_ref < upper)

    # The generator's hand-lowered rows equal the core's lowering of the
    # linear_ineq block (same rows, same order, same offsets).
    from ipax.problem.linear_ineq import lower_linear_inequalities

    lowered = lower_linear_inequalities(lin, x_ref, np)
    probe = np.random.default_rng(1).uniform(-1.0, 2.0, size=N)
    np.testing.assert_allclose(
        lowered.ineq_constraints(probe), op.ineq_constraints(probe), rtol=1e-13
    )
    np.testing.assert_allclose(
        lowered.ineq_jacobian(probe).dense_matrix(),
        op.ineq_jacobian(probe).dense_matrix(),
        rtol=0,
        atol=0,
    )

    result = _solve(lin)
    assert result.status is ipax.Status.OPTIMAL
    ax = lin.matrix @ result.x
    assert np.all(ax[two] >= lower[two] - 1e-8)
    assert np.all(ax <= upper + 1e-8)
    assert np.all(result.x >= -1e-8)


def test_two_sided_zero_reproduces_the_one_sided_draw():
    base = make_tall_sparse_problem(N, M, density=0.05, seed=3)
    ranged = make_tall_sparse_problem(N, M, density=0.05, seed=3, two_sided=0.5)
    assert (base.matrix != ranged.matrix).nnz == 0
    np.testing.assert_array_equal(base.row_bounds[1], ranged.row_bounds[1])
    assert np.all(np.isinf(base.row_bounds[0]))


def test_generator_rejects_bad_arguments():
    with pytest.raises(ValueError, match="declaration"):
        make_tall_sparse_problem(N, M, declaration="nonlinear")
    with pytest.raises(ValueError, match="two_sided"):
        make_tall_sparse_problem(N, M, two_sided=1.5)


def test_linear_ineq_declaration_has_no_nonlinear_rows():
    problem = make_tall_sparse_problem(N, M, declaration="linear_ineq")
    with pytest.raises(NotImplementedError):
        problem.ineq_constraints(np.zeros(N))
    with pytest.raises(NotImplementedError):
        problem.ineq_jacobian(np.zeros(N))
    matrix, lower, upper = problem.linear_ineq()
    assert isinstance(matrix, ipax.CSROperator)
    assert lower.shape == upper.shape == (M,)
    assert make_tall_sparse_problem(N, M).linear_ineq() is None


# --- the asv suite -----------------------------------------------------------


def test_asv_suite_runs_one_small_cell(monkeypatch):
    # Shrink the first tracked cell so the smoke run stays fast; the methods,
    # options and route decoding are the suite's own.
    monkeypatch.setattr(bench, "_problem_size", lambda n_vars, excess: (N, M))
    cell = bench.tracked_cells()[0]
    monkeypatch.setattr(bench, "tracked_cells", lambda: [cell])
    assert len(cell) == len(bench.TallLinearIneq.param_names)

    timing = bench.TallLinearIneq()
    timing.setup(*cell)
    assert timing.time_solve(*cell) is None
    assert timing.peakmem_solve(*cell) is None

    tracker = bench.TallLinearIneqTrack()
    cache = tracker.setup_cache()
    assert set(cache) == {cell}
    tracker.setup(cache, *cell)
    assert tracker.track_success(cache, *cell) == 1.0
    assert 0 < tracker.track_iterations(cache, *cell) < 200
    assert 0.0 <= tracker.track_kkt_residual(cache, *cell) <= 1e-8
    assert tracker.track_route(cache, *cell) in set(bench.ROUTE_CODES.values())


def test_asv_suite_prunes_to_the_tracked_grid():
    """Every cell either builds or is skipped by ``setup`` (asv's
    NotImplementedError convention), and the kept cells are exactly
    ``TRACKED_CELLS`` × both declarations."""
    import itertools

    for cell in itertools.product(*bench.TallLinearIneq.params):
        if bench.skip_reason(*cell) is not None:
            with pytest.raises(NotImplementedError):
                bench.TallLinearIneq().setup(*cell)
            with pytest.raises(NotImplementedError):
                bench.TallLinearIneqTrack().setup({}, *cell)
    kept = bench.tracked_cells()
    expected = {
        (*base, declaration)
        for base in bench.TRACKED_CELLS
        for declaration in bench.DECLARATIONS
    }
    assert set(kept) == expected
    assert len(kept) == len(expected)
    assert bench.TallLinearIneqTrack.params == bench.TallLinearIneq.params
    # Every tracked cell is reachable from the params grid.
    for base in bench.TRACKED_CELLS:
        for value, axis in zip(base, bench.TallLinearIneq.params, strict=False):
            assert value in axis


def test_route_code_mapping_is_stable():
    assert bench.ROUTE_CODES == {
        ("dense", "condensed"): 1,
        ("dense", "augmented"): 2,
        ("krylov", "condensed"): 3,
        ("sparse", "augmented"): 4,
        ("sparse-NE", "normal_equations"): 5,
    }
    with pytest.raises(ValueError, match="unmapped route"):
        bench.route_code_of("magic (x)", "condensed")
    with pytest.raises(ValueError, match="unmapped route"):
        bench.route_code_of("dense", "normal_equations")


def _solver_classes_with_route_labels() -> set[type]:
    found: set[type] = set()
    for info in pkgutil.iter_modules(ipax.linalg.__path__):
        module = importlib.import_module(f"ipax.linalg.{info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if callable(getattr(obj, "describe", None)) and callable(
                getattr(obj, "kkt_form", None)
            ):
                found.add(obj)
    return found


def test_route_codes_cover_every_route_the_solvers_emit():
    """Solve through every built-in route and decode ``Result.routes``.

    The guard below fails when a new solver class with ``describe()``/
    ``kkt_form()`` appears in ``ipax.linalg`` — extend this test (and
    ``ROUTE_CODES``) for it instead of letting ``track_route`` raise nightly.
    """
    from ipax.linalg.dense import DenseSolver
    from ipax.linalg.krylov import KrylovSolver
    from ipax.linalg.sparse import SparseDirectSolver

    assert _solver_classes_with_route_labels() == {
        DenseSolver,
        KrylovSolver,
        SparseDirectSolver,
    }

    configs: list[dict] = [
        {"linsolve": "dense"},
        # The dense augmented form keeps an explicit (exact) Hessian block; with
        # L-BFGS it falls back to the condensed form.
        {
            "linsolve": "dense",
            "dense": DenseOptions(kkt_route="augmented"),
            "hessian": "exact",
        },
        {"linsolve": "krylov"},
    ]
    if get_sparse_adapter(np) is not None:
        configs += [
            {"linsolve": "sparse", "sparse": SparseOptions(kkt_route="augmented")},
            {
                "linsolve": "sparse",
                "sparse": SparseOptions(kkt_route="normal_equations"),
            },
        ]
    seen: set[int] = set()
    for config in configs:
        problem = make_tall_sparse_problem(
            N, M, density=0.05, declaration="linear_ineq", two_sided=0.2
        )
        result = _solve(problem, **config)
        assert result.routes is not None, config
        seen.add(bench.route_code(result.routes))
    if get_sparse_adapter(np) is not None:
        assert seen == set(bench.ROUTE_CODES.values())
