"""Cross-backend conformance tests."""

from __future__ import annotations

import pytest

from ipax import Options, solve
from ipax.testing.backends import import_namespace
from ipax.testing.problems import UnconstrainedQuadratic
from tests._helpers import array, implemented, to_float


def test_namespace_fixture_creates_float64_arrays_when_available(namespace):
    x = array(namespace, [1.0, 2.0])
    if hasattr(namespace, "float64"):
        assert x.dtype == namespace.float64
    assert x.shape == (2,)


def test_oracle_solution_agrees_across_available_backends(all_available_backends):
    if len(all_available_backends) < 2:
        pytest.skip("needs at least two installed Array-API backends")

    solutions = []
    for name in all_available_backends:
        xp = import_namespace(name)
        Q = array(xp, [[4.0, 1.0], [1.0, 3.0]])
        b = array(xp, [1.0, 2.0])
        problem = UnconstrainedQuadratic(Q, b, xp)
        solution = problem.known_solution()
        solutions.append(
            tuple(to_float(solution[idx]) for idx in range(solution.shape[0]))
        )

    reference = solutions[0]
    for solution in solutions[1:]:
        assert solution == pytest.approx(reference, rel=1e-9, abs=1e-9)


def test_same_solution_across_backends(all_available_backends):
    if len(all_available_backends) < 2:
        pytest.skip("needs at least two installed Array-API backends")

    solutions = []
    for name in all_available_backends:
        xp = import_namespace(name)
        Q = array(xp, [[4.0, 1.0], [1.0, 3.0]])
        b = array(xp, [1.0, 2.0])
        problem = UnconstrainedQuadratic(Q, b, xp)
        with implemented("solver"):
            result = solve(
                problem,
                array(xp, [0.0, 0.0]),
                options=Options(hessian="exact", linsolve="dense"),
            )
        solutions.append(
            tuple(to_float(result.x[idx]) for idx in range(result.x.shape[0]))
        )

    reference = solutions[0]
    for solution in solutions[1:]:
        assert solution == pytest.approx(reference, rel=1e-6, abs=1e-6)


def test_array_api_strict_raises_on_out_of_standard_calls():
    try:
        strict = import_namespace("array_api_strict")
    except ImportError:
        pytest.skip("array-api-strict is not installed")

    def missing_lstsq():
        return strict.linalg.lstsq

    assert not hasattr(strict.linalg, "lstsq")
    with pytest.raises(AttributeError):
        missing_lstsq()
