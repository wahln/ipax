"""``ipax.solve`` checks its arguments up front (API review B15, B6, C15, C16).

Wrong argument types raise ``TypeError``; a start point or bounds that do not
match the problem raise :class:`ipax.ProblemDefinitionError` (a
``ValueError``). Before 0.14 these failed with unrelated errors from deep
inside the solve (a broadcast ``ValueError``, ``IndexError``,
``AttributeError: 'dict' object has no attribute 'scaling'``).
"""

from __future__ import annotations

import pytest

import ipax
from ipax.backend.operators import MatrixFreeJacobian, as_operator
from ipax.problem.base import Problem
from tests._helpers import array


class _Shifted(Problem):
    """``sum((x - 1)^2)`` with optional bounds."""

    def __init__(self, xp, n=2, bounds=(None, None)):
        self._xp, self._n, self._bounds = xp, n, bounds

    @property
    def n_vars(self) -> int:
        return self._n

    def bounds(self):
        return self._bounds

    def objective(self, x):
        return self._xp.sum((x - 1.0) * (x - 1.0))

    def gradient(self, x):
        return 2.0 * (x - 1.0)


def _x0(xp, n=2):
    return array(xp, [0.0] * n)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"options": {"max_iter": 10}}, "options must be an ipax.Options"),
        ({"warm_start": {"y_eq": None}}, "warm_start must be an ipax.WarmStart"),
        ({"callback": "print"}, "callback must be callable"),
    ],
    ids=["options-dict", "warm-start-dict", "callback-string"],
)
def test_wrong_argument_types_raise_type_error(namespace, kwargs, match):
    with pytest.raises(TypeError, match=match):
        ipax.solve(_Shifted(namespace), _x0(namespace), **kwargs)


def test_a_start_point_that_is_not_an_array_raises_type_error():
    with pytest.raises(TypeError, match="x0 must be an array"):
        ipax.solve(_Shifted(None), [0.0, 0.0])


def test_a_start_point_of_the_wrong_length_is_rejected(namespace):
    with pytest.raises(
        ipax.ProblemDefinitionError, match=r"x0 has 3 entries.*n_vars = 2"
    ):
        ipax.solve(_Shifted(namespace), _x0(namespace, 3))


def test_a_start_point_that_is_not_1d_is_rejected(namespace):
    x0 = array(namespace, [[0.0, 0.0]])
    with pytest.raises(ipax.ProblemDefinitionError, match="x0 must be 1-D"):
        ipax.solve(_Shifted(namespace), x0)


@pytest.mark.parametrize("side", [0, 1], ids=["lower", "upper"])
def test_bounds_of_the_wrong_shape_are_rejected(namespace, side):
    bounds = [None, None]
    bounds[side] = array(namespace, [0.0, 0.0, 0.0])
    problem = _Shifted(namespace, bounds=tuple(bounds))
    name = "lower" if side == 0 else "upper"
    with pytest.raises(ipax.ProblemDefinitionError, match=f"{name} bound has shape"):
        ipax.solve(problem, _x0(namespace))


def test_n_iter_counts_iterations_performed(namespace):
    result = ipax.solve(_Shifted(namespace), _x0(namespace))
    assert result.history[0].iteration == 0  # the start point
    assert result.n_iter == result.history[-1].iteration
    assert result.n_iter == len(result.history) - 1


def test_n_iter_is_zero_when_the_solve_exits_at_the_start(namespace):
    bounds = (array(namespace, [1.0, 1.0]), array(namespace, [0.0, 0.0]))
    result = ipax.solve(_Shifted(namespace, bounds=bounds), _x0(namespace))
    assert result.status is ipax.Status.INFEASIBLE
    assert result.n_iter == 0


def test_as_operator_points_scipy_sparse_users_to_csr_operator():
    sp = pytest.importorskip("scipy.sparse")
    import numpy as np

    with pytest.raises(
        TypeError, match=r"CSROperator\(A\.indptr, A\.indices, A\.data, A\.shape\)"
    ):
        as_operator(sp.csr_matrix(np.eye(3)))


def test_as_operator_rejects_non_array_objects():
    with pytest.raises(TypeError, match="LinearOperator"):
        as_operator(object())


def test_matrix_free_jacobian_without_rmatvec_says_what_to_pass(namespace):
    op = MatrixFreeJacobian((2, 3), matvec=lambda v: v[:2])
    with pytest.raises(NotImplementedError, match=r"rmatvec="):
        op.rmatvec(array(namespace, [1.0, 1.0]))


def test_a_problem_that_is_not_a_problem_raises_type_error(namespace):
    with pytest.raises(TypeError, match=r"problem must be an ipax\.Problem"):
        ipax.solve(lambda x: x, _x0(namespace))


def test_a_bound_that_is_not_an_array_is_rejected(namespace):
    problem = _Shifted(namespace, bounds=([0.0, 0.0], None))
    with pytest.raises(
        ipax.ProblemDefinitionError, match="lower bound must be an array"
    ):
        ipax.solve(problem, _x0(namespace))


def test_sparse_route_without_a_sparse_adapter_raises_unsupported_backend():
    xps = pytest.importorskip("array_api_strict")
    from ipax.backend.namespace import capabilities

    x0 = xps.asarray([0.0, 0.0])
    if capabilities(xps).has_sparse_adapter:
        pytest.skip("array_api_strict gained a sparse adapter")
    with pytest.raises(ipax.UnsupportedBackendError, match="sparse adapter"):
        ipax.solve(_Shifted(xps), x0, options=ipax.Options(linsolve="sparse"))
    with pytest.raises(RuntimeError):  # 0.13 code caught RuntimeError here
        ipax.solve(_Shifted(xps), x0, options=ipax.Options(linsolve="sparse"))
