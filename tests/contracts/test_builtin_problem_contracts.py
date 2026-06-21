"""Contract coverage for built-in and oracle ``Problem`` implementations."""

from __future__ import annotations

from ipax.problem.function import FunctionProblem, LinearProblem, QuadraticProblem
from ipax.testing.problems import (
    BoundConstrainedQP,
    EqualityConstrainedQP,
    UnconstrainedQuadratic,
)
from tests._helpers import array, implemented
from tests.contracts.test_problem_contract import ProblemContract


class TestUnconstrainedQuadraticOracle(ProblemContract):
    def make_problem(self, namespace):
        Q = array(namespace, [[4.0, 1.0], [1.0, 3.0]])
        b = array(namespace, [1.0, 2.0])
        problem = UnconstrainedQuadratic(Q, b, namespace)
        return problem, array(namespace, [0.25, -0.75])


class TestBoundConstrainedQPOracle(ProblemContract):
    def make_problem(self, namespace):
        return BoundConstrainedQP(namespace), array(namespace, [0.2, 0.8])


class TestEqualityConstrainedQPOracle(ProblemContract):
    def make_problem(self, namespace):
        return EqualityConstrainedQP(namespace), array(namespace, [0.25, 0.75])


class TestFunctionProblem(ProblemContract):
    def make_problem(self, namespace):
        def objective(x):
            return namespace.sum((x - array(namespace, [1.0, -2.0])) ** 2)

        def gradient(x):
            return 2.0 * (x - array(namespace, [1.0, -2.0]))

        with implemented("FunctionProblem"):
            problem = FunctionProblem(2, objective, gradient=gradient)
        return problem, array(namespace, [0.5, 1.0])


class TestQuadraticProblem(ProblemContract):
    def make_problem(self, namespace):
        with implemented("QuadraticProblem"):
            problem = QuadraticProblem(
                array(namespace, [[2.0, 0.0], [0.0, 4.0]]),
                array(namespace, [-1.0, 3.0]),
            )
        return problem, array(namespace, [0.5, -0.5])


class TestLinearProblem(ProblemContract):
    def make_problem(self, namespace):
        with implemented("LinearProblem"):
            problem = LinearProblem(array(namespace, [1.0, -2.0]))
        return problem, array(namespace, [0.5, -0.5])
