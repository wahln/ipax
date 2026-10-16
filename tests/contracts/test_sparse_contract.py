"""Reusable contract battery for ``SparseDirectSolver`` adapters."""

from __future__ import annotations

import pytest

from ipax.linalg.solver import LinearSolveError
from tests._helpers import array, assert_allclose, implemented

pytestmark = pytest.mark.sparse


class SparseDirectSolverContract:
    """Mixin battery. Subclasses provide ``make_adapter(namespace)``."""

    implementation_reason = "sparse adapters"

    def make_adapter(self, namespace):  # pragma: no cover - overridden
        raise NotImplementedError

    def test_build_from_coo_triplets(self, namespace, tol):
        with implemented(self.implementation_reason):
            adapter = self.make_adapter(namespace)
            rows = namespace.asarray([0, 0, 1, 2])
            cols = namespace.asarray([0, 2, 1, 2])
            values = array(namespace, [2.0, -1.0, 3.0, 4.0])
            op = adapter.from_coo(rows, cols, values, shape=(3, 3))
            v = array(namespace, [1.0, 2.0, 3.0])
            actual = op.matvec(v)

        expected = array(namespace, [-1.0, 6.0, 12.0])
        assert_allclose(namespace, actual, expected, **tol)

    def test_factor_and_solve(self, namespace, tol):
        with implemented(self.implementation_reason):
            adapter = self.make_adapter(namespace)
            rows = namespace.asarray([0, 1, 1, 2])
            cols = namespace.asarray([0, 0, 1, 2])
            values = array(namespace, [2.0, 1.0, 3.0, 4.0])
            K = adapter.from_coo(rows, cols, values, shape=(3, 3))
            rhs = array(namespace, [2.0, 7.0, 8.0])
            solver = adapter.solver()
            solver.factor(K)
            actual = solver.solve(rhs)

        assert_allclose(namespace, actual, array(namespace, [1.0, 2.0, 2.0]), **tol)

    def test_reports_inertia_when_available(self, namespace):
        with implemented("inertia reporting"):
            adapter = self.make_adapter(namespace)
            rows = namespace.asarray([0, 1])
            cols = namespace.asarray([0, 1])
            values = array(namespace, [1.0, -1.0])
            K = adapter.from_coo(rows, cols, values, shape=(2, 2))
            solver = adapter.solver(require_inertia=True)
            solver.factor(K)

        assert solver.inertia == (1, 1, 0)

    # -- failure contract: numerical failures are signalled, not raised raw --
    #
    # The IPM escalates δ_w on ``LinearSolveError``, on a non-finite solution,
    # or on a reported inertia that misses its target (a zero eigenvalue
    # always does); a backend-native exception escaping the adapter would
    # abort the run instead (Wächter & Biegler 2006, §3.1). Adapters translate
    # their backend's failures at this boundary.

    @pytest.mark.parametrize("symmetric", [True, None], ids=["symmetric", "general"])
    def test_singular_matrix_is_classified(self, namespace, symmetric):
        with implemented(self.implementation_reason):
            adapter = self.make_adapter(namespace)
            K = adapter.from_coo(
                namespace.asarray([0, 0, 1, 1]),
                namespace.asarray([0, 1, 0, 1]),
                array(namespace, [1.0, 1.0, 1.0, 1.0]),
                shape=(2, 2),
                symmetric=symmetric,
            )
            solver = adapter.solver()
            try:
                solver.factor(K)
                x = solver.solve(array(namespace, [1.0, 2.0]))
            except LinearSolveError:
                return

        inertia_fn = getattr(solver, "inertia_or_none", None)
        inertia = inertia_fn() if inertia_fn is not None else None
        # An LDLᵀ may complete a singular factorization (Feral does, with a
        # finite but meaningless solution); then its inertia must say so.
        singular_inertia = inertia is not None and inertia[2] > 0
        assert singular_inertia or not bool(namespace.all(namespace.isfinite(x)))

    @pytest.mark.parametrize("symmetric", [True, None], ids=["symmetric", "general"])
    def test_non_finite_matrix_raises_linear_solve_error(self, namespace, symmetric):
        with implemented(self.implementation_reason):
            adapter = self.make_adapter(namespace)
            K = adapter.from_coo(
                namespace.asarray([0, 1]),
                namespace.asarray([0, 1]),
                array(namespace, [float("nan"), 1.0]),
                shape=(2, 2),
                symmetric=symmetric,
            )
            solver = adapter.solver()
            with pytest.raises(LinearSolveError):
                solver.factor(K)
                solver.solve(array(namespace, [1.0, 2.0]))
