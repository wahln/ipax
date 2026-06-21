"""Reusable contract battery for ``LinearOperator`` implementations."""

from __future__ import annotations

from tests._helpers import array, assert_allclose, implemented, transpose


class LinearOperatorContract:
    """Mixin battery. Subclasses provide ``make_operator`` and ``make_dense``."""

    implementation_reason = "operators"

    def make_operator(self, namespace):  # pragma: no cover - overridden
        raise NotImplementedError

    def make_dense(self, namespace):  # pragma: no cover - overridden
        raise NotImplementedError

    def test_shape_consistency(self, namespace):
        with implemented(self.implementation_reason):
            op = self.make_operator(namespace)
            dense = self.make_dense(namespace)

        assert op.shape == dense.shape

    def test_matvec_matches_dense(self, namespace, tol):
        with implemented(self.implementation_reason):
            op = self.make_operator(namespace)
            dense = self.make_dense(namespace)
            v = array(namespace, [1.0 + idx for idx in range(op.shape[1])])
            actual = op.matvec(v)
            expected = namespace.matmul(dense, v)

        assert_allclose(namespace, actual, expected, **tol)

    def test_adjoint_identity(self, namespace, tol):
        with implemented(self.implementation_reason):
            op = self.make_operator(namespace)
            v = array(namespace, [1.0 + idx for idx in range(op.shape[1])])
            w = array(namespace, [2.0 + idx for idx in range(op.shape[0])])
            Av = op.matvec(v)
            Atw = op.rmatvec(w)

        left = namespace.sum(Av * w)
        right = namespace.sum(v * Atw)
        assert_allclose(namespace, left, right, **tol)

    def test_matmat_matches_dense(self, namespace, tol):
        with implemented(self.implementation_reason):
            op = self.make_operator(namespace)
            dense = self.make_dense(namespace)
            V = transpose(
                namespace,
                array(
                    namespace,
                    [
                        [1.0 + row + col for row in range(op.shape[1])]
                        for col in range(2)
                    ],
                ),
            )
            actual = op.matmat(V)
            expected = namespace.matmul(dense, V)

        assert_allclose(namespace, actual, expected, **tol)
