"""Contract coverage for the built-in ``LinearOperator`` classes."""

from __future__ import annotations

import pytest

from ipax.backend.operators import (
    Dense,
    Diagonal,
    Identity,
    LowRank,
    MatrixFreeJacobian,
)
from ipax.backend.sparse import get_sparse_adapter
from tests._helpers import array, float_dtype, transpose
from tests.contracts.test_operator_contract import LinearOperatorContract


class TestDenseOperator(LinearOperatorContract):
    def make_dense(self, namespace):
        return array(namespace, [[2.0, -1.0, 0.5], [0.0, 3.0, 4.0]])

    def make_operator(self, namespace):
        return Dense(self.make_dense(namespace))


class TestDiagonalOperator(LinearOperatorContract):
    def make_dense(self, namespace):
        return array(namespace, [[2.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, 4.0]])

    def make_operator(self, namespace):
        return Diagonal(array(namespace, [2.0, -3.0, 4.0]))


class TestIdentityOperator(LinearOperatorContract):
    def make_dense(self, namespace):
        dtype = float_dtype(namespace)
        if dtype is None:
            return namespace.eye(3)
        return namespace.eye(3, dtype=dtype)

    def make_operator(self, namespace):
        return Identity(3)


class TestLowRankOperator(LinearOperatorContract):
    def _factors(self, namespace):
        U = array(namespace, [[1.0, 2.0], [0.0, -1.0], [3.0, 0.5]])
        V = array(namespace, [[-1.0, 0.0], [2.0, 1.0], [0.5, 4.0]])
        return U, V

    def make_dense(self, namespace):
        U, V = self._factors(namespace)
        return namespace.matmul(U, transpose(namespace, V))

    def make_operator(self, namespace):
        U, V = self._factors(namespace)
        return LowRank(U, V)


class TestMatrixFreeJacobian(LinearOperatorContract):
    def make_dense(self, namespace):
        return array(namespace, [[1.0, 2.0, 0.0], [-1.0, 0.5, 3.0]])

    def make_operator(self, namespace):
        A = self.make_dense(namespace)
        return MatrixFreeJacobian(
            A.shape,
            matvec=lambda v: namespace.matmul(A, v),
            rmatvec=lambda v: namespace.matmul(transpose(namespace, A), v),
        )


@pytest.mark.sparse
class TestSparseOperator(LinearOperatorContract):
    # COO triplets for [[2, -1, 0.5], [0, 3, 4]] (the Dense battery's matrix).
    _rows = (0, 0, 0, 1, 1)
    _cols = (0, 1, 2, 1, 2)
    _vals = (2.0, -1.0, 0.5, 3.0, 4.0)

    def make_dense(self, namespace):
        return array(namespace, [[2.0, -1.0, 0.5], [0.0, 3.0, 4.0]])

    def make_operator(self, namespace):
        adapter = get_sparse_adapter(namespace)
        if adapter is None:
            pytest.skip(f"no sparse adapter for backend {namespace.__name__!r}")
        return adapter.from_coo(
            namespace.asarray(self._rows),
            namespace.asarray(self._cols),
            array(namespace, self._vals),
            shape=(2, 3),
        )
