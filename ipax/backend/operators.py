# Copyright 2026 Niklas Wahl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The ``LinearOperator`` hierarchy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

from ipax.backend.namespace import array_namespace

if TYPE_CHECKING:
    from ipax.typing import Array


class LinearOperator(ABC):
    """Backend-agnostic linear map exposing ``matvec`` and optional adjoints."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]: ...

    @abstractmethod
    def matvec(self, v: Array) -> Array:
        """Compute ``A @ v``."""

    def rmatvec(self, v: Array) -> Array:
        """Compute ``A.T @ v``. Defaults to unsupported."""
        raise NotImplementedError

    def matmat(self, V: Array) -> Array:
        """Compute ``A @ V`` by applying ``matvec`` column-wise."""
        xp = array_namespace(V)
        columns = tuple(self.matvec(V[:, idx]) for idx in range(int(V.shape[1])))
        return xp.stack(columns, axis=1)

    def rmatmat(self, V: Array) -> Array:
        """Compute ``A.T @ V`` by applying ``rmatvec`` column-wise."""
        xp = array_namespace(V)
        columns = tuple(self.rmatvec(V[:, idx]) for idx in range(int(V.shape[1])))
        return xp.stack(columns, axis=1)

    def dense_structured_solve(self, rhs: Array) -> Array:
        """Solve ``A x = rhs`` exactly by exploiting special operator structure.

        Optional capability used by :class:`~ipax.linalg.dense.DenseSolver` to skip
        the dense ``n × n`` materialization when the operator has a known
        structured inverse (e.g. an L-BFGS diagonal-plus-low-rank condensed block
        solved by the Woodbury identity, or the equality saddle's Schur
        complement). Defaults to unavailable; the dense solver then falls back to
        materializing and factoring the operator.
        """
        del rhs
        raise NotImplementedError("operator does not expose a structured dense solve")

    def dense_matrix(self, like: Array | None = None) -> Array:
        """Return an explicit dense matrix when the operator already has one.

        Optional capability for dense direct solves. Unlike ``matmat(I)``, this
        lets explicit operators hand back or assemble their matrix without an
        identity probe and the extra kernels it causes on device backends. Operators
        without explicit dense structure leave the default unavailable. Operators
        with no stored array (e.g. :class:`Identity`) may use ``like`` as a
        dtype/backend template.

        The result may alias the operator's backing array (e.g. :class:`Dense`
        returns it without copying), so callers must treat it as **read-only** —
        an in-place mutation would corrupt the operator.
        """
        del like
        raise NotImplementedError("operator does not expose a dense matrix")

    def diagonal(self, like: Array | None = None) -> Array:
        """Return the main diagonal as a rank-1 array.

        Optional capability: only operators that can produce it *cheaply* (i.e.
        without ``n`` matvecs) override this. The Jacobi preconditioner of the
        matrix-free Krylov solver consumes it when available and silently falls
        back to no preconditioning otherwise, so a missing diagonal is never an
        error. Operators that do not carry an array namespace internally may use
        ``like`` as a dtype/backend template.
        """
        del like
        raise NotImplementedError("operator does not expose a cheap diagonal")

    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        """Emit COO structure ``(rows, cols, values, shape)`` as Array-API vectors.

        Optional capability and the mechanism behind invariant #4: operators with
        *explicit* structure override this so a sparse-direct adapter can assemble
        and factor the actual matrix while the core stays backend-agnostic (the
        core emits index/value vectors; the adapter builds the sparse object).
        Duplicate ``(row, col)`` entries are summed by the adapter, so block
        assemblies may emit overlapping diagonals freely.

        Matrix-free / product operators (``LBFGSOperator``, ``MatrixFreeJacobian``,
        :class:`Composite`) cannot express their structure as triplets and leave
        the default, which signals that the sparse-direct route is unavailable for
        the system they appear in. Operators with no stored array (e.g.
        :class:`Identity`) may use ``like`` as a dtype/backend template.
        """
        del like
        raise NotImplementedError("operator does not expose sparse COO structure")

    def coo_pattern_signature(self) -> object | None:
        """Stable identity for the row/column pattern emitted by :meth:`to_coo`.

        Sparse-direct adapters may use this to split structure from numeric
        values without comparing device index arrays. Return ``None`` unless the
        operator can guarantee that its COO row/column vectors are determined by
        structural metadata rather than by current numeric values. This must be
        conservative: a value-dependent exact Hessian that drops zero entries
        should leave the default so the sparse solver reanalyzes.
        """
        return None

    def gram_diagonal(self, weights: Array) -> Array:
        """Return ``diag(Aᵀ diag(weights) A)`` — the weighted column energies.

        Entry ``k`` is ``Σ_i weights_i · A_ik²``. This is the inequality term
        ``diag(∇gᵀ Σ_s ∇g)`` of the condensed Newton matrix (Breedveld 2017,
        eq. 18), needed to build the Jacobi preconditioner matrix-free. Optional,
        like :meth:`diagonal`: only operators that can produce it without ``n``
        matvecs override it.
        """
        raise NotImplementedError("operator does not expose a cheap Gram diagonal")

    def row_gram_diagonal(self, weights: Array) -> Array:
        """Return ``diag(A diag(weights) Aᵀ)`` — the weighted *row* energies.

        Entry ``j`` is ``Σ_k weights_k · A_jk²`` (length = number of rows). This is
        the dual transpose of :meth:`gram_diagonal` and supplies the approximate
        Schur-complement diagonal ``diag(∇c diag(N)⁻¹ ∇cᵀ)`` of the equality
        saddle, used to build an SPD block-diagonal preconditioner for MINRES.
        Optional, like :meth:`gram_diagonal`.
        """
        raise NotImplementedError("operator does not expose a cheap row-Gram diagonal")

    def row_inf_norms(self, like: Array | None = None) -> Array:
        """Return ``max(abs(A), axis=1)`` without probing rows individually.

        This optional capability is used by gradient-based problem scaling.
        Matrix-free operators should leave it unavailable unless they can
        provide all row norms in one application-specific operation.
        """
        del like
        raise NotImplementedError("operator does not expose cheap row infinity norms")

    def spd_preconditioner_diagonal(self) -> Array:
        """Return a strictly positive diagonal of an SPD preconditioner.

        For an indefinite operator (the equality saddle), ``diagonal`` is itself
        indefinite and unusable for a symmetric preconditioner; this instead
        returns the diagonal of an SPD *block* preconditioner (PD primal block,
        approximate-Schur dual block). The Krylov solver applies it to MINRES by
        symmetric scaling. Optional; defaults to "not available".
        """
        raise NotImplementedError("operator does not expose an SPD preconditioner")

    def symmetry_hint(self) -> bool | None:
        """Declare ``A == Aᵀ`` when known cheaply by construction; ``None`` if not.

        Optional capability mirroring :meth:`preferred_krylov_method`: an operator
        assembled to be symmetric (the KKT condensed/saddle blocks, whose ``C``/
        ``Cᵀ`` off-diagonals share one value array) returns ``True`` so the
        sparse-direct route can skip the per-iteration O(nnz) ``A − Aᵀ`` numerical
        symmetry test. ``None`` means "unknown — test numerically", preserving the
        default behavior for generic operators.
        """
        return None

    def preferred_krylov_method(self) -> Literal["cg", "minres"] | None:
        """Return the Krylov method this operator should use, when constrained.

        Most operators leave this unset so the solver follows user options. A
        symmetric indefinite saddle can return ``"minres"`` to avoid first trying
        CG and discovering indefiniteness through a failed curvature test.
        """
        return None

    def lbfgs_inverse_apply(self) -> Callable[[Array], Array]:
        """Return an L-BFGS-aware approximate inverse ``M⁻¹ ≈ K⁻¹`` (SPD).

        For the condensed Newton operator whose Hessian block is a compact L-BFGS
        approximation ``B = ξI − U M⁻¹ Uᵀ`` (§4.3), the dominant structure is
        "diagonal minus low-rank", which the Sherman–Morrison–Woodbury identity
        inverts in ``O(n·m)``. The result is an SPD preconditioner exact up to the
        off-diagonal of the inequality Gram term, so CG/GMRES converge in very few
        iterations. Optional; defaults to "not available".
        """
        raise NotImplementedError("operator has no L-BFGS-aware preconditioner")

    def __matmul__(self, v: Array) -> Array:
        return self.matvec(v)


class Dense(LinearOperator):
    """Wrap an Array-API matrix."""

    def __init__(self, A: Array) -> None:
        if len(A.shape) != 2:
            raise ValueError("Dense operator requires a rank-2 array")
        self._A = A

    @property
    def shape(self) -> tuple[int, int]:
        return int(self._A.shape[0]), int(self._A.shape[1])

    def matvec(self, v: Array) -> Array:
        xp = array_namespace(self._A, v)
        return xp.matmul(self._A, v)

    def rmatvec(self, v: Array) -> Array:
        xp = array_namespace(self._A, v)
        return xp.matmul(xp.permute_dims(self._A, (1, 0)), v)

    def matmat(self, V: Array) -> Array:
        xp = array_namespace(self._A, V)
        return xp.matmul(self._A, V)

    def rmatmat(self, V: Array) -> Array:
        xp = array_namespace(self._A, V)
        return xp.matmul(xp.permute_dims(self._A, (1, 0)), V)

    def dense_matrix(self, like: Array | None = None) -> Array:
        # Returns the backing array *by reference* (no copy): callers — e.g.
        # DenseSolver, which caches it — must treat the result as read-only, as
        # an in-place mutation would corrupt this operator.
        del like
        return self._A

    def diagonal(self, like: Array | None = None) -> Array:
        del like
        xp = array_namespace(self._A)
        return xp.linalg.diagonal(self._A)

    def gram_diagonal(self, weights: Array) -> Array:
        xp = array_namespace(self._A, weights)
        return xp.sum(xp.expand_dims(weights, axis=1) * self._A * self._A, axis=0)

    def row_gram_diagonal(self, weights: Array) -> Array:
        xp = array_namespace(self._A, weights)
        return xp.sum(self._A * self._A * xp.expand_dims(weights, axis=0), axis=1)

    def row_inf_norms(self, like: Array | None = None) -> Array:
        del like
        xp = array_namespace(self._A)
        m, n = self.shape
        if n == 0:
            return xp.zeros((m,), dtype=self._A.dtype)
        return xp.max(xp.abs(self._A), axis=1)

    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        del like
        xp = array_namespace(self._A)
        m, n = self.shape
        rows = xp.reshape(
            xp.broadcast_to(xp.expand_dims(xp.arange(m), axis=1), (m, n)), (m * n,)
        )
        cols = xp.reshape(
            xp.broadcast_to(xp.expand_dims(xp.arange(n), axis=0), (m, n)), (m * n,)
        )
        values = xp.reshape(self._A, (m * n,))
        return rows, cols, values, (m, n)

    def coo_pattern_signature(self) -> object:
        return ("dense", self.shape)


class Diagonal(LinearOperator):
    """Diagonal operator from a vector ``d``."""

    def __init__(self, d: Array) -> None:
        if len(d.shape) != 1:
            raise ValueError("Diagonal operator requires a rank-1 array")
        self._d = d

    @property
    def shape(self) -> tuple[int, int]:
        n = int(self._d.shape[0])
        return n, n

    def matvec(self, v: Array) -> Array:
        return self._d * v

    def rmatvec(self, v: Array) -> Array:
        return self.matvec(v)

    def matmat(self, V: Array) -> Array:
        xp = array_namespace(self._d, V)
        return xp.expand_dims(self._d, axis=1) * V

    def rmatmat(self, V: Array) -> Array:
        return self.matmat(V)

    def dense_matrix(self, like: Array | None = None) -> Array:
        del like
        xp = array_namespace(self._d)
        n = int(self._d.shape[0])
        return xp.eye(n, dtype=self._d.dtype) * self._d

    def diagonal(self, like: Array | None = None) -> Array:
        del like
        return self._d

    def gram_diagonal(self, weights: Array) -> Array:
        # A = diag(d) ⇒ diag(Aᵀ W A)_k = d_k² · weights_k.
        return self._d * self._d * weights

    def row_inf_norms(self, like: Array | None = None) -> Array:
        del like
        xp = array_namespace(self._d)
        return xp.abs(self._d)

    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        del like
        xp = array_namespace(self._d)
        n = int(self._d.shape[0])
        idx = xp.arange(n)
        return idx, idx, self._d, (n, n)

    def coo_pattern_signature(self) -> object:
        return ("diagonal", self.shape)


class Identity(LinearOperator):
    """Identity operator ``I_n``."""

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError("Identity dimension must be non-negative")
        self._n = n

    @property
    def shape(self) -> tuple[int, int]:
        return self._n, self._n

    def matvec(self, v: Array) -> Array:
        return v

    def rmatvec(self, v: Array) -> Array:
        return v

    def matmat(self, V: Array) -> Array:
        return V

    def rmatmat(self, V: Array) -> Array:
        return V

    def dense_matrix(self, like: Array | None = None) -> Array:
        if like is None:
            raise NotImplementedError("Identity dense matrix requires a template array")
        xp = array_namespace(like)
        return xp.eye(self._n, dtype=like.dtype)

    def diagonal(self, like: Array | None = None) -> Array:
        if like is None:
            raise NotImplementedError("Identity diagonal requires a template array")
        xp = array_namespace(like)
        return xp.ones((self._n,), dtype=like.dtype)

    def gram_diagonal(self, weights: Array) -> Array:
        # A = I ⇒ diag(Aᵀ W A) = weights.
        return weights

    def row_inf_norms(self, like: Array | None = None) -> Array:
        if like is None:
            raise NotImplementedError("Identity row norms require a template array")
        xp = array_namespace(like)
        return xp.ones((self._n,), dtype=like.dtype)

    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        if like is None:
            raise NotImplementedError("Identity COO requires a template array")
        xp = array_namespace(like)
        idx = xp.arange(self._n)
        ones = xp.ones((self._n,), dtype=like.dtype)
        return idx, idx, ones, (self._n, self._n)

    def coo_pattern_signature(self) -> object:
        return ("identity", self.shape)


class LowRank(LinearOperator):
    """Operator ``U @ V.T``."""

    def __init__(self, U: Array, V: Array) -> None:
        if len(U.shape) != 2 or len(V.shape) != 2:
            raise ValueError("LowRank factors must be rank-2 arrays")
        if U.shape[1] != V.shape[1]:
            raise ValueError("LowRank factors must have the same rank dimension")
        self._U = U
        self._V = V

    @property
    def shape(self) -> tuple[int, int]:
        return int(self._U.shape[0]), int(self._V.shape[0])

    def matvec(self, v: Array) -> Array:
        xp = array_namespace(self._U, self._V, v)
        return xp.matmul(
            self._U,
            xp.matmul(xp.permute_dims(self._V, (1, 0)), v),
        )

    def rmatvec(self, v: Array) -> Array:
        xp = array_namespace(self._U, self._V, v)
        return xp.matmul(
            self._V,
            xp.matmul(xp.permute_dims(self._U, (1, 0)), v),
        )

    def matmat(self, V: Array) -> Array:
        xp = array_namespace(self._U, self._V, V)
        return xp.matmul(
            self._U,
            xp.matmul(xp.permute_dims(self._V, (1, 0)), V),
        )

    def rmatmat(self, V: Array) -> Array:
        xp = array_namespace(self._U, self._V, V)
        return xp.matmul(
            self._V,
            xp.matmul(xp.permute_dims(self._U, (1, 0)), V),
        )

    def dense_matrix(self, like: Array | None = None) -> Array:
        del like
        xp = array_namespace(self._U, self._V)
        return xp.matmul(self._U, xp.permute_dims(self._V, (1, 0)))


class MatrixFreeJacobian(LinearOperator):
    """User/autodiff callbacks giving ``Jv`` and optionally ``J.Tv``."""

    def __init__(
        self,
        shape: tuple[int, int],
        matvec: Callable[[Array], Array],
        rmatvec: Callable[[Array], Array] | None = None,
        row_inf_norms: Callable[[], Array] | None = None,
    ) -> None:
        if shape[0] < 0 or shape[1] < 0:
            raise ValueError("operator dimensions must be non-negative")
        self._shape = shape
        self._matvec = matvec
        self._rmatvec = rmatvec
        self._row_inf_norms = row_inf_norms

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def matvec(self, v: Array) -> Array:
        return self._matvec(v)

    def rmatvec(self, v: Array) -> Array:
        if self._rmatvec is None:
            raise NotImplementedError("matrix-free adjoint callback is not available")
        return self._rmatvec(v)

    def row_inf_norms(self, like: Array | None = None) -> Array:
        del like
        if self._row_inf_norms is None:
            raise NotImplementedError("matrix-free row norms are not available")
        return self._row_inf_norms()


class Composite(LinearOperator):
    """Lazy product composition, e.g. ``J.T @ Sigma @ J``."""

    def __init__(self, *terms: LinearOperator) -> None:
        if not terms:
            raise ValueError("Composite requires at least one operator")
        for left, right in pairwise(terms):
            if left.shape[1] != right.shape[0]:
                raise ValueError("operator shapes are not composable")
        self._terms = terms

    @property
    def shape(self) -> tuple[int, int]:
        return self._terms[0].shape[0], self._terms[-1].shape[1]

    def matvec(self, v: Array) -> Array:
        result = v
        for term in reversed(self._terms):
            result = term.matvec(result)
        return result

    def rmatvec(self, v: Array) -> Array:
        result = v
        for term in self._terms:
            result = term.rmatvec(result)
        return result

    def matmat(self, V: Array) -> Array:
        result = V
        for term in reversed(self._terms):
            result = term.matmat(result)
        return result

    def rmatmat(self, V: Array) -> Array:
        result = V
        for term in self._terms:
            result = term.rmatmat(result)
        return result

    def dense_matrix(self, like: Array | None = None) -> Array:
        result = self._terms[-1].dense_matrix(like)
        for term in reversed(self._terms[:-1]):
            left = term.dense_matrix(result)
            xp = array_namespace(left, result)
            result = xp.matmul(left, result)
        return result


class VStack(LinearOperator):
    """Vertical stack of operators sharing the variable (column) dimension.

    The adjoint sums each block's contribution; the optional structure-exposing
    capabilities (``to_coo``, ``row_gram_diagonal``, ``row_inf_norms``) propagate
    block-wise so a stacked Jacobian keeps the sparse-direct, Krylov-preconditioner
    and gradient-scaling routes available when every block supports them.
    """

    def __init__(self, ops: tuple[LinearOperator, ...]) -> None:
        if not ops:
            raise ValueError("VStack requires at least one operator")
        self._n = ops[0].shape[1]
        for op in ops:
            if op.shape[1] != self._n:
                raise ValueError("VStack operators must share the column dimension")
        self._ops = ops
        self._rows = tuple(op.shape[0] for op in ops)

    @property
    def shape(self) -> tuple[int, int]:
        return sum(self._rows), self._n

    def matvec(self, v: Array) -> Array:
        xp = array_namespace(v)
        return xp.concat(tuple(op.matvec(v) for op in self._ops))

    def rmatvec(self, v: Array) -> Array:
        xp = array_namespace(v)
        result = None
        offset = 0
        for op, rows in zip(self._ops, self._rows, strict=True):
            piece = op.rmatvec(v[offset : offset + rows])
            result = piece if result is None else result + piece
            offset += rows
        assert result is not None
        return xp.asarray(result)

    def matmat(self, V: Array) -> Array:
        xp = array_namespace(V)
        return xp.concat(tuple(op.matmat(V) for op in self._ops), axis=0)

    def rmatmat(self, V: Array) -> Array:
        xp = array_namespace(V)
        result = None
        offset = 0
        for op, rows in zip(self._ops, self._rows, strict=True):
            piece = op.rmatmat(V[offset : offset + rows, :])
            result = piece if result is None else result + piece
            offset += rows
        assert result is not None
        return xp.asarray(result)

    def dense_matrix(self, like: Array | None = None) -> Array:
        xp = None
        template = like
        parts: list[Array] = []
        for op in self._ops:
            piece = op.dense_matrix(template)
            if xp is None:
                xp = array_namespace(piece)
            if template is None:
                template = piece
            parts.append(piece)
        assert xp is not None
        return xp.concat(tuple(parts), axis=0)

    def row_gram_diagonal(self, weights: Array) -> Array:
        # Rows are stacked, so the weighted row energies concatenate. Propagates
        # NotImplementedError if any block cannot supply them cheaply.
        xp = array_namespace(weights)
        return xp.concat(tuple(op.row_gram_diagonal(weights) for op in self._ops))

    def gram_diagonal(self, weights: Array) -> Array:
        # Row weights are stacked with the blocks; split by row range and sum each
        # block's column-energy contribution.
        result = None
        offset = 0
        for op, rows in zip(self._ops, self._rows, strict=True):
            piece = op.gram_diagonal(weights[offset : offset + rows])
            result = piece if result is None else result + piece
            offset += rows
        assert result is not None
        xp = array_namespace(result)
        return xp.asarray(result)

    def row_inf_norms(self, like: Array | None = None) -> Array:
        # Stacked rows ⇒ concatenate each block's row norms (used by scaling).
        xp = None
        parts: list[Array] = []
        for op in self._ops:
            piece = op.row_inf_norms(like)
            if xp is None:
                xp = array_namespace(piece)
            parts.append(piece)
        assert xp is not None
        return xp.concat(tuple(parts))

    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        # Vertically stacked blocks ⇒ concatenate triplets with row offsets.
        del like
        rows_parts: list[Array] = []
        cols_parts: list[Array] = []
        vals_parts: list[Array] = []
        offset = 0
        xp = None
        for op, n_rows in zip(self._ops, self._rows, strict=True):
            r, c, v, _ = op.to_coo()
            if xp is None:
                xp = array_namespace(v)
            rows_parts.append(r + offset)
            cols_parts.append(c)
            vals_parts.append(v)
            offset += n_rows
        assert xp is not None
        return (
            xp.concat(tuple(rows_parts)),
            xp.concat(tuple(cols_parts)),
            xp.concat(tuple(vals_parts)),
            (offset, self._n),
        )

    def coo_pattern_signature(self) -> object | None:
        parts = tuple(op.coo_pattern_signature() for op in self._ops)
        if any(part is None for part in parts):
            return None
        return ("vstack", self.shape, parts)


def as_operator(obj: Array | LinearOperator) -> LinearOperator:
    """Normalize a rank-2 dense array or existing operator."""
    if isinstance(obj, LinearOperator):
        return obj
    if not hasattr(obj, "shape"):
        raise TypeError("expected an Array-API array or LinearOperator")
    if len(obj.shape) != 2:
        raise ValueError("only rank-2 dense arrays can be normalized as operators")
    return Dense(obj)


__all__ = [
    "Composite",
    "Dense",
    "Diagonal",
    "Identity",
    "LinearOperator",
    "LowRank",
    "MatrixFreeJacobian",
    "VStack",
    "as_operator",
]
