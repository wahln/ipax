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

    def coo_values(self, like: Array | None = None) -> Array:
        """Return only the COO value vector, in :meth:`to_coo`'s exact order.

        Optional fast path behind the values-only sparse refactor: in an
        interior-point solve the COO row/column pattern is fixed (the solver
        caches it keyed on :meth:`coo_pattern_signature`) while only the values
        move, so an operator that can recompute its values *without* rebuilding
        the index vectors overrides this. The default recomputes the full triplet
        and discards the indices — always correct, but forgoes the speedup. The
        returned order must match :meth:`to_coo` exactly (a shared contract test
        guards the two against drift).
        """
        return self.to_coo(like)[2]

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

    def gram(
        self,
        weights: Array,
        *,
        accumulate_dtype: str | None = None,
        hinted_only: bool = False,
    ) -> Array:
        """Return the full weighted Gram matrix ``Aᵀ diag(weights) A`` (dense ``n×n``).

        The matrix analogue of :meth:`gram_diagonal`: the condensed inequality term
        ``∇gᵀ Σ_s ∇g`` of the Newton system (Breedveld 2017, eq. 18). For a sparse
        ``A`` this is formed by sparse arithmetic — scale rows by ``weights`` then a
        sparse Gram product — yielding the small dense ``n×n`` result *without*
        densifying ``A`` to ``m×n`` first. That is the whole point at radiotherapy
        scale, where ``m ≫ n``: the dense condensed route would otherwise materialize
        a ``m×n`` Jacobian (gigabytes) to form an ``n×n`` matrix. Optional, like
        :meth:`gram_diagonal`; the dense KKT route falls back to densifying ``A`` when
        it is unavailable.

        ``accumulate_dtype`` (dtype *name*, e.g. ``"float32"``) requests the
        accumulation itself in reduced precision — the mixed-precision dense
        route (``DenseOptions.gram_dtype``); the *returned* matrix keeps the
        operator's native result dtype regardless. Best-effort: implementations
        may ignore the request (returning the exact Gram is always valid), and
        wrappers forwarding :meth:`gram` must forward it. Callers tolerate
        pre-keyword implementations (``TypeError``) by retrying without it.

        ``hinted_only`` restricts the reduction to the parts of the operator
        whose own data supports it (:meth:`gram_accumulate_dtype_hint`): a
        composite honors the request block by block, leaving
        full-precision blocks exact. That is what makes the reduction safe on
        a block assembled from mixed-precision sources. With
        ``hinted_only=False`` (an explicit user request) the reduction applies
        throughout, hint or not.
        """
        del accumulate_dtype, hinted_only
        raise NotImplementedError("operator does not expose a full weighted Gram")

    def gram_capable(self) -> bool:
        """Whether :meth:`gram` is expected to succeed (no densify fallback).

        A cheap *structural* probe — no Gram is formed — used by solver
        auto-selection to prefer the condensed normal-equations route for tall
        (``n ≪ m``) inequality Jacobians. Wrapper operators that forward
        :meth:`gram` to an inner operator must override this to forward the
        probe as well. Conservative: the default only reports the override.
        """
        return type(self).gram is not LinearOperator.gram

    def gram_accumulate_dtype_hint(self) -> str | None:
        """Dtype name the Gram may be *accumulated* in without losing data
        information, or ``None`` (no reduction opportunity — the default).

        The discovery half of ``DenseOptions(gram_dtype="auto")``: an operator
        whose stored values carry only float32 information — float32 storage,
        or float64 values that are exact float32 upcasts *declared as such* by
        their producer — answers ``"float32"``, and the mixed-precision dense
        route then accumulates reduced with iterative-refinement
        certification. Metadata only, never a value scan (a float64 matrix
        that merely happens to be float32-representable must NOT hint — silent
        heuristics would shift behavior across whole benchmark corpora).
        Wrappers forwarding :meth:`gram` forward this too. A *stack* reports
        reduced when **any** block does, because it honors the request per
        block (see ``hinted_only`` in :meth:`gram`) — its float64 blocks stay
        exact, so a block assembled from mixed-precision sources still gets
        the reduction where its data justifies it.
        """
        return None

    def gram_coo(self, weights: Array) -> tuple[Array, Array, Array, tuple[int, int]]:
        """``Aᵀ diag(weights) A`` as *sparse* COO triplets (invariant #4).

        The sparse sibling of :meth:`gram`, for the sparse condensed
        normal-equations route: when ``A`` has localized (banded/block) rows,
        the Gram stays sparse and an ``n×n`` sparse factorization of the
        condensed matrix beats both the dense route (O(n²) memory) and
        matrix-free Krylov. Contract: for a fixed operator, the returned
        *pattern* (rows/cols, canonical order) is identical across calls with
        different ``weights`` — only the values move — so sparse-direct
        callers may cache structure and symbolic analyses. Duplicate entries,
        if any, are summed by the consumer. Optional; on non-localized
        sparsity the result may be near-dense — route selection consults
        :meth:`gram_fill_estimate` before committing to this form.
        """
        del weights
        raise NotImplementedError("operator does not expose a sparse COO Gram")

    def gram_coo_capable(self) -> bool:
        """Whether :meth:`gram_coo` is expected to succeed.

        Structural probe like :meth:`gram_capable`; wrappers forwarding
        :meth:`gram_coo` must forward this too.
        """
        return type(self).gram_coo is not LinearOperator.gram_coo

    def gram_fill_estimate(self) -> float | None:
        """Estimated density of the Gram *pattern* ``nnz(AᵀA)/n²``, or ``None``.

        The cheap structural probe behind sparse normal-equations
        auto-selection: the route wins only when ``AᵀA`` stays sparse
        (localized/banded rows), and whether it does cannot be read off
        ``nnz(A)`` — scattered rows of the same density fill the Gram
        completely. Adapters estimate it from sampled column overlap without
        forming the Gram; ``None`` means unknown (no explicit sparse
        structure), which selection treats as a veto. Wrappers forwarding
        :meth:`gram_coo` must forward this too.
        """
        return None

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

    def lbfgs_block_preconditioner_apply(self) -> Callable[[Array], Array]:
        """Return an L-BFGS-aware block-diagonal preconditioner for a saddle.

        Only the equality saddle (which wraps a condensed block with an L-BFGS
        compact form) provides this: the block-diagonal ``diag(N⁻¹, S⁻¹)`` with the
        Woodbury ``N⁻¹`` on the (1,1) block and an approximate-Schur diagonal on the
        (2,2) block. Non-diagonal, so it is applied by GMRES rather than the MINRES
        diagonal-scaling path. Optional; defaults to "not available".
        """
        raise NotImplementedError("operator has no L-BFGS block preconditioner")

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

    def gram_accumulate_dtype_hint(self) -> str | None:
        xp = array_namespace(self._A)
        return "float32" if self._A.dtype == xp.float32 else None

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

    def coo_values(self, like: Array | None = None) -> Array:
        del like
        xp = array_namespace(self._A)
        m, n = self.shape
        return xp.reshape(self._A, (m * n,))

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

    def coo_values(self, like: Array | None = None) -> Array:
        del like
        return self._d

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

    def coo_values(self, like: Array | None = None) -> Array:
        if like is None:
            raise NotImplementedError("Identity COO values require a template array")
        xp = array_namespace(like)
        return xp.ones((self._n,), dtype=like.dtype)

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

    def gram(
        self,
        weights: Array,
        *,
        accumulate_dtype: str | None = None,
        hinted_only: bool = False,
    ) -> Array:
        # Jᵀ diag(w) J for J = [J1; …; Jk] is Σ_b Jbᵀ diag(w_b) Jb — the vertical
        # stack sums each block's Gram over its own row (weight) range. Propagates
        # NotImplementedError if any block cannot form its Gram.
        result = None
        offset = 0
        for op, rows in zip(self._ops, self._rows, strict=True):
            piece = op.gram(
                weights[offset : offset + rows],
                accumulate_dtype=accumulate_dtype,
                hinted_only=hinted_only,
            )
            result = piece if result is None else result + piece
            offset += rows
        assert result is not None
        xp = array_namespace(result)
        return xp.asarray(result)

    def gram_capable(self) -> bool:
        # The stacked Gram succeeds only when every block's does.
        return all(op.gram_capable() for op in self._ops)

    def gram_accumulate_dtype_hint(self) -> str | None:
        # Any block suffices, because ``gram(hinted_only=True)`` applies the
        # reduction per block: the float32-sourced blocks accumulate reduced
        # and the genuinely-float64 ones stay exact. That is strictly better
        # than the two alternatives — requiring unanimity forfeits the
        # reduction on a block that is overwhelmingly float32 (radiotherapy
        # VMAT plans are 96% float32 by nonzero, held back by one float64
        # constraint), and a size-weighted rule would silently reduce
        # declared-float64 data.
        for op in self._ops:
            hint = op.gram_accumulate_dtype_hint()
            if hint is not None:
                return hint
        return None

    def gram_coo(self, weights: Array) -> tuple[Array, Array, Array, tuple[int, int]]:
        # Σ_b Jbᵀ diag(w_b) Jb as concatenated n×n triplets: overlapping
        # entries across blocks are duplicates the consumer sums. Per-block
        # patterns are stable, so the concatenation is too.
        rows_parts: list[Array] = []
        cols_parts: list[Array] = []
        vals_parts: list[Array] = []
        offset = 0
        xp = None
        for op, n_rows in zip(self._ops, self._rows, strict=True):
            r, c, v, _ = op.gram_coo(weights[offset : offset + n_rows])
            if xp is None:
                xp = array_namespace(v)
            rows_parts.append(r)
            cols_parts.append(c)
            vals_parts.append(v)
            offset += n_rows
        assert xp is not None
        return (
            xp.concat(tuple(rows_parts)),
            xp.concat(tuple(cols_parts)),
            xp.concat(tuple(vals_parts)),
            (self._n, self._n),
        )

    def gram_coo_capable(self) -> bool:
        return all(op.gram_coo_capable() for op in self._ops)

    def gram_fill_estimate(self) -> float | None:
        # The stacked Gram pattern is the union of the blocks' patterns, so
        # the fill is bounded by the (capped) sum of the block fills — a safe
        # over-estimate for the sparse-NE gate. Unknown anywhere ⇒ unknown.
        total = 0.0
        for op in self._ops:
            estimate = op.gram_fill_estimate()
            if estimate is None:
                return None
            total += estimate
        return min(1.0, total)

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

    def coo_values(self, like: Array | None = None) -> Array:
        # Stacked rows ⇒ concatenate each block's values in block order, matching
        # to_coo without recomputing the row-offset index vectors.
        del like
        xp = None
        parts: list[Array] = []
        for op in self._ops:
            piece = op.coo_values()
            if xp is None:
                xp = array_namespace(piece)
            parts.append(piece)
        assert xp is not None
        return xp.concat(tuple(parts))

    def coo_pattern_signature(self) -> object | None:
        parts = tuple(op.coo_pattern_signature() for op in self._ops)
        if any(part is None for part in parts):
            return None
        return ("vstack", self.shape, parts)


class _SparseStructured(LinearOperator):
    """Backend-agnostic sparse operator carrying Array-API COO triplets.

    The public sparse operators (:class:`COOOperator`, :class:`CSROperator`,
    :class:`CSCOperator`) all reduce to this. It honors invariant #4 exactly: it
    stores only Array-API index/value vectors and *emits structure* through
    :meth:`to_coo`/:meth:`coo_values`, while the actual sparse linear algebra
    (``matvec``-family, ``diagonal``, Gram diagonals, row norms) is delegated to
    the per-backend adapter resolved from the value array's namespace — the core
    never builds or holds a concrete sparse matrix itself.

    Pattern reuse is opt-in. By default :meth:`coo_pattern_signature` is ``None``
    (conservative: the sparse-direct route re-analyzes every factorization). A
    caller whose sparsity pattern is fixed across a solve — the usual case, where
    only the numeric values move iteration to iteration — passes a stable
    ``pattern_key`` to unlock the symbolic-analysis and structure-cache fast paths.
    """

    def __init__(
        self,
        rows: Array,
        cols: Array,
        values: Array,
        shape: tuple[int, int],
        *,
        symmetric: bool | None = None,
        pattern_key: object | None = None,
        values_dtype_hint: str | None = None,
    ) -> None:
        if len(rows.shape) != 1 or len(cols.shape) != 1 or len(values.shape) != 1:
            raise ValueError("COO rows, cols, and values must be rank-1 arrays")
        if rows.shape[0] != values.shape[0] or cols.shape[0] != values.shape[0]:
            raise ValueError("COO rows, cols, and values must have equal length")
        if len(shape) != 2 or shape[0] < 0 or shape[1] < 0:
            raise ValueError("shape must be a non-negative (rows, cols) pair")
        self._rows = rows
        self._cols = cols
        self._values = values
        self._shape = (int(shape[0]), int(shape[1]))
        self._symmetric = symmetric
        self._pattern_key = pattern_key
        # Declared source precision of the values (see
        # ``gram_accumulate_dtype_hint``): a producer whose float64 values are
        # exact float32 upcasts (upcasting is lossless, so the reduced-data
        # cache recovers the source bits exactly) passes "float32" here.
        self._values_dtype_hint = values_dtype_hint
        # Lazily built, cached adapter operator for the heavy linear algebra.
        self._delegate: LinearOperator | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def gram_accumulate_dtype_hint(self) -> str | None:
        if self._values_dtype_hint is not None:
            return self._values_dtype_hint
        xp = array_namespace(self._values)
        return "float32" if self._values.dtype == xp.float32 else None

    def _adapter_op(self) -> LinearOperator:
        """Resolve (and cache) the backend adapter operator for sparse algebra."""
        if self._delegate is None:
            from typing import Any

            from ipax.backend.sparse import get_sparse_adapter

            xp = array_namespace(self._values)
            adapter: Any = get_sparse_adapter(xp)
            if adapter is None:
                raise NotImplementedError(
                    "sparse matrix algebra requires a backend sparse adapter; "
                    "install the sparse extra (e.g. ipax[sparse-cpu]) or use a "
                    "matrix-free operator"
                )
            self._delegate = adapter.from_coo(
                self._rows,
                self._cols,
                self._values,
                shape=self._shape,
                symmetric=self._symmetric,
                pattern_signature=self.coo_pattern_signature(),
            )
        return self._delegate

    # Structure emission — pure Array API, no adapter needed (invariant #4).
    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        del like
        return self._rows, self._cols, self._values, self._shape

    def coo_values(self, like: Array | None = None) -> Array:
        del like
        return self._values

    def coo_pattern_signature(self) -> object | None:
        if self._pattern_key is None:
            return None
        return ("sparse-structured", self._shape, self._symmetric, self._pattern_key)

    def symmetry_hint(self) -> bool | None:
        return self._symmetric

    # Sparse linear algebra — delegated to the per-backend adapter.
    def matvec(self, v: Array) -> Array:
        return self._adapter_op().matvec(v)

    def rmatvec(self, v: Array) -> Array:
        return self._adapter_op().rmatvec(v)

    def matmat(self, V: Array) -> Array:
        return self._adapter_op().matmat(V)

    def rmatmat(self, V: Array) -> Array:
        return self._adapter_op().rmatmat(V)

    def diagonal(self, like: Array | None = None) -> Array:
        return self._adapter_op().diagonal(like)

    def gram_diagonal(self, weights: Array) -> Array:
        return self._adapter_op().gram_diagonal(weights)

    def gram(
        self,
        weights: Array,
        *,
        accumulate_dtype: str | None = None,
        hinted_only: bool = False,
    ) -> Array:
        if (
            hinted_only
            and accumulate_dtype is not None
            and self.gram_accumulate_dtype_hint() != accumulate_dtype
        ):
            # This operator's data does not support the reduction; the caller
            # asked for it only where it does, so accumulate exactly.
            accumulate_dtype = None
        return self._adapter_op().gram(weights, accumulate_dtype=accumulate_dtype)

    def gram_capable(self) -> bool:
        # Capability lives in the backend adapter operator (e.g. scipy/cupy
        # implement ``gram``; torch/jax sparse do not yet). No adapter ⇒ the
        # matvec family itself would fail, so report incapable rather than raise.
        try:
            return self._adapter_op().gram_capable()
        except NotImplementedError:
            return False

    def gram_coo(self, weights: Array) -> tuple[Array, Array, Array, tuple[int, int]]:
        return self._adapter_op().gram_coo(weights)

    def gram_coo_capable(self) -> bool:
        try:
            return self._adapter_op().gram_coo_capable()
        except NotImplementedError:
            return False

    def gram_fill_estimate(self) -> float | None:
        try:
            return self._adapter_op().gram_fill_estimate()
        except NotImplementedError:
            return None

    def row_gram_diagonal(self, weights: Array) -> Array:
        return self._adapter_op().row_gram_diagonal(weights)

    def row_inf_norms(self, like: Array | None = None) -> Array:
        return self._adapter_op().row_inf_norms(like)

    def dense_matrix(self, like: Array | None = None) -> Array:
        # Densifies via the adapter (an identity probe): a convenience for the
        # dense route, not the intended fast path for a sparse operator.
        del like
        xp = array_namespace(self._values)
        n = self._shape[1]
        return self._adapter_op().matmat(xp.eye(n, dtype=self._values.dtype))


class COOOperator(_SparseStructured):
    """Sparse operator from Array-API COO triplets ``(rows, cols, values)``.

    The canonical way to hand the solver a sparse Jacobian or Hessian without a
    concrete sparse library in user code: the triplets are plain Array-API
    integer/float vectors in the caller's backend. Duplicate ``(row, col)``
    entries are summed (matching the adapter), so block assemblies may overlap
    freely. Pass ``symmetric=True`` for a symmetric block (e.g. a Hessian) to skip
    the adapter's numerical symmetry test, and ``pattern_key`` to declare the
    pattern fixed across a solve and unlock symbolic-analysis reuse.
    """


def _expand_indptr(xp: object, indptr: Array, nnz: int) -> Array:
    """Compressed ``indptr`` → the major-axis index of each stored entry.

    Entry ``e`` belongs to major index ``i`` where ``indptr[i] ≤ e < indptr[i+1]``;
    ``searchsorted(..., 'right') − 1`` recovers it in one pure Array-API pass (no
    scatter primitive, which the standard lacks).
    """
    arange = xp.arange(nnz)  # type: ignore[attr-defined]
    return xp.searchsorted(indptr, arange, side="right") - 1  # type: ignore[attr-defined]


class CSROperator(_SparseStructured):
    """Sparse operator from CSR arrays ``(indptr, indices, data)``.

    Ergonomic constructor for callers already holding a CSR matrix (the row
    pointer has length ``shape[0] + 1``). The compressed rows are expanded to COO
    coordinates once at construction; everything else is shared with
    :class:`COOOperator`. CSR/CSC carry no factorization-time speed advantage over
    COO here — the solver factorizes the *assembled* KKT matrix, not this block —
    so choose whichever matches the data you already have.
    """

    def __init__(
        self,
        indptr: Array,
        indices: Array,
        data: Array,
        shape: tuple[int, int],
        *,
        symmetric: bool | None = None,
        pattern_key: object | None = None,
        values_dtype_hint: str | None = None,
    ) -> None:
        if len(shape) != 2:
            raise ValueError("shape must be a (rows, cols) pair")
        if int(indptr.shape[0]) != int(shape[0]) + 1:
            raise ValueError("CSR indptr must have length shape[0] + 1")
        xp = array_namespace(data)
        rows = _expand_indptr(xp, indptr, int(data.shape[0]))
        super().__init__(
            rows,
            indices,
            data,
            shape,
            symmetric=symmetric,
            pattern_key=pattern_key,
            values_dtype_hint=values_dtype_hint,
        )


class CSCOperator(_SparseStructured):
    """Sparse operator from CSC arrays ``(indptr, indices, data)``.

    Ergonomic constructor for callers already holding a CSC matrix (the column
    pointer has length ``shape[1] + 1``). The compressed columns are expanded to
    COO coordinates once at construction; see :class:`CSROperator` on why the
    compressed format is a convenience, not a performance, choice.
    """

    def __init__(
        self,
        indptr: Array,
        indices: Array,
        data: Array,
        shape: tuple[int, int],
        *,
        symmetric: bool | None = None,
        pattern_key: object | None = None,
        values_dtype_hint: str | None = None,
    ) -> None:
        if len(shape) != 2:
            raise ValueError("shape must be a (rows, cols) pair")
        if int(indptr.shape[0]) != int(shape[1]) + 1:
            raise ValueError("CSC indptr must have length shape[1] + 1")
        xp = array_namespace(data)
        cols = _expand_indptr(xp, indptr, int(data.shape[0]))
        super().__init__(
            indices,
            cols,
            data,
            shape,
            symmetric=symmetric,
            pattern_key=pattern_key,
            values_dtype_hint=values_dtype_hint,
        )


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
    "COOOperator",
    "CSCOperator",
    "CSROperator",
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
