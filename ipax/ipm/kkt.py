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

"""Assemble KKT systems behind ``LinearOperator`` interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ipax.backend.namespace import array_namespace
from ipax.backend.operators import Diagonal, Identity, LinearOperator

if TYPE_CHECKING:
    from collections.abc import Callable

    from ipax.linalg.regularize import RegularizationState
    from ipax.typing import Array


@dataclass(frozen=True, slots=True)
class _Border:
    """A symmetric border block appended to a KKT system, with **zero RHS**.

    Couples ``size`` new auxiliary variables to existing ones: the connection
    triplets ``(conn_rows, conn_cols, conn_values)`` give ``C`` (auxiliary-local
    row × existing-column), placed symmetrically as ``C`` and ``Cᵀ``; the inner
    triplets give the auxiliary diagonal block ``E``. Eliminating the auxiliaries
    contributes the Schur term ``−Cᵀ E⁻¹ C`` to the logical block. Two uses:

    - **Inequalities** (indefinite augmented route): ``C = ∇g``, ``E = −Σ_s⁻¹`` ⇒
      Schur term ``+∇gᵀ Σ_s ∇g`` — the condensed Gram term, *never formed densely*.
    - **Limited-memory Hessian**: ``C = Uᵀ``, ``E = M`` ⇒ Schur term
      ``−U M⁻¹ Uᵀ`` (IPOPT limited-memory; Byrd–Nocedal–Schnabel 1994).

    ``conn_cols`` index existing variables (the primal block ``[0, n)``), so a
    border composes unchanged when the logical block grows (e.g. the condensed
    block bordered into the equality saddle).
    """

    size: int
    conn_rows: Array
    conn_cols: Array
    conn_values: Array
    inner_rows: Array
    inner_cols: Array
    inner_values: Array


@dataclass(frozen=True, slots=True)
class _Assembly:
    """A logical sparse COO block plus zero-RHS symmetric :class:`_Border` blocks.

    The assembled matrix nests every border after the logical block; its Schur
    complement onto the logical block is the true KKT operator the driver solves
    (the condensed ``N`` / the equality saddle). The auxiliary variables have a
    zero right-hand side and are discarded, so :class:`SparseDirectSolver` just
    pads the RHS and truncates the solution across the size gap.
    """

    rows: Array
    cols: Array
    values: Array
    logical_size: int
    borders: tuple[_Border, ...] = ()


def _grid_indices(xp: object, n_rows: int, n_cols: int) -> tuple[Array, Array]:
    """Row- and column-index vectors of a dense ``n_rows × n_cols`` block (COO)."""
    rows = xp.reshape(  # type: ignore[attr-defined]
        xp.broadcast_to(  # type: ignore[attr-defined]
            xp.expand_dims(xp.arange(n_rows), axis=1),  # type: ignore[attr-defined]
            (n_rows, n_cols),
        ),
        (n_rows * n_cols,),
    )
    cols = xp.reshape(  # type: ignore[attr-defined]
        xp.broadcast_to(  # type: ignore[attr-defined]
            xp.expand_dims(xp.arange(n_cols), axis=0),  # type: ignore[attr-defined]
            (n_rows, n_cols),
        ),
        (n_rows * n_cols,),
    )
    return rows, cols


def _lowrank_border(xp: object, u: Array, m: Array) -> _Border:
    """Border for the L-BFGS low-rank term ``−U M⁻¹ Uᵀ`` (``C = Uᵀ``, ``E = M``)."""
    n_primal = int(u.shape[0])
    r = int(u.shape[1])
    # U entry (i, j) couples primal column i to auxiliary row j: C = Uᵀ.
    grid_rows, grid_cols = _grid_indices(xp, n_primal, r)
    conn_rows = grid_cols  # auxiliary-local (rank) index j
    conn_cols = grid_rows  # existing primal index i
    conn_values = xp.reshape(u, (n_primal * r,))  # type: ignore[attr-defined]
    m_rows, m_cols = _grid_indices(xp, r, r)
    return _Border(
        r,
        conn_rows,
        conn_cols,
        conn_values,
        m_rows,
        m_cols,
        xp.reshape(m, (r * r,)),  # type: ignore[attr-defined]
    )


def _inequality_border(ineq_jac: LinearOperator, sigma_s: LinearOperator) -> _Border:
    """Border keeping ``∇g`` explicit with the ``−Σ_s⁻¹`` block (augmented route).

    ``C = ∇g`` couples each slack multiplier ``Δλ`` to the primal variables, and
    ``E = −Σ_s⁻¹`` (``Σ_s = Λ/S`` strictly positive in the interior) is the slack
    block. The Schur term is the condensed ``∇gᵀ Σ_s ∇g`` — assembled implicitly,
    so the sparse factor stays as sparse as ``∇g`` itself.
    """
    g_rows, g_cols, g_values, shape = ineq_jac.to_coo()
    m_ineq = shape[0]
    xp = array_namespace(g_values)
    diag = xp.arange(m_ineq)
    inner = -1.0 / sigma_s.diagonal()  # −Σ_s⁻¹
    return _Border(m_ineq, g_rows, g_cols, g_values, diag, diag, inner)


def _border(a: _Assembly) -> tuple[Array, Array, Array, tuple[int, int]]:
    """Materialize the bordered COO matrix and its (augmented) shape."""
    xp = array_namespace(a.values)
    rows, cols, values = a.rows, a.cols, a.values
    offset = a.logical_size
    for border in a.borders:
        # Connection C at (offset + conn_rows, conn_cols) and its transpose Cᵀ.
        rows = xp.concat((rows, offset + border.conn_rows, border.conn_cols))
        cols = xp.concat((cols, border.conn_cols, offset + border.conn_rows))
        values = xp.concat((values, border.conn_values, border.conn_values))
        # Inner auxiliary block E on the trailing diagonal corner.
        rows = xp.concat((rows, offset + border.inner_rows))
        cols = xp.concat((cols, offset + border.inner_cols))
        values = xp.concat((values, border.inner_values))
        offset += border.size
    return rows, cols, values, (offset, offset)


def _logical_assembly(op: LinearOperator) -> _Assembly:
    """Logical COO block (+ any borders) for a KKT sub-block operator."""
    if isinstance(op, _CondensedOperator):
        return op.assemble()
    rows, cols, values, shape = op.to_coo()
    return _Assembly(rows, cols, values, logical_size=shape[0])


def _woodbury_factors(
    d: Array, u: Array, m: Array
) -> tuple[Array, Array, Array, Array]:
    """Precompute the reusable pieces of ``(diag(d) − U M⁻¹ Uᵀ)⁻¹``.

    Returns ``(d, D⁻¹U, Uᵀ, M − Uᵀ D⁻¹ U)``. The inner ``r × r`` factor is the
    expensive part, so callers that apply the inverse repeatedly (the L-BFGS Krylov
    preconditioner) factor once and reuse across :func:`_woodbury_solve` applies.
    """
    xp = array_namespace(d, u, m)
    inv_d_u = u / xp.expand_dims(d, axis=1)
    u_t = xp.permute_dims(u, (1, 0))
    inner = m - xp.matmul(u_t, inv_d_u)
    return d, inv_d_u, u_t, inner


def _woodbury_solve(factors: tuple[Array, Array, Array, Array], rhs: Array) -> Array:
    """Apply ``(diag(d) − U M⁻¹ Uᵀ)⁻¹`` to ``rhs`` from :func:`_woodbury_factors`.

    ``rhs`` may be a vector or a matrix (columns solved independently): one
    diagonal inverse plus one ``r × r`` solve, never forming the ``n × n`` operator.
    """
    d, inv_d_u, u_t, inner = factors
    xp = array_namespace(rhs, inv_d_u)
    if len(rhs.shape) == 1:
        inv_d_rhs = rhs / d
    elif len(rhs.shape) == 2:
        inv_d_rhs = rhs / xp.expand_dims(d, axis=1)
    else:
        raise ValueError("Woodbury solve requires a vector or matrix RHS")
    z = xp.linalg.solve(inner, xp.matmul(u_t, inv_d_rhs))
    return inv_d_rhs + xp.matmul(inv_d_u, z)


def _diagonal_solve(d: Array, rhs: Array) -> Array:
    """Apply ``diag(d)^-1`` to a vector or columns of a matrix RHS."""
    xp = array_namespace(d, rhs)
    if len(rhs.shape) == 1:
        return rhs / d
    if len(rhs.shape) == 2:
        return rhs / xp.expand_dims(d, axis=1)
    raise ValueError("diagonal solve requires a vector or matrix RHS")


def _add_dense_diagonal(matrix: Array, diagonal: Array) -> Array:
    """Return ``matrix + diag(diagonal)`` without an identity matmat probe."""
    xp = array_namespace(matrix, diagonal)
    n = int(diagonal.shape[0])
    return matrix + xp.eye(n, dtype=matrix.dtype) * diagonal


class _CondensedOperator(LinearOperator):
    """Lazy ``W + Sigma_x + J.T @ Sigma_s @ J + delta_w * I`` operator."""

    def __init__(
        self,
        W: LinearOperator,
        sigma_x: LinearOperator,
        sigma_s: LinearOperator,
        ineq_jac: LinearOperator,
        delta_w: float,
    ) -> None:
        if W.shape[0] != W.shape[1]:
            raise ValueError("W must be square")
        if sigma_x.shape != W.shape:
            raise ValueError("sigma_x must match W")
        if ineq_jac.shape[1] != W.shape[0]:
            raise ValueError("ineq_jac has incompatible variable dimension")
        if sigma_s.shape != (ineq_jac.shape[0], ineq_jac.shape[0]):
            raise ValueError("sigma_s must match the inequality dimension")
        self._W = W
        self._sigma_x = sigma_x
        self._sigma_s = sigma_s
        self._ineq_jac = ineq_jac
        self._delta_w = delta_w

    @property
    def shape(self) -> tuple[int, int]:
        return self._W.shape

    def matvec(self, v: Array) -> Array:
        result = self._W.matvec(v) + self._sigma_x.matvec(v)
        if self._ineq_jac.shape[0] > 0:
            jv = self._ineq_jac.matvec(v)
            result = result + self._ineq_jac.rmatvec(self._sigma_s.matvec(jv))
        if self._delta_w != 0.0:
            result = result + self._delta_w * v
        return result

    def rmatvec(self, v: Array) -> Array:
        return self.matvec(v)

    def matmat(self, V: Array) -> Array:
        result = self._W.matmat(V) + self._sigma_x.matmat(V)
        if self._ineq_jac.shape[0] > 0:
            jv = self._ineq_jac.matmat(V)
            result = result + self._ineq_jac.rmatmat(self._sigma_s.matmat(jv))
        if self._delta_w != 0.0:
            result = result + self._delta_w * V
        return result

    def rmatmat(self, V: Array) -> Array:
        return self.matmat(V)

    def dense_matrix(self, like: Array | None = None) -> Array:
        """Materialize the condensed dense block from explicit operator pieces."""
        template = like
        if template is None:
            try:
                template = self._sigma_x.diagonal()
            except NotImplementedError:
                template = None

        dense = self._W.dense_matrix(template)

        if isinstance(self._sigma_x, (Diagonal, Identity)):
            dense = _add_dense_diagonal(dense, self._sigma_x.diagonal(dense))
        else:
            dense = dense + self._sigma_x.dense_matrix(dense)

        if self._ineq_jac.shape[0] > 0:
            xp = array_namespace(dense)
            jac = self._ineq_jac.dense_matrix(dense)
            if isinstance(self._sigma_s, (Diagonal, Identity)):
                sigma_s_jac = xp.expand_dims(self._sigma_s.diagonal(jac), axis=1) * jac
            else:
                sigma_s_jac = xp.matmul(self._sigma_s.dense_matrix(jac), jac)
            dense = dense + xp.matmul(xp.permute_dims(jac, (1, 0)), sigma_s_jac)

        if self._delta_w != 0.0:
            xp = array_namespace(dense)
            dense = dense + self._delta_w * xp.eye(self.shape[0], dtype=dense.dtype)
        return dense

    def dense_structured_solve(self, rhs: Array) -> Array:
        """Exact dense solve for ``D - U M⁻¹ Uᵀ`` L-BFGS condensed blocks.

        For bound-only L-BFGS systems, ``N = D - U M⁻¹ Uᵀ`` with
        ``D = ξI + Σ_x + δ_w I``. The Woodbury identity solves ``N rhs = b``
        through one diagonal inverse and one compact ``2m × 2m`` solve, avoiding
        the dense ``n × n`` materialization. Inequality Gram terms are intentionally
        excluded here so this direct path remains exact.
        """
        if self._ineq_jac.shape[0] > 0:
            raise NotImplementedError(
                "structured dense solve does not handle inequality Gram terms"
            )
        if not isinstance(self._sigma_x, Diagonal):
            raise NotImplementedError(
                "structured dense solve requires a diagonal Sigma_x block"
            )
        sigma_x = self._sigma_x.diagonal()
        if isinstance(self._W, (Diagonal, Identity)):
            d = self._W.diagonal(sigma_x) + sigma_x
            if self._delta_w != 0.0:
                d = d + self._delta_w
            return _diagonal_solve(d, rhs)

        compact_form = getattr(self._W, "compact_form", None)
        if compact_form is None:
            raise NotImplementedError(
                "structured dense solve requires an L-BFGS compact Hessian"
            )
        xi, u, m_lbfgs = compact_form()
        d = xi + sigma_x
        if self._delta_w != 0.0:
            d = d + self._delta_w
        return _woodbury_solve(_woodbury_factors(d, u, m_lbfgs), rhs)

    def diagonal(self, like: Array | None = None) -> Array:
        """Cheap diagonal for Jacobi preconditioning.

        ``diag(W) + diag(Σ_x) + diag(∇gᵀ Σ_s ∇g) + δ_w``. Propagates
        ``NotImplementedError`` when any block lacks a cheap diagonal (e.g. an
        L-BFGS ``W`` with no curvature pairs yet), so the solver falls back to no
        preconditioning rather than paying ``n`` matvecs.
        """
        d = self._W.diagonal(like) + self._sigma_x.diagonal(like)
        if self._ineq_jac.shape[0] > 0:
            d = d + self._ineq_jac.gram_diagonal(self._sigma_s.diagonal())
        if self._delta_w != 0.0:
            d = d + self._delta_w
        return d

    def assemble(self) -> _Assembly:
        """Assemble the condensed Newton operator ``N`` as a bordered system.

        The logical block holds the diagonal-ish primal part; every dense product
        ``N`` would otherwise form is kept implicit as a zero-RHS :class:`_Border`
        whose Schur complement reproduces it exactly:

        - **Hessian ``W``.** A *diagonal-plus-low-rank* ``W = diag(d) − U M⁻¹ Uᵀ``
          (it exposes ``diagonal_low_rank_form``: the L-BFGS compact Hessian, or a
          matrix-free RT-like ``diag(h) + C Cᵀ``) contributes only ``d`` to the
          logical block and a low-rank border for ``−U M⁻¹ Uᵀ`` (IPOPT limited-
          memory; §4.3) — the dense term is never formed. Otherwise an assemblable
          (analytic/sparse) ``W`` emits its triplets directly. A ``W`` that is
          neither (e.g. an autodiff-HVP black box) propagates ``NotImplementedError``.
        - **Inequalities.** The Gram term ``∇gᵀ Σ_s ∇g`` is kept implicit as the
          indefinite augmented border (``∇g`` with the ``−Σ_s⁻¹`` slack block), so
          the factor stays as sparse as ``∇g`` instead of densifying the product.

        With both present (the RT case: low-rank Hessian + inequality caps) the two
        borders stack — their Schur terms add, recovering ``N`` exactly.
        """
        n = self._W.shape[0]
        sigma_x_diag = self._sigma_x.diagonal()
        xp = array_namespace(sigma_x_diag)
        idx = xp.arange(n)
        borders: list[_Border] = []

        low_rank_form = getattr(self._W, "diagonal_low_rank_form", None)
        if low_rank_form is not None:
            # Diagonal-plus-low-rank Hessian: diagonal d + low-rank border −U M⁻¹ Uᵀ.
            try:
                d, u, m = low_rank_form()
            except NotImplementedError:
                # L-BFGS before the first curvature pair: W = I, no low-rank part.
                d = xp.ones((n,), dtype=sigma_x_diag.dtype)
                u = m = None
            rows, cols, values = idx, idx, d + sigma_x_diag + self._delta_w
            if u is not None and m is not None and int(u.shape[1]) > 0:
                borders.append(_lowrank_border(xp, u, m))
        else:
            # Assemblable Hessian: W triplets + the Σ_x and δ_w diagonals.
            wr, wc, wv, _ = self._W.to_coo()
            shift_diag = sigma_x_diag
            if self._delta_w != 0.0:
                shift_diag = shift_diag + self._delta_w
            # Reserve the full diagonal so later δ_w activation changes values,
            # not sparsity. Sparse solvers can then reuse symbolic analysis.
            rows = xp.concat((wr, idx))
            cols = xp.concat((wc, idx))
            values = xp.concat((wv, shift_diag))

        if self._ineq_jac.shape[0] > 0:
            borders.append(_inequality_border(self._ineq_jac, self._sigma_s))

        return _Assembly(rows, cols, values, logical_size=n, borders=tuple(borders))

    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        """Emit the condensed block as COO triplets (invariant #4).

        Borders the :meth:`assemble` logical block with its low-rank / inequality
        borders, if any; the returned shape is the *augmented* size, while
        :attr:`shape` stays the logical ``n × n``.
        """
        del like
        return _border(self.assemble())

    def symmetry_hint(self) -> bool:
        """The condensed block (and its symmetric borders) is symmetric exactly.

        ``W`` is symmetric, the Σ_x/δ_w terms are diagonal, and each border places
        ``C``/``Cᵀ`` from one shared value array, so the assembled COO is exactly
        ``A == Aᵀ`` — the sparse-direct route can skip its numerical symmetry test.
        """
        return True

    def coo_pattern_signature(self) -> object | None:
        """Stable sparse structure key, or ``None`` for value-dependent patterns."""
        low_rank_form = getattr(self._W, "diagonal_low_rank_form", None)
        if low_rank_form is not None:
            try:
                _, u, _ = low_rank_form()
            except NotImplementedError:
                hessian_signature: object = ("diagonal_low_rank", self._W.shape, 0)
            else:
                hessian_signature = (
                    "diagonal_low_rank",
                    self._W.shape,
                    int(u.shape[1]),
                )
        else:
            hessian_signature = self._W.coo_pattern_signature()
            if hessian_signature is None:
                return None

        ineq_signature: object | None = None
        if self._ineq_jac.shape[0] > 0:
            ineq_signature = self._ineq_jac.coo_pattern_signature()
            if ineq_signature is None:
                return None

        return (
            "condensed",
            self.shape,
            hessian_signature,
            self._sigma_x.shape,
            self._ineq_jac.shape,
            ineq_signature,
        )

    def expected_inertia(self) -> tuple[int, int, int] | None:
        """IPOPT target inertia ``(n₊, n₋, n₀)`` of the assembled bordered system.

        For an *assemblable* Hessian the condensed block ``N`` is positive
        definite once regularized (``n`` positive eigenvalues), and each
        inequality contributes one negative ``−Σ_s⁻¹`` border row, so the matrix
        the sparse solver actually factors should have inertia ``(n, m_I, 0)``.
        A correct LDLᵀ factorization can *succeed* with the wrong inertia (a
        non-descent step), which is exactly what the inertia check catches.

        Returns ``None`` for a diagonal-plus-low-rank Hessian (L-BFGS /
        matrix-free): its low-rank border carries an inner block whose inertia is
        not known cheaply, and Powell-damped L-BFGS keeps ``W`` PD anyway, so the
        factorization-failure escalation suffices there.
        """
        if getattr(self._W, "diagonal_low_rank_form", None) is not None:
            return None
        return (self._W.shape[0], self._ineq_jac.shape[0], 0)

    def primal_block(self) -> LinearOperator | None:
        """The condensed block ``N`` that must be PD, or ``None`` to skip the check.

        ``N`` (this operator) is what the regularized Newton step needs positive
        definite; a dense solver can probe it with a Cholesky factorization. As
        with :meth:`expected_inertia`, a diagonal-plus-low-rank Hessian
        (L-BFGS / matrix-free) returns ``None``: it is PD by Powell damping and a
        Cholesky on its *materialized* form would defeat the matrix-free intent.
        """
        if getattr(self._W, "diagonal_low_rank_form", None) is not None:
            return None
        return self

    def lbfgs_inverse_apply(self) -> Callable[[Array], Array]:
        """L-BFGS-aware approximate inverse via Sherman–Morrison–Woodbury (§5.2).

        With ``W = ξI − U M⁻¹ Uᵀ`` (the compact L-BFGS Hessian) the condensed
        operator is ``N = D̃ − U M⁻¹ Uᵀ`` for the diagonal

            D̃ = ξ + diag(Σ_x) + δ_w + diag(∇gᵀ Σ_s ∇g),

        approximating only the *off-diagonal* of the inequality Gram term. The
        Woodbury identity then gives the exact inverse of that approximation::

            N⁻¹ r = D̃⁻¹ (r + U z),   z = (M − Uᵀ D̃⁻¹ U)⁻¹ (Uᵀ D̃⁻¹ r),

        an SPD operator costing one ``2k×2k`` solve plus ``O(n·k)`` per apply, with
        ``D̃`` and the inner factor precomputed once. Raises ``NotImplementedError``
        when ``W`` exposes no L-BFGS compact form (e.g. a matrix-free Hessian or
        no curvature pairs yet).
        """
        compact_form = getattr(self._W, "compact_form", None)
        if compact_form is None:
            raise NotImplementedError(
                "L-BFGS-aware preconditioner requires an L-BFGS Hessian block"
            )
        xi, u, m_lbfgs = compact_form()
        d_tilde = xi + self._sigma_x.diagonal()
        if self._ineq_jac.shape[0] > 0:
            d_tilde = d_tilde + self._ineq_jac.gram_diagonal(self._sigma_s.diagonal())
        if self._delta_w != 0.0:
            d_tilde = d_tilde + self._delta_w

        # Factor the 2k×2k inner block once (nonsingular for PD N); each apply is
        # then one diagonal inverse plus one small solve via Sherman–Morrison–Woodbury.
        factors = _woodbury_factors(d_tilde, u, m_lbfgs)
        return lambda r: _woodbury_solve(factors, r)


def build_condensed_operator(
    W: LinearOperator,
    sigma_x: LinearOperator,
    sigma_s: LinearOperator,
    ineq_jac: LinearOperator,
    reg: RegularizationState,
) -> LinearOperator:
    """Assemble the condensed normal-equations operator."""
    return _CondensedOperator(W, sigma_x, sigma_s, ineq_jac, reg.delta_w)


class _SaddleOperator(LinearOperator):
    """Bordered ``(Δx, Δy)`` quasidefinite saddle for equality constraints.

    Wraps the condensed inequality/bound block ``N`` (already carrying
    ``δ_w I``) and borders it with the equality Jacobian and the negative
    ``δ_c`` regularization on the (2,2) block (Friedlander–Orban 2012)::

        ┌ N        ∇cᵀ ┐
        └ ∇c    −δ_c I ┘
    """

    def __init__(
        self,
        condensed_n: LinearOperator,
        eq_jac: LinearOperator,
        delta_c: float,
    ) -> None:
        n = condensed_n.shape[0]
        if condensed_n.shape[1] != n:
            raise ValueError("condensed block must be square")
        if eq_jac.shape[1] != n:
            raise ValueError("eq_jac has incompatible variable dimension")
        self._n = n
        self._m = eq_jac.shape[0]
        self._condensed = condensed_n
        self._eq_jac = eq_jac
        self._delta_c = delta_c

    @property
    def shape(self) -> tuple[int, int]:
        size = self._n + self._m
        return size, size

    def matvec(self, v: Array) -> Array:
        xp = array_namespace(v)
        v_x = v[: self._n]
        v_y = v[self._n :]
        top = self._condensed.matvec(v_x) + self._eq_jac.rmatvec(v_y)
        bottom = self._eq_jac.matvec(v_x) - self._delta_c * v_y
        return xp.concat((top, bottom))

    def rmatvec(self, v: Array) -> Array:
        return self.matvec(v)

    def matmat(self, V: Array) -> Array:
        xp = array_namespace(V)
        v_x = V[: self._n, :]
        v_y = V[self._n :, :]
        top = self._condensed.matmat(v_x) + self._eq_jac.rmatmat(v_y)
        bottom = self._eq_jac.matmat(v_x) - self._delta_c * v_y
        return xp.concat((top, bottom), axis=0)

    def rmatmat(self, V: Array) -> Array:
        return self.matmat(V)

    def dense_matrix(self, like: Array | None = None) -> Array:
        """Materialize the bordered saddle block from explicit dense pieces."""
        n_dense = self._condensed.dense_matrix(like)
        if self._m == 0:
            return n_dense

        xp = array_namespace(n_dense)
        eq = self._eq_jac.dense_matrix(n_dense)
        top = xp.concat((n_dense, xp.permute_dims(eq, (1, 0))), axis=1)
        bottom = xp.concat(
            (eq, -self._delta_c * xp.eye(self._m, dtype=n_dense.dtype)), axis=1
        )
        return xp.concat((top, bottom), axis=0)

    def dense_structured_solve(self, rhs: Array) -> Array:
        """Exact Schur-complement solve using a structured condensed inverse.

        Propagates ``NotImplementedError`` from the condensed block (e.g. a
        non-L-BFGS Hessian or an inequality Gram term), so the dense solver falls
        back to materialization for systems without exploitable structure.
        """
        xp = array_namespace(rhs)
        if len(rhs.shape) == 1:
            vector_rhs = True
            rhs_x = xp.expand_dims(rhs[: self._n], axis=1)
            rhs_y = xp.expand_dims(rhs[self._n :], axis=1)
        elif len(rhs.shape) == 2:
            vector_rhs = False
            rhs_x = rhs[: self._n, :]
            rhs_y = rhs[self._n :, :]
        else:
            raise ValueError("saddle structured dense solve requires vector/matrix RHS")

        if self._m == 0:
            solution = self._condensed.dense_structured_solve(rhs_x)
            return solution[:, 0] if vector_rhs else solution

        identity_y = xp.eye(self._m, dtype=rhs.dtype)
        eq_t = self._eq_jac.rmatmat(identity_y)
        n_rhs = int(rhs_x.shape[1])
        batch_rhs = xp.concat((rhs_x, eq_t), axis=1)
        solved = self._condensed.dense_structured_solve(batch_rhs)
        n_inv_rhs = solved[:, :n_rhs]
        n_inv_eq_t = solved[:, n_rhs:]

        schur = self._eq_jac.matmat(n_inv_eq_t)
        if self._delta_c != 0.0:
            schur = schur + self._delta_c * identity_y
        rhs_schur = self._eq_jac.matmat(n_inv_rhs) - rhs_y
        dy = xp.linalg.solve(schur, rhs_schur)
        dx = n_inv_rhs - xp.matmul(n_inv_eq_t, dy)
        solution = xp.concat((dx, dy), axis=0)
        return solution[:, 0] if vector_rhs else solution

    def preferred_krylov_method(self) -> Literal["minres"]:
        """The equality saddle is symmetric indefinite, so MINRES is the route."""
        return "minres"

    def assemble(self) -> _Assembly:
        """Assemble the bordered quasidefinite saddle in ``[Δx | Δy]`` order.

        Extends the condensed block's logical assembly (size ``n``) with the
        equality Jacobian's symmetric off-diagonal blocks ``∇c`` / ``∇cᵀ`` and the
        ``−δ_c`` (2,2) diagonal, growing the logical size to ``n + m``. Any borders
        from the condensed block (L-BFGS low-rank, inequalities) are carried
        through unchanged — their connections live in the primal range ``[0, n)``,
        which the equality rows sit *after*, so they still border the full saddle
        correctly.
        """
        inner = _logical_assembly(self._condensed)
        xp = array_namespace(inner.values)
        n, m = self._n, self._m
        er, ec, ev, _ = self._eq_jac.to_coo()
        # ∇c at rows [n, n+m) × cols [0, n); ∇cᵀ is its transpose block.
        rows = xp.concat((inner.rows, er + n, ec))
        cols = xp.concat((inner.cols, ec, er + n))
        values = xp.concat((inner.values, ev, ev))
        if m > 0:
            didx = xp.arange(m) + n
            delta = xp.full((m,), -self._delta_c, dtype=values.dtype)
            rows = xp.concat((rows, didx))
            cols = xp.concat((cols, didx))
            values = xp.concat((values, delta))
        return _Assembly(rows, cols, values, logical_size=n + m, borders=inner.borders)

    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        """Emit the bordered saddle as COO triplets (invariant #4).

        Propagates ``NotImplementedError`` only from a non-L-BFGS matrix-free
        ``W`` or a matrix-free Jacobian (``∇c`` / ``∇g``). The returned shape is
        the *augmented* size when L-BFGS / inequality borders are present.
        """
        del like
        return _border(self.assemble())

    def symmetry_hint(self) -> bool:
        """The saddle is symmetric: ``∇c``/``∇cᵀ`` mirror one value array, the
        ``−δ_c`` (2,2) block is diagonal, and the condensed block is symmetric."""
        return True

    def coo_pattern_signature(self) -> object | None:
        condensed_signature = self._condensed.coo_pattern_signature()
        if condensed_signature is None:
            return None

        eq_signature: object | None = None
        if self._m > 0:
            eq_signature = self._eq_jac.coo_pattern_signature()
            if eq_signature is None:
                return None

        return (
            "saddle",
            self.shape,
            condensed_signature,
            self._eq_jac.shape,
            eq_signature,
        )

    def expected_inertia(self) -> tuple[int, int, int] | None:
        """IPOPT target inertia of the assembled saddle, or ``None``.

        The equality border adds the ``−δ_c`` (2,2) block, contributing ``m``
        more negative eigenvalues than the condensed block, so the target is the
        condensed target with ``m`` extra negatives — ``(n, m_E + m_I, 0)``.
        Propagates ``None`` when the condensed block does (low-rank Hessian).
        """
        inner = getattr(self._condensed, "expected_inertia", None)
        target = inner() if inner is not None else None
        if target is None:
            return None
        n_pos, n_neg, n_zero = target
        return (n_pos, n_neg + self._m, n_zero)

    def primal_block(self) -> LinearOperator | None:
        """The condensed ``N`` block of the saddle that must be PD, or ``None``.

        With ``N`` PD and ``δ_c > 0`` the saddle is quasidefinite (Friedlander–
        Orban), so probing ``N`` is sufficient for the dense PD guard.
        """
        fn = getattr(self._condensed, "primal_block", None)
        return fn() if fn is not None else None

    def spd_preconditioner_diagonal(self) -> Array:
        """SPD block-diagonal preconditioner for MINRES.

        The saddle is symmetric *indefinite*, so its own diagonal (with the
        ``−δ_c`` block) is not a valid symmetric preconditioner. We instead return
        the diagonal of the SPD block preconditioner

            ┌ diag(N)                              0 ┐
            └ 0        δ_c + diag(∇c diag(N)⁻¹ ∇cᵀ)  ┘

        — the PD primal Jacobi block stacked with a positive approximate-Schur
        dual block (Murphy–Golub–Wathen 2000). Propagates ``NotImplementedError``
        when ``diag(N)`` is unavailable (e.g. a matrix-free Hessian); the dual
        block falls back to the SPD identity when ``∇c`` cannot supply its row
        energies cheaply. The Krylov solver applies this by symmetric scaling.
        """
        diag_n = self._condensed.diagonal()
        xp = array_namespace(diag_n)
        try:
            schur = self._eq_jac.row_gram_diagonal(1.0 / diag_n)
            dual = self._delta_c + schur
        except NotImplementedError:
            # Primal-only preconditioning: SPD identity on the (strictly positive)
            # dual block keeps the whole preconditioner SPD.
            dual = xp.ones((self._m,), dtype=diag_n.dtype)
        return xp.concat((diag_n, dual))


def build_saddle_operator(
    condensed_n: LinearOperator,
    eq_jac: LinearOperator,
    delta_c: float,
) -> LinearOperator:
    """Border the condensed block with equalities into the quasidefinite saddle."""
    return _SaddleOperator(condensed_n, eq_jac, delta_c)


__all__ = [
    "build_condensed_operator",
    "build_saddle_operator",
]
