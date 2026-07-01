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

"""Lagrangian-Hessian strategies.

- **L-BFGS (default):** compact representation, memory ``m ∈ [5, 20]``, curvature
  pairs ``(δ_k, γ_k)`` from ``(x, ∇_x L)``. **Powell damping** enforces
  ``δᵀγ > 0`` so ``B`` stays PD ⇒ condensed ``N`` is PD ⇒ Cholesky/CG valid.
  Exposes both a matrix-free ``Hv`` (compact form, ``O(n·m)``) and a low-rank
  ``ξI + UVᵀ`` form for the dense/sparse routes.
- **Autodiff-HVP:** exact Hessian–vector products when the backend supports
  double-backprop and the user enables it.
- **Exact passthrough:** the operator from ``Problem.lagrangian_hessian``.
- **L-SR1 (future work, not exposed):** a limited-memory SR1 update would allow
  an *indefinite* Hessian approximation (capturing negative curvature on
  nonconvex problems), but that breaks the condensed normal-equations route's
  "no inertia oracle required" property (which relies on a PD (1,1) block via
  Powell damping). It would need the indefinite augmented/inertia route or a
  trust-region globalization, so it is a deliberate scope decision — see AGENTS.md
  Direction — and is not offered as a ``HessianMode`` until then.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipax.backend.namespace import array_namespace
from ipax.backend.operators import LinearOperator

if TYPE_CHECKING:
    from ipax.options import LBFGSOptions
    from ipax.typing import Array, Namespace


# Powell damping (Powell 1978; Nocedal & Wright §18.3): keep δᵀγ ≥ κ · δᵀBδ so
# the BFGS update stays positive definite even on nonconvex Lagrangians.
_POWELL_KAPPA = 0.2


class LBFGSOperator(LinearOperator):
    """Compact, Powell-damped L-BFGS Hessian approximation of the Lagrangian.

    Stores the most recent ``m`` curvature pairs ``(δ_k, γ_k)`` and applies the
    *direct* (Hessian, not inverse) compact representation of Byrd, Nocedal &
    Schnabel (1994, eq. 3.2)::

        B = ξI − U M⁻¹ Uᵀ,   U = [ξS  Y],
        M = ┌ ξ SᵀS   L ┐,   L = strict-lower(SᵀY),  D = diag(δ_kᵀγ_k)
            └ Lᵀ      −D ┘

    With ``LBFGSOptions.initial_scaling`` enabled, ``ξ`` is
    ``(γ_kᵀγ_k)/(δ_kᵀγ_k)`` from the newest pair. Otherwise the compact update
    uses the unscaled identity seed. With no pairs the operator is always the
    identity, so the first IPM step is a Newton-seed step.
    """

    def __init__(self, n: int, options: LBFGSOptions) -> None:
        if n < 0:
            raise ValueError("L-BFGS dimension must be non-negative")
        self._n = n
        self._options = options
        self._memory = max(1, int(options.memory))
        # Curvature history as n×k matrices (None ⇒ no pairs yet ⇒ B = I).
        self._s: Array | None = None
        self._y: Array | None = None
        # Cached compact-form pieces, rebuilt whenever the history changes.
        self._xi: float = 1.0
        self._u: Array | None = None
        self._m: Array | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self._n, self._n

    def matvec(self, v: Array) -> Array:
        return self._apply(v)

    def rmatvec(self, v: Array) -> Array:
        return self._apply(v)  # B is symmetric

    def matmat(self, V: Array) -> Array:
        return self._apply(V)

    def rmatmat(self, V: Array) -> Array:
        return self._apply(V)  # B is symmetric

    def dense_matrix(self, like: Array | None = None) -> Array:
        """Materialize the compact Hessian directly from ``B = xi*I - U M^-1 U.T``."""
        if self._u is None or self._m is None:
            if like is None:
                raise NotImplementedError(
                    "L-BFGS dense matrix requires a template before the first pair"
                )
            xp = array_namespace(like)
            return xp.eye(self._n, dtype=like.dtype)

        xp = array_namespace(self._u)
        identity = xp.eye(self._n, dtype=self._u.dtype)
        correction = xp.matmul(
            self._u,
            xp.linalg.solve(self._m, xp.permute_dims(self._u, (1, 0))),
        )
        return self._xi * identity - correction

    def _apply(self, x: Array) -> Array:
        """Apply ``B = ξI − U M⁻¹ Uᵀ`` to a vector or a batch of columns.

        Shared by matvec/matmat: ``xp.linalg.solve`` takes a vector or a matrix
        RHS, so the compact form is evaluated in one pass whether ``x`` is 1-D
        (single application) or 2-D (batched materialization for the dense route).
        """
        if self._u is None or self._m is None:
            return x  # B = ξI with ξ = 1 (identity seed)
        xp = array_namespace(x)
        u_t_x = xp.matmul(xp.permute_dims(self._u, (1, 0)), x)
        z = xp.linalg.solve(self._m, u_t_x)
        return self._xi * x - xp.matmul(self._u, z)

    def diagonal(self, like: Array | None = None) -> Array:
        """Diagonal of the compact Hessian ``B = ξI − U M⁻¹ Uᵀ`` (§4.3).

        ``diag(B)_k = ξ − (U M⁻¹ Uᵀ)_kk`` computed in ``O(n·k)`` from the cached
        compact factors — no ``n×n`` matrix. Raised before the first curvature
        pair (``B = I`` then, but no array is on hand to size the result), which
        the Jacobi preconditioner treats as "no diagonal available".
        """
        if self._u is None or self._m is None:
            if like is not None:
                xp = array_namespace(like)
                return xp.ones((self._n,), dtype=like.dtype)
            raise NotImplementedError(
                "L-BFGS diagonal is unavailable before the first curvature pair"
            )
        del like
        xp = array_namespace(self._u)
        # z = M⁻¹ Uᵀ (2k×n); (U M⁻¹ Uᵀ)_kk = Σ_j U_kj z_jk.
        z = xp.linalg.solve(self._m, xp.permute_dims(self._u, (1, 0)))
        correction = xp.sum(self._u * xp.permute_dims(z, (1, 0)), axis=1)
        return self._xi - correction

    def compact_form(self) -> tuple[float, Array, Array]:
        """Return the compact-form factors ``(ξ, U, M)`` of ``B = ξI − U M⁻¹ Uᵀ``.

        Consumed by the L-BFGS-aware preconditioner (§5.2), which folds the
        low-rank ``U M⁻¹ Uᵀ`` into a Woodbury inverse. Raised before the first
        curvature pair (``B = I``, no low-rank part), which callers treat as
        "no L-BFGS structure available".
        """
        if self._u is None or self._m is None:
            raise NotImplementedError(
                "L-BFGS compact form is unavailable before the first curvature pair"
            )
        return self._xi, self._u, self._m

    def diagonal_low_rank_form(self) -> tuple[Array, Array, Array]:
        """Return ``(d, U, M)`` with ``B == diag(d) − U M⁻¹ Uᵀ`` (sparse-assemblable).

        The diagonal-plus-low-rank view of the compact Hessian, consumed by the
        sparse-direct route to factor ``B`` as a bordered system instead of
        forming the dense ``U M⁻¹ Uᵀ`` (§4.3, IPOPT limited-memory). ``d = ξ·1``
        (the scaled identity seed); ``U``, ``M`` as in :meth:`compact_form` (``M``
        is **not** inverted — it becomes the border's trailing block). Raised
        before the first curvature pair (``B = I``), which the assembler treats as
        a plain unit diagonal. This is the generic hook any *diagonal-plus-low-
        rank* operator implements; the L-BFGS case is one instance.
        """
        if self._u is None or self._m is None:
            raise NotImplementedError(
                "L-BFGS low-rank form is unavailable before the first curvature pair"
            )
        xp = array_namespace(self._u)
        d = self._xi * xp.ones((self._n,), dtype=self._u.dtype)
        return d, self._u, self._m

    def update(self, delta: Array, gamma: Array) -> None:
        """Push a curvature pair, applying Powell damping to keep ``δᵀγ > 0``.

        ``delta = x⁺ − x`` and ``gamma = ∇_xL(x⁺) − ∇_xL(x)`` (same multipliers).
        A pair that is still non-positive after damping is dropped rather than
        corrupting the approximation.
        """
        xp = array_namespace(delta, gamma)
        s = delta
        y = gamma
        bs = self.matvec(s)
        s_bs = float(xp.sum(s * bs))
        s_y = float(xp.sum(s * y))

        if self._options.powell_damping and s_y < _POWELL_KAPPA * s_bs:
            denom = s_bs - s_y
            # denom > 0 here since s_y < κ·s_bs ≤ s_bs (and s_bs ≥ 0 for PD B).
            theta = (1.0 - _POWELL_KAPPA) * s_bs / denom if denom > 0.0 else 1.0
            y = theta * y + (1.0 - theta) * bs
            s_y = float(xp.sum(s * y))

        if s_y <= 0.0:
            return  # safeguard: skip a pair that cannot keep B positive definite

        col_s = xp.reshape(s, (self._n, 1))
        col_y = xp.reshape(y, (self._n, 1))
        if self._s is None or self._y is None:
            self._s, self._y = col_s, col_y
        else:
            self._s = xp.concat((self._s, col_s), axis=1)
            self._y = xp.concat((self._y, col_y), axis=1)
            k = int(self._s.shape[1])
            if k > self._memory:
                self._s = self._s[:, k - self._memory :]
                self._y = self._y[:, k - self._memory :]
        self._rebuild(xp)

    def _rebuild(self, xp: Namespace) -> None:
        """Recompute the cached compact-form factors ``U`` and ``M``."""
        s = self._s
        y = self._y
        assert s is not None and y is not None
        k = int(s.shape[1])

        # Initial scaling ξ from the newest pair (Nocedal & Wright eq. 7.20,
        # inverted for the direct Hessian): ξ = γᵀγ / δᵀγ. When disabled,
        # use the unscaled identity seed for the compact update.
        s_last = s[:, k - 1]
        y_last = y[:, k - 1]
        s_y_last = float(xp.sum(s_last * y_last))
        y_y_last = float(xp.sum(y_last * y_last))
        xi = (
            y_y_last / s_y_last
            if self._options.initial_scaling and s_y_last > 0.0
            else 1.0
        )
        self._xi = xi

        s_t = xp.permute_dims(s, (1, 0))
        s_s = xp.matmul(s_t, s)  # k×k, SᵀS
        s_y = xp.matmul(s_t, y)  # k×k, (SᵀY)_{ij} = δ_iᵀγ_j
        lower = xp.tril(s_y, k=-1)  # strict lower triangle L
        diag = xp.sum(s * y, axis=0)  # diag(D)_i = δ_iᵀγ_i
        d_mat = xp.eye(k, dtype=s.dtype) * diag

        top = xp.concat((xi * s_s, lower), axis=1)
        bottom = xp.concat((xp.permute_dims(lower, (1, 0)), -d_mat), axis=1)
        self._m = xp.concat((top, bottom), axis=0)
        self._u = xp.concat((xi * s, y), axis=1)  # n×2k


__all__ = ["LBFGSOperator"]
