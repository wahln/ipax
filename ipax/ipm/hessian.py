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
        # Gram blocks ``SᵀS`` and ``SᵀY`` (k×k), maintained incrementally: a new
        # pair borders them with one row/column (O(n·k)) and a dropped pair
        # slices them — recomputing both from scratch each update is O(n·k²).
        self._ss: Array | None = None
        self._sy: Array | None = None
        self._yy: Array | None = None
        # Namespace of the curvature arrays, learned at the first update so the
        # per-apply paths need not re-resolve it from their inputs.
        self._xp: Namespace | None = None
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

    def _namespace(self) -> Namespace:
        """Namespace of the stored pairs (learned at the first update; resolved
        lazily for operators assembled from explicit factors)."""
        if self._xp is None:
            assert self._u is not None
            self._xp = array_namespace(self._u)
        return self._xp

    def dense_matrix(self, like: Array | None = None) -> Array:
        """Materialize the compact Hessian directly from ``B = xi*I - U M^-1 U.T``."""
        if self._u is None or self._m is None:
            if like is None:
                raise NotImplementedError(
                    "L-BFGS dense matrix requires a template before the first pair"
                )
            xp = array_namespace(like)
            return xp.eye(self._n, dtype=like.dtype)

        xp = self._namespace()
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
        xp = self._namespace()
        u_t_x = xp.matmul(xp.permute_dims(self._u, (1, 0)), x)
        # A singular middle matrix leaves the low-rank correction undefined, and
        # letting the backend's error escape aborts the whole solve (S2MPJ
        # ``LINSPANH`` on ``lbfgs/krylov``). ``SᵀS`` degenerates when a stored
        # ``δ = x⁺ − x`` is (nearly) collinear with an existing column — an exact
        # zero is already dropped by the curvature safeguard in ``update``, a
        # near-duplicate is not. Fall back to the identity seed ``ξx`` for this
        # application: it keeps ``B`` positive definite, which is what the
        # condensed route needs, and only costs the curvature information the
        # unusable factors could not have supplied anyway.
        # Backends disagree on how a singular solve fails: NumPy raises
        # ``LinAlgError``, Torch its own type, and some return inf/nan instead.
        # The non-finite check is on ``z`` rather than on the finished product:
        # the garbage originates in the solve, and ``z`` is ``2k``-sized (k =
        # L-BFGS memory) where the result is ``n``-sized. This is the matvec —
        # scanning ``n`` here allocates a temporary and forces a host
        # synchronisation on *every* application, the pattern that cost ~23x on
        # GPU in ``barrier.py``.
        try:
            z = xp.linalg.solve(self._m, u_t_x)
        except Exception:
            return self._xi * x
        if not bool(xp.all(xp.isfinite(z))):
            return self._xi * x
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
        xp = self._namespace()
        # z = M⁻¹ Uᵀ (2k×n); (U M⁻¹ Uᵀ)_kk = Σ_j U_kj z_jk.
        #
        # A singular M makes the correction — and therefore the diagonal —
        # undefined. That is the same situation as "before the first curvature
        # pair" from the caller's point of view, so report it the same way:
        # ``spd_preconditioner_diagonal`` propagates ``NotImplementedError`` and
        # the Krylov solver drops to a preconditioner that needs no diagonal.
        # Letting the backend's own error escape instead aborted the whole solve
        # (S2MPJ ``LINSPANH`` on ``lbfgs/krylov``). Backends disagree on how they
        # fail — NumPy raises ``LinAlgError``, Torch its own type, and some
        # return inf/nan — so both paths are covered.
        try:
            z = xp.linalg.solve(self._m, xp.permute_dims(self._u, (1, 0)))
        except Exception as exc:
            raise NotImplementedError(
                "L-BFGS diagonal is unavailable: the compact middle matrix is singular"
            ) from exc
        correction = xp.sum(self._u * xp.permute_dims(z, (1, 0)), axis=1)
        diagonal = self._xi - correction
        if not bool(xp.all(xp.isfinite(diagonal))):
            raise NotImplementedError(
                "L-BFGS diagonal is unavailable: the compact middle matrix is "
                "numerically singular"
            )
        return diagonal

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

    def gram_blocks(self) -> tuple[Array, Array, Array]:
        """Return the cached Gram blocks ``(SᵀS, SᵀY, YᵀY)`` of the window.

        Maintained incrementally by :meth:`update`, so a consumer can form
        ``UᵀU`` for ``U = [ξS  Y]`` in O(k²) without touching the ``n×2k``
        factor — the dense structured solve uses this when ``Σ_x ≡ 0`` (no
        bounds), where the Woodbury inner factor is ``M − UᵀU/(ξ + δ_w)``.
        Raised before the first curvature pair.
        """
        if self._ss is None or self._sy is None or self._yy is None:
            raise NotImplementedError(
                "L-BFGS Gram blocks are unavailable before the first curvature pair"
            )
        return self._ss, self._sy, self._yy

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
        xp = self._namespace()
        d = self._xi * xp.ones((self._n,), dtype=self._u.dtype)
        return d, self._u, self._m

    def update(self, delta: Array, gamma: Array) -> None:
        """Push a curvature pair, applying Powell damping to keep ``δᵀγ > 0``.

        ``delta = x⁺ − x`` and ``gamma = ∇_xL(x⁺) − ∇_xL(x)`` (same multipliers).
        A pair that is still non-positive after damping is dropped rather than
        corrupting the approximation.
        """
        xp = array_namespace(delta, gamma)
        self._xp = xp
        # A non-finite curvature pair (the Lagrangian gradient overflowed to
        # inf/NaN at a trial iterate — e.g. an exp/rational element function
        # evaluated far outside its safe range) would be appended and corrupt the
        # compact form *permanently*, since the poisoned column survives in the
        # memory window. The Powell/positive-curvature safeguards below do not
        # catch it: ``s_y`` is then NaN and every ``<``/``<=`` comparison is
        # False. Drop the pair so the approximation stays finite and the solver
        # can still take a (steepest-descent-like) step to escape the region.
        # (One host sync for both checks: each ``bool()`` syncs on a GPU.)
        if not bool(xp.all(xp.isfinite(delta)) & xp.all(xp.isfinite(gamma))):
            return
        s = delta
        y = gamma
        bs = self.matvec(s)
        s_bs = float(xp.sum(s * bs))
        s_y = float(xp.sum(s * y))

        # A pair that strongly *contradicts* positive curvature is dropped
        # rather than damped: the Powell blend below would fabricate PD
        # evidence out of it (θ-mixing y toward Bs), and that fabricated
        # curvature steers the search from the history for m more updates
        # (S2MPJ ORTHRGDS: s·y/s·Bs down to −25 damped ⇒ 1000+ iterations at
        # a worse optimum; skipped ⇒ ~20 to IPOPT's objective). Milder
        # indefiniteness falls through to the damping branch, where the
        # full-corpus A/B shows the blend genuinely helps.
        skip_ratio = self._options.damping_skip_ratio
        if skip_ratio is not None and s_bs > 0.0 and s_y < -skip_ratio * s_bs:
            return

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
        row_s = xp.permute_dims(col_s, (1, 0))
        row_y = xp.permute_dims(col_y, (1, 0))
        if self._s is None or self._y is None:
            self._s, self._y = col_s, col_y
            self._ss = xp.matmul(row_s, col_s)
            self._sy = xp.matmul(row_s, col_y)
            self._yy = xp.matmul(row_y, col_y)
        else:
            assert self._ss is not None and self._sy is not None
            assert self._yy is not None
            s_t = xp.permute_dims(self._s, (1, 0))
            y_t = xp.permute_dims(self._y, (1, 0))
            # Border the Gram blocks with the new pair: (SᵀS)_{i,new} = δ_iᵀδ_new
            # (symmetric), (SᵀY)_{i,new} = δ_iᵀγ_new (column) and
            # (SᵀY)_{new,j} = δ_newᵀγ_j (row).
            ss_col = xp.matmul(s_t, col_s)  # k×1
            sy_col = xp.matmul(s_t, col_y)  # k×1
            sy_row = xp.matmul(row_s, self._y)  # 1×k
            yy_col = xp.matmul(y_t, col_y)  # k×1
            ss_corner = xp.matmul(row_s, col_s)  # 1×1
            sy_corner = xp.matmul(row_s, col_y)  # 1×1
            yy_corner = xp.matmul(row_y, col_y)  # 1×1
            self._ss = xp.concat(
                (
                    xp.concat((self._ss, ss_col), axis=1),
                    xp.concat((xp.permute_dims(ss_col, (1, 0)), ss_corner), axis=1),
                ),
                axis=0,
            )
            self._sy = xp.concat(
                (
                    xp.concat((self._sy, sy_col), axis=1),
                    xp.concat((sy_row, sy_corner), axis=1),
                ),
                axis=0,
            )
            self._yy = xp.concat(
                (
                    xp.concat((self._yy, yy_col), axis=1),
                    xp.concat((xp.permute_dims(yy_col, (1, 0)), yy_corner), axis=1),
                ),
                axis=0,
            )
            self._s = xp.concat((self._s, col_s), axis=1)
            self._y = xp.concat((self._y, col_y), axis=1)
            k = int(self._s.shape[1])
            if k > self._memory:
                drop = k - self._memory
                self._s = self._s[:, drop:]
                self._y = self._y[:, drop:]
                self._ss = self._ss[drop:, drop:]
                self._sy = self._sy[drop:, drop:]
                self._yy = self._yy[drop:, drop:]
        self._rebuild(xp, s_y_last=s_y)

    def _rebuild(self, xp: Namespace, *, s_y_last: float) -> None:
        """Recompute the cached compact-form factors ``U`` and ``M``.

        ``s_y_last = δ_kᵀγ_k`` of the newest (damped) pair is already known to
        the caller, and the Gram blocks come from the incremental cache, so
        this costs O(k²) plus the O(n·k) copy into ``U``.
        """
        s = self._s
        y = self._y
        s_s = self._ss
        s_y = self._sy
        assert s is not None and y is not None
        assert s_s is not None and s_y is not None
        k = int(s.shape[1])

        # Initial scaling ξ from the newest pair; when disabled, the unscaled
        # identity seed. Two standard estimates (LBFGSOptions.seed_formula):
        # "direct" (Nocedal & Wright eq. 7.20, inverted for the direct
        # Hessian) is ξ = γᵀγ/δᵀγ; "scalar1" (IPOPT's
        # limited_memory_initialization) is ξ = δᵀγ/δᵀδ. They differ by the
        # δ–γ misalignment factor 1/cos²∠(δ,γ) ≥ 1, which badly-scaled least
        # squares can drive to ~1e15 (S2MPJ NELSONLS) — an over-stiff seed
        # freezes the primal step and feeds the Powell-damping test an
        # inflated δᵀBδ.
        xi = 1.0
        if self._options.initial_scaling and s_y_last > 0.0:
            if self._options.seed_formula == "scalar1":
                s_s_last = float(s_s[k - 1, k - 1])
                xi = s_y_last / s_s_last if s_s_last > 0.0 else 1.0
            else:
                y_last = y[:, k - 1]
                y_y_last = float(xp.sum(y_last * y_last))
                xi = y_y_last / s_y_last
        self._xi = xi

        lower = xp.tril(s_y, k=-1)  # strict lower triangle L
        d_mat = s_y * xp.eye(k, dtype=s.dtype)  # diag(D)_i = δ_iᵀγ_i = (SᵀY)_ii

        top = xp.concat((xi * s_s, lower), axis=1)
        bottom = xp.concat((xp.permute_dims(lower, (1, 0)), -d_mat), axis=1)
        self._m = xp.concat((top, bottom), axis=0)
        self._u = xp.concat((xi * s, y), axis=1)  # n×2k


__all__ = ["LBFGSOperator"]
