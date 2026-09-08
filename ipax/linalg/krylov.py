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

"""``KrylovSolver`` — matrix-free, default at scale.

Conjugate Gradients on the PD condensed normal-equations system (Breedveld 2017,
eq. 18/20) and MINRES on the symmetric-*indefinite* equality saddle
(Friedlander–Orban 2012). Everything is expressed with
matvecs + vector ops in the Array-API standard, so no KKT matrix is ever formed
and the same code runs on every backend.

The default ``method="cg"`` *prefers* CG but is robust: if it encounters
non-positive curvature ``pᵀKp ≤ 0`` (i.e. the operator is not PD, as happens for
the equality saddle) it transparently restarts with MINRES. ``method="minres"``
forces MINRES; ``method="gmres"`` runs restarted, left-preconditioned GMRES(m) —
a robust alternative that also handles symmetric-indefinite systems directly.

Preconditioning (§5.2), all matrix-free:
- ``jacobi`` — a strictly positive (SPD) diagonal: the operator's own diagonal for
  CG/GMRES, or the equality saddle's SPD *block* diagonal (PD primal Jacobi block
  plus a positive approximate-Schur dual block, ``spd_preconditioner_diagonal``)
  applied to MINRES by symmetric scaling. **Exception:** when the operator
  reports its Woodbury inverse as *exact* (``lbfgs_inverse_is_exact`` — a
  bound-only L-BFGS condensed block, no inequality Gram term) ``jacobi`` and
  ``auto`` solve with that inverse *directly* (reported as ``pc=lbfgs-exact``,
  ``last_method="direct"``): one Woodbury apply verified by a true-residual
  check, with working-precision iterative refinement (Carson & Higham 2018)
  covering round-off, falling back to the CG-preconditioned route when
  refinement stalls — no CG loop on the fast path, and neither the O(n·k²)
  L-BFGS diagonal nor its two host syncs are paid.
  ``KrylovOptions.exact_lbfgs_inverse=False`` restores the plain diagonal
  (the A/B lever).
- ``lbfgs`` — an L-BFGS-aware Sherman–Morrison–Woodbury inverse. On the condensed
  (equality-free) operator it is ``N⁻¹`` (``lbfgs_inverse_apply``), an SPD operator
  used directly by CG/GMRES. On the equality **saddle** it is the block-diagonal
  ``diag(N⁻¹, S⁻¹)`` (``lbfgs_block_preconditioner_apply``; Murphy–Golub–Wathen
  2000) — the Woodbury ``N⁻¹`` on the (1,1) block and the reciprocal
  approximate-Schur diagonal on the (2,2) block. Being non-diagonal it is applied
  by **GMRES** (the default ``cg`` route switches to GMRES when this preconditioner
  is available), since MINRES admits only a diagonal. It degrades to ``jacobi``
  where no L-BFGS compact form is available (e.g. before the first curvature pair,
  or an exact/matrix-free Hessian).
- ``auto`` — start with cheap ``jacobi`` (or the exact inverse where it applies,
  as above) and self-promote to ``lbfgs`` the first time a solve struggles: a convergence failure the Woodbury inverse could rescue
  triggers an immediate promoted retry, and a merely slow success (more than
  ``auto_switch_ratio`` of the iteration budget) promotes for the next solve. The
  flag is sticky for the life of the solver, so the extra Woodbury cost is paid
  only on ill-conditioned systems while well-conditioned ones stay on Jacobi.
All preconditioners fall back to none when no suitable structure exists.

References: Hestenes & Stiefel 1952 (CG); Paige & Saunders 1975 (MINRES); Saad &
Schultz 1986 (GMRES); Murphy, Golub & Wathen 2000 (block preconditioning of
saddle systems); Byrd, Nocedal & Schnabel 1994 (compact L-BFGS).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from ipax.backend.namespace import array_namespace
from ipax.backend.operators import MatrixFreeJacobian
from ipax.linalg.solver import LinearSolveError

if TYPE_CHECKING:
    from ipax.backend.operators import LinearOperator
    from ipax.options import KrylovOptions
    from ipax.typing import Array, Namespace


class KrylovConvergenceError(LinearSolveError):
    """Raised when an iterative solve does not reach the tolerance in time.

    The IPM driver catches this (like a failed dense factorization) and escalates
    the primal regularization ``δ_w`` (§4.4), which improves conditioning and the
    positive-definiteness the condensed CG path relies on.
    """


class _IndefiniteOperatorError(Exception):
    """Internal signal: CG hit non-positive curvature; retry with MINRES."""


def _is_resource_failure(exc: BaseException) -> bool:
    """Whether a backend exception is an allocation failure, not numerics.

    Device backends raise their own types (torch's CUDA ``OutOfMemoryError`` is
    a ``RuntimeError``, JAX reports ``RESOURCE_EXHAUSTED``) which cannot be
    named in the core (invariant #1); recognise them by class name and message.
    """
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        "outofmemory" in name
        or "out of memory" in message
        or ("resource_exhausted" in message)
    )


# Woodbury-apply budget for the direct exact-inverse solve: one apply plus up
# to two working-precision iterative-refinement rounds (they converge when
# cond(N)·u ≲ 1 — Carson & Higham 2018; the exact-inverse case is the
# best-conditioned instance of their fixed-precision setting). A residual
# still above tolerance after the budget does NOT fail the solve: the dispatch
# falls back to CG preconditioned with the same inverse — the pre-direct route,
# which is Galerkin-optimal per apply and therefore at least as strong — so
# the budget is purely a fast-path/fallback split, not a robustness knob (the
# same pattern as ``_MAX_REG_ATTEMPTS`` in the driver).
_MAX_EXACT_APPLIES = 3


def _inner(xp: Namespace, a: Array, b: Array) -> float:
    return float(xp.sum(a * b))


def _norm(xp: Namespace, a: Array) -> float:
    return float(xp.sqrt(xp.sum(a * a)))


def _true_residual_norm(
    xp: Namespace,
    K: LinearOperator,
    x: Array,
    b: Array,
) -> float:
    return _norm(xp, K.matvec(x) - b)


def _givens(a: float, b: float) -> tuple[float, float]:
    """Givens rotation ``(c, s)`` with ``c·a + s·b = √(a²+b²)`` and ``−s·a + c·b = 0``."""
    if b == 0.0:
        return 1.0, 0.0
    r = (a * a + b * b) ** 0.5
    return a / r, b / r


class KrylovSolver:
    """Matrix-free CG / MINRES with optional Jacobi preconditioning."""

    def __init__(self, options: KrylovOptions) -> None:
        self._options = options
        self._operator: LinearOperator | None = None
        # Diagnostics from the most recent solve (consumed by tests/benchmarks).
        self.last_iterations: int = 0
        self.last_residual: float = 0.0
        self.last_method: str = ""
        # ``preconditioner="auto"``: sticky flag, set once a solve struggles, that
        # promotes the effective preconditioner from Jacobi to L-BFGS (§5.2).
        self._auto_promoted: bool = False
        # Set per solve (reset at the top of ``solve``) when the default/auto
        # mode applied the operator's *exact* condensed Woodbury inverse instead
        # of Jacobi (bound-only L-BFGS systems); reported by ``describe``.
        self._exact_inverse_active: bool = False
        # Sticky opt-out (like ``_auto_promoted``): a solve that broke down
        # under the exact inverse — a numerically singular L-BFGS middle
        # matrix, the ``_apply`` fallback case — retries on Jacobi and this
        # solver stays on Jacobi thereafter, so a persistently singular window
        # never pays a failed exact-inverse CG on every IPM iteration.
        self._exact_inverse_blocked: bool = False
        # Inexact-Newton forcing: the most recent outer KKT residual hinted by the
        # driver, or ``None`` before the first hint (then the fixed ``rtol`` is used).
        self._outer_residual: float | None = None

    def set_outer_residual(self, residual: float) -> None:
        """Record the current outer KKT residual for the adaptive inner tolerance.

        Non-finite or non-positive hints are ignored (the fixed ``rtol`` then
        applies) so a degenerate residual can never loosen the solve.
        """
        if math.isfinite(residual) and residual > 0.0:
            self._outer_residual = residual

    def is_direct(self) -> bool:
        """``False``: ``factor`` binds the operator; each solve is a Krylov run.

        The bound-only L-BFGS exact-inverse dispatch (``_exact_inverse_solve``)
        is the one solve that is not, but it applies only to problems without
        constraints — which never run a second-order correction — so the
        driver's use of this hook is unaffected by it.
        """
        return False

    def _effective_rtol(self) -> float:
        """Inner relative tolerance for this solve — fixed, or the forcing term.

        With ``adaptive_tol`` (and a residual hint) this is the Eisenstat–Walker
        forcing term ``clip(η · ‖r_outer‖, rtol, adaptive_rtol_max)``: loose while
        the IPM is far from optimal (so an ill-conditioned early system a tight
        ``rtol`` cannot reach still yields a usable inexact-Newton step) and
        tightening to ``rtol`` as the outer residual falls. Otherwise the fixed
        ``rtol``.
        """
        opts = self._options
        if not opts.adaptive_tol or self._outer_residual is None:
            return opts.rtol
        target = opts.adaptive_eta * self._outer_residual
        return min(opts.adaptive_rtol_max, max(opts.rtol, target))

    def describe(self) -> str:
        """Human-readable label for diagnostics, incl. method/preconditioner."""
        pc: str = self._options.preconditioner
        if pc == "auto":
            pc = f"auto:{self._effective_preconditioner()}"
        if self._exact_inverse_active:
            pc = "auto:lbfgs-exact" if pc.startswith("auto") else "lbfgs-exact"
        return f"krylov ({self._options.method}, pc={pc})"

    def kkt_form(self) -> str:
        """The KKT assembly iterated on (``Result.routes.kkt_form``): the
        condensed normal-equations block (bordered into the equality saddle
        where needed), applied matrix-free."""
        return "condensed"

    def _effective_preconditioner(self) -> str:
        """Resolve the preconditioner actually in force this solve.

        In ``"auto"`` mode this is Jacobi until a solve struggles, then L-BFGS
        (sticky for the life of this solver instance / IPM run); otherwise it is
        the configured mode verbatim.
        """
        mode = self._options.preconditioner
        if mode == "auto":
            return "lbfgs" if self._auto_promoted else "jacobi"
        return mode

    def factor(self, K: LinearOperator) -> None:
        """Store the operator. Krylov methods need no factorization."""
        if K.shape[0] != K.shape[1]:
            raise ValueError("KrylovSolver requires a square operator")
        self._operator = K

    def solve(self, rhs: Array) -> Array:
        if self._operator is None:
            raise RuntimeError("factor() must be called before solve()")
        K = self._operator
        dim = K.shape[0]
        if int(rhs.shape[0]) != dim:
            raise ValueError("right-hand side dimension does not match operator")

        xp = array_namespace(rhs)
        rtol = self._effective_rtol()
        max_iter = (
            self._options.max_iter
            if self._options.max_iter is not None
            else 2 * dim + 100
        )

        # ``preconditioner="auto"``: run with the current effective preconditioner;
        # on a convergence failure the condensed Woodbury could rescue, promote and
        # retry the same solve once, then (on success) promote for the next solve if
        # this one was slow. In any non-auto mode this is a single ``_dispatch`` call.
        self._exact_inverse_active = False
        try:
            solution = self._dispatch(K, rhs, xp, max_iter, rtol)
        except KrylovConvergenceError:
            if self._exact_inverse_active:
                # The exact Woodbury inverse broke down (numerically singular
                # L-BFGS middle matrix): retry this solve on plain Jacobi, the
                # route these modes took before the exact inverse existed, and
                # stay there for the rest of this solver's life.
                self._exact_inverse_blocked = True
                self._exact_inverse_active = False
                solution = self._dispatch(K, rhs, xp, max_iter, rtol)
            elif not self._auto_can_promote(K):
                raise
            else:
                self._auto_promoted = True
                solution = self._dispatch(K, rhs, xp, max_iter, rtol)
        else:
            self._auto_promote_if_slow(K, max_iter)
        return solution

    def _dispatch(
        self,
        K: LinearOperator,
        rhs: Array,
        xp: Namespace,
        max_iter: int,
        rtol: float,
    ) -> Array:
        """Route to CG / MINRES / GMRES per method, preferred route, and pc."""
        method = self._options.method
        preferred = K.preferred_krylov_method()
        # A non-diagonal L-BFGS block preconditioner diag(N⁻¹, S⁻¹) for the
        # indefinite saddle can only be applied by GMRES (left preconditioning);
        # MINRES admits a diagonal only. So on the default (``cg``) route, when the
        # saddle offers that preconditioner and it is in force, take GMRES instead
        # of the CG→MINRES fallback. An explicit ``minres``/``gmres`` is honored.
        if (
            method == "cg"
            and self._effective_preconditioner() == "lbfgs"
            and preferred == "minres"
        ):
            return self._gmres(K, rhs, xp, max_iter, rtol)
        if method == "gmres":
            return self._gmres(K, rhs, xp, max_iter, rtol)
        if method == "minres":
            # Explicit MINRES is honored verbatim — no fallback.
            return self._preconditioned_minres(K, rhs, xp, max_iter, rtol)
        if method == "cg" and preferred == "minres":
            # Default cg route on an equality saddle. MINRES is fragile on
            # ill-conditioned indefinite saddles — it can return a garbage/non-finite
            # step at iteration 1, which the driver answers with an unbounded (and
            # counterproductive) δ_w escalation → numerical_error. GMRES minimizes the
            # true residual with restarts and is markedly more robust; and the SPD
            # *diagonal* (incl. the approximate-Schur dual block) actively hurts GMRES
            # on these saddles, so fall back **unpreconditioned** (S2MPJ Task 7:
            # HAGER*/DTOC*/CATENARY numerical_error → optimal).
            try:
                return self._preconditioned_minres(K, rhs, xp, max_iter, rtol)
            except KrylovConvergenceError:
                return self._gmres(K, rhs, xp, max_iter, rtol, minv=lambda r: r)
        if method != "cg":
            raise ValueError("Krylov method must be 'cg', 'minres', or 'gmres'")

        # method == "cg": prefer CG, fall back to MINRES on indefiniteness.
        if self._exact_inverse_eligible(K):
            # The Woodbury apply is the exact N⁻¹ here — solve directly instead
            # of paying the CG loop's inner products and host syncs around it.
            return self._exact_inverse_solve(K, rhs, xp, max_iter, rtol)
        precond = self._make_preconditioner(K, rhs, xp)
        try:
            return self._cg(K, rhs, xp, max_iter, rtol, precond)
        except _IndefiniteOperatorError:
            return self._preconditioned_minres(K, rhs, xp, max_iter, rtol)

    # -- auto preconditioner promotion (§5.2) -----------------------------

    def _auto_open(self) -> bool:
        """Whether an ``"auto"`` solver is still eligible to promote at all."""
        return self._options.preconditioner == "auto" and not self._auto_promoted

    def _auto_can_promote(self, K: LinearOperator) -> bool:
        """Whether auto may switch Jacobi → the condensed Woodbury inverse here.

        The *only* promotion target under either trigger (failure rescue or a
        slow-but-successful solve) is the **near-exact condensed Woodbury inverse**
        ``N⁻¹`` (equality-free), whose preconditioning reliably helps. The
        *approximate* saddle block preconditioner is never promoted to: its
        approximate Schur diagonal can yield worse steps than the slow-but-stable
        Jacobi solve and outright *diverge* a saddle Jacobi handles (S2MPJ
        HS109/CLNLBEAM/ACOPP). The block stays reachable via explicit
        ``preconditioner="lbfgs"``; auto leaves saddle robustness to the GMRES
        fallback in ``_dispatch``.
        """
        return self._auto_open() and self._lbfgs_condensed_available(K)

    def _auto_promote_if_slow(self, K: LinearOperator, max_iter: int) -> None:
        """Promote after a slow-but-successful solve (iterations over threshold).

        The cheap slowness test decides *first*: the availability probe inside
        :meth:`_auto_can_promote` builds the full Woodbury factors (O(n·k²)),
        which a fast successful solve must not pay on every iteration.
        """
        if not self._auto_open():
            return
        if self.last_iterations <= self._options.auto_switch_ratio * max_iter:
            return
        if self._lbfgs_condensed_available(K):
            self._auto_promoted = True

    def _lbfgs_condensed_available(self, K: LinearOperator) -> bool:
        """True when ``K`` offers the near-exact condensed Woodbury inverse ``N⁻¹``.

        This is the equality-free condensed operator only; the equality saddle
        exposes the block preconditioner instead, not this near-exact inverse.
        """
        try:
            K.lbfgs_inverse_apply()
        except NotImplementedError:
            return False
        return True

    # -- preconditioning --------------------------------------------------

    def _exact_inverse_eligible(self, K: LinearOperator) -> bool:
        """Whether this solve may use the operator's exact Woodbury ``N⁻¹``.

        The same gate for the direct dispatch and the GMRES preconditioner: the
        effective mode is the default Jacobi (an explicit ``none``/``lbfgs``
        keeps its documented behavior), the ``exact_lbfgs_inverse`` A/B lever
        is on, no earlier breakdown blocked it, and the operator reports its
        Woodbury apply as the exact inverse (bound-only L-BFGS block).
        """
        return (
            self._effective_preconditioner() == "jacobi"
            and self._options.exact_lbfgs_inverse
            and not self._exact_inverse_blocked
            and K.lbfgs_inverse_is_exact()
        )

    def _exact_inverse_solve(
        self, K: LinearOperator, b: Array, xp: Namespace, max_iter: int, rtol: float
    ) -> Array:
        """Direct solve via the exact condensed Woodbury inverse (§5.2).

        Bound-only L-BFGS systems: ``x = N⁻¹ b`` in one apply (Byrd, Nocedal &
        Schnabel 1994 compact form). Wrapping this apply in CG — the previous
        route — paid the loop's inner products (three host syncs) and vector
        updates just to confirm convergence; here one true-residual check does
        that, with up to ``_MAX_EXACT_APPLIES − 1`` working-precision
        iterative-refinement rounds ``x += N⁻¹ r`` (Carson & Higham 2018)
        covering round-off on an ill-conditioned late-barrier ``D̃``.
        (RT-scale study, n = 50k: 27 CG iterations + 17 ms Jacobi diagonal →
        one verified apply; step solve 34 → 6 ms.)

        Refinement is *weaker* per apply than a CG iteration (which is
        Galerkin-optimal over the preconditioned Krylov space), so a stalled
        or non-finite refinement never fails the solve here: it falls back to
        exactly the pre-direct route — CG preconditioned with the same inverse
        — whose breakdown paths keep the established semantics (sticky Jacobi
        via :meth:`solve`). A backend-native error out of the Woodbury apply
        (an exactly singular L-BFGS middle matrix) is converted to
        :class:`KrylovConvergenceError` so it takes that same path instead of
        escaping the driver's δ_w ladder.

        ``last_iterations`` counts Woodbury applies on the fast path (1 in the
        regular case) with ``last_method="direct"``; the fallback records as
        CG/MINRES, exactly as before this fast path existed.
        """
        self._exact_inverse_active = True
        b_norm = _norm(xp, b)
        if b_norm == 0.0:
            self._record(0, 0.0, "direct")
            return xp.zeros_like(b)
        tol = rtol * b_norm

        raw_apply = K.lbfgs_inverse_apply()

        def apply_inverse(v: Array) -> Array:
            try:
                return raw_apply(v)
            except Exception as exc:  # backend-native LinAlgError and kin
                if _is_resource_failure(exc):
                    # Out-of-memory is not a singular window: relabeling it
                    # would sticky-disable the fast path and hide the cause
                    # behind a slower Jacobi retry.
                    raise
                raise KrylovConvergenceError(
                    f"exact Woodbury apply failed: {exc}"
                ) from exc

        x = apply_inverse(b)
        for applies in range(1, _MAX_EXACT_APPLIES + 1):
            r = b - K.matvec(x)
            r_norm = _norm(xp, r)
            if r_norm <= tol:
                self._record(applies, r_norm, "direct")
                return x
            if not math.isfinite(r_norm):
                break  # a non-finite apply: refinement cannot recover it
            if applies < _MAX_EXACT_APPLIES:
                x = x + apply_inverse(r)
        try:
            return self._cg(K, b, xp, max_iter, rtol, apply_inverse)
        except _IndefiniteOperatorError:
            return self._preconditioned_minres(K, b, xp, max_iter, rtol)

    def _make_preconditioner(
        self, K: LinearOperator, rhs: Array, xp: Namespace
    ) -> Callable[[Array], Array]:
        """Build the ``M⁻¹`` apply for CG/GMRES; identity when unavailable (§5.2).

        ``lbfgs`` uses the operator's Sherman–Morrison–Woodbury inverse when it
        exposes an L-BFGS compact form, otherwise it degrades to Jacobi. ``jacobi``
        uses a strictly positive (SPD) diagonal — the saddle's SPD block diagonal
        when offered, else the operator's own diagonal.
        """
        mode = self._effective_preconditioner()
        if mode == "none":
            return lambda r: r
        if self._exact_inverse_eligible(K):
            # Bound-only L-BFGS block (no inequality Gram term): the
            # Sherman–Morrison–Woodbury apply (§5.2; Byrd, Nocedal & Schnabel
            # 1994 compact form) is the *exact* ``N⁻¹`` — strictly better than
            # Jacobi, whose O(n·k²) L-BFGS diagonal costs the same order as
            # the Woodbury factor plus two host syncs. The default CG dispatch
            # short-circuits to the *direct* ``_exact_inverse_solve`` (which
            # carries the RT-scale measurements) before ever building a
            # preconditioner, so this branch serves the explicit
            # ``method="gmres"`` route.
            self._exact_inverse_active = True
            return K.lbfgs_inverse_apply()
        if mode == "lbfgs":
            # Prefer the saddle block preconditioner diag(N⁻¹, S⁻¹) (non-diagonal,
            # GMRES-only); else the condensed Woodbury inverse; else Jacobi.
            try:
                return K.lbfgs_block_preconditioner_apply()
            except NotImplementedError:
                pass
            try:
                return K.lbfgs_inverse_apply()
            except NotImplementedError:
                pass  # no L-BFGS structure here → fall back to Jacobi
        elif mode != "jacobi":
            raise ValueError(
                "Krylov preconditioner must be 'none', 'jacobi', or 'lbfgs'"
            )
        d = self._spd_diagonal(K, rhs, xp)
        if d is None:
            return lambda r: r
        inv_d = 1.0 / d
        return lambda r: inv_d * r

    def _spd_diagonal(
        self, K: LinearOperator, like: Array, xp: Namespace
    ) -> Array | None:
        """A strictly positive diagonal for Jacobi, or ``None`` if unavailable.

        Prefers the operator's SPD preconditioner diagonal (the equality saddle's
        PD-primal / approximate-Schur-dual blocks); otherwise its plain diagonal.
        Anything non-positive or unavailable disables preconditioning.
        """
        try:
            d = K.spd_preconditioner_diagonal()
        except NotImplementedError:
            try:
                d = K.diagonal(like)
            except NotImplementedError:
                return None
        if not bool(xp.all(d > 0.0)):
            return None
        return d

    # -- iterative methods ------------------------------------------------

    def _cg(
        self,
        K: LinearOperator,
        b: Array,
        xp: Namespace,
        max_iter: int,
        rtol: float,
        precond: Callable[[Array], Array],
    ) -> Array:
        """Preconditioned conjugate gradients (Hestenes & Stiefel 1952)."""
        x = xp.zeros_like(b)
        b_norm = _norm(xp, b)
        if b_norm == 0.0:
            self._record(0, 0.0, "cg")
            return x

        r = b  # residual at x = 0
        z = precond(r)
        p = z
        rz = _inner(xp, r, z)
        # For an SPD preconditioner ⟨r, M⁻¹r⟩ > 0 whenever r ≠ 0; an exact zero
        # (underflow on an ill-scaled problem — MGH09LS reached 0/0 in the β
        # update under pc=auto — or an annihilating approximate M⁻¹), a negative
        # value or a non-finite one is a breakdown of the preconditioned inner
        # product: surface it as non-convergence so the driver escalates δ_w
        # instead of crashing with ZeroDivisionError.
        if not (rz > 0.0 and math.isfinite(rz)):
            raise KrylovConvergenceError(
                f"CG breakdown: preconditioned inner product {rz:.3e} at entry"
            )
        tol = rtol * b_norm
        r_norm = b_norm

        for it in range(1, max_iter + 1):
            kp = K.matvec(p)
            p_kp = _inner(xp, p, kp)
            if p_kp <= 0.0:
                # Non-positive curvature ⇒ K is not PD; CG is invalid here.
                raise _IndefiniteOperatorError
            alpha = rz / p_kp
            x = x + alpha * p
            r = r - alpha * kp
            r_norm = _norm(xp, r)
            if r_norm <= tol:
                self._record(it, r_norm, "cg")
                return x
            z = precond(r)
            rz_next = _inner(xp, r, z)
            if not (rz_next > 0.0 and math.isfinite(rz_next)):
                raise KrylovConvergenceError(
                    f"CG breakdown: preconditioned inner product {rz_next:.3e} "
                    f"at iteration {it} (residual {r_norm:.3e})",
                    iterate=x,
                )
            beta = rz_next / rz
            p = z + beta * p
            rz = rz_next

        # The truncated iterate rides along: CG minimizes the energy norm over
        # the Krylov space it has built, so it is a descent direction for the
        # SPD quadratic model (Steihaug 1983) even when short of ``rtol``.
        raise KrylovConvergenceError(
            f"CG did not converge in {max_iter} iterations "
            f"(residual {r_norm:.3e}, tolerance {tol:.3e})",
            iterate=x,
        )

    def _gmres(
        self,
        K: LinearOperator,
        b: Array,
        xp: Namespace,
        max_iter: int,
        rtol: float,
        minv: Callable[[Array], Array] | None = None,
    ) -> Array:
        """Restarted, left-preconditioned GMRES(m) (Saad & Schultz 1986).

        Arnoldi with Householder-free modified Gram–Schmidt and incremental Givens
        rotations; the small Hessenberg/rotation bookkeeping is kept in Python
        floats so the kernel stays backend-agnostic (matvecs + vector ops only).
        Convergence is decided on the *true* relative residual ``‖b − Kx‖`` at each
        restart, with the cheap Arnoldi estimate ending a cycle early. Works for
        symmetric *indefinite* systems too, so it is a robust alternative to MINRES.

        ``minv`` overrides the preconditioner apply; the MINRES-failure fallback
        passes the identity to run **unpreconditioned** GMRES (the SPD diagonal
        hurts GMRES on ill-conditioned equality saddles — S2MPJ Task 7).
        """
        if minv is None:
            minv = self._make_preconditioner(K, b, xp)
        restart = self._options.gmres_restart

        x = xp.zeros_like(b)
        b_norm = _norm(xp, b)
        if b_norm == 0.0:
            self._record(0, 0.0, "gmres")
            return x
        tol = rtol * b_norm
        # Preconditioned tolerance for the inner cycle's cheap estimate.
        precond_tol = rtol * _norm(xp, minv(b))

        total = 0
        while total < max_iter:
            residual = b - K.matvec(x)
            res_norm = _norm(xp, residual)
            if res_norm <= tol:
                self._record(total, res_norm, "gmres")
                return x

            z0 = minv(residual)
            beta = _norm(xp, z0)
            if beta == 0.0:
                break
            v = [z0 / beta]
            g = [beta]
            r_cols: list[list[float]] = []  # rotated upper-Hessenberg columns
            cs: list[float] = []
            sn: list[float] = []

            cycle = min(restart, max_iter - total)
            for j in range(cycle):
                total += 1
                w = minv(K.matvec(v[j]))
                h = []
                for i in range(j + 1):
                    hij = _inner(xp, w, v[i])
                    w = w - hij * v[i]
                    h.append(hij)
                h_next = _norm(xp, w)
                # Apply the accumulated rotations to the new column.
                for i in range(j):
                    h[i], h[i + 1] = (
                        cs[i] * h[i] + sn[i] * h[i + 1],
                        -sn[i] * h[i] + cs[i] * h[i + 1],
                    )
                c, s = _givens(h[j], h_next)
                cs.append(c)
                sn.append(s)
                h[j] = c * h[j] + s * h_next
                r_cols.append(h)
                g.append(-s * g[j])
                g[j] = c * g[j]

                if h_next > 0.0 and math.isfinite(h_next):
                    v.append(w / h_next)
                # End the cycle on convergence, an Arnoldi (happy) breakdown
                # ``h_next == 0``, or a non-finite operator image (NaN/Inf from a bad
                # iterate) — continuing would index a basis vector never appended.
                # A non-finite residual leaves the outer true-residual check below to
                # raise ``KrylovConvergenceError`` so the driver escalates δ_w.
                if (
                    not math.isfinite(h_next)
                    or h_next == 0.0
                    or abs(g[j + 1]) <= precond_tol
                ):
                    break

            # Back-substitution on the upper-triangular rotated Hessenberg.
            k = len(r_cols)
            y = [0.0] * k
            for i in reversed(range(k)):
                acc = g[i]
                for jj in range(i + 1, k):
                    acc -= r_cols[jj][i] * y[jj]
                diag = r_cols[i][i]
                # The rotated diagonal is √(h[i]² + h_next²), so it vanishes only
                # when K·vᵢ vanished after orthogonalization (a singular operator,
                # e.g. an identically-zero exact Hessian at the start point) — the
                # projected least-squares column is rank-deficient. Take the
                # minimal-norm choice yᵢ = 0 and let the outer true-residual check
                # decide, raising KrylovConvergenceError so the driver escalates
                # δ_w instead of crashing on a raw ZeroDivisionError.
                if diag == 0.0 or not math.isfinite(diag):
                    y[i] = 0.0
                    continue
                y[i] = acc / diag
            for i in range(k):
                x = x + y[i] * v[i]

        res_norm = _true_residual_norm(xp, K, x, b)
        if res_norm <= tol:
            self._record(total, res_norm, "gmres")
            return x
        raise KrylovConvergenceError(
            f"GMRES did not converge in {max_iter} iterations "
            f"(residual {res_norm:.3e}, tolerance {tol:.3e})",
            iterate=x,
        )

    def _minres_preconditioner_diagonal(
        self, K: LinearOperator, b: Array, xp: Namespace
    ) -> Array | None:
        """SPD diagonal preconditioner for MINRES, or ``None`` for unpreconditioned.

        MINRES applies a preconditioner by symmetric scaling, which only admits a
        *diagonal* SPD ``D`` — so the non-diagonal ``lbfgs`` inverse is not
        representable here and degrades to the same SPD diagonal as ``jacobi``.
        """
        mode = self._effective_preconditioner()
        if mode == "none":
            return None
        if mode not in ("jacobi", "lbfgs"):
            raise ValueError(
                "Krylov preconditioner must be 'none', 'jacobi', or 'lbfgs'"
            )
        return self._spd_diagonal(K, b, xp)

    def _preconditioned_minres(
        self,
        K: LinearOperator,
        b: Array,
        xp: Namespace,
        max_iter: int,
        rtol: float,
    ) -> Array:
        """MINRES with an SPD diagonal preconditioner via symmetric scaling.

        A symmetric positive-definite diagonal ``D`` is applied by solving the
        symmetrically scaled system ``K̂ x̂ = b̂`` with ``K̂ = D^{-1/2} K D^{-1/2}``
        (still symmetric), then recovering ``x = D^{-1/2} x̂``. This is
        mathematically preconditioned MINRES while reusing the unpreconditioned
        kernel unchanged. Falls back to plain MINRES when no SPD diagonal exists.
        """
        d = self._minres_preconditioner_diagonal(K, b, xp)
        if d is None:
            return self._minres(K, b, xp, max_iter, rtol)
        scale = 1.0 / xp.sqrt(d)
        scaled = MatrixFreeJacobian(K.shape, lambda v: scale * K.matvec(scale * v))
        x_hat = self._minres(scaled, scale * b, xp, max_iter, rtol)
        return scale * x_hat

    def _minres(
        self,
        K: LinearOperator,
        b: Array,
        xp: Namespace,
        max_iter: int,
        rtol: float,
    ) -> Array:
        """Unpreconditioned MINRES for symmetric (possibly indefinite) ``K``.

        Lanczos tridiagonalization with Givens rotations (Paige & Saunders 1975),
        following the SOL/SciPy reference loop. The estimated residual norm is the
        rotation residual ``phibar``; we stop on a relative-residual criterion.
        """
        x = xp.zeros_like(b)
        beta1 = _norm(xp, b)
        if beta1 == 0.0:
            self._record(0, 0.0, "minres")
            return x
        eps = float(xp.finfo(b.dtype).eps)

        # Lanczos / rotation state.
        r1 = b
        r2 = b
        y = b
        oldb = 0.0
        beta = beta1
        dbar = 0.0
        epsln = 0.0
        phibar = beta1
        cs = -1.0
        sn = 0.0
        w = xp.zeros_like(b)
        w2 = xp.zeros_like(b)

        tol = rtol * beta1

        for itn in range(1, max_iter + 1):
            # --- Lanczos step: generate v_k and the next residual r2. ---
            v = (1.0 / beta) * y
            y = K.matvec(v)
            if itn >= 2:
                y = y - (beta / oldb) * r1
            alfa = _inner(xp, v, y)
            y = y - (alfa / beta) * r2
            r1 = r2
            r2 = y
            oldb = beta
            beta = _norm(xp, r2)

            # --- apply the previous rotation, then compute the new one. ---
            oldeps = epsln
            delta = cs * dbar + sn * alfa
            gbar = sn * dbar - cs * alfa
            epsln = sn * beta
            dbar = -cs * beta

            gamma = (gbar * gbar + beta * beta) ** 0.5
            gamma = max(gamma, eps)
            cs = gbar / gamma
            sn = beta / gamma
            phi = cs * phibar
            phibar = sn * phibar

            # --- update the solution. ---
            denom = 1.0 / gamma
            w1 = w2
            w2 = w
            w = (v - oldeps * w1 - delta * w2) * denom
            x = x + phi * w

            if phibar <= tol:
                true_residual = _true_residual_norm(xp, K, x, b)
                if true_residual <= tol:
                    self._record(itn, true_residual, "minres")
                    return x
            if beta <= eps:  # Lanczos breakdown ⇒ the Krylov space is exhausted.
                true_residual = _true_residual_norm(xp, K, x, b)
                if true_residual <= tol:
                    self._record(itn, true_residual, "minres")
                    return x
                raise KrylovConvergenceError(
                    "MINRES Lanczos breakdown before convergence "
                    f"(residual {true_residual:.3e}, tolerance {tol:.3e})",
                    iterate=x,
                )

        true_residual = _true_residual_norm(xp, K, x, b)
        raise KrylovConvergenceError(
            f"MINRES did not converge in {max_iter} iterations "
            f"(residual {true_residual:.3e}, tolerance {tol:.3e})",
            iterate=x,
        )

    def _record(self, iterations: int, residual: float, method: str) -> None:
        self.last_iterations = iterations
        self.last_residual = residual
        self.last_method = method


__all__ = ["KrylovConvergenceError", "KrylovSolver"]
