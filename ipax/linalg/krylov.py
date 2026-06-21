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
  applied to MINRES by symmetric scaling.
- ``lbfgs`` — an L-BFGS-aware Sherman–Morrison–Woodbury inverse of the condensed
  operator (``lbfgs_inverse_apply``), an SPD operator used directly by CG/GMRES;
  it degrades to ``jacobi`` where no L-BFGS compact form is available (and on the
  MINRES path, which can only apply a diagonal preconditioner).
All preconditioners fall back to none when no suitable structure exists.

References: Hestenes & Stiefel 1952 (CG); Paige & Saunders 1975 (MINRES); Saad &
Schultz 1986 (GMRES); Murphy, Golub & Wathen 2000 (block preconditioning of
saddle systems); Byrd, Nocedal & Schnabel 1994 (compact L-BFGS).
"""

from __future__ import annotations

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

    def describe(self) -> str:
        """Human-readable label for diagnostics, incl. method/preconditioner."""
        return f"krylov ({self._options.method}, pc={self._options.preconditioner})"

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
        rtol = self._options.rtol
        max_iter = (
            self._options.max_iter
            if self._options.max_iter is not None
            else 2 * dim + 100
        )

        method = self._options.method
        preferred = K.preferred_krylov_method()
        if method == "gmres":
            return self._gmres(K, rhs, xp, max_iter, rtol)
        if method == "minres" or (method == "cg" and preferred == "minres"):
            return self._preconditioned_minres(K, rhs, xp, max_iter, rtol)
        if method != "cg":
            raise ValueError("Krylov method must be 'cg', 'minres', or 'gmres'")

        # method == "cg": prefer CG, fall back to MINRES on indefiniteness.
        precond = self._make_preconditioner(K, rhs, xp)
        try:
            return self._cg(K, rhs, xp, max_iter, rtol, precond)
        except _IndefiniteOperatorError:
            return self._preconditioned_minres(K, rhs, xp, max_iter, rtol)

    # -- preconditioning --------------------------------------------------

    def _make_preconditioner(
        self, K: LinearOperator, rhs: Array, xp: Namespace
    ) -> Callable[[Array], Array]:
        """Build the ``M⁻¹`` apply for CG/GMRES; identity when unavailable (§5.2).

        ``lbfgs`` uses the operator's Sherman–Morrison–Woodbury inverse when it
        exposes an L-BFGS compact form, otherwise it degrades to Jacobi. ``jacobi``
        uses a strictly positive (SPD) diagonal — the saddle's SPD block diagonal
        when offered, else the operator's own diagonal.
        """
        mode = self._options.preconditioner
        if mode == "none":
            return lambda r: r
        if mode == "lbfgs":
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
            beta = rz_next / rz
            p = z + beta * p
            rz = rz_next

        raise KrylovConvergenceError(
            f"CG did not converge in {max_iter} iterations "
            f"(residual {r_norm:.3e}, tolerance {tol:.3e})"
        )

    def _gmres(
        self, K: LinearOperator, b: Array, xp: Namespace, max_iter: int, rtol: float
    ) -> Array:
        """Restarted, left-preconditioned GMRES(m) (Saad & Schultz 1986).

        Arnoldi with Householder-free modified Gram–Schmidt and incremental Givens
        rotations; the small Hessenberg/rotation bookkeeping is kept in Python
        floats so the kernel stays backend-agnostic (matvecs + vector ops only).
        Convergence is decided on the *true* relative residual ``‖b − Kx‖`` at each
        restart, with the cheap Arnoldi estimate ending a cycle early. Works for
        symmetric *indefinite* systems too, so it is a robust alternative to MINRES.
        """
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

                if h_next > 0.0:
                    v.append(w / h_next)
                if abs(g[j + 1]) <= precond_tol or h_next == 0.0:
                    break

            # Back-substitution on the upper-triangular rotated Hessenberg.
            k = len(r_cols)
            y = [0.0] * k
            for i in reversed(range(k)):
                acc = g[i]
                for jj in range(i + 1, k):
                    acc -= r_cols[jj][i] * y[jj]
                y[i] = acc / r_cols[i][i]
            for i in range(k):
                x = x + y[i] * v[i]

        res_norm = _true_residual_norm(xp, K, x, b)
        if res_norm <= tol:
            self._record(total, res_norm, "gmres")
            return x
        raise KrylovConvergenceError(
            f"GMRES did not converge in {max_iter} iterations "
            f"(residual {res_norm:.3e}, tolerance {tol:.3e})"
        )

    def _minres_preconditioner_diagonal(
        self, K: LinearOperator, b: Array, xp: Namespace
    ) -> Array | None:
        """SPD diagonal preconditioner for MINRES, or ``None`` for unpreconditioned.

        MINRES applies a preconditioner by symmetric scaling, which only admits a
        *diagonal* SPD ``D`` — so the non-diagonal ``lbfgs`` inverse is not
        representable here and degrades to the same SPD diagonal as ``jacobi``.
        """
        mode = self._options.preconditioner
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
                    f"(residual {true_residual:.3e}, tolerance {tol:.3e})"
                )

        true_residual = _true_residual_norm(xp, K, x, b)
        raise KrylovConvergenceError(
            f"MINRES did not converge in {max_iter} iterations "
            f"(residual {true_residual:.3e}, tolerance {tol:.3e})"
        )

    def _record(self, iterations: int, residual: float, method: str) -> None:
        self.last_iterations = iterations
        self.last_residual = residual
        self.last_method = method


__all__ = ["KrylovConvergenceError", "KrylovSolver"]
