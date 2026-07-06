"""S2MPJ problem loader (download-gated, NumPy-evaluated, backend-bridged).

`S2MPJ <https://github.com/GrattonToint/S2MPJ>`_ translates the CUTEst SIF test
collection (including the full Hock-Schittkowski set) into **pure Python** problem
files — no Fortran, no SIF decoder, no compilation, cross-platform. Each problem
exposes value/gradient/Jacobian/Hessian through a ``CUTEst_problem`` base class
(``fgx``, ``cJx``, ``LHxyv`` …) using NumPy + SciPy.

Because that evaluation is hardcoded to NumPy/SciPy, the problems cannot run
*natively* under an arbitrary Array-API backend. :class:`_S2MPJProblem` instead
**bridges**: it evaluates in NumPy and converts inputs/outputs to/from the target
namespace, so ipax's solver runs its linear algebra on any (CPU) backend while the
model is evaluated on the host. This is for accuracy / cross-backend consistency
benchmarking; it forces a host sync per evaluation, so it is *not* for GPU
performance work (use the synthetic RT generator / TROTS surrogate there).

S2MPJ's generic ``evalgrsum`` evaluation loop is interpretive (per-element
``eval()`` dispatch, ``lil_matrix`` row assembly) and dominates sweep wall-time,
so the bridge routes ``fx/fgx/cx/cJx`` — and the exact-Hessian ``LgHxy`` —
through the **precompiled evaluator** in :mod:`benchmarks.corpus._s2mpj_fast`
whenever it verifies against the original methods at build time (else it falls
back to the originals; the Hessian is gated by its own verification, so it can
fall back alone — see that module's docstring for the mechanism and measured
speedups).

S2MPJ has no license file, so its files are **not vendored**. Point
``IPAX_S2MPJ_DIR`` (or the ``directory`` argument) at a local checkout; the loader
returns ``[]`` when it is absent, mirroring the other external corpora.

S2MPJ stores each constraint as a two-sided range ``clower ≤ c(x) ≤ cupper`` with
an equality when the two coincide. The adapter maps that onto ipax's eq/ineq
split, lowering finite inequality sides to one-sided rows (``c − cupper ≤ 0`` and
``clower − c ≤ 0``) exactly as the native ``linear_ineq`` lowering does.

S2MPJ also exposes the **exact Lagrangian Hessian** (``LgHxy``/``LHxyv``, convention
``L = f + yᵀc``), so the adapter can drive ipax's exact-Hessian route — not only the
default L-BFGS. :class:`_S2MPJExactProblem` implements ``lagrangian_hessian`` by
mapping ipax's ``(σ, y_eq, y_ineq)`` onto S2MPJ's single multiplier vector ``Y``
(equality rows ``+y_eq``; lowered lower-side rows ``−y_ineq`` because their curvature
is ``−∇²c``; lowered upper-side rows ``+y_ineq``) and honoring ``σ`` on the objective
term (so it stays correct under gradient-based scaling, where ``σ ≠ 1``). With
``sparse=True`` the Jacobians and the Hessian are returned as
:class:`~ipax.backend.sparse.numpy_scipy.SparseOperator` (true COO sparsity, for the
sparse-direct route) rather than densified arrays.
"""

from __future__ import annotations

import functools
import importlib
import os
import re
import sys
from typing import TYPE_CHECKING, Any

from ipax.problem.base import Problem

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarks.corpus import BenchmarkProblem
    from ipax.backend.operators import LinearOperator
    from ipax.typing import Array, Namespace, Scalar


def _to_numpy(x: Array) -> Any:
    """Convert an Array-API array to a 1-D NumPy array (host bridge)."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return np.reshape(x, (-1,))
    try:
        return np.reshape(np.from_dlpack(x), (-1,))
    except (TypeError, RuntimeError, BufferError, ValueError):
        return np.reshape(np.asarray(x), (-1,))


def _from_numpy(xp: Namespace, arr: Any) -> Array:
    """Convert a NumPy array back into the target namespace as float64 (1-/2-D)."""
    import numpy as np

    return xp.asarray(np.asarray(arr, dtype=np.float64))


class _S2MPJProblem(Problem):
    """Bridge a NumPy-evaluated S2MPJ problem onto an Array-API namespace.

    The wrapped ``instance`` is a CUTEst_problem from S2MPJ; ``xp`` is the target
    namespace. Equality vs inequality constraints and the finite inequality sides
    are precomputed once from ``clower``/``cupper``.

    ``sparse`` controls how Jacobians (and, in :class:`_S2MPJExactProblem`, the
    Hessian) cross the bridge: ``False`` densifies to Array-API arrays; ``True``
    wraps the native ``scipy.sparse`` structure in a ``SparseOperator`` so the
    sparse-direct route factors true sparsity. This base class supplies no
    Lagrangian Hessian, so the solver uses its default L-BFGS.
    """

    def __init__(
        self,
        instance: Any,
        xp: Namespace,
        *,
        sparse: bool = False,
        feasibility: bool = False,
    ) -> None:
        import numpy as np

        self._sparse = sparse
        # Dataset-sourced expected outcome (set by the loader from the source file):
        # the documented ``LO SOLTN`` objective, the ``infeasible`` marker, and the
        # CUTEst classification. Defaults here so a directly-constructed instance is
        # still valid; the loader overwrites them.
        self.expected_objective: float | None = None
        self.expected_infeasible: bool = False
        self.pbclass: str | None = None

        # A problem with no objective group is a *feasibility* / nonlinear-equations
        # system, not a minimization problem (S2MPJ's ``fx`` would error). By default
        # we reject it; with ``feasibility=True`` we run it as ``min 0`` s.t. the
        # constraints, turning the IPM into a feasibility finder. ``H`` is the
        # optional explicit quadratic objective.
        self._no_objective = len(getattr(instance, "objgrps", ())) == 0 and not hasattr(
            instance, "H"
        )
        if self._no_objective and not feasibility:
            raise NotImplementedError("S2MPJ problem has no objective function")

        self._inst = instance
        self.xp = xp
        self._n = int(instance.n)
        m = int(getattr(instance, "m", 0))
        self._m = m

        # Precompiled evaluator, verified against the original methods at build
        # time (None → keep the originals). Cached on the shared instance, so the
        # per-config rebuilds of the same problem verify once.
        from benchmarks.corpus._s2mpj_fast import fast_evaluator

        self._fast = fast_evaluator(instance)

        # Per-point memo of the full constraint value/Jacobian. S2MPJ bundles all
        # constraints in one ``cx``/``cJx`` call (each ~10–16 ms of pure-Python
        # ``eval`` + scipy-sparse assembly — ~100× the objective), and the IPM
        # evaluates equalities and inequalities at the *same* point back-to-back, so
        # a size-1 cache keyed on the point halves the dominant benchmark cost.
        self._cx_key: bytes | None = None
        self._cx_val: Any = None
        self._cJx_key: bytes | None = None
        self._cJx_val: Any = None
        # Same-point memo for (f, ∇f): the driver requests the gradient right
        # after accepting a point whose objective the line search just computed
        # (and fgx returns both), so one evaluation serves both calls.
        self._fgx_key: bytes | None = None
        self._fgx_f: float = 0.0
        self._fgx_g: Any = None

        # ``clower``/``cupper`` exist only when the problem has constraints.
        clower_attr = getattr(instance, "clower", None)
        cupper_attr = getattr(instance, "cupper", None)
        if m > 0 and clower_attr is not None and cupper_attr is not None:
            clower = np.reshape(np.asarray(clower_attr, dtype=float), (-1,))
            cupper = np.reshape(np.asarray(cupper_attr, dtype=float), (-1,))
            finite_l = np.isfinite(clower)
            finite_u = np.isfinite(cupper)
            eq_mask = finite_l & finite_u & (clower == cupper)
            self._eq_idx = np.where(eq_mask)[0]
            self._eq_rhs = clower[self._eq_idx]
            self._lo_idx = np.where(finite_l & ~eq_mask)[0]  # clower ≤ c ⇒ clower−c ≤ 0
            self._up_idx = np.where(finite_u & ~eq_mask)[0]  # c ≤ cupper ⇒ c−cupper ≤ 0
            self._lo_rhs = clower[self._lo_idx]
            self._up_rhs = cupper[self._up_idx]
        else:
            empty_i = np.zeros((0,), dtype=int)
            empty_f = np.zeros((0,), dtype=float)
            self._eq_idx = self._lo_idx = self._up_idx = empty_i
            self._eq_rhs = self._lo_rhs = self._up_rhs = empty_f

        # Signed row selectors, built once: ``P @ J`` replaces per-call fancy
        # row-slicing + vstack when splitting the bundled Jacobian into the
        # eq/ineq blocks (lower rows carry −1: their lowering is ``clower − c``).
        import scipy.sparse as sp

        n_eq, n_lo, n_up = len(self._eq_idx), len(self._lo_idx), len(self._up_idx)
        self._P_eq = sp.csr_matrix(
            (np.ones(n_eq), (np.arange(n_eq), self._eq_idx)), shape=(n_eq, m)
        )
        self._P_ineq = sp.csr_matrix(
            (
                np.concatenate((-np.ones(n_lo), np.ones(n_up))),
                (
                    np.arange(n_lo + n_up),
                    np.concatenate((self._lo_idx, self._up_idx)),
                ),
            ),
            shape=(n_lo + n_up, m),
        )

        lower_attr = getattr(instance, "xlower", None)
        upper_attr = getattr(instance, "xupper", None)
        self._lower = (
            None
            if lower_attr is None
            else _from_numpy(xp, np.reshape(np.asarray(lower_attr, dtype=float), (-1,)))
        )
        self._upper = (
            None
            if upper_attr is None
            else _from_numpy(xp, np.reshape(np.asarray(upper_attr, dtype=float), (-1,)))
        )

    @property
    def n_vars(self) -> int:
        return self._n

    def bounds(self) -> tuple[Array | None, Array | None]:
        return self._lower, self._upper

    # -- evaluation dispatch: verified fast evaluator, else the original -----
    def _eval_fx(self, x_np: Any) -> Any:
        return (self._fast or self._inst).fx(x_np)

    def _eval_fgx(self, x_np: Any) -> Any:
        return (self._fast or self._inst).fgx(x_np)

    def _eval_cx(self, x_np: Any) -> Any:
        return (self._fast or self._inst).cx(x_np)

    def _eval_cJx(self, x_np: Any) -> Any:
        return (self._fast or self._inst).cJx(x_np)

    def _eval_lghxy(self, x_col: Any, y_col: Any) -> Any:
        # The Hessian fast path is gated by its own verification flag: a
        # Hessian-only mismatch keeps the fast fx/cx while LgHxy falls back.
        if self._fast is not None and self._fast.lghxy_ok:
            return self._fast.LgHxy(x_col, y_col)
        return self._inst.LgHxy(x_col, y_col)

    def objective(self, x: Array) -> Scalar:
        import numpy as np

        if self._no_objective:  # feasibility problem: minimize a constant 0
            return _from_numpy(self.xp, np.asarray(0.0))
        # S2MPJ's auto-generated evaluations use Python ``float**`` and can raise
        # OverflowError on the wild trial points a line search probes (e.g.
        # LUKVLE4C's ``100*GVAR**6``). Return +inf so the solver simply rejects
        # the trial rather than crashing. Also return an xp float64 0-d (not a
        # Python float) so the value keeps the backend dtype — a Python float
        # re-cast via ``xp.asarray`` would silently drop to float32 on Torch.
        x_np = _to_numpy(x)
        if self._fgx_key is not None and x_np.tobytes() == self._fgx_key:
            return _from_numpy(self.xp, np.asarray(self._fgx_f))
        try:
            val = np.asarray(self._eval_fx(x_np))
        except (OverflowError, FloatingPointError):
            val = np.asarray(np.inf)
        return _from_numpy(self.xp, val)

    def gradient(self, x: Array) -> Array:
        import numpy as np

        if self._no_objective:  # constant objective ⇒ zero gradient
            return _from_numpy(self.xp, np.zeros((self._n,)))
        x_np = _to_numpy(x)
        key = x_np.tobytes()
        if key != self._fgx_key:
            try:
                f, g = self._eval_fgx(x_np)
            except (OverflowError, FloatingPointError):
                return _from_numpy(self.xp, np.full((self._n,), np.inf))
            self._fgx_key = key
            self._fgx_f = float(np.asarray(f).reshape(-1)[0])
            self._fgx_g = np.reshape(np.asarray(g, dtype=float), (-1,))
        return _from_numpy(self.xp, self._fgx_g)

    # -- shared per-point constraint value / Jacobian (memoized) -----------
    def _cx(self, x_np: Any) -> Any:
        """S2MPJ full constraint vector ``c(x)``, memoized on the last point."""
        import numpy as np

        key = x_np.tobytes()
        if key != self._cx_key:
            self._cx_key = key
            self._cx_val = np.reshape(
                np.asarray(self._eval_cx(x_np), dtype=float), (-1,)
            )
        return self._cx_val

    def _cJx(self, x_np: Any) -> Any:
        """S2MPJ constraint Jacobian ``(c, ∇c)``, memoized on the last point."""
        import numpy as np

        key = x_np.tobytes()
        if key != self._cJx_key:
            self._cJx_key = key
            self._cJx_val = self._eval_cJx(x_np)
            # cJx returns c(x) alongside the Jacobian; seed the value memo so the
            # driver's same-point eq/ineq *value* requests skip a full cx call.
            c, _jac = self._cJx_val
            self._cx_key = key
            self._cx_val = np.reshape(np.asarray(c, dtype=float), (-1,))
        return self._cJx_val

    # -- equalities (present only when S2MPJ has clower == cupper rows) ----
    def eq_constraints(self, x: Array) -> Array:
        if self._eq_idx.shape[0] == 0:
            raise NotImplementedError
        c = self._cx(_to_numpy(x))
        return _from_numpy(self.xp, c[self._eq_idx] - self._eq_rhs)

    def eq_jacobian(self, x: Array) -> Array | LinearOperator:
        if self._eq_idx.shape[0] == 0:
            raise NotImplementedError
        _c, jac = self._cJx(_to_numpy(x))
        rows = self._P_eq @ jac.tocsr()
        if self._sparse:
            return self._sparse_op(rows)
        return _from_numpy(self.xp, rows.toarray())

    # -- inequalities (lowered finite sides of the two-sided ranges) -------
    def ineq_constraints(self, x: Array) -> Array:
        if self._lo_idx.shape[0] == 0 and self._up_idx.shape[0] == 0:
            raise NotImplementedError
        import numpy as np

        c = self._cx(_to_numpy(x))
        lo = self._lo_rhs - c[self._lo_idx]
        up = c[self._up_idx] - self._up_rhs
        return _from_numpy(self.xp, np.concatenate((lo, up)))

    def ineq_jacobian(self, x: Array) -> Array | LinearOperator:
        if self._lo_idx.shape[0] == 0 and self._up_idx.shape[0] == 0:
            raise NotImplementedError
        _c, jac = self._cJx(_to_numpy(x))
        # Lower side ``clower − c`` contributes ``−∇c``; upper side ``c − cupper``
        # contributes ``+∇c`` — both encoded in the signed selector.
        rows = self._P_ineq @ jac.tocsr()
        if self._sparse:
            return self._sparse_op(rows)
        return _from_numpy(self.xp, rows.toarray())

    # -- shared Hessian helpers (used by the exact subclass) ---------------
    def _sparse_op(self, matrix: Any) -> LinearOperator:
        """Wrap a host ``scipy.sparse`` matrix as a COO-exposing operator."""
        from ipax.backend.sparse.numpy_scipy import SparseOperator

        return SparseOperator(matrix, self.xp)

    def _s2mpj_multipliers(self, y_eq: Array, y_ineq: Array) -> Any:
        """Map ipax ``(y_eq, y_ineq)`` onto S2MPJ's per-constraint vector ``Y``.

        S2MPJ's Lagrangian is ``f + Σ_i Y_i c_i`` over the *original* constraint
        rows, so ``Σ_i Y_i ∇²c_i`` must reproduce ipax's curvature term
        ``Σ y_eq·∇²c + Σ y_ineq·∇²g``. Equality rows take ``+y_eq``; lowered
        lower-side rows take ``−y_ineq`` (their ``g = clower − c`` has
        ``∇²g = −∇²c``); lowered upper-side rows take ``+y_ineq``. A range
        constraint that appears in both blocks accumulates both contributions.
        """
        import numpy as np

        Y = np.zeros((self._m,), dtype=float)
        if self._m == 0:
            return Y
        yeq = _to_numpy(y_eq)
        yineq = _to_numpy(y_ineq)
        n_lo = int(self._lo_idx.shape[0])
        if self._eq_idx.shape[0]:
            np.add.at(Y, self._eq_idx, yeq)
        if n_lo:
            np.add.at(Y, self._lo_idx, -yineq[:n_lo])
        if self._up_idx.shape[0]:
            np.add.at(Y, self._up_idx, yineq[n_lo:])
        return Y

    def _lagrangian_hessian_matrix(
        self, x: Array, y_eq: Array, y_ineq: Array, sigma: Scalar
    ) -> Any:
        """Assemble ``σ∇²f + Σ y·∇²c`` as a host ``scipy.sparse`` matrix.

        ``LgHxy`` returns ``∇²f + Σ Y_i ∇²c_i`` (no objective scaling). For the
        common ``σ > 0`` case (gradient-based scaling passes ``σ = s_f`` through
        the scaling wrapper) the identity ``σ∇²f + Σ Y∇²c = σ·(∇²f + Σ(Y/σ)∇²c)``
        folds the objective scaling into the multipliers, so one ``LgHxy`` call
        suffices — ``LgHxy`` costs a full Hessian assembly (fast or original),
        so the old two-call ``(σ−1)∇²f`` correction doubled the exact-Hessian
        bridge cost. ``σ ≤ 0`` keeps the general two-call form.
        """
        import numpy as np

        s = float(sigma)
        xcol = np.reshape(_to_numpy(x), (-1, 1))
        Y = self._s2mpj_multipliers(y_eq, y_ineq)
        if s == 1.0:
            _lval, _grad, hess = self._eval_lghxy(xcol, np.reshape(Y, (-1, 1)))
        elif s > 0.0:
            _lval, _grad, hess = self._eval_lghxy(xcol, np.reshape(Y / s, (-1, 1)))
            hess = s * hess
        else:
            _lval, _grad, hess = self._eval_lghxy(xcol, np.reshape(Y, (-1, 1)))
            _l0, _g0, hess_f = self._eval_lghxy(xcol, np.zeros((self._m, 1)))
            hess = hess + (s - 1.0) * hess_f
        return hess


class _S2MPJExactProblem(_S2MPJProblem):
    """S2MPJ bridge that also supplies the **exact Lagrangian Hessian**.

    Defining ``lagrangian_hessian`` on the class advertises it through ipax's
    derivative resolution (``_provides`` compares against the base ``Problem``),
    so the solver takes the exact-Hessian route instead of L-BFGS. The matrix is
    returned dense or as a ``SparseOperator`` per the ``sparse`` flag.
    """

    def lagrangian_hessian(
        self,
        x: Array,
        y_eq: Array,
        y_ineq: Array,
        sigma: Scalar = 1.0,
    ) -> Array | LinearOperator:
        import numpy as np

        hess = self._lagrangian_hessian_matrix(x, y_eq, y_ineq, sigma)
        if self._sparse:
            import scipy.sparse as sp

            return self._sparse_op(sp.csr_matrix(hess))
        return _from_numpy(self.xp, np.asarray(hess.toarray()))


def s2mpj_dir(directory: str | None = None) -> str | None:
    """Resolve the S2MPJ checkout directory, or ``None`` if unavailable.

    Uses ``directory`` if given, else the ``IPAX_S2MPJ_DIR`` environment variable.
    A directory qualifies only if it holds ``s2mpjlib.py`` and ``python_problems``.
    """
    root = directory or os.environ.get("IPAX_S2MPJ_DIR")
    if not root:
        return None
    if not os.path.isfile(os.path.join(root, "s2mpjlib.py")):
        return None
    if not os.path.isdir(os.path.join(root, "python_problems")):
        return None
    return root


def list_s2mpj_problems(directory: str | None = None) -> list[str]:
    """All S2MPJ problem names available in the checkout (sorted).

    Globs ``python_problems/*.py`` (the actual files present, more reliable than
    the repo's ``list_of_python_problems`` which can include editor backups),
    dropping the ``s2mpjlib`` support module. Returns ``[]`` when no checkout is
    found. Note: not every name is benchmarkable — objective-free problems are
    rejected at build time, and the runner caps problem size.
    """
    import glob

    root = s2mpj_dir(directory)
    if root is None:
        return []
    pattern = os.path.join(root, "python_problems", "*.py")
    names = sorted(
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(pattern)
        if os.path.basename(path) != "s2mpjlib.py"
    )
    return names


@functools.lru_cache(maxsize=2048)
def _problem_metadata(root: str, name: str) -> tuple[str | None, float | None, bool]:
    """Parse ``(pbclass, expected_objective, expected_infeasible)`` from the source.

    These come from the dataset itself, not our own judgement: ``self.pbclass`` is
    the CUTEst classification; ``# LO SOLTN <value>`` is the SIF author's documented
    solution objective (present on ~72% of the corpus); and an explicit
    ``Solution (infeasible)`` / ``Source: an infeasible problem`` comment marks the
    deliberately-infeasible problems (e.g. BURKEHAN). Missing fields → ``None`` /
    ``False``.
    """
    path = os.path.join(root, "python_problems", name + ".py")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None, None, False

    pbclass = None
    match = re.search(r'self\.pbclass\s*=\s*"([^"]+)"', text)
    if match:
        pbclass = match.group(1)

    expected_objective: float | None = None
    match = re.search(r"(?im)^\s*#\s*LO\s+SOLTN\s+([-+0-9.eEdD]+)", text)
    if match:
        try:  # SIF may use Fortran 'D' exponents
            expected_objective = float(
                match.group(1).replace("D", "E").replace("d", "e")
            )
        except ValueError:
            expected_objective = None

    infeasible = bool(
        re.search(r"(?i)solution\s*\(\s*infeasible\s*\)", text)
        or re.search(r"(?i)source:\s*an?\s+infeasible", text)
    )
    return pbclass, expected_objective, infeasible


def _ensure_on_path(root: str) -> None:
    problems = os.path.join(root, "python_problems")
    for entry in (root, problems):
        if entry not in sys.path:
            sys.path.insert(0, entry)


@functools.lru_cache(maxsize=8)
def _instantiate(root: str, name: str, size: int | None = None) -> Any:
    """Instantiate an S2MPJ problem, optionally at a target variable count.

    Cached (small LRU) because the runner builds the same problem once per config
    (gate + every linear-solver route), and S2MPJ's pure-Python construction is the
    sweep's bottleneck — sharing the read-only instance across the per-problem
    config fan-out turns ~5 builds into one. ``size`` selects a scalable problem's
    dimension (``PROBLEM(N)``); a problem that is not size-parametrized (or rejects
    ``N``) falls back to its SIF default, so callers may request a size uniformly.
    """
    _ensure_on_path(root)
    cls = getattr(importlib.import_module(name), name)
    if size is not None:
        try:
            return cls(size)
        except Exception:  # not scalable / invalid N → SIF default size
            return cls()
    return cls()


# A small, curated default selection: constrained Hock-Schittkowski problems that
# the L-BFGS path should solve. The loader accepts any S2MPJ problem name.
_DEFAULT_NAMES = ("HS21", "HS35", "HS71", "HS6", "HS7", "HS8", "HS28")


def s2mpj_problems(
    names: Sequence[str] | None = None,
    *,
    directory: str | None = None,
    backends: tuple[str, ...] = ("numpy",),
    hessian: str = "lbfgs",
    sparse: bool = False,
    size: int | None = None,
    feasibility: bool = False,
) -> list[BenchmarkProblem]:
    """Return :class:`BenchmarkProblem`s for the named S2MPJ problems.

    Returns ``[]`` when no S2MPJ checkout is available (so callers may extend the
    corpus unconditionally). ``backends`` restricts each case to host-bridgeable
    namespaces (CPU NumPy/Torch); the default is NumPy only. ``hessian="exact"``
    builds problems that supply the analytic Lagrangian Hessian (else default
    L-BFGS); ``sparse=True`` returns Jacobians/Hessian as ``SparseOperator`` for
    the sparse-direct route. ``size`` requests a target variable count for the
    scalable problems (others keep their SIF default) — the lever for a
    scaling-focused sweep that reaches the sparse route's intended regime.
    ``feasibility=True`` admits the objective-free problems (CUTEst feasibility /
    nonlinear-equation systems), running them as ``min 0`` subject to the
    constraints instead of rejecting them. Each built problem carries the
    dataset's expected outcome (``expected_objective``/``expected_infeasible``/
    ``pbclass``) for authoritative scoring.
    """
    from benchmarks.corpus import BenchmarkProblem

    root = s2mpj_dir(directory)
    if root is None:
        return []

    selected = tuple(names) if names is not None else _DEFAULT_NAMES
    cls = _S2MPJExactProblem if hessian == "exact" else _S2MPJProblem

    def _make(name: str) -> BenchmarkProblem:
        def build(xp: Namespace) -> tuple[Problem, Array]:
            import numpy as np

            instance = _instantiate(root, name, size)
            problem = cls(instance, xp, sparse=sparse, feasibility=feasibility)
            pbclass, expected_objective, expected_infeasible = _problem_metadata(
                root, name
            )
            problem.pbclass = pbclass
            problem.expected_objective = expected_objective
            problem.expected_infeasible = expected_infeasible
            x0 = _from_numpy(xp, np.reshape(instance.x0, (-1,)))
            return problem, x0

        return BenchmarkProblem(
            name=f"s2mpj/{name}",
            kind="NLP",
            tags=("s2mpj", "cutest"),
            build=build,
            backends=backends,
        )

    return [_make(name) for name in selected]


__all__ = ["list_s2mpj_problems", "s2mpj_dir", "s2mpj_problems"]
