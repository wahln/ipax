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

S2MPJ has no license file, so its files are **not vendored**. Point
``IPAX_S2MPJ_DIR`` (or the ``directory`` argument) at a local checkout; the loader
returns ``[]`` when it is absent, mirroring the other external corpora.

S2MPJ stores each constraint as a two-sided range ``clower ≤ c(x) ≤ cupper`` with
an equality when the two coincide. The adapter maps that onto ipax's eq/ineq
split, lowering finite inequality sides to one-sided rows (``c − cupper ≤ 0`` and
``clower − c ≤ 0``) exactly as the native ``linear_ineq`` lowering does. No
Lagrangian Hessian is supplied, so the solver uses its default L-BFGS — the path
this sweep is meant to exercise across the corpus.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import TYPE_CHECKING, Any

from ipax.problem.base import Problem

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarks.corpus import BenchmarkProblem
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
    """

    def __init__(self, instance: Any, xp: Namespace) -> None:
        import numpy as np

        # Reject feasibility/least-squares problems with no objective group: they
        # are not minimization problems and cannot be benchmarked as such (S2MPJ's
        # ``fx`` would error). ``H`` is the optional explicit quadratic objective.
        if len(getattr(instance, "objgrps", ())) == 0 and not hasattr(instance, "H"):
            raise NotImplementedError("S2MPJ problem has no objective function")

        self._inst = instance
        self.xp = xp
        self._n = int(instance.n)
        m = int(getattr(instance, "m", 0))
        self._m = m

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

    def objective(self, x: Array) -> Scalar:
        import numpy as np

        # S2MPJ's auto-generated evaluations use Python ``float**`` and can raise
        # OverflowError on the wild trial points a line search probes (e.g.
        # LUKVLE4C's ``100*GVAR**6``). Return +inf so the solver simply rejects
        # the trial rather than crashing. Also return an xp float64 0-d (not a
        # Python float) so the value keeps the backend dtype — a Python float
        # re-cast via ``xp.asarray`` would silently drop to float32 on Torch.
        try:
            val = np.asarray(self._inst.fx(_to_numpy(x)))
        except (OverflowError, FloatingPointError):
            val = np.asarray(np.inf)
        return _from_numpy(self.xp, val)

    def gradient(self, x: Array) -> Array:
        import numpy as np

        try:
            _f, g = self._inst.fgx(_to_numpy(x))
            g = np.reshape(g, (-1,))
        except (OverflowError, FloatingPointError):
            g = np.full((self._n,), np.inf)
        return _from_numpy(self.xp, g)

    # -- equalities (present only when S2MPJ has clower == cupper rows) ----
    def eq_constraints(self, x: Array) -> Array:
        if self._eq_idx.shape[0] == 0:
            raise NotImplementedError
        import numpy as np

        c = np.reshape(np.asarray(self._inst.cx(_to_numpy(x)), dtype=float), (-1,))
        return _from_numpy(self.xp, c[self._eq_idx] - self._eq_rhs)

    def eq_jacobian(self, x: Array) -> Array:
        if self._eq_idx.shape[0] == 0:
            raise NotImplementedError
        _c, jac = self._inst.cJx(_to_numpy(x))
        return _from_numpy(self.xp, jac.toarray()[self._eq_idx, :])

    # -- inequalities (lowered finite sides of the two-sided ranges) -------
    def ineq_constraints(self, x: Array) -> Array:
        if self._lo_idx.shape[0] == 0 and self._up_idx.shape[0] == 0:
            raise NotImplementedError
        import numpy as np

        c = np.reshape(np.asarray(self._inst.cx(_to_numpy(x)), dtype=float), (-1,))
        lo = self._lo_rhs - c[self._lo_idx]
        up = c[self._up_idx] - self._up_rhs
        return _from_numpy(self.xp, np.concatenate((lo, up)))

    def ineq_jacobian(self, x: Array) -> Array:
        if self._lo_idx.shape[0] == 0 and self._up_idx.shape[0] == 0:
            raise NotImplementedError
        import numpy as np

        _c, jac = self._inst.cJx(_to_numpy(x))
        dense = jac.toarray()
        rows = np.concatenate((-dense[self._lo_idx, :], dense[self._up_idx, :]), axis=0)
        return _from_numpy(self.xp, rows)


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


def _ensure_on_path(root: str) -> None:
    problems = os.path.join(root, "python_problems")
    for entry in (root, problems):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _instantiate(root: str, name: str) -> Any:
    _ensure_on_path(root)
    module = importlib.import_module(name)
    return getattr(module, name)()


# A small, curated default selection: constrained Hock-Schittkowski problems that
# the L-BFGS path should solve. The loader accepts any S2MPJ problem name.
_DEFAULT_NAMES = ("HS21", "HS35", "HS71", "HS6", "HS7", "HS8", "HS28")


def s2mpj_problems(
    names: Sequence[str] | None = None,
    *,
    directory: str | None = None,
    backends: tuple[str, ...] = ("numpy",),
) -> list[BenchmarkProblem]:
    """Return :class:`BenchmarkProblem`s for the named S2MPJ problems.

    Returns ``[]`` when no S2MPJ checkout is available (so callers may extend the
    corpus unconditionally). ``backends`` restricts each case to host-bridgeable
    namespaces (CPU NumPy/Torch); the default is NumPy only.
    """
    from benchmarks.corpus import BenchmarkProblem

    root = s2mpj_dir(directory)
    if root is None:
        return []

    selected = tuple(names) if names is not None else _DEFAULT_NAMES

    def _make(name: str) -> BenchmarkProblem:
        def build(xp: Namespace) -> tuple[Problem, Array]:
            import numpy as np

            instance = _instantiate(root, name)
            problem = _S2MPJProblem(instance, xp)
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
