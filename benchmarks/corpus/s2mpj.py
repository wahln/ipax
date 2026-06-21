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

        self._inst = instance
        self.xp = xp
        self._n = int(instance.n)
        m = int(getattr(instance, "m", 0))

        clower = np.reshape(np.asarray(instance.clower, dtype=float), (-1,))
        cupper = np.reshape(np.asarray(instance.cupper, dtype=float), (-1,))
        finite_l = np.isfinite(clower)
        finite_u = np.isfinite(cupper)
        eq_mask = finite_l & finite_u & (clower == cupper)
        self._eq_idx = np.where(eq_mask)[0]
        self._eq_rhs = clower[self._eq_idx]
        self._lo_idx = np.where(finite_l & ~eq_mask)[0]  # clower ≤ c  ⇒ clower − c ≤ 0
        self._up_idx = np.where(finite_u & ~eq_mask)[0]  # c ≤ cupper  ⇒ c − cupper ≤ 0
        self._lo_rhs = clower[self._lo_idx]
        self._up_rhs = cupper[self._up_idx]
        self._m = m

        lower = np.reshape(np.asarray(instance.xlower, dtype=float), (-1,))
        upper = np.reshape(np.asarray(instance.xupper, dtype=float), (-1,))
        self._lower = _from_numpy(xp, lower)
        self._upper = _from_numpy(xp, upper)

    @property
    def n_vars(self) -> int:
        return self._n

    def bounds(self) -> tuple[Array, Array]:
        return self._lower, self._upper

    def objective(self, x: Array) -> Scalar:
        import numpy as np

        # Return an xp float64 0-d (not a Python float) so the value keeps the
        # target backend's dtype — a Python float re-cast via ``xp.asarray`` would
        # silently drop to float32 on some backends (e.g. Torch).
        return _from_numpy(self.xp, np.asarray(self._inst.fx(_to_numpy(x))))

    def gradient(self, x: Array) -> Array:
        import numpy as np

        _f, g = self._inst.fgx(_to_numpy(x))
        return _from_numpy(self.xp, np.reshape(g, (-1,)))

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


__all__ = ["s2mpj_dir", "s2mpj_problems"]
