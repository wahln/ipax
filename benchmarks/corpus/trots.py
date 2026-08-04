"""TROTS loader (download-gated, NumPy-evaluated, backend-bridged).

`TROTS <https://www.sebastiaanbreedveld.nl/trots/>`_ — the Radiotherapy
Optimisation Test Set (Breedveld & Heijmen, *Data in Brief* 2017) — is a set of
real radiotherapy treatment-planning problems stored as MATLAB v7.3 (HDF5) files.
Each file holds a ``problem`` list (weighted objectives + constraints), a ``data``
structure of pencil-beam **dose matrices** (sparse or dense), and the reference
``solutionX`` the dataset authors obtained. This module turns one such file into
an ipax :class:`~ipax.problem.base.Problem`.

The dataset is **not vendored** (size + license): point ``IPAX_TROTS_DIR`` (or the
``directory`` argument) at a local copy of the ``.mat`` files; the loaders return
``[]``/``None`` when it is absent, mirroring the other external corpora.

Cost functions (data description §5.1), with ``d = A x + b`` the dose vector:

* **type 1 — linear.** A single-row matrix is a *mean* (linear in ``x``). A
  multi-row matrix is the pointwise *maximum* (a *minimise* entry) or *minimum*
  (a *maximise* entry) — nonsmooth, so it is lowered to the standard **minimax**
  form: an auxiliary variable ``t`` with ``A x - t ≤ 0`` (max) or ``t - A x ≤ 0``
  (min) and a linear ``±w·t`` objective term. As a *constraint* it becomes the
  elementwise linear block ``A x ≤ bound`` / ``A x ≥ bound``.
* **type 2 — quadratic.** ``½ xᵀA x + bᵀx + c`` (the fluence-smoothing terms).
* **type 3 — gEUD.** ``(mean_i d_iᵃ)^{1/a}``.
* **type 4 — LTCP.** ``mean_i exp(-α(d_i - d_p))`` (tumour control).
* **type 5 — DVH.** ``mean_i r_i/(1+r_i)`` with ``r_i = (d_i/d_c)^p`` (Breedveld
  2017's smoothed partial-volume approximation).

The scalarised objective is ``Σ_k factor_k·w_k·f_k`` over the active objective
entries (``factor = +1`` minimise, ``-1`` maximise). Evaluated at the stored
``solutionX`` this reproduces the reference ``Objective Function Value`` from the
``Results/*.txt`` files (validated to the significant figures each solve reports),
which is the loader's ground-truth check — see :func:`objective_at`.

The **linear** blocks (elementwise dose constraints + minimax links) have a
constant Jacobian: they are lowered once to one-sided ``G z + h ≤ 0`` rows and
stacked *after* the smooth **nonlinear** cost-function rows in
``ineq_constraints``/``ineq_jacobian`` (so the Lagrangian-Hessian multipliers line
up with the curved rows, and the trailing linear rows contribute no Hessian term).
The constant block is emitted as a cached Array-API-native sparse operator rather
than ipax's :meth:`~ipax.problem.base.Problem.linear_ineq` because that route's
two-sided lowering densifies the matrix — infeasible at RT row counts. Jacobians
and the exact Lagrangian Hessian cross to the backend as Array-API-native
:class:`~ipax.backend.operators.CSROperator` /
:class:`~ipax.backend.operators.CSCOperator` blocks (true sparsity for the
sparse-direct route), never a concrete backend sparse object (invariant #4).

Because these problems have far more constraints than variables (``n ≈ 1e3``,
``m`` up to several ``1e5``), the **condensed** normal-equations route
(``linsolve="dense"`` or matrix-free ``"krylov"``) — which consumes the sparse
dose matrices through matvecs and forms only the ``n×n`` system — is far cheaper
per iteration than factoring the full ``(n+m)`` saddle with ``linsolve="sparse"``.

Like the S2MPJ bridge, evaluation runs in NumPy/SciPy on the host and converts
to/from the target namespace, so ipax's linear algebra runs on any CPU backend
while the (large, host-resident) dose matrices stay in SciPy. This forces a host
sync per evaluation — it is for accuracy/cross-backend work, not GPU performance.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ipax.problem.base import Problem

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarks.corpus import BenchmarkProblem
    from ipax.backend.operators import LinearOperator
    from ipax.typing import Array, Namespace, Scalar


# --------------------------------------------------------------------------- #
# Host bridge helpers
# --------------------------------------------------------------------------- #
def _to_numpy(x: Array) -> Any:
    """Array-API array → 1-D NumPy array (host bridge).

    Device arrays (CuPy) refuse implicit host conversion; ``.get()`` is the
    explicit device→host copy. The problem callbacks evaluate host-side with
    SciPy regardless of the solve namespace — at ``n ≈ 10³`` the per-iteration
    transfer is ~16 kB, negligible next to the device-side KKT solve.
    """
    import numpy as np

    if isinstance(x, np.ndarray):
        return np.reshape(x, (-1,))
    if hasattr(x, "__cuda_array_interface__") and hasattr(x, "get"):
        return np.reshape(np.asarray(x.get()), (-1,))
    try:
        return np.reshape(np.from_dlpack(x), (-1,))
    except (TypeError, RuntimeError, BufferError, ValueError):
        return np.reshape(np.asarray(x), (-1,))


def _from_numpy(xp: Namespace, arr: Any) -> Array:
    """NumPy array → target namespace as float64."""
    import numpy as np

    return xp.asarray(np.asarray(arr, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Raw file structures
# --------------------------------------------------------------------------- #
@dataclass
class TROTSMatrix:
    """One ``data.matrix`` entry: a dose/fluence/quadratic operator."""

    name: str
    mtype: int  # 0 dose (d = Ax+b), 1 fluence, 2 quadratic/square
    matrix: Any  # scipy.sparse matrix (voxels × vars) or (vars × vars) for type 2
    b: Any  # offset vector (NumPy), possibly all-zero
    c: float  # scalar for the quadratic form


@dataclass
class TROTSEntry:
    """One ``problem`` list entry: an objective or a constraint."""

    name: str
    is_constraint: bool
    minimise: bool  # objective: minimise f; constraint: maximum (≤) constraint
    ctype: int  # cost-function type (1 linear, 2 quad, 3 gEUD, 4 LTCP, 5 DVH)
    data_id: int  # 1-based index into the matrix list
    params: Any  # NumPy parameter vector for the cost function
    bound: float  # constraint rhs / objective aspired value ("Objective" field)
    weight: float
    priority: int
    active: bool


@dataclass
class TROTSInstance:
    """A parsed TROTS problem file (matrices loaded lazily on demand)."""

    identifier: str
    n: int  # number of decision variables (misc.size)
    real: int  # original RT variable count (misc.real)
    entries: list[TROTSEntry]
    solution: Any  # reference solutionX (NumPy), or None
    # Least-squares warm-start data (misc.Initialise*): the tumour dose matrices,
    # their per-matrix reference dose, and the regularisation matrix id (0 = none).
    init_matrix_ids: Any = field(default=None)  # 1-based ids, or None
    init_reference_dose: Any = field(default=None)  # scalar dose per init matrix
    init_reg_matrix_id: int = 0
    _path: str = field(repr=False, default="")
    _matrix_cache: dict[int, TROTSMatrix] = field(repr=False, default_factory=dict)

    def matrix(self, data_id: int) -> TROTSMatrix:
        """Load (and cache) the ``data.matrix`` entry ``data_id`` (1-based)."""
        if data_id not in self._matrix_cache:
            self._matrix_cache[data_id] = _load_matrix(self._path, data_id - 1)
        return self._matrix_cache[data_id]


# --------------------------------------------------------------------------- #
# HDF5 reading
# --------------------------------------------------------------------------- #
def _h5_str(f: Any, struct: str, field_name: str, i: int) -> str:
    ref = f[struct][field_name][i, 0]
    val = f[ref][()]
    import numpy as np

    try:
        return "".join(chr(int(c)) for c in np.asarray(val).ravel())
    except (TypeError, ValueError):
        return ""


def _h5_vec(f: Any, struct: str, field_name: str, i: int) -> Any:
    import numpy as np

    return np.asarray(f[f[struct][field_name][i, 0]][()], dtype=float).ravel()


def _load_matrix(path: str, j: int) -> TROTSMatrix:
    """Read ``data.matrix[j]`` (0-based) into a :class:`TROTSMatrix`."""
    import h5py
    import numpy as np
    import scipy.sparse as sp

    with h5py.File(path, "r") as f:
        mats = f["data"]["matrix"]
        ref = mats["A"][j, 0]
        obj = f[ref]
        mtype = int(np.asarray(f[mats["Type"][j, 0]][()]).ravel()[0])
        name = _h5_str(f, "data/matrix", "Name", j)
        b = np.asarray(f[mats["b"][j, 0]][()], dtype=float).ravel()
        c_arr = np.asarray(f[mats["c"][j, 0]][()]).ravel()
        c = float(c_arr[0]) if c_arr.size else 0.0
        if isinstance(obj, h5py.Group):
            # MATLAB sparse: CSC arrays (data, ir, jc) with a MATLAB_sparse row
            # count. An all-zero matrix (nnz = 0) is written with only ``jc`` —
            # ``data`` and ``ir`` are omitted entirely (Head-and-Neck_05's
            # 'Brainstem' matrix), so synthesize the empty arrays.
            if "data" in obj:
                data = np.asarray(obj["data"][()]).ravel()
                ir = np.asarray(obj["ir"][()]).ravel().astype(np.int64)
            else:
                data = np.zeros(0)
                ir = np.zeros(0, dtype=np.int64)
            jc = np.asarray(obj["jc"][()]).ravel().astype(np.int64)
            nrows = int(obj.attrs["MATLAB_sparse"])
            ncols = jc.size - 1
            matrix: Any = sp.csc_matrix((data, ir, jc), shape=(nrows, ncols))
        else:
            # Dense matrices are stored transposed (vars × voxels): un-transpose.
            matrix = sp.csr_matrix(np.asarray(obj[()], dtype=float).T)
    return TROTSMatrix(name=name, mtype=mtype, matrix=matrix, b=b, c=c)


@functools.lru_cache(maxsize=8)
def _load_instance(path: str, only_active: bool) -> TROTSInstance:
    """Parse the ``problem`` list + ``misc`` of a TROTS file (matrices lazy)."""
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        misc = f["data"]["misc"]
        n = int(np.asarray(misc["size"][()]).ravel()[0])
        real = int(np.asarray(misc["real"][()]).ravel()[0])
        solution = None
        if "solutionX" in f:
            solution = np.asarray(f["solutionX"][()], dtype=float).ravel()
        identifier = os.path.splitext(os.path.basename(path))[0]

        # Least-squares warm-start data (§4 misc): tumour matrix ids, their target
        # dose, and the regularisation matrix id (a 0 / empty entry means none).
        init_ids = np.asarray(misc["InitialiseMatrixID"][()]).ravel().astype(int)
        init_dose = np.asarray(misc["InitialiseReferenceDose"][()]).ravel()
        reg_raw = np.asarray(misc["InitialiseRegularisationMatrixID"][()]).ravel()
        reg_id = int(reg_raw[0]) if reg_raw.size and int(reg_raw[0]) > 0 else 0

        entries: list[TROTSEntry] = []
        n_entries = f["problem"]["dataID"].shape[0]
        for i in range(n_entries):
            active_vec = _h5_vec(f, "problem", "Active", i)
            active = bool(active_vec[0]) if active_vec.size else True
            if only_active and not active:
                continue
            entries.append(
                TROTSEntry(
                    name=_h5_str(f, "problem", "Name", i),
                    is_constraint=bool(_h5_vec(f, "problem", "IsConstraint", i)[0]),
                    minimise=bool(_h5_vec(f, "problem", "Minimise", i)[0]),
                    ctype=int(_h5_vec(f, "problem", "Type", i)[0]),
                    data_id=int(_h5_vec(f, "problem", "dataID", i)[0]),
                    params=_h5_vec(f, "problem", "Parameters", i),
                    bound=float(_h5_vec(f, "problem", "Objective", i)[0]),
                    weight=float(_h5_vec(f, "problem", "Weight", i)[0]),
                    priority=int(_h5_vec(f, "problem", "Priority", i)[0]),
                    active=active,
                )
            )
    return TROTSInstance(
        identifier=identifier,
        n=n,
        real=real,
        entries=entries,
        solution=solution,
        init_matrix_ids=init_ids,
        init_reference_dose=init_dose,
        init_reg_matrix_id=reg_id,
        _path=path,
    )


def load_trots_file(path: str, *, only_active: bool = True) -> TROTSInstance:
    """Load a TROTS ``.mat`` file into a :class:`TROTSInstance`.

    ``only_active`` drops the entries the dataset flags ``Active = False`` (disabled
    objectives/constraints), matching what the reference solve optimised.
    """
    return _load_instance(os.path.abspath(path), only_active)


# --------------------------------------------------------------------------- #
# Cost-function evaluation (on the dose vector d, host NumPy)
# --------------------------------------------------------------------------- #
# Default DVH steepness when a DVH entry supplies only the critical dose ``d_c``
# (the Liver relative-DVH constraints do this): the reference solves used ``p = 5``,
# reported in the ``DVH Parameters`` line of those ``Results/*.txt`` files.
_DVH_DEFAULT_STEEPNESS = 5.0


def _dvh_params(par: Any) -> tuple[float, float]:
    """``(d_c, p)`` for a DVH entry; ``p`` defaults to :data:`_DVH_DEFAULT_STEEPNESS`."""
    dc = float(par[0])
    p = float(par[1]) if par.size >= 2 else _DVH_DEFAULT_STEEPNESS
    return dc, p


def _cost_value(ctype: int, d: Any, par: Any) -> float:
    """Scalar cost-function value on dose vector ``d`` (types 1/3/4/5)."""
    import numpy as np

    if ctype == 1:  # linear: mean (1 row) or pointwise max/min (handled by caller)
        raise AssertionError("linear cost handled separately")
    if ctype == 3:  # gEUD
        a = float(par[0])
        return float(np.mean(np.power(np.abs(d), a)) ** (1.0 / a))
    if ctype == 4:  # LTCP
        dp, alpha = float(par[0]), float(par[1])
        return float(np.mean(np.exp(-alpha * (d - dp))))
    if ctype == 5:  # DVH (smoothed)
        dc, p = _dvh_params(par)
        # φ = r/(1+r) = 1 − s with s = 1/(1+r): for a high-dose voxel r overflows,
        # but s → 0 cleanly (1/∞), so 1 − s → 1 with no inf/inf nan.
        with np.errstate(over="ignore"):
            r = np.power(d / dc, p)
        s = 1.0 / (1.0 + r)
        return float(np.mean(1.0 - s))
    raise ValueError(f"unsupported cost-function type {ctype}")


def _cost_grad_d(ctype: int, d: Any, par: Any) -> Any:
    """∂f/∂d for the smooth nonlinear cost functions (per-voxel gradient)."""
    import numpy as np

    m = d.size
    if ctype == 3:  # gEUD: f = S^{1/a}, S = mean(d^a)
        a = float(par[0])
        s = float(np.mean(np.power(np.abs(d), a)))
        return (s ** (1.0 / a - 1.0)) * np.power(np.abs(d), a - 1.0) * np.sign(d) / m
    if ctype == 4:  # LTCP
        dp, alpha = float(par[0]), float(par[1])
        return (-alpha / m) * np.exp(-alpha * (d - dp))
    if ctype == 5:  # DVH
        dc, p = _dvh_params(par)
        # dφ/dd = (p/d)·(1−s)·s with s = 1/(1+r) — the overflow-safe form of
        # (p r/d)/(1+r)² (r s² = (1−s)s). → 0 as d → 0 (p > 1) and as r → ∞ (s → 0),
        # so a high-dose voxel gives 0 rather than an inf/inf nan.
        safe = d > 0.0
        g = np.zeros_like(d)
        ds = d[safe]
        with np.errstate(over="ignore"):
            r = np.power(ds / dc, p)
        s = 1.0 / (1.0 + r)
        g[safe] = (1.0 / m) * (p / ds) * (1.0 - s) * s
        return g
    raise ValueError(f"unsupported cost-function type {ctype}")


def _cost_hess_d(ctype: int, d: Any, par: Any) -> tuple[Any, Any, Any]:
    """Second-derivative pieces of ``∂²f/∂d²`` for the exact Hessian.

    Returns ``(diag, u, coef)`` such that ``∂²f/∂d² = diag(diag) + coef·u uᵀ``
    (the rank-1 term is absent for LTCP/DVH, where ``u`` is zeros and ``coef`` 0).
    ``Aᵀ (∂²f/∂d²) A`` is then the ``x``-block curvature.
    """
    import numpy as np

    m = d.size
    zeros = np.zeros_like(d)
    if ctype == 4:  # LTCP: convex, diagonal
        dp, alpha = float(par[0]), float(par[1])
        h = (alpha * alpha / m) * np.exp(-alpha * (d - dp))
        return h, zeros, 0.0
    if ctype == 5:  # DVH: φ'' = (p/d²)(1−s)s·[ −2p(1−s) + (p−1) ], s = 1/(1+r)
        # The overflow-safe rewrite of (p r/d²)/(1+r)²·[−2p r/(1+r) + (p−1)] using
        # r s² = (1−s)s and r s = (1−s); → 0 as d → 0 (p > 1) and as r → ∞ (s → 0).
        dc, p = _dvh_params(par)
        safe = d > 0.0
        h = np.zeros_like(d)
        ds = d[safe]
        with np.errstate(over="ignore"):
            r = np.power(ds / dc, p)
        s = 1.0 / (1.0 + r)
        h[safe] = (p / ds**2) * (1.0 - s) * s * (-2.0 * p * (1.0 - s) + (p - 1.0))
        return h / m, zeros, 0.0
    if ctype == 3:  # gEUD: diagonal + rank-1 (from S depending on all d_i)
        a = float(par[0])
        da = np.power(np.abs(d), a)
        s = float(np.mean(da))
        diag = s ** (1.0 / a - 1.0) * (a - 1.0) * np.power(np.abs(d), a - 2.0) / m
        u = np.power(np.abs(d), a - 1.0) * np.sign(d) / m  # (1/m) d^{a-1}
        coef = (1.0 / a - 1.0) * a * s ** (1.0 / a - 2.0)
        return diag, u, coef
    raise ValueError(f"unsupported cost-function type {ctype}")


# --------------------------------------------------------------------------- #
# Objective evaluation at a point (ground-truth check vs the reference results)
# --------------------------------------------------------------------------- #
def objective_at(instance: TROTSInstance, x: Any) -> float:
    """The scalarised TROTS objective ``Σ factor·w·f_k`` at ``x`` (host NumPy).

    Matches the reference ``Objective Function Value`` when ``x`` is the stored
    ``solutionX`` — the loader's correctness oracle.
    """
    import numpy as np

    x = np.asarray(x, dtype=float).ravel()
    total = 0.0
    for e in instance.entries:
        if e.is_constraint or not e.active:
            continue
        mat = instance.matrix(e.data_id)
        factor = 1.0 if e.minimise else -1.0
        if e.ctype == 2:  # quadratic
            ax = mat.matrix @ x
            lin = float(mat.b @ x) if mat.b.size == x.size else 0.0
            f = 0.5 * float(x @ ax) + lin + mat.c
        elif e.ctype == 1:  # linear: mean / pointwise max (min for maximise)
            d = mat.matrix @ x + (mat.b if mat.b.size == mat.matrix.shape[0] else 0.0)
            f = (
                float(d[0])
                if d.size == 1
                else (float(np.max(d)) if e.minimise else float(np.min(d)))
            )
        else:
            d = mat.matrix @ x + (mat.b if mat.b.size == mat.matrix.shape[0] else 0.0)
            f = _cost_value(e.ctype, d, e.params)
        total += factor * e.weight * f
    return total


# --------------------------------------------------------------------------- #
# The ipax Problem
# --------------------------------------------------------------------------- #
def _is_linear_mean(mat: TROTSMatrix) -> bool:
    return mat.matrix.shape[0] == 1


class TROTSProblem(Problem):
    """A TROTS treatment-planning problem as an ipax :class:`Problem`.

    Variables are ``z = [x, t]``: the RT fluence ``x ≥ 0`` (length ``instance.n``)
    plus one auxiliary ``t`` per multi-row linear objective (the minimax epigraph
    variable). The linear dose constraints and minimax links form the constant
    :meth:`linear_ineq` block; the smooth gEUD/LTCP/DVH/quadratic cost functions
    form the nonlinear objective and ``ineq_constraints``. There are no equality
    constraints (TROTS ranges are one-sided).

    The base class supplies no Lagrangian Hessian (default L-BFGS);
    :class:`TROTSExactProblem` adds the analytic one.
    """

    def __init__(
        self,
        instance: TROTSInstance,
        xp: Namespace,
        *,
        sparse: bool = False,
    ) -> None:
        import numpy as np

        self._inst = instance
        self.xp = xp
        self._sparse = sparse
        self._n = int(instance.n)
        self.expected_objective: float | None = None  # set by the loader

        # Partition the active entries into the reformulation's building blocks.
        self._minimax: list[tuple[TROTSMatrix, float, bool]] = []  # (mat, w, minimise)
        self._lin_obj_coef = np.zeros((self._n,), dtype=float)  # mean-linear terms
        self._lin_obj_const = 0.0
        self._nl_obj: list[
            tuple[TROTSMatrix, float, int, Any]
        ] = []  # (mat,coef,type,par)
        self._quad_obj: list[tuple[TROTSMatrix, float]] = []  # (mat, coef)
        self._nl_con: list[tuple[TROTSMatrix, float, int, Any, float]] = []
        self._quad_con: list[tuple[TROTSMatrix, float, float]] = []  # (mat,sign,bound)
        lin_rows: list[Any] = []  # scipy rows over x (padded later)
        lin_lo: list[float] = []
        lin_hi: list[float] = []

        for e in instance.entries:
            factor = 1.0 if e.minimise else -1.0
            if not e.is_constraint:  # objective
                if e.ctype == 1:
                    mat = instance.matrix(e.data_id)
                    if _is_linear_mean(mat):
                        row = np.asarray(mat.matrix.todense()).ravel()
                        self._lin_obj_coef += factor * e.weight * row
                        b0 = float(mat.b[0]) if mat.b.size else 0.0
                        self._lin_obj_const += factor * e.weight * b0
                    else:
                        self._minimax.append((mat, e.weight, e.minimise))
                elif e.ctype == 2:
                    self._quad_obj.append(
                        (instance.matrix(e.data_id), factor * e.weight)
                    )
                else:
                    self._nl_obj.append(
                        (
                            instance.matrix(e.data_id),
                            factor * e.weight,
                            e.ctype,
                            e.params,
                        )
                    )
            else:  # constraint
                mat = instance.matrix(e.data_id)
                if e.ctype == 1:
                    A = mat.matrix
                    off = mat.b if mat.b.size == A.shape[0] else 0.0
                    # minimise ⇒ maximum constraint (A x + off ≤ bound); else minimum.
                    if e.minimise:
                        lin_rows.append(A)
                        lin_lo.extend([-np.inf] * A.shape[0])
                        lin_hi.extend(
                            (np.full(A.shape[0], e.bound) - off).tolist()
                            if np.ndim(off)
                            else [e.bound - float(off)] * A.shape[0]
                        )
                    else:
                        lin_rows.append(A)
                        lin_lo.extend(
                            (np.full(A.shape[0], e.bound) - off).tolist()
                            if np.ndim(off)
                            else [e.bound - float(off)] * A.shape[0]
                        )
                        lin_hi.extend([np.inf] * A.shape[0])
                elif e.ctype == 2:
                    # ½xᵀAx + bᵀx + c (⋚ bound) → sign·(f - bound) ≤ 0.
                    sign = 1.0 if e.minimise else -1.0
                    self._quad_con.append((mat, sign, e.bound))
                else:
                    sign = 1.0 if e.minimise else -1.0
                    self._nl_con.append((mat, sign, e.ctype, e.params, e.bound))

        self._n_aux = len(self._minimax)
        self._n_vars = self._n + self._n_aux

        # Assemble the constant linear-inequality block over z = [x, t]:
        # the elementwise dose constraints plus one minimax link per aux var.
        import scipy.sparse as sp

        blocks: list[Any] = []
        lo: list[float] = list(lin_lo)
        hi: list[float] = list(lin_hi)
        if lin_rows:
            A_x = sp.vstack([sp.csr_matrix(r) for r in lin_rows], format="csr")
            blocks.append(
                sp.hstack(
                    [A_x, sp.csr_matrix((A_x.shape[0], self._n_aux))], format="csr"
                )
            )
        for k, (mat, _w, minimise) in enumerate(self._minimax):
            A = sp.csr_matrix(mat.matrix)
            m = A.shape[0]
            e_k = sp.csr_matrix(
                (np.full(m, -1.0 if minimise else 1.0), (np.arange(m), np.full(m, k))),
                shape=(m, self._n_aux),
            )
            # minimise(max): A x - t ≤ 0 ; maximise(min): -A x + t ≤ 0.
            sgn = 1.0 if minimise else -1.0
            blocks.append(sp.hstack([sgn * A, e_k], format="csr"))
            off = mat.b if mat.b.size == m else np.zeros(m)
            lo.extend([-np.inf] * m)
            hi.extend((-sgn * off).tolist())
        self._A_lin = (
            sp.vstack(blocks, format="csr")
            if blocks
            else sp.csr_matrix((0, self._n_vars))
        )
        self._lin_lo = np.asarray(lo, dtype=float)
        self._lin_hi = np.asarray(hi, dtype=float)

        # Lower the constant two-sided rows to one-sided form ``G z + h ≤ 0`` here
        # (each row has exactly one finite side): a finite upper ``A z ≤ hi`` →
        # ``A z − hi``; a finite lower ``A z ≥ lo`` → ``lo − A z``. The whole linear
        # block is emitted through ``ineq_constraints`` as a cached sparse operator
        # — ipax's ``linear_ineq`` lowering densifies, which is infeasible at RT
        # scale, so we lower the sparse block ourselves and keep it sparse.
        hi_fin = np.isfinite(self._lin_hi)
        lo_fin = np.isfinite(self._lin_lo)
        g_blocks: list[Any] = []
        h_parts: list[Any] = []
        A_csr = self._A_lin.tocsr()
        if bool(hi_fin.any()):
            g_blocks.append(A_csr[hi_fin])
            h_parts.append(-self._lin_hi[hi_fin])
        if bool(lo_fin.any()):
            g_blocks.append(-A_csr[lo_fin])
            h_parts.append(self._lin_lo[lo_fin])
        self._G = (
            sp.vstack(g_blocks, format="csr")
            if g_blocks
            else sp.csr_matrix((0, self._n_vars))
        )
        self._h = np.concatenate(h_parts) if h_parts else np.zeros((0,), dtype=float)
        self._n_lin_ineq = int(self._G.shape[0])
        self._n_nl_ineq = len(self._nl_con) + len(self._quad_con)
        self._G_op: LinearOperator | None = None  # cached constant sparse operator

        # Objective linear coefficient over z (mean terms on x; minimax on t).
        self._c_obj = np.zeros((self._n_vars,), dtype=float)
        self._c_obj[: self._n] = self._lin_obj_coef
        for k, (_mat, w, minimise) in enumerate(self._minimax):
            self._c_obj[self._n + k] = (1.0 if minimise else -1.0) * w

    # -- dimensions & bounds ------------------------------------------------
    @property
    def n_vars(self) -> int:
        return self._n_vars

    def bounds(self) -> tuple[Array | None, Array | None]:
        import numpy as np

        lower = np.zeros((self._n_vars,), dtype=float)
        lower[self._n :] = -np.inf  # aux minimax variables are free
        upper = np.full((self._n_vars,), np.inf, dtype=float)
        return _from_numpy(self.xp, lower), _from_numpy(self.xp, upper)

    def _pad_aux(self, x: Any) -> Any:
        """Append minimax aux values ``t_k = max/min(A_k x)`` consistent with ``x``."""
        import numpy as np

        t = np.zeros((self._n_aux,), dtype=float)
        for k, (mat, _w, minimise) in enumerate(self._minimax):
            d = mat.matrix @ x
            t[k] = float(np.max(d)) if minimise else float(np.min(d))
        return np.concatenate([x, t])

    def least_squares_fluence(self, ridge: float = 1e-8) -> Any:
        """The TROTS least-squares warm-start fluence (host NumPy, length ``n``).

        Solves the normal equations ``(A_initᵀ A_init + R) x = A_initᵀ (d_ref − b)``
        for the tumour matrices ``A_init`` (``misc.InitialiseMatrixID``) driven to
        their per-matrix reference dose ``d_ref`` (``misc.InitialiseReferenceDose``),
        with the optional quadratic regularisation ``R``
        (``misc.InitialiseRegularisationMatrixID``). This is the physically motivated
        start the dataset ships — a fluence that already delivers roughly the
        prescribed dose to the targets — far closer to the feasible region than a
        uniform map. Negative fluences are clipped to zero (non-negativity). Returns
        a uniform vector when the file carries no initialisation matrices.
        """
        import numpy as np
        import scipy.sparse as sp

        inst = self._inst
        ids = (
            np.asarray(inst.init_matrix_ids).ravel()
            if inst.init_matrix_ids is not None
            else np.zeros((0,), dtype=int)
        )
        if ids.size == 0:
            return np.ones((self._n,), dtype=float)

        dose = np.asarray(inst.init_reference_dose, dtype=float).ravel()
        blocks: list[Any] = []
        rhs_parts: list[Any] = []
        for k, mid in enumerate(ids):
            mat = inst.matrix(int(mid))
            a = sp.csr_matrix(mat.matrix)
            blocks.append(a)
            target = (dose[k] if k < dose.size else 0.0) - (
                mat.b if mat.b.size == a.shape[0] else 0.0
            )
            rhs_parts.append(np.full(a.shape[0], 1.0) * target)
        a_init = sp.vstack(blocks, format="csr")
        normal = np.asarray((a_init.T @ a_init).todense())
        rhs = a_init.T @ np.concatenate(rhs_parts)
        if inst.init_reg_matrix_id:
            r = inst.matrix(inst.init_reg_matrix_id).matrix
            r = np.asarray((0.5 * (r + r.T)).todense())
            normal = normal + r
        # A small ridge (relative to the block's scale) keeps the solve well-posed
        # when A_init is rank-deficient or no regularisation matrix is supplied.
        normal = normal + ridge * (np.trace(normal) / max(self._n, 1)) * np.eye(self._n)
        x = np.linalg.solve(normal, rhs)
        return np.maximum(x, 0.0)

    def initial_point(self, scale: float = 1.0, *, warm_start: bool = True) -> Array:
        """A strictly-positive fluence start with consistent minimax aux values.

        With ``warm_start`` (default) and initialisation data present, uses the
        dataset's least-squares fluence (:meth:`least_squares_fluence`); otherwise a
        uniform ``scale`` map. A tiny positive floor is applied so the log-barrier
        has a non-degenerate interior start on the ``x ≥ 0`` bounds.
        """
        import numpy as np

        has_init = (
            self._inst.init_matrix_ids is not None
            and np.asarray(self._inst.init_matrix_ids).size > 0
        )
        if warm_start and has_init:
            x = self.least_squares_fluence()
            peak = float(np.max(x))
            x = np.maximum(x, 1e-6 * peak if peak > 0.0 else 1e-6)
        else:
            x = np.full((self._n,), scale, dtype=float)
        return _from_numpy(self.xp, self._pad_aux(x))

    # -- objective ----------------------------------------------------------
    def objective(self, z: Array) -> Scalar:
        import numpy as np

        zc = _to_numpy(z)
        x = zc[: self._n]
        total = float(self._c_obj @ zc) + self._lin_obj_const
        for mat, coef in self._quad_obj:
            ax = mat.matrix @ x
            lin = float(mat.b @ x) if mat.b.size == x.size else 0.0
            total += coef * (0.5 * float(x @ ax) + lin + mat.c)
        for mat, coef, ctype, par in self._nl_obj:
            d = mat.matrix @ x + (mat.b if mat.b.size == mat.matrix.shape[0] else 0.0)
            try:
                total += coef * _cost_value(ctype, d, par)
            except (OverflowError, FloatingPointError):
                return _from_numpy(self.xp, np.asarray(np.inf))
        return _from_numpy(self.xp, np.asarray(total))

    def gradient(self, z: Array) -> Array:
        import numpy as np

        zc = _to_numpy(z)
        x = zc[: self._n]
        g = np.array(self._c_obj, dtype=float)
        for mat, coef in self._quad_obj:
            gx = mat.matrix @ x
            if mat.b.size == x.size:
                gx = gx + mat.b
            g[: self._n] += coef * gx
        for mat, coef, ctype, par in self._nl_obj:
            d = mat.matrix @ x + (mat.b if mat.b.size == mat.matrix.shape[0] else 0.0)
            gd = _cost_grad_d(ctype, d, par)
            g[: self._n] += coef * (mat.matrix.T @ gd)
        return _from_numpy(self.xp, g)

    # -- inequality constraints (nonlinear cost rows first, then linear) ----
    # The elementwise dose constraints and minimax links are constant-Jacobian
    # rows; ipax's ``linear_ineq`` would densify them, so they are lowered in
    # ``__init__`` to ``G z + h ≤ 0`` and stacked *after* the smooth cost-function
    # rows here. The nonlinear rows lead so the Lagrangian-Hessian multipliers line
    # up with them (the trailing linear multipliers carry no curvature).
    def _linear_ineq_operator(self) -> LinearOperator:
        if self._G_op is None:
            self._G_op = self._csr_operator(self._G, symmetric=False)
        return self._G_op

    def ineq_constraints(self, z: Array) -> Array:
        import numpy as np

        if self._n_nl_ineq == 0 and self._n_lin_ineq == 0:
            raise NotImplementedError
        zc = _to_numpy(z)
        x = zc[: self._n]
        vals: list[float] = []
        for mat, sign, ctype, par, bound in self._nl_con:
            d = mat.matrix @ x + (mat.b if mat.b.size == mat.matrix.shape[0] else 0.0)
            vals.append(sign * (_cost_value(ctype, d, par) - bound))
        for mat, sign, bound in self._quad_con:
            ax = mat.matrix @ x
            lin = float(mat.b @ x) if mat.b.size == x.size else 0.0
            f = 0.5 * float(x @ ax) + lin + mat.c
            vals.append(sign * (f - bound))
        nl = np.asarray(vals, dtype=float)
        if self._n_lin_ineq == 0:
            return _from_numpy(self.xp, nl)
        lin_vals = self._G @ zc + self._h
        return _from_numpy(self.xp, np.concatenate([nl, lin_vals]))

    def ineq_jacobian(self, z: Array) -> Array | LinearOperator:
        import numpy as np
        import scipy.sparse as sp

        from ipax.backend.operators import VStack

        if self._n_nl_ineq == 0 and self._n_lin_ineq == 0:
            raise NotImplementedError
        lin_op = self._linear_ineq_operator() if self._n_lin_ineq else None
        if self._n_nl_ineq == 0:
            assert lin_op is not None
            return lin_op
        x = _to_numpy(z)[: self._n]
        rows: list[Any] = []
        for mat, sign, ctype, par, _bound in self._nl_con:
            d = mat.matrix @ x + (mat.b if mat.b.size == mat.matrix.shape[0] else 0.0)
            gd = _cost_grad_d(ctype, d, par)
            rows.append(sign * (mat.matrix.T @ gd))
        for mat, sign, _bound in self._quad_con:
            gx = mat.matrix @ x
            if mat.b.size == x.size:
                gx = gx + mat.b
            rows.append(sign * gx)
        J_x = np.stack(rows) if rows else np.zeros((0, self._n))
        J = sp.hstack(
            [sp.csr_matrix(J_x), sp.csr_matrix((len(rows), self._n_aux))], format="csr"
        )
        nl_op = self._csr_operator(J, symmetric=False)
        if lin_op is None:
            return nl_op
        return VStack((nl_op, lin_op))

    # -- sparse-operator helper --------------------------------------------
    def _csr_operator(self, csr: Any, *, symmetric: bool) -> LinearOperator:
        """Wrap a host ``scipy.csr_matrix`` as an Array-API ``CSROperator``."""
        import numpy as np

        from ipax.backend.operators import CSROperator

        csr = csr.tocsr()
        csr.sort_indices()
        xp = self.xp
        return CSROperator(
            xp.asarray(np.asarray(csr.indptr, dtype=np.int64)),
            xp.asarray(np.asarray(csr.indices, dtype=np.int64)),
            _from_numpy(xp, csr.data),
            (int(csr.shape[0]), int(csr.shape[1])),
            symmetric=symmetric,
        )


class TROTSExactProblem(TROTSProblem):
    """TROTS problem that also supplies the exact Lagrangian Hessian.

    Only the quadratic and gEUD/LTCP/DVH terms have curvature; the linear and
    minimax pieces contribute none. The ``x``-block Hessian ``σ·∇²f + Σ y·∇²g`` is
    assembled as a host sparse matrix and returned padded with zero ``t`` rows and
    columns as a :class:`~ipax.backend.operators.CSCOperator`.
    """

    def lagrangian_hessian(
        self,
        x: Array,
        y_eq: Array,
        y_ineq: Array,
        sigma: Scalar = 1.0,
    ) -> Array | LinearOperator:
        import numpy as np
        import scipy.sparse as sp

        z = _to_numpy(x)
        xv = z[: self._n]
        s = float(sigma)
        H = sp.csr_matrix((self._n, self._n))

        for mat, coef in self._quad_obj:
            A = mat.matrix
            H = H + (s * coef) * 0.5 * (A + A.T)
        for mat, coef, ctype, par in self._nl_obj:
            d = mat.matrix @ xv + (mat.b if mat.b.size == mat.matrix.shape[0] else 0.0)
            H = H + (s * coef) * self._term_hessian(mat, ctype, par, d)

        yv = _to_numpy(y_ineq) if y_ineq is not None else np.zeros((0,))
        row = 0
        for mat, sign, ctype, par, _bound in self._nl_con:
            d = mat.matrix @ xv + (mat.b if mat.b.size == mat.matrix.shape[0] else 0.0)
            H = H + (yv[row] * sign) * self._term_hessian(mat, ctype, par, d)
            row += 1
        for mat, sign, _bound in self._quad_con:
            A = mat.matrix
            H = H + (yv[row] * sign) * 0.5 * (A + A.T)
            row += 1

        # Pad with zero aux (t) rows/cols and return as a symmetric CSC operator.
        H_full = sp.bmat(
            [[H, None], [None, sp.csr_matrix((self._n_aux, self._n_aux))]],
            format="csc",
        )
        return self._csc_operator(H_full)

    @staticmethod
    def _term_hessian(mat: TROTSMatrix, ctype: int, par: Any, d: Any) -> Any:
        """``Aᵀ (∂²f/∂d²) A`` for one nonlinear term (host sparse)."""
        import numpy as np
        import scipy.sparse as sp

        diag, u, coef = _cost_hess_d(ctype, d, par)
        A = mat.matrix
        H = (A.T @ sp.diags(diag) @ A).tocsr()
        if coef != 0.0:  # gEUD rank-1 term: Aᵀ (coef u uᵀ) A = coef (Aᵀu)(Aᵀu)ᵀ
            v = np.asarray(A.T @ u).ravel()
            H = H + coef * sp.csr_matrix(np.outer(v, v))
        return H

    def _csc_operator(self, csc: Any) -> LinearOperator:
        import numpy as np

        from ipax.backend.operators import CSCOperator

        csc = csc.tocsc()
        csc.sort_indices()
        xp = self.xp
        return CSCOperator(
            xp.asarray(np.asarray(csc.indptr, dtype=np.int64)),
            xp.asarray(np.asarray(csc.indices, dtype=np.int64)),
            _from_numpy(xp, csc.data),
            (int(csc.shape[0]), int(csc.shape[1])),
            symmetric=True,
        )


# --------------------------------------------------------------------------- #
# Reference results (Results/*.txt)
# --------------------------------------------------------------------------- #
@dataclass
class TROTSReference:
    """A parsed ``Results/<case>_result.txt`` reference solve."""

    objective: float
    n_iter: int
    significant_figures: int
    x: Any  # NumPy vector of the reference plan
    header: dict[str, float]


def parse_reference_result(path: str) -> TROTSReference:
    """Parse a TROTS ``Results/*.txt`` reference-solve file."""
    import numpy as np

    header: dict[str, float] = {}
    values: list[float] = []
    reading = False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if reading:
                try:
                    values.append(float(line.strip()))
                except ValueError:
                    continue
                continue
            if line.startswith("Optimised result"):
                reading = True
                continue
            # "Key words   <number>" — split on the last whitespace-run token.
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                try:
                    header[parts[0].strip()] = float(parts[1])
                except ValueError:
                    pass
    return TROTSReference(
        objective=header.get("Objective Function Value", float("nan")),
        n_iter=int(header.get("Number of Iterations", 0)),
        significant_figures=int(header.get("Significant Figures", 0)),
        x=np.asarray(values, dtype=float),
        header=header,
    )


# --------------------------------------------------------------------------- #
# Corpus factory (download-gated)
# --------------------------------------------------------------------------- #
# The first patient of each data group (used for the initial verification sweep).
_DEFAULT_CASES = (
    "Protons_01",
    "Prostate_BT_01",
    "Prostate_CK_01",
    "Prostate_VMAT_101",
    "Liver_01",
    "Head-and-Neck_01",
)


def trots_dir(directory: str | None = None) -> str | None:
    """Resolve the TROTS data directory, or ``None`` if unavailable.

    Uses ``directory`` if given, else ``IPAX_TROTS_DIR``. Qualifies only if it
    holds at least one ``*.mat`` file.
    """
    import glob

    root = directory or os.environ.get("IPAX_TROTS_DIR")
    if not root or not os.path.isdir(root):
        return None
    if not glob.glob(os.path.join(root, "*.mat")):
        return None
    return root


def list_trots_cases(directory: str | None = None) -> list[str]:
    """All TROTS case names available in the directory (sorted)."""
    import glob

    root = trots_dir(directory)
    if root is None:
        return []
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(root, "*.mat"))
    )


def reference_for(case: str, directory: str | None = None) -> TROTSReference | None:
    """The reference result for ``case``, if a ``Results/*.txt`` file exists."""
    root = trots_dir(directory)
    if root is None:
        return None
    path = os.path.join(root, "Results", f"{case}_result.txt")
    if not os.path.isfile(path):
        return None
    return parse_reference_result(path)


def trots_problems(
    names: Sequence[str] | None = None,
    *,
    directory: str | None = None,
    backends: tuple[str, ...] = ("numpy",),
    hessian: str = "lbfgs",
    sparse: bool = False,
) -> list[BenchmarkProblem]:
    """:class:`BenchmarkProblem`s for the named TROTS cases (``[]`` if no data).

    ``hessian="exact"`` supplies the analytic Lagrangian Hessian (else default
    L-BFGS); ``sparse=True`` is a hint carried for symmetry with the other corpora
    (the Jacobians/Hessian are always emitted as Array-API sparse operators).
    """
    from benchmarks.corpus import BenchmarkProblem

    root = trots_dir(directory)
    if root is None:
        return []

    selected = tuple(names) if names is not None else _DEFAULT_CASES
    cls = TROTSExactProblem if hessian == "exact" else TROTSProblem

    def _make(case: str) -> BenchmarkProblem:
        def build(xp: Namespace) -> tuple[Problem, Array]:
            instance = load_trots_file(os.path.join(root, f"{case}.mat"))
            problem = cls(instance, xp, sparse=sparse)
            ref = reference_for(case, root)
            problem.expected_objective = None if ref is None else ref.objective
            return problem, problem.initial_point()

        return BenchmarkProblem(
            name=f"trots/{case}",
            kind="RT",
            tags=("trots", "rt", "sparse"),
            build=build,
            backends=backends,
        )

    return [_make(case) for case in selected]


__all__ = [
    "TROTSEntry",
    "TROTSExactProblem",
    "TROTSInstance",
    "TROTSMatrix",
    "TROTSProblem",
    "TROTSReference",
    "list_trots_cases",
    "load_trots_file",
    "objective_at",
    "parse_reference_result",
    "reference_for",
    "trots_dir",
    "trots_problems",
]
