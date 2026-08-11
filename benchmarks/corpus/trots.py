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

Like the S2MPJ bridge, evaluation runs in NumPy/SciPy on the host by default and
converts to/from the target namespace, so ipax's linear algebra runs on any CPU
backend while the (large, host-resident) dose matrices stay in SciPy. For **CuPy
namespaces** the evaluation matrices are mirrored to the device once and the
callbacks evaluate device-side (the elementwise cost math dispatches from NumPy's
functions to CuPy via NEP-18), so a GPU solve no longer pays a host round-trip
per callback — only scalar reductions sync. The exact Lagrangian Hessian
(:class:`TROTSExactProblem`) still assembles host-side (per-term sparse SpGEMM;
it serves the accuracy tests, not the RT performance runs).
"""

from __future__ import annotations

import functools
import hashlib
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
    # Dtype of the matrix values AS STORED IN THE FILE (TROTS dose matrices
    # are float32 at the source). Any later float64 promotion is an exact
    # upcast, so this is the precision the data actually carries — the
    # metadata behind ``DenseOptions(gram_dtype="auto")``.
    source_dtype: str = "float64"


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


# --------------------------------------------------------------------------- #
# On-disk matrix cache
# --------------------------------------------------------------------------- #
# Parsing the MATLAB v7.3 dose matrices dominates a TROTS run's startup (tens of
# seconds per case; ``Prostate_CK_01`` measured 35-56 s), and every benchmark
# invocation re-pays it. Each parsed matrix is therefore mirrored to a plain
# ``.npz`` beside the dataset and reloaded from there on later runs. The cache is
# keyed by the source file's absolute-path digest, size and mtime plus a format
# version, so editing or replacing a ``.mat`` misses rather than serving a stale
# matrix (and two datasets sharing one cache directory cannot alias), and it is
# strictly an optimization: any failure to read or write it falls back to
# parsing. Set ``IPAX_TROTS_CACHE`` to a directory to relocate it, or to
# ``off`` to disable it.
_CACHE_FORMAT = 1
_CACHE_DIRNAME = ".ipax_trots_cache"
_CACHE_OFF = frozenset({"", "0", "off", "no", "none", "false"})


def _cache_dir_for(path: str) -> str | None:
    """Cache directory for ``path``'s parsed matrices, or ``None`` if disabled."""
    setting = os.environ.get("IPAX_TROTS_CACHE")
    if setting is not None:
        if setting.strip().lower() in _CACHE_OFF:
            return None
        root = setting
    else:
        root = os.path.join(os.path.dirname(os.path.abspath(path)), _CACHE_DIRNAME)
    try:
        stat = os.stat(path)
        # The absolute path's digest disambiguates same-named files: pointing
        # IPAX_TROTS_CACHE at one shared directory for several datasets would
        # otherwise let two "Protons_01.mat" with equal size and mtime alias,
        # and a load could silently return a matrix from the wrong dataset.
        # basename stays in the key so the directory remains human-readable.
        source = hashlib.sha256(
            os.path.abspath(path).encode("utf-8", "surrogatepass")
        ).hexdigest()[:16]
        key = (
            f"{os.path.basename(path)}.{source}"
            f".{stat.st_size}.{stat.st_mtime_ns}.v{_CACHE_FORMAT}"
        )
    except OSError:
        return None
    return os.path.join(root, key)


def _compact_indices(matrix: Any) -> Any:
    """Narrow a sparse matrix's index arrays to int32 when they fit.

    The MATLAB reader hands SciPy int64 ``ir``/``jc``, which SciPy then keeps —
    8 bytes per stored entry for dose matrices whose indices never approach
    2³¹. Narrowing halves index memory (~330 MB on ``Prostate_VMAT_101``'s
    82 M nonzeros), shrinks the on-disk cache by the same amount, and is what
    SciPy would have chosen had the arrays arrived as int32. Applied on both
    the parse and cache-read paths so a cold and a warm load are identical.
    """
    import numpy as np

    limit = np.iinfo(np.int32).max
    if matrix.nnz > limit or max(matrix.shape, default=0) > limit:
        return matrix
    if matrix.indices.dtype != np.int32:
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
    if matrix.indptr.dtype != np.int32:
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix


def _read_cached_matrix(cache_file: str) -> TROTSMatrix | None:
    """Load a cached :class:`TROTSMatrix`, or ``None`` on any miss/corruption."""
    import numpy as np
    import scipy.sparse as sp

    try:
        with np.load(cache_file) as z:  # allow_pickle stays off by default
            builder = sp.csc_matrix if str(z["fmt"]) == "csc" else sp.csr_matrix
            values = z["data"]
            # Values may be stored narrowed (see the writer); widening back is
            # exact, so the rebuilt matrix equals the parsed one bit for bit.
            if "value_dtype" in z.files and str(z["value_dtype"]) != str(values.dtype):
                values = values.astype(str(z["value_dtype"]))
            matrix = _compact_indices(
                builder(
                    (values, z["indices"], z["indptr"]),
                    shape=tuple(int(v) for v in z["shape"]),
                )
            )
            return TROTSMatrix(
                name=str(z["name"]),
                mtype=int(z["mtype"]),
                matrix=matrix,
                b=z["b"],
                c=float(z["c"]),
                source_dtype=str(z["source_dtype"]),
            )
    except (OSError, ValueError, KeyError, EOFError):
        # Absent, truncated (a run killed mid-write), or written by an
        # incompatible NumPy: re-parse instead of failing the load.
        return None


def _write_cached_matrix(cache_file: str, mat: TROTSMatrix) -> None:
    """Mirror a parsed matrix to ``cache_file`` (atomically; best effort)."""
    import numpy as np

    compressed = _compact_indices(
        mat.matrix.tocsc() if mat.matrix.format == "csc" else mat.matrix.tocsr()
    )
    # The dense-stored matrices are widened to float64 at parse time even
    # though the file holds float32 (``source_dtype`` records that), which
    # would otherwise double their footprint here — the corpus caches into
    # gigabytes. Store the narrow values when the widening is provably
    # lossless (checked, not assumed) and widen again on read.
    values = compressed.data
    stored = values
    if values.dtype == np.float64 and mat.source_dtype == "float32":
        narrowed = values.astype(np.float32)
        if np.array_equal(narrowed.astype(np.float64), values):
            stored = narrowed
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    # Write-then-rename so a concurrent reader never observes a partial file
    # (the benchmark runners load the same case from several processes).
    tmp = f"{cache_file}.{os.getpid()}.tmp"
    try:
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                fmt=np.asarray(compressed.format),
                data=stored,
                value_dtype=np.asarray(str(values.dtype)),
                indices=compressed.indices,
                indptr=compressed.indptr,
                shape=np.asarray(compressed.shape, dtype=np.int64),
                b=np.asarray(mat.b),
                c=np.asarray(mat.c),
                name=np.asarray(mat.name),
                mtype=np.asarray(mat.mtype),
                source_dtype=np.asarray(mat.source_dtype),
            )
        os.replace(tmp, cache_file)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _load_matrix(path: str, j: int) -> TROTSMatrix:
    """Read ``data.matrix[j]`` (0-based) into a :class:`TROTSMatrix`.

    Served from the on-disk cache when possible (see :func:`_cache_dir_for`).
    """
    cache_dir = _cache_dir_for(path)
    cache_file = os.path.join(cache_dir, f"m{j:05d}.npz") if cache_dir else None
    if cache_file is not None:
        cached = _read_cached_matrix(cache_file)
        if cached is not None:
            return cached

    mat = _parse_matrix(path, j)

    if cache_file is not None:
        try:
            _write_cached_matrix(cache_file, mat)
        except OSError:
            pass  # read-only location, full disk, ... — the parse still stands
    return mat


def _parse_matrix(path: str, j: int) -> TROTSMatrix:
    """Read ``data.matrix[j]`` (0-based) straight from the HDF5 file."""
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
            source_dtype = str(data.dtype)
            jc = np.asarray(obj["jc"][()]).ravel().astype(np.int64)
            nrows = int(obj.attrs["MATLAB_sparse"])
            ncols = jc.size - 1
            matrix: Any = _compact_indices(
                sp.csc_matrix((data, ir, jc), shape=(nrows, ncols))
            )
        else:
            # Dense matrices are stored transposed (vars × voxels): un-transpose.
            raw = np.asarray(obj[()])
            source_dtype = str(raw.dtype)
            matrix = _compact_indices(sp.csr_matrix(np.asarray(raw, dtype=float).T))
    return TROTSMatrix(
        name=name,
        mtype=mtype,
        matrix=matrix,
        b=b,
        c=c,
        source_dtype=source_dtype,
    )


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
        # Per-row "is this row's data float32 in the file?" flags, appended in the
        # same order as the rows themselves so the lowered block below can be
        # grouped by source precision.
        lin_f32: list[Any] = []

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
                    lin_f32.append(
                        np.full(A.shape[0], mat.source_dtype == "float32", dtype=bool)
                    )
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
            lin_f32.append(np.full(m, mat.source_dtype == "float32", dtype=bool))
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
        row_f32 = (
            np.concatenate(lin_f32)
            if lin_f32
            else np.zeros((self._A_lin.shape[0],), dtype=bool)
        )
        g_blocks: list[Any] = []
        h_parts: list[Any] = []
        n_f32 = 0
        A_csr = self._A_lin.tocsr()
        # Emit the float32-sourced rows first, so that group is *contiguous* in the
        # lowered ``G`` and can be handed to the solver as an operator of its own
        # carrying its own accumulate hint (see ``_linear_ineq_operator``). Row
        # order within a group — and hence the whole block whenever the plan is not
        # mixed — is unchanged from the ungrouped assembly.
        for is_f32 in (True, False):
            group = row_f32 == is_f32
            sel_hi = hi_fin & group
            sel_lo = lo_fin & group
            if bool(sel_hi.any()):
                g_blocks.append(A_csr[sel_hi])
                h_parts.append(-self._lin_hi[sel_hi])
                n_f32 += int(sel_hi.sum()) if is_f32 else 0
            if bool(sel_lo.any()):
                g_blocks.append(-A_csr[sel_lo])
                h_parts.append(self._lin_lo[sel_lo])
                n_f32 += int(sel_lo.sum()) if is_f32 else 0
        self._G = (
            sp.vstack(g_blocks, format="csr")
            if g_blocks
            else sp.csr_matrix((0, self._n_vars))
        )
        self._h = np.concatenate(h_parts) if h_parts else np.zeros((0,), dtype=float)
        self._n_lin_ineq = int(self._G.shape[0])
        self._n_nl_ineq = len(self._nl_con) + len(self._quad_con)
        # The lin-block values are the source matrices' entries times ±1 (plus the
        # ±1 minimax links), so a row drawn from a matrix that is float32 in the
        # file carries only float32 information even though the assembled ``G`` is
        # float64. Those rows lead the block (grouped above), and the count is what
        # ``_linear_ineq_operator`` splits on so the solver's ``gram_dtype="auto"``
        # default can reduce their accumulate while the genuinely-float64 rows stay
        # exact. Radiotherapy plans are routinely mixed: a VMAT plan measures ~96%
        # float32 by nonzero, held back by a single float64 constraint matrix.
        self._n_lin_f32 = n_f32
        self._G_op: LinearOperator | None = None  # cached constant sparse operator

        # Objective linear coefficient over z (mean terms on x; minimax on t).
        self._c_obj = np.zeros((self._n_vars,), dtype=float)
        self._c_obj[: self._n] = self._lin_obj_coef
        for k, (_mat, w, minimise) in enumerate(self._minimax):
            self._c_obj[self._n + k] = (1.0 if minimise else -1.0) * w

        # Device-side evaluation (CuPy namespaces): keep z and the evaluation
        # matrices on the device so the per-callback dose products (incl. the
        # big constant G spmv) run on the GPU instead of round-tripping to
        # SciPy. The elementwise cost math (`_cost_value` & friends) is written
        # against NumPy's public functions, which dispatch to CuPy via the
        # NEP-18/ufunc protocols, so the same code serves both spaces. Host
        # NumPy/SciPy remains the path for every non-CUDA namespace.
        self._eval_device = False
        self._cxs: Any = None  # cupyx.scipy.sparse, when active
        self._dev_mats: dict[int, Any] = {}  # id(TROTSMatrix) -> device CSR
        self._dev_vecs: dict[int, Any] = {}  # id(TROTSMatrix) -> device b
        self._G_eval: Any = self._G
        self._h_eval: Any = self._h
        self._c_obj_eval: Any = self._c_obj
        try:
            probe = xp.asarray(0.0)
        except Exception:  # namespace probing only — any failure means "host"
            probe = None
        if probe is not None and hasattr(probe, "__cuda_array_interface__"):
            try:
                import cupyx.scipy.sparse as cxs
            except ImportError:
                cxs = None
            if cxs is not None:
                self._cxs = cxs
                self._eval_device = True
                self._G_eval = cxs.csr_matrix(self._G)
                self._h_eval = xp.asarray(self._h)
                self._c_obj_eval = xp.asarray(self._c_obj)

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

    # -- evaluation-space accessors -----------------------------------------
    # Host (default): NumPy vectors + the SciPy matrices as loaded. Device
    # (CuPy namespaces): the same objects mirrored once to the GPU, keyed by
    # the TROTSMatrix identity (matrices are immutable for the problem's
    # lifetime). The evaluation loops below are written once against these
    # accessors; NumPy's functions dispatch to CuPy on device arrays.
    def _eval_z(self, z: Array) -> Any:
        if self._eval_device:
            return self.xp.reshape(self.xp.asarray(z), (-1,))
        return _to_numpy(z)

    def _eval_mat(self, mat: TROTSMatrix) -> Any:
        if not self._eval_device:
            return mat.matrix
        dev = self._dev_mats.get(id(mat))
        if dev is None:
            dev = self._cxs.csr_matrix(mat.matrix.tocsr())
            self._dev_mats[id(mat)] = dev
        return dev

    def _eval_b(self, mat: TROTSMatrix) -> Any:
        if not self._eval_device:
            return mat.b
        dev = self._dev_vecs.get(id(mat))
        if dev is None:
            dev = self.xp.asarray(mat.b)
            self._dev_vecs[id(mat)] = dev
        return dev

    def _dose(self, mat: TROTSMatrix, x: Any) -> Any:
        """``d = A x (+ b)`` in evaluation space (offset only when per-row)."""
        d = self._eval_mat(mat) @ x
        if mat.b.size == mat.matrix.shape[0]:
            d = d + self._eval_b(mat)
        return d

    # -- objective ----------------------------------------------------------
    def objective(self, z: Array) -> Scalar:
        import numpy as np

        zc = self._eval_z(z)
        x = zc[: self._n]
        total = float(self._c_obj_eval @ zc) + self._lin_obj_const
        for mat, coef in self._quad_obj:
            ax = self._eval_mat(mat) @ x
            lin = float(self._eval_b(mat) @ x) if mat.b.size == self._n else 0.0
            total += coef * (0.5 * float(x @ ax) + lin + mat.c)
        for mat, coef, ctype, par in self._nl_obj:
            d = self._dose(mat, x)
            try:
                total += coef * _cost_value(ctype, d, par)
            except (OverflowError, FloatingPointError):
                return _from_numpy(self.xp, np.asarray(np.inf))
        return _from_numpy(self.xp, np.asarray(total))

    def gradient(self, z: Array) -> Array:
        zc = self._eval_z(z)
        x = zc[: self._n]
        g = self._c_obj_eval.copy()
        for mat, coef in self._quad_obj:
            gx = self._eval_mat(mat) @ x
            if mat.b.size == self._n:
                gx = gx + self._eval_b(mat)
            g[: self._n] += coef * gx
        for mat, coef, ctype, par in self._nl_obj:
            d = self._dose(mat, x)
            gd = _cost_grad_d(ctype, d, par)
            g[: self._n] += coef * (self._eval_mat(mat).T @ gd)
        if self._eval_device:
            return g
        return _from_numpy(self.xp, g)

    # -- inequality constraints (nonlinear cost rows first, then linear) ----
    # The elementwise dose constraints and minimax links are constant-Jacobian
    # rows; ipax's ``linear_ineq`` would densify them, so they are lowered in
    # ``__init__`` to ``G z + h ≤ 0`` and stacked *after* the smooth cost-function
    # rows here. The nonlinear rows lead so the Lagrangian-Hessian multipliers line
    # up with them (the trailing linear multipliers carry no curvature).
    def _linear_ineq_operator(self) -> LinearOperator:
        if self._G_op is None:
            k = self._n_lin_f32
            if k in (0, self._n_lin_ineq):
                self._G_op = self._csr_operator(
                    self._G,
                    symmetric=False,
                    values_dtype_hint="float32" if k else None,
                )
            else:
                from ipax.backend.operators import VStack

                # Mixed sources: hand the two groups over separately, so a reduced
                # Gram accumulate applies to the float32-sourced rows alone. The
                # stack is equivalent row-for-row to the single operator — a
                # vertical stack's Gram is the sum of its blocks' Grams.
                self._G_op = VStack(
                    (
                        self._csr_operator(
                            self._G[:k], symmetric=False, values_dtype_hint="float32"
                        ),
                        self._csr_operator(self._G[k:], symmetric=False),
                    )
                )
        return self._G_op

    def ineq_constraints(self, z: Array) -> Array:
        import numpy as np

        if self._n_nl_ineq == 0 and self._n_lin_ineq == 0:
            raise NotImplementedError
        zc = self._eval_z(z)
        x = zc[: self._n]
        vals: list[float] = []
        for mat, sign, ctype, par, bound in self._nl_con:
            d = self._dose(mat, x)
            vals.append(sign * (_cost_value(ctype, d, par) - bound))
        for mat, sign, bound in self._quad_con:
            ax = self._eval_mat(mat) @ x
            lin = float(self._eval_b(mat) @ x) if mat.b.size == self._n else 0.0
            f = 0.5 * float(x @ ax) + lin + mat.c
            vals.append(sign * (f - bound))
        nl = np.asarray(vals, dtype=float)
        if self._n_lin_ineq == 0:
            return _from_numpy(self.xp, nl)
        lin_vals = self._G_eval @ zc + self._h_eval
        if self._eval_device:
            return np.concatenate([self.xp.asarray(nl), lin_vals])
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
        x = self._eval_z(z)[: self._n]
        rows: list[Any] = []
        for mat, sign, ctype, par, _bound in self._nl_con:
            d = self._dose(mat, x)
            gd = _cost_grad_d(ctype, d, par)
            rows.append(sign * (self._eval_mat(mat).T @ gd))
        for mat, sign, _bound in self._quad_con:
            gx = self._eval_mat(mat) @ x
            if mat.b.size == self._n:
                gx = gx + self._eval_b(mat)
            rows.append(sign * gx)
        spx = self._cxs if self._eval_device else sp
        J_x = np.stack(rows) if rows else np.zeros((0, self._n))
        J = spx.hstack(
            [spx.csr_matrix(J_x), spx.csr_matrix((len(rows), self._n_aux))],
            format="csr",
        )
        nl_op = self._csr_operator(J, symmetric=False)
        if lin_op is None:
            return nl_op
        return VStack((nl_op, lin_op))

    # -- sparse-operator helper --------------------------------------------
    def _csr_operator(
        self,
        csr: Any,
        *,
        symmetric: bool,
        values_dtype_hint: str | None = None,
    ) -> LinearOperator:
        """Wrap a host SciPy (or device cupyx) CSR as an Array-API ``CSROperator``."""
        import numpy as np

        from ipax.backend.operators import CSROperator

        csr = csr.tocsr()
        csr.sort_indices()
        xp = self.xp
        if hasattr(csr.data, "__cuda_array_interface__"):
            # Already device-resident (device evaluation path): no host bounce.
            return CSROperator(
                xp.asarray(csr.indptr, dtype=xp.int64),
                xp.asarray(csr.indices, dtype=xp.int64),
                xp.asarray(csr.data, dtype=xp.float64),
                (int(csr.shape[0]), int(csr.shape[1])),
                symmetric=symmetric,
                values_dtype_hint=values_dtype_hint,
            )
        return CSROperator(
            xp.asarray(np.asarray(csr.indptr, dtype=np.int64)),
            xp.asarray(np.asarray(csr.indices, dtype=np.int64)),
            _from_numpy(xp, csr.data),
            (int(csr.shape[0]), int(csr.shape[1])),
            symmetric=symmetric,
            values_dtype_hint=values_dtype_hint,
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
