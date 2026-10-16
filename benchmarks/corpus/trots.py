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
constant Jacobian and are declared through ipax's public
:meth:`~ipax.problem.base.Problem.linear_ineq` as the original two-sided rows
``lo ≤ A z ≤ hi``. The solver lowers them to one-sided rows at the triplet level
— never densified — and stacks them *after* the smooth **nonlinear**
cost-function rows of ``ineq_constraints``/``ineq_jacobian``, slicing their
multipliers off before the Lagrangian Hessian (so :class:`TROTSExactProblem`
sees exactly the curved rows' multipliers). Jacobians, the linear block and the
exact Lagrangian Hessian cross to the backend as Array-API-native
:class:`~ipax.backend.operators.CSROperator` /
:class:`~ipax.backend.operators.CSCOperator` blocks (true sparsity for the
sparse-direct route), never a concrete backend sparse object (invariant #4).
On NumPy, CuPy and Torch namespaces each precision group of the linear block
is a :class:`TiledGramCSROperator` — the same CSR rows, with a per-structure
tiled Gram (Breedveld et al. 2017, §4.2 eq. (48), §5.2) behind the ``gram``
hook the condensed dense route calls, run by host BLAS on NumPy and in the
namespace (cuBLAS ``syrk`` / ``matmul``) on CuPy and Torch;
``TROTSProblem(..., tiled_gram=False)`` keeps the plain operator.

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
(:class:`TROTSExactProblem`) evaluates host-side: a :class:`TROTSExactHessian`
with one (tiled) product per unique curved structure, the dose-space
diagonals of all its terms summed (van Haveren & Breedveld 2019, eqs. (8)–(9);
Breedveld et al. 2017, eq. (48)), and the gEUD rank-1 terms as a low-rank
factor. On CuPy and Torch namespaces its tiled structures run through the
same device kernel (:class:`TROTSDeviceExactHessian`); on other namespaces
with a sparse adapter (JAX), or when no structure is tiled, the grouped
Hessian is formed as host sparse once per call and crossed as a
``CSCOperator`` (no per-product host round-trip). ``grouped_hessian=False``
keeps the original per-term SpGEMM path.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ipax.backend.operators import CSROperator, LinearOperator
from ipax.problem.base import Problem

_LOG = logging.getLogger(__name__)

# Breedveld et al. 2017 §3.1 initialisation constants: the floor applied to the
# primal fluence (after eq. (24)) and to y0, z0, w0 (eqs. (26)–(28)), and the
# LSQR settings for the least-squares multipliers of eq. (26). The floor makes
# high LSQR precision moot (Protons_01: every multiplier is < 0.04 and lands on
# the floor; 1e-6 gives the identical start in 3.4 s where 1e-8 took 24 s).
_BREEDVELD_START_FLOOR = 5.0
_BREEDVELD_LSQR_TOL = 1e-6
_BREEDVELD_LSQR_ITER = 200

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from benchmarks.corpus import BenchmarkProblem
    from ipax.options import ScalingMethod, ScalingOptions
    from ipax.result import WarmStart
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


# Robustness guards at the cost functions' singular points (van Haveren &
# Breedveld 2019, Phys. Med. Biol. 64:16NT01, and its corrigendum 65:079501,
# give the closed forms; these keep them finite where the closed form is not):
#
# * gEUD powers with a negative exponent (``|d|^(a-2)`` in the Hessian for
#   ``a < 2``, ``|d|^(a-1)`` in the gradient for ``a < 1``) are evaluated at
#   ``max(|d|, _GEUD_DOSE_FLOOR)``: at ``d = 0`` the curvature is unbounded
#   (``1 < a < 2``) or multiplied by ``a - 1 = 0`` (mean, ``a = 1``), and the
#   unguarded form returns ``inf`` / ``nan``. 1e-12 Gy is ~10 orders below any
#   dose that matters, so values away from ``d = 0`` are bitwise unchanged.
#   For strongly negative exponents the floor rises with ``|e|`` to
#   ``_GEUD_POWER_MAX ** (-1/|e|)`` (``1e-12^e`` alone overflows float64 once
#   ``e < ≈ -25.6``), so ``floor^e ≤ 1e300`` stays finite for every finite ``a``;
#   the raised floor stays below 5e-10 Gy for ``|e| ≤ 32`` (``a ≥ -30``).
# * A negative exponent with a voxel at (or numerically at) zero dose makes
#   ``S = mean(d^a)`` infinite: the gEUD value is then 0 and the closed forms'
#   prefactors ``S^(1/a-1)`` / ``S^(1/a-2)`` vanish, so every gradient and
#   Hessian entry evaluates to 0 — returned explicitly (as for ``S = 0``
#   below; the gEUD is not differentiable there) instead of ``0·∞ = nan``.
# * An all-zero structure dose gives ``S = mean(d^a) = 0``, where the gEUD is
#   not differentiable (a scaled ``a``-norm at the origin): the gradient and
#   Hessian return 0 — the minimum-norm subgradient and no curvature — instead
#   of a Python ``ZeroDivisionError`` from ``0.0 ** negative``.
# * The LTCP exponent ``-α(d - d_p)`` is clipped at ``_LTCP_EXP_CLIP``: e^500 ≈
#   1e217 leaves headroom for the voxel mean (a sum of ~1e5 terms), while the
#   clip needs a dose ~500/α below the prescription — never a normal input.
#   Below the clip the *value* is constant while :func:`_cost_grad_d` /
#   :func:`_cost_hess_d` still return the (clipped) ``α/m·e^500`` factors, so a
#   finite-difference derivative harness would flag that region as
#   inconsistent; it is a guard, not a smooth extension.
_GEUD_DOSE_FLOOR = 1e-12
_GEUD_POWER_MAX = 1e300
_LTCP_EXP_CLIP = 500.0


def _geud_power(ad: Any, e: float) -> Any:
    """``|d|^e`` with ``|d|`` floored for ``e < 0`` so the power stays finite:
    at :data:`_GEUD_DOSE_FLOOR`, raised for ``|e| ≳ 25`` until ``floor^e`` is at
    most :data:`_GEUD_POWER_MAX` (see the guards above)."""
    import numpy as np

    if e < 0.0:
        floor = max(_GEUD_DOSE_FLOOR, _GEUD_POWER_MAX ** (1.0 / e))
        ad = np.maximum(ad, floor)
    return np.power(ad, e)


def _geud_mean_power(ad: Any, a: float) -> float:
    """``S = mean(|d|^a)``; ``inf`` (without a warning) when ``a < 0`` and a
    voxel sits at or numerically near zero dose (see the guards above)."""
    import numpy as np

    with np.errstate(divide="ignore", over="ignore"):
        return float(np.mean(np.power(ad, a)))


def _ltcp_exp(d: Any, dp: float, alpha: float) -> Any:
    """``exp(-α(d - d_p))`` with the exponent clipped at :data:`_LTCP_EXP_CLIP`."""
    import numpy as np

    # Below the clip the value is constant but the derivative callers keep the
    # clipped factor: FD-vs-analytic checks disagree there (see the guards).
    return np.exp(np.minimum(-alpha * (d - dp), _LTCP_EXP_CLIP))


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
        return _geud_mean_power(np.abs(d), a) ** (1.0 / a)  # S = inf (a < 0) → 0
    if ctype == 4:  # LTCP
        dp, alpha = float(par[0]), float(par[1])
        return float(np.mean(_ltcp_exp(d, dp, alpha)))
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
        ad = np.abs(d)
        s = _geud_mean_power(ad, a)
        if s == 0.0 or s == float("inf"):  # all-zero / cold-voxel: see the guards
            return np.zeros_like(d)
        return (s ** (1.0 / a - 1.0)) * _geud_power(ad, a - 1.0) * np.sign(d) / m
    if ctype == 4:  # LTCP
        dp, alpha = float(par[0]), float(par[1])
        return (-alpha / m) * _ltcp_exp(d, dp, alpha)
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
        h = (alpha * alpha / m) * _ltcp_exp(d, dp, alpha)
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
        # van Haveren & Breedveld 2019, eqs. (8)–(9): diag(e2) + c2·e1 e1ᵀ.
        a = float(par[0])
        ad = np.abs(d)
        s = _geud_mean_power(ad, a)
        if s == 0.0 or s == float("inf"):  # all-zero / cold-voxel: see the guards
            return zeros, zeros, 0.0
        diag = s ** (1.0 / a - 1.0) * (a - 1.0) * _geud_power(ad, a - 2.0) / m
        u = _geud_power(ad, a - 1.0) * np.sign(d) / m  # (1/m) d^{a-1}
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


# --------------------------------------------------------------------------- #
# Tiled / compacted Gram of the linear block (Breedveld et al. 2017, §4.2/§5.2)
# --------------------------------------------------------------------------- #
# The condensed dense route forms ``N = W + Aᵀ Σ A`` once per IPM iteration, and
# at radiotherapy scale the linear block's Gram dominates the iteration (≈ 80% of
# the wall on Protons_01). The generic SciPy adapter accumulates it by chunked
# dense ``syrk`` over *all* ``n`` columns, yet a dose-matrix row block touches
# only the beamlets that reach its voxels. Breedveld 2017 §4.2 (eq. 48) forms
# the product per structure on compacted, spatially-ordered tiles, with the
# weights of every row that shares a structure matrix merged first; §5.2
# threads it one tile per core. This is that kernel at the application edge
# (SciPy/BLAS on the host), plugged into the core through the ``gram``
# override of a ``CSROperator`` subclass — the core lowering keeps a subclass
# intact (``_SignedRowSelection``) and hands it per-source-row summed weights.
@dataclass(frozen=True)
class TiledGramConfig:
    """Knobs of :class:`TiledGramCSROperator` (defaults measured on Protons_01).

    ``block_rows`` is the tile height ``B`` (rows of the RCM-ordered structure
    per tile). ``flop_ratio_threshold`` is the gate: when the tiled plan's
    arithmetic is above this fraction of the plain path's (``m·n²``), the
    operator keeps the plain adapter Gram — dense-ish plans (VMAT) gain nothing
    from compaction. A tile whose sparse-product arithmetic ``Σ nnz_row²`` is at
    most ``sparse_tile_flop_ratio`` of its dense ``syrk`` arithmetic ``b·c²`` is
    multiplied sparsely instead of densified (wide but nearly empty tiles, e.g.
    finite-difference smoothing rows); SciPy's SpGEMM runs ~100× below BLAS
    rate, so the break-even measured on the TROTS tiles sits near 1%.

    ``threads`` is the number of accumulation *lanes* (§5.2: one tile per
    worker at a time, private accumulators summed at the end); ``1`` runs the
    tiles sequentially. The lane count — and with it the summation order, so
    the result's last bits — depends only on this setting and the data, never
    on the machine: a pool of ``min(threads, os.cpu_count())`` workers serves
    the lanes. A Gram uses fewer lanes when there is little work
    (``lane_min_flops`` of BLAS per lane amortizes its private ``n²``
    accumulator) or when the accumulators would exceed ``lane_budget_bytes``
    (``n ≈ 10⁴`` ⇒ sequential). BLAS keeps its own threading either way
    (measured: pinning it to one thread makes the pool slower, since the
    compacted scatter holds the GIL).

    ``device_sparse_tiles`` (CuPy namespaces) multiplies the sparse tiles of
    exact (float64) Grams by cuSPARSE SpGEMM; ``False`` densifies them like
    every other tile. Reduced-precision requests and Torch densify them too.
    Host (NumPy) runs ignore it.

    Densifying a sparse tile is priced wrongly by the gate: the gate charges
    it ``Σ nnz_row²`` (its sparse-product arithmetic), but a densified tile
    costs the dense ``b·c²`` (twice that on Torch, whose ``matmul`` has no
    one-triangle ``syrk``) and a ``b×c`` temporary. So on Torch and for
    float32 requests the gate statistic underestimates the device cost of
    plans with sparse tiles. ``device_densify_budget_bytes`` caps the damage:
    a sparse tile whose densified ``b×c`` float64 temporary would exceed it
    is never densified on a device — CuPy multiplies it by cuSPARSE SpGEMM at
    every precision (and whatever ``device_sparse_tiles`` says), Torch by the
    host SciPy SpGEMM of the plan's tile (one weight download per such tile;
    free on CPU tensors). Host (NumPy) runs ignore it (they never densify a
    sparse tile).

    On a device namespace (CuPy / Torch) the exact Hessian of
    :class:`TROTSExactProblem` ignores ``flop_ratio_threshold`` for its
    curved structures: a structure the gate keeps off is planned anyway
    (lazily, on the first dense Hessian), because the tiled plan never costs
    more flops than the full-width chunks the gate compares it with, and the
    only other device option is a host product crossed per call. The linear
    block keeps the gate. ``tiled_gram=False`` is the only opt-out (the
    grouped Hessian is then crossed as a ``CSCOperator``).
    """

    block_rows: int = 2048
    flop_ratio_threshold: float = 0.5
    sparse_tile_flop_ratio: float = 0.01
    threads: int = 8
    lane_min_flops: float = 2.5e8
    lane_budget_bytes: int = 1 << 30
    device_sparse_tiles: bool = True
    device_densify_budget_bytes: int = 256 << 20


@dataclass(frozen=True)
class GramSegment:
    """Rows ``[start, stop)`` of a :class:`TiledGramCSROperator` are
    ``[sign·A, aux_coef·e_aux]``: a signed copy of one structure matrix ``A``
    (``rows × n_x`` host SciPy sparse, its columns the leading ``n_x`` operator
    columns) plus an optional constant auxiliary column (the minimax epigraph
    variable). Segments with equal ``key`` share ``A`` and one tiled plan: their
    weights are summed before the product (Breedveld 2017, eq. 48)."""

    start: int
    stop: int
    structure: Any
    key: Any
    sign: float = 1.0
    aux_col: int | None = None
    aux_coef: float = 0.0


@dataclass
class _Tile:
    start: int  # rows [start, stop) of the RCM-permuted structure
    stop: int
    cols: Any  # sorted active columns (int64)
    csr: Any  # compacted tile (stop-start) × len(cols), SciPy CSR
    sparse: bool  # multiply sparsely instead of densify + syrk
    csr_t: Any = None  # transpose (sparse tiles only)
    row_of_nnz: Any = None  # nnz → tile row (sparse tiles only)
    flops: float = 0.0


@dataclass
class _StructurePlan:
    structure: Any  # the original host matrix (rows × n_x), kept by reference
    perm: Any  # RCM row order
    tiles: list[_Tile]
    uses: list[GramSegment]


def _row_counts(a: Any) -> Any:
    """Stored entries per row of a SciPy sparse matrix, without converting it."""
    import numpy as np
    import scipy.sparse as sp

    if getattr(a, "format", None) == "csr":
        return np.diff(a.indptr)
    if getattr(a, "format", None) == "csc":
        return np.bincount(a.indices, minlength=a.shape[0])
    return np.diff(sp.csr_matrix(a).indptr)


def _rcm_row_order(a: Any) -> Any:
    """Row order of a reverse Cuthill–McKee sweep of the bipartite graph
    ``[[0, A], [Aᵀ, 0]]`` — rows sharing beamlets become neighbours, so a tile of
    consecutive rows touches few columns (Breedveld 2017 §5.2's spatial tiling).
    RCM: Cuthill & McKee 1969 / George 1971, via SciPy's
    ``scipy.sparse.csgraph.reverse_cuthill_mckee``."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    m = a.shape[0]
    pattern = sp.csr_matrix(
        (np.ones(a.nnz, dtype=np.int8), a.indices, a.indptr), shape=a.shape
    )
    graph = sp.bmat([[None, pattern], [pattern.T, None]], format="csr")
    order = np.asarray(reverse_cuthill_mckee(graph, symmetric_mode=True))
    return order[order < m].astype(np.int64)


def _plan_structure(
    a: Any, cfg: TiledGramConfig, store_dtype: Any
) -> tuple[Any, list[_Tile]]:
    """RCM-permute ``a`` and cut it into compacted ``block_rows``-row tiles.

    ``store_dtype`` float32 is a request, honoured only when every (canonical,
    duplicate-summed) value of ``a`` is an exact float32 widening — checked
    once per structure, as the matrix cache writer does. The tiles also serve
    exact (float64) Gram and Hessian requests, so a structure whose values are
    not float32-exact is stored in float64 instead; the caller's
    ``values_dtype_hint`` then only governs the reduced-accumulate path, where
    the float64 tile is rounded to float32 at product time exactly as the plain
    adapter's float32 accumulate rounds it.
    """
    import numpy as np
    import scipy.sparse as sp

    a = sp.csr_matrix(a, copy=True)  # canonicalize a private copy, never
    a.sum_duplicates()  # the caller's (shared, cached) structure matrix
    a.sort_indices()
    if np.dtype(store_dtype) == np.float32 and not np.array_equal(
        a.data.astype(np.float32).astype(np.float64), a.data
    ):
        store_dtype = np.float64  # not float32-exact: keep exact requests exact
    m = a.shape[0]
    perm = _rcm_row_order(a) if m > cfg.block_rows else np.arange(m, dtype=np.int64)
    ap = a[perm]
    tiles: list[_Tile] = []
    for start in range(0, m, cfg.block_rows):
        stop = min(m, start + cfg.block_rows)
        lo, hi = int(ap.indptr[start]), int(ap.indptr[stop])
        if hi == lo:
            continue  # an all-zero tile contributes nothing
        idx = ap.indices[lo:hi]
        cols, local = np.unique(idx, return_inverse=True)
        b, c = stop - start, int(cols.shape[0])
        csr = sp.csr_matrix(
            (
                ap.data[lo:hi].astype(store_dtype),
                local.astype(np.int32),
                (ap.indptr[start : stop + 1] - lo).astype(np.int32),
            ),
            shape=(b, c),
        )
        # Kernel choice by arithmetic, not density: the sparse product costs
        # Σ nnz_row² multiply-adds (plus its c×c output), the dense syrk b·c².
        sparse_flops = float(np.sum(np.diff(csr.indptr).astype(float) ** 2))
        dense_flops = float(b) * c * c
        sparse = sparse_flops <= cfg.sparse_tile_flop_ratio * dense_flops
        tile = _Tile(start, stop, cols.astype(np.int64), csr, sparse)
        if sparse:
            tile.csr_t = csr.T.tocsr()
            tile.row_of_nnz = np.repeat(
                np.arange(b, dtype=np.int32), np.diff(csr.indptr)
            )
            tile.flops = sparse_flops
        else:
            tile.flops = dense_flops
        tiles.append(tile)
    return perm, tiles


def _tile_gram_into(
    acc: Any, tile: _Tile, wp: Any, swp: Any, bad: Any, acc_dtype: Any, syrk: Any
) -> None:
    """Add ``tileᵀ diag(w) tile`` into ``acc``'s lower triangle (compacted scatter).

    Only the lower triangle of ``acc`` is meaningful; the caller mirrors once.
    """
    import numpy as np
    import scipy.sparse as sp

    w_t = wp[tile.start : tile.stop]
    if tile.sparse:
        data = tile.csr.data * w_t[tile.row_of_nnz]
        scaled = sp.csr_matrix(
            (data, tile.csr.indices, tile.csr.indptr), shape=tile.csr.shape
        )
        g = (tile.csr_t @ scaled).tocoo()
        keep = g.row >= g.col  # the lower triangle suffices (sorted cols)
        acc[tile.cols[g.row[keep]], tile.cols[g.col[keep]]] += g.data[keep]
        return
    dense = tile.csr.toarray()
    if dense.dtype != acc_dtype:
        dense = dense.astype(acc_dtype)
    if bad is None or not bool(bad[tile.start : tile.stop].any()):
        # Σ ≥ 0: symmetric rank-k update on √w-scaled rows, one triangle.
        dense *= swp[tile.start : tile.stop, None]
        g = syrk(1.0, dense.T, trans=0, lower=1)
    else:
        # Signed / non-finite weights (exact-Hessian nonconvex case, Breedveld
        # 2017 §5.1.2): no square root — general product for this tile.
        g = dense.T @ (w_t.astype(acc_dtype)[:, None] * dense)
    acc[np.ix_(tile.cols, tile.cols)] += g


def _mirror_lower(out: Any) -> Any:
    """Copy the lower triangle onto the upper one (bitwise-exact symmetry)."""
    import numpy as np

    out = np.ascontiguousarray(out)
    n = out.shape[0]
    for j in range(n):
        out[j, j + 1 :] = out[j + 1 :, j]
    return out


@dataclass
class TiledGramPlan:
    """The host-side tiled-Gram plan of one row block, built before (and
    independently of) its operator, so a caller can consult the gate without
    paying for the operator's Array-API arrays.

    ``flop_ratio`` is the gate statistic: Σ_r b_r·c_r² over the tiles (sparse
    tiles: Σ nnz_row²) against the full-width accumulate's m·n² — i.e. the
    mean compaction Σ c_r²/(R·n²) for uniform tiles. A cheap pre-gate rejects
    a plan before any permutation work when Σ_rows nnz_row² (per distinct
    structure) — a lower bound, since a tile is at least as wide as its widest
    row — already exceeds the threshold; ``flop_ratio`` is then that bound.

    ``gated_structures`` holds the structures of a plan the gate rejected
    after planning them (the pre-gate passed); it is empty for a tiled plan
    and for a pre-gate rejection. The device exact Hessian reuses them
    instead of planning the structure twice; other callers drop the plan.
    """

    shape: tuple[int, int]
    config: TiledGramConfig
    structures: list[_StructurePlan]
    nx: int
    flop_ratio: float
    tiled: bool
    seconds: float
    gated_structures: list[_StructurePlan] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        shape: tuple[int, int],
        segments: Sequence[GramSegment],
        *,
        values_dtype_hint: str | None = None,
        config: TiledGramConfig | None = None,
        registry: dict[Any, Any] | None = None,
    ) -> TiledGramPlan:
        """Plan ``segments`` (see the class docstring for the gate).

        ``registry`` (optional) shares per-structure tiles between plans: it
        maps ``(segment key, requested tile storage dtype, config)`` to the
        structure's ``(matrix, perm, tiles)``. A hit is reused instead of
        re-planned, and the structures of a plan that passes the gate are
        registered. The device exact Hessian of :class:`TROTSExactProblem`
        also registers the structures it plans past the gate (see
        :class:`TiledGramConfig`), under the same key. Either way a structure
        matrix used by both the linear block and the exact Hessian holds one
        tile copy, whichever side plans it first.
        """
        import time

        import numpy as np

        cfg = config if config is not None else TiledGramConfig()
        m, n = int(shape[0]), int(shape[1])
        store_dtype = np.float32 if values_dtype_hint == "float32" else np.float64
        t0 = time.perf_counter()
        grouped: dict[Any, tuple[Any, list[GramSegment]]] = {}
        covered = 0
        for seg in sorted(segments, key=lambda sg: sg.start):
            if seg.stop - seg.start != seg.structure.shape[0]:
                raise ValueError("segment rows must match its structure matrix")
            if seg.sign not in (1.0, -1.0):
                # The merged weights assume |sign| = 1 (eq. 48: signs square
                # away); a scaled copy would need sign² in the weight sum.
                raise ValueError(f"segment sign must be +1 or -1, got {seg.sign}")
            if not 0 <= seg.start <= seg.stop <= m or seg.structure.shape[1] > n:
                raise ValueError("segment lies outside the operator")
            if seg.start != covered:
                raise ValueError("segments must tile the operator rows exactly")
            covered = seg.stop
            if seg.aux_col is not None and not (
                seg.structure.shape[1] <= seg.aux_col < n
            ):
                raise ValueError("aux column must lie right of the structure")
            entry = grouped.setdefault(seg.key, (seg.structure, []))
            if entry[0] is not seg.structure:
                raise ValueError("segments sharing a key must share the structure")
            entry[1].append(seg)
        if covered != m:
            raise ValueError("segments must tile the operator rows exactly")
        plain_flops = float(m) * n * n
        # Pre-gate (see the class docstring): dense plans such as VMAT's
        # float32 group are rejected here without paying for RCM.
        lower_bound = sum(
            float(np.sum(_row_counts(a).astype(float) ** 2))
            for a, _uses in grouped.values()
        )
        structures: list[_StructurePlan] = []
        fresh: list[tuple[Any, Any]] = []
        if plain_flops > 0 and lower_bound / plain_flops > cfg.flop_ratio_threshold:
            tiled_flops = lower_bound
        else:
            for key, (a, uses) in grouped.items():
                reg_key = (key, np.dtype(store_dtype).str, cfg)
                hit = registry.get(reg_key) if registry is not None else None
                if hit is not None and hit[0] is a:
                    perm, tiles = hit[1], hit[2]
                else:
                    perm, tiles = _plan_structure(a, cfg, store_dtype)
                    fresh.append((reg_key, (a, perm, tiles)))
                structures.append(_StructurePlan(a, perm, tiles, uses))
            tiled_flops = sum(t.flops for p in structures for t in p.tiles)
        ratio = tiled_flops / plain_flops if plain_flops > 0 else 1.0
        tiled = ratio <= cfg.flop_ratio_threshold
        if tiled and registry is not None:
            registry.update(fresh)
        return cls(
            shape=(m, n),
            config=cfg,
            structures=structures if tiled else [],
            nx=max((p.structure.shape[1] for p in structures), default=0),
            flop_ratio=ratio,
            tiled=tiled,
            seconds=time.perf_counter() - t0,
            # Gated after planning: handed back (callers that do not reuse
            # them drop the tile copies with the plan).
            gated_structures=[] if tiled else structures,
        )


class TiledGramCSROperator(CSROperator):
    """A ``CSROperator`` whose :meth:`gram` is the per-structure tiled product.

    Behaves exactly like the plain :class:`~ipax.backend.operators.CSROperator`
    over ``csr`` (same rows, same order, same matvec family, same COO for the
    sparse routes) — the row permutation and tiling are private to the Gram
    plan. ``segments`` describe which structure matrix each row range copies
    (see :class:`GramSegment`); omitted, the whole matrix is one segment.

    The Gram ``Aᵀ diag(w) A`` is formed per structure ``s`` as
    ``A_sᵀ diag(Σ_uses w_u) A_s`` (Breedveld 2017, eq. 48 — rows sharing a
    matrix merge their weights, so a min/max pair or a constraint reused as a
    minimax link is multiplied once), over RCM-ordered row tiles compacted to
    their active columns: dense ``syrk`` on ``√w``-scaled tiles (``gemm`` where a
    tile's merged weights are negative or non-finite), a sparse product on
    nearly-empty tiles, one-triangle accumulation and a single mirror at the
    end, plus the rank-1 borders of the auxiliary columns (``Aᵀw`` matvecs).
    ``accumulate_dtype="float32"`` densifies and runs ``ssyrk`` in float32 and
    accumulates the tiles into a float64 result. The gate
    (:attr:`TiledGramConfig.flop_ratio_threshold`) keeps the plain adapter Gram
    when compaction would not pay.

    On NumPy namespaces the kernels are host BLAS/SciPy (threaded lanes). On
    CuPy and Torch namespaces the same plan runs in the namespace: the tiles
    are materialized once on the operator's device (``device_cache`` shares
    them between operators of one problem) and multiplied there — cuBLAS
    ``syrk`` on CuPy, ``matmul`` on Torch — see :func:`_device_accumulate`.
    The result is an array of the solve namespace on that device.

    On NumPy a weight vector with at most half of its entries nonzero — the
    restoration phase's active sets — is multiplied from those rows alone
    (:class:`_ActiveRowGram`) instead of through every tile.
    """

    def __init__(
        self,
        csr: Any,
        xp: Namespace,
        *,
        segments: Sequence[GramSegment] | None = None,
        values_dtype_hint: str | None = None,
        config: TiledGramConfig | None = None,
        plan: TiledGramPlan | None = None,
        device_cache: dict[Any, Any] | None = None,
    ) -> None:
        import numpy as np
        import scipy.sparse as sp

        from ipax.backend.namespace import device as device_of

        csr = sp.csr_matrix(csr)
        csr.sort_indices()
        super().__init__(
            xp.asarray(np.asarray(csr.indptr, dtype=np.int64)),
            xp.asarray(np.asarray(csr.indices, dtype=np.int64)),
            _from_numpy(xp, csr.data),
            (int(csr.shape[0]), int(csr.shape[1])),
            symmetric=False,
            values_dtype_hint=values_dtype_hint,
        )
        self._xp_ns = xp
        if plan is None:
            plan = TiledGramPlan.build(
                self.shape,
                segments
                if segments is not None
                else (GramSegment(0, self.shape[0], csr, key="whole"),),
                values_dtype_hint=values_dtype_hint,
                config=config,
            )
        elif plan.shape != self.shape:
            raise ValueError("the plan was built for a different shape")
        if segments is not None:
            # Cheap consistency check (O(#segments)): the rows the segments
            # describe must carry exactly the operator's stored entries.
            expected = sum(
                int(sg.structure.nnz) + (sg.stop - sg.start) * (sg.aux_col is not None)
                for sg in segments
            )
            if expected != int(csr.nnz):
                raise ValueError(
                    "segments do not describe the operator's entries "
                    f"({expected} stored entries vs {int(csr.nnz)})"
                )
        self.config = plan.config
        self._plans = plan.structures if plan.tiled else []
        self._nx = plan.nx
        #: The gate statistic (see :attr:`TiledGramPlan.flop_ratio`).
        self.flop_ratio = plan.flop_ratio
        #: Whether :meth:`gram` runs the tiled kernel (else the plain adapter).
        self.tiled = plan.tiled
        #: Wall seconds spent building the plan (permutations + tiles).
        self.plan_seconds = plan.seconds
        self._memo: dict[str | None, tuple[Any, Any]] = {}
        self.gram_compute_count = 0  # observability: actual tiled products
        #: Host restoration Grams from the active rows (see _ActiveRowGram).
        self.active_rows = _ActiveRowGram(csr, self.config)
        #: Device execution context (CuPy / Torch namespaces), else None.
        self._device: _DeviceGramContext | None = None
        kind = None if _is_host_numpy(xp) else _device_kind(xp)
        if kind is not None:
            self._device = _DeviceGramContext(
                xp,
                kind,
                device_of(self.coo_values()),
                device_cache,
                sparse_tiles=self.config.device_sparse_tiles,
                densify_budget_bytes=self.config.device_densify_budget_bytes,
            )

    # -- the Gram ---------------------------------------------------------------
    def gram(
        self,
        weights: Array,
        *,
        accumulate_dtype: str | None = None,
        hinted_only: bool = False,
    ) -> Array:
        if not self.tiled:
            return super().gram(
                weights, accumulate_dtype=accumulate_dtype, hinted_only=hinted_only
            )
        import numpy as np

        if (
            hinted_only
            and accumulate_dtype is not None
            and self.gram_accumulate_dtype_hint() != accumulate_dtype
        ):
            accumulate_dtype = None  # this block's data does not support it
        if self._device is not None:
            return self._device_gram(weights, accumulate_dtype)
        w = np.asarray(_to_numpy(weights), dtype=np.float64)
        # Restoration's active-set Grams (a few nonzero weights): the active
        # rows alone rather than every tile (see _ActiveRowGram).
        compact = self.active_rows.gram(w, accumulate_dtype)
        if compact is not None:
            return self._xp_ns.asarray(compact)
        # Last-weights memo, like the adapter's: δ_w retries re-request the
        # identical Σ within one IPM iteration.
        acc_dtype = np.dtype(accumulate_dtype or np.float64)
        if acc_dtype not in (np.float32, np.float64):
            acc_dtype = np.dtype(np.float64)  # best-effort: no BLAS syrk for it
        if acc_dtype == np.float64:
            accumulate_dtype = None  # one memo slot for every exact request
        memo = self._memo.get(accumulate_dtype)
        if memo is not None and np.array_equal(w, memo[0]):
            return self._xp_ns.asarray(memo[1])
        out = self._tiled_gram(w, acc_dtype)
        self._memo[accumulate_dtype] = (np.array(w, copy=True), out)
        self.gram_compute_count += 1
        return self._xp_ns.asarray(out)

    def _tiled_gram(self, w: Any, acc_dtype: Any) -> Any:
        import numpy as np

        n = self.shape[1]
        nx = self._nx
        out = np.zeros((n, n), dtype=np.float64)
        work: list[tuple[_Tile, Any, Any, Any]] = []
        for plan in self._plans:
            a = plan.structure
            ws = np.zeros((a.shape[0],), dtype=np.float64)
            for seg in plan.uses:
                w_u = w[seg.start : seg.stop]
                ws += w_u  # signs square away (eq. 48)
                if seg.aux_col is not None:
                    # Border of the constant aux column: rows [s·a_i, c·e_k]
                    # give (x,t) = s·c·Aᵀw and (t,t) = c²·Σw — lower triangle.
                    k = seg.aux_col
                    border = np.asarray(a.T @ w_u).reshape(-1)
                    out[k, : a.shape[1]] += (seg.sign * seg.aux_coef) * border
                    out[k, k] += seg.aux_coef**2 * float(np.sum(w_u))
            work.extend(_tile_work(ws, plan.perm, plan.tiles, acc_dtype))
        _accumulate_tiles(out[:nx, :nx], work, self.config, acc_dtype)
        return _mirror_lower(out)

    # -- the device Gram (CuPy / Torch) -----------------------------------------
    def device_structures(self) -> list[_DeviceStructure]:
        """The device tiles of every planned structure (materialized on first
        use, then cached in the context's ``device_cache``)."""
        ctx = self._device
        if ctx is None:
            return []
        rows = self.config.block_rows
        return [
            ctx.structure(p.perm, p.tiles, p.structure.shape[0], rows)
            for p in self._plans
        ]

    def _device_gram(self, weights: Array, accumulate_dtype: str | None) -> Array:
        """The device Gram: one host read per call (see _device_accumulate)."""
        import numpy as np

        ctx = self._device
        assert ctx is not None
        xp = ctx.xp
        acc_name = accumulate_dtype or "float64"
        if acc_name not in ("float32", "float64"):
            acc_name = "float64"  # best-effort, as on the host
        key = None if acc_name == "float64" else acc_name
        w = xp.reshape(xp.asarray(weights, device=ctx.device), (-1,))
        if w.dtype != xp.float64:
            w = xp.astype(w, xp.float64)
        items: list[tuple[_DeviceStructure, Any]] = []
        aux: list[GramSegment] = []
        for plan, ds in zip(self._plans, self.device_structures(), strict=True):
            ws = None
            for seg in plan.uses:
                w_u = w[seg.start : seg.stop]
                ws = w_u if ws is None else ws + w_u  # signs square away (eq. 48)
                if seg.aux_col is not None:
                    aux.append(seg)
            items.append((ds, ws))
        prepared, flag_dev = _device_prepare(ctx, items, acc_name)
        # Last-weights memo (δ_w retries). Its comparison is a device
        # reduction read back together with the per-tile weight flags, so a
        # call costs one host read whether it hits or computes (a hit wastes
        # only the few launches of the weight preparation above).
        memo = self._memo.get(key)
        same = None
        if memo is not None and memo[0].shape == w.shape:
            same = xp.reshape(xp.all(w == memo[0]), (1,))
        parts = [p for p in (same, flag_dev) if p is not None]
        host = (
            np.asarray(ctx.to_host(xp.concat(parts)), dtype=bool)
            if parts
            else np.zeros((0,), dtype=bool)
        )
        if same is not None:
            if bool(host[0]):
                assert memo is not None
                return memo[1]
            host = host[1:]
        out = self._device_tiled_gram(w, acc_name, prepared, host, aux)
        self._memo[key] = (xp.asarray(w, copy=True), out)
        self.gram_compute_count += 1
        return out

    def _device_tiled_gram(
        self,
        w: Any,
        acc_name: str,
        prepared: list[tuple[_DeviceStructure, Any, Any]],
        flags: Any,
        aux: list[GramSegment],
    ) -> Any:
        ctx = self._device
        assert ctx is not None
        xp = ctx.xp
        n = self.shape[1]
        out = xp.zeros((n, n), dtype=xp.float64, device=ctx.device)
        _device_run(ctx, out, prepared, flags, acc_name)
        if aux:
            # Borders of the constant aux columns (lower triangle, row k):
            # a segment's rows are [s·A, c·e_k], so the operator's own
            # transpose product with the weights masked to that segment is
            # r = [s·Aᵀw_u; c·Σw_u·e_k] and row k gains c·r — (x,t) = s·c·Aᵀw_u,
            # (t,t) = c²·Σw_u — one SpMM for every aux segment at once.
            masked = xp.zeros(
                (self.shape[0], len(aux)), dtype=xp.float64, device=ctx.device
            )
            for j, seg in enumerate(aux):
                masked[seg.start : seg.stop, j] = w[seg.start : seg.stop]
            r = self.rmatmat(masked)
            for j, seg in enumerate(aux):
                assert seg.aux_col is not None
                out[seg.aux_col, :] += seg.aux_coef * r[:, j]
        return ctx.mirror_lower(out)


class _ActiveRowGram:
    """``Aᵀ diag(w) A`` from the rows of ``A`` with a nonzero weight (host).

    The Gauss–Newton restoration phase asks the linear block for
    ``Jᵀ diag(1[g > 0]) J`` (Wächter & Biegler 2006, §3.3) once per
    Levenberg–Marquardt step, and only a handful of rows is violated: on
    Prostate_VMAT_101 about 100 of the 81279 lowered rows. Forming the full
    product there was ~45% of each restoration's wall. When at most
    ``max_fraction`` of the weights are nonzero, :meth:`gram` forms the
    product from those rows alone (:meth:`_product`) — the same matrix, since
    a zero-weight row adds nothing, at a different round-off; otherwise it
    returns None and the caller runs its usual product. Interior-point weights
    (``Σ_s > 0``) never take this path, so those products are untouched — as
    long as the lowered rows cover more than half of the source rows (a row
    with no finite bound gets weight 0 in the core's signed row selection;
    every TROTS linear row has exactly one finite side).
    """

    def __init__(
        self, csr: Any, config: TiledGramConfig, max_fraction: float = 0.5
    ) -> None:
        self.csr = csr
        self.config = config
        self.max_fraction = max_fraction
        self._memo: dict[str | None, tuple[Any, Any]] = {}
        #: Observability: compacted products actually formed.
        self.count = 0
        #: Observability: compacted ``rmatvec`` calls (the operator's own).
        self.rmatvec_count = 0

    def _product(self, wa: Any, active: Any, acc_dtype: Any) -> Any:
        """``A_Sᵀ diag(w_S) A_S`` by BLAS over ``block_rows`` chunks of the
        active rows ``S`` (the plain adapter's chunked accumulation: one
        ``syrk`` target — ``gemm`` for signed or non-finite weights — and the
        lower triangle mirrored once), accumulated in ``acc_dtype`` and
        returned in float64."""
        import numpy as np
        from scipy.linalg.blas import get_blas_funcs

        n = int(self.csr.shape[1])
        if active.size == 0:
            return np.zeros((n, n), dtype=np.float64)
        sub = self.csr[active]
        use_syrk = bool(np.all(np.isfinite(wa) & (wa >= 0.0)))
        if use_syrk:
            syrk = get_blas_funcs("syrk", dtype=acc_dtype)
            scale = np.sqrt(wa).astype(acc_dtype)
            acc = np.zeros((n, n), dtype=acc_dtype, order="F")
        else:
            acc = np.zeros((n, n), dtype=acc_dtype)
        step = self.config.block_rows
        for start in range(0, sub.shape[0], step):
            stop = min(sub.shape[0], start + step)
            block = sub[start:stop].toarray().astype(acc_dtype, copy=False)
            if use_syrk:
                block *= scale[start:stop, None]
                acc = syrk(
                    1.0, block.T, beta=1.0, c=acc, trans=0, lower=1, overwrite_c=1
                )
            else:
                acc += block.T @ (wa[start:stop, None].astype(acc_dtype) * block)
        # syrk fills the lower triangle only; mirroring it also makes the
        # gemm product exactly symmetric, like the tiled kernel's.
        return _mirror_lower(acc.astype(np.float64))

    def gram(self, w: Any, accumulate_dtype: str | None) -> Any | None:
        """The compacted product (host NumPy ``n×n``), or None to decline.

        ``w`` is the host float64 weight vector; ``accumulate_dtype`` the
        caller's resolved request (``None`` or ``"float32"``).
        """
        import numpy as np

        active = np.flatnonzero(w != 0.0)  # non-finite weights stay in
        if active.size > self.max_fraction * w.shape[0]:
            return None
        acc_dtype = np.dtype(accumulate_dtype or np.float64)
        if acc_dtype not in (np.float32, np.float64):
            acc_dtype = np.dtype(np.float64)
        key = None if acc_dtype == np.float64 else acc_dtype.name
        # Last-weights memo: a rejected LM trial retries at a larger damping
        # with the same active set.
        memo = self._memo.get(key)
        if memo is not None and np.array_equal(w, memo[0]):
            return memo[1]
        out = self._product(w[active], active, acc_dtype)
        # Returned as is on a memo hit: callers treat it as read-only (the
        # adapter's contract; restoration and VStack add out of place).
        self._memo[key] = (np.array(w, copy=True), out)
        self.count += 1
        return out


def _resolve_accumulate(
    op: LinearOperator, accumulate_dtype: str | None, hinted_only: bool
) -> str | None:
    """The accumulate request an operator honours (``hinted_only``: only a
    block whose data hint matches reduces)."""
    if (
        hinted_only
        and accumulate_dtype is not None
        and op.gram_accumulate_dtype_hint() != accumulate_dtype
    ):
        return None  # this block's data does not support it
    return accumulate_dtype


class ActiveRowGramCSROperator(CSROperator):
    """A host ``CSROperator`` whose Gram skips zero-weight rows.

    The linear-block group a :class:`TiledGramPlan` gates off on a NumPy
    namespace (dense-ish rows, e.g. Prostate_VMAT_101's float32 group: 65524
    rows at 64% density). Everything is the plain operator, :meth:`gram`
    included — the adapter's chunked dense accumulation with its memo and
    mixed-precision path — except for sparse weight vectors (the restoration
    phase's active sets), which :class:`_ActiveRowGram` serves from the
    active rows alone; :meth:`rmatvec` likewise skips the zero rows of a
    mostly-zero vector (restoration's gradient ``Jᵀ max(g, 0)``).

    Being a subclass, the operator is lowered by the core as a signed row
    selection around it (:func:`ipax.problem.linear_ineq._lower_two_sided`),
    which folds the selection into the Gram weights, instead of rewriting its
    triplets (on Prostate_VMAT_101 the ``ipax.solve`` setup drops from 10.3 s
    to 5.1 s and the Breedveld start from 31.9 s to 27.0 s, both of which
    lowered the 82M-entry group). For a group whose rows are all
    upper-bounded (Prostate_VMAT_101's) the folded weights are the lowered
    ones in the same order, so the interior-point products are bit-identical
    to the triplet-lowered plain operator's; a group mixing min and max rows
    (Liver_01's float32 group) sums its products in the source row order
    instead of the lowered one — round-off only.
    """

    def __init__(
        self,
        csr: Any,
        xp: Namespace,
        *,
        values_dtype_hint: str | None = None,
        config: TiledGramConfig | None = None,
    ) -> None:
        import numpy as np
        import scipy.sparse as sp

        if not _is_host_numpy(xp):
            raise ValueError("ActiveRowGramCSROperator is host-NumPy only")
        csr = sp.csr_matrix(csr)
        csr.sort_indices()
        super().__init__(
            xp.asarray(np.asarray(csr.indptr, dtype=np.int64)),
            xp.asarray(np.asarray(csr.indices, dtype=np.int64)),
            _from_numpy(xp, csr.data),
            (int(csr.shape[0]), int(csr.shape[1])),
            symmetric=False,
            values_dtype_hint=values_dtype_hint,
        )
        self._xp_ns = xp
        self.active_rows = _ActiveRowGram(
            csr, config if config is not None else TiledGramConfig()
        )

    def gram(
        self,
        weights: Array,
        *,
        accumulate_dtype: str | None = None,
        hinted_only: bool = False,
    ) -> Array:
        import numpy as np

        w = np.asarray(_to_numpy(weights), dtype=np.float64).reshape(-1)
        out = self.active_rows.gram(
            w, _resolve_accumulate(self, accumulate_dtype, hinted_only)
        )
        if out is not None:
            return self._xp_ns.asarray(out)
        return super().gram(
            weights, accumulate_dtype=accumulate_dtype, hinted_only=hinted_only
        )

    def rmatvec(self, v: Array) -> Array:
        """``Aᵀv``; a mostly-zero ``v`` from its nonzero rows alone.

        Restoration's Gauss–Newton gradient ``Jᵀ max(g, 0)`` has as few
        nonzeros as its active set. The compacted product adds the same
        terms in the same (row) order as the full CSC product — only the
        zero rows' ``±0`` terms are skipped — so it is the same vector.
        """
        import numpy as np

        y = _to_numpy_nd(v)
        if y.ndim != 1:
            return super().rmatvec(v)
        nz = np.flatnonzero(y)
        if nz.size > self.active_rows.max_fraction * y.shape[0]:
            return super().rmatvec(v)
        out = self.active_rows.csr[nz].T @ y[nz]
        self.active_rows.rmatvec_count += 1
        return self._xp_ns.asarray(np.asarray(out, dtype=np.float64))


def _tile_work(
    w: Any, perm: Any, tiles: Sequence[_Tile], acc_dtype: Any
) -> list[tuple[_Tile, Any, Any, Any]]:
    """Work items of one structure: its tiles with the RCM-permuted weights,
    their square roots (syrk) and the signed / non-finite mask (gemm)."""
    import numpy as np

    wp = w[perm] if perm is not None else w
    finite_nonneg = np.isfinite(wp) & (wp >= 0.0)
    bad = None if bool(finite_nonneg.all()) else ~finite_nonneg
    swp = np.sqrt(np.where(finite_nonneg, wp, 0.0)).astype(acc_dtype)
    return [(tile, wp, swp, bad) for tile in tiles]


def _accumulate_tiles(
    acc: Any,
    work: list[tuple[_Tile, Any, Any, Any]],
    cfg: TiledGramConfig,
    acc_dtype: Any,
) -> None:
    """Add every ``(tile, w_perm, sqrt_w_perm, bad)`` item's weighted Gram into
    the lower triangle of ``acc`` (an ``nx×nx`` view), on the configured lanes.

    Shared by the linear block's :class:`TiledGramCSROperator` and the exact
    Hessian's :class:`TROTSExactHessian`. The lane count is a function of the
    config and the data only (see :class:`TiledGramConfig`), so the round-off
    does not depend on the machine.
    """
    from scipy.linalg.blas import get_blas_funcs

    if not work:
        return
    nx = int(acc.shape[0])
    syrk = get_blas_funcs("syrk", dtype=acc_dtype)
    total_flops = sum(item[0].flops for item in work)
    threads = min(
        max(1, int(cfg.threads)),
        len(work),
        max(1, int(total_flops // max(cfg.lane_min_flops, 1.0))),
        max(1, int(cfg.lane_budget_bytes) // max(1, 8 * nx * nx)),
    )
    if threads == 1:
        for tile, wp, swp, bad in work:  # scatter straight into the view
            _tile_gram_into(acc, tile, wp, swp, bad, acc_dtype, syrk)
    else:
        acc += _threaded_tiles(work, nx, acc_dtype, syrk, threads)


def _threaded_tiles(
    work: list[tuple[_Tile, Any, Any, Any]],
    nx: int,
    acc_dtype: Any,
    syrk: Any,
    threads: int,
) -> Any:
    """One tile per worker at a time (Breedveld 2017 §5.2), private
    accumulators summed at the end; tiles are dealt to workers by flops."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    lanes: list[list[tuple[_Tile, Any, Any, Any]]] = [[] for _ in range(threads)]
    load = [0.0] * threads
    for item in sorted(work, key=lambda it: -it[0].flops):
        lane = min(range(threads), key=load.__getitem__)
        lanes[lane].append(item)
        load[lane] += item[0].flops

    def run(items: list[tuple[_Tile, Any, Any, Any]]) -> Any:
        acc = np.zeros((nx, nx))
        for tile, wp, swp, bad in items:
            _tile_gram_into(acc, tile, wp, swp, bad, acc_dtype, syrk)
        return acc

    workers = max(1, min(threads, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(run, [lane for lane in lanes if lane]))
    total = parts[0]
    for part in parts[1:]:
        total += part
    return total


# --------------------------------------------------------------------------- #
# Device (CuPy / Torch) execution of the tiled plans
# --------------------------------------------------------------------------- #
# The plan (RCM order, compacted tile column sets, tile values) is host index /
# value data, so the same plan drives a device kernel: each tile is materialized
# once in the solve namespace as a flat dense-scatter index ``row·c + col`` plus
# its values (≈ the host plan's 12 B/nnz, 8 B/nnz for float32-stored tiles),
# densified per call with one scatter, scaled by √w and multiplied into a c×c
# block — cuBLAS ``syrk`` on CuPy (one triangle, as the CuPy adapter does),
# ``matmul`` on Torch (no public syrk: twice the flops) — which is scatter-added
# at ``ix_(cols, cols)`` into the n×n result and mirrored once. Tiles run
# sequentially on the stream (kernel launches are asynchronous; no lanes), so
# the summation order is fixed by the plan. The host plan (its tile CSRs,
# ≈ 12 B/nnz) is retained next to the device copy: it is the registry's
# sharing key, and Torch's over-budget sparse tiles multiply it on the host
# (see TiledGramConfig.device_densify_budget_bytes).
#
# Host synchronizations per computing Gram: one host read (the per-tile
# weight flags, with the memo comparison folded into it when the memo slot is
# filled), plus — on exact CuPy Grams — at least one per sparse tile inside
# cuSPARSE SpGEMM, which reads the product's nnz back to size its output
# (unavoidable through cupyx). Torch's over-budget sparse tiles add one
# weight download each (none on CPU tensors).
_DEVICE_MIRROR_BLOCK = 1024


def _device_kind(xp: Namespace) -> str | None:
    """``"cupy"``/``"torch"`` when ``xp`` produces CuPy/Torch arrays (the device
    kernel's namespaces), else None (NumPy, JAX, ``array_api_strict``: JAX
    arrays are immutable and ``array_api_strict`` has no sparse adapter, so
    both keep the plain operator)."""
    try:
        probe = xp.asarray(0.0)
    except Exception:  # namespace probing only
        return None
    module = type(probe).__module__.split(".")[0]
    return module if module in ("cupy", "torch") else None


@dataclass
class _DeviceTile:
    start: int  # rows [start, stop) of the RCM-permuted structure
    stop: int
    width: int  # compacted columns c
    lo: int  # the tile's slice of the structure's flat / data arrays
    hi: int
    cols: Any  # device int64 (sorted, unique)
    sparse: bool  # CuPy SpGEMM kernel (else densify + syrk / matmul)
    csr: Any = None  # CuPy sparse tiles: device CSR (b×c) and its transpose
    csr_t: Any = None  # (float64: the product's dtype, so cupyx never recasts it)
    row_of_nnz: Any = None
    #: A sparse tile over the densify budget: never densified (CuPy: SpGEMM
    #: at every precision; Torch: the host SciPy product of ``host``).
    capped: bool = False
    host: Any = None  # the host plan's _Tile (Torch capped tiles only)


@dataclass
class _DeviceStructure:
    rows: int
    block_rows: int
    perm: Any  # device int64 RCM order
    flat: Any  # device int32/int64 dense-scatter index row·c + col
    data: Any  # device values (the tile storage dtype)
    tiles: list[_DeviceTile]
    nbytes: int  # device bytes held by this structure's tiles


class _DeviceGramContext:
    """Namespace primitives and the per-structure device tile cache of the
    device tiled Gram (shared by :class:`TiledGramCSROperator` and
    :class:`TROTSDeviceExactHessian`).

    ``cache`` maps ``(id(host tiles), *signature)`` to ``(host tiles,
    _DeviceStructure)``, where the signature is the namespace kind, the
    device and the kernel-selection knobs (so contexts that would materialize
    a structure differently never share an entry); the loader passes one
    dict per problem, so a structure planned once (plan registry) is also
    materialized on the device once.
    """

    def __init__(
        self,
        xp: Namespace,
        kind: str,
        device: Any,
        cache: dict[Any, Any] | None = None,
        *,
        sparse_tiles: bool = True,
        densify_budget_bytes: int | None = None,
    ) -> None:
        # Normalize to the Array API compat namespace: the device kernels use
        # ``device=`` keywords (``asarray``/``zeros``) that a raw ``cupy`` module
        # does not accept, so a problem built on plain ``cupy`` would fail at
        # its first Gram (the compat wrapper adds the keyword; a no-op for one).
        from ipax.backend.namespace import array_namespace

        self.xp = array_namespace(xp.asarray(0.0))
        self.kind = kind
        self.device = device
        self.cache: dict[Any, Any] = cache if cache is not None else {}
        # Sparse tiles keep a sparse kernel only where the namespace has a
        # sparse×sparse product we can drive from here (cuSPARSE via cupyx);
        # Torch densifies them (see the class comment above).
        self.sparse_tiles = bool(sparse_tiles) and kind == "cupy"
        #: Densified-tile cap (bytes of a b×c float64 temporary); see
        #: TiledGramConfig.device_densify_budget_bytes.
        self.densify_budget = int(
            densify_budget_bytes
            if densify_budget_bytes is not None
            else TiledGramConfig.device_densify_budget_bytes
        )
        self.signature = (kind, str(device), self.sparse_tiles, self.densify_budget)
        self._syrk: Any = None
        if kind == "cupy":
            try:
                from cupy import cublas

                self._syrk = cublas.syrk
            except ImportError:  # pragma: no cover - degraded CuPy install
                self._syrk = None

    # -- transfers ------------------------------------------------------------
    def put(self, host: Any, dtype: Any = None) -> Any:
        import numpy as np

        arr = np.ascontiguousarray(host if dtype is None else host.astype(dtype))
        return self.xp.asarray(arr, device=self.device)

    def to_host(self, x: Any) -> Any:
        import numpy as np

        if self.kind == "cupy":
            import cupy

            return cupy.asnumpy(x)
        if self.kind == "torch":
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def dtype(self, name: str) -> Any:
        return getattr(self.xp, name)

    # -- the tile cache ---------------------------------------------------------
    def structure(
        self, perm: Any, tiles: list[_Tile], rows: int, block_rows: int
    ) -> _DeviceStructure:
        key = (id(tiles), *self.signature)
        hit = self.cache.get(key)
        if hit is not None and hit[0] is tiles:
            return hit[1]
        ds = self._materialize(perm, tiles, rows, block_rows)
        self.cache[key] = (tiles, ds)
        return ds

    def _materialize(
        self, perm: Any, tiles: list[_Tile], rows: int, block_rows: int
    ) -> _DeviceStructure:
        import numpy as np

        flats: list[Any] = []
        values_parts: list[Any] = []
        dtiles: list[_DeviceTile] = []
        offset = 0
        largest = 0
        for t in tiles:
            b, c = t.stop - t.start, int(t.cols.shape[0])
            counts = np.diff(t.csr.indptr)
            local_row = np.repeat(np.arange(b, dtype=np.int64), counts)
            flats.append(local_row * c + t.csr.indices.astype(np.int64))
            values_parts.append(t.csr.data)
            largest = max(largest, b * c)
            # A sparse tile whose densified float64 b×c temporary would
            # exceed the budget is never densified (TiledGramConfig).
            capped = bool(t.sparse) and b * c * 8 > self.densify_budget
            dt = _DeviceTile(
                t.start,
                t.stop,
                c,
                offset,
                offset + int(t.csr.nnz),
                self.put(t.cols, np.int64),
                bool(t.sparse)
                and (self.sparse_tiles or (capped and self.kind == "cupy")),
                capped=capped,
            )
            if capped and not dt.sparse:
                dt.host = t  # no device SpGEMM here: the host SciPy product
            dtiles.append(dt)
            offset += int(t.csr.nnz)
        index_dtype = np.int32 if largest < 2**31 else np.int64
        store = tiles[0].csr.dtype if tiles else np.float64
        flat = np.concatenate(flats) if flats else np.zeros(0, dtype=np.int64)
        data = (
            np.concatenate(values_parts) if values_parts else np.zeros(0, dtype=store)
        )
        flat_dev = self.put(flat, index_dtype)
        data_dev = self.put(data)
        nbytes = flat.size * np.dtype(index_dtype).itemsize + data.nbytes
        nbytes += int(np.asarray(perm).size) * 8
        for t, dt in zip(tiles, dtiles, strict=True):
            if dt.sparse:
                import cupyx.scipy.sparse as cxs

                dt.csr = cxs.csr_matrix(
                    (
                        data_dev[dt.lo : dt.hi],
                        self.put(t.csr.indices, np.int32),
                        self.put(t.csr.indptr, np.int32),
                    ),
                    shape=t.csr.shape,
                )
                # The plan's tiles are canonical (sorted, duplicate-free), and
                # so is their transpose: say so, or cupyx checks it with a
                # kernel and a host sync. The transpose is stored in float64,
                # the product's dtype, so cupyx does not recast a float32
                # tile's transpose on every call.
                dt.csr.has_canonical_format = True
                csr_t = dt.csr.T.tocsr()
                if csr_t.dtype != np.float64:
                    csr_t = csr_t.astype(np.float64)
                csr_t.has_canonical_format = True
                dt.csr_t = csr_t
                dt.row_of_nnz = self.put(
                    np.repeat(
                        np.arange(t.stop - t.start, dtype=np.int32),
                        np.diff(t.csr.indptr),
                    )
                )
                # csr's indices/indptr, csr_t's indices/indptr/data, row_of_nnz
                # (csr's data is a view of the structure's values).
                nnz = int(t.csr.nnz)
                nbytes += 3 * nnz * 4 + nnz * 8 + (t.stop - t.start + dt.width + 2) * 4
        return _DeviceStructure(
            rows=int(rows),
            block_rows=int(block_rows),
            perm=self.put(np.asarray(perm), np.int64),
            flat=flat_dev,
            data=data_dev,
            tiles=dtiles,
            nbytes=int(nbytes),
        )

    # -- kernels ----------------------------------------------------------------
    def gram_block(self, dense: Any) -> Any:
        """``denseᵀ dense`` (c×c): lower triangle via cuBLAS ``syrk`` on CuPy
        (upper left zero), the full product via ``matmul`` on Torch."""
        if self._syrk is not None:
            return self._syrk("T", dense, lower=True)
        return self.xp.matmul(self.xp.permute_dims(dense, (1, 0)), dense)

    def scatter_block(self, out: Any, cols: Any, block: Any) -> None:
        """``out[ix_(cols, cols)] += block`` (unique ``cols``: no collisions)."""
        rows_ix, cols_ix = cols[:, None], cols[None, :]
        if self.kind == "cupy":
            import cupy

            cupy.add.at(out, (rows_ix, cols_ix), block)  # one kernel, casts
        else:
            if block.dtype != out.dtype:
                block = self.xp.astype(block, out.dtype)
            out.index_put_((rows_ix, cols_ix), block, accumulate=True)

    def scatter_coo(self, out: Any, rows: Any, cols: Any, vals: Any) -> None:
        """``out[rows[k], cols[k]] += vals[k]`` (duplicates accumulate)."""
        if self.kind == "cupy":
            import cupy

            cupy.add.at(out, (rows, cols), vals)
        else:
            if vals.dtype != out.dtype:
                vals = self.xp.astype(vals, out.dtype)
            out.index_put_((rows, cols), vals, accumulate=True)

    def mirror_lower(self, out: Any) -> Any:
        """Copy the lower triangle onto the upper one in place (bitwise-exact
        symmetry), in ``_DEVICE_MIRROR_BLOCK`` panels: bounded temporaries and
        O(n/block) launches."""
        xp = self.xp
        n = int(out.shape[0])
        blk = _DEVICE_MIRROR_BLOCK
        for j0 in range(0, n, blk):
            j1 = min(n, j0 + blk)
            diag = out[j0:j1, j0:j1]
            out[j0:j1, j0:j1] = xp.tril(diag) + xp.permute_dims(
                xp.tril(diag, k=-1), (1, 0)
            )
            if j1 < n:
                out[j0:j1, j1:] = xp.permute_dims(out[j1:, j0:j1], (1, 0))
        return out


def _device_accumulate(
    ctx: _DeviceGramContext,
    out: Any,
    items: list[tuple[_DeviceStructure, Any]],
    acc_name: str,
) -> None:
    """Add every structure's ``A_sᵀ diag(w_s) A_s`` into ``out``'s lower triangle.

    ``items`` pairs each device structure with its merged weights (device, the
    structure's original row order). The device twin of
    :func:`_accumulate_tiles`: √w-scaled ``syrk``/``matmul`` on tiles whose
    weights are finite and nonnegative, ``Dᵀ(w∘D)`` on the others (Breedveld
    2017 §5.1.2), sparse tiles by cuSPARSE on CuPy for exact requests (and
    over-budget sparse tiles never densified; see
    :attr:`TiledGramConfig.device_densify_budget_bytes`). The per-tile weight
    test is one reduction per structure and one host read per call; exact
    CuPy Grams add the SpGEMM output-size read of each sparse tile (see the
    section comment above).
    """
    import numpy as np

    prepared, flag_dev = _device_prepare(ctx, items, acc_name)
    if flag_dev is None:
        return
    flags = np.asarray(ctx.to_host(flag_dev), dtype=bool)
    _device_run(ctx, out, prepared, flags, acc_name)


def _device_prepare(
    ctx: _DeviceGramContext,
    items: list[tuple[_DeviceStructure, Any]],
    acc_name: str,
) -> tuple[list[tuple[_DeviceStructure, Any, Any]], Any]:
    """Per structure: the RCM-permuted weights, their square roots (in the
    accumulate dtype) and the per-tile "signed / non-finite" flags — all on
    the device, no host read. Returns ``(prepared, flags)``; ``flags`` is
    the device bool vector (one entry per ``block_rows`` rows of every
    structure, in order), or None when ``items`` is empty."""
    xp = ctx.xp
    acc = ctx.dtype(acc_name)
    prepared: list[tuple[_DeviceStructure, Any, Any]] = []
    flag_parts: list[Any] = []
    for ds, w_s in items:
        wp = xp.take(w_s, ds.perm)
        good = xp.isfinite(wp) & (wp >= 0.0)
        swp = xp.astype(xp.sqrt(xp.where(good, wp, xp.zeros_like(wp))), acc)
        nblocks = -(-ds.rows // ds.block_rows)
        bad = ~good
        pad = nblocks * ds.block_rows - ds.rows
        if pad:
            bad = xp.concat((bad, xp.zeros((pad,), dtype=xp.bool, device=ctx.device)))
        flag_parts.append(xp.any(xp.reshape(bad, (nblocks, ds.block_rows)), axis=1))
        prepared.append((ds, wp, swp))
    if not prepared:
        return prepared, None
    return prepared, xp.concat(flag_parts)


def _device_run(
    ctx: _DeviceGramContext,
    out: Any,
    prepared: list[tuple[_DeviceStructure, Any, Any]],
    flags: Any,
    acc_name: str,
) -> None:
    """The tile products of :func:`_device_prepare`'s output, with ``flags``
    already on the host (NumPy bool, same layout)."""
    xp = ctx.xp
    acc = ctx.dtype(acc_name)
    offset = 0
    for ds, wp, swp in prepared:
        for t in ds.tiles:
            bad_tile = bool(flags[offset + t.start // ds.block_rows])
            if t.sparse and (acc_name == "float64" or t.capped):
                # Exact requests (a reduced-precision request densifies
                # sparse tiles: float32 syrk outruns cuSPARSE's float64
                # SpGEMM there; measured on Protons_01, RTX 4070 Laptop) and
                # over-budget tiles, which are never densified.
                _device_sparse_tile_into(ctx, out, t, wp[t.start : t.stop])
                continue
            if t.capped:
                # Over budget with no device SpGEMM (Torch): the host product.
                _host_sparse_tile_into(ctx, out, t, wp[t.start : t.stop])
                continue
            b, c = t.stop - t.start, t.width
            dense = xp.zeros((b * c,), dtype=acc, device=ctx.device)
            vals = ds.data[t.lo : t.hi]
            if vals.dtype != acc:
                vals = xp.astype(vals, acc)
            dense[ds.flat[t.lo : t.hi]] = vals
            dense = xp.reshape(dense, (b, c))
            if not bad_tile:
                # Σ ≥ 0: symmetric rank-k update on √w-scaled rows.
                dense *= swp[t.start : t.stop, None]
                block = ctx.gram_block(dense)
            else:
                # Signed / non-finite weights: no square root, general product.
                w_t = xp.astype(wp[t.start : t.stop], acc)
                block = xp.matmul(xp.permute_dims(dense, (1, 0)), w_t[:, None] * dense)
            ctx.scatter_block(out, t.cols, block)
        offset += -(-ds.rows // ds.block_rows)


def _device_sparse_tile_into(
    ctx: _DeviceGramContext, out: Any, t: _DeviceTile, w_t: Any
) -> None:
    """Sparse tile on CuPy: ``tileᵀ diag(w) tile`` by cuSPARSE SpGEMM, its lower
    triangle scattered at the tile's columns (upper entries are added as zeros
    rather than filtered, which would cost a host sync).

    Host syncs: the SpGEMM reads the product's nnz back to size its output
    (inside cupyx, unavoidable). The scaled copy carries the tile's canonical
    indices, so it is flagged canonical — otherwise cupyx's
    ``sum_duplicates`` would check it with a kernel and a second sync — and
    it is float64 like the stored transpose, so neither operand is recast.
    """
    import cupyx.scipy.sparse as cxs

    xp = ctx.xp
    data = t.csr.data * w_t[t.row_of_nnz]
    if data.dtype != xp.float64:
        data = xp.astype(data, xp.float64)
    scaled = cxs.csr_matrix((data, t.csr.indices, t.csr.indptr), shape=t.csr.shape)
    scaled.has_canonical_format = True
    g = (t.csr_t @ scaled).tocoo()
    vals = xp.where(g.row >= g.col, g.data, xp.zeros_like(g.data))
    ctx.scatter_coo(out, xp.take(t.cols, g.row), xp.take(t.cols, g.col), vals)


def _host_sparse_tile_into(
    ctx: _DeviceGramContext, out: Any, t: _DeviceTile, w_t: Any
) -> None:
    """Over-budget sparse tile without a device SpGEMM (Torch): the host SciPy
    product of the plan's tile (the NumPy kernel's sparse branch), its lower
    triangle scattered onto the device. One weight download (free on CPU
    tensors) and one small upload of the product."""
    import numpy as np
    import scipy.sparse as sp

    tile = t.host
    w = np.asarray(ctx.to_host(w_t), dtype=np.float64)
    data = tile.csr.data * w[tile.row_of_nnz]
    scaled = sp.csr_matrix(
        (data, tile.csr.indices, tile.csr.indptr), shape=tile.csr.shape
    )
    g = (tile.csr_t @ scaled).tocoo()
    keep = g.row >= g.col
    ctx.scatter_coo(
        out,
        ctx.put(tile.cols[g.row[keep]], np.int64),
        ctx.put(tile.cols[g.col[keep]], np.int64),
        ctx.put(g.data[keep], np.float64),
    )


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

    ``tiled_gram`` (default ``True``) declares each precision group of the
    linear block as a :class:`TiledGramCSROperator` when its plan passes the
    gate (NumPy: host BLAS; CuPy / Torch: the device kernel; other namespaces
    keep the plain operator); a :class:`TiledGramConfig` sets the knobs,
    ``False`` keeps the plain ``CSROperator`` for A/B comparisons.

    ``float32_group_min_share`` (default 5%) gates the precision grouping of
    the linear block: its float32-sourced rows form their own hinted group
    (which ``gram_dtype="auto"`` accumulates in float32) only when they hold at
    least this share of the block's nonzeros; below it they fold into the
    float64 group unhinted, so a negligible float32 share does not put every
    dense solve on the refinement route. ``0`` always splits. The fold keeps
    every product exact but changes the lowered row order (one group instead
    of two, so also the multiplier order), summation orders, and the float64
    group's tiled plan and gate statistic, which then include the float32
    structures. :attr:`float32_nnz_share` and :attr:`float32_group_folded`
    report the decision.
    """

    def __init__(
        self,
        instance: TROTSInstance,
        xp: Namespace,
        *,
        sparse: bool = False,
        tiled_gram: bool | TiledGramConfig = True,
        float32_group_min_share: float = 0.05,
    ) -> None:
        import numpy as np

        if not 0.0 <= float(float32_group_min_share) <= 1.0:
            raise ValueError(
                "float32_group_min_share must lie in [0, 1], got "
                f"{float32_group_min_share!r}"
            )
        self._inst = instance
        self.xp = xp
        self._sparse = sparse
        # Kept for the host twin (see _host_twin): the share gate fixes the
        # lowered row order, so the twin must be built with the same value.
        self._float32_group_min_share = float(float32_group_min_share)
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
        # Per assembled block (same order as the rows): which structure matrix
        # it copies and how — the tiled Gram's segment description (see
        # :class:`GramSegment`): (matrix, sign, aux column, aux coefficient).
        lin_meta: list[tuple[TROTSMatrix, float, int | None, float]] = []

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
                    lin_meta.append((mat, 1.0, None, 0.0))
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
            lin_meta.append((mat, sgn, self._n + k, -sgn))
            off = mat.b if mat.b.size == m else np.zeros(m)
            lo.extend([-np.inf] * m)
            hi.extend((-sgn * off).tolist())
        # Deliberately a local, not an attribute: only the precision-grouped copy
        # below is kept, so the host holds the linear block once.
        A_lin = (
            sp.vstack(blocks, format="csr")
            if blocks
            else sp.csr_matrix((0, self._n_vars))
        )
        lin_lo = np.asarray(lo, dtype=float)
        lin_hi = np.asarray(hi, dtype=float)
        row_f32 = (
            np.concatenate(lin_f32)
            if lin_f32
            else np.zeros((A_lin.shape[0],), dtype=bool)
        )
        # Precision-grouping gate: a float32 group holding a negligible share
        # of the block's nonzeros buys no Gram reduction worth having, yet its
        # hint puts every dense solve on the mixed-precision refinement route,
        # whose certificate matvecs run over the *whole* block (Protons_01:
        # 0.07% float32 by nnz, 0.91 vs 0.82 s/iter against gram_dtype
        # "native" at identical iterations). Below the threshold the float32
        # rows fold into the float64 group without a hint. Their values are
        # exact float64 widenings, so every product is unchanged, but the fold
        # is not free of side effects: the block becomes one group, so the
        # lowered row order (and with it the multiplier order in y_ineq) is
        # the single group's, summation orders change, and the float32
        # structures join the float64 group's tiled plan and gate statistic
        # (planned under no hint, which TROTSExactProblem mirrors so the
        # plan registry shares their tiles).
        total_nnz = int(A_lin.nnz)
        f32_nnz = int(np.sum(np.diff(A_lin.indptr)[row_f32])) if total_nnz else 0
        #: Share of the linear block's nonzeros drawn from float32 sources.
        self.float32_nnz_share = f32_nnz / total_nnz if total_nnz else 0.0
        #: Whether the share gate folded the float32-sourced rows (if any,
        #: including an all-empty float32 group) into the float64 group.
        self.float32_group_folded = bool(row_f32.any()) and (
            self.float32_nnz_share < float(float32_group_min_share)
        )
        if self.float32_group_folded:
            row_f32 = np.zeros_like(row_f32)
            lin_f32 = [np.zeros_like(flags) for flags in lin_f32]
        #: Keys (``id`` of the TROTSMatrix) of the linear block's structures.
        self._lin_structure_keys = {id(mat) for mat, _s, _c, _k in lin_meta}
        # Group the two-sided rows by source precision, float32-sourced first.
        # The lin-block values are the source matrices' entries times ±1 (plus
        # the ±1 minimax links), so a row drawn from a matrix that is float32 in
        # the file carries only float32 information even though the assembled
        # block is float64. Each group becomes its own block of the
        # ``linear_ineq`` operator (see :meth:`linear_ineq`); ipax lowers a
        # ``VStack`` block by block, so the float32 group keeps its
        # ``values_dtype_hint`` and the solver's ``gram_dtype="auto"`` default
        # reduces its Gram accumulate while the genuinely-float64 rows stay
        # exact. Radiotherapy plans are routinely mixed: a VMAT plan measures
        # ~96% float32 by nonzero, held back by a single float64 constraint
        # matrix. Row order within a group is the assembly order.
        A_csr = A_lin.tocsr()
        self._lin_groups: list[tuple[Any, str | None]] = []
        # Per group: the tiled-Gram segments (row ranges within the group, in
        # the group's row order — assembly order restricted to the group).
        self._lin_segments: list[list[GramSegment]] = []
        lo_parts: list[Any] = []
        hi_parts: list[Any] = []
        for is_f32 in (True, False):
            group = row_f32 == is_f32
            if bool(group.any()):
                self._lin_groups.append((A_csr[group], "float32" if is_f32 else None))
                lo_parts.append(lin_lo[group])
                hi_parts.append(lin_hi[group])
                segments: list[GramSegment] = []
                offset = 0
                for (mat, sgn, aux_col, aux_coef), flags in zip(
                    lin_meta, lin_f32, strict=True
                ):
                    rows = int(flags.shape[0])
                    if rows == 0 or bool(flags[0]) != is_f32:
                        continue
                    segments.append(
                        GramSegment(
                            offset,
                            offset + rows,
                            mat.matrix,
                            key=id(mat),
                            sign=sgn,
                            aux_col=aux_col,
                            aux_coef=aux_coef,
                        )
                    )
                    offset += rows
                self._lin_segments.append(segments)
        self._lin_lo = np.concatenate(lo_parts) if lo_parts else np.zeros((0,))
        self._lin_hi = np.concatenate(hi_parts) if hi_parts else np.zeros((0,))
        self._n_lin_rows = int(A_lin.shape[0])  # two-sided; one finite side each
        # Tiled per-structure Gram of the linear block (host NumPy namespaces
        # only; see TiledGramCSROperator). ``False`` keeps the plain operator.
        if tiled_gram is True:
            self._tiled_cfg: TiledGramConfig | None = TiledGramConfig()
        elif tiled_gram is False:
            self._tiled_cfg = None
        else:
            self._tiled_cfg = tiled_gram
        self._n_nl_ineq = len(self._nl_con) + len(self._quad_con)
        # Per-structure tile registry shared by the linear block and the exact
        # Hessian (see TiledGramPlan.build). Only TROTSExactProblem enables it:
        # the L-BFGS problem keeps the linear block's plan scoped to its
        # operator, exactly as before.
        self._plan_registry: dict[Any, Any] | None = None
        # Device tiled Gram (CuPy / Torch namespaces): the namespace kind, and
        # the per-problem device tile cache shared by the linear block's
        # operators (rebuilt per linear_ineq() call) and the exact Hessian.
        self._device_kind = None if _is_host_numpy(xp) else _device_kind(xp)
        self._device_tiles: dict[Any, Any] = {}
        if self._device_kind is not None:
            # Device namespaces keep one plan per structure across
            # linear_ineq() calls, so its device tiles are materialized once
            # (the cache is keyed by the plan's tile list).
            self._plan_registry = {}

        # Objective linear coefficient over z (mean terms on x; minimax on t).
        self._c_obj = np.zeros((self._n_vars,), dtype=float)
        self._c_obj[: self._n] = self._lin_obj_coef
        for k, (_mat, w, minimise) in enumerate(self._minimax):
            self._c_obj[self._n + k] = (1.0 if minimise else -1.0) * w

        # Device-side evaluation (CuPy namespaces): keep z and the dose matrices
        # on the device so the per-callback products run on the GPU instead of
        # round-tripping to SciPy. The big constant linear block is not
        # evaluated here at all: it is declared through ``linear_ineq`` and the
        # solver evaluates its lowered operator, which already lives in the
        # solve namespace. The elementwise cost math (`_cost_value` &
        # friends) is written against NumPy's public functions, which dispatch to
        # CuPy via the NEP-18/ufunc protocols, so the same code serves both
        # spaces. Host NumPy/SciPy remains the path for every non-CUDA namespace.
        self._eval_device = False
        self._cxs: Any = None  # cupyx.scipy.sparse, when active
        self._dev_mats: dict[int, Any] = {}  # id(TROTSMatrix) -> device CSR
        self._dev_vecs: dict[int, Any] = {}  # id(TROTSMatrix) -> device b
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

    def initial_point(
        self,
        scale: float = 1.0,
        *,
        warm_start: bool = True,
        strategy: Literal["ipax", "breedveld"] = "ipax",
    ) -> Array:
        """A strictly-positive fluence start with consistent minimax aux values.

        With ``warm_start`` (default) and initialisation data present, the base
        fluence is the dataset's least-squares fluence
        (:meth:`least_squares_fluence`; Breedveld et al. 2017, eq. (23));
        otherwise a uniform ``scale`` map. ``strategy`` then shapes it:

        * ``"ipax"`` (default) applies a tiny positive floor (``1e-6·max``) so
          the log-barrier has a non-degenerate interior start on the ``x ≥ 0``
          bounds, leaving the fluence otherwise untouched.
        * ``"breedveld"`` is the paper's §3.1 primal start: the fluence is
          pulled halfway towards its mean, ``x0 = ½x + ½·mean(x)`` (eq. (24)),
          and every element below 5 is raised to 5 — a well-centred start
          far from the bounds. Pair it with :meth:`breedveld_warm_start` for
          the matching dual start and ``μ0``.

        The minimax aux values are padded consistently with the final fluence
        (:meth:`_pad_aux`). Returned in the problem's namespace.
        """
        import numpy as np

        if strategy not in ("ipax", "breedveld"):
            raise ValueError(
                f"unknown initial_point strategy {strategy!r}; "
                "expected 'ipax' or 'breedveld'"
            )
        has_init = (
            self._inst.init_matrix_ids is not None
            and np.asarray(self._inst.init_matrix_ids).size > 0
        )
        if warm_start and has_init:
            x = self.least_squares_fluence()
            if strategy == "ipax":
                peak = float(np.max(x))
                x = np.maximum(x, 1e-6 * peak if peak > 0.0 else 1e-6)
        else:
            x = np.full((self._n,), scale, dtype=float)
        if strategy == "breedveld":
            # Breedveld et al. 2017, eq. (24), then the §3.1 floor at 5.
            x = 0.5 * x + 0.5 * float(np.mean(x))
            x = np.maximum(x, _BREEDVELD_START_FLOOR)
        return _from_numpy(self.xp, self._pad_aux(x))

    def _host_twin(self) -> TROTSProblem:
        """A host-NumPy copy of this problem for host-side preprocessing.

        Built from the same instance and the same ``sparse`` /
        ``float32_group_min_share`` arguments, so its lowered inequality rows —
        whose order the share gate fixes — match this problem's row for row.
        It is the plain :class:`TROTSProblem` without the tiled Gram: only
        values, gradients and Jacobian products are needed, which neither the
        exact Hessian nor the tile plans change. On a host NumPy namespace the
        problem itself is returned.
        """
        import numpy as np

        if _is_host_numpy(self.xp) and not self._eval_device:
            return self
        return TROTSProblem(
            self._inst,
            np,
            sparse=self._sparse,
            tiled_gram=False,
            float32_group_min_share=self._float32_group_min_share,
        )

    def breedveld_warm_start(
        self,
        x0: Array,
        *,
        scaling: ScalingOptions | ScalingMethod | None = None,
        lsqr: Literal["auto", "host", "device"] = "auto",
    ) -> tuple[WarmStart, float]:
        """Breedveld et al. 2017 §3.1 dual start and ``μ0`` at the primal ``x0``.

        Computed on the problem exactly as the solver sees it —
        ``linear_ineq`` lowered by
        :func:`ipax.problem.linear_ineq.lower_linear_inequalities`, the same
        function :func:`ipax.solve` applies — so the multipliers follow the
        solver's ``y_ineq`` order (nonlinear rows, then the lowered linear
        block). With ``g`` the one-sided inequalities ``g(x) ≤ 0``:

        * ``y0`` solves ``∇g(x0)ᵀ y0 = −∇f(x0)`` in the least-squares sense by
          LSQR (eq. (26); ``atol = btol = 1e-6``, at most 200 iterations), then
          ``y0 ← max(y0, 5)``;
        * ``z0 = max(∇f(x0) + ∇g(x0)ᵀ y0, 5)`` on the bounded (fluence)
          entries (eq. (27)); the free minimax aux entries get ``0``;
        * the slacks ``w0 = max(−g(x0), 5)`` (eq. (28)) — ipax's ``s`` with
          ``g(x) + s = 0``;
        * ``μ0 = (w0ᵀ y0 + z0ᵀ (x0 − x_L)) / (m + n_bounded)``, the average
          complementarity (eq. (10)), for ``BarrierOptions(mu_init=μ0)``.

        ``lsqr`` picks where this runs. ``"host"``: NumPy/SciPy on a host twin
        of the problem (:meth:`_host_twin`), SciPy's LSQR. ``"device"``: in
        the problem's CuPy namespace on the GPU, with :func:`_lsqr` (the same
        Paige–Saunders iteration and stopping rules as SciPy's) on the
        device-resident lowered operator — no host twin, no host copy of the
        linear block. Both give the same start to round-off (the floored
        arrays and ``μ0``; the unfloored LSQR iterates drift apart by LSQR's
        own round-off amplification, well below the floor). ``"auto"``
        (default) is ``"device"`` on CuPy namespaces and ``"host"`` otherwise
        (NumPy, Torch, JAX, ``array_api_strict``).

        The floor of 5 and eq. (10) are in the problem's original units (the
        paper's μ0 has no centering factor σ). ``scaling`` is the solve's
        ``Options.scaling`` (a :class:`~ipax.options.ScalingOptions` or a
        method string; ``None`` means the ipax default, gradient-based):
        :func:`ipax.solve` maps a warm start into its scaled subproblem
        (``s̃ = d·s``, ``ỹ = y·s_f/d``, ``z̃ = s_f·z``), so every
        complementarity pair the driver sees is ``s_f`` times the original,
        while ``mu_init`` is taken as given. The returned ``μ0`` is therefore
        the *solver-space* average complementarity ``s_f·μ0``, with the
        objective factor ``s_f = min(1, max_gradient / ‖∇f(x0)‖∞)`` computed
        exactly as the solver does (Wächter & Biegler 2006, §3.8; ``s_f = 1``
        for ``method="none"``). The warm-start arrays stay in original units.

        Returns ``(WarmStart(s, y_ineq, z_lower), μ0)`` with the arrays in
        ``x0``'s namespace and on its device. Raises ``ValueError`` when the
        problem has neither inequalities nor finite lower bounds (eq. (10) is
        then undefined), or for ``lsqr="device"`` on a non-CuPy namespace.
        """
        import numpy as np

        from ipax.backend.namespace import array_namespace, device
        from ipax.result import WarmStart

        lsq = self._breedveld_least_squares(x0, lsqr)
        ns, y = lsq.ns, lsq.y
        if lsq.jac is None:
            jt_y = ns.zeros_like(lsq.x)
        else:
            y = ns.maximum(y, _BREEDVELD_START_FLOOR)
            jt_y = lsq.vec(lsq.jac.rmatvec(y))
        # eq. (27): bound multipliers from stationarity, on bounded entries.
        bounded = ns.isfinite(lsq.lower)
        z = ns.where(bounded, ns.maximum(lsq.grad + jt_y, _BREEDVELD_START_FLOOR), 0.0)
        # eq. (28): slacks, g(x0) + w0 = 0 pushed inside by the floor.
        w = ns.maximum(-lsq.g, _BREEDVELD_START_FLOOR)
        # eq. (10): the average complementarity.
        gap = ns.where(bounded, lsq.x - ns.where(bounded, lsq.lower, 0.0), 0.0)
        n_bounded = int(ns.count_nonzero(bounded))
        m = int(lsq.g.shape[0])
        denom = m + n_bounded
        if denom == 0:
            raise ValueError(
                "Breedveld start: the problem has no inequalities and no finite "
                "lower bounds, so the average complementarity μ0 (eq. (10)) is "
                "undefined"
            )
        mu0 = (float(w @ y) + float(z @ gap)) / denom
        # Solver-space μ0: ipax.solve scales every complementarity pair by the
        # objective factor s_f (see the docstring).
        mu0 *= _objective_scale_factor(_to_numpy(lsq.grad), scaling)
        # Back to x0's namespace and device (the compat namespace accepts
        # ``device=`` also where the problem was built on a raw module); a
        # no-op for the device path, whose arrays already live there.
        xp = array_namespace(x0)
        dev = device(x0)

        def put(arr: Any) -> Array:
            if lsq.on_device:
                return xp.asarray(arr, dtype=xp.float64, device=dev)
            return xp.asarray(np.asarray(arr, dtype=np.float64), device=dev)

        warm = WarmStart(s=put(w), y_ineq=put(y), z_lower=put(z))
        return warm, mu0

    def _breedveld_least_squares(
        self, x0: Array, lsqr: Literal["auto", "host", "device"] = "auto"
    ) -> _BreedveldLSQ:
        """The pieces of :meth:`breedveld_warm_start` before its floors.

        ``x0``, ``∇f(x0)``, ``g(x0)``, the lowered inequality Jacobian, the
        lower bounds and the *unfloored* LSQR multipliers of eq. (26), all in
        one namespace: NumPy on the host twin (``lsqr="host"``) or the
        problem's CuPy namespace (``lsqr="device"``; see
        :meth:`breedveld_warm_start` for ``"auto"``).
        """
        import numpy as np

        from ipax.backend.namespace import array_namespace
        from ipax.problem.linear_ineq import lower_linear_inequalities

        if lsqr not in ("auto", "host", "device"):
            raise ValueError(
                f"unknown lsqr {lsqr!r}; expected 'auto', 'host' or 'device'"
            )
        if lsqr == "device" and not self._eval_device:
            raise ValueError(
                "Breedveld start: lsqr='device' needs a CuPy namespace "
                "(with cupyx) on the problem"
            )
        on_device = self._eval_device and lsqr != "host"
        if on_device:
            ns: Any = array_namespace(x0)
            x = ns.asarray(x0, dtype=ns.float64)
            problem: TROTSProblem = self
        else:
            ns = np
            x = np.asarray(_to_numpy(x0), dtype=np.float64)
            problem = self._host_twin()

        def as_vec(v: Any) -> Any:
            return _as_float_vec(v, ns, on_device)

        low = lower_linear_inequalities(problem, x, ns)
        grad = as_vec(low.gradient(x))
        try:
            g = as_vec(low.ineq_constraints(x))
            jac = low.ineq_jacobian(x)
        except NotImplementedError:  # no inequalities at all
            g, jac = ns.zeros((0,), dtype=ns.float64), None
        if jac is not None and int(g.shape[0]) == 0:
            jac = None
        lower = as_vec(problem.bounds()[0])

        if jac is None:
            y = ns.zeros((0,), dtype=ns.float64)
        else:
            op = jac
            m, n_vars = int(g.shape[0]), int(x.shape[0])
            if on_device:
                # Breedveld et al. 2017, eq. (26): least-squares multipliers,
                # A = ∇g(x0)ᵀ applied as op.rmatvec, Aᵀ as op.matvec.
                y, istop, itn = _lsqr(
                    lambda v: as_vec(op.rmatvec(v)),
                    lambda u: as_vec(op.matvec(u)),
                    -grad,
                    m,
                    ns,
                    atol=_BREEDVELD_LSQR_TOL,
                    btol=_BREEDVELD_LSQR_TOL,
                    iter_lim=_BREEDVELD_LSQR_ITER,
                )
            else:
                import scipy.sparse.linalg as spla

                jt = spla.LinearOperator(
                    (n_vars, m),
                    matvec=lambda v: as_vec(op.rmatvec(np.ravel(v))),
                    rmatvec=lambda u: as_vec(op.matvec(np.ravel(u))),
                    dtype=np.float64,
                )
                # Breedveld et al. 2017, eq. (26): least-squares multipliers.
                y, istop, itn = spla.lsqr(
                    jt,
                    -grad,
                    atol=_BREEDVELD_LSQR_TOL,
                    btol=_BREEDVELD_LSQR_TOL,
                    iter_lim=_BREEDVELD_LSQR_ITER,
                )[:3]
            if istop == 7:  # SciPy's code for "iteration limit reached"
                _LOG.warning(
                    "Breedveld start: LSQR stopped at its iteration limit (%d) "
                    "before reaching atol = btol = %g; using its last iterate",
                    itn,
                    _BREEDVELD_LSQR_TOL,
                )
        return _BreedveldLSQ(
            ns=ns,
            on_device=on_device,
            x=x,
            grad=grad,
            g=g,
            jac=jac,
            lower=lower,
            y=y,
        )

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

    # -- linear inequalities (constant block, lowered by ipax) ---------------
    def linear_ineq(self) -> tuple[LinearOperator, Array, Array] | None:
        """The constant two-sided block ``lo ≤ A z ≤ hi``.

        The elementwise dose constraints and the minimax links, as an
        Array-API-native :class:`~ipax.backend.operators.CSROperator` — a
        :class:`~ipax.backend.operators.VStack` of the float32-sourced and the
        float64 row groups when the plan mixes them, so each keeps its own
        precision hint — with the original two-sided bounds (every row has
        exactly one finite side). ipax lowers it into one-sided rows, never
        densified — plain groups at the triplet level, the loader's operator
        subclasses as a signed row selection around them — and appends them
        after the nonlinear rows of :meth:`ineq_constraints`; the lowered row
        order is the core's (per group: finite-lower rows, then finite-upper
        rows), not the loader's.

        Built fresh on every call and deliberately not cached: the lowering
        emits an operator of its own, so holding this one would keep a second
        copy of the largest array in the problem (a second device copy on a
        CUDA namespace). Only the host CSR groups are retained.
        """
        if not self._lin_groups:
            return None
        from ipax.backend.operators import VStack

        ops = [
            self._linear_block_operator(csr, hint, segments)
            for (csr, hint), segments in zip(
                self._lin_groups, self._lin_segments, strict=True
            )
        ]
        matrix = ops[0] if len(ops) == 1 else VStack(tuple(ops))
        return (
            matrix,
            _from_numpy(self.xp, self._lin_lo),
            _from_numpy(self.xp, self._lin_hi),
        )

    def _linear_block_operator(
        self, csr: Any, hint: str | None, segments: list[GramSegment]
    ) -> LinearOperator:
        """One precision group of the linear block: tiled Gram when it pays.

        The :class:`TiledGramCSROperator` is used on NumPy (host BLAS) and on
        CuPy / Torch namespaces (the device kernel on the operator's device)
        when its plan passes the flop-ratio gate. A group the gate keeps off
        is an :class:`ActiveRowGramCSROperator` on NumPy (the plain product,
        restoration's active sets from the active rows alone) and the plain
        ``CSROperator`` elsewhere, as is every group on any other namespace
        (JAX, ``array_api_strict``) — which the core lowers at the triplet
        level exactly as before.
        """

        if self._tiled_cfg is not None:
            if _is_host_numpy(self.xp) or self._device_kind is not None:
                plan = TiledGramPlan.build(
                    csr.shape,
                    segments,
                    values_dtype_hint=hint,
                    config=self._tiled_cfg,
                    registry=self._plan_registry,
                )
                if plan.tiled:
                    try:
                        return TiledGramCSROperator(
                            csr,
                            self.xp,
                            segments=segments,
                            values_dtype_hint=hint,
                            plan=plan,
                            device_cache=self._device_tiles,
                        )
                    except ValueError as exc:
                        # The segments disagree with the assembled rows (a
                        # loader bookkeeping bug, e.g. explicit zeros dropped
                        # on one side): the plain operator is always correct.
                        _LOG.warning(
                            "tiled Gram disabled for a linear-block group: %s", exc
                        )
                elif _is_host_numpy(self.xp) and not self._eval_device:
                    # Gated off on the host: the plain product in the
                    # interior-point iterations, the active rows only in
                    # restoration (see ActiveRowGramCSROperator).
                    return ActiveRowGramCSROperator(
                        csr, self.xp, values_dtype_hint=hint, config=self._tiled_cfg
                    )
        return self._csr_operator(csr, symmetric=False, values_dtype_hint=hint)

    # -- nonlinear inequality constraints (smooth cost-function rows) --------
    # Only the curved rows live here; the constant linear block is declared
    # through ``linear_ineq`` and appended after these rows by the solver, so
    # the Lagrangian-Hessian multipliers line up with the rows below.
    def ineq_constraints(self, z: Array) -> Array:
        import numpy as np

        if self._n_nl_ineq == 0:
            raise NotImplementedError  # the linear block alone: see linear_ineq
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
        return _from_numpy(self.xp, np.asarray(vals, dtype=float))

    def ineq_jacobian(self, z: Array) -> Array | LinearOperator:
        import numpy as np
        import scipy.sparse as sp

        if self._n_nl_ineq == 0:
            raise NotImplementedError  # the linear block alone: see linear_ineq
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
        J_x = np.stack(rows)
        J = spx.hstack(
            [spx.csr_matrix(J_x), spx.csr_matrix((len(rows), self._n_aux))],
            format="csr",
        )
        return self._csr_operator(J, symmetric=False)

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


def _objective_scale_factor(
    grad: Any, scaling: ScalingOptions | ScalingMethod | None
) -> float:
    """The objective factor ``s_f`` :func:`ipax.solve` applies at ``x0``.

    Normalizes ``scaling`` like :class:`ipax.Options` does (``None`` → the
    default :class:`~ipax.options.ScalingOptions`, a string → its method) and
    mirrors the core's gradient-based rule (Wächter & Biegler 2006, §3.8):
    ``min(1, max_gradient / ‖∇f(x0)‖∞)``, identity for an empty or small
    gradient and for ``method="none"``.
    """
    import numpy as np

    from ipax.options import ScalingOptions

    if scaling is None:
        opts = ScalingOptions()
    elif isinstance(scaling, str):
        opts = ScalingOptions(method=scaling)
    else:
        opts = scaling
    if opts.method == "none" or int(np.size(grad)) == 0:
        return 1.0
    norm = float(np.max(np.abs(grad)))
    if norm <= opts.max_gradient:
        return 1.0
    return opts.max_gradient / norm


def _as_float_vec(v: Any, ns: Any, on_device: bool) -> Any:
    """``v`` as a 1-D float64 vector: in ``ns`` on the device, else NumPy."""
    import numpy as np

    if on_device:
        return ns.reshape(ns.asarray(v, dtype=ns.float64), (-1,))
    return np.asarray(_to_numpy(v), dtype=np.float64)


@dataclass
class _BreedveldLSQ:
    """:meth:`TROTSProblem._breedveld_least_squares`' result, in one namespace.

    ``y`` is the unfloored LSQR solution of eq. (26) (empty without
    inequalities, when ``jac`` is ``None``)."""

    ns: Any
    on_device: bool
    x: Any
    grad: Any
    g: Any
    jac: Any
    lower: Any
    y: Any

    def vec(self, v: Any) -> Any:
        return _as_float_vec(v, self.ns, self.on_device)


def _ieee_div(a: float, b: float) -> float:
    """``a / b`` with IEEE-754 semantics on a zero divisor.

    SciPy's ``lsqr`` runs its recurrence on NumPy scalars, where ``x / 0``
    gives ``±inf`` (``nan`` for ``0 / 0``) and the stopping tests then act on
    it; a Python-float division raises ``ZeroDivisionError`` instead. Used for
    the recurrence divisions of :func:`_lsqr` so the transcription degrades
    the way SciPy does rather than crashing.
    """
    import math

    if b != 0.0:
        return a / b
    if a == 0.0 or math.isnan(a):
        return math.nan
    return math.copysign(math.inf, a) * math.copysign(1.0, b)


def _sym_ortho(a: float, b: float) -> tuple[float, float, float]:
    """Stable Givens rotation ``(c, s, r)`` with ``[c s; -s c]ᵀ[a; b] = [r; 0]``.

    Choi's SymOrtho (S.-C. Choi 2006, PhD thesis, Stanford), as in SciPy's
    ``scipy.sparse.linalg.lsqr``; host scalars."""
    import math

    if b == 0.0:
        return math.copysign(1.0, a) if a != 0.0 else 0.0, 0.0, abs(a)
    if a == 0.0:
        return 0.0, math.copysign(1.0, b), abs(b)
    if abs(b) > abs(a):
        tau = a / b
        s = math.copysign(1.0, b) / math.sqrt(1.0 + tau * tau)
        c = s * tau
        return c, s, b / s
    tau = b / a
    c = math.copysign(1.0, a) / math.sqrt(1.0 + tau * tau)
    s = c * tau
    return c, s, a / c


def _lsqr(
    matvec: Any,
    rmatvec: Any,
    b: Any,
    n: int,
    xp: Any,
    *,
    atol: float,
    btol: float,
    iter_lim: int,
    conlim: float = 1e8,
) -> tuple[Any, int, int]:
    """LSQR for ``min ‖A x − b‖₂`` in the namespace of ``b`` (e.g. on a GPU).

    Paige & Saunders (1982), "LSQR: An algorithm for sparse linear equations
    and sparse least squares", ACM TOMS 8(1), 43–71 — transcribed from SciPy's
    ``scipy.sparse.linalg.lsqr`` (no damping, ``x0 = 0``) with the same
    stopping rules and ``istop`` codes: a device run follows the host SciPy
    run to round-off, up to the round-off amplification that is LSQR's own
    (Golub–Kahan loses orthogonality; on TROTS Protons_01 a 1e-16 relative
    perturbation of SciPy's matvec moves its final iterate by ~1e-4).
    ``matvec(v) = A v`` (length ``len(b)``), ``rmatvec(u) = Aᵀ u`` (length
    ``n``). The vectors stay in ``xp``; the recurrence scalars are host floats
    (three norms per iteration sync).

    cupyx's ``lsqr`` is not this algorithm — it is a direct sparse-QR solve
    of a *square* CSR system (cuSOLVER ``csrlsvqr``) — so it cannot solve the
    rectangular eq. (26) through an operator.

    Returns ``(x, istop, itn)`` (``istop = 7``: iteration limit reached).
    """
    import math

    eps = float(xp.finfo(xp.float64).eps)
    ctol = 1.0 / conlim if conlim > 0 else 0.0
    anorm = acond = ddnorm = res2 = xnorm = xxnorm = z = 0.0
    cs2, sn2 = -1.0, 0.0
    itn = istop = 0

    def norm(v: Any) -> float:
        return float(xp.linalg.vector_norm(v))

    u = b
    bnorm = norm(b)
    x = xp.zeros((n,), dtype=b.dtype)
    beta = bnorm
    if beta > 0:
        u = (1.0 / beta) * u
        v = rmatvec(u)
        alfa = norm(v)
    else:
        v = xp.zeros_like(x)
        alfa = 0.0
    if alfa > 0:
        v = (1.0 / alfa) * v
    w = v
    rhobar, phibar = alfa, beta
    if alfa * beta == 0:
        return x, istop, itn

    while itn < iter_lim:
        itn += 1
        # Bidiagonalization: beta·u = A v − alfa·u, alfa·v = Aᵀ u − beta·v.
        u = matvec(v) - alfa * u
        beta = norm(u)
        if beta > 0:
            u = (1.0 / beta) * u
            anorm = math.sqrt(anorm**2 + alfa**2 + beta**2)
            v = rmatvec(u) - beta * v
            alfa = norm(v)
            if alfa > 0:
                v = (1.0 / alfa) * v
        # Plane rotation eliminating the subdiagonal beta.
        cs, sn, rho = _sym_ortho(rhobar, beta)
        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi
        # Update x and w; ‖dk‖ = ‖w‖/|rho| with dk = w/rho.
        # IEEE division (SciPy's NumPy scalars give inf/nan at rho = 0).
        t1 = _ieee_div(phi, rho)
        t2 = _ieee_div(-theta, rho)
        ddnorm += _ieee_div(norm(w), abs(rho)) ** 2
        x = x + t1 * w
        w = v + t2 * w
        # Plane rotation on the right; estimate ‖x‖.
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = _ieee_div(rhs, gambar)
        xnorm = math.sqrt(xxnorm + zbar**2)
        gamma = math.sqrt(gambar**2 + theta**2)
        cs2 = _ieee_div(gambar, gamma)
        sn2 = _ieee_div(theta, gamma)
        z = _ieee_div(rhs, gamma)
        xxnorm += z**2
        # Convergence tests (SciPy's order: the last satisfied test wins).
        acond = anorm * math.sqrt(ddnorm)
        rnorm = math.sqrt(phibar**2 + res2)
        arnorm = alfa * abs(tau)
        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm + eps)
        test3 = 1.0 / (acond + eps)
        t1 = test1 / (1.0 + anorm * xnorm / bnorm)
        rtol = btol + atol * anorm * xnorm / bnorm
        if itn >= iter_lim:
            istop = 7
        if 1.0 + test3 <= 1.0:
            istop = 6
        if 1.0 + test2 <= 1.0:
            istop = 5
        if 1.0 + t1 <= 1.0:
            istop = 4
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1
        if istop != 0:
            break
    return x, istop, itn


def _is_host_numpy(xp: Namespace) -> bool:
    """Whether ``xp`` produces NumPy ndarrays (the host-BLAS kernels' domain).

    Torch (CPU or CUDA), CuPy, JAX and ``array_api_strict`` are not: their
    solves get operators whose data already lives in the namespace."""
    import numpy as np

    try:
        return isinstance(xp.asarray(0.0), np.ndarray)
    except Exception:  # namespace probing only
        return False


def _to_numpy_nd(x: Array) -> Any:
    """Array-API array → NumPy array of the same shape (host bridge)."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "__cuda_array_interface__") and hasattr(x, "get"):
        return np.asarray(x.get())
    try:
        return np.from_dlpack(x)
    except (TypeError, RuntimeError, BufferError, ValueError):
        return np.asarray(x)


@dataclass
class _CurvedStructure:
    """One structure matrix carrying nonlinear (curved) cost terms.

    ``perm``/``tiles`` hold its tiled-Gram plan when the per-structure gate
    passes; otherwise ``csr`` (a CSR view of the matrix, shared with the
    caller when it already is CSR) serves full-width dense row chunks on the
    host. On a device namespace a gated-off structure is planned anyway
    (``device_perm``/``device_tiles``): see :class:`TROTSDeviceExactHessian`.
    """

    mat: TROTSMatrix
    perm: Any = None
    tiles: list[_Tile] | None = None
    csr: Any = None
    flop_ratio: float = 1.0
    #: Device namespaces only: the plan of a structure the gate keeps off
    #: (the device kernel runs it like any tiled structure).
    device_perm: Any = None
    device_tiles: list[_Tile] | None = None
    #: ``(A∘A)ᵀ`` (host CSR), built on the first :meth:`TROTSExactHessian.diagonal`.
    sq_t: Any = None
    #: Host gated-off structures: the full-width row chunks of ``csr``,
    #: built on the first dense Hessian (see :func:`_host_chunks`).
    host_chunks: list[_Tile] | None = None


def _host_chunks(cs: _CurvedStructure, nx: int, cfg: TiledGramConfig) -> list[_Tile]:
    """The full-width ``block_rows`` row chunks of a gated-off structure.

    Constant for the problem's lifetime, so they are cut once and kept on
    ``cs``: zero-copy CSR windows onto ``cs.csr`` (views of its data and
    indices; only the ``block_rows + 1`` row pointers are new), where slicing
    ``csr[start:stop]`` on every dense Hessian copied each chunk — ~0.09 s of
    the 0.33 s host Hessian on Prostate_VMAT_101 (two gated-off structures,
    18M entries). The densified values are those of the slices, so the
    product is unchanged bit for bit.
    """
    import numpy as np
    import scipy.sparse as sp

    if cs.host_chunks is None:
        a = cs.csr
        cols = np.arange(nx, dtype=np.int64)
        chunks: list[_Tile] = []
        for start in range(0, a.shape[0], cfg.block_rows):
            stop = min(a.shape[0], start + cfg.block_rows)
            lo, hi = int(a.indptr[start]), int(a.indptr[stop])
            if hi == lo:
                continue
            # Assigned, not passed to the constructor (which copies the
            # arrays): the window views the structure's own data/indices.
            window = sp.csr_matrix((stop - start, a.shape[1]), dtype=a.dtype)
            window.data = a.data[lo:hi]
            window.indices = a.indices[lo:hi]
            window.indptr = a.indptr[start : stop + 1] - lo
            chunks.append(
                _Tile(
                    start,
                    stop,
                    cols,
                    window,
                    False,
                    flops=float(stop - start) * nx * nx,
                )
            )
        cs.host_chunks = chunks
    return cs.host_chunks


class TROTSExactHessian(LinearOperator):
    """The exact Lagrangian Hessian of a TROTS problem, grouped per structure.

    Every TROTS cost function is, in dose space, a diagonal plus at most a
    rank-1 term; in beamlet space ``∇²h = c1·Aᵀdiag(e2)A + c2·(Aᵀe1)(Aᵀe1)ᵀ``
    (van Haveren & Breedveld 2019, Phys. Med. Biol. 64:16NT01, eqs. (8)–(9);
    corrigendum 65:079501). The diagonals of all terms on one structure add,
    so the curvature is one product per unique structure (Breedveld et al.
    2017, eq. (48)):

        H_x = Σ_k A_kᵀ diag(w_k) A_k + Q + U Vᵀ,

    with ``w_k`` the multiplier-weighted sum of the structure's dose-space
    diagonals (signed in general: DVH curvature is indefinite, maximised
    terms flip sign), ``Q`` the beamlet-space quadratic terms and ``U Vᵀ``
    the gEUD rank-1 terms (a :class:`~ipax.backend.operators.LowRank`). The
    minimax auxiliary variables ``t`` get zero rows and columns.

    :meth:`matvec` is the exact float64 product (what Krylov and the
    mixed-precision refinement apply); :meth:`dense_matrix` forms the tiled
    products (:class:`TiledGramCSROperator`'s kernels, signed-weight gemm
    where needed; full-width dense row chunks for structures the gate keeps
    off), is exactly symmetric, and is memoized on the operator (one
    operator per IPM iteration). :meth:`to_coo` serves the sparse-direct
    route through the plain SciPy product. Host-evaluated (NumPy/SciPy);
    results cross to the solve namespace — which is why
    :class:`TROTSExactProblem` hands this operator to the solver on NumPy
    namespaces only (elsewhere it crosses :meth:`host_sparse` once).

    ``low_rank_factors`` is ``(U, V)`` (host NumPy, ``nx×r``) or None; the
    operator wraps them in a :class:`~ipax.backend.operators.LowRank`
    (:attr:`low_rank`) and keeps the factors (:attr:`low_rank_factors`).
    """

    def __init__(
        self,
        xp: Namespace,
        n_vars: int,
        nx: int,
        terms: list[tuple[_CurvedStructure, Any]],
        quad: Any,
        low_rank_factors: tuple[Any, Any] | None,
        config: TiledGramConfig,
        stats: dict[str, float] | None = None,
        quad_builder: Callable[[], Any] | None = None,
    ) -> None:
        from ipax.backend.operators import LowRank

        self._xp_ns = xp
        self._n_vars = int(n_vars)
        self._nx = int(nx)
        #: ``(structure, w_k)`` per curved structure (dose-space weights).
        self.terms = terms
        self._quad = quad
        # Forms ``quad`` on first use instead (the device operator, whose
        # dense and CuPy matvec paths never read the host Q).
        self._quad_builder = quad_builder
        #: gEUD rank-1 factors ``(U, V)`` (host NumPy) or None.
        self.low_rank_factors = low_rank_factors
        #: gEUD rank-1 part ``U Vᵀ`` (:class:`LowRank` over host NumPy) or None.
        self.low_rank = (
            LowRank(low_rank_factors[0], low_rank_factors[1])
            if low_rank_factors is not None
            else None
        )
        self.config = config
        self._stats = stats
        self._dense: Any = None

    @property
    def quad(self) -> Any:
        """Beamlet-space quadratic part ``Q`` (host SciPy, ``nx×nx``) or None."""
        if self._quad_builder is not None:
            self._quad, self._quad_builder = self._quad_builder(), None
        return self._quad

    @property
    def shape(self) -> tuple[int, int]:
        return self._n_vars, self._n_vars

    def symmetry_hint(self) -> bool | None:
        return True

    # -- products ------------------------------------------------------------
    def _apply_host(self, v: Any) -> Any:
        import numpy as np

        v = np.asarray(v, dtype=np.float64)
        vx = v[: self._nx]
        out = np.zeros_like(v)
        ox = out[: self._nx]
        for cs, w in self.terms:
            a = cs.mat.matrix
            av = a @ vx
            ox += a.T @ (w.reshape((-1,) + (1,) * (av.ndim - 1)) * av)
        if self.quad is not None:
            ox += self.quad @ vx
        if self.low_rank is not None:
            ox += (
                self.low_rank.matvec(vx) if vx.ndim == 1 else (self.low_rank.matmat(vx))
            )
        return out

    def matvec(self, v: Array) -> Array:
        return self._xp_ns.asarray(self._apply_host(_to_numpy_nd(v)))

    def rmatvec(self, v: Array) -> Array:
        return self.matvec(v)  # symmetric

    def matmat(self, V: Array) -> Array:
        return self.matvec(V)

    def rmatmat(self, V: Array) -> Array:
        return self.matvec(V)

    # -- explicit forms -------------------------------------------------------
    def dense_matrix(self, like: Array | None = None) -> Array:
        del like
        if self._dense is None:
            import time

            t0 = time.perf_counter()
            self._dense = self._dense_host()
            if self._stats is not None:
                self._stats["dense_seconds"] += time.perf_counter() - t0
                self._stats["dense_calls"] += 1
        return self._xp_ns.asarray(self._dense)

    def _dense_host(self) -> Any:
        import numpy as np
        import scipy.sparse as sp

        nx = self._nx
        cfg = self.config
        out = np.zeros((self._n_vars, self._n_vars), dtype=np.float64)
        work: list[tuple[_Tile, Any, Any, Any]] = []
        for cs, w in self.terms:
            if cs.tiles is not None:
                work.extend(_tile_work(w, cs.perm, cs.tiles, np.float64))
                continue
            # Gated-off (dense-ish) structure: full-width dense row chunks —
            # the plain adapter's product, on the same lanes and kernels.
            work.extend(_tile_work(w, None, _host_chunks(cs, nx, cfg), np.float64))
        acc = out[:nx, :nx]
        _accumulate_tiles(acc, work, cfg, np.float64)
        if self.quad is not None:
            q = sp.coo_matrix(self.quad)
            np.add.at(acc, (q.row, q.col), q.data)
        if self.low_rank is not None:
            acc += np.asarray(self.low_rank.dense_matrix())
        return _mirror_lower(out)  # lower triangle mirrored: exactly symmetric

    def diagonal(self, like: Array | None = None) -> Array:
        """``Σ_k (A_k∘A_k)ᵀ w_k + diag(Q) + rowsum(U∘V)`` — no matrix formed."""
        del like
        return self._xp_ns.asarray(self._diagonal_host())

    def _diagonal_host(self) -> Any:
        """The diagonal as a host NumPy vector (see :meth:`diagonal`)."""
        import numpy as np

        d = np.zeros((self._n_vars,), dtype=np.float64)
        dx = d[: self._nx]
        for cs, w in self.terms:
            if cs.sq_t is None:  # constant per structure: built once, kept
                a = cs.mat.matrix
                cs.sq_t = a.multiply(a).T.tocsr()
            dx += np.asarray(cs.sq_t @ w).reshape(-1)
        if self.quad is not None:
            dx += self.quad.diagonal()
        if self.low_rank_factors is not None:
            u, v = self.low_rank_factors
            dx += np.sum(u * v, axis=1)
        return d

    def host_sparse(self) -> Any:
        """``H`` over ``z`` as host SciPy CSR via the plain SciPy product (the
        sparse-direct route; the old per-term path's products, grouped)."""
        import numpy as np
        import scipy.sparse as sp

        nx = self._nx
        h = sp.csr_matrix((nx, nx))
        for cs, w in self.terms:
            a = sp.csr_matrix(cs.mat.matrix)
            h = h + (a.T @ sp.diags(w) @ a).tocsr()
        if self.quad is not None:
            h = h + self.quad
        if self.low_rank is not None:
            h = h + sp.csr_matrix(np.asarray(self.low_rank.dense_matrix()))
        n_aux = self._n_vars - nx
        return sp.bmat([[h, None], [None, sp.csr_matrix((n_aux, n_aux))]], format="csr")

    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        import numpy as np

        del like
        coo = self.host_sparse().tocoo()
        xp = self._xp_ns
        return (
            xp.asarray(np.asarray(coo.row, dtype=np.int64)),
            xp.asarray(np.asarray(coo.col, dtype=np.int64)),
            _from_numpy(xp, coo.data),
            self.shape,
        )


class _DeviceQuadratic:
    """The beamlet-space quadratic part ``Q = Σ_t mult_t·Q_t`` of the exact
    Hessian on a device namespace, with no per-call host product or crossing.

    Only the multipliers of the (symmetrised, constant) quadratic terms change
    between calls, so the pattern of the sum is planned once on the host: the
    union of the terms' patterns in CSR order and, per union entry, its
    contributions stacked in ``K`` zero-padded layers (``K`` = the largest
    number of terms sharing one entry). A call then uploads the multipliers
    (one small transfer) and forms ``Σ_j val[j]·mult[term[j]]`` on the device;
    the dense route scatters the lower triangle at unique positions, so no
    index collides and the result is deterministic. The device copies are
    made once per device-context signature.

    The layers store ``K·U`` values for ``T`` contributions (``U`` union
    entries); :attr:`fill_ratio` is ``K·U / T`` (1 when no entry is shared
    by more terms than the others). The TROTS smoothing terms share little:
    ``Prostate_CK_01`` (25 terms) builds a small plan, and the stat is logged
    at debug level. A ``U × n_terms`` CSR matrix applied to the multipliers
    (one SpMV) would store exactly ``T`` values, at the price of a
    library-defined summation order; the layered sum keeps a fixed order.
    """

    def __init__(self, terms: Sequence[Any], n: int) -> None:
        import numpy as np
        import scipy.sparse as sp

        rows, cols, vals, owner = [], [], [], []
        for t, q in enumerate(terms):
            c = sp.coo_matrix(q)
            c.sum_duplicates()
            rows.append(np.asarray(c.row, dtype=np.int64))
            cols.append(np.asarray(c.col, dtype=np.int64))
            vals.append(np.asarray(c.data, dtype=np.float64))
            owner.append(np.full(c.nnz, t, dtype=np.int64))
        # (At least one term: the problem builds a plan only when it has one.)
        key = np.concatenate(rows) * n + np.concatenate(cols)
        val, term = np.concatenate(vals), np.concatenate(owner)
        order = np.lexsort((term, key))  # by entry, then by term (sum order)
        key, val, term = key[order], val[order], term[order]
        uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
        rank = np.arange(key.shape[0]) - first[inv]
        self.n = int(n)
        self.layers = int(rank.max()) + 1 if rank.size else 0
        self.val = np.zeros((self.layers, uniq.shape[0]))
        self.term = np.zeros((self.layers, uniq.shape[0]), dtype=np.int64)
        self.val[rank, inv] = val
        self.term[rank, inv] = term
        #: The union pattern (CSR order: sorted by row, then column).
        self.row = uniq // n
        self.col = uniq % n
        self.indptr = np.concatenate(
            ([0], np.cumsum(np.bincount(self.row, minlength=n)))
        ).astype(np.int64)
        #: Union positions of the lower triangle (the mirror source).
        self.lower = np.flatnonzero(self.row >= self.col)
        #: Stored layer values per contribution (``K·U / T``, see above).
        self.fill_ratio = self.val.size / max(1, int(key.shape[0]))
        _LOG.debug(
            "device Q plan: %d terms, %d contributions, %d union entries, "
            "%d layers (fill ratio %.2f)",
            len(terms),
            int(key.shape[0]),
            int(uniq.shape[0]),
            self.layers,
            self.fill_ratio,
        )
        self._device: dict[Any, dict[str, Any]] = {}

    def on_device(self, ctx: _DeviceGramContext) -> dict[str, Any]:
        """The device copies for ``ctx``'s namespace/device (made once)."""
        import numpy as np

        hit = self._device.get(ctx.signature)
        if hit is None:
            hit = {
                "val": ctx.put(self.val, np.float64),
                "term": ctx.put(self.term.reshape(-1), np.int64),
                "indices": ctx.put(self.col, np.int32),
                "indptr": ctx.put(self.indptr, np.int32),
                "lower": ctx.put(self.lower, np.int64),
                "low_rows": ctx.put(self.row[self.lower], np.int64),
                "low_cols": ctx.put(self.col[self.lower], np.int64),
            }
            self._device[ctx.signature] = hit
        return hit

    def values(self, ctx: _DeviceGramContext, mults: Any) -> Any:
        """``Q``'s values on the union pattern for the host multipliers
        ``mults`` (one upload of ``n_terms`` floats)."""
        import numpy as np

        xp = ctx.xp
        dev = self.on_device(ctx)
        nnz = int(self.row.shape[0])
        m = ctx.put(np.asarray(mults, dtype=np.float64))
        w = xp.reshape(xp.take(m, dev["term"]), (self.layers, nnz))
        return xp.sum(dev["val"] * w, axis=0)


class TROTSDeviceExactHessian(TROTSExactHessian):
    """:class:`TROTSExactHessian` with its products on a CuPy / Torch device.

    What :class:`TROTSExactProblem` returns on those namespaces when tiling
    is on (``tiled_gram`` not ``False``) and the Hessian has a curved
    structure or a quadratic term (otherwise the host Hessian is crossed as
    a ``CSCOperator``, as before):

    * :meth:`dense_matrix` runs every curved structure through the device
      tiled kernel (:func:`_device_accumulate`, sharing the linear block's
      device tiles through the problem's cache). A structure the flop-ratio
      gate keeps off runs through its own (ungated) plan, made on the first
      dense Hessian by ``device_planner`` (so matvec-only and sparse-direct
      routes never plan it): on the host the alternative is BLAS on
      full-width dense row chunks, but here it was a host SciPy SpGEMM plus
      a crossing of its product (``Prostate_CK_01``: 36 s per iteration),
      and the tiled plan never costs more flops than full-width chunks.
      ``Q`` is formed on the device from its constant union pattern
      (:class:`_DeviceQuadratic`); ``U Vᵀ`` is a device matmul; one triangle
      is mirrored (exactly symmetric). Memoized like the host operator. No
      per-call transfer is larger than a weight vector or the ``n×r`` gEUD
      factors.
    * :meth:`matvec` is the exact float64 product. With device-resident
      structure matrices (CuPy evaluation, ``sparse_module`` = ``cupyx``) it
      runs on the device; otherwise (Torch) it is a host round trip per
      product — download, SciPy products, upload onto the context's device.
    * :meth:`diagonal` and :meth:`to_coo` are the host forms, uploaded once.

    Torch is validated on CPU tensors only; Torch-CUDA is unsupported and
    untested (the problem callbacks bridge through NumPy).
    """

    def __init__(
        self,
        *args: Any,
        context: _DeviceGramContext,
        device_matrix: Any = None,
        sparse_module: Any = None,
        quad_plan: _DeviceQuadratic | None = None,
        quad_mults: Any = None,
        device_planner: Callable[[_CurvedStructure], tuple[Any, list[_Tile]]]
        | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._ctx = context
        self._device_planner = device_planner
        self._device_matrix = device_matrix
        self._sparse_module = sparse_module
        self._quad_plan = quad_plan
        self._quad_mults = quad_mults
        self._quad_values: Any = None  # Q on the union pattern (device)
        self._dev_weights: list[Any] | None = None  # w_k per term (device)
        self._dev_low_rank: Any = None  # (U, V) on the device
        self._dev_products: Any = None  # (terms, Q, U, V) on the device

    # Device copies of the per-call data, uploaded once per operator and
    # shared by dense_matrix and matvec.
    def _device_quad_values(self) -> Any:
        if self._quad_values is None and self._quad_plan is not None:
            self._quad_values = self._quad_plan.values(self._ctx, self._quad_mults)
        return self._quad_values

    def _device_weights(self) -> list[Any]:
        if self._dev_weights is None:
            import numpy as np

            self._dev_weights = [
                self._ctx.put(np.asarray(w, np.float64)) for _cs, w in self.terms
            ]
        return self._dev_weights

    def _device_low_rank(self) -> Any:
        if self._dev_low_rank is None and self.low_rank_factors is not None:
            import numpy as np

            u, v = self.low_rank_factors
            self._dev_low_rank = (
                self._ctx.put(np.asarray(u, np.float64)),
                self._ctx.put(np.asarray(v, np.float64)),
            )
        return self._dev_low_rank

    # -- products ------------------------------------------------------------
    def _device_factors(self) -> Any:
        if self._dev_products is None:
            ctx = self._ctx
            terms = [
                (self._device_matrix(cs.mat), w)
                for (cs, _w), w in zip(self.terms, self._device_weights(), strict=True)
            ]
            quad = None
            plan = self._quad_plan
            if plan is not None:
                dev = plan.on_device(ctx)
                quad = self._sparse_module.csr_matrix(
                    (self._device_quad_values(), dev["indices"], dev["indptr"]),
                    shape=(plan.n, plan.n),
                )
                quad.has_canonical_format = True  # CSR order, duplicate-free
            u, v = self._device_low_rank() or (None, None)
            self._dev_products = (terms, quad, u, v)
        return self._dev_products

    def matvec(self, v: Array) -> Array:
        ctx = self._ctx
        xp = ctx.xp
        if self._device_matrix is None or self._sparse_module is None:
            # No device-resident structure matrices (Torch): a host round
            # trip — download (``to_host`` handles CPU and CUDA tensors),
            # SciPy products, upload onto the context's device.
            return ctx.put(self._apply_host(ctx.to_host(v)))
        v = xp.asarray(v, device=ctx.device)
        nx = self._nx
        vx = v[:nx]
        terms, quad, u, vv = self._device_factors()
        ox = xp.zeros_like(vx)
        for a, w in terms:
            av = a @ vx
            ox = ox + a.T @ ((w[:, None] if av.ndim == 2 else w) * av)
        if quad is not None:
            ox = ox + quad @ vx
        if u is not None:
            ox = ox + xp.matmul(u, xp.matmul(xp.permute_dims(vv, (1, 0)), vx))
        if v.shape[0] == nx:
            return ox
        tail = xp.zeros_like(v[nx:])
        return xp.concat((ox, tail), axis=0)

    # -- explicit forms -------------------------------------------------------
    def dense_matrix(self, like: Array | None = None) -> Array:
        del like
        if self._dense is None:
            import time

            t0 = time.perf_counter()
            self._dense = self._dense_device()
            if self._stats is not None:
                self._stats["dense_seconds"] += time.perf_counter() - t0
                self._stats["dense_calls"] += 1
        return self._dense

    def _dense_device(self) -> Any:
        ctx = self._ctx
        xp = ctx.xp
        nx = self._nx
        out = xp.zeros(
            (self._n_vars, self._n_vars), dtype=xp.float64, device=ctx.device
        )
        items: list[tuple[_DeviceStructure, Any]] = []
        for (cs, _w), w in zip(self.terms, self._device_weights(), strict=True):
            if cs.tiles is not None:
                perm, tiles = cs.perm, cs.tiles
            else:  # gated off: its device plan, made on first use
                assert self._device_planner is not None, cs.mat.name
                perm, tiles = self._device_planner(cs)
            if not tiles:
                continue  # an all-zero structure contributes nothing
            ds = ctx.structure(
                perm, tiles, cs.mat.matrix.shape[0], self.config.block_rows
            )
            items.append((ds, w))
        _device_accumulate(ctx, out, items, "float64")
        if self._quad_plan is not None:
            dev = self._quad_plan.on_device(ctx)
            ctx.scatter_coo(  # unique lower-triangle positions: no collisions
                out,
                dev["low_rows"],
                dev["low_cols"],
                xp.take(self._device_quad_values(), dev["lower"]),
            )
        low_rank = self._device_low_rank()
        if low_rank is not None:
            u, v = low_rank
            out[:nx, :nx] += xp.matmul(u, xp.permute_dims(v, (1, 0)))
        return ctx.mirror_lower(out)

    def diagonal(self, like: Array | None = None) -> Array:
        del like
        return self._ctx.put(self._diagonal_host())  # host form, one upload

    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        import numpy as np

        del like
        coo = self.host_sparse().tocoo()
        ctx = self._ctx
        return (
            ctx.put(coo.row, np.int64),
            ctx.put(coo.col, np.int64),
            ctx.put(coo.data, np.float64),
            self.shape,
        )


class TROTSExactProblem(TROTSProblem):
    """TROTS problem that also supplies the exact Lagrangian Hessian.

    Only the quadratic and gEUD/LTCP/DVH terms have curvature; the linear and
    minimax pieces contribute none. By default (``grouped_hessian=True``) the
    ``x``-block Hessian ``σ·∇²f + Σ y·∇²g`` is a :class:`TROTSExactHessian`:
    one (tiled) product per unique curved structure with the dose-space
    diagonals of all its terms summed (Breedveld et al. 2017, eq. (48); van
    Haveren & Breedveld 2019, eqs. (8)–(9)), the gEUD rank-1 terms as a
    low-rank factor and the beamlet-space quadratics as a constant-pattern
    sparse sum. Structure plans are built once, at construction, and shared
    with the linear block through the problem's plan registry.

    The grouped operator is host-resident (its products run in NumPy/SciPy),
    so it is returned as-is only on NumPy namespaces. On CuPy and Torch the
    curved structures are planned too (a structure the gate keeps off gets a
    device plan anyway) and a :class:`TROTSDeviceExactHessian` runs their
    products through the device tiled kernel (sharing the linear block's
    device tiles), with the quadratic terms summed on the device over their
    constant pattern. On any other namespace with an ipax sparse adapter
    (JAX), with ``tiled_gram=False``, or when the Hessian has no curved or
    quadratic term at all, the grouped Hessian is formed once per call on
    the host (:meth:`TROTSExactHessian.host_sparse`: one SciPy SpGEMM per
    structure) and crossed to the namespace as a
    :class:`~ipax.backend.operators.CSCOperator`, like the per-term path —
    so a Krylov matvec or a dense-route ``dense_matrix`` runs on the
    namespace's own arrays (device-resident on a GPU) instead of
    round-tripping to the host on every product. A namespace without a
    sparse adapter (``array_api_strict``) cannot apply a ``CSCOperator`` at
    all, so it keeps the host operator, whose products cross per call.

    ``grouped_hessian=False`` keeps the original per-term path — one SciPy
    SpGEMM ``Aᵀ diag(h) A`` per term, returned padded with zero ``t``
    rows/columns as a :class:`~ipax.backend.operators.CSCOperator` — for
    parity tests and A/B runs.

    :attr:`hessian_stats` counts wall seconds and calls. ``build_seconds`` is
    the time inside :meth:`lagrangian_hessian`, which covers different work
    per path: the per-term path and the non-NumPy grouped path form the whole
    sparse Hessian there, whereas the NumPy grouped path only forms the
    weights and the operator; its products are timed separately in
    ``dense_seconds`` (the memoized :meth:`TROTSExactHessian.dense_matrix`;
    Krylov matvecs are not timed). Compare paths on
    ``build_seconds + dense_seconds``.
    """

    def __init__(
        self,
        instance: TROTSInstance,
        xp: Namespace,
        *,
        sparse: bool = False,
        tiled_gram: bool | TiledGramConfig = True,
        grouped_hessian: bool = True,
        float32_group_min_share: float = 0.05,
    ) -> None:
        import scipy.sparse as sp

        super().__init__(
            instance,
            xp,
            sparse=sparse,
            tiled_gram=tiled_gram,
            float32_group_min_share=float32_group_min_share,
        )
        self._grouped_hessian = bool(grouped_hessian)
        #: Wall seconds / call counts of the Hessian (observability).
        self.hessian_stats: dict[str, float] = {
            "build_seconds": 0.0,
            "build_calls": 0,
            "dense_seconds": 0.0,
            "dense_calls": 0,
        }
        if not self._grouped_hessian:
            return
        # The host operator on NumPy namespaces, and on namespaces with no
        # sparse adapter to cross to (see the class docstring).
        from ipax.backend.sparse import get_sparse_adapter

        self._hess_host = not self._eval_device and (
            _is_host_numpy(xp) or get_sparse_adapter(xp) is None
        )
        # CuPy / Torch: tiled structures run on the device (the device tiled
        # kernel, sharing the linear block's device tiles); see
        # TROTSDeviceExactHessian.
        self._hess_device = not self._hess_host and self._device_kind is not None
        self._plan_registry = {}
        self._hess_cfg = (
            self._tiled_cfg if self._tiled_cfg is not None else TiledGramConfig()
        )
        # One entry per unique curved structure (insertion order = first use).
        self._curved: dict[int, _CurvedStructure] = {}
        curved_mats = [mat for mat, _c, _t, _p in self._nl_obj] + [
            mat for mat, _s, _t, _p, _b in self._nl_con
        ]
        for mat in curved_mats:
            if id(mat) in self._curved:
                continue
            cs = _CurvedStructure(mat)
            a = mat.matrix
            if (
                self._tiled_cfg is not None
                and a.shape[0] > 0
                and (self._hess_host or self._hess_device)
            ):
                hint = self._hessian_hint(mat)
                plan = TiledGramPlan.build(
                    (int(a.shape[0]), self._n),
                    [GramSegment(0, int(a.shape[0]), a, key=id(mat))],
                    values_dtype_hint=hint,
                    config=self._tiled_cfg,
                    registry=self._plan_registry,
                )
                cs.flop_ratio = plan.flop_ratio
                if plan.tiled:
                    cs.perm = plan.structures[0].perm
                    cs.tiles = plan.structures[0].tiles
                elif self._hess_device and plan.gated_structures:
                    # Gated off on a device after being planned: keep that
                    # plan for the device kernel (see _device_structure_plan)
                    # rather than planning the structure a second time.
                    gp = plan.gated_structures[0]
                    self._register_device_plan(cs, gp.perm, gp.tiles)
            if cs.tiles is None and self._hess_host:
                # Row slicing needs CSR, not sorted indices: never sort in
                # place (a CSR source is shared with the instance cache).
                cs.csr = sp.csr_matrix(a)
            self._curved[id(mat)] = cs
        # Beamlet-space quadratic terms: constant symmetric matrices (only
        # their multipliers change per call).
        self._quad_obj_sym = [
            (0.5 * (mat.matrix + mat.matrix.T)).tocsr() for mat, _c in self._quad_obj
        ]
        self._quad_con_sym = [
            (0.5 * (mat.matrix + mat.matrix.T)).tocsr()
            for mat, _s, _b in self._quad_con
        ]
        # Device namespaces: their sum's constant pattern, planned once, so a
        # call forms Q on the device from the multipliers alone.
        self._quad_device: _DeviceQuadratic | None = None
        if (
            self._hess_device
            and self._tiled_cfg is not None
            and (self._quad_obj_sym or self._quad_con_sym)
        ):
            self._quad_device = _DeviceQuadratic(
                self._quad_obj_sym + self._quad_con_sym, self._n
            )

    def _hessian_hint(self, mat: TROTSMatrix) -> str | None:
        """The tile-storage request of a curved structure's plan.

        float32 is a request: _plan_structure falls back to float64 unless
        the values are float32-exact."""
        hint = "float32" if mat.source_dtype == "float32" else None
        if (
            hint is not None
            and self.float32_group_folded
            and id(mat) in self._lin_structure_keys
        ):
            # The share gate folded this structure's linear rows into the
            # unhinted float64 group, which plans it without a hint: plan it
            # the same way, so the registry key matches and one tile set
            # serves the block and the Hessian (the exact products are
            # identical either way).
            hint = None
        return hint

    def _registry_key(self, mat: TROTSMatrix) -> tuple[Any, ...]:
        """The plan-registry key the linear block would use for ``mat``."""
        import numpy as np

        store = np.float32 if self._hessian_hint(mat) == "float32" else np.float64
        return (id(mat), np.dtype(store).str, self._tiled_cfg)

    def _register_device_plan(
        self, cs: _CurvedStructure, perm: Any, tiles: list[_Tile]
    ) -> None:
        assert self._plan_registry is not None
        self._plan_registry[self._registry_key(cs.mat)] = (cs.mat.matrix, perm, tiles)
        cs.device_perm, cs.device_tiles = perm, tiles

    def _device_structure_plan(self, cs: _CurvedStructure) -> tuple[Any, list[_Tile]]:
        """The device plan of a curved structure the gate keeps off, made on
        the first dense Hessian and cached on ``cs`` (see
        :class:`TiledGramConfig`: on a device the Hessian ignores the gate).

        The registry is consulted first, so a structure the linear block
        already planned (or planned later, from this entry) holds one tile
        set; a fresh plan is registered under the linear block's key.
        """
        import numpy as np

        if cs.device_tiles is None:
            a = cs.mat.matrix
            if a.shape[0] == 0:  # no rows, no tiles (nothing to register)
                cs.device_perm, cs.device_tiles = np.zeros((0,), np.int64), []
            else:
                assert self._plan_registry is not None
                assert self._tiled_cfg is not None
                hit = self._plan_registry.get(self._registry_key(cs.mat))
                if hit is not None and hit[0] is a:
                    cs.device_perm, cs.device_tiles = hit[1], hit[2]
                else:
                    store = (
                        np.float32
                        if self._hessian_hint(cs.mat) == "float32"
                        else np.float64
                    )
                    perm, tiles = _plan_structure(a, self._tiled_cfg, store)
                    self._register_device_plan(cs, perm, tiles)
        assert cs.device_tiles is not None
        return cs.device_perm, cs.device_tiles

    def lagrangian_hessian(
        self,
        x: Array,
        y_eq: Array,
        y_ineq: Array,
        sigma: Scalar = 1.0,
    ) -> Array | LinearOperator:
        import time

        t0 = time.perf_counter()
        if self._grouped_hessian:
            grouped = self._grouped_lagrangian_hessian(x, y_ineq, sigma)
            # Non-NumPy namespace: the device operator (CuPy / Torch with
            # tiling on and a curved or quadratic term), else cross the
            # grouped Hessian once per call instead of round-tripping every
            # product (class docstring).
            op: LinearOperator = (
                grouped
                if self._hess_host or isinstance(grouped, TROTSDeviceExactHessian)
                else self._csc_operator(grouped.host_sparse())
            )
        else:
            op = self._per_term_lagrangian_hessian(x, y_ineq, sigma)
        self.hessian_stats["build_seconds"] += time.perf_counter() - t0
        self.hessian_stats["build_calls"] += 1
        return op

    def _grouped_lagrangian_hessian(
        self, x: Array, y_ineq: Array, sigma: Scalar
    ) -> TROTSExactHessian:
        import numpy as np

        z = _to_numpy(x)
        xv = z[: self._n]
        s = float(sigma)
        yv = _to_numpy(y_ineq) if y_ineq is not None else np.zeros((0,))
        doses: dict[int, Any] = {}  # d_k = A_k x (+b), once per structure
        weights: dict[int, Any] = {}
        u_cols: list[Any] = []
        v_cols: list[Any] = []

        def add(mat: TROTSMatrix, mult: float, ctype: int, par: Any) -> None:
            if mult == 0.0:
                return  # no contribution (skips the per-voxel work too)
            key = id(mat)
            d = doses.get(key)
            if d is None:
                d = mat.matrix @ xv
                if mat.b.size == mat.matrix.shape[0]:
                    d = d + mat.b
                doses[key] = d
            diag, u, coef = _cost_hess_d(ctype, d, par)
            w = weights.get(key)
            if w is None:
                weights[key] = mult * diag
            else:
                w += mult * diag
            if coef != 0.0:  # gEUD: rank-1 c2·(Aᵀe1)(Aᵀe1)ᵀ, never densified
                au = np.asarray(mat.matrix.T @ u, dtype=np.float64).reshape(-1)
                u_cols.append(au)
                v_cols.append((mult * coef) * au)

        for mat, coef, ctype, par in self._nl_obj:
            add(mat, s * coef, ctype, par)
        row = 0
        for mat, sign, ctype, par, _bound in self._nl_con:
            add(mat, float(yv[row]) * sign, ctype, par)
            row += 1
        # One multiplier per quadratic term: objective terms first, then the
        # constraint rows (the order of _quad_obj_sym + _quad_con_sym).
        obj_mults = [s * coef for _mat, coef in self._quad_obj]
        con_mults = []
        for _mat, sign, _b in self._quad_con:
            con_mults.append(float(yv[row]) * sign)
            row += 1
        quad_syms = (self._quad_obj_sym, self._quad_con_sym)

        def quad_sum() -> Any:
            """``Q = Σ mult·sym`` (host SciPy); zero-multiplier constraint
            terms are skipped; None when no term contributes."""
            quad = None
            for sym, mult in zip(quad_syms[0], obj_mults, strict=True):
                quad = mult * sym if quad is None else quad + mult * sym
            for sym, mult in zip(quad_syms[1], con_mults, strict=True):
                if mult != 0.0:
                    quad = mult * sym if quad is None else quad + mult * sym
            return quad

        low_rank = (
            (np.stack(u_cols, axis=1), np.stack(v_cols, axis=1)) if u_cols else None
        )
        terms = [(self._curved[key], w) for key, w in weights.items()]
        if (
            self._hess_device
            and self._tiled_cfg is not None
            and (terms or self._quad_device is not None)
        ):
            # Every curved structure has a device plan (tiled, or planned
            # past the gate at construction) and Q a device pattern.
            from ipax.backend.namespace import device as device_of

            assert self._device_kind is not None
            context = _DeviceGramContext(
                self.xp,
                self._device_kind,
                device_of(self.xp.asarray(x)),
                self._device_tiles,
                sparse_tiles=self._hess_cfg.device_sparse_tiles,
                densify_budget_bytes=self._hess_cfg.device_densify_budget_bytes,
            )
            return TROTSDeviceExactHessian(
                self.xp,
                self._n_vars,
                self._n,
                terms,
                None,
                low_rank,
                self._hess_cfg,
                stats=self.hessian_stats,
                quad_builder=quad_sum,  # host fallbacks only (diagonal, COO)
                device_planner=self._device_structure_plan,
                context=context,
                device_matrix=self._eval_mat if self._eval_device else None,
                sparse_module=self._cxs if self._eval_device else None,
                quad_plan=self._quad_device,
                quad_mults=np.asarray(obj_mults + con_mults, dtype=np.float64),
            )
        return TROTSExactHessian(
            self.xp,
            self._n_vars,
            self._n,
            terms,
            quad_sum(),
            low_rank,
            self._hess_cfg,
            stats=self.hessian_stats,
        )

    def _per_term_lagrangian_hessian(
        self, x: Array, y_ineq: Array, sigma: Scalar
    ) -> LinearOperator:
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
        """``Aᵀ (∂²f/∂d²) A`` for one nonlinear term (host sparse; per-term path)."""
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
    "ActiveRowGramCSROperator",
    "GramSegment",
    "TROTSEntry",
    "TROTSExactProblem",
    "TROTSInstance",
    "TROTSMatrix",
    "TROTSProblem",
    "TROTSReference",
    "TiledGramCSROperator",
    "TiledGramConfig",
    "TiledGramPlan",
    "list_trots_cases",
    "load_trots_file",
    "objective_at",
    "parse_reference_result",
    "reference_for",
    "trots_dir",
    "trots_problems",
]
