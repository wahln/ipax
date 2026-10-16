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

"""CuPy sparse adapter (ADAPTER -- concrete-library import allowed here).

Builds ``cupyx.scipy.sparse`` matrices from Array-API COO triplets and solves
them on CUDA. The preferred direct route is NVIDIA cuDSS, accessed through the
official ``nvmath-python`` bindings (``nvmath.bindings.cudss``), which handle
library discovery and status checking and report inertia for symmetric/Hermitian
indefinite factorizations. If the bindings (or the cuDSS library they wrap) are
not installed, the adapter falls back to ``cupyx.scipy.sparse.linalg.spsolve``
for solve-only use.

Numerical failures are classified at this boundary (invariant #4), because
neither CUDA route raises on its own: a KKT matrix with non-finite entries is
rejected up front (as the SciPy adapter's ``_reject_non_finite`` does), and
cuDSS's static pivoting silently replaces a too-small pivot by
``PIVOT_EPSILON`` — an exactly singular system then "solves" to a finite,
meaningless vector (``±1/ε``) with a plausible inertia. A solve on such a
perturbed factorization is therefore iteratively refined against the
unperturbed matrix (GESP pairs static pivoting with refinement; Li & Demmel
2003) and then certified by its residual (see
:meth:`CuDSSSparseSolver._certify_perturbed_solve`), raising
:class:`LinearSolveError` when it does not solve the system at all.
Factorizations without perturbed pivots are neither refined nor certified.

``ipax`` deliberately does not choose a CUDA toolkit, CuPy wheel, or cuDSS
runtime for the user. The ``[sparse-cuda]`` extra installs only the
``nvmath-python`` bindings; see the installation documentation for the
user-managed CUDA dependencies.
"""

from __future__ import annotations

import ctypes
import math
import numbers
import warnings
from typing import TYPE_CHECKING, Any, cast

# Allowed import boundary (invariants #1, #4): backend/sparse/ adapters only.
import cupy
import cupyx
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg

try:  # High-level cuBLAS wrappers (ship with CuPy; syrk since v9).
    from cupy import cublas as _cublas
except ImportError:  # pragma: no cover - degraded CuPy install
    _cublas = None

try:  # cuSPARSE's generic SpMV with an explicit algorithm (see _csr_matmul).
    from cupy.cuda import device as _cuda_device
    from cupy_backends.cuda.libs import cusparse as _cusparse_lib
    from cupyx import cusparse as _cupyx_cusparse
except (ImportError, AttributeError):  # degraded install / the unit-test fakes
    _cuda_device = None
    _cusparse_lib = None
    _cupyx_cusparse = None

from ipax.backend.operators import LinearOperator
from ipax.backend.sparse._canonical import (
    CompiledLowerTriangle,
    compile_compressed,
    compile_lower_triangle,
    index_dtype,
)
from ipax.backend.sparse._routing import to_xp_array
from ipax.linalg.solver import LinearSolveError

if TYPE_CHECKING:
    from ipax.typing import Array, Namespace


# ``cudaDataType_t`` values from CUDA ``library_types.h``. These are ABI-stable
# CUDA enum values (not cuDSS-specific) and are what ``matrix_create_csr`` /
# ``matrix_create_dn`` expect for their ``value_type`` / ``index_type`` args.
_CUDSS_R_32F = 0
_CUDSS_R_64F = 1
_CUDSS_R_32I = 10
_CUDSS_R_64I = 24

# Cached nvmath cuDSS bindings module (``nvmath.bindings.cudss``).
_CUDSS_BINDINGS: Any | None = None

# ``cudssStatus_t`` codes that, on a *matrix creation* call, mean the runtime is
# unusable rather than the system numerically hard — the canonical cause is a
# cuDSS library whose ABI does not match the installed nvmath-python bindings
# (nvmath's own high-level API refuses such a pairing up front; the raw
# bindings do not check). Creation takes no matrix values, so no amount of
# regularization can repair it — degrade to spsolve instead.
_CUDSS_STATUS_INVALID_VALUE = 3
_CUDSS_STATUS_NOT_SUPPORTED = 4

# Density crossover / chunk cap for the weighted-Gram dense-GEMM strategy —
# mirrors ``ipax.backend.sparse.numpy_scipy`` (kept local to avoid importing
# the SciPy adapter into GPU-only environments); see that module for the flop
# model and TROTS measurements behind the values.
_GRAM_DENSE_MIN_DENSITY = 0.05
_GRAM_DENSE_CHUNK_ELEMENTS = 50_000_000
# Row-block size for mirroring the syrk-computed lower triangle to the upper
# one: bounded temporaries (block² indices) and O(n/block) kernel launches
# instead of either a full-size n² index gather or n single-row copies.
_GRAM_MIRROR_BLOCK = 1024
# Relative residual ``‖b − A x‖∞ / ‖b‖∞`` above which a cuDSS solve on a
# perturbed-pivot factorization is rejected as singular (see
# ``CuDSSSparseSolver._certify_perturbed_solve``): the forcing-term bound of
# an inexact Newton step, set where only a solve that failed to solve the
# system at all trips it — an inconsistent singular system leaves an O(1)
# residual (0.25 on the 2×2 contract case), a near-singular consistent one
# ~1e-8 (a degenerate L-BFGS border). Judged *after* the refinement below.
_PERTURBED_SOLVE_MAX_RESIDUAL = 1e-2
# Iterative refinement of a solve on a perturbed-pivot factorization (and
# only there): ``x += (A+E)⁻¹ (b − A x)`` against the unperturbed ``A``. Static
# pivoting is paired with refinement by design — the perturbation ``E`` only
# has to be small enough for the refinement to converge (Li & Demmel 2003,
# GESP; Higham 2002, *Accuracy and Stability of Numerical Algorithms*, 2nd
# ed., ch. 12: each step contracts the error by ≈ ‖(A+E)⁻¹E‖). The stopping
# rules follow IPOPT's ``PDFullSpaceSolver::Solve`` (IPOPT 3.14 options
# ``max_refinement_steps = 10``, ``residual_ratio_max = 1e-10``,
# ``residual_improvement_factor = 1``): stop after
# ``_PERTURBED_MAX_REFINEMENT_STEPS`` steps, once IPOPT's residual ratio
# ``‖r‖∞ / (min(‖x‖∞, 1e6 ‖b‖∞) + ‖b‖∞)`` (its ``ComputeResidualRatio``,
# ``max_cond = 1e6``) is at most ``_PERTURBED_REFINEMENT_TOL``, or at the
# first step that does not reduce the certified ratio ``‖r‖∞ / ‖b‖∞``. Two
# deliberate deviations: each column keeps its *best* iterate rather than
# the last, because ‖(A+E)⁻¹E‖ > 1 does occur (a ROTDISC KKT system: 1.3e-5
# unrefined, 20 after five steps, 8e7 after ten); and a solve already at the
# target takes no step (IPOPT: ``min_refinement_steps = 1``). The target is
# floored at ``_PERTURBED_REFINEMENT_EPS_FACTOR`` unit roundoffs of the
# factorization's dtype, so a float32 system stops at its attainable
# accuracy instead of stagnating toward 1e-10. cuDSS's own in-solve
# refinement (``CUDSS_CONFIG_IR_N_STEPS``) was measured and not used: it runs
# a fixed step count (``IR_TOL`` is ignored by the pinned cuDSS 0.7 runtime),
# so it can neither stop early nor stop that divergence.
_PERTURBED_MAX_REFINEMENT_STEPS = 10
_PERTURBED_REFINEMENT_TOL = 1e-10
_PERTURBED_REFINEMENT_IMPROVEMENT = 1.0
_PERTURBED_REFINEMENT_EPS_FACTOR = 100.0
_IPOPT_RESIDUAL_MAX_COND = 1e6

# Run-to-run reproducible sparse products. CuPy's ``csr @ x`` always passes
# cuSPARSE's ``*_ALG_DEFAULT``, whose CSR kernels accumulate with atomics in
# two cases: the transposed product ``Aᵀy`` (every cuSPARSE algorithm), and a
# forward product with long rows, which the default algorithm splits across
# thread blocks. Either returns different last bits from call to call (30
# distinct results in 30 repeats on a 3000×20000 matrix with 15000-entry
# rows, 2026-10-02 probe), and an interior-point trajectory amplifies them.
# The cuSPARSE documentation lists ``CUSPARSE_SPMV_CSR_ALG2`` as the
# bitwise-deterministic CSR SpMV for the *non-transposed* operation, so the
# adapter calls it through the generic API (``_csr_matmul``) and turns every
# transposed product into a non-transposed one on an explicit, cached ``Aᵀ``
# (``SparseOperator._transpose_csr``). ``cusparseSpMVAlg_t`` value from
# ``cusparse.h`` (CUDA ≥ 11.2; CuPy's bindings call it
# ``CUSPARSE_CSRMV_ALG2``); ``cusparseOperation_t`` likewise.
_SPMV_CSR_ALG2 = 3
_CUSPARSE_OPERATION_NON_TRANSPOSE = 0
# Multi-column products run in one of two ordered kernels (``_csr_matmul``):
# one thread per output entry, its row summed serially, while the longest row
# has at most this many entries; one warp per output entry beyond it (32
# strided partial sums, combined by a fixed shuffle tree), where a serial
# thread would wait on the long rows. Measured 2026-10-02 (RTX 4070 Laptop,
# float64, 2/8/64 columns, interleaved medians, cuSPARSE's SpMM as the
# reference): the thread kernel was at or below SpMM with rows of up to 1024
# entries but 3–6× slower on rows of 15 000–52 000; the warp kernel was
# within 0.8–1.2× of SpMM on those long rows, but up to 3× slower than the
# thread kernel on short ones.
_ORDERED_SPMM_MAX_ROW = 1024
# Largest explicit ``Aᵀ`` (values + indices + row pointers) an operator
# builds for its transposed products. A larger operator builds none and uses
# cuSPARSE's transposed product, which is not reproducible run to run — the
# one exception to the adapter's reproducible products. (The sparse-Gram
# strategies need ``Aᵀ`` for their SpGEMM and keep it whatever the cap, as
# they did before; the transposed products then use that copy.) The cached copy costs
# memory in every mode and the alternatives cost time in every mode, and
# the ``deterministic`` option never reaches the Jacobian operators to choose
# between them. Measured 2026-10-02 on the worst case at hand,
# Head-and-Neck_01's 1.1e8-nonzero dose block (CuPy dense route, 4
# iterations): caching its 1.24 GiB transpose took the pool's peak from 6.51
# to 7.75 GiB used (8.91 GiB reserved, past the 8 GiB card) and the run from
# 54 to 68 s; transposing row windows of 2**24 entries on every call instead
# (7× a cuSPARSE transposed SpMV on the Protons_01 KKT) took it to 6.85 GiB
# and 66 s. 512 MiB still caches every S2MPJ operator and an operator of
# the Protons_01 KKT's size (4.1e7 nonzeros, 476 MiB).
_TRANSPOSE_CACHE_MAX_BYTES = 512 * 2**20


def _generic_products_available() -> bool:
    """Whether cuSPARSE's generic SpMV can be called with an algorithm."""
    if _cusparse_lib is None or _cupyx_cusparse is None or _cuda_device is None:
        return False
    try:
        if cupy.cuda.runtime.is_hip:  # hipSPARSE has no such algorithm choice
            return False
        return bool(_cupyx_cusparse.check_availability("spmv"))
    except Exception:  # pragma: no cover - runtime probe failure: degrade
        return False


# Decided once at import (a capability of the installed CuPy/CUDA, not state).
_GENERIC_PRODUCTS = _generic_products_available()

# Ordered duplicate sum for the COO→CSR build (see ``_SegmentSumMap``): output
# slot ``i`` adds its emitted triplets ``vals[order[starts[i]:starts[i+1]]]``
# serially, in emission order, starting from 0 — the order of NumPy's
# sequential ``np.add.at`` in the SciPy adapter, so both builds agree to the
# bit. ``cupy.add.at`` sums with atomics in a different order on every call.
_SEGMENT_SUM_KERNEL = cupy.ElementwiseKernel(
    "raw T vals, raw I order, raw I starts",
    "T out",
    """
    const I lo = starts[i];
    const I hi = starts[i + 1];
    T acc = 0;
    for (I k = lo; k < hi; ++k) {
        acc += vals[order[k]];
    }
    out = acc;
    """,
    "ipax_ordered_segment_sum",
)

# ``out[i] = Σ_p data[p] · x[col·n + indices[p]]`` over row ``i % m`` of a
# CSR matrix, in storage order, for column ``col = i // m`` of a
# column-major ``x`` (see ``_csr_matmul``).
_ORDERED_SPMM_KERNEL = cupy.ElementwiseKernel(
    "raw T data, raw I indptr, raw I indices, raw T x, int64 m, int64 n",
    "T out",
    """
    const long long row = i % m;
    const long long base = (i / m) * n;
    T acc = 0;
    for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
        acc += data[p] * x[base + indices[p]];
    }
    out = acc;
    """,
    "ipax_ordered_csr_spmm",
)

# Its long-row variant: one warp per output entry ``(row, col)``; lane ``l``
# sums entries ``l, l+32, …`` of the row in order, and a fixed shuffle tree
# adds the 32 partial sums — the same order on every run.
_ORDERED_SPMM_WARP_SOURCE = r"""
template <typename T, typename I>
__global__ void ipax_ordered_csr_spmm_warp(
    const T* data, const I* indptr, const I* indices, const T* x, T* out,
    long long m, long long n, long long total)
{
    const long long entry =
        (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (entry >= total) return;  // uniform across the warp
    const long long row = entry % m;
    const long long base = (entry / m) * n;
    T acc = 0;
    for (I p = indptr[row] + lane; p < indptr[row + 1]; p += 32) {
        acc += data[p] * x[base + indices[p]];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) out[entry] = acc;
}
"""
_ORDERED_SPMM_WARP_NAMES = {
    ("float64", "int32"): "ipax_ordered_csr_spmm_warp<double, int>",
    ("float64", "int64"): "ipax_ordered_csr_spmm_warp<double, long long>",
    ("float32", "int32"): "ipax_ordered_csr_spmm_warp<float, int>",
    ("float32", "int64"): "ipax_ordered_csr_spmm_warp<float, long long>",
}
# Compiled on the first ``get_function`` (NVRTC), not at import.
_ORDERED_SPMM_WARP_MODULE = cupy.RawModule(
    code=_ORDERED_SPMM_WARP_SOURCE,
    name_expressions=list(_ORDERED_SPMM_WARP_NAMES.values()),
)
_WARP_THREADS_PER_BLOCK = 256


def _mirror_lower_to_upper(out: Any) -> None:
    """Mirror the lower triangle of square ``out`` onto the upper, in place.

    Pure copies (bitwise-exact, incl. -0.0), blocked to ``_GRAM_MIRROR_BLOCK``
    rows: each diagonal block mirrors internally via a block-sized index
    gather (temporaries bounded by block², never n²), and the panel right of
    it is a single transposed-slice copy — O(n/block) kernel launches total,
    where a per-row copy loop would launch n times.
    """
    n = int(out.shape[0])
    block = _GRAM_MIRROR_BLOCK
    tri = None
    for j0 in range(0, n, block):
        j1 = min(n, j0 + block)
        if j1 - j0 == block:
            if tri is None:
                tri = cupy.triu_indices(block, 1)
            rows, cols = tri
        else:
            rows, cols = cupy.triu_indices(j1 - j0, 1)
        diag = out[j0:j1, j0:j1]
        diag[rows, cols] = diag[cols, rows]
        if j1 < n:
            out[j0:j1, j1:] = out[j1:, j0:j1].T


def _resolve_refinement_policy(
    max_refinement_steps: int | None, perturbed_max_residual: float | None
) -> tuple[int, float]:
    """Validated ``(max_steps, threshold)`` of the perturbed-pivot policy.

    ``None`` takes the module default (``_PERTURBED_MAX_REFINEMENT_STEPS`` /
    ``_PERTURBED_SOLVE_MAX_RESIDUAL``), read at call time.
    """
    # Typed ``object``: callers (runner flags, user code) may pass anything.
    steps: object = (
        _PERTURBED_MAX_REFINEMENT_STEPS
        if max_refinement_steps is None
        else max_refinement_steps
    )
    threshold: object = (
        _PERTURBED_SOLVE_MAX_RESIDUAL
        if perturbed_max_residual is None
        else perturbed_max_residual
    )
    if (
        isinstance(steps, bool)
        or not isinstance(steps, numbers.Integral)
        or int(steps) < 0
    ):
        raise ValueError(f"max_refinement_steps must be an int >= 0, got {steps!r}")
    # An infinite threshold would silently disable the certificate; NaN and
    # non-positive values fail ``0 < x``.
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, numbers.Real)
        or not 0.0 < float(threshold) < math.inf
    ):
        raise ValueError(
            f"perturbed_max_residual must be finite and > 0, got {threshold!r}"
        )
    return int(steps), float(threshold)


def _resolve_deterministic(deterministic: object) -> bool:
    """Validated reproducible-factorization flag (a ``bool``)."""
    if not isinstance(deterministic, bool):
        raise ValueError(f"deterministic must be a bool, got {deterministic!r}")
    return deterministic


class _CuDSSUnavailableError(ImportError):
    """The nvmath binding or user-managed cuDSS runtime cannot be loaded."""


def _to_cupy(arr: Array) -> Any:
    """Return ``arr`` as a CuPy array, using DLPack where available."""
    if isinstance(arr, cupy.ndarray):
        return arr
    try:
        return cupy.from_dlpack(arr)
    except (TypeError, RuntimeError, BufferError, ValueError):
        return cupy.asarray(arr)


def _to_index(arr: Array) -> Any:
    """Device integer-index vector for COO/CSR assembly."""
    return _to_cupy(arr).astype(cupy.int64, copy=False)


def _scatter_add(out: Any, idx: Any, vals: Any) -> None:
    """Device unbuffered scatter-add (sums duplicate targets) for canonical maps."""
    # ``cupyx.scatter_add`` delegates to the deprecated ``ndarray.scatter_add``
    # (warns as of CuPy 14); ``cupy.add.at`` is its documented replacement.
    cupy.add.at(out, idx, vals)


def _csr_matmul(a: Any, x: Any, *, max_row: int | None = None) -> Any:
    """``a @ x`` for a CSR ``a``, bitwise reproducible run to run.

    A vector (or a one-column matrix) is one non-transposed SpMV with
    ``CUSPARSE_SPMV_CSR_ALG2`` (see ``_SPMV_CSR_ALG2``). No cuSPARSE SpMM
    algorithm was reproducible on long rows (the default and ``CSR_ALG1/2/3``
    all gave 30 distinct results in 30 repeats on rows of 15 000 entries), so
    a wider ``x`` runs through an ordered kernel: ``_ORDERED_SPMM_KERNEL``
    when the longest row has at most ``_ORDERED_SPMM_MAX_ROW`` entries, the
    warp variant otherwise. ``max_row`` is that longest row; callers that
    hold the structure pass it cached, else it costs one reduction and host
    sync per multi-column call. The kernels let NVCC contract ``acc += a·x``
    into an FMA, so a column of ``matmat`` can differ from ``matvec`` of the
    same vector in the last bits — each is reproducible on its own. The
    dtype handling mirrors CuPy's ``csr @ x`` (both operands promoted to the
    common type, a matrix result Fortran-ordered). Where the generic API or
    the dtype is unavailable (hipSPARSE, integer or complex data) this is
    CuPy's own product, reproducible or not.
    """
    if not _GENERIC_PRODUCTS or x.ndim not in (1, 2):
        return a @ x
    dtype = cupy.result_type(a.dtype, x.dtype)
    if dtype not in (cupy.float32, cupy.float64):
        return a @ x
    m, n = (int(d) for d in a.shape)
    if int(x.shape[0]) != n:
        raise ValueError(f"dimension mismatch: {a.shape} @ {x.shape}")
    if a.dtype != dtype:
        a = a.astype(dtype)
    vector = x.ndim == 1
    k = 1 if vector else int(x.shape[1])
    if a.nnz == 0 or m * k == 0:
        return cupy.zeros((m,) if vector else (m, k), dtype=dtype, order="F")
    x = cupy.asfortranarray(x, dtype=dtype)
    if k > 1:
        if max_row is None:
            max_row = _max_row(a.indptr)
        return _ordered_spmm(a, x, max_row)
    out = cupy.zeros((m,) if vector else (m, 1), dtype=dtype, order="F")
    # Host-pointer alpha = 1, beta = 0 (CuPy's handles use host pointer mode).
    scalar = ctypes.c_float if dtype == cupy.float32 else ctypes.c_double
    one, zero = scalar(1.0), scalar(0.0)
    mat = _cupyx_cusparse.SpMatDescriptor.create(a)
    vec_x = _cupyx_cusparse.DnVecDescriptor.create(x.reshape(-1))
    vec_y = _cupyx_cusparse.DnVecDescriptor.create(out.reshape(-1))
    args = (
        _cuda_device.get_cusparse_handle(),
        _CUSPARSE_OPERATION_NON_TRANSPOSE,
        ctypes.addressof(one),
        mat.desc,
        vec_x.desc,
        ctypes.addressof(zero),
        vec_y.desc,
        _cuda_data_type(dtype),
        _SPMV_CSR_ALG2,
    )
    buffer = cupy.empty(_cusparse_lib.spMV_bufferSize(*args), cupy.int8)
    _cusparse_lib.spMV(*args, _ptr(buffer))
    return out


def _max_row(indptr: Any) -> int:
    """Entries in the longest row of a CSR ``indptr`` (one host sync)."""
    if int(indptr.shape[0]) < 2:
        return 0
    return int(cupy.max(cupy.diff(indptr)))


def _ordered_spmm(a: Any, x: Any, max_row: int) -> Any:
    """``a @ x`` for a column-major ``x`` of ≥ 2 columns, in a fixed order."""
    m, n = (int(d) for d in a.shape)
    k = int(x.shape[1])
    flat = cupy.empty((m * k,), dtype=a.dtype)
    columns = x.T.reshape(-1)  # x's columns back to back (F order)
    if max_row <= _ORDERED_SPMM_MAX_ROW:
        _ORDERED_SPMM_KERNEL(a.data, a.indptr, a.indices, columns, m, n, flat)
    else:
        key = (str(a.dtype), str(a.indices.dtype))
        kernel = _ORDERED_SPMM_WARP_MODULE.get_function(_ORDERED_SPMM_WARP_NAMES[key])
        total = m * k
        blocks = -(-total * 32 // _WARP_THREADS_PER_BLOCK)
        sizes = (cupy.int64(m), cupy.int64(n), cupy.int64(total))
        args = (a.data, a.indptr, a.indices, columns, flat, *sizes)
        kernel((blocks,), (_WARP_THREADS_PER_BLOCK,), args)
    # Output entry ``i`` is ``(i % m, i // m)``: a column-major ``(m, k)``.
    return flat.reshape((k, m)).T


class _CSRStructure:
    """Value-independent facts of one canonical CSR pattern, computed once.

    The longest row (which ordered kernel a multi-column product takes) and
    ``Aᵀ``'s structure. The first :meth:`transpose` runs ``csr2csc``; a
    second value array on the same pattern builds the transpose plan, the
    permutation ``perm`` taking ``A``'s value array to ``Aᵀ``'s (a stable
    argsort of the column indices, so rows stay ascending within each column
    — the layout ``csr2csc`` produces, bit for bit), after which ``Aᵀ`` for
    new values is one gather, ``data[perm]``. The adapter shares one
    structure among the operators it builds on one pattern signature (see
    :class:`_SegmentSumMap`) — the facade's KKT operators, every iteration.
    An operator built by a fresh adapter (a problem's ``COOOperator``
    Jacobian is lowered through a new one each iteration) gets a fresh
    structure and so pays one ``csr2csc``, never the plan.
    """

    __slots__ = ("_max_row", "_perm", "_t_structure", "indices", "indptr", "shape")

    def __init__(self, indptr: Any, indices: Any, shape: tuple[int, int]) -> None:
        self.indptr = indptr
        self.indices = indices
        self.shape = (int(shape[0]), int(shape[1]))
        self._max_row: int | None = None
        self._t_structure: _CSRStructure | None = None
        self._perm: Any = None

    @property
    def max_row(self) -> int:
        """Entries in the longest row (one host sync, the first time)."""
        if self._max_row is None:
            self._max_row = _max_row(self.indptr)
        return self._max_row

    def transpose(self, data: Any) -> tuple[Any, _CSRStructure]:
        """``Aᵀ`` as a CSR for the value array ``data``, and its structure."""
        m, n = self.shape
        if self._t_structure is None:
            at = cupyx.scipy.sparse.csr_matrix(
                (data, self.indices, self.indptr), shape=self.shape
            ).T.tocsr()
            self._t_structure = _CSRStructure(at.indptr, at.indices, (n, m))
            return at, self._t_structure
        if self._perm is None:
            perm = cupy.argsort(self.indices)  # stable: rows ascending per column
            self._perm = perm.astype(self.indices.dtype, copy=False)
        t_structure = self._t_structure
        matrix = cupyx.scipy.sparse.csr_matrix(
            (data[self._perm], t_structure.indices, t_structure.indptr),
            shape=t_structure.shape,
        )
        matrix.has_canonical_format = True
        return matrix, t_structure


class _SegmentSumMap:
    """A fixed COO→canonical-CSR map with a reproducible duplicate sum.

    The structure comes from :func:`compile_compressed` (sorted unique
    ``(row, col)`` keys, compiled once per pattern); the duplicate sum does
    not use its scatter-add, which on CUDA is ``cupy.add.at`` — atomics,
    summed in a different order on every call when a slot has three or more
    emitted entries (30 distinct results in 30 repeats at ~15 entries per
    slot, 2026-10-02 probe). Instead the triplets are sorted once by slot
    (a stable argsort of the inverse map, so emission order within a slot),
    and :meth:`data` adds each slot's run serially (``_SEGMENT_SUM_KERNEL``):
    one launch, no atomics, the SciPy adapter's ``np.add.at`` order. The
    order and run boundaries replace the int64 inverse map, in int32 where
    the triplet count allows, so the map holds no more index memory than
    before; applying it took half the time of ``cupy.add.at`` on the
    Protons_01 KKT (19.7 vs 39.3 ms, 4.3e7 triplets).
    """

    __slots__ = ("_order", "_starts", "nnz", "structure")

    def __init__(self, rows: Any, cols: Any, shape: tuple[int, int]) -> None:
        compiled = compile_compressed(
            cupy,
            _scatter_add,
            major=rows,
            minor=cols,
            n_major=shape[0],
            n_minor=shape[1],
        )
        inverse = compiled.inverse
        nnz, structure = compiled.nnz, (compiled.indptr, compiled.indices)
        del compiled  # its inverse map is replaced by the order/starts below
        n_triplets = int(inverse.shape[0])
        idx_dtype = index_dtype(cupy, n_triplets + 1)
        # CuPy's argsort is stable (documented), so a slot's run keeps the
        # emission order. ``searchsorted`` rather than ``bincount`` for the
        # run boundaries: CuPy 14.1's bincount faulted (illegal address) at
        # minlength ≈ 4e7, the Protons_01 KKT's nonzero count.
        # Intermediates are dropped as soon as they are used: at 4.3e7
        # triplets each int64 temporary is 0.34 GB.
        order = cupy.argsort(inverse)
        sorted_slots = inverse[order]
        del inverse
        self._order = order.astype(idx_dtype, copy=False)
        del order
        self._starts = cupy.searchsorted(
            sorted_slots, cupy.arange(nnz + 1), side="left"
        ).astype(idx_dtype, copy=False)
        del sorted_slots
        self.nnz = nnz
        self.structure = _CSRStructure(*structure, shape)

    def data(self, values: Any) -> Any:
        """Canonical ``data`` for ``values``: each slot's entries, in order."""
        out = cupy.empty((self.nnz,), dtype=values.dtype)
        if self.nnz:
            _SEGMENT_SUM_KERNEL(values, self._order, self._starts, out)
        return out

    def csr(self, values: Any) -> Any:
        """The canonical CSR matrix for ``values``."""
        structure = self.structure
        matrix = cupyx.scipy.sparse.csr_matrix(
            (self.data(values), structure.indices, structure.indptr),
            shape=structure.shape,
        )
        matrix.has_canonical_format = True
        return matrix


def _to_float(value: Any) -> float:
    """Bring a scalar-like CuPy value to a Python ``float``."""
    return float(cupy.asnumpy(value))


def _ptr(arr: Any) -> int:
    """Device pointer (``intptr_t``) for a CuPy array; nvmath takes plain ints."""
    return int(arr.data.ptr)


def _value_type(dtype: Any) -> int:
    """cuDSS scalar type for real CuPy matrix values."""
    normalized = cupy.dtype(dtype)
    if normalized == cupy.dtype(cupy.float32):
        return _CUDSS_R_32F
    if normalized == cupy.dtype(cupy.float64):
        return _CUDSS_R_64F
    raise TypeError("cuDSS sparse solver supports only float32/float64 values")


# The same ``cudaDataType_t`` values serve cuSPARSE's ``computeType``.
_cuda_data_type = _value_type


def _index_type(dtype: Any) -> int:
    """cuDSS scalar type for integer CSR offsets/indices."""
    normalized = cupy.dtype(dtype)
    if normalized == cupy.dtype(cupy.int32):
        return _CUDSS_R_32I
    if normalized == cupy.dtype(cupy.int64):
        return _CUDSS_R_64I
    raise TypeError("cuDSS sparse solver supports only int32/int64 indices")


def _is_symmetric(matrix: Any) -> bool:
    """Return whether a CuPy sparse matrix is numerically symmetric."""
    if matrix.shape[0] != matrix.shape[1]:
        return False
    diff = (matrix - matrix.T).tocoo()
    diff.eliminate_zeros()
    if int(diff.nnz) == 0:
        return True
    matrix_scale = _to_float(cupy.max(cupy.abs(matrix.data))) if matrix.nnz else 0.0
    diff_scale = _to_float(cupy.max(cupy.abs(diff.data)))
    n = int(matrix.shape[0])
    tol = max(1.0, matrix_scale) * n * float(cupy.finfo(cupy.float64).eps)
    return diff_scale <= tol


def _canonical_csr(matrix: Any) -> Any:
    """CSR with summed duplicates and sorted column indices.

    A CSR already flagged canonical is returned as is. Anything else goes
    through a one-off :class:`_SegmentSumMap`, never CuPy's
    ``sum_duplicates``, whose duplicate sum is an ``atomicAdd`` and so not
    reproducible run to run.
    """
    if getattr(matrix, "format", None) == "csr" and matrix.has_canonical_format:
        return matrix
    coo = matrix.tocoo()  # expands the compressed axis; sums nothing
    return _SegmentSumMap(coo.row, coo.col, coo.shape).csr(coo.data)


def _device_id(matrix: Any) -> int:
    """CUDA device containing a CuPy sparse matrix's value storage."""
    device = getattr(matrix.data, "device", None)
    if device is not None and hasattr(device, "id"):
        return int(device.id)
    # Test doubles and third-party CuPy-compatible arrays may not expose
    # ``ndarray.device``; their active CUDA device is the best available signal.
    return int(cupy.cuda.runtime.getDevice())


def _is_cudss_load_error(exc: BaseException) -> bool:
    """Return whether ``exc`` means the optional cuDSS runtime is unavailable."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ImportError, OSError)):
            return True
        cls = type(current)
        if cls.__name__ in {
            "DynamicLibNotFoundError",
            "LibraryNotFoundError",
        } and cls.__module__.startswith("cuda.pathfinder"):
            return True
        current = current.__cause__ or current.__context__
    return False


class SparseOperator(LinearOperator):
    """``LinearOperator`` backed by a ``cupyx.scipy.sparse`` CSR matrix.

    Unlike the SciPy adapter the CSR form is kept eager: cuDSS consumes CSR, which
    is also the matvec format, so it is not wasted on the direct-solve route. The
    only per-iteration trim here is caching the symmetry verdict (and honoring the
    assembler's structural hint) instead of recomputing ``A − Aᵀ`` every factor.

    The products are bitwise reproducible run to run: forward products use
    cuSPARSE's deterministic SpMV or an ordered kernel (:func:`_csr_matmul`),
    and transposed ones (``rmatvec``, ``rmatmat``, ``gram_diagonal``, the
    sparse Gram's ``Aᵀ`` factor) run forward on an explicit ``Aᵀ`` CSR,
    built on the first transposed product and kept for the operator's value
    array (:meth:`_transpose_csr`). That ``Aᵀ`` is a second copy of the
    matrix in device memory (its values only, when the adapter shares the
    pattern's transpose plan); an operator that is only factored or only
    applied forward never builds it. ``structure`` is the adapter's shared
    per-pattern :class:`_CSRStructure`; without it the operator keeps its
    own. (The sparse Gram's SpGEMM itself is cuSPARSE's default algorithm,
    observed but not documented to be reproducible.)
    """

    def __init__(
        self,
        matrix: Any,
        xp: Namespace,
        *,
        symmetric: bool | None = None,
        pattern_signature: object | None = None,
        canonical: bool = False,
        structure: _CSRStructure | None = None,
    ) -> None:
        # ``canonical=True`` marks ``matrix`` as an already-canonical CSR from the
        # compiled COO map (values-only refactor fast path), so the per-iteration
        # ``tocsr().sum_duplicates().sort_indices()`` is skipped.
        self._matrix = matrix if canonical else _canonical_csr(matrix)
        self._xp = xp
        # Shared per-pattern structure (adapter-built operators), else own.
        self._structure = structure
        self._transpose_structure: _CSRStructure | None = None
        # Structural symmetry hint from the assembler (None ⇒ test numerically).
        self._symmetric_hint = symmetric
        self._pattern_signature = pattern_signature
        # Aᵀ as an explicit CSR (see _transpose_csr), keyed on the value array
        # it was built from; shared by the transposed products and the Gram.
        self._transpose: Any = None
        self._transpose_source: Any = None
        self._squared_transpose: Any = None  # (A∘A)ᵀ, sharing Aᵀ's indices
        # Gram-path caches — mirror of the SciPy adapter (see its ``gram``):
        # a last-weights memo plus one-time scaled-buffer / row-map symbolic
        # work, leaving only value work + the SpGEMM on a miss.
        self._gram_scaled: Any = None  # diag(w)A same-pattern buffer
        self._gram_row_index: Any = None  # nnz → row map for w expansion
        # One memo slot per accumulate_dtype (see the SciPy adapter's comment:
        # mixed/exact alternation with identical weights must not evict).
        self._gram_memo: dict[str | None, tuple[Any, Any]] = {}
        self._gram_compute_count = 0  # observability/testing: actual SpGEMM runs
        self._reduced_csr: Any = None  # cast-once reduced-precision data CSR
        self._native_streak = False  # last gram() request was native (see gram)
        self._squared: Any = None  # A∘A (row_gram_diagonal), sharing A's indices
        self._squared_source: Any = None
        # Sparse-Gram (COO) memo for the sparse normal-equations route.
        self._gram_coo_weights: Any = None
        self._gram_coo_value: Any = None
        self._symmetric: bool | None = None

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self._matrix.shape
        return int(rows), int(cols)

    @property
    def cupy_matrix(self) -> Any:
        """The wrapped device CSR matrix (consumed by CUDA sparse solvers)."""
        return self._matrix

    def is_symmetric(self) -> bool:
        """Whether ``A == Aᵀ``: honor the assembler's hint, else test numerically.

        The structural hint (set by the KKT condensed/saddle assembler) is
        authoritative when present, sparing the per-iteration O(nnz) device-side
        ``A − Aᵀ`` test and its host syncs.
        """
        if self._symmetric_hint is not None:
            return self._symmetric_hint
        if self._symmetric is None:
            self._symmetric = _is_symmetric(self._matrix)
        return self._symmetric

    def coo_pattern_signature(self) -> object | None:
        return self._pattern_signature

    def _own_structure(self) -> _CSRStructure:
        if self._structure is None:
            a = self._matrix
            self._structure = _CSRStructure(a.indptr, a.indices, self.shape)
        return self._structure

    def _transpose_csr(self) -> Any:
        """``Aᵀ`` as an explicit CSR, built once per value array.

        cuSPARSE's transposed SpMV/SpMM scatter with atomics (every
        algorithm), so a transposed product runs as a forward one on ``Aᵀ``
        instead. The transpose is a permutation of the values (no
        arithmetic), reproducible bit for bit: one gather through the shared
        pattern's plan when the adapter supplied one, else ``csr2csc``.
        Keyed on the identity of the wrapped matrix's value array: replacing
        ``cupy_matrix.data`` rebuilds it (in-place writes into that array are
        not detected — operators are values, built anew by ``from_coo`` when
        values move).
        """
        data = self._matrix.data
        if self._transpose is None or self._transpose_source is not data:
            if self._structure is not None:
                self._transpose, self._transpose_structure = self._structure.transpose(
                    data
                )
            else:
                at = self._matrix.T.tocsr()
                if self._transpose_structure is None:
                    self._transpose_structure = _CSRStructure(
                        at.indptr, at.indices, at.shape
                    )
                self._transpose = at
            self._transpose_source = data
            self._squared_transpose = None
        return self._transpose

    def _product(self, x: Any) -> Any:
        """``A @ x`` (device), with the cached longest row."""
        ordered = x.ndim == 2 and _GENERIC_PRODUCTS  # else no kernel choice
        max_row = self._own_structure().max_row if ordered else None
        return _csr_matmul(self._matrix, x, max_row=max_row)

    def _transpose_fits_cache(self) -> bool:
        """Whether the explicit ``Aᵀ`` fits ``_TRANSPOSE_CACHE_MAX_BYTES``."""
        a = self._matrix
        size = int(a.nnz) * (a.data.itemsize + a.indices.itemsize)
        size += (int(a.shape[1]) + 1) * a.indptr.itemsize
        return bool(size <= _TRANSPOSE_CACHE_MAX_BYTES)

    def _transposed_product(self, x: Any, *, squared: bool = False) -> Any:
        """``Aᵀ @ x`` (or ``(A∘A)ᵀ @ x``) on the cached explicit transpose."""
        cached = (
            self._transpose is not None and self._transpose_source is self._matrix.data
        )
        if not cached and not self._transpose_fits_cache():
            # Over budget and not built by the sparse Gram (which keeps its
            # Aᵀ whatever the cap): cuSPARSE's product, not reproducible.
            return (self._squared_csr if squared else self._matrix).T @ x
        at = self._transpose_csr()
        if squared:
            if self._squared_transpose is None:
                self._squared_transpose = cupyx.scipy.sparse.csr_matrix(
                    (at.data * at.data, at.indices, at.indptr), shape=at.shape
                )
            at = self._squared_transpose
        structure = self._transpose_structure
        assert structure is not None  # set by _transpose_csr
        max_row = structure.max_row if x.ndim == 2 and _GENERIC_PRODUCTS else None
        return _csr_matmul(at, x, max_row=max_row)

    def matvec(self, v: Array) -> Array:
        return cast("Array", to_xp_array(self._product(_to_cupy(v)), self._xp))

    def rmatvec(self, v: Array) -> Array:
        out = self._transposed_product(_to_cupy(v))
        return cast("Array", to_xp_array(out, self._xp))

    def matmat(self, V: Array) -> Array:
        return cast("Array", to_xp_array(self._product(_to_cupy(V)), self._xp))

    def rmatmat(self, V: Array) -> Array:
        out = self._transposed_product(_to_cupy(V))
        return cast("Array", to_xp_array(out, self._xp))

    def diagonal(self, like: Array | None = None) -> Array:
        del like
        return cast("Array", to_xp_array(self._matrix.diagonal(), self._xp))

    @property
    def _squared_csr(self) -> Any:
        """The elementwise square ``A∘A``, built once per value array (the
        row Gram diagonal is recomputed every IPM iteration). Same pattern as
        ``A``, so it shares ``A``'s index arrays and adds one value array."""
        data = self._matrix.data
        if self._squared is None or self._squared_source is not data:
            a = self._matrix
            self._squared = cupyx.scipy.sparse.csr_matrix(
                (data * data, a.indices, a.indptr), shape=a.shape
            )
            self._squared_source = data
        return self._squared

    def gram_diagonal(self, weights: Array) -> Array:
        # diag(Aᵀ diag(w) A)_k = Σ_i w_i A_ik² = (A∘A)ᵀ w (column energies),
        # forward on (A∘A)ᵀ = Aᵀ∘Aᵀ, which shares the cached Aᵀ's indices.
        out = self._transposed_product(_to_cupy(weights), squared=True)
        return cast("Array", to_xp_array(cupy.asarray(out).reshape(-1), self._xp))

    def gram(
        self,
        weights: Array,
        *,
        accumulate_dtype: str | None = None,
        hinted_only: bool = False,
    ) -> Array:
        if (
            hinted_only
            and accumulate_dtype is not None
            and self.gram_accumulate_dtype_hint() != accumulate_dtype
        ):
            accumulate_dtype = None  # this matrix's data does not support it
        # Aᵀ diag(w) A as a dense n×n matrix via sparse arithmetic (row-scale then
        # sparse Gram), densifying only the small n×n result — mirror of the SciPy
        # adapter's cached ``gram`` (see its comment for the caching rationale).
        # The memo compare costs one device reduction + host sync per call; the
        # SpGEMM it skips is orders of magnitude more work at n ≪ m scale.
        w = _to_cupy(weights).reshape(-1)
        if accumulate_dtype is None:
            # Free the reduced-data device copy (multi-GB at RT nnz) only on the
            # SECOND consecutive native request — mirror of the SciPy adapter.
            # One native request is an exact rebuild after a failed refinement,
            # after which ``DenseSolver`` re-engages mixed on the next
            # factorization; freeing then would force a multi-GB device recast
            # on every intermittent hard iteration.
            if self._native_streak:
                self._reduced_csr = None
            self._native_streak = True
        else:
            self._native_streak = False
        memo = self._gram_memo.get(accumulate_dtype)
        if memo is not None and bool(cupy.array_equal(w, memo[0])):
            return cast("Array", to_xp_array(memo[1], self._xp))

        a = self._matrix
        m, n = a.shape
        size = int(m) * int(n)
        if size > 0 and a.nnz / size >= _GRAM_DENSE_MIN_DENSITY:
            # Dense-ish rows: chunked dense GEMM (cuBLAS) beats SpGEMM — the
            # CPU-calibrated crossover of the SciPy adapter, if anything more
            # pronounced on GPU tensor pipelines.
            if accumulate_dtype is not None:
                # Mixed-precision accumulate — mirror of the SciPy adapter:
                # cast-once reduced-data CSR, reduced-dtype cuBLAS
                # accumulation (the fp32/fp64 rate ratio is the whole point
                # on consumer GPUs), upcast only the n×n result. Dense
                # strategy only; the SpGEMM branch ignores the request.
                acc = cupy.dtype(accumulate_dtype)
                gram = self._gram_dense_accumulate(
                    self._reduced_data_csr(acc), w.astype(acc, copy=False)
                )
                gram = gram.astype(cupy.result_type(a.dtype, w.dtype), copy=False)
            else:
                gram = self._gram_dense_accumulate(a, w)
        else:
            transpose = self._transpose_csr()
            if self._gram_scaled is None:
                self._gram_scaled = a.copy()
                # nnz→row map, device-native: row of stored element k is the
                # indptr bucket containing k (no host round-trip of an O(m)
                # row-count vector, which cupy.repeat's list repeats would need).
                self._gram_row_index = (
                    cupy.searchsorted(a.indptr, cupy.arange(int(a.nnz)), side="right")
                    - 1
                )
            scaled = self._gram_scaled
            scaled.data = a.data * w[self._gram_row_index]
            gram = cupy.asarray((transpose @ scaled).toarray())
        self._gram_memo[accumulate_dtype] = (w.copy(), gram)
        self._gram_compute_count += 1
        return cast("Array", to_xp_array(gram, self._xp))

    def _reduced_data_csr(self, dtype: Any) -> Any:
        """Same-structure CSR with the data cast once to ``dtype`` (device).

        Index arrays are shared with the wrapped matrix; only the value array
        is duplicated, so reduced-precision accumulates read half the bytes
        with no per-chunk cast."""
        if self._reduced_csr is None or self._reduced_csr.dtype != dtype:
            a = self._matrix
            self._reduced_csr = cupyx.scipy.sparse.csr_matrix(
                (a.data.astype(dtype), a.indices, a.indptr), shape=a.shape
            )
        return self._reduced_csr

    def gram_coo(self, weights: Array) -> tuple[Array, Array, Array, tuple[int, int]]:
        # ``Aᵀ diag(w) A`` kept SPARSE for the sparse normal-equations route —
        # mirror of the SciPy adapter's ``gram_coo`` (see its comment for the
        # pattern-stability contract), on the device via cuSPARSE SpGEMM.
        w = _to_cupy(weights).reshape(-1)
        if self._gram_coo_value is not None and bool(
            cupy.array_equal(w, self._gram_coo_weights)
        ):
            product = self._gram_coo_value
        else:
            a = self._matrix
            transpose = self._transpose_csr()
            if self._gram_scaled is None:
                self._gram_scaled = a.copy()
                self._gram_row_index = (
                    cupy.searchsorted(a.indptr, cupy.arange(int(a.nnz)), side="right")
                    - 1
                )
            scaled = self._gram_scaled
            scaled.data = a.data * w[self._gram_row_index]
            product = (transpose @ scaled).tocoo()
            self._gram_coo_weights = w.copy()
            self._gram_coo_value = product
            self._gram_compute_count += 1
        n = int(self.shape[1])
        return (
            cast("Array", to_xp_array(product.row, self._xp)),
            cast("Array", to_xp_array(product.col, self._xp)),
            cast("Array", to_xp_array(product.data, self._xp)),
            (n, n),
        )

    def gram_coo_capable(self) -> bool:
        return True

    def gram_accumulate_dtype_hint(self) -> str | None:
        # Storage-level metadata only — mirror of the SciPy adapter.
        return "float32" if self._matrix.dtype == cupy.float32 else None

    def gram_fill_estimate(self) -> float | None:
        """Estimated Gram-pattern density — the SciPy adapter's sampled
        column-overlap estimate on the host-transferred index structure.

        A one-time route-selection probe, so the two index-array transfers
        (a few MB) are irrelevant; running the 128-column union loop on the
        GPU would instead cost hundreds of tiny kernel launches and syncs.
        Reuses :func:`ipax.backend.sparse.numpy_scipy._estimate_gram_fill`
        (a CuPy stack without SciPy skips the probe rather than raising).
        """
        try:
            from ipax.backend.sparse.numpy_scipy import _estimate_gram_fill
        except ImportError:
            return None
        m, n = self.shape
        if m == 0 or n == 0:
            return 0.0
        csr = self._matrix.tocsr()
        csc = self._matrix.tocsc()
        return _estimate_gram_fill(
            cupy.asnumpy(csr.indptr),
            cupy.asnumpy(csr.indices),
            cupy.asnumpy(csc.indptr),
            cupy.asnumpy(csc.indices),
            n,
        )

    @staticmethod
    def _gram_dense_accumulate(a: Any, w: Any) -> Any:
        """``Aᵀ diag(w) A`` by cuBLAS over zero-copy row windows of ``a``.

        Same chunked dense strategy as the SciPy adapter: densify
        ``m_chunk × n`` row windows (bounded by ``_GRAM_DENSE_CHUNK_ELEMENTS``).
        For finite nonnegative weights the chunk is scaled in place by ``√w``
        and accumulated with the *symmetric* rank-k update
        (``cupy.cublas.syrk``): half the GEMM FLOPs and no ``w ∘ block``
        temporary, with the lower triangle mirrored once at the end (bitwise
        symmetric result). Mixed-sign or non-finite weights fall back to the
        general ``blockᵀ @ (w_chunk ∘ block)`` accumulation — see the SciPy
        adapter's docstring for the overflow-envelope caveat of the ``√w``
        split (the two forms saturate differently at float32 extremes).
        """
        from cupyx.scipy import sparse as cupyx_sparse

        m, n = a.shape
        dtype = cupy.result_type(a.dtype, w.dtype)
        chunk = max(1, _GRAM_DENSE_CHUNK_ELEMENTS // max(int(n), 1))
        # One consolidated host transfer of the chunk-boundary offsets instead
        # of two 0-d syncs per chunk inside the loop. The weight-gate reduction
        # below is one more sync per gram call — same cost class as the memo
        # compare in :meth:`gram`, negligible next to the accumulate.
        starts = [*range(0, m, chunk), m]
        bounds = [int(v) for v in cupy.asnumpy(a.indptr[cupy.asarray(starts)])]
        use_syrk = (
            _cublas is not None
            and dtype in (cupy.float32, cupy.float64)
            and bool(cupy.all((w >= 0.0) & cupy.isfinite(w)))
        )
        out = cupy.zeros((n, n), dtype=dtype)
        if use_syrk:
            sqrt_w = cupy.sqrt(w.astype(dtype, copy=False))
        for i, start in enumerate(starts[:-1]):
            end = starts[i + 1]
            lo, hi = bounds[i], bounds[i + 1]
            window = cupyx_sparse.csr_matrix(
                (a.data[lo:hi], a.indices[lo:hi], a.indptr[start : end + 1] - lo),
                shape=(end - start, n),
            )
            block = cupy.asarray(window.toarray(), dtype=dtype)
            if use_syrk:
                block *= sqrt_w[start:end, None]  # in place — block is fresh
                # out := blockᵀ @ block + out, lower triangle only (verified
                # in-place on C-order ``out`` for float32/float64).
                _cublas.syrk("T", block, out=out, alpha=1.0, beta=1.0, lower=True)
            else:
                out += block.T @ (w[start:end, None] * block)
        if use_syrk:
            _mirror_lower_to_upper(out)
        return out

    def row_gram_diagonal(self, weights: Array) -> Array:
        # diag(A diag(w) Aᵀ)_j = Σ_k w_k A_jk² = (A∘A) w (row energies).
        w = _to_cupy(weights)
        ordered = w.ndim == 2 and _GENERIC_PRODUCTS
        max_row = self._own_structure().max_row if ordered else None
        out = _csr_matmul(self._squared_csr, w, max_row=max_row)
        return cast("Array", to_xp_array(cupy.asarray(out).reshape(-1), self._xp))

    def row_inf_norms(self, like: Array | None = None) -> Array:
        del like
        m, n = self.shape
        if n == 0:
            norms = cupy.zeros((m,), dtype=self._matrix.dtype)
        else:
            norms = abs(self._matrix).max(axis=1).toarray().reshape((m,))
        return cast("Array", to_xp_array(norms, self._xp))

    def to_coo(
        self, like: Array | None = None
    ) -> tuple[Array, Array, Array, tuple[int, int]]:
        del like
        coo = self._matrix.tocoo()
        return (
            cast("Array", to_xp_array(coo.row, self._xp)),
            cast("Array", to_xp_array(coo.col, self._xp)),
            cast("Array", to_xp_array(coo.data, self._xp)),
            self.shape,
        )


def _reject_non_finite(matrix: Any) -> None:
    """Raise :class:`LinearSolveError` if any stored value is inf/NaN.

    Mirrors ``ipax.backend.sparse.numpy_scipy._reject_non_finite``: a
    non-finite KKT entry means upstream derivatives overflowed at a bad trial
    iterate, a numerical failure the IPM recovers from (δ_w ladder, then
    step-failure restoration; Wächter & Biegler 2006, §3.1 → §3.3). Neither
    cuDSS nor ``spsolve`` raises on one — cuDSS factors it to a NaN solution
    with a plausible inertia. One host synchronization.
    """
    if matrix.nnz and not bool(cupy.isfinite(matrix.data).all()):
        raise LinearSolveError(
            "sparse KKT matrix has non-finite entries (inf/NaN); the upstream "
            "derivatives overflowed at this iterate"
        )


def _require_csr(K: LinearOperator) -> tuple[Any, Namespace]:
    """Extract a square CuPy CSR matrix + namespace from ``SparseOperator``."""
    if not isinstance(K, SparseOperator):
        raise TypeError(
            "CuPy sparse solver requires a SparseOperator built from COO "
            f"triplets; got {type(K).__name__}"
        )
    matrix = K.cupy_matrix
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("sparse solver requires a square operator")
    return matrix, K._xp


def _load_cudss() -> Any:
    """Import the nvmath-python cuDSS bindings lazily.

    ``nvmath.bindings.cudss`` wraps NVIDIA cuDSS, dlopening the library and
    raising :class:`cuDSSError` on non-success statuses. Kept lazy (not a
    module-level import) so the CuPy adapter still degrades to the spsolve
    fallback when nvmath/cuDSS is absent.
    """
    global _CUDSS_BINDINGS
    if _CUDSS_BINDINGS is not None:
        return _CUDSS_BINDINGS
    try:
        from nvmath.bindings import cudss
    except ImportError as exc:
        raise ImportError(
            "NVIDIA cuDSS bindings are unavailable; after installing a "
            "CUDA-compatible CuPy package, install `ipax[sparse-cuda]` for "
            "the nvmath bindings"
        ) from exc
    _CUDSS_BINDINGS = cudss
    return cudss


class CuPySpsolveSolver:
    """Solve-only fallback backed by ``cupyx.scipy.sparse.linalg.spsolve``."""

    def __init__(self, *, require_inertia: bool = False) -> None:
        self._require_inertia = require_inertia
        self._matrix: Any | None = None
        self._xp: Namespace | None = None

    def describe(self) -> str:
        return "CuPy spsolve (GPU)"

    def factor(self, K: LinearOperator) -> None:
        if self._require_inertia:
            raise NotImplementedError(
                "GPU sparse inertia requires a user-managed NVIDIA cuDSS runtime "
                "compatible with the installed CUDA toolkit"
            )
        matrix, xp = _require_csr(K)
        # spsolve re-factorizes on every solve, so the O(nnz) check here is
        # noise beside it.
        _reject_non_finite(matrix)
        self._matrix = matrix
        self._xp = xp

    def solve(self, rhs: Array) -> Array:
        if self._matrix is None or self._xp is None:
            raise RuntimeError("factor() must be called before solve()")
        try:
            x = cupyx.scipy.sparse.linalg.spsolve(self._matrix, _to_cupy(rhs))
        except RuntimeError as exc:
            raise LinearSolveError("CuPy sparse solve failed") from exc
        return cast("Array", to_xp_array(x, self._xp))

    @property
    def inertia(self) -> tuple[int, int, int]:
        """Inertia is unavailable from the ``spsolve`` fallback."""
        raise RuntimeError("inertia is unavailable without cuDSS")

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """The ``spsolve`` fallback is not inertia-revealing."""
        return None


class CuDSSSparseSolver:
    """Sparse CUDA direct solver backed by NVIDIA cuDSS via nvmath-python.

    cuDSS can solve general sparse systems. Inertia is available only for
    symmetric/Hermitian non-positive-definite matrix types, so this wrapper
    requests a symmetric lower-triangle LDL^T factorization whenever inertia is
    required.

    A solve on a factorization in which cuDSS perturbed pivots (static
    pivoting) is iteratively refined against the unperturbed matrix (at most
    ``max_refinement_steps`` steps; 0 disables it) and then certified: a
    right-hand-side column whose relative residual ``‖b − A x‖∞ / ‖b‖∞``
    still exceeds ``perturbed_max_residual`` raises
    :class:`LinearSolveError`. Both default to the module constants
    ``_PERTURBED_MAX_REFINEMENT_STEPS`` and ``_PERTURBED_SOLVE_MAX_RESIDUAL``
    (read at construction). A factorization without perturbed pivots is
    neither refined nor certified.

    ``deterministic=True`` sets cuDSS's ``DETERMINISTIC_MODE`` on the solver's
    config, so repeated factorizations of the same matrix are bitwise
    identical. The default cuDSS factorization is not: its parallel
    reductions run in a varying order (six fresh solvers on one captured
    ROTDISC KKT system gave six solutions differing by up to 1.4e2). The mode
    costs 1.2-3x per factorization (measured on cuDSS 0.7.1; analysis
    unchanged). A runtime without the parameter warns and factors in the
    default mode.
    """

    def __init__(
        self,
        *,
        require_inertia: bool = False,
        max_refinement_steps: int | None = None,
        perturbed_max_residual: float | None = None,
        deterministic: bool = False,
    ) -> None:
        self._require_inertia = require_inertia
        self._max_refinement_steps, self._perturbed_max_residual = (
            _resolve_refinement_policy(max_refinement_steps, perturbed_max_residual)
        )
        self._deterministic = _resolve_deterministic(deterministic)
        # Outcome of the last perturbed-pivot solve (host values the
        # refinement already read, kept for diagnostics/instrumentation):
        # refinement steps taken and the certified worst-column ratio.
        self.last_refinement_steps = 0
        self.last_residual_ratio: float | None = None
        self._cudss: Any | None = None
        self._handle: int | None = None
        self._config: int | None = None
        self._data: int | None = None
        self._matrix: int | None = None
        self._row_offsets: Any | None = None
        self._col_indices: Any | None = None
        self._values: Any | None = None
        self._pattern: tuple[int, int, int, int, int, int, int] | None = None
        self._pattern_key: object | None = None
        # Cached full-CSR → lower-triangle map for the symmetric route, keyed on
        # the pattern signature + (n, nnz) so a grown pattern recompiles.
        self._lower: CompiledLowerTriangle | None = None
        self._lower_key: tuple[object, int, int] | None = None
        # Persistent dense RHS/solution descriptors and their backing device
        # buffers, reused across solves (and the factor phase) so each solve is a
        # buffer copy + execute instead of create/destroy of cuDSS dn matrices.
        self._rhs_matrix: int | None = None
        self._sol_matrix: int | None = None
        self._b_buf: Any | None = None
        self._x_buf: Any | None = None
        self._dn_shape: tuple[int, int] | None = None
        self._xp: Namespace | None = None
        self._dtype: Any | None = None
        self._inertia: tuple[int, int, int] | None = None
        # Lazy best-effort inertia state: whether the current factor is a
        # symmetric LDLᵀ (inertia defined), its size, and whether the read was
        # already attempted (so ``inertia_or_none`` syncs the host at most once).
        self._symmetric_factor = False
        self._n_factored = 0
        self._inertia_read = False
        self._device_id: int | None = None
        # Failure certificate of the current factorization, settled lazily at
        # its first solve so a factor pays no host sync of its own: the
        # device-side "all values finite" flag, the full matrix the residual
        # of a perturbed-pivot solve is measured against, and cuDSS's count of
        # perturbed pivots (``None`` until read).
        self._values_finite: Any | None = None
        # The factored matrix (read by the S2MPJ runner's certificate
        # instrumentation) and its operator (the refinement's residual
        # products, with the operator's cached longest row).
        self._residual_matrix: Any | None = None
        self._residual_operator: Any | None = None
        self._npivots: int | None = None

    def describe(self) -> str:
        return "cuDSS (GPU)"

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Release cuDSS resources owned by the solver."""
        cudss = self._cudss
        if cudss is None:
            return
        device_id = self._device_id

        def release() -> None:
            self._destroy_matrix()
            self._destroy_dense()
            if self._data is not None and self._handle is not None:
                cudss.data_destroy(self._handle, self._data)
                self._data = None
            if self._config is not None:
                cudss.config_destroy(self._config)
                self._config = None
            if self._handle is not None:
                cudss.destroy(self._handle)
                self._handle = None

        try:
            if device_id is None:
                release()
            else:
                with cupy.cuda.Device(device_id):
                    release()
        finally:
            self._cudss = None
            self._device_id = None
            self._xp = None
            self._dtype = None
            self._inertia = None
            self._symmetric_factor = False
            self._inertia_read = False
            self._values_finite = None
            self._residual_matrix = None
            self._residual_operator = None
            self._npivots = None

    def factor(self, K: LinearOperator) -> None:
        matrix, xp = _require_csr(K)
        assert isinstance(K, SparseOperator)  # narrowed by _require_csr
        device_id = _device_id(matrix)
        with cupy.cuda.Device(device_id):
            symmetric = K.is_symmetric()
            if self._require_inertia and not symmetric:
                raise ValueError("cuDSS inertia is defined only for symmetric matrices")

            self._ensure_context(device_id)
            self._bind_current_stream()
            cudss = self._cudss
            assert cudss is not None

            pattern_signature = K.coo_pattern_signature()
            factor_matrix = matrix
            matrix_type = int(cudss.MatrixType.GENERAL)
            matrix_view = int(cudss.MatrixViewType.FULL)
            if symmetric:
                factor_matrix = self._lower_triangle_csr(matrix, pattern_signature)
                matrix_type = int(cudss.MatrixType.SYMMETRIC)
                matrix_view = int(cudss.MatrixViewType.LOWER)

            self._xp = xp
            # Queued on device, read by the first solve (``_certify_factor``).
            self._values_finite = (
                cupy.isfinite(matrix.data).all() if matrix.nnz else None
            )
            self._residual_matrix = matrix
            self._residual_operator = K
            self._npivots = None
            self._factor_csr(
                factor_matrix,
                matrix_type,
                matrix_view,
                pattern_signature=pattern_signature,
            )
            # Under ``require_inertia`` read eagerly (implausible inertia must
            # fail the factor); otherwise defer to ``inertia_or_none`` so
            # callers that never consult it pay no per-factor host sync.
            self._symmetric_factor = symmetric
            self._n_factored = int(matrix.shape[0])
            self._inertia_read = self._require_inertia
            if self._require_inertia:
                # Already synchronizing for the inertia: settle the failure
                # certificate here too, so a non-finite matrix raises at
                # factor time (as on the CPU adapters) instead of reporting
                # a plausible inertia the δ_w ladder would act on.
                self._certify_factor()
            self._inertia = (
                self._read_inertia(self._n_factored) if self._require_inertia else None
            )

    def solve(self, rhs: Array) -> Array:
        if (
            self._cudss is None
            or self._handle is None
            or self._config is None
            or self._data is None
            or self._matrix is None
            or self._xp is None
            or self._dtype is None
            or self._device_id is None
        ):
            raise RuntimeError("factor() must be called before solve()")
        with cupy.cuda.Device(self._device_id):
            self._bind_current_stream()
            perturbed = self._certify_factor() > 0

            b = cupy.asarray(_to_cupy(rhs), dtype=self._dtype)
            was_vector = len(b.shape) == 1
            if was_vector:
                b = cupy.reshape(b, (int(b.shape[0]), 1))
            x = self._solve_columns(b)
            if perturbed:
                x = self._certify_perturbed_solve(b, x)
            result = x[:, 0] if was_vector else x
            return cast("Array", to_xp_array(result, self._xp))

    def _solve_columns(self, b: Any) -> Any:
        """One cuDSS SOLVE phase for an ``(n, k)`` right-hand side (a copy out)."""
        cudss = self._cudss
        assert cudss is not None
        n, ncols = int(b.shape[0]), int(b.shape[1])
        # Reuse the persistent dense descriptors/buffers, then copy this RHS
        # into the bound buffer — no per-solve cuDSS dn create/destroy.
        self._ensure_dense(n, ncols)
        assert self._b_buf is not None and self._x_buf is not None
        self._b_buf[...] = b
        self._guard(
            cudss.execute,
            self._handle,
            int(cudss.Phase.SOLVE),
            self._config,
            self._data,
            self._matrix,
            self._sol_matrix,
            self._rhs_matrix,
        )
        # Copy out of the reused solution buffer before the next solve
        # overwrites it (the buffer outlives this call).
        return self._x_buf.copy()

    def _certify_factor(self) -> int:
        """Settle the current factorization's failure certificate (once).

        Raises :class:`LinearSolveError` for a matrix with non-finite entries
        (see :func:`_reject_non_finite`) and returns the number of pivots
        cuDSS perturbed. Deferred from ``factor`` to the first solve (unless
        ``require_inertia`` already synchronizes the factor) so the
        factorization itself stays free of host syncs. It costs two host
        reads once per factorization — the finiteness flag and the
        ``NPIVOTS`` query — taken while the stream is already drained by the
        solve the IPM synchronizes on right after (its finiteness check).
        """
        if self._npivots is None:
            finite = self._values_finite
            if finite is not None and not bool(finite):
                raise LinearSolveError(
                    "sparse KKT matrix has non-finite entries (inf/NaN); the "
                    "upstream derivatives overflowed at this iterate"
                )
            self._npivots = self._read_npivots()
        return self._npivots

    def _certify_perturbed_solve(self, b: Any, x: Any) -> Any:
        """Refine a solve on a perturbed-pivot factor, then certify it.

        Returns the refined solution; raises :class:`LinearSolveError` unless
        it solves ``A x = b``.

        cuDSS's static pivoting replaces a pivot below ``PIVOT_EPSILON`` by
        ``±ε`` and so factors ``A + E`` instead of ``A`` (the GESP strategy;
        Li & Demmel 2003, ACM TOMS 29(2)). When ``A`` is merely
        ill-conditioned, or the tiny pivot is an artefact of the ordering (a
        zero diagonal a 2×2 block would pair off — the minimax epigraph
        variable of a MAKELA-type KKT system), the unrefined solve is an
        inexact solve of ``A x = b`` — residuals of 1e-5..1e-2 on the CuPy
        corpus sweep, enough to stall the IPM — and iterative refinement
        against ``A`` recovers it: ``x ← x + (A+E)⁻¹ (b − A x)`` (Higham
        2002, ch. 12), stopped by IPOPT's rules (see
        ``_PERTURBED_MAX_REFINEMENT_STEPS``), each column keeping its best
        iterate.

        But when ``A x = b`` has no solution — ``A`` singular, ``b`` outside
        its range — the ``+ε`` pivot turns the missing component into an
        ``O(‖b‖/ε)`` spike along the null space: a finite, meaningless vector
        (the ``±1e13`` of a 2×2 singular system), with a plausible inertia.
        Its residual is the component of ``b`` outside ``range(A)``, which no
        solution and no refinement step can reduce, so the solve fails the
        inexact-Newton condition ``‖b − A x‖ ≤ η ‖b‖`` (Dembo, Eisenstat &
        Steihaug 1982) for any useful forcing term: rejected at ``η =
        perturbed_max_residual`` (default ``_PERTURBED_SOLVE_MAX_RESIDUAL``),
        judged on the refined residual. The rejection is the numerical
        failure it is — the δ_w/δ_c ladder then regularizes the system
        (Wächter & Biegler 2006, §3.1), as on the CPU when SuperLU raises or
        Feral reports a zero pivot. Each column of an ``(n, k)`` right-hand
        side is certified and keeps its best iterate on its own (a small
        inconsistent column must not hide behind a large consistent one);
        the loop stops when the worst column stops improving.

        Cost, only on perturbed factorizations: one SpMV and one host sync
        (the ratio read that decides whether to refine, which is also the
        certificate), plus one cuDSS SOLVE phase, one SpMV and one host sync
        per refinement step taken — typically 1–3, at most
        ``max_refinement_steps``. The per-step sync is the early exit: a
        fixed-count refinement without it runs every step even when the
        first one reached the target, and cannot stop where
        ``‖(A+E)⁻¹E‖ > 1`` makes it diverge.
        """
        self.last_refinement_steps = 0
        self.last_residual_ratio = None
        operator = self._residual_operator
        assert operator is not None
        ncols = int(b.shape[1])
        if ncols == 0:
            return x
        b_norm = cupy.max(cupy.abs(b), axis=0)
        b_nonzero = b_norm > 0
        b_safe = cupy.where(b_nonzero, b_norm, 1.0)
        target = max(
            _PERTURBED_REFINEMENT_TOL,
            _PERTURBED_REFINEMENT_EPS_FACTOR * float(cupy.finfo(self._dtype).eps),
        )

        def measure(x_: Any) -> tuple[Any, Any, Any]:
            """Residual and per-column (certified, IPOPT-stopping) ratios."""
            r_ = b - operator._product(x_)  # its cached longest row
            r_norm = cupy.max(cupy.abs(r_), axis=0)
            # Certified ratio ‖r‖∞/‖b‖∞; a zero column of b has ratio 0 only
            # when solved exactly. NaN counts as ∞ (never certified).
            cert = cupy.where(
                b_nonzero, r_norm / b_safe, cupy.where(r_norm == 0, 0.0, cupy.inf)
            )
            cert = cupy.where(cupy.isnan(cert), cupy.inf, cert)
            # IPOPT's ComputeResidualRatio: ‖r‖/(min(‖x‖, max_cond ‖b‖) + ‖b‖).
            denom = (
                cupy.minimum(
                    cupy.max(cupy.abs(x_), axis=0), _IPOPT_RESIDUAL_MAX_COND * b_norm
                )
                + b_norm
            )
            stop = cupy.where(
                denom > 0, r_norm / cupy.where(denom > 0, denom, 1.0), r_norm
            )
            stop = cupy.where(cupy.isnan(stop), cupy.inf, stop)
            return r_, cert, stop

        def worst(cert: Any, stop: Any) -> tuple[float, float]:
            """Worst-column ratios, both in one host sync."""
            pair = cupy.asnumpy(cupy.stack((cupy.max(cert), cupy.max(stop))))
            return float(pair[0]), float(pair[1])

        r, cert, stop = measure(x)
        worst_cert, worst_stop = worst(cert, stop)
        steps = 0
        while (
            steps < self._max_refinement_steps
            and worst_stop > target
            and worst_cert < math.inf
        ):
            x_new = x + self._solve_columns(r)
            r_new, cert_new, stop_new = measure(x_new)
            steps += 1
            if ncols == 1:
                # The dominant vector case: compare on the host after the sync.
                new_cert, new_stop = worst(cert_new, stop_new)
                if new_cert < worst_cert:
                    x, r = x_new, r_new
            else:
                # Keep each column's best iterate (refinement diverges where
                # ‖(A+E)⁻¹E‖ > 1, and then the unrefined column is better).
                better = cert_new < cert
                x = cupy.where(better, x_new, x)
                r = cupy.where(better, r_new, r)
                cert = cupy.where(better, cert_new, cert)
                stop = cupy.where(better, stop_new, stop)
                new_cert, new_stop = worst(cert, stop)
            improved = new_cert < _PERTURBED_REFINEMENT_IMPROVEMENT * worst_cert
            if new_cert < worst_cert:
                worst_cert, worst_stop = new_cert, new_stop
            if not improved:
                break
        self.last_refinement_steps = steps
        self.last_residual_ratio = worst_cert
        if not worst_cert <= self._perturbed_max_residual:
            raise LinearSolveError(
                f"cuDSS perturbed {self._npivots} pivot(s) and the solve does not "
                "satisfy the system: it is singular for this right-hand side "
                f"(relative residual {worst_cert:.1e} after {steps} refinement "
                "step(s))"
            )
        return x

    @property
    def inertia(self) -> tuple[int, int, int]:
        """The factored operator's inertia ``(n_pos, n_neg, n_zero)``."""
        if self._inertia is None:
            raise RuntimeError(
                "inertia is unavailable: create the solver with "
                "require_inertia=True and call factor() first"
            )
        return self._inertia

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """Best-effort LDLᵀ inertia of the current symmetric factor.

        Parity with the Feral CPU adapter: inertia is a free by-product of the
        symmetric factorization and the IPM's inertia-guided δ_w correction
        depends on it, so it is reported even without ``require_inertia``. The
        ``data_get`` is a host sync, so it runs lazily and at most once per
        factor; a GENERAL (non-symmetric) factor has no defined inertia and
        reports ``None``.

        A factorization with perturbed pivots (``NPIVOTS > 0``) reports
        cuDSS's count too, although that is the inertia of ``A + E``: cuDSS
        counts each perturbed pivot by the sign of the ``±ε`` it substituted
        (its ``DIAG`` shows them as ``±1e-13``). On 30 captured perturbed
        CUTEst KKT systems the count disagreed with a dense Bunch–Kaufman
        LDLᵀ on 5 — zero pivots counted positive (``CATENARY``) and flipped
        signs (``KIRBY2LS``, ``ACOPR30``). Withholding it (``None`` whenever
        ``NPIVOTS > 0``) was measured and rejected: on the S2MPJ corpus it
        lost 9 ``exact/sparse`` rows (no fixes), where the rejected
        perturbed factors were genuinely indefinite and the δ_w correction
        was needed (Wächter & Biegler 2006, §3.1), and was a wash on
        ``lbfgs/sparse`` (5 fixed, 6 broken, confirmed by lone re-runs).
        Counting the perturbed pivots
        as zero is worse still: Bunch–Kaufman finds them nonzero on 27 of
        the 30 systems. See the S2MPJ benchmark report (2026-10-01).
        """
        if self._inertia is not None:
            return self._inertia
        if (
            self._inertia_read
            or not self._symmetric_factor
            or self._data is None
            or self._device_id is None
        ):
            return None
        self._inertia_read = True
        with cupy.cuda.Device(self._device_id):
            try:
                self._inertia = self._read_inertia(self._n_factored)
            except LinearSolveError:
                self._inertia = None
        return self._inertia

    def _guard(self, fn: Any, *args: Any) -> Any:
        """Call an nvmath cuDSS function, mapping ``cuDSSError`` failures."""
        assert self._cudss is not None
        try:
            return fn(*args)
        except self._cudss.cuDSSError as exc:
            raise LinearSolveError(f"cuDSS {fn.__name__} failed: {exc}") from exc

    def _runtime_version(self) -> str:
        """Best-effort ``major.minor.patch`` of the loaded cuDSS runtime."""
        get_property = getattr(self._cudss, "get_property", None)
        if get_property is None:
            return "unknown"
        try:
            # libraryPropertyType: MAJOR_VERSION=0, MINOR_VERSION=1, PATCH_LEVEL=2.
            return ".".join(str(int(get_property(i))) for i in (0, 1, 2))
        except Exception:
            return "unknown"

    def _guard_create(self, fn: Any, *args: Any) -> Any:
        """Like :meth:`_guard`, but for *matrix creation* calls.

        Creation rejections (INVALID_VALUE / NOT_SUPPORTED) carry no numerical
        content — the classic cause is a cuDSS runtime that does not match the
        nvmath-python bindings (observed: cuDSS 0.8 under nvmath 0.9, which was
        built against 0.7). They are raised as :class:`_CuDSSUnavailableError`
        so the dispatching solver degrades to spsolve instead of feeding an
        unfixable failure into the driver's regularization ladder.
        """
        assert self._cudss is not None
        try:
            return fn(*args)
        except self._cudss.cuDSSError as exc:
            status = getattr(exc, "status", None)
            if status is not None and int(status) in (
                _CUDSS_STATUS_INVALID_VALUE,
                _CUDSS_STATUS_NOT_SUPPORTED,
            ):
                raise _CuDSSUnavailableError(
                    f"cuDSS {fn.__name__} rejected its arguments ({exc}); the "
                    f"loaded cuDSS runtime (version {self._runtime_version()}) "
                    "likely does not match the version this nvmath-python "
                    "release was built against — install the matching "
                    "nvidia-cudss wheel"
                ) from exc
            raise LinearSolveError(f"cuDSS {fn.__name__} failed: {exc}") from exc

    def _ensure_context(self, device_id: int) -> None:
        if self._cudss is not None and self._device_id == device_id:
            return
        if self._cudss is not None:
            self.close()
        cudss = _load_cudss()
        self._cudss = cudss
        self._device_id = device_id
        try:
            self._handle = self._guard(cudss.create)
            self._config = self._guard(cudss.config_create)
            self._data = self._guard(cudss.data_create, self._handle)
        except Exception as exc:
            self.close()
            if _is_cudss_load_error(exc):
                raise _CuDSSUnavailableError(
                    "the user-managed NVIDIA cuDSS runtime could not be loaded"
                ) from exc
            raise
        if self._deterministic:
            self._set_deterministic_mode()

    def _set_deterministic_mode(self) -> None:
        """Set ``DETERMINISTIC_MODE = 1`` on the config (before any execute).

        A config parameter is read by the analysis/factorization phases that
        run with the config, so it is set right after ``config_create``. The
        value is a 32-bit int (``cudssConfigSet`` takes it by address and
        size). A binding without the parameter, or a runtime rejecting it,
        cannot honour the request: warn and keep the default mode.
        """
        cudss = self._cudss
        param = getattr(getattr(cudss, "ConfigParam", None), "DETERMINISTIC_MODE", None)
        reason = "the nvmath cuDSS bindings have no DETERMINISTIC_MODE parameter"
        if cudss is not None and param is not None:
            value = ctypes.c_int32(1)
            try:
                self._guard(
                    cudss.config_set,
                    self._config,
                    int(param),
                    ctypes.addressof(value),
                    ctypes.sizeof(value),
                )
                return
            # A mapped cuDSSError, or a binding whose config_set signature
            # differs (TypeError/ValueError): degrade, do not fail the factor.
            except (LinearSolveError, TypeError, ValueError) as exc:
                reason = str(exc)
        warnings.warn(
            "deterministic cuDSS factorizations were requested but are "
            f"unavailable ({reason}; cuDSS {self._runtime_version()}); "
            "factoring in the default, non-reproducible mode",
            RuntimeWarning,
            stacklevel=3,
        )

    def _bind_current_stream(self) -> None:
        """Run cuDSS on CuPy's current stream for correct dependency ordering."""
        if self._cudss is None or self._handle is None:
            raise RuntimeError("cuDSS context was not initialized")
        stream = cupy.cuda.get_current_stream()
        self._guard(self._cudss.set_stream, self._handle, int(stream.ptr))

    def _lower_triangle_csr(self, matrix: Any, signature: object | None) -> Any:
        """Lower triangle of a canonical symmetric CSR for the cuDSS LDLᵀ route.

        On a stable ``signature`` the lower-triangle structure is fixed, so the
        per-iteration ``tril(...)`` + re-canonicalization collapses to one cached
        boolean gather of the full canonical CSR data. ``None`` (value-dependent
        pattern) falls back to the from-scratch ``tril`` build.
        """
        n = int(matrix.shape[0])
        if signature is None:
            return _canonical_csr(cupyx.scipy.sparse.tril(matrix))
        key = (signature, n, int(matrix.nnz))
        if self._lower is None or self._lower_key != key:
            self._lower = compile_lower_triangle(cupy, matrix.indptr, matrix.indices, n)
            self._lower_key = key
        lower = self._lower
        data = lower.data(matrix.data)
        return cupyx.scipy.sparse.csr_matrix(
            (data, lower.indices, lower.indptr), shape=(n, n)
        )

    def _factor_csr(
        self,
        matrix: Any,
        matrix_type: int,
        matrix_view: int,
        *,
        pattern_signature: object | None,
    ) -> None:
        cudss = self._cudss
        if (
            cudss is None
            or self._handle is None
            or self._config is None
            or self._data is None
        ):
            raise RuntimeError("cuDSS context was not initialized")

        values = cupy.ascontiguousarray(matrix.data)
        # Optimization #3: int32 indices halve index bandwidth and speed up cuDSS
        # analysis/solve whenever the dimensions and nnz fit a signed 32-bit int.
        idx_dtype = index_dtype(
            cupy, int(matrix.shape[0]), int(matrix.shape[1]), int(matrix.nnz)
        )
        index_type = _index_type(cupy.dtype(idx_dtype))
        pattern = (
            int(matrix.shape[0]),
            int(matrix.shape[1]),
            int(matrix.nnz),
            matrix_type,
            matrix_view,
            index_type,
            _value_type(values.dtype),
        )

        self._dtype = values.dtype
        structure_key = (
            pattern_signature,
            matrix_type,
            matrix_view,
        )
        same_pattern = (
            pattern_signature is not None
            and pattern == self._pattern
            and structure_key == self._pattern_key
            and self._matrix is not None
            and self._row_offsets is not None
            and self._col_indices is not None
        )
        if same_pattern:
            # Deliberately no ``array_equal`` on CSR indices here: on CUDA that
            # comparison is device work plus a host sync. A non-None signature is
            # a contract that row/column coordinates are structural; value-
            # dependent sparse operators must return ``None`` and reanalyze.
            self._values = values
            self._guard(cudss.matrix_set_values, self._matrix, _ptr(values))
            phase = int(cudss.Phase.FACTORIZATION)
        else:
            row_offsets = cupy.ascontiguousarray(
                matrix.indptr.astype(idx_dtype, copy=False)
            )
            indices = cupy.ascontiguousarray(
                matrix.indices.astype(idx_dtype, copy=False)
            )
            self._destroy_matrix()
            self._row_offsets = row_offsets
            self._col_indices = indices
            self._values = values
            self._pattern = pattern
            self._pattern_key = structure_key if pattern_signature is not None else None
            self._matrix = self._create_csr_matrix(
                matrix,
                row_offsets,
                indices,
                values,
                matrix_type,
                matrix_view,
            )
            phase = int(cudss.Phase.ANALYSIS) | int(cudss.Phase.FACTORIZATION)

        # The analysis/factorization phases need RHS/solution descriptors bound,
        # but their contents are unused here — point them at the persistent dense
        # buffers (reused by solve), not a throwaway per-factor allocation.
        n = int(matrix.shape[0])
        self._ensure_dense(n, 1)
        self._guard(
            cudss.execute,
            self._handle,
            phase,
            self._config,
            self._data,
            self._matrix,
            self._sol_matrix,
            self._rhs_matrix,
        )

    def _ensure_dense(self, n: int, ncols: int) -> None:
        """Allocate (or reuse) the persistent dense RHS/solution descriptors.

        cuDSS dense matrices are reused across solves and the factor phase: the
        descriptors are bound once to fixed device buffers of shape ``(n, ncols)``
        and only rebuilt when that shape changes (e.g. a grown L-BFGS border
        changes the augmented system size).
        """
        if self._dn_shape == (n, ncols) and self._rhs_matrix is not None:
            return
        self._destroy_dense()
        # Fortran order to match the COL_MAJOR descriptors (cuDSS supports only
        # column-major dense matrices; ROW_MAJOR returns NOT_SUPPORTED).
        self._b_buf = cupy.zeros((n, ncols), dtype=self._dtype, order="F")
        self._x_buf = cupy.zeros((n, ncols), dtype=self._dtype, order="F")
        self._rhs_matrix = self._create_dense_matrix(n, ncols, self._b_buf)
        self._sol_matrix = self._create_dense_matrix(n, ncols, self._x_buf)
        self._dn_shape = (n, ncols)

    def _destroy_dense(self) -> None:
        if self._cudss is not None:
            if self._rhs_matrix is not None:
                self._cudss.matrix_destroy(self._rhs_matrix)
            if self._sol_matrix is not None:
                self._cudss.matrix_destroy(self._sol_matrix)
        self._rhs_matrix = None
        self._sol_matrix = None
        self._b_buf = None
        self._x_buf = None
        self._dn_shape = None

    def _create_csr_matrix(
        self,
        matrix: Any,
        row_offsets: Any,
        indices: Any,
        values: Any,
        matrix_type: int,
        matrix_view: int,
    ) -> int:
        cudss = self._cudss
        if cudss is None:
            raise RuntimeError("cuDSS context was not initialized")
        # cuDSS shares one index type for row offsets and column indices; both
        # carry the same dtype chosen in ``_factor_csr`` (int32 when the system
        # fits a signed 32-bit int, else int64).
        return cast(
            int,
            self._guard_create(
                cudss.matrix_create_csr,
                int(matrix.shape[0]),
                int(matrix.shape[1]),
                int(matrix.nnz),
                _ptr(row_offsets),
                0,
                _ptr(indices),
                _ptr(values),
                _index_type(indices.dtype),
                _value_type(values.dtype),
                matrix_type,
                matrix_view,
                int(cudss.IndexBase.ZERO),
            ),
        )

    def _create_dense_matrix(self, nrows: int, ncols: int, values: Any) -> int:
        cudss = self._cudss
        if cudss is None:
            raise RuntimeError("cuDSS context was not initialized")
        # cuDSS supports only column-major dense matrices (cudssMatrixCreateDn
        # returns NOT_SUPPORTED for ROW_MAJOR), so ld is the row count and the
        # backing buffers are allocated Fortran-order in _ensure_dense.
        return cast(
            int,
            self._guard_create(
                cudss.matrix_create_dn,
                int(nrows),
                int(ncols),
                int(nrows),
                _ptr(values),
                _value_type(values.dtype),
                int(cudss.Layout.COL_MAJOR),
            ),
        )

    def _read_inertia(self, n: int) -> tuple[int, int, int]:
        cudss = self._cudss
        if cudss is None or self._handle is None or self._data is None:
            raise RuntimeError("cuDSS context was not initialized")
        # cuDSS validates the output-buffer size exactly and (observed on 0.7)
        # fills INERTIA as int32[2], rejecting an int64[2] buffer with
        # INVALID_VALUE — so try the known layout first, then the widened one
        # for runtimes that grow it.
        last_error: LinearSolveError | None = None
        for count_type in (ctypes.c_int32, ctypes.c_int64):
            counts = (count_type * 2)()
            written = ctypes.c_size_t()
            try:
                self._guard(
                    cudss.data_get,
                    self._handle,
                    self._data,
                    int(cudss.DataParam.INERTIA),
                    ctypes.addressof(counts),
                    ctypes.sizeof(counts),
                    ctypes.addressof(written),
                )
            except LinearSolveError as exc:
                last_error = exc
                continue
            pos = int(counts[0])
            neg = int(counts[1])
            zero = n - pos - neg
            if pos < 0 or neg < 0 or zero < 0:
                raise LinearSolveError("cuDSS returned an implausible inertia")
            return pos, neg, zero
        assert last_error is not None
        raise last_error

    def _read_npivots(self) -> int:
        """Pivots cuDSS perturbed in the current factorization (``NPIVOTS``).

        ``0`` when the runtime does not report it, which leaves the solve
        uncertified exactly as before this check existed. Same int32-then-
        int64 buffer probing as :meth:`_read_inertia`.
        """
        cudss = self._cudss
        param = getattr(getattr(cudss, "DataParam", None), "NPIVOTS", None)
        if cudss is None or param is None or self._handle is None or self._data is None:
            return 0
        for count_type in (ctypes.c_int32, ctypes.c_int64):
            count = count_type()
            written = ctypes.c_size_t()
            try:
                self._guard(
                    cudss.data_get,
                    self._handle,
                    self._data,
                    int(param),
                    ctypes.addressof(count),
                    ctypes.sizeof(count),
                    ctypes.addressof(written),
                )
            except LinearSolveError:
                continue
            return max(0, int(count.value))
        return 0

    def _destroy_matrix(self) -> None:
        if self._cudss is not None and self._matrix is not None:
            self._cudss.matrix_destroy(self._matrix)
        self._matrix = None
        self._row_offsets = None
        self._col_indices = None
        self._values = None
        self._pattern = None
        self._pattern_key = None


class CuPySparseSolver:
    """CUDA sparse-direct solver preferring cuDSS and falling back to spsolve."""

    def __init__(
        self,
        *,
        require_inertia: bool = False,
        prefer_cudss: bool = True,
        max_refinement_steps: int | None = None,
        perturbed_max_residual: float | None = None,
        deterministic: bool = False,
    ) -> None:
        self._require_inertia = require_inertia
        self._prefer_cudss = prefer_cudss
        # Validated up front and forwarded to the cuDSS solver (see
        # :class:`CuDSSSparseSolver`); unused by the spsolve fallback.
        self._max_refinement_steps, self._perturbed_max_residual = (
            _resolve_refinement_policy(max_refinement_steps, perturbed_max_residual)
        )
        self._deterministic = _resolve_deterministic(deterministic)
        self._inner: CuDSSSparseSolver | CuPySpsolveSolver | None = None
        self._cudss_unavailable = False

    def describe(self) -> str:
        return self._inner.describe() if self._inner is not None else "CuPy (GPU)"

    def factor(self, K: LinearOperator) -> None:
        if self._prefer_cudss and not self._cudss_unavailable:
            if not isinstance(self._inner, CuDSSSparseSolver):
                self._inner = CuDSSSparseSolver(
                    require_inertia=self._require_inertia,
                    max_refinement_steps=self._max_refinement_steps,
                    perturbed_max_residual=self._perturbed_max_residual,
                    deterministic=self._deterministic,
                )
            try:
                self._inner.factor(K)
            except ImportError as exc:
                self._cudss_unavailable = True
                self._inner = None
                if self._require_inertia:
                    raise NotImplementedError(
                        "GPU sparse inertia requires a user-managed NVIDIA cuDSS "
                        "runtime compatible with the installed CUDA toolkit"
                    ) from exc
                # Loud on purpose: spsolve re-factorizes from scratch on every
                # solve (no symbolic reuse), typically orders of magnitude
                # slower per IPM iteration — a silent downgrade here has cost
                # real users long, unexplained runs.
                warnings.warn(
                    f"cuDSS is unavailable ({exc}); falling back to CuPy "
                    "spsolve, which re-factorizes on every solve",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                return

        if not isinstance(self._inner, CuPySpsolveSolver):
            self._inner = CuPySpsolveSolver(require_inertia=self._require_inertia)
        self._inner.factor(K)

    def solve(self, rhs: Array) -> Array:
        if self._inner is None:
            raise RuntimeError("factor() must be called before solve()")
        return self._inner.solve(rhs)

    @property
    def inertia(self) -> tuple[int, int, int]:
        """The factored operator's inertia ``(n_pos, n_neg, n_zero)``."""
        if self._inner is None:
            raise RuntimeError("factor() must be called before reading inertia")
        return self._inner.inertia

    def inertia_or_none(self) -> tuple[int, int, int] | None:
        """Best-effort inertia of the dispatched inner solver; ``None`` if absent."""
        if self._inner is None:
            return None
        return self._inner.inertia_or_none()


class CuPySparseAdapter:
    """Factory pairing CuPy COO assembly with CUDA sparse-direct solvers.

    The adapter persists across an interior-point solve (the facade caches one
    instance), so it memoizes the COO→canonical-CSR map for the most recent
    ``pattern_signature``. On a fixed KKT pattern this replaces the per-iteration
    device ``tocsr().sum_duplicates().sort_indices()`` with a single scatter-add
    (see :mod:`ipax.backend.sparse._canonical`).

    ``max_refinement_steps`` / ``perturbed_max_residual`` set the cuDSS
    solvers' perturbed-pivot policy (see :class:`CuDSSSparseSolver`); ``None``
    keeps the module defaults. :meth:`configure_perturbed_pivots` changes it
    after construction — the route by which
    ``SparseOptions.perturbed_pivot_refinement_steps`` /
    ``perturbed_pivot_max_residual`` reach this adapter. ``deterministic``
    likewise sets the cuDSS solvers' reproducible mode (default off), and
    :meth:`configure_deterministic` is how ``SparseOptions.deterministic``
    reaches it.
    """

    def __init__(
        self,
        *,
        max_refinement_steps: int | None = None,
        perturbed_max_residual: float | None = None,
        deterministic: bool = False,
    ) -> None:
        self._max_refinement_steps, self._perturbed_max_residual = (
            _resolve_refinement_policy(max_refinement_steps, perturbed_max_residual)
        )
        self._deterministic = _resolve_deterministic(deterministic)
        self._signature: object | None = None
        self._compiled: _SegmentSumMap | None = None
        self._meta: tuple[tuple[int, int], int] | None = None

    def configure_perturbed_pivots(
        self, *, refinement_steps: int | None, max_residual: float | None
    ) -> None:
        """Set the perturbed-pivot policy of the solvers made from now on.

        The optional adapter hook through which the ``SparseDirectSolver``
        facade forwards ``SparseOptions``' policy (called once, before
        :meth:`solver`); ``None`` takes the module default. Validated as in
        the constructor; a rejected policy leaves the current one in place.
        """
        self._max_refinement_steps, self._perturbed_max_residual = (
            _resolve_refinement_policy(refinement_steps, max_residual)
        )

    def configure_deterministic(self, *, enabled: bool) -> None:
        """Switch cuDSS's reproducible mode for the solvers made from now on.

        The optional adapter hook through which the ``SparseDirectSolver``
        facade forwards ``SparseOptions.deterministic`` (called once, before
        :meth:`solver`, and only when the option is on). A non-bool is
        rejected and leaves the current setting.
        """
        self._deterministic = _resolve_deterministic(enabled)

    def from_coo(
        self,
        rows: Array,
        cols: Array,
        values: Array,
        *,
        shape: tuple[int, int],
        symmetric: bool | None = None,
        pattern_signature: object | None = None,
    ) -> SparseOperator:
        """Build a :class:`SparseOperator` from Array-API COO triplets.

        ``symmetric`` is an optional structural hint from the assembler; ``None``
        leaves the operator to test symmetry numerically when first asked.
        ``pattern_signature`` keys the compiled-map cache (a stable signature
        reuses the canonical CSR structure and recomputes only values) and is
        also used by cuDSS to reuse its symbolic analysis without comparing
        device index arrays.
        """
        from ipax.backend.namespace import array_namespace

        xp = array_namespace(values)
        cp_values = _to_cupy(values)

        if pattern_signature is not None:
            csr = self._canonical_csr(
                _to_index(rows),
                _to_index(cols),
                cp_values,
                shape,
                pattern_signature,
                int(cp_values.shape[0]),
            )
            assert self._compiled is not None  # set by _canonical_csr
            return SparseOperator(
                csr,
                xp,
                symmetric=symmetric,
                pattern_signature=pattern_signature,
                canonical=True,
                structure=self._compiled.structure,
            )

        matrix = cupyx.scipy.sparse.coo_matrix(
            (cp_values, (_to_index(rows), _to_index(cols))),
            shape=shape,
        )
        return SparseOperator(
            matrix,
            xp,
            symmetric=symmetric,
            pattern_signature=pattern_signature,
        )

    def _canonical_csr(
        self,
        rows: Any,
        cols: Any,
        values: Any,
        shape: tuple[int, int],
        signature: object,
        n_triplets: int,
    ) -> Any:
        """Canonical CSR via the cached compiled map, recompiling on a new pattern.

        Keyed on ``(signature, shape, n_triplets)`` so a reused signature with a
        structurally different system (a grown L-BFGS border) recompiles rather
        than misapplying the stale map.
        """
        meta = (shape, n_triplets)
        if self._compiled is None or self._signature != signature or self._meta != meta:
            # Rows as the major axis ⇒ canonical CSR structure directly; the
            # duplicate sum is the map's ordered, reproducible one.
            self._compiled = _SegmentSumMap(rows, cols, shape)
            self._signature = signature
            self._meta = meta
        return self._compiled.csr(values)

    def solver(self, *, require_inertia: bool = False) -> CuPySparseSolver:
        """Return the CUDA sparse-direct solver (cuDSS default, spsolve fallback)."""
        return CuPySparseSolver(
            require_inertia=require_inertia,
            max_refinement_steps=self._max_refinement_steps,
            perturbed_max_residual=self._perturbed_max_residual,
            deterministic=self._deterministic,
        )


__all__ = [
    "CuDSSSparseSolver",
    "CuPySparseAdapter",
    "CuPySparseSolver",
    "CuPySpsolveSolver",
    "SparseOperator",
]
