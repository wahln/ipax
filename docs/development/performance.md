# Performance proposals

This page records **proposed development work**, not available solver options or
measured JIT speedups. The initial scope is L-BFGS with no equality or general
inequality constraints; finite lower and upper bounds are allowed. It also covers
the fully unbounded case. For current behavior, see the
[linear-algebra layer](../concepts/linalg.md) and
[routing hints](../guide/routing-hints.md).

For affine constraints, see the [linear-inequality review](linear-inequality-performance.md),
which covers bounds plus L-BFGS across all three solve routes.

The pure Array API implementation remains the reference and fallback. Any
backend-specific acceleration belongs in an explicitly labeled adapter, with
linear algebra injected through the existing operator/solver interfaces.

## Compact systems to accelerate

The L-BFGS representation and bound-only condensed system have the form

\[
W = \xi I - U M^{-1} U^T, \qquad
N = D - U M^{-1} U^T, \qquad
H = M - U^T D^{-1} U,
\]

where \(U=[\xi S\;Y]\), \(D=\xi I+\Sigma_x+\delta_w I\), and \(M\) and
\(H\) are only \(2k\times2k\) for a history of \(k\) curvature pairs.

Dense uses a structured Woodbury solve, default Krylov can apply that same exact
inverse with residual verification, and sparse factors the equivalent bordered
system. Current Woodbury caches retain the small coefficient matrix \(H\), not
its numerical LU factorization. Each `xp.linalg.solve` factors it again; the
Hessian application similarly solves with \(M\).

Although the damped Hessian and condensed system are positive definite, their
compact middle matrices are indefinite. They need a suitable pivoted solve;
Cholesky is not a replacement for these small solves.

## Optional backend adapters

### Reuse small factorizations

The first candidate is a reusable pivoted factor/solve adapter for the compact
matrices:

- Cache the factor of \(M\) per accepted L-BFGS generation.
- Cache the factor of \(H\) per condensed operator and L-BFGS generation, so
  changing bound diagonals or regularization invalidates the factor.
- Retain vector and matrix right-hand-side support and solve from the factor
  rather than storing an explicit inverse.

For NumPy, SciPy's `lu_factor` and `lu_solve` provide this separation. Repeated
small solves then cost \(O(k^2)\) after factorization, rather than refactoring in
\(O(k^3)\). This benefits dense and Krylov Woodbury applications, and all routes
through common L-BFGS Hessian applications.
[SciPy factor reuse](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_solve.html).

### Avoid unnecessary CUDA synchronization

PyTorch documents that `torch.linalg.solve` synchronizes CUDA with the CPU.
Evaluate `solve_ex` or reusable LU factors in an adapter, combining status reads
with an existing host decision where possible.
[PyTorch solve behavior](https://docs.pytorch.org/docs/main/generated/torch.linalg.solve.html).

Preserve existing failure behavior: a Hessian application with unusable compact
coefficients falls back to its seed; a failed Newton inverse must reach the
appropriate refinement, iterative fallback, or regularization path. Merely
suppressing factorization errors would change solver correctness. Resource
failures such as out-of-memory must still propagate.

### Reduce temporary arrays around BLAS operations

Evaluate GEMV/GEMM alpha/beta accumulation, reusable scratch buffers, and symmetric
rank-k products for the diagonal weighted-Gram blocks. The cross block still
needs a general product. These may avoid full-length multiply/add temporaries.

Measure layout conversion, packing, workspace, and float64 costs. Eagerly
concatenating the implicit \(U=[\xi S\;Y]\) can increase memory traffic even if
it reduces the number of BLAS calls. Existing SciPy and CuPy/cuBLAS facilities
may be sufficient without adding a dependency.

### NVIDIA Python tools and possible extras

`nvmath-python` is already part of `ipax[sparse-cuda]`, and the sparse adapter
uses its cuDSS bindings. That functionality is already implemented.

Further candidates include nvmath host BLAS planning and, later, `nvmath.device`
with `numba-cuda` for small on-device linear algebra. NVIDIA documents device
APIs for cuBLASDx and cuSOLVERDx as experimental. CUDA Python execution and
binding APIs can help manage streams and resources; they do not automatically
accelerate existing Array API calls.
[CUDA Python suite](https://nvidia.github.io/cuda-python/latest/),
[nvmath device APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/index.html).

If prototypes justify them, possible extras are a CPU small-factor group using
SciPy, a CPU JIT group using Numba, and an experimental CUDA JIT group using
`numba-cuda` plus the required NVIDIA runtimes. These are dependency candidates,
not currently declared extras. CUDA/runtime compatibility must be established
before selecting package constraints.

Adapter state must live in explicit objects and respect backend, device, dtype,
stream, and generation lifetimes. Preserve a pure Array API fallback and keep
new solve strategies out of `ipm/driver.py`.

## JIT compilation candidates

Start with pure numerical helpers. The outer solver includes callbacks,
data-dependent line-search decisions, scalar conversions, and mutable history;
compiling that whole object graph is a much larger project.

| Technology | Initial target | Main limitations |
|---|---|---|
| NumPy/Numba | Fused bound arithmetic, residual reductions, and fixed-array L-BFGS update kernels | Vendor BLAS already handles large products. Compile explicit array kernels; keep `fastmath=False` initially to preserve non-finite safeguards and reduction semantics. |
| `torch.compile` | Pure tensor helpers for bound reductions and compact setup/apply | Scalar reads and data-dependent Python control can break graphs. Changing history shapes and Python scalar parameters such as `xi` can cause recompilation. |
| `cupy.fuse` | Bound gaps, diagonal construction, dual recovery, and simple reductions | Fusion covers supported elementwise/reduction expressions, not the entire Woodbury BLAS/solve sequence. Reuse decorated functions across calls. |
| JAX JIT | Pure compact and bound kernels with explicit state | Traced values cannot drive ordinary Python branching. Use stable shapes and structured control flow for decisions inside compiled kernels. |
| `numba-cuda` and NVIDIA device math | Fused GPU pointwise/reduction kernels, then a small compact-solve prototype | Launch and resource costs can outweigh the small arithmetic workload; numerical validation and maintenance costs are higher. |

These are applicability judgments based on the implementation and documented
compiler boundaries, not performance guarantees.
[Numba performance guidance](https://numba.readthedocs.io/en/stable/user/performance-tips.html),
[PyTorch graph breaks](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html),
[CuPy fusion](https://docs.cupy.dev/en/stable/reference/generated/cupy.fuse.html),
[JAX control flow](https://docs.jax.dev/en/latest/201/control-flow.html).

NVIDIA cuTile is another exploratory kernel-development option. It does not
change the project's scope guardrail against custom mixed-precision tiled BLAS
kernels without discussion.
[NVIDIA cuTile](https://developer.nvidia.com/cuda/tile).

Growing L-BFGS history can specialize once per window length or use a validated
fixed-size representation. Naively zero-padding an indefinite compact matrix
can make the solve singular. Cache compiled functions outside iteration loops;
keep changing numerical parameters as array inputs where that avoids repeated
specialization. Objective/gradient callback compilation is a separate benchmark
from solver-kernel compilation.

## Validation before adoption

1. Benchmark the current pure implementation and candidate on identical inputs,
   separating factor/setup, repeated solve, L-BFGS update, and full IPM time.
2. Record cold compilation, warm execution, memory use, kernel launches, and
   synchronization. Fewer Python scalar reads alone do not establish a speedup.
3. Check actual linear residuals and final scaled KKT accuracy on well-conditioned,
   ill-conditioned, and nonconvex problems, including active bounds and rejected
   or degenerate curvature pairs. Retain singular/non-finite fallback tests.
4. Pass relevant contracts and NumPy/PyTorch/strict tests for portable changes;
   test adapters on their actual target devices, including missing-extra fallback.
5. Add an optional dependency or enable an acceleration only after repeatable
   gains outweigh setup, numerical, and maintenance costs.

The repository's `benchmarks/runners/lbfgs_bound_only.py` measures compact setup
and solve costs for this problem family. The general device-efficiency runner
uses exact Hessians by default, so its default workload is not a substitute for
bound-only L-BFGS measurements.
