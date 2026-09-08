# Linear inequalities with bounds and L-BFGS

This review covers dense, Krylov, and sparse solves for an objective with affine
inequalities and variable bounds, using the current compact L-BFGS Hessian.
There are no equality or nonlinear constraints in this scope. Sections 1 and 2
separate implemented changes from further portable candidates; sections 3 and 4
are **suggestions only**, with no new options or dependencies enabled.
For the complementary bound-only analysis, see [Performance proposals](performance.md).

## Current execution and costs

Write the lowered inequalities as \(g(x)=Ax-b\leq0\), with slacks \(s>0\),
multipliers \(\lambda>0\), and \(\Sigma_s=\operatorname{diag}(\lambda/s)\).
With \(k\) L-BFGS pairs, the condensed Newton operator is

\[
N=D-UM^{-1}U^T+A^T\Sigma_s A,\qquad
D=\xi I+\Sigma_x+\delta_w I,\qquad U=[\xi S\;Y].
\]

The bound-only Woodbury inverse ceases to be exact when the inequality Gram
term has off-diagonal entries. The implementation conservatively disables its
exact-inverse shortcut whenever inequality rows are present.

| Route | Current behavior with inequalities and L-BFGS | Main cost to measure |
|---|---|---|
| Dense | Materializes the Hessian and the condensed \(n\times n\) matrix. Sparse Jacobians can supply a Gram without densifying \(A\); a plain `Dense` Jacobian uses the materialization fallback. | Gram formation, \(O(n^2)\) storage, and dense factorization. The L-BFGS `primal_block()` is absent, so the PD guard never probes; before this review each `xp.linalg.solve` factored the dense matrix again (see the Cholesky reuse below). |
| Krylov | Applies \(Nv\) using \(Av\), diagonal weighting, \(A^T\), and the compact Hessian. Default Jacobi uses the exact diagonal; the optional L-BFGS preconditioner replaces the inequality Gram by its diagonal. | Two Jacobian products per iteration, compact solves, and scalar reductions. Strong constraint coupling can defeat both preconditioners. |
| Sparse augmented | Keeps \(A\) explicit in a border with \(-\Sigma_s^{-1}\) and keeps the compact Hessian border. For \(k\) accepted pairs the assembled size is \(n+m+2k\). | Numeric factorization, fill, structure/value assembly, and host/device transfers. |
| Sparse normal equations | Forms the sparse Gram and retains the compact Hessian border, giving size \(n+2k\). | Whether \(A^TA\) stays sparse. Localized rows can favor this route; sparse but widely overlapping rows can produce a dense Gram. |

`factor()` on `DenseSolver` prepares state; materialization and numerical solving
are lazy. Repeated predictor/corrector right-hand sides therefore deserve their
own timing. The sparse adapters already support factor/solve separation and
pattern reuse; those are existing capabilities, not new proposals.

Two representation details matter before tuning kernels:

- `linear_ineq=(A, lower, upper)` currently accepts concrete dense arrays only.
  The lowering rejects `LinearOperator` inputs, despite the broader annotation.
  Sparse or matrix-free affine inequalities can currently be expressed through
  `ineq_constraints(x)=A.matvec(x)-b` and a constant `ineq_jacobian` callback.
- Two finite row bounds produce separate signed rows when dense inequalities
  are lowered. Choosing `linsolve="sparse"` does not make a dense `A` sparse:
  `Dense.to_coo()` emits all entries, including numerical zeros. Use a structural
  sparse operator through the callbacks when its sparsity is important.

Auto-selection also uses size, aspect ratio, and available Gram/fill metadata.
The measurements below do not establish new crossover thresholds.

## 1. Portable Array API refactors

**Implemented: owned temporary reuse for `Dense.gram_diagonal`.** Compute
`weighted = weights[:, None] * A`, then `weighted *= A`, and reduce columns.
The temporary belongs to the operation; neither the Jacobian nor its weights
are mutated. Mutable backends can reuse one \(m\times n\) buffer in place of
two. Python augmented assignment also permits an immutable backend to return a
new array, preserving the result without promising that backend a memory saving.
No squared-Jacobian cache is retained across calls. `Dense.row_gram_diagonal`
(the equality-saddle Schur diagonal) mirrors the same order and buffer reuse.

Multiplication order is deliberate. For float64, \(A_{ij}=10^{200}\) and
\(w_i=10^{-200}\) give a finite weighted square, while squaring \(A\) first
overflows. An exploratory `(A*A).T @ weights` was faster but rejected for this
reason. A `vecdot` replacement was slower in the initial CPU and CUDA tests.

**Implemented: preserve batched adjoints through row scaling.** `_RowScaled`
now evaluates \(J^T(DV)\) through the inner `rmatmat`, instead of falling back
to one `rmatvec` and one column extraction per right-hand side, then stacking.
This supports batched condensed-operator applications, including matrix RHS
refinement, and fixes empty RHS batches when the inner operator supports them.
Ordinary vector Newton solves do not gain from this change.

**Implemented: Cholesky reuse for the PD-by-construction L-BFGS block.** The
dense route materializes \(N\) whenever an inequality Gram term rules out the
Woodbury structured solve, but `primal_block()` is `None` for L-BFGS, so the
PD guard (whose Cholesky factor the condensed route already reuses) never ran
and every right-hand side paid a fresh `xp.linalg.solve` LU. The operator now
declares `positive_definite_hint()` — `True` for the Powell-damped compact
L-BFGS Hessian and the condensed block built on it — and `DenseSolver` takes a
Cholesky of the materialized \(N\) on that declaration alone (half the LU
flops), keeping the factor for the corrector/SOC back-solves through the
existing `get_dense_cholesky_solve` gap-filler. It is a reuse optimization,
not a guard: a numerical Cholesky failure falls back to the LU path unchanged,
and backends without the back-substitution primitive (array-api-strict, JAX)
skip the factorization rather than paying for a factor they cannot apply.
Explicit-Hessian blocks still go through the probing guard exactly as before.
A breakdown is bookkept the way the mixed-precision route bookkeeps its
failures: `DenseOptions.pd_hint_failure_limit` *consecutive* breakdowns stop
the attempts for the rest of the solver's life (a success resets the count, so
one ill-conditioned iterate does not forfeit the reuse everywhere else), and
`DenseSolver.describe()` carries a sticky `pd-hint->lu` marker once any
breakdown has occurred, so `Result.routes` never reports a clean `dense` run
when the hinted factorization fell back to LU. A breakdown under the
mixed-precision route is marked but counts toward *neither* kill switch: the
reduced matrix may be non-PD by precision noise alone — no evidence about the
exact block — and the refinement pass that follows the LU solve is the
certificate that judges it.

Further candidates require broader work and remain unimplemented:

- **Preserve affine provenance through lowering and scaling.** At fixed current
  multipliers, affine Jacobian terms cancel from the L-BFGS curvature difference:
  \(\gamma=\nabla f(x_{new})-\nabla f(x_{old})\). The driver currently evaluates
  both Lagrangian gradients. Avoiding the redundant adjoint product would benefit
  every solve route and avoid subtracting two large, equal dual contributions.
  This needs trusted provenance, including correct slicing for mixed constraints;
  object identity alone is insufficient for mutable nonlinear Jacobians.
- **Reuse affine line-search products.** Evaluate
  \(g(x+\alpha\Delta x)=g(x)+\alpha A\Delta x\) after one product per direction.
  Track the direction and point explicitly and invalidate on restoration or a
  changed direction. Near-active constraints need rounding/error tests before
  replacing callback evaluation throughout globalization.
- **Preserve structure through scaling.** `Dense` lacks the optional full `gram`
  hook, so scaled dense Jacobians materialize an extra scaled matrix. A full Gram
  hook would allow scaling to fold into weights and dense/sparse stacks to sum
  per-block Grams. It would also change tall-problem auto-selection and interact
  with `gram_dtype`/per-block precision hints, so it needs a separate routing and
  numerical review. `_RowScaled` also lacks `coo_values`; stable-pattern sparse
  assembly can consequently rebuild indices through `to_coo`. A solution must
  survive wrapper recreation and invalidate correctly when structure changes.
- **Lower sparse/operator row bounds without a dense copy.** Signed row selection
  could preserve the original operator and avoid duplicate two-sided storage.
  It must forward all product, Gram, and structural hooks to retain each route's
  capabilities; this is more than changing the input annotation.

## 2. Portable GPU refactors

The two retained changes use only Array API operations. Buffer reuse reduces
peak live storage on mutable GPU arrays; batching changes many small adjoint
products into the inner operator's matrix product. Neither change removes a
host synchronization, and no kernel-launch profiler was used to claim counts.

The largest remaining synchronization target is iterative linear solving.
CG converts curvature, residual norms, and preconditioned inner products to
Python scalars. These checks occur repeatedly when the inequality term makes
CG genuinely iterative. Some independent diagnostics could share a reduction or
host decision, but the recurrence has dependencies: positivity, non-finite, and
convergence checks cannot simply be deleted. Measure full iterations, not just
the number of `float(...)` calls. Sparse backend synchronization also needs
adapter-level profiling; a portable COO vector does not imply a transfer-free
factorization.

Avoid caching \(A^T\Sigma_s A\) merely because \(A\) is constant: slack weights
change every Newton step. Constant index patterns and numeric factors have
different invalidation requirements. Sparse adapters already cache useful
structure and, in some implementations, squared values for Gram diagonals.
Duplicating those caches in the core would add memory without establishing a gain.

## 3. Optional backend adapters

Priority candidates, all retaining the Array API reference fallback:

1. **Keep the compact middle systems on pivoted solvers.** The dense factor
   reuse above covers \(N\); the compact \(M\) and Woodbury middle systems are
   indefinite even though the Hessian is positive definite, so any adapter for
   them needs a pivoted (LU/Bunch–Kaufman) factorization, never a Cholesky.
   [SciPy LU factorization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_factor.html).
2. **Fuse weighted reductions and accumulate around BLAS.** A backend contraction
   or reduction kernel for \(\sum_i(w_iA_{ij})A_{ij}\) can eliminate the full
   weighted temporary while preserving multiplication order. GEMV/GEMM alpha/beta
   accumulation can combine the Hessian, bounds, and constraint contributions.
   For dense Gram formation, evaluate symmetric rank-k products of weighted rows,
   including square-root scaling, packing, and rounding costs in the comparison.
3. **Keep solve status and Krylov scalars on device where safe.** PyTorch's
   `solve_ex(check_errors=False)` avoids its documented CUDA error-check sync;
   a later status check must still feed existing failure/fallback handling.
   This is particularly relevant to compact solves inside repeated Krylov
   applications. [PyTorch solve_ex](https://docs.pytorch.org/docs/main/generated/torch.linalg.solve_ex.html).
4. **Exploit existing sparse planning before adding another layer.** Verify actual
   symbolic reuse with constant \(A\), changing \(\Sigma_s\), and a growing then
   full L-BFGS window. cuDSS through `nvmath-python` is already implemented in
   `ipax[sparse-cuda]`. NVIDIA's higher-level direct solver separates planning,
   factorization, and solve; investigate it only if profiling finds an uncovered
   lifecycle/workspace opportunity in the current bindings.
   [nvmath DirectSolver](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/sparse/generated/nvmath.sparse.advanced.DirectSolver.html).

Factors and workspaces belong to explicit adapter objects, keyed by device,
dtype, stream, operator generation, and relevant values. Never reuse a numeric
factor on a pattern match alone. Optional dependencies stay lazy and unavailable
adapters fall back cleanly.

## 4. JIT candidates and possible extras

| Technology | Constrained-route target | Boundary to respect |
|---|---|---|
| NumPy + Numba | Weighted column reductions, slack/bound arithmetic, residual assembly | Compile explicit array kernels, leaving BLAS to its library; start with `fastmath=False`. Choose loop order with memory layout in mind. |
| `torch.compile` | Pure tensor kernels for Gram diagonals, \(\Sigma_s\), RHS construction, and recovery | Scalar reads and callbacks break graphs. Full CG/IPM compilation needs explicit state/control flow; changing history lengths can trigger recompilation. |
| `cupy.fuse` | Elementwise slack/bound expressions and supported weighted reductions | Fusion does not compile the entire sparse factorization or BLAS/compact-solve chain. Cache decorated kernels outside iteration loops. |
| JAX JIT | Pure affine products and residual/diagonal kernels with explicit state | Fixed shapes and structured control flow are needed for traced decisions. Validate on a working JAX environment separately. |
| `numba-cuda` + NVIDIA device APIs | Fused reductions first; compact device solves only as a later prototype | Launch/resource cost, robust pivoting, and float64 behavior must be measured. Do not replace vendor sparse factorization with an unvalidated custom solver. |

These are implementation-based applicability judgments, not measured compiler
speedups. Compiler boundaries and numerical caveats are described in
[Numba performance guidance](https://numba.readthedocs.io/en/stable/user/performance-tips.html),
[PyTorch troubleshooting](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html),
and [CuPy fusion](https://docs.cupy.dev/en/stable/reference/generated/cupy.fuse.html).

Potential extras are Numba for CPU kernels and `numba-cuda` with the required
NVIDIA device math/runtime packages for CUDA prototypes. The
[CUDA Python suite](https://nvidia.github.io/cuda-python/latest/index.html)
provides execution/binding components, including `cuda.core` and `cuda.bindings`;
installing them alone does not fuse Array API calls. The
[nvmath device APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/index.html)
are another option for device-side math. Keep the existing scope exclusion of
custom mixed-precision tiled BLAS kernels. No extra was added in this review.

## Measurements and validation

The reproducible runner is `benchmarks/runners/lbfgs_linear_ineq.py`. Its before
classes reproduce the two changed operator methods and the LU-only dense L-BFGS
path from `c19f336`; all other code is shared. Each route step is one
factorization and two right-hand sides (the predictor/corrector pattern). Run with one CPU BLAS thread (set `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1` before launching Python); Torch is pinned to one thread.

```bash
python -m benchmarks.runners.lbfgs_linear_ineq
```

It separates large dense kernels from small fixed Newton systems. Route cases
use the same mathematical Jacobian in dense storage for dense/Krylov and
structural COO for sparse routes: two adjacent nonzeros per row, \(m=2n\),
\(n=128,512\), five L-BFGS pairs, nonzero bound diagonals, and slack-weight spreads
of 1 and \(10^4\). Sparse symbolic state is reused after warmup. Each case repeats identical
coefficients, so numeric Gram caches can also hit; this does not measure every
rebuild along an IPM trajectory. These localized rows favor sparse normal equations; they are not representative of all sparse
matrices. Dense and sparse controls do not use the changed Gram-diagonal helper.

Isolated large-kernel medians (10,000 by 1,000 float64 Jacobian, 15 samples of
seven calls, synchronized CUDA sample boundaries) were:

| Backend | Gram diagonal before / after (ms) | Scaled adjoint, eight RHSs, before / after (ms) |
|---|---|---|
| NumPy 2.4.6, CPU | 40.07 / 27.74 | 19.22 / 10.74 |
| Torch 2.12.0, CPU | 21.41 / 18.39 | 19.79 / 5.39 |
| CuPy, RTX 4070 Laptop GPU | 2.10 / 2.11 | 2.74 / 2.35 |

The Gram-buffer change improves these CPU timings; CUDA time is essentially
unchanged. It removes one 80 MB matrix-sized live temporary on mutable backends,
by the allocation structure of the expression; this is not a measurement of
allocator-reserved peak memory. Batched adjoints improve all three measured
backends. Neither table entry is a whole-IPM speedup. The isolated repeat can be
run with `--sizes= --repeats 15 --calls 7`; full route measurements use seven
samples of three calls by default.

Dense-route medians for the Cholesky reuse (one factorization, two right-hand
sides, \(m=2n\), single BLAS thread; the Krylov and sparse rows of the same run
are controls that do not touch the change and moved within noise) were:

| Backend | \(n\) | Slack spread | LU-only before / Cholesky after (ms) |
|---|---|---|---|
| NumPy 2.4.6, CPU | 512 | 1 / \(10^4\) | 23.9 / 23.0 · 28.3 / 26.7 |
| NumPy 2.4.6, CPU | 1024 | 1 / \(10^4\) | 165.0 / 149.8 · 151.4 / 142.0 |
| Torch 2.12.0, CPU | 512 | 1 / \(10^4\) | 14.8 / 12.0 · 15.0 / 12.6 |
| Torch 2.12.0, CPU | 1024 | 1 / \(10^4\) | 112.1 / 88.8 · 112.5 / 84.5 |

The step time is dominated by forming the dense Gram \(A^T\Sigma_sA\)
(\(2mn^2\) flops against \(n^3/3\) for the Cholesky), so the factor-level
saving of half an LU plus one \(O(n^3)\) re-solve shows up as 4–10% (NumPy)
and 19–33% (Torch) of the whole step at this aspect ratio; taller problems
dilute it further, more right-hand sides per factorization amplify it.
Residuals were unchanged to round-off (\(\le 4\times10^{-14}\) relative).

On the local NumPy, Torch CPU, and CuPy/cuDSS RTX 4070 Laptop GPU runs, every
fixed-system relative infinity-norm residual was below \(2.0\times10^{-10}\).
At \(n=512\), both Jacobi and L-BFGS preconditioning took 11 CG iterations at
unit weights and 231 with the larger weight spread. Thus compact Hessian
preconditioning alone did not address the dominant constraint coupling.
These counts support investigating conditioning and scalar synchronization;
they do not justify a universal route change.

The regression suite checks NumPy, Torch, and Array API Strict, plus local CuPy.
It covers input preservation, overflow-sensitive multiplication order, empty and
batched adjoints, and a known QP optimum with an active inequality and active
bound through dense/Krylov/sparse. Both dense `linear_ineq` lowering and constant
COO callbacks are checked, with L-BFGS explicitly selected and gradient-based
scaling enabled. JAX and Torch CUDA were not validated in this environment.

**Full-corpus sweep gate (v31, 2026-09-07).** The S2MPJ sweep at the canonical
budget (`--max-iter 10000 --max-time 300`, objective-free problems included)
moved the corpus from 4608 to 4616 correct of 6600 (+8 net, 15 fixed / 7 broken)
with zero linear-solver route changes. On `lbfgs/dense`, the only configuration
whose solve path the Cholesky reuse changes, no correctness flag flipped and
46 of 1098 rows changed their iteration count — the round-off signature of a
Cholesky replacing an LU. Every broken row is a Krylov configuration (the
ORTH*/KISSING/HS111/PALMER churn family plus two cap-boundary cases), where the
only touched code is the saddle preconditioner's `row_gram_diagonal` order.
The same-iteration-count wall-time ratio was 0.97 corpus-wide and 0.88 on
`lbfgs/dense`, but the untouched sparse configurations spread 1.05–1.10 on the
same machine, so the sweep is consistent with the microbenchmark, not a
measurement of it.

Full IPM speedups, peak allocator measurements, cold compilation, and profiled
kernel/synchronization counts remain separate measurements. Apply the
[adoption criteria](performance.md#validation-before-adoption) before enabling
any of the adapter or compiler suggestions.
