# The linear-algebra layer

The IPM iteration depends only on two protocols; everything scale- and
sparsity-related lives behind them.

## `LinearOperator`

A backend-agnostic `matvec` (+ optional `rmatvec`/`matmat`). Subclasses:
`Dense`, `Diagonal`, `Identity`, `LowRank`, `LBFGSOperator`,
`MatrixFreeJacobian`, `SparseOperator`, `Composite`. Every `Problem`
Jacobian/Hessian normalizes into one of these.

## `LinearSolver`

`solve(K, rhs) -> step` for the augmented or condensed KKT operator.

| Solver | Use | Purity |
|---|---|---|
| `DenseSolver` | ≲ 1e4 vars; Cholesky/solve | pure Array API |
| `KrylovSolver` | default at scale; matrix-free CG/MINRES/GMRES | pure Array API |
| `SparseDirectSolver` | sparse Jacobians; per-backend factorization | adapter |

Selection is automatic (size, constraint shape, Jacobian density, estimated
Gram fill, namespace capabilities) and user-overridable via `Options.linsolve`.
Adding a solver never touches `ipm/driver.py` (invariant #3).

### Mixed-precision Gram accumulation (dense route)

For tall inequality problems the dense condensed route spends 80–90% of its
per-iteration wall forming the Gram term `∇gᵀ Σ_s ∇g` (O(m·n²) FLOPs). The
default `DenseOptions(gram_dtype="auto")` prefers the precision the
constraint data actually carries: when the inequality Jacobian *declares*
float32-grade data (`gram_accumulate_dtype_hint()` — float32 storage, or
float64 values that are exact float32 upcasts declared by their producer, as
the TROTS loader does for cases whose dose matrices are float32 in the file),
the accumulation
runs in float32 — ~2× on CPUs (twice the SIMD width), up to the fp32/fp64
rate ratio on GPUs — while everything else stays float64. Fully-float64
problems are untouched: the hint is declared metadata, never a value scan,
so behavior only changes where the data provably carries no float64
information. `gram_dtype="float32"` forces the reduction, `"native"`
disables it. Both respect the working precision the solve actually runs in —
the reduced dtype must be strictly narrower than it, so a float32 solve of
float32 data stays a plain float32 solve rather than paying a refinement
pass for bit-identical arithmetic. Full
working accuracy is restored per solve by **iterative refinement** against the
exact float64 operator matvec (Carson & Higham 2018): each O(n²)-solve +
matvec correction contracts the error by ρ ≈ κ(N)·u₃₂, so a handful of steps
reach `refine_tol` — and a solve that runs out of budget or plateaus is still
accepted when its *measured exact residual* clears the looser
`refine_accept_tol` certificate (mid-barrier Newton steps need nowhere near
direct-solve accuracy). A solve missing even that level — or a
positive-definiteness failure the exact matrix does not reproduce (both the
signature of a κ(N)·u₃₂ ≳ 1 stretch) — rebuilds the exact matrix for that
factorization; `refine_failure_limit` *consecutive* failures return the
instance to native precision for good. Conditioning along an IPM run is not
monotone (μ jumps, δ_w, re-centering), which is why one hard iteration is not
terminal. Every returned step carries a measured exact-residual certificate —
only accumulation cost is ever traded. (The PD probe itself runs on the
approximate matrix; the refinement rejection is what catches the
masked-indefinite case, and the pair is pinned by a regression test.) The
option applies to the inequality/bound **condensed** assembly —
equality-constrained saddle systems currently assemble exactly and ignore it.
`Result.routes` reports the engaged route as `dense (gram=float32)`, or
`dense (gram=float32->native)` after a self-disable.

!!! warning "Known limitation: matrix-free Krylov on equality saddles"

    On **equality-constrained** problems the matrix-free `KrylovSolver` borders
    the condensed system into an indefinite saddle and, by default (Jacobi
    preconditioner), solves it with MINRES. A diagonal preconditioner is too weak
    for *ill-conditioned* saddles: MINRES can stall and the solve is reported as
    `numerical_error`. This affects a number of equality-heavy CUTEst problems
    (optimal-control / network models) when `linsolve="krylov"` is forced.

    An opt-in **block-Schur preconditioner** is now available for the L-BFGS route:
    `KrylovOptions(preconditioner="lbfgs")` builds the block-diagonal
    `diag(N⁻¹, S⁻¹)` (Murphy–Golub–Wathen), applying the L-BFGS-aware Woodbury
    inverse on the (1,1) block, and switches the saddle solve to GMRES (which can
    left-apply the non-diagonal preconditioner). It measurably helps saddles whose
    bottleneck is the *linear solve* once L-BFGS has warmed up, but does not help
    the first-iteration cold-start failures (before any curvature pair, the L-BFGS
    Hessian is `ξI` and the step direction — not the solve — is the issue). The
    same problems still solve through the **dense** route (the automatic choice
    below ~1e4 variables), so default usage is unaffected; the residual gap is for
    large, matrix-free, equality-constrained models.

    To get the block preconditioner *only where it pays off*, use
    `KrylovOptions(preconditioner="auto")`: it runs the cheap Jacobi diagonal
    until a solve struggles — a convergence failure triggers an immediate promoted
    retry, and a slow success (over `auto_switch_ratio` of the iteration budget)
    promotes for subsequent solves — then stays on the L-BFGS block/Woodbury
    preconditioner for the rest of that solve. Well-conditioned systems never pay
    the extra Woodbury cost.

## Sparsity as an adapter concern

The standard has no sparse type. The core emits structure as Array-API
integer/float `(row, col, value)` vectors; thin per-backend wrappers in
`backend/sparse/` build and factor the actual sparse matrix. The IPM never sees
a backend-specific sparse object.

### The two sparse KKT forms

`SparseDirectSolver` can assemble the KKT system two ways
(`SparseOptions(kkt_route=...)`):

- **`"augmented"`** (default): factors the bordered symmetric-indefinite
  emission — inequalities stay explicit as a `∇g` / `−Σ_s⁻¹` border, so the
  factor is as sparse as `∇g` itself, and equalities border in as a
  quasidefinite saddle. Works for every constraint mix; the LDLᵀ inertia
  drives the IPM's δ_w correction (Wächter & Biegler 2006, §3.1).
- **`"normal_equations"`**: condenses the inequality Gram term `∇gᵀ Σ_s ∇g`
  *sparsely* into the logical `n×n` block and factors that instead (Breedveld
  2017, §2: the condensed system is `n×n` however large `m` grows). For tall
  (`m ≫ n`) problems with **localized** Jacobian rows (banded/block
  dose-influence structure) this replaces an `(n+m)`-sized bordered factor —
  or a Krylov iteration whose counts blow up on the ill-conditioned late-IPM
  `Σ` — with one small sparse Cholesky-sized factorization per iteration.
  Measured on banded tall QPs (`m = 10n`): 3–4.6× faster per iteration than
  the bordered factor at `n` = 10k–30k, and optimal in 62 s at `n` = 20k where
  matrix-free Krylov ran 50+ minutes unconverged. Equality constraints
  border in as a Schur/equality block — `∇c` stays explicit next to the
  condensed-Gram block in the factored quasidefinite saddle — so all
  constraint mixes are supported; the Hessian must be one the fold supports
  (L-BFGS keeps its low-rank border; a COO-emittable analytic Hessian folds
  directly).

The trap in the normal-equations form is *Gram fill-in*: scattered rows of the
same nnz fill `∇gᵀ∇g` almost completely, and a sparse factorization of a
near-dense `n×n` block is hopeless. `LinearOperator.gram_fill_estimate()` is
the cheap structural probe that resolves this: the pattern of Gram column `j`
is exactly the union of the column patterns of the rows touching `j`, so
sampling evenly spaced columns estimates the Gram-pattern density without
forming the Gram (within 1% of the exact fill at `n` = 10k–20k, for tens of
milliseconds, once, at selection time). `linsolve="auto"` uses it: tall
sparse-Jacobian problems whose estimated fill stays below ~1% take the
normal-equations route automatically; everything else keeps the previous
selection. The form remains directly selectable via
`SparseOptions(kkt_route="normal_equations")` for cases the conservative
auto-gate declines.

!!! note "Inertia is a backend capability"

    The inertia-guided δ_w correction engages only when the backend
    factorization reports the factor's inertia (Feral and cuDSS do, for
    free; SuperLU does not). When the KKT operator has an inertia target but
    the factorization cannot report one, `SparseDirectSolver` logs a
    one-time warning — on nonconvex problems a wrong-inertia factor could
    otherwise pass undetected. Install `ipax[sparse-cpu]` (Feral) on CPU or
    use cuDSS on CUDA to close the gap.

### Per-backend factorization routing

Two backends own a concrete factorization: SciPy (CPU — Feral LDLᵀ / SuperLU)
and CuPy (CUDA — NVIDIA cuDSS via nvmath-python, with a `spsolve` fallback).
Torch and JAX have no native sparse-direct path of their own, so they **reuse**
those two: the `DeviceRoutingSparseAdapter` reads the COO buffer's
`__dlpack_device__` and reinterprets it — zero-copy via DLPack where the
libraries allow — onto the SciPy adapter for host arrays and the CuPy/cuDSS
adapter for CUDA arrays. So Torch-CPU and JAX-CPU factor through Feral/SuperLU,
and Torch-CUDA and JAX-GPU factor through cuDSS, with results handed back in the
caller's namespace. Routing is by *device*, not by library name.
