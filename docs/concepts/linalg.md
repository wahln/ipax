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

Selection is automatic (size + density + namespace capabilities) and
user-overridable via `Options.linsolve`. Adding a solver never touches
`ipm/driver.py` (invariant #3).

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

## Sparsity as an adapter concern

The standard has no sparse type. The core emits structure as Array-API
integer/float `(row, col, value)` vectors; thin per-backend wrappers in
`backend/sparse/` build and factor the actual sparse matrix. The IPM never sees
a backend-specific sparse object.

Two backends own a concrete factorization: SciPy (CPU — Feral LDLᵀ / SuperLU)
and CuPy (CUDA — NVIDIA cuDSS via nvmath-python, with a `spsolve` fallback).
Torch and JAX have no native sparse-direct path of their own, so they **reuse**
those two: the `DeviceRoutingSparseAdapter` reads the COO buffer's
`__dlpack_device__` and reinterprets it — zero-copy via DLPack where the
libraries allow — onto the SciPy adapter for host arrays and the CuPy/cuDSS
adapter for CUDA arrays. So Torch-CPU and JAX-CPU factor through Feral/SuperLU,
and Torch-CUDA and JAX-GPU factor through cuDSS, with results handed back in the
caller's namespace. Routing is by *device*, not by library name.
