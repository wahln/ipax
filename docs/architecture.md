# Architecture & invariants

The canonical contributor rules live in `AGENTS.md`. This page is the
orientation map.

## Package layout

```
ipax/
  typing.py options.py result.py solve.py
  problem/   Problem ABC, FunctionProblem, derivative resolution, autodiff adapters
  backend/   namespace resolution, LinearOperator hierarchy, sparse adapters,
             Array API gap-fillers (dense Cholesky solve, bulk scalar reads)
  linalg/    LinearSolver protocol: dense / krylov / sparse-direct + regularization
  ipm/       barrier, kkt, step, filter_ls, restoration, breedveld_ls, hessian, init, driver
  testing/   analytic oracle problems shared with benchmarks
```

## The two protocols that carry the design

- **`LinearOperator`** (`backend/operators.py`) — every Jacobian and the
  Lagrangian Hessian is one of these (`Dense`, `Diagonal`, `LowRank`,
  `LBFGSOperator`, `MatrixFreeJacobian`, `SparseOperator`, `Composite`). The IPM
  loop never inspects a concrete matrix.
- **`LinearSolver`** (`linalg/solver.py`) — every KKT solve goes through this:
  `DenseSolver`, `KrylovSolver` (matrix-free, default at scale),
  `SparseDirectSolver`. Auto-selected by size/density/capabilities.

## Non-negotiable invariants

1. **No concrete array library in the core** — only in `backend/sparse/` and
   `problem/autodiff/`. Enforced by `scripts/check_purity.py` and
   `tests/test_purity_gate.py`.
2. **Stay inside the Array API standard** (main namespace + optional `linalg`).
   `array-api-strict` is the purity gate in the test matrix.
3. **Linear algebra is injected**, never hard-wired into `ipm/driver.py`.
4. **Sparsity is an adapter concern** — the core emits index/value vectors only.
5. **No global mutable state** — solver state lives in explicit objects.

## Two KKT solve routes

- **Condensed normal-equations** (Array-API-native default): PD via Powell-damped
  L-BFGS + Friedlander–Orban primal–dual regularization — **no inertia oracle**.
  With an *exact* (assemblable) Hessian the dense reference solver additionally
  Cholesky-probes the condensed block `N`: `xp.linalg.solve` (LU) would accept an
  indefinite `N` and return a non-descent step, so a failed Cholesky drives the
  same δ_w escalation, steering nonconvex solves away from saddle points.
- **Indefinite augmented** (sparse-direct only): inequalities and the L-BFGS
  low-rank term stay explicit as symmetric *border* blocks (`∇g` with `−Σ_s⁻¹`),
  so the bordered matrix the sparse LDLᵀ factors is indefinite but its Schur
  complement is exactly the condensed `N`. When the backend reports inertia
  (Feral / cuDSS), the IPM runs the IPOPT inertia-guided δ_w correction —
  bumping δ_w until the factor's inertia matches the operator's target, which
  steers nonconvex solves away from saddle points.

See `concepts/algorithm.md` for the math.
