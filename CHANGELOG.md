# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Public sparse operators `COOOperator`, `CSROperator`, and `CSCOperator`
  (exported from `ipax`), so a `Problem` can return a sparse Jacobian or Hessian
  without a concrete sparse library in user code. They carry Array-API
  index/value vectors and emit structure for the sparse-direct route
  (`to_coo`/`coo_values`/`coo_pattern_signature`) in pure Array API, delegating
  the matvec-family, diagonal, and Gram-diagonal algebra to the per-backend
  adapter (invariant #4). Pattern reuse is opt-in via `pattern_key` (declare the
  sparsity fixed across a solve to unlock symbolic-analysis and structure-cache
  reuse). CSR/CSC are ergonomic constructors over the same core — the solver
  factorizes the assembled KKT matrix, not the user block, so the compressed
  format is a convenience, not a performance, choice. The SciPy and CuPy adapter
  operators gained `gram_diagonal`/`row_gram_diagonal`/`rmatmat` so these
  operators are first-class on the condensed-Jacobi and saddle-MINRES
  preconditioner paths.

### Changed
- Faster dense KKT solve: the condensed, saddle, and L-BFGS Hessian operators now
  implement native batched `matmat`/`rmatmat` (and `rmatmat` is available across the
  `LinearOperator` subclasses), so materializing the operator for the dense route
  pushes the whole identity through each sub-block once instead of looping
  `matvec` column-by-column — far fewer kernel launches at scale. The Cholesky
  positive-definite guard also reuses the already-materialized matrix instead of
  re-forming the primal block. Bound-only L-BFGS dense solves now use an exact
  Woodbury compact solve (and equality saddles use the corresponding Schur
  complement) before falling back to full materialization. Explicit operators now
  expose a `dense_matrix` hook, so the fallback dense route can assemble exact
  dense/saddle blocks directly instead of probing with an identity matmul; the
  materialized matrix is cached across repeated `solve()` calls after `factor()`,
  and diagonal exact-Hessian bound-only systems use a direct diagonal solve.
- Faster sparse-direct KKT solves: explicit operators and KKT assemblers now
  expose conservative `coo_pattern_signature()` metadata, letting sparse adapters
  split stable structure from per-iteration values. The CuPy/cuDSS route reuses
  symbolic analysis with `matrix_set_values` for same-pattern factors instead of
  comparing GPU CSR index arrays, while unknown/value-dependent exact-Hessian
  patterns still force re-analysis. Condensed and equality-saddle regularization
  diagonals are reserved so `delta_w`/`delta_c` activation changes values rather
  than sparsity, and the sparse facade guards cached adapters against backend or
  device changes.
- Faster sparse assembly (values-only refactor): the SciPy and CuPy sparse
  adapters now compile the COO→canonical-CSR/CSC transform once per fixed
  `pattern_signature` into a gather/segment-sum map, then recompute only the value
  array on subsequent factorizations — replacing the per-iteration
  `coo_matrix(...).tocsc()` / `tocsr().sum_duplicates().sort_indices()` sort
  (`O(nnz log nnz)`, never amortized by the factorization's symbolic reuse) with a
  single `O(nnz)` scatter-add. The cuDSS symmetric route likewise caches its
  full-CSR→lower-triangle map instead of re-running `tril(...)` plus a
  re-canonicalization each iteration, and the device-routing adapter (Torch/JAX)
  now persists its delegate so the cache survives across iterations. The cuDSS
  route also uploads int32 CSR offsets/indices when the system fits a signed
  32-bit integer (halving index bandwidth), falling back to int64 otherwise.
- Faster sparse assembly (cached COO structure): `LinearOperator` gained an
  optional `coo_values()` hook (the values-only counterpart of `to_coo()`), and
  the condensed/saddle KKT blocks plus `Dense`/`Diagonal`/`Identity`/`VStack`
  implement it. The `SparseDirectSolver` facade now caches the COO row/column
  vectors keyed on `coo_pattern_signature` and recomputes only the values each
  iteration, so the per-step assembly skips rebuilding the index vectors — most
  notably the L-BFGS low-rank border's `O(n·m)` index grids — on a fixed pattern.
- Faster cuDSS solves (descriptor reuse): the cuDSS dense RHS/solution
  descriptors and their device buffers are now created once and reused across
  solves and the factorization phase (rebuilt only when the system size changes),
  replacing the per-solve `matrix_create_dn`/`matrix_destroy` churn with a buffer
  copy.
- `Options.hessian` now defaults to `"auto"` (use a supplied analytic
  `lagrangian_hessian`, else L-BFGS). The default behavior is unchanged, but the
  explicit modes are now honored literally even when an analytic Hessian exists:
  `"lbfgs"` always uses the limited-memory approximation and `"autodiff-hvp"`
  always uses autodiff Hessian-vector products. Previously a supplied
  `lagrangian_hessian` silently overrode `hessian="lbfgs"`, so the L-BFGS route
  could not be exercised on a problem that also defined an exact Hessian.

### Removed
- The unimplemented `"lsr1"` value of `Options.hessian`. It was never wired to a
  solver path — selecting it silently ran L-BFGS (or the analytic Hessian) — so
  it is dropped rather than left as a false promise. A limited-memory SR1 update
  remains possible future work but requires the indefinite augmented/inertia route
  (its indefinite approximation is incompatible with the condensed route's
  PD-by-Powell-damping design); see the note in `ipax/ipm/hessian.py`.

### Fixed
- The filter line search no longer diverges (and falsely reports `infeasible`) on
  problems where a quasi-Newton step's barrier objective collapses while the
  constraint violation explodes. The filter is now initialized with the
  Wächter & Biegler 2006 eq. (18) guard region `{θ ≥ θ_max}`,
  `θ_max = 1e4·max(1, θ(x₀))`, so a trial with non-finite or wildly large `θ` is
  rejected outright before the switching/Armijo test (surfaced by the L-BFGS route
  on Hock-Schittkowski HS7 in the S2MPJ sweep).
- The sparse-direct route no longer aborts the whole solve with an uncaught
  `ValueError` when an upstream Hessian/Jacobian overflows to inf/NaN at a bad
  trial iterate. The sparse adapter now rejects a non-finite KKT matrix with a
  `LinearSolveError`, which the IPM regularization loop already handles (δ_w
  escalation, then a graceful step-failure classification) — matching the dense
  route. This clears the S2MPJ sweep's `solve_error` cluster (EIGMAXA, EIGMINA,
  LUKVLE8, MESH, BRATU1D, RAT42LS, RAT43, INDEF); several now converge to an
  `optimal`/`acceptable` point once the failed factorization is recovered instead
  of crashing.
- Two robustness guards for problems whose element functions overflow (exp /
  rational terms evaluated far outside their safe range), which previously drove
  the L-BFGS route to a `numerical_error` on the S2MPJ sweep:
  - The L-BFGS update now drops a **non-finite curvature pair** instead of
    appending it. A non-finite gradient difference `γ = grad L(x+) - grad L(x)`
    would otherwise corrupt the compact form permanently (the poisoned column
    survives the memory window, and the Powell/positive-curvature safeguards miss
    it because the curvature `δ·γ` is then NaN).
  - The filter line search now rejects a trial whose **barrier objective `φ_t` is
    non-finite** (not only non-finite `θ_t`). A step that overshoots into a region
    where the objective evaluates to `-inf` would otherwise pass the Armijo test
    (`-inf` is below any finite bound) instead of backtracking to a finite iterate.
- The L-BFGS route now recovers from a step that overshoots into a region where
  the **objective gradient is non-finite** while `θ`/`φ` are still finite (e.g.
  RAT42LS's logistic, whose residual stays finite while its derivative becomes
  `inf/inf = NaN`). The filter line search evaluates only `θ`/`φ`, so it accepted
  such a step and the next KKT solve was poisoned (`numerical_error`). A
  `grad_finite(α)` check is now threaded into the line search on the L-BFGS route:
  a non-finite-gradient trial is unacceptable, so the search backtracks to a
  damped step in the finite region (handing off to restoration only if the whole
  ray is bad). Moves RAT42LS from `numerical_error` to `optimal` on the sparse
  route. The exact-Hessian route, whose scaled steps do not overshoot, skips the
  extra gradient evaluation.

## [0.3.0] - 2026-06-26

### Added
- Documentation: a published **S2MPJ / CUTEst benchmark** page
  (`docs/benchmarks/s2mpj.md`) recording the latest full-corpus run — system
  information, per-configuration metrics for the `{lbfgs, exact} × {dense, krylov,
  sparse}` matrix, the optimization-vs-feasibility split, and the dataset-sourced
  scoring methodology.
- The per-iteration log table reprints its column header every
  `HEADER_REPEAT_INTERVAL` (10) rows so it stays readable on long runs, and marks
  any iterate that already satisfies every enabled acceptable-stopping criterion
  (before the required consecutive count) with a trailing `*`.
- Expanded the Hock–Schittkowski analytic-oracle set in `ipax.testing.problems`
  with `HS9`, `HS21`, `HS28`, and `HS71`, covering active bound multipliers, a
  degenerate (zero) equality multiplier, a non-unique periodic optimum, and the
  full equality+inequality+bounds constraint mix. Each is exercised across every
  backend in the integration suite, wired into the QC benchmark corpus, and
  checked by a new finite-difference derivative-consistency test that also
  back-fills the previously untested HS oracles.
- Two-sided linear inequalities (`Problem.linear_ineq`, `l ≤ A x ≤ u`) are now
  solved: the constant-data block is lowered into the standard one-sided
  inequality machinery (finite lower rows → `l − A x ≤ 0`, finite upper rows →
  `A x − u ≤ 0`, both-finite rows yield a range pair), so the IPM, gradient
  scaling, and every solver route handle it with no special-casing and the block
  contributes no Lagrangian-Hessian term. Previously `solve` raised
  `NotImplementedError` despite the interface being documented. A matrix-free
  (operator) `linear_ineq` matrix still raises with guidance to use
  `ineq_constraints` instead.

- S2MPJ scoring now uses the **dataset's own documented outcome** instead of
  convergence alone: the loader parses each source file for the CUTEst
  classification (`pbclass`), the SIF author's solution objective
  (`# LO SOLTN`, present on ~72% of the corpus), and an explicit
  `Solution (infeasible)` / `Source: an infeasible problem` marker. A case is
  scored *correct* when it reaches the documented objective, or — for a
  documented-infeasible problem like BURKEHAN — when it **detects infeasibility**
  (previously flagged as a failure). The report shows the gap to the documented
  optimum (`Δf*`) and annotates `infeasible (exp)`. The objective-free problems
  (CUTEst feasibility / nonlinear-equation systems) can now be run via
  `--include-objective-free` as `min 0` subject to the constraints, and one
  configuration can be swept per process with `--config` for parallel runs.
- S2MPJ sweep gained a size-aware run strategy for tractable full-corpus runs:
  **per-route variable caps** (dense 2000, Krylov 10000, sparse 25000) so each
  config runs only on problems that fit its route — small problems are
  cross-validated across every route while larger ones fall through to Krylov and
  the sparse-direct route — with `--max-vars` kept as a global ceiling; **sized
  instantiation** (`--size N`, with `PROBLEM(N)` for the scalable problems and a
  SIF-default fallback for the rest) to reach the sparse route's intended large-`n`
  regime; an optional subprocess **build-time guard** (`--max-build-seconds`) that
  abandons a pathological O(n²) pure-Python construction before it stalls an
  unattended sweep; and per-problem instance **caching** so the per-config fan-out
  rebuilds each problem once instead of up to five times.
- S2MPJ benchmark sweep now exercises the **exact Lagrangian Hessian** and the
  **sparse-direct route**, not only L-BFGS. The adapter (`_S2MPJExactProblem`) wires
  S2MPJ's `LgHxy`/`LHxyv` (convention `L = f + yᵀc`) into ipax's exact-Hessian route,
  mapping `(σ, y_eq, y_ineq)` onto S2MPJ's single multiplier vector with the correct
  signs for lowered inequality sides (lower `−y`, upper `+y`) and honoring `σ` on the
  objective term so it stays correct under gradient-based scaling. With `sparse=True`
  the Jacobians and Hessian cross as `SparseOperator`s (true COO sparsity) for the
  sparse-direct (Feral/cuDSS) factorization. The runner's regular matrix is now
  `{lbfgs, exact} × {dense, krylov, sparse}` (`exact/sparse` factors true sparsity —
  raise `--max-vars` to reach the large, sparse models), and its `--scaling` now
  defaults to `gradient-based` to match the solver default rather than benchmarking a
  scaling-off configuration users do not get.
- S2MPJ benchmark corpus: `benchmarks/corpus/s2mpj.py` loads the pure-Python
  S2MPJ translations of the CUTEst/Hock–Schittkowski problems (no Fortran/SIF
  toolchain) and bridges their NumPy/SciPy evaluation onto any CPU Array-API
  backend, mapping S2MPJ's two-sided `clower ≤ c(x) ≤ cupper` constraints onto
  ipax's eq/ineq split. A `benchmarks/runners/s2mpj.py` L-BFGS accuracy sweep
  consumes it. `list_s2mpj_problems()` enumerates a checkout and the runner's
  `--all` flag sweeps the **entire CUTEst set**, with `--max-vars`/`--max-iter`/
  `--max-time` caps, per-problem isolation, automatic skipping of objective-free
  problems, and a status summary. Download-gated (`IPAX_S2MPJ_DIR`); not vendored
  (S2MPJ has no license) and not part of per-PR CI — the loader returns `[]` and
  the gated tests skip when no checkout is present.

### Changed
- **Gradient-based scaling is now the default** (`ScalingOptions.method`
  `"none"` → `"gradient-based"`), matching IPOPT. Across the full CUTEst/S2MPJ
  corpus this solves a net **+67** problems (≈92 recovered, mostly from the
  slow-converging `max_iter` bucket; ≈23 regressed, of which only 3–4 genuinely
  fail — hard nonconvex/minimax cases that diverge under scaling — and the rest
  merely converge slower). The returned `x`, objective, and multipliers are
  reported in the original problem's units; pass `scaling="none"` to opt out.
- Promoted the driver's private vertical-stack operator to a public
  `ipax.backend.operators.VStack` (now also exposing `row_inf_norms` for
  gradient scaling), reused by both the equality assembly and the new
  linear-inequality lowering.

### Fixed
- A stall at a near-optimal iterate is now reported `ACCEPTABLE` instead of
  being discarded. Near a solution the condensed system is ill-conditioned (μ
  driven below the achieved KKT residual), so the Newton step can come out
  non-finite and the line search can fail to make progress even though the
  iterate is essentially optimal. The solver now salvages such an iterate —
  whether the failure is a non-finite **step solve** (previously
  `NUMERICAL_ERROR`) or the line search **handing off to restoration**
  (previously a false `INFEASIBLE`) — when its scaled KKT components are within a
  relaxed multiple (IPOPT `acceptable_tol` ≈ 1e2 × `tol`) of the optimality
  tolerances, rather than throwing away a usable solution.
- Fixed variables (`x_L == x_U`) — common in CUTEst-style models — no longer make
  the solve fail at the first iteration. Such a variable has no strict barrier
  interior, so `z = μ/(x − x_L)` was singular and the first Newton step came out
  non-finite (`numerical_error`). The solver now relaxes fixed / near-degenerate
  bound pairs symmetrically about their midpoint (IPOPT
  `fixed_variable_treatment='relax_bounds'`), leaving well-separated bounds
  untouched. Surfaced by the S2MPJ sweep, where it accounted for the bulk of the
  first-iteration `numerical_error` failures.
- The filter line-search switching condition no longer raises `OverflowError` on
  a badly-scaled iterate. Python's `float ** s_phi` raises instead of returning
  `inf` once the result exceeds the double range (an enormous directional
  derivative `dphi`), which crashed the whole solve; the power now uses IEEE
  overflow semantics (`→ inf`). Surfaced by the S2MPJ INDEF sweep. The S2MPJ
  benchmark adapter likewise sanitizes overflow in its NumPy-bridged
  objective/gradient (returning `inf`), so a trial point that overflows the
  problem's own generated `float**` is rejected rather than crashing
  (e.g. LUKVLE4C, which then solves).
- Feasibility restoration no longer crashes on a numerically singular or
  extreme-scale Gauss-Newton system. The damped (Levenberg–Marquardt) step now
  treats a failed/non-finite linear solve as a rejected step — growing the
  damping (up to a ceiling) and retrying — instead of letting the backend's
  ``solve`` raise (e.g. numpy ``LinAlgError: Singular matrix`` when a constraint
  Jacobian blows up far from feasibility). Surfaced by the S2MPJ HS7 sweep; the
  solve now degrades to a reported status rather than raising.
- `configure_verbosity` no longer attaches a second console handler when the
  application has already configured its own handler on the `"ipax"` logger,
  which previously printed every iteration record twice. Propagation to ancestor
  handlers (and `caplog`) is unchanged.

## [0.2.0] - 2026-06-21

### Added
- GPU/device-efficiency profiling harness: `DeviceMetrics` and
  `measure_device_solve` in `benchmarks/harness` (host↔device sync counter plus a
  CuPy GPU/CPU time split), the `benchmarks/runners/device_efficiency.py` CLI
  runner (GPU-gated, a no-op on CI), and kernel micro-benchmarks (matvec / dense
  solve / one Newton step) parametrized over every installed backend.
- Device reporting: backend `capabilities()` now discovers available devices via
  the Array-API inspection API (`__array_namespace_info__().devices()`) instead of
  assuming CPU, and `Result.device` records where the solve ran — surfaced in the
  tier-1 result summary.

### Changed
- Vectorized `fraction_to_boundary`, removing a per-element Python loop that forced
  `O(n)` host↔device synchronizations per call (and it is called six times per
  iteration). On a CUDA backend this cut the per-iteration sync count from
  thousands — scaling linearly with `n` — to a small constant, speeding up
  matrix-free GPU iterations by roughly 20× at scale; iterates and results are
  unchanged (CPU behavior is identical).
- CI uploads coverage with `codecov/codecov-action@v5` using a repository
  `CODECOV_TOKEN`.

## [0.1.1] - 2026-06-21

### Fixed
- CI `test` jobs no longer error while collecting `--doctest-modules` over the
  `ipax` package when an optional backend is absent. The JAX/CuPy autodiff and
  sparse adapters import their concrete library at module top level (the
  invariant #1 carve-out), so importing them fails when that backend is not
  installed — the normal CI case (neither JAX nor CuPy is installed there). A
  root `conftest.py` now skips those adapter modules during collection unless the
  backend can be imported; runtime dispatch already tolerated their absence.
- CI `lint` job no longer fails `mypy` on the NumPy/SciPy sparse adapter when
  NumPy is not installed in the minimal `[lint]` environment. `numpy.*` joins the
  optional-backend `ignore_missing_imports` overrides alongside torch/jax/cupy/scipy.

### Added
- Pre-commit hooks: `kacl-verify` enforces the Keep a Changelog format of this
  file, and `validate-pyproject` schema-validates `pyproject.toml`; both also run
  in the CI `lint` job.
- CI packaging job (sdist + wheel build with `twine check`) on `main`/`develop`
  pushes and pull requests, plus a tag-triggered `release.yml` workflow that
  publishes to PyPI (Trusted Publishing) and creates a GitHub release with notes
  drawn from this changelog and the tag annotation.

### Changed
- Project version corrected to `0.1.1` in both `pyproject.toml` (previously a
  stale `0.0.0`) and `ipax.__version__`.

## [0.1.0] - 2026-06-21

### Added
- Primal–dual interior-point solver for general NLP with equality, inequality,
  and bound constraints (`ipax.solve`, `Problem` interface).
- Capability-graded derivatives: analytic → autodiff → finite-difference for
  gradients/Jacobians; analytic → autodiff-HVP → Powell-damped L-BFGS for the
  Lagrangian Hessian.
- Pluggable linear algebra behind `LinearOperator` / `LinearSolver`: dense
  (Cholesky/solve), matrix-free Krylov (CG/MINRES/GMRES), and per-backend
  sparse-direct routes with automatic selection.
- IPOPT-style filter line search with second-order correction and feasibility
  restoration; optional Breedveld step controller; optional Mehrotra–Gondzio
  higher-order corrections.
- Inertia-guided δ_w regularization on the sparse-direct route: when the backend
  reports the LDLᵀ inertia (Feral / cuDSS), the IPM bumps δ_w until the factor's
  inertia matches the KKT operator's target, steering nonconvex solves away from
  saddle points. Falls back to factorization-failure escalation otherwise.
- Positive-definiteness guard on the dense reference route: with an exact Hessian
  the condensed block ``N`` is Cholesky-probed before the LU solve, so an
  indefinite ``N`` triggers δ_w escalation instead of a silent non-descent step
  (the dense analog of the sparse inertia check). Pure Array API.
- Gradient-based NLP auto-scaling, warm-start seeding, and layered diagnostics.
- Multi-backend (NumPy + PyTorch in CI; CuPy/JAX supported) with `array-api-strict`
  as the purity gate; import-purity gate (`scripts/check_purity.py`) enforcing
  invariants #1/#4.
- Contract batteries (`tests/contracts/`) plus unit/property/integration/backends/
  regression layers; benchmark suite (`benchmarks/`, asv); MkDocs documentation.

[Unreleased]: https://github.com/wahln/ipax/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/wahln/ipax/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/wahln/ipax/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/wahln/ipax/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/wahln/ipax/releases/tag/v0.1.0
