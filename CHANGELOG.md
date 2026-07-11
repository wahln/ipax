# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Dense route: the PD-probe Cholesky factor is now reused for the solve.**
  `DenseSolver` already Cholesky-probes the condensed block `N` to reject
  indefinite systems (the δ_w escalation guard); that factor was then
  discarded and the same matrix re-factored with LU by `xp.linalg.solve`.
  A new backend gap-filler `get_dense_cholesky_solve` (`ipax/backend/dense/`;
  the Array API `linalg` extension has no triangular solve — BLAS `trsm` /
  LAPACK `potrs`) back-substitutes the kept factor instead: SciPy `cho_solve`
  on NumPy, `torch.cholesky_solve` on Torch (CPU/CUDA), two
  `cupyx.scipy.linalg.solve_triangular` sweeps on CuPy. Every solve against
  one factorization drops from O(n³) to O(n²), which also makes
  Mehrotra/Gondzio corrector and SOC back-solves near-free on the dense
  route. Measured (NumPy, single-threaded BLAS, n=1000–4000, m=n/2):
  1.2–1.4× per factor+solve, 1.7–1.9× for the factor+3-back-solves pattern.
  Backends without the primitive (array-api-strict, JAX) and the equality
  saddle route (indefinite bordered matrix; only its leading `N` block is
  probed) keep the LU path unchanged.

## [0.5.0] - 2026-07-10

### Added
- **More informative progress logging.** The `verbose >= 1` (info) tier now
  opens with a condensed one-line problem/solver headline before the result
  summary — variable/bound counts, equality/inequality counts, the resolved
  linear solver, and the resolved Hessian source (`format_setup`). The
  `verbose >= 2` per-iteration table gains an `ls` column reporting the
  number of backtracking trial step sizes the line search (filter or
  Breedveld) evaluated to reach each row, and rows produced by a
  feasibility-restoration jump are now tagged with a trailing `R`
  (`IterationRecord.line_search_iters`/`.restored`, combinable with the
  existing acceptable-iterate `*` tag).
- **Sparse condensed normal-equations route**
  (`Options(linsolve="sparse", sparse=SparseOptions(kkt_route="normal_equations"))`).
  The condensed matrix ``N = W + Σ_x + δ_w I + ∇gᵀ Σ_s ∇g`` is assembled
  *sparsely*: a new ``LinearOperator.gram_coo`` capability (SciPy and CuPy
  adapters, forwarded through `COOOperator`/`CSROperator`, `VStack` and the
  gradient-scaling wrapper) emits ``∇gᵀ Σ_s ∇g`` as pattern-stable COO
  triplets, and the condensed KKT operator folds them into its logical
  ``n×n`` block (`normal_equations_coo`; an L-BFGS Hessian keeps its
  low-rank border). `SparseDirectSolver(form="normal_equations")` factors
  the result with full structure/symbolic reuse and translates the factor's
  inertia back into bordered terms (Haynsworth additivity: the eliminated
  ``−Σ_s⁻¹`` block's ``m_I`` negatives), so the inertia-guided δ_w
  correction works unchanged. For tall (``m ≫ n``) problems with
  **localized** Jacobian rows (banded/block dose-influence structure) this
  replaces a ``(n+m)``-sized bordered factor or a Krylov iteration with one
  small sparse Cholesky-sized factorization per iteration (Breedveld 2017
  §2). Measured on banded tall QPs (``m = 10n``, 20 nnz/row): 3–4.6× faster
  per iteration than the bordered augmented factor at ``n`` = 10k–30k
  (6.2 vs 15.6 s at 20k), 32× faster than the dense condensed route at 10k
  (1.27 vs 40.4 s, 237 MB vs 3.3 GB peak), and on par with matrix-free
  Krylov on early well-conditioned iterations — but decisively ahead at
  convergence, where the late-IPM Σ blows up CG's inner iteration counts
  while the direct factor's cost stays flat: at ``n`` = 20k the sparse-NE
  solve reaches OPTIMAL (KKT 4e-9) in 25 iterations / 62 s, while the
  Krylov route had not converged after 50+ minutes. Deliberately opt-in:
  with non-localized rows the Gram fills in toward dense ``n²``, which no
  cheap structural probe can detect.
- **Tall sparse QP generator** (`benchmarks.generators.make_tall_sparse_problem`):
  seeded ``m ≫ n`` bound-constrained QPs with a Gram-capable sparse-operator
  Jacobian, random or banded (``bandwidth=``) column structure — the
  measurement vehicle for the tall-selection thresholds and the sparse-NE
  route.
- **Tall auto-selection is now density-gated.** The tall-crossover
  measurement (2026-07, tall QPs at ``m = 10n``) showed the dense condensed
  route beating Krylov 5.4× at ``n = 5000`` but *losing* (75.5 vs
  46.5 s/iteration) at ``n = 10 000`` with a ~1%-dense Jacobian: the
  tall-dense advantage comes from the adapters' dense-GEMM Gram, which only
  engages at ≥ ~5% density. `select_solver`'s tall branch now also probes
  ``nnz/(m·n)`` of the inequality Jacobian and keeps Krylov below 5%
  density; dense-ish Jacobians (TROTS dose matrices, 11–53%) keep the
  measured 13–19× dense-route wins.
- **Stall detection** (`Options.max_stall_iter`, default 25; new
  `Status.STALLED`): consecutive frozen iterations — no accepted step and a
  bit-for-bit unchanged KKT error — now terminate honestly (or as
  `ACCEPTABLE` within the relaxed KKT tolerance) instead of re-deriving the
  identical rejected direction until `max_iter` (S2MPJ POWELLBSLS burned
  10 001 iterations this way). Limit cycles that keep moving the iterate reset
  the counter and still run to the ordinary budgets.
- **Condensed normal-equations tuning for tall (`n ≪ m`) problems.** The
  SciPy/CuPy sparse adapters now cache the Gram fast path per operator: a
  last-weights memo returns the memoized `∇gᵀ Σ_s ∇g` for the bit-identical
  Σ_s re-requested by δ_w retries, SOC and Mehrotra re-solves within an IPM
  iteration, and cache misses reuse a one-time `Aᵀ` CSR transpose +
  same-pattern scaled buffer (plus a shared `A∘A` for the Gram diagonals), so
  only value work and the SpGEMM itself remain per iteration. Dense-ish
  Jacobians (row density ≥ 5%, e.g. TROTS dose matrices at ~30%) switch to a
  chunked dense-GEMM accumulation instead — SpGEMM's Σ nnz_row² hash
  arithmetic plus its per-call symbolic pass is the wrong algorithm there
  (measured 27× on TROTS Prostate_CK: 90 s → 3.4 s per Gram). Solver
  auto-selection gained the same awareness: `select_solver` now accepts the
  inequality row count and a lazy `gram_capable()` probe (new
  `LinearOperator.gram_capable`, forwarded by row scaling and `VStack`) and
  keeps the dense condensed route up to 20k variables when `m ≥ 10·n`, where
  one `n×n` factorization per iteration beats Krylov matvecs through the huge
  Jacobian (Breedveld et al. 2017). `DenseOptions.augmented_max_size`
  (default 20 000) guards the dense augmented route against materializing the
  `(n+m_eq+m_I)²` bordered matrix at such row counts — it falls back to the
  condensed route before allocating, losing only the inertia diagnostic.
- **Barrier μ oracles, orthogonal to higher-order corrections.**
  `Options(mu_schedule=...)` is now honored and grew a value: `"adaptive"`
  re-targets μ every iteration by the LOQO centrality rule (Nocedal, Wächter &
  Waltz 2009, eq. (3.6)), `"breedveld"` applies the steplength-driven
  duality-gap reduction σ(α) = ((α−1)/(α+10))² of Breedveld et al. 2017
  (eqs. (10)–(12)), `"probing"` runs the Mehrotra σ-rule from an affine probe
  (also standalone, without a corrector), and `"monotone"` is the guarded
  Fiacco–McCormick loop. Mehrotra/Gondzio correctors now *consume* the oracle's
  μ target instead of choosing it, so any oracle can steer any corrector.
  Previously the non-monotone values were accepted but silently ran the
  monotone schedule.
- **Centrality floor for the free-mode oracles**
  (`BarrierOptions.kappa_centrality`, default `1e-2`): μ never drops below
  κ_cent·max(dual, primal infeasibility), the practical rendering of the
  centrality condition in El-Bakry et al. (1996)'s Newton interior-point
  convergence theory. Without it an aggressive oracle can crush μ near a
  saddle while the iterate is far from stationary, pinning it to the boundary
  beyond the reach of the KKT-error fallback's complementarity-based re-entry.
- **Free-mode KKT-error safeguard** (NWW 2009, §5.1, Algorithm A): a
  non-monotone μ oracle is suspended when the scaled KKT error stops making
  sufficient progress (`fallback_kappa`/`fallback_window`, paper defaults
  κ = 0.9999, l_max = 5), falling back to the monotone reduction re-initialized
  at 0.8× the average complementarity until the error recovers —
  the counterpart of IPOPT's `adaptive_mu_globalization="kkt-error"`.
  `BarrierOptions(fallback="never")` restores pure free mode.
- **TROTS radiotherapy benchmark corpus** (`benchmarks/corpus/trots.py`,
  download-gated on `IPAX_TROTS_DIR`). Loads the TROTS treatment-planning problems
  (MATLAB v7.3/HDF5) into an ipax `Problem`: all cost functions (linear
  max/min/mean, quadratic, gEUD, LTCP, smoothed DVH) with the weighted
  scalarisation, the pointwise-max objectives lowered to a minimax epigraph, the
  elementwise dose constraints emitted as Array-API sparse operators, an exact
  Lagrangian Hessian variant, and the dataset's least-squares warm start
  (`misc.Initialise*`). Objective evaluated at the reference `solutionX` reproduces
  the published `Objective Function Value` to the significant figures each solve
  reports (validated across the proton/photon/liver groups). Runner + gated tests
  included.
- **Sparse-Gram condensation for the dense KKT route.** `LinearOperator` gains an
  optional `gram(weights)` capability returning the dense `n×n` weighted Gram
  `Aᵀ diag(w) A`, implemented on the sparse operators (`COO`/`CSR`/`CSC`, the
  per-backend `SparseOperator`, and `VStack`) by sparse arithmetic. The condensed
  dense route (`ipm/kkt.py`) now forms the inequality term `∇gᵀ Σ_s ∇g` through it
  instead of densifying the Jacobian to `m×n` first — the decisive change for
  problems with far more constraints than variables (`n ≪ m`), where the old path
  materialized a gigabytes-scale dense Jacobian to build a small `n×n` matrix
  (e.g. a TROTS proton case: a 369k×1101 Jacobian ≈ 3.2 GB, now a ~4 s sparse
  Gram). Falls back to the previous densify-and-multiply when the Jacobian exposes
  no `gram` (dense/matrix-free) or `Σ_s` is non-diagonal.

### Fixed
- **The cuDSS route now actually works against a real runtime** (first
  hardware validation of the GPU sparse-direct path; previously it was only
  exercised through NumPy-backed fakes). Three defects found by running on
  a live cuDSS 0.7/CUDA 12 stack, each now also modeled by the fake so the
  CPU test suite guards them:
  - **Dense RHS/solution descriptors were created ``ROW_MAJOR``**, which
    cuDSS rejects (`NOT_SUPPORTED` — it supports only column-major dense
    matrices). They are now ``COL_MAJOR`` over Fortran-ordered device
    buffers with ``ld = nrows``.
  - **The LDLᵀ inertia was read as ``int64[2]``**, which cuDSS rejects
    (`INVALID_VALUE`; observed layout is ``int32[2]``). The read now tries
    the observed layout first and falls back to the widened one. Inertia is
    also reported **best-effort** through ``inertia_or_none`` (lazily, one
    host sync at most per factor, symmetric factors only) instead of only
    under ``require_inertia`` — parity with the Feral CPU adapter, so the
    IPM's inertia-guided δ_w correction now engages on the cuDSS route
    (verified on GPU: the saddle-trap QP now escapes to the minimizer).
  - **A cuDSS runtime whose ABI does not match the nvmath-python bindings**
    (e.g. cuDSS 0.8 under nvmath 0.9, which is built against 0.7; nvmath's
    high-level API refuses the pairing, the raw bindings do not check)
    failed every ``matrix_create`` with `INVALID_VALUE`, which the driver's
    regularization ladder retried until reporting an inscrutable
    ``numerical_error`` at iteration 0. Creation-time rejections now
    degrade to the spsolve fallback instead.
  Additionally, **every silent degrade to CuPy spsolve now emits a
  ``RuntimeWarning``** (missing runtime or ABI mismatch): spsolve
  re-factorizes from scratch on every solve, and two real environments hit
  the silent fallback without noticing.
- **The cuDSS symmetric route no longer crashes on CuPy < 14.** The
  compiled full-CSR → lower-triangle map (`compile_lower_triangle`) built its
  nnz→row ownership vector with ``xp.repeat(arange(n), diff(indptr))``;
  NumPy accepts array-valued repeat counts but CuPy only does from v14, so
  the first symmetric factorization through a real cuDSS runtime raised
  ``ValueError: cupy.ndaray cannot be specified as `repeats` argument.`` on
  older CuPy. The map is now derived with ``searchsorted`` over ``indptr``
  (the same construction as the adapters' Gram row maps), which is
  version-independent and avoids materializing the repeat. The fake-CuPy
  test fixture that masked this now models the restrictive ``repeat``
  semantics, and the canonical-map tests run against both a CuPy-semantics
  namespace stub and (GPU-gated) real CuPy.
- **Unbounded detection requires the objective to diverge, not just ‖x‖.**
  The diverging-iterates test fired on the iterate norm alone, misreporting
  convergent problems whose iterates wander astronomically far before
  returning (S2MPJ KOEBHELB: ‖x‖ grows monotonically past 1e22 across ~30
  iterations, then converges to f = 112 — flagged ``unbounded`` on the
  exact/dense and exact/sparse routes in every sweep since v5) or whose
  optimum is legitimately huge. "Unbounded below" means f → −∞, so
  `Status.UNBOUNDED` now additionally requires the objective to sit below
  ``−diverging_iterates_tol``. KOEBHELB exact/dense: ``unbounded`` →
  ``optimal`` (KKT 2.4e-9); genuinely unbounded problems (INDEF,
  f → −3.6e21) are detected at the same iteration as before.
- **The Mehrotra corrector degrades gracefully when the correction hurts.**
  The corrected direction was accepted unconditionally, but its ``−ΔΔ`` term
  presumes a Newton predictor whose full step reduces complementarity — on
  nonconvex or quasi-Newton curvature it can instead collapse the maximal
  fraction-to-boundary step, leaving the solve at a convergence knife-edge
  (S2MPJ HS71 × ``exact/dense+mehrotra``: converged in ~10 iterations or
  stalled at KKT ≈ 36 depending on last-bit arithmetic across Torch builds).
  A corrected step that retains less than half the predictor's combined
  boundary step length is now discarded for the plain centered Newton step
  at the same μ target (the direction ``corrections="none"`` computes),
  falling back to the affine predictor only if that re-solve fails. The
  Gondzio corrector builds on the same guarded base. The HS71
  ``exclude_configs`` workaround in the QC corpus is removed; the gate runs
  200/200 with the corrector configs swept again.
- **Budget exhaustion at a near-optimal iterate now reports `ACCEPTABLE`.**
  The driver already returns the best accepted iterate on failure statuses;
  when that iterate satisfies the relaxed KKT tolerances (1e2 × the
  optimality tolerances, IPOPT ``acceptable_tol``), `MAX_ITER`/`MAX_TIME`
  now report `ACCEPTABLE` like the stall and step-failure salvage paths
  already did — previously a run that oscillated at the acceptable level
  without holding it for the consecutive-iteration window read as a harsh
  budget failure from an essentially optimal point (S2MPJ budget-cluster
  audit: DIAMON2DLS at KKT 6.7e-7 reported ``max_time``).
- **CuPy adapter no longer calls the deprecated ``ndarray.scatter_add``.**
  ``cupyx.scatter_add`` delegates to it and warns as of CuPy 14 (which
  escalates to an error under pytest's warning filter, breaking every CuPy
  ``from_coo`` assembly); the canonical-map scatter now uses its documented
  replacement ``cupy.add.at``.
- **Restoration solves the reduced Gauss-Newton system on the free set.**
  With variables pinned at their bounds, the full-space GN step is dominated
  by the blocked components; the box projection swallowed it and the
  Levenberg–Marquardt loop degraded into a microscopic gradient crawl (S2MPJ
  DRUGDIS: θ 0.19 → 0.16 in a full 200-iteration restoration budget; UBH5
  crawled through nine consecutive budgets). Bound-blocked variables (the
  binding set of the projected-gradient test) now have their rows/columns
  removed from the normal system — projected Newton on the free set
  (Bertsekas 1999, §2.3) — restoring the GN rate: DRUGDIS reaches θ 8e-4 and
  UBH5 drops 4× further per budget, box-stationary certificates arrive in
  seconds instead of full budgets (MESH: 300 s → 3 s), and CSFI2/DEGENLPB/
  CORE2 gain converged finishes on the false-infeasible subset.
- **Uncertified restoration stalls no longer report `INFEASIBLE`.** The
  feasibility restoration now distinguishes *how* it ended: only a
  stationarity-type exit (projected gradient ≈ 0, or no descent at the
  Levenberg–Marquardt damping ceiling) certifies local infeasibility
  (Wächter & Biegler 2006, §3.3). A trailing-window plateau or an exhausted
  iteration budget resumes the main loop while restoration keeps reducing the
  violation between stalls, spends the one-shot x0-anchored probe before
  giving up, and then terminates as the honest `Status.RESTORATION_FAILED`
  (previously never emitted) — an early window exit could relabel an
  out-of-budget failure as a false claim about the problem (S2MPJ:
  LAKES/NASH/SWOPF on the Krylov routes). On the 53-problem false-infeasible
  subset × 6 configs: +5 correct / +9 converged, ~16 false-INFEASIBLE rows per
  config relabeled, zero regressions on previously-optimal runs, and the
  genuinely infeasible control (BURKEHAN) still reports `INFEASIBLE`
  everywhere.
- **CG breakdown on a vanished preconditioned inner product.** ``⟨r, M⁻¹r⟩``
  can underflow to an exact zero with a nonzero residual on an ill-scaled
  problem (MGH09LS under ``preconditioner="auto"`` crashed with
  ``ZeroDivisionError: 0/0`` in the β update); a zero, negative or non-finite
  preconditioned inner product now raises ``KrylovConvergenceError`` so the
  driver escalates δ_w — MGH09LS lbfgs/krylov pc=auto: crash → OPTIMAL.
- **Second-chance restoration anchored at the starting point.** Feasibility
  restoration is a local method: entered from a wandered-off iterate it can
  converge to a nonzero local minimizer of the constraint infeasibility on a
  perfectly feasible problem, and the driver then declared `INFEASIBLE`. The
  2026-07 S2MPJ audit found 28 of the 52 falsely-INFEASIBLE problems (160
  config-rows) restorable directly from the user's `x0` (CATENARY, CRESC4/50,
  ELATTAR, GASOIL, HS39/87/90/91/101/102/111, MESH, ROBOT, SWOPF, TRAINH,
  TRUSPYR1, UBH5, ...). The driver now probes a believed local-infeasibility
  claim once with a restoration anchored at the starting point and resumes the
  main loop when that reaches believable feasibility (CATENARY exact/dense:
  INFEASIBLE → OPTIMAL; TRUSPYR1: INFEASIBLE → ACCEPTABLE). Genuine
  infeasibility verdicts are unaffected — the probe simply fails there too.
- **Restoration Levenberg–Marquardt damping control.** A rejected trial step
  now retries with a larger damping *without* rebuilding the Jacobian and
  Gauss-Newton normal matrix (they belong to the unchanged iterate); the outer
  budget spent on genuinely converging runs grew 80 → 200 iterations, paid for
  by a trailing-window progress guard that exits accept/reject limit cycles
  near a nonzero local minimizer early. Stationarity is now tested on the
  **projected** gradient, so an active-bound stall (MANNE-class) is recognized
  as first-order stationary for the bound-constrained infeasibility problem
  instead of grinding the damping to its ceiling with one Jacobian rebuild per
  projection-swallowed trial.
- **False "locally infeasible" verdicts on feasible problems** (S2MPJ audit:
  52 feasible problems reported INFEASIBLE across the configs, most since
  v8). Two changes: a local-infeasibility claim is now *vetoed* when an
  accepted iterate already reached the verdict's own believe-threshold — a
  diverged endgame (degenerate duals at tiny μ) is a stall at a feasible
  problem, not evidence of infeasibility, and reports `Status.STALLED`
  instead; and every failure status now returns the accepted iterate with
  the lowest scaled KKT error rather than the final iteration's wreckage
  (DEGENLPA: previously INFEASIBLE at a diverged point, now STALLED
  returning its essentially optimal iterate, f = 3.05986 at KKT 1.7e-6).
  The W&B eq. (16) κ_Σ dual clip was prototyped as a further cure and
  measured on the same subset: it fixed nothing and broke CRESC4, so it was
  deliberately not shipped (regression:
  `tests/regression/test_false_infeasible_veto.py`).
- **Iterative KKT solves could freeze the solver at feasible iterates with a
  non-descent direction.** CG on an indefinite condensed operator can
  "succeed" with a garbage direction — success is not a descent certificate —
  and at θ = 0 the filter line search rightly rejects every ascent trial while
  restoration is a no-op, so nothing ever changed (85 of the 126 S2MPJ v9
  `exact/krylov` regressions, mostly `*LS` problems). A step with a
  meaningful positive barrier-directional-derivative at a feasible iterate is
  now treated like a failed factorization: δ_w is escalated and the system
  re-solved until the direction is descent — the Krylov analogue of the dense
  route's Cholesky PD-probe (Wächter & Biegler 2006, §3.1 inertia-correction
  semantics). POWELLBSLS/SANTALS/PALMER2A now solve to the reference optima
  on `exact/krylov` (regression:
  `tests/regression/test_feasible_ascent_stall.py`).
- **Non-monotone μ oracles ran away on ill-scaled problems** (found via the
  radiotherapy example: every free-mode oracle stagnated at a huge constant μ
  with L-BFGS, with or without correctors). Five stacked defects, each with a
  regression test (`tests/regression/test_mu_oracle_inflation.py`):
  the monotone reducer computed κ_μ·μ^θ_μ — which *increases* for μ ≥ 25 and
  locked any inflated μ forever — and is now κ_μ·min(μ, μ^θ_μ), an aggressive
  variant of Wächter & Biegler eq. (7) that keeps the historical (S2MPJ-tuned)
  pace for μ ≤ 1 while decreasing linearly from any inflated magnitude;
  the Mehrotra probing σ = (μ_aff/μ)³ was unguarded above 1
  and exploded μ on a quasi-Newton affine probe (now clipped at 1); the filter
  line search's θ-progress branch degenerated to "0 ≤ 0" at feasible iterates
  (θ = 0) and accepted arbitrary ascent steps (W&B §2.3: only f-type/φ-decrease
  acceptance applies there); the Mehrotra −ΔΔ corrector targets built from a
  low-quality affine direction were unbounded (now clipped into the symmetric
  neighbourhood [γμ, μ/γ], Colombo & Gondzio 2008); and a corrected direction
  that loses the barrier descent property (NWW 2009 §5) is now discarded for
  the plain centered step toward the same μ.
- **S2MPJ benchmark: batched evaluation preserves overflow semantics.** The
  original (and per-element) S2MPJ code raises `OverflowError` at wild
  line-search trial points (Python-float powers via `to_scalar`), which the
  bridge maps to a cleanly rejected trial; the vectorized batch stayed in
  NumPy and silently returned inf/nan instead. A non-finite batched result
  now re-evaluates that point on the per-element path, restoring exact
  behavioral parity (regression test with an overflow-prone generated-style
  element).

### Changed
- **Default μ schedule stays `"monotone"`**, now by measurement: the S2MPJ v10
  paired A/B (full CUTEst matrix, identical flags per arm) had monotone beat
  probing on every solver/Hessian config — 3837 vs 3770 correct overall — an
  aggressive oracle can crash μ faster than the duals converge, stalling just
  above tolerance. `"probing"` remains a supported opt-in (it uniquely solves
  12–18 problems per config that monotone misses). One behavioral change vs
  the previous release survives: Mehrotra/Gondzio corrections now aim at the
  oracle's μ, so with the monotone default they no longer imply the probing
  σ-rule — pass `mu_schedule="probing"` alongside `corrections=` to restore
  the old pairing.
- **S2MPJ benchmark: precompiled Lagrangian Hessian.** The fast S2MPJ evaluator
  (`benchmarks/corpus/_s2mpj_fast.py`) now also replaces the interpretive
  `LgHxy` path used by the exact-Hessian sweep configs: the Hessian's COO
  structure is compiled once per instance and each call only fills values
  (measured ~3–70× per call, median ~10× over a 96-problem sample; e.g. the
  ACOPP14 `exact/dense` solve drops from 1.7 s to 0.9 s with `LgHxy` gone from
  the profile top). Verified at build time against the original `LgHxy` with
  its own independent fallback — a Hessian-only mismatch or an oversized
  structure keeps the fast `fx/cx` and reverts just the Hessian to the
  original (94/96 sampled problems verify; 2 very large ones fall back by the
  structure-size guard).
- **S2MPJ benchmark: `cJx` fills a precompiled CSR layout.** The constraint
  Jacobian's sparsity is fixed per instance, so its canonical CSR
  `indices`/`indptr` and all element/linear-term scatter positions are now
  computed once at build time; each call only fills the `data` vector (no
  per-call `searchsorted`, no COO→CSR sort). ~2× per `cJx` call on
  element-light problems (HS71, DTOC1L), ~20–40% on element-heavy ones.
- **S2MPJ benchmark: per-elftype batched element evaluation.** S2MPJ's
  generated element functions are scalar-coded (`EV_[i,0]`, `np.zeros(dim)`,
  `self.elpar[iel_]`), so the fast evaluator now vectorizes them with a
  mechanical AST rewrite and evaluates all same-type elements in one call per
  type, scattering gradients through precomputed CSR slots (the objective
  gradient becomes support-based in the same stroke). Every batched type is
  verified numerically against its per-element original at build time and
  demoted to the per-element path on any mismatch or unsupported construct;
  the whole-evaluator verification against the original methods still guards
  the composition. On a 96-problem corpus sample all 158 element types (277k
  element occurrences) batch, with no verification fallbacks. Measured:
  ~4–25× per `cx`/`cJx` call over the pre-batching evaluator (ACOPP14
  1.03→0.04 ms; ~25–250× vs S2MPJ's original `evalgrsum`), ~2× per `fgx`,
  and an element-heavy L-BFGS solve runs ~2× faster end-to-end.
- **S2MPJ benchmark: remaining per-call hot spots batched/precomputed.**
  Non-TRIVIAL *group* functions now batch per gftype through the same AST
  transform + verify-or-demote pipeline (the `gL2` least-squares corpus is
  the main beneficiary: LUKSAN11LS `fgx` 15×, DTOC1L 9×, PALMER1C 5×); the
  bridge splits the bundled Jacobian into eq/ineq blocks with precomputed
  signed row selectors (`P @ J` instead of per-call fancy-slicing + vstack)
  and memoizes `(f, ∇f)` per point so a same-point objective request after a
  gradient reuses one `fgx` evaluation; and the fast Lagrangian Hessian's
  COO→CSR reduction (sort order + duplicate-run starts) is precomputed, so
  per-call assembly is a gather + `np.add.reduceat` with no COO sort.

## [0.4.0] - 2026-07-05

### Added
- **Adaptive Krylov tolerance (inexact Newton).** The matrix-free solver now drives
  an Eisenstat–Walker forcing sequence by default: the inner solve is only as tight
  as the current outer KKT residual demands —
  `inner_rtol = clip(adaptive_eta · ‖KKT residual‖, rtol, adaptive_rtol_max)` — so
  early iterations solve loosely (fast, and *robust*: an ill-conditioned initial KKT
  system that the fixed `rtol=1e-10` could never reach — the inner CG/MINRES/GMRES
  just fails and triggers a δ_w runaway → `numerical_error` — now yields a usable
  step) and tighten toward `rtol` as the IPM converges. New `KrylovOptions`:
  `adaptive_tol` (default `True`), `adaptive_eta` (`0.1`), `adaptive_rtol_max`
  (`1e-8`, calibrated to the outer tol — a looser cap drives step-sensitive IPM
  problems into infeasibility). Set `adaptive_tol=False` to force the fixed `rtol`.
  The `LinearSolver` protocol gained a `set_outer_residual` hint (no-op for the
  direct solvers). On the S2MPJ Krylov route this both *solves* problems that
  previously errored (e.g. MRIBASIS: `numerical_error` → `optimal`) and *speeds up*
  ones that timed out chasing 1e-10 (NET1/LAUNCH/CLNLBEAM: `max_time` → `optimal`).
- **Unbounded-problem detection.** An IPOPT-style diverging-iterates test now reports
  `Status.UNBOUNDED` (previously in the enum but never emitted) when `‖x‖_∞` exceeds
  the new `Options.diverging_iterates_tol` (default `1e20`, `None` disables). An
  unbounded-below objective drove the iterate off to infinity while the KKT residual
  never fell and only surfaced as a misleading `numerical_error` once the runaway
  iterate overflowed to non-finite (e.g. the CUTEst INDEF problem: objective marching
  to `-1e155`); it is now stopped early and labelled honestly (INDEF → `unbounded` at
  iteration 27 instead of `numerical_error` at 186).
- Opt-in **block-Schur preconditioner** for the matrix-free Krylov route on
  equality saddles: `KrylovOptions(preconditioner="lbfgs")` now builds the
  block-diagonal `diag(N⁻¹, S⁻¹)` (Murphy–Golub–Wathen) on `_SaddleOperator` —
  the L-BFGS-aware Woodbury inverse `N⁻¹` on the (1,1) block and the reciprocal
  approximate-Schur diagonal on the (2,2) block. Because it is non-diagonal, the
  default `cg` route switches to **GMRES** (which left-applies it) for the saddle
  when it is available; MINRES admits only a diagonal. It clusters the saddle
  spectrum far better than the diagonal Jacobi block and cuts iterations on
  ill-conditioned equality-constrained systems once L-BFGS has curvature pairs.
  Purely opt-in: the default preconditioner remains `jacobi`, so default behavior
  is unchanged, and it degrades to Jacobi before the first curvature pair or with
  a non-L-BFGS Hessian.
- Adaptive **`KrylovOptions(preconditioner="auto")`** mode that self-tunes between
  the two preconditioners: it starts on the cheap Jacobi diagonal and promotes to
  the L-BFGS preconditioner the first time a solve struggles. Two triggers, with
  deliberately asymmetric aggressiveness so promotion never regresses a solve:
  a **convergence failure** is rescued by retrying the same solve with *any*
  available L-BFGS structure (the alternative is a definite failure and δ_w
  escalation), whereas a merely **slow success** (more than the new
  `auto_switch_ratio` fraction, default `0.5`, of the iteration budget) promotes
  only to the *near-exact condensed Woodbury inverse* `N⁻¹`. The approximate
  saddle block preconditioner is never promoted speculatively — on a
  rank-deficient equality Jacobian its approximate Schur diagonal can yield worse
  steps than the slow-but-stable Jacobi solve (observed on the ACOPP power-flow
  cluster). Promotion is sticky for the life of the solver instance (one IPM run),
  so ill-conditioned systems get the stronger preconditioner while well-conditioned
  ones never pay the per-iteration Woodbury cost. The default remains `jacobi`;
  `auto` is opt-in.
- The S2MPJ / CUTEst benchmark harness now scores a second **`converged`** tier
  alongside `correct`: a case that reaches a valid KKT point (a small scaled KKT
  residual at a success status — which already bounds primal infeasibility)
  counts as converged regardless of *which* optimum it found, so a solve to a
  different local minimum on a nonconvex problem is credited as genuine
  convergence rather than a failure. `correct` (which additionally matches the
  dataset-documented objective) is a strict subset. Both counts appear in the
  runner summary, the per-config table, and the report header; the per-case table
  flags a converged-but-not-`correct` case with `≈` (and a non-converged one with
  `⚠️`). Pure benchmark/reporting change — no solver behavior is affected.
- **Precompiled S2MPJ evaluator** (`benchmarks/corpus/_s2mpj_fast.py`): profiling
  the S2MPJ sweep showed ~90% of solve wall-time inside `s2mpjlib`'s interpretive
  `evalgrsum` loop (per-element `eval()` string dispatch, `lil_matrix` row
  assembly, per-group sparse slicing of the linear term). The bridge now compiles
  the group-partially-separable structure once per instance — element/group
  functions resolved via `getattr`, all linear terms as one `A @ x − gconst`
  product, Jacobians assembled as COO triplets on the precomputed support — while
  the generated element/group math itself is untouched. It is **verified against
  the original methods at build time** (start point + a perturbed point) and
  falls back to them on any unsupported feature or mismatch, so a corpus oddity
  cannot corrupt scores. Measured: 8–130× per constraint evaluation, 6–37×
  end-to-end solves (AIRPORT 68.8 s → 1.9 s, BATCH 38.5 s → 1.7 s); many former
  `max_time` rows now run to completion. Benchmark-side only.
- The S2MPJ sweep runner gained **`--jobs N`**: problems fan out over worker
  processes (each worker runs one problem's whole config matrix, keeping the
  shared instance + verified evaluator amortized). Reports are now flushed in
  sorted row order, so they are deterministic and diff cleanly regardless of
  completion order; the per-problem crash-surviving flush is kept, and on a
  worker's native crash the `.inflight` file lists the candidate culprits for
  `--resume --exclude`.
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
- **Acceptable-level termination is now enabled by default** (IPOPT convention:
  `acceptable_tol = 1e-6` — 1e2 × the optimality tolerance — held for
  `acceptable_iter = 15` consecutive iterations). The mechanism existed
  (`AcceptableStoppingOptions`, `Status.ACCEPTABLE`) but every tolerance defaulted
  to `None`, so a solve whose achievable KKT floor sits between 1e-8 and 1e-6 —
  degenerate optima, ill-conditioned least squares — ground thousands of
  iterations into `max_iter`/`max_time` at an essentially optimal point (v6 S2MPJ
  sweep: PALMER1A spent 300 s at kkt 4.2e-8; it now stops ACCEPTABLE at the same
  point in 1.2 s). New defaults: `dual_inf_tol = constr_viol_tol = compl_inf_tol
  = 1e-6`, `n_iter = 15`; set all three to `None` to restore the old
  grind-to-the-cap behavior.
- Cheaper S2MPJ exact-Hessian bridge: under gradient-based scaling (`σ = s_f ≠ 1`)
  the bridge assembled `σ∇²f + Σ y∇²c` with **two** `LgHxy` calls (one for the
  `(σ−1)∇²f` correction); it now folds the scaling into the multipliers via
  `σ·LgHxy(x, Y/σ)` for `σ > 0` — one interpretive Hessian assembly instead of
  two. The bridge also seeds its constraint-value memo from `cJx`'s returned
  `c(x)`, so the driver's same-point value requests after a Jacobian evaluation
  skip a full `cx` call. Benchmark-side only.
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
- **δ_c escalation no longer contaminates a pure-δ_w repair** (regression in the
  rank-deficient-∇c fix, caught by the v6 S2MPJ sweep: HS61 went from optimal at
  kkt 1e-14 in 12 iterations to cycling at kkt ~1e3 until `max_time`). Exploded
  equality multipliers can leave ~1e8–1e9 negative curvature in the primal block;
  that is repaired by δ_w *alone* (HS61: δ_w = 2.3e9), and the resulting dual step
  is exactly what heals the multipliers. With `delta_c_trigger = 1e4`, δ_c grew
  alongside δ_w during that climb, so the eventually-successful factorization
  carried a meaningful δ_c whose relaxed dual step never repaired the multipliers.
  The regularization ladder is now **two-phase**: phase 1 escalates δ_w alone
  (`delta_c_trigger` default raised 1e4 → 1e10, above any legitimate primal
  repair); only when that ladder is exhausted is the failure attributed to a
  singular dual block, and phase 2 then *resets δ_w to its floor* and escalates
  (small δ_w, growing δ_c) together — a huge leftover δ_w would poison the very
  solve δ_c is meant to rescue. The rank-deficient motivations are preserved and
  improved: ACOPP14 now reaches OPTIMAL in ~15 iterations on the exact routes
  (the phase-2 steps carry a small δ_w instead of ≥1e4), and BT1/DISC2 stay
  optimal. Regression test: HS61 (gated) alongside the existing BT1/DISC2 case.
- **Matrix-free MINRES failures on equality saddles now fall back to GMRES** instead
  of surfacing as `numerical_error`. On the default `method="cg"` route an equality
  saddle is solved with MINRES, which is fragile on ill-conditioned indefinite
  saddles: it can return a garbage/non-finite step at iteration 1, which the driver
  answers with an unbounded (and, on the iterative route, counterproductive) `δ_w`
  escalation to `1e40` → `numerical_error` on step 1. GMRES — minimizing the true
  residual with restarts — is markedly more robust here, and because the SPD
  *diagonal* (incl. the approximate-Schur dual block) actively hurts GMRES on these
  saddles, the fallback runs **unpreconditioned**. Recovers a large slice of the
  S2MPJ optimal-control cluster on the Krylov route (HAGER*/DTOC*/CATENARY:
  `numerical_error` → `optimal`). An explicit `method="minres"` is honored verbatim
  (no fallback); the default `cg` route gets it automatically.
- **`preconditioner="auto"` no longer regresses equality-constrained saddles.** The
  failure-rescue trigger previously promoted a struggling saddle solve to the
  *approximate* block preconditioner, whose approximate Schur diagonal can diverge a
  solve plain Jacobi handles (S2MPJ HS109/CLNLBEAM/DISC2/EIGMAXA turned `optimal` →
  `numerical_error` under auto). Auto now only ever promotes to the **near-exact
  condensed Woodbury inverse** (equality-free) under *either* trigger — never the
  saddle block — so it is `≡ jacobi` on saddles (zero regressions) while keeping the
  equality-free acceleration. Saddle robustness is the GMRES fallback's job above;
  the block preconditioner stays reachable via explicit `preconditioner="lbfgs"`.
- The KKT-solve regularization loop now escalates the **dual regularization `δ_c`**
  alongside `δ_w` on a failed saddle solve. `δ_w` regularizes only the (1,1)
  primal block, so a **rank-deficient equality Jacobian** — e.g. AC optimal
  power-flow's reference-bus degeneracy — left the bordered saddle singular in the
  (2,2) dual block, and the solver escalated `δ_w` to `1e27` uselessly (into
  numerical singularity) before reporting `numerical_error` at iteration 1. `δ_c`
  now grows (capped by `RegularizationOptions.delta_c_max`, default `1e-1`) within
  the failing solve and resets on the next step (Wächter & Biegler 2006, §3.1).
  This unsticks the S2MPJ lbfgs/krylov `numerical_error` cluster (ACOPP*/ACOPR*
  now make progress instead of dying on step 1) and fixes solves with redundant /
  linearly-dependent equality constraints on every route. `δ_c` escalation is a
  **last resort**, gated on `δ_w` first growing past `delta_c_trigger` (default
  `1e4`): a rank-deficient `∇c` leaves `δ_w` running to its ceiling uselessly (only
  `δ_c` repairs the (2,2) block), whereas an ordinary indefinite (1,1) block is
  fixed by a *small* `δ_w`. Escalating `δ_c` while `δ_w` is still small perturbs the
  (2,2) block needlessly and — on the inertia route — changes the very inertia the
  check tests against, which diverged well-conditioned equality problems (e.g.
  BT1/DISC2/FLETCHER on the exact routes). The trigger keeps `δ_c` away from them
  while still activating for the genuinely rank-deficient case.
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
- The GMRES fallback no longer crashes with a raw `ZeroDivisionError` on a
  **singular KKT system** (QC gate: HS9 with `hessian="exact", linsolve="krylov"`
  from the standard start, where the exact Lagrangian Hessian is identically zero
  so the first condensed (1,1) block is the zero operator). When the operator
  image of a basis vector vanishes after orthogonalization, the rotated
  Hessenberg diagonal is exactly zero; the back-substitution now takes the
  minimal-norm choice for that rank-deficient column and lets the true-residual
  check raise the controlled `KrylovConvergenceError`, so the driver escalates
  δ_w and recovers (HS9: OPTIMAL in 5 iterations). The Jacobi preconditioner
  build also skips its `1/diag(N)` divide-by-zero warning for a non-positive
  primal diagonal it was going to discard anyway.
- The solver no longer reports a **feasible iterate as `infeasible`**. Feasibility
  restoration can stall a hair above its own (differently scaled) tolerance at a
  point that is feasible by the driver's constraint-violation metric — a
  degenerate optimum where constraint qualification fails (HS13), or a limit cycle
  that keeps re-reaching feasibility (HS56) — and then declared local
  infeasibility, which contradicts the point being feasible. The restoration
  verdict is now believed only when the restored iterate's violation genuinely
  exceeds the constraint-violation tolerance (by `1e3×`, far below any
  bounded-away-from-zero infeasible stationary point); otherwise the main
  iteration resumes from the feasible point. This clears the false `infeasible`
  on HS13/HS56/HS72/HS102 in the S2MPJ sweep — several now converge to the true
  optimum once the limit cycle is broken — while genuinely-infeasible problems
  (e.g. `BURKEHAN`) are still detected.

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

[Unreleased]: https://github.com/wahln/ipax/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/wahln/ipax/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/wahln/ipax/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/wahln/ipax/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/wahln/ipax/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/wahln/ipax/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/wahln/ipax/releases/tag/v0.1.0
