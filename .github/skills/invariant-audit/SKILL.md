---
name: invariant-audit
description: >-
  Audit a code change to the ipax interior-point solver against its five
  non-negotiable architectural invariants, its math/citation conventions, its
  GPU/device-efficiency rules, and its testing discipline. Use this whenever
  reviewing a diff that touches anything under ipax/ (the core solver
  package) — it catches semantic violations that lint/type/purity checks
  cannot see, such as a NumPy-only attribute slipping through an
  Array-API-generic code path, or a concrete solve hard-wired into the
  iteration loop.
---

# Invariant audit

You are reviewing a change to `ipax`, a pure-Python, Array API–conformant
primal–dual interior-point solver for large-scale constrained optimization
(radiotherapy-treatment-planning scale). Full project context is in
`AGENTS.md` at the repository root — read it if you haven't. This skill is
the review checklist; it does not replace reading the actual diff and
surrounding code.

## Scope

Review only what the diff introduces or modifies. Don't relitigate
pre-existing code the diff doesn't touch, and don't invent issues — if a
change is clean against every check below, say so plainly rather than
padding the review.

## 1. The five non-negotiable invariants

1. **No concrete array library in the core.** Beyond a literal `import
   numpy`/`torch`/`cupy`/`jax`/`scipy` (which `scripts/check_purity.py`
   already catches), look for it smuggled in less directly: a helper that
   returns a concrete-backend object, an assumption about a NumPy-only
   attribute or method, `np.`-style calls copy-pasted into generic code.
   The namespace must come from `array_namespace(x)`
   (`ipax/backend/namespace.py`) and only `xp.*` / `xp.linalg.*` may be
   used. **Exception:** `ipax/backend/sparse/**`, `ipax/backend/dense/**`,
   and `ipax/problem/autodiff/**` are the labeled adapters and may import
   concrete libraries — but even there, flag a concrete object escaping
   back across the adapter boundary into `ipm/` or another core module.
2. **Stay inside the Array API standard** (main namespace + the `linalg`
   extension: `cholesky, eigh/eigvalsh, qr, svd, solve, inv, pinv, slogdet,
   matrix_norm, vector_norm, …`). It does **not** provide LDLᵀ-with-inertia,
   triangular solve, `lstsq`, or any sparse type. Flag use of any of those
   outside the adapters, and flag `out=`/in-place ops or non-standard fancy
   indexing. A genuine gap-filler belongs in `backend/` with a comment
   naming the missing primitive.
3. **Linear algebra is injected, never hard-wired.** All Jacobians/Hessians
   should be `LinearOperator`s (`ipax/backend/operators.py`); all KKT
   solves should go through the `LinearSolver` protocol
   (`ipax/linalg/solver.py`). Flag any concrete factor/solve call reaching
   directly into `ipax/ipm/driver.py` or another loop body — adding a new
   solve strategy must not require touching the driver.
4. **Sparsity is an adapter concern.** The core (`ipm/`, `linalg/solver.py`'s
   public surface) should only see Array-API index/value vectors describing
   structure; only `backend/sparse/` adapters build and factor an actual
   sparse matrix. Flag the IPM or core `linalg/` code touching a backend
   sparse object directly.
5. **No module-level mutable global state.** Flag global singletons or
   mutable module-level caches. Solver state lives in explicit objects
   threaded through the call chain, not `localStorage`-style globals.

If the change *appears* to require breaking invariant 1–4 to work, that
requirement is itself the finding — raise it for discussion in the review,
don't let a workaround merge silently.

## 2. Math & citation conventions

- Symbols match `AGENTS.md`'s math conventions: `W, Sigma_x, Sigma_s, N, mu,
  tau, alpha, theta, phi`; standard form with slacks `g(x)+s=0, s≥0`;
  multipliers `y` (equalities), `λ` (inequalities/slacks), `z_L, z_U`
  (bounds).
- Every algorithmic step cites the paper/equation it implements, e.g.
  `# Wächter & Biegler 2006, eq. (19)` or `# Breedveld 2017, eq. (18)`.
  Flag uncited numerical choices and magic numbers in the loop body —
  configuration values belong in the frozen dataclasses in
  `ipax/options.py`.
- Regularization follows Friedlander–Orban primal–dual (`δ_w` on the (1,1)
  block, `−δ_c` on the (2,2) block; escalate `δ_w` from `1e-6`, doubling, on
  Cholesky/factorization failure per Breedveld). Convergence is measured as
  the scaled KKT ∞-norm against `ε_tol`. Flag deviations from this scheme
  that aren't explained.

## 3. GPU / device efficiency

The solver runs the same code path on CuPy/JAX/Torch-CUDA; the iteration
loop is where Array-API GPU performance is won or lost. Flag, at low cost
and without demanding an algorithm change:

- **Host–device sync in the hot loop:** `float(x)`, `.item()`, `bool(x)`,
  `int(x)`, or a Python `if`/`while` on a 0-d device-array value inside the
  driver's iteration. Scalars like `theta/phi/alpha/kkt_error` should stay
  0-d device arrays as long as possible, read to Python **once per
  iteration**, consolidated — not repeatedly.
- **Needless materialization or host transfer:** `asarray` round-trips, or
  converting to a Python list/scalar and back for no functional reason.
- **dtype promotion / hard-coded dtype:** mixing float32/float64, or a
  literal `float64` instead of reading the dtype from the inputs (floats
  are float64 by default in this project, but it must come from the data,
  never be hard-coded).
- **Allocation churn:** obviously avoidable temporaries in the loop body
  (the standard has no `out=`, so the bar is "avoidable", not "zero").
- **JAX-lazy hazard:** note when new data-dependent Python control flow will
  force materialization on a lazy/JIT backend — this is structural, so flag
  it rather than demanding an immediate fix.

## 4. Testing discipline

- New solver behavior ships with failing-first tests parametrized over the
  namespace fixture (NumPy + PyTorch minimum, per `tests/conftest.py`).
- New `Problem`/`LinearOperator`/`LinearSolver`/`SparseDirectSolver`
  implementations exercise the matching shared battery in
  `tests/contracts/`.
- Bug fixes ship a regression test under `tests/regression/`.
- Flag core behavior added with no multi-backend test coverage.

## Output format

Group findings by severity, most severe first. For each: `path:line` — the
invariant number, paper citation, or `AGENTS.md` section it violates — what
is wrong, why it matters, and the concrete fix.

```
### Blockers (must fix before merge)
- ipax/ipm/driver.py:142 — [invariant #3] hard-wires a Cholesky solve in
  the loop instead of routing through the injected LinearSolver. This
  breaks pluggability — a new solve strategy would require editing the
  driver. Fix: call self._solver.solve(...) instead.

### Warnings
- ...

### Nits
- ...

**Verdict:** clean / N blockers, M warnings.
```

Be specific and cite the invariant number, paper equation, or `AGENTS.md`
section every time — a finding without a citation back to the rule it
breaks is not actionable.
