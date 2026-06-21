# AGENTS.md — `ipax`

Operational guide for any agent or contributor working in this repository. This is
the **canonical** instructions file; `CLAUDE.md` imports it.

`ipax` is a pure-Python, **Python Array API–conformant** primal–dual interior-point
solver for large-scale nonlinear constrained optimization, tuned for radiotherapy
(RT) treatment-planning scale. The math, the KKT reduction, and the design rules
that govern changes live in this file; cite the source papers (see References) for the
algorithmic details.

---

## What this project is

- General NLP: `min f(x)` s.t. `c(x)=0`, `g(x)≤0`, `x_L ≤ x ≤ x_U`, solved by a
  log-barrier interior-point method with Lagrange multipliers.
- **Scale target:** `1e3`–`1e5` variables; dense *or* sparse Jacobians/Hessians.
- **Default Hessian:** L-BFGS (compact, Powell-damped) of the Lagrangian.
- **Globalization:** filter line-search (IPOPT, default); Breedveld step controller
  as an alternative mode.
- **Backends:** NumPy + PyTorch in CI; CuPy/JAX (incl. GPU) supported via the same
  code path. Inspirations: IPOPT (Wächter & Biegler 2006), Breedveld et al. (2017).

---

## Non-negotiable invariants (read first)

1. **No concrete array library in the core.** Never `import numpy`/`torch`/`cupy`/
   `jax` inside `ipax/` except in `ipax/backend/sparse/` and other explicitly
   labeled adapters. Get the namespace from the input arrays:
   ```python
   from ipax.backend.namespace import array_namespace
   xp = array_namespace(x)          # then use xp.*, xp.linalg.*
   ```
2. **Stay inside the standard.** Use only the Array API main namespace + the optional
   `linalg` extension. The extension provides `cholesky, eigh/eigvalsh, qr, svd,
   solve, inv, pinv, slogdet, matrix_norm, vector_norm, …`. It does **NOT** provide
   LDLᵀ-with-inertia, triangular solve, `lstsq`, or any sparse type — do not assume
   them. Gap-fillers go in `backend/` with a comment naming the missing primitive.
3. **Linear algebra is injected, never hard-wired.** All Jacobians/Hessians are
   `LinearOperator`s; all KKT solves go through the `LinearSolver` protocol. Adding a
   solve strategy must not touch `ipm/driver.py`.
4. **Sparsity is an adapter concern.** The core emits structure as Array-API
   index/value vectors; per-backend wrappers in `backend/sparse/` build the actual
   sparse matrix and factor it. The IPM never sees a backend sparse object.
5. **No `localStorage`-style global state.** Solver state lives in explicit objects;
   no module-level mutable singletons.

If a change appears to require breaking 1–4, stop and discuss in the PR description
instead of working around it.

---

## Architecture map

```
ipax/
  typing.py options.py result.py solve.py
  problem/      base.py (Problem ABC), function.py, derivatives.py,
                finitediff.py, autodiff/{jax,torch}.py
  backend/      namespace.py, operators.py,
                sparse/{numpy_scipy,cupy,torch,jax,_routing}.py
  linalg/       solver.py, dense.py, krylov.py, regularize.py
  ipm/          barrier, kkt, step, filter_ls, restoration, breedveld_ls,
                hessian, init, driver
  testing/      problems.py (analytic oracles), backends.py
tests/          contracts/, unit/, property/, integration/, backends/, regression/
examples/       minimal runnable examples for the current implementation surface
benchmarks/     generators/, corpus/, harness, runners, reports
```

- **`Problem`** (`problem/base.py`): user-facing ABC. Required: `n_vars`,
  `objective`. Optional (resolved by `derivatives.py`): `gradient`, nonlinear
  `eq/ineq_constraints` + Jacobians, and `lagrangian_hessian`. **Linear constraints
  are declared separately** (`linear_eq`/`linear_ineq`, constant data) from nonlinear
  ones — constant Jacobian, no Hessian term, assembled once. Derivative precedence:
  analytic → autodiff → finite-diff (grads/Jacobians); analytic → autodiff-HVP →
  **L-BFGS** (Hessian). Never form a dense Hessian at scale.
- **`LinearOperator`** (`backend/operators.py`): `matvec`/`rmatvec`/`matmat`;
  subclasses `Dense, Diagonal, LowRank, LBFGSOperator, MatrixFreeJacobian,
  SparseOperator, Composite`. All `Problem` Jacobians/Hessians normalize to these.
- **`LinearSolver`** (`linalg/solver.py`): `DenseSolver` (Cholesky/solve),
  `KrylovSolver` (CG/MINRES/GMRES, matrix-free — default at scale), and
  `SparseDirectSolver` (per-backend). Auto-selected by size/density/capabilities.

---

## Math conventions (match the code to these)

- Standard form uses slacks: `g(x)+s=0, s≥0`. Multipliers: `y` (equalities),
  `λ` (inequalities/slacks), `z_L,z_U` (bounds). `diag(v)=V`; `e`=all-ones.
- Two KKT solve routes: the **condensed normal-equations** route (Array-API-native,
  default for dense & matrix-free; PD via Powell-damped L-BFGS + primal–dual
  regularization, so **no inertia oracle required**) and the **indefinite augmented**
  route (sparse-direct backends with LDLᵀ inertia only).
- Regularization: Friedlander–Orban primal–dual (`δ_w` on (1,1), `−δ_c` on (2,2));
  on Cholesky failure escalate `δ_w` from `1e-6`, doubling, per Breedveld.
- Convergence: scaled KKT ∞-norm ≤ `ε_tol` (default `1e-8`), IPOPT scaling
  `s_d, s_c`.

Keep symbol names in code aligned with the math conventions above (`W, Sigma_x,
Sigma_s, N, mu, tau, alpha, theta, phi`).

---

## Coding standards

- Python ≥ 3.10, full type hints, `from __future__ import annotations`.
- Public API documented with docstrings; numerical choices cite the paper/eq.
- Config via frozen dataclasses in `options.py`; no magic numbers in the loop body.
- Formatting/lint: `ruff` (format + lint) and `mypy` clean before commit.
- Pure functions where practical; no hidden global state; explicit `xp`/device
  threading.
- Floats are `float64` by default; never hard-code a dtype — read it from inputs.

---

## Testing & verification

- **Test-driven.** For every change write the failing tests first, then implement
  to green. New protocol implementations must pass the shared **contract test
  batteries** (`tests/contracts/`) for `Problem`, `LinearOperator`, `LinearSolver`,
  `SparseDirectSolver` — this is what keeps "pluggable" honest.
- **Always multi-backend.** Tests parametrize over the namespace fixture
  (`tests/conftest.py`): NumPy + PyTorch minimum; CuPy/JAX/GPU when available.
  `array-api-strict` is the **purity gate** (raises on out-of-standard calls).
- **Purity check** in CI: fail if banned imports appear outside allowed adapters.
- **Oracles:** verify KKT conditions at the solution; closed-form QP/LP optima,
  Hock–Schittkowski set, cross-check vs `scipy.optimize` and `cyipopt` when installed.
- **Derivative harness:** FD vs analytic vs autodiff grad/Jacobian + Hessian-vector
  checks (also a public utility usable on any user `Problem`).
- **Benchmarks are separate** (`benchmarks/`, not `tests/`): synthetic RT-like
  generators (`1e3`–`1e5` vars, 5–50% density) + standard sets, tracked over commits
  with `asv`; run nightly, not per-PR.
- A non-trivial change is not done until the relevant tests pass on **both** NumPy
  and PyTorch. Add a regression test with every bug fix.

### Commands

`scripts/check.py` is the **single verification entrypoint** — it runs the same
gates CI runs (`format → lint → types → purity → test`), so "is my change done?"
is one command instead of reconstructing flags from `ci.yml`, the pre-commit
config and this file. Prefer it over invoking the tools by hand.

```bash
python scripts/check.py                      # all gates (multi-backend tests included)
python scripts/check.py --fast               # skip the slow test gate (lint/types/purity only)
python scripts/check.py lint types           # only the named gates
python scripts/check.py --list               # show the gates and their commands

# Equivalent raw invocations (what the gates wrap):
pytest -q                                    # full test suite
IPAX_BACKENDS=numpy,torch,array_api_strict pytest -q
ruff check . && ruff format --check . && mypy ipax
python scripts/check_purity.py               # import-purity gate (invariants #1/#4)
asv run                                      # tracked perf benchmarks (benchmarks/)
```

Enforcement layers (same `check_purity.py` logic, different triggers): the
**pre-commit** hook runs it at commit; a **Claude Code PostToolUse hook**
(`scripts/hooks/purity_guard.py`, wired in `.claude/settings.json`) runs it the
moment an edit touches `ipax/`, so a purity violation is caught immediately
rather than at commit. Other agents (e.g. aider's `lint-cmd`) can point at the
same script.

### Agent tooling

Claude Code config lives in `.claude/` and is checked in:

- **`invariant-auditor`** subagent (`.claude/agents/invariant-auditor.md`) — a
  read-only reviewer that audits a diff against the five invariants, the math/citation
  conventions, the testing discipline, and the GPU/device-efficiency rules (the
  *semantic* checks the purity hook and contract tests cannot see). Invoke it
  explicitly before a commit or PR — e.g. *"use the invariant-auditor to review my
  changes"*.
- **`/verify`** (`.claude/commands/verify.md`) — run `scripts/check.py` and summarize.
- **`/tdd`** (`.claude/commands/tdd.md`) — drive a change through the mandated
  red→green→verify→regression loop, multi-backend and invariant-aware.

The auditor's rubric is the static half of GPU performance work; the measured half is
a future `benchmarks/runners/device_efficiency.py` (sync/kernel profiling on real
hardware), deferred until GPU CI exists.

---

## Scope guardrails

**In:** equality + inequality + bound constraints; L-BFGS + exact Hessian; dense,
matrix-free, and sparse-direct solver routes; filter line-search + restoration;
optional Mehrotra–Gondzio higher-order corrections; multi-backend.

**Out (do not add without discussion):** Breedveld's dose-matrix `N=AᵀDA+Q+T`
condensation, graph permutation/tiling, mixed-precision tiled BLAS, chained
dose-influence products; hard-coded RT cost-functions (LTCP/gEUD/dose-volume);
multi-criteria / Pareto / beam-angle layers. Keep dimensionality & sparsity in
mind, but the RT-specific kernels belong in a separate downstream layer.

---

## Direction

`ipax` is **beta**. The solver surface above is complete; near-term work
is **maintenance and measurement**, not new features. The project is also actively
**looking for adopters** — early real-world use on the `Problem`/`solve` API is what
drives the road to `1.0.0`.

### Road to 1.0.0

`1.0.0` is the point where the public API carries a semver stability promise. It is
gated on these exit criteria (the day-to-day work below is *how* we get there, not a
new feature list):

1. **API stability.** The public surface — `solve`, `Problem`, `Options`, `Result`,
   and the `LinearOperator`/`LinearSolver` protocols — holds stable across the `0.x`
   series, with any deprecations carried for at least one minor release.
2. **Published documentation.** The MkDocs site (`docs/`: API reference + guides) is
   built and published.
3. **GPU & sparse-direct verification.** The CuPy/cuDSS and Torch/JAX-CUDA sparse
   routes and the MUMPS inertia path are exercised on real hardware in CI, or
   documented as provisional until GPU CI exists.
4. **Adopter validation.** At least one external workload runs on `ipax`, with the
   feedback folded back into the API.
5. **Tracked baselines.** `asv` performance baselines and the QC accuracy sweep are
   tracked across releases, with cross-checks vs IPOPT/SciPy/OSQP green.

Near-term work is **maintenance and measurement**, not new features:

- **Bug-fixing.** Tighten correctness on hard nonconvex/ill-conditioned problems;
  every fix ships with a regression test (`tests/regression/`).
- **Performance.** Drive the `benchmarks/` scaling and micro suites (`asv`), reduce
  per-iteration cost and memory across the dense/matrix-free/sparse routes, and
  profile device efficiency once GPU CI exists.
- **Accuracy benchmarking.** Grow the QC sweep and reference cross-checks
  (vs IPOPT/SciPy/OSQP), track scaled-KKT accuracy and robustness across the corpus
  and backends over time.

New algorithmic features beyond the current scope are out until discussed.

---

## Documentation

Documentation lives in the folder docs and uses mkdocs.

Minimal runnable examples live in `examples/`. Keep them short, feature-scoped,
and runnable from the repository root with the local virtual environment. Examples
may import a concrete backend such as NumPy at the application edge; the `ipax/`
core must remain backend-agnostic.

---

## References

- Wächter, A. & Biegler, L. T. (2006). "On the implementation of an interior-point
  filter line-search algorithm for large-scale nonlinear programming."
  *Mathematical Programming* 106(1), 25–57. <https://doi.org/10.1007/s10107-004-0559-y>
- Breedveld, S., van den Berg, B. & Heijmen, B. (2017). "An interior-point
  implementation developed and tuned for radiation therapy treatment planning."
  *Computational Optimization and Applications* 68(2), 209–242.
  <https://doi.org/10.1007/s10589-017-9919-4>. TROTS dataset <http://www.trots.eu>.
- Array API `linalg` extension:
  <https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html>
- `array-api-compat` / `array-api-extra`:
  <https://data-apis.org/array-api-compat/> · <https://data-apis.org/array-api-extra/>
- Friedlander, M. P. & Orban, D. (2012). "A primal–dual regularized interior-point
  method for convex quadratic programs." *Mathematical Programming Computation* 4(1),
  71–107. <https://doi.org/10.1007/s12532-012-0035-2>
