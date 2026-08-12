# Copilot review instructions — ipax

`ipax` is a pure-Python, Array API–conformant primal–dual interior-point solver.
The canonical project rules live in [`AGENTS.md`](../AGENTS.md) (root) — read it
before reviewing; the notes below are review-specific emphasis, not a
replacement for it.

## What to prioritize

1. **The five non-negotiable invariants** (`AGENTS.md`, "Non-negotiable
   invariants"): no concrete array library in the core, stay inside the Array
   API standard + `linalg` extension, linear algebra injected via the
   `LinearOperator`/`LinearSolver` protocols (never hard-wired in `ipm/
   driver.py`), sparsity confined to `backend/sparse/` adapters, no
   module-level mutable state. Use the `invariant-audit` skill for the
   detailed checklist — it catches semantic violations (e.g. a NumPy-only
   attribute slipping through `xp.*`) that grep-based hooks miss.
2. **Citations.** Every algorithmic step (barrier update, regularization,
   line-search/filter logic, KKT construction) should cite the source
   paper/equation, e.g. `# Wächter & Biegler 2006, eq. (19)`. Flag uncited
   numerical choices and magic numbers in the loop body — those belong in the
   frozen dataclasses in `options.py`.
3. **Symbol conventions.** Code should use the math conventions' symbol names
   (`W, Sigma_x, Sigma_s, N, mu, tau, alpha, theta, phi`) rather than
   ad hoc renamings, so the code stays traceable to the math.
4. **Testing discipline.** New solver behavior needs failing-first tests
   parametrized across backends (NumPy + PyTorch minimum; see
   `tests/conftest.py`'s namespace fixture). Bug fixes need a regression test
   under `tests/regression/`. New `Problem`/`LinearOperator`/`LinearSolver`
   implementations need to pass the relevant `tests/contracts/` battery.
5. **Scope guardrails.** `AGENTS.md`'s "Scope guardrails" section lists
   what's explicitly out (RT-specific cost functions, dose-matrix
   condensation, Pareto/beam-angle layers, custom tiled-BLAS kernels, etc.).
   Flag new code that drifts into that territory as a design question, not a
   silent addition.

## What NOT to flag

- Concrete backend imports (`numpy`, `torch`, `scipy`, sparse types) inside
  `ipax/backend/sparse/`, `ipax/problem/autodiff/`, `examples/`, or
  `benchmarks/` — those are the designated adapter/application-edge
  boundaries where a concrete library is expected.
- Style already enforced mechanically by `ruff`/`mypy`/`scripts/
  check_purity.py` — focus review effort on what those gates can't see
  (semantics, citations, test coverage, architectural drift), not on
  formatting.

## Verification

If your review environment can execute commands, prefer running
`python scripts/check.py` (the single verification entrypoint — format,
lint, types, purity, tests) over reconstructing the gates by hand; see
`.github/workflows/copilot-code-review.yml` for the environment it needs.

## Tone

Be specific: cite the invariant number, paper/equation, or `AGENTS.md`
section for every finding. If a change is clean, say so — don't invent
issues to fill a review.
