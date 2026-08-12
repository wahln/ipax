---
applyTo: "benchmarks/**/*.py"
---

`benchmarks/` is explicitly out of the Array-API purity scope (`AGENTS.md`,
"Testing & verification": "Benchmarks are separate ... not tests"). Review
these files with a different bar than `ipax/` core or `tests/`:

- **Concrete backends are fine.** Importing NumPy/SciPy/Torch directly (e.g.
  in `benchmarks/generators/`, `benchmarks/harness`, `benchmarks/runners/`)
  is expected and should not be flagged as an invariant #1 violation.
- **Not a correctness gate.** Benchmarks are tracked over commits with `asv`
  and run nightly, not per-PR — don't hold benchmark changes to the same
  "needs a failing-first test" bar as `ipax/` or `tests/` changes. Do still
  flag benchmarks that silently change what they measure (e.g. altering a
  generator's problem distribution) without updating the tracked baseline
  or calling it out in the PR description, since that quietly invalidates
  historical comparisons.
- **RT-scale realism.** Synthetic generators should stay in the project's
  stated target range (`1e3`–`1e5` variables, 5–50% density) unless the
  change is deliberately adding a new regime — flag generators that drift
  outside that range without comment.
- **No algorithmic changes smuggled in.** `benchmarks/` should measure
  `ipax`, not reimplement pieces of it. Flag any benchmark-only code that
  duplicates solver logic that belongs in `ipax/` instead.
- **Reports stay reproducible.** Prefer benchmark code that regenerates
  `benchmarks/reports/` output deterministically (fixed seeds, pinned
  problem sets) over ad hoc scripts whose output can't be reproduced by CI.
