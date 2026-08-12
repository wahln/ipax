---
applyTo: "tests/**/*.py"
---

`ipax` is test-driven and multi-backend by rule (`AGENTS.md`, "Testing &
verification"). When reviewing changes under `tests/`:

- **Multi-backend by default.** New tests that exercise solver/protocol
  behavior should parametrize over the namespace fixture in
  `tests/conftest.py`, covering NumPy and PyTorch at minimum (CuPy/JAX when
  the fixture makes them available). A test hard-coded to one backend is a
  finding unless there's a stated reason it's backend-specific (e.g.
  asserting on a backend's own error type).
- **Contract batteries.** A new `Problem`, `LinearOperator`, `LinearSolver`,
  or `SparseDirectSolver` implementation should be exercised by the matching
  shared battery in `tests/contracts/`, not just a bespoke ad hoc test —
  that's what keeps the "pluggable" claim in `AGENTS.md` honest.
- **Regression tests.** Every bug fix should add a test under
  `tests/regression/` that fails before the fix and passes after, so the
  specific failure mode can't silently return.
- **Oracles over hand-picked assertions.** Prefer checking KKT conditions at
  the solution, closed-form QP/LP optima, or cross-checks against
  `scipy.optimize`/`cyipopt` (when installed) over asserting a single
  numeric value with no derivation — flag magic-number assertions that
  don't explain where the expected value comes from.
- **`array-api-strict` purity.** Tests that add solver-path coverage should
  be runnable under the `array_api_strict` backend where practical — that
  backend is the purity gate; a test that only works on NumPy because it
  relies on a NumPy-only behavior is itself a purity leak worth flagging.
- **Derivative checks.** Changes to gradients/Jacobians/Hessians should be
  covered by (or reuse) the FD-vs-analytic-vs-autodiff derivative harness
  rather than a one-off numeric comparison.

Benchmarks (`benchmarks/`) are a separate, non-gating suite — see
`.github/instructions/benchmarks.instructions.md` for those rules; don't
apply the multi-backend/contract requirements above to files under
`benchmarks/`.
