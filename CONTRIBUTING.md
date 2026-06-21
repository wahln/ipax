# Contributing to `ipax`

Thanks for working on `ipax`. The canonical, detailed contributor guide is
[`AGENTS.md`](./AGENTS.md) — it is the single source of truth for the math
conventions, architecture, and the rules below. This file is the short version.

## Non-negotiable invariants

These are hard constraints. A change that appears to require breaking one should be
raised for discussion in the PR, not worked around silently.

1. **No concrete array library in the core.** Never import `numpy`/`torch`/`cupy`/
   `jax` inside `ipax/` except in `ipax/backend/sparse/` and `ipax/problem/autodiff/`.
   Get the namespace from the input arrays:
   ```python
   from ipax.backend.namespace import array_namespace
   xp = array_namespace(x)
   ```
2. **Stay inside the Array API standard** — main namespace + the optional `linalg`
   extension only. Gap-fillers (triangular solve, `lstsq`, …) go in `backend/` with a
   comment naming the missing primitive.
3. **Linear algebra is injected, never hard-wired.** Jacobians/Hessians are
   `LinearOperator`s; all KKT solves go through the `LinearSolver` protocol. Adding a
   solve strategy must not touch `ipm/driver.py`.
4. **Sparsity is an adapter concern.** The core emits structure as Array-API
   index/value vectors; per-backend wrappers in `backend/sparse/` build and factor the
   sparse matrix. The IPM never sees a backend sparse object.
5. **No global mutable state.** Solver state lives in explicit objects.

The purity boundary (invariants #1/#4) is enforced by `scripts/check_purity.py` — at
commit via a pre-commit hook and, for Claude Code, via a PostToolUse hook the moment
an edit touches `ipax/`.

## Coding standards

- Python ≥ 3.10, full type hints, `from __future__ import annotations`.
- Public API documented with docstrings; numerical choices cite the paper/equation
  (e.g. `# Wächter & Biegler 2006, eq. (19)`).
- Config via frozen dataclasses in `options.py`; no magic numbers in the loop body.
- Floats are `float64` by default — never hard-code a dtype; read it from inputs.
- `ruff` (format + lint) and `mypy` clean before commit.

## Testing

- **Test-driven.** Write the failing test first, then implement to green.
- **Always multi-backend.** Numerical tests parametrize over the namespace fixture
  (NumPy + PyTorch minimum; CuPy/JAX/GPU when available); `array-api-strict` is the
  purity gate.
- New protocol implementations must pass the shared **contract batteries**
  (`tests/contracts/`).
- Every bug fix ships with a regression test in `tests/regression/`.

## Verification

`scripts/check.py` is the single verification entrypoint and runs exactly what CI
runs (`format → lint → types → purity → test`):

```bash
python scripts/check.py            # all gates (multi-backend tests included)
python scripts/check.py --fast     # skip the slow test gate
python scripts/check.py --list     # show the gates and the raw commands they wrap
```

## Direction

`ipax` is **beta** and **looking for adopters** — trying it on a real
workload and reporting back is one of the most valuable contributions right now.
The solver surface is complete; near-term work is **maintenance and measurement**
rather than new features:

- **Bug-fixing** — correctness on hard nonconvex / ill-conditioned problems, each fix
  with a regression test.
- **Performance** — drive the `benchmarks/` scaling and micro suites (`asv`); reduce
  per-iteration cost and memory across the dense / matrix-free / sparse routes.
- **Accuracy benchmarking** — grow the QC sweep and reference cross-checks
  (vs IPOPT/SciPy/OSQP); track scaled-KKT accuracy and robustness over time.

### Road to 1.0.0

`1.0.0` carries a semver stability promise for the public API. It is gated on:
**(1)** a stable public surface (`solve`, `Problem`, `Options`, `Result`,
`LinearOperator`/`LinearSolver`) across the `0.x` series, deprecations carried one
minor release; **(2)** the published MkDocs site; **(3)** GPU/sparse-direct
(CuPy/cuDSS, Torch/JAX-CUDA, MUMPS inertia) verified on real hardware in CI or
documented as provisional; **(4)** at least one external adopter workload with
feedback folded back in; **(5)** tracked `asv` and QC baselines with reference
cross-checks green. See [`README.md`](./README.md#road-to-100) and
[`AGENTS.md`](./AGENTS.md#road-to-100) for the full version.

New algorithmic features beyond the current scope (see the scope guardrails in
`AGENTS.md`) are out until discussed.
