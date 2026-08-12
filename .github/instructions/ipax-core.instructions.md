---
applyTo: "ipax/**/*.py"
---

This is the `ipax` core library — the part that must stay Array-API-conformant
and backend-agnostic per `AGENTS.md`'s non-negotiable invariants. Apply the
strict rules below, with the adapter exceptions noted.

## Adapter exceptions

These subpaths are the project's explicitly labeled adapters and are allowed
(expected) to import a concrete array/autodiff library — do not flag `import
numpy`/`torch`/`cupy`/`jax`/`scipy` there:

- `ipax/backend/sparse/**` — per-backend sparse construction/factorization.
- `ipax/backend/dense/**` — per-backend dense Cholesky/solve implementations.
- `ipax/problem/autodiff/**` — per-backend autodiff (JAX/Torch) gradient and
  HVP wrappers.

Even inside these adapters, still flag: a concrete type or object escaping
back into `ipm/`, `linalg/solver.py`'s public protocol, or any other core
module (invariant #4 — the IPM must never see a backend sparse/dense object
directly, only through `LinearOperator`/`LinearSolver`); and any adapter that
doesn't satisfy the shared `tests/contracts/` battery for its protocol.

## Everywhere else in `ipax/`

- **No concrete array library.** The namespace must come from
  `array_namespace(x)` (`ipax/backend/namespace.py`); only `xp.*` /
  `xp.linalg.*` calls are allowed. Flag `np.*`-style calls, NumPy-only
  attribute/method assumptions, or a concrete backend smuggled in through a
  helper.
- **Stay inside the Array API standard** (main namespace + the `linalg`
  extension). Flag use of anything the extension doesn't provide — LDLᵀ with
  inertia, triangular solve, `lstsq`, `out=`/in-place ops, non-standard fancy
  indexing, any sparse type — outside the adapters above. A genuine gap
  belongs in `backend/` with a comment naming the missing primitive.
- **Linear algebra is injected, never hard-wired.** All Jacobians/Hessians
  should be `LinearOperator`s (`ipax/backend/operators.py`); all KKT solves
  should go through the `LinearSolver` protocol (`ipax/linalg/solver.py`).
  Flag any concrete factor/solve call reaching directly into
  `ipax/ipm/driver.py` or another loop body — adding a solve strategy must
  not require touching the driver.
- **No module-level mutable state.** Flag global singletons or mutable
  module-level caches. Solver state belongs in explicit objects passed
  through the call chain.
- **Citations and symbols.** Algorithmic steps (barrier update,
  regularization, filter/line-search, KKT construction, restoration) should
  cite the source paper/equation (see `AGENTS.md` References). Code should
  use the project's symbol names (`W, Sigma_x, Sigma_s, N, mu, tau, alpha,
  theta, phi`) rather than renaming them ad hoc. Numerical constants belong
  in the frozen dataclasses in `ipax/options.py`, not as magic numbers in a
  loop body.
- **GPU / device efficiency** (no-cost to check, since the same loop runs on
  CuPy/JAX/Torch-CUDA): flag `float(x)`/`.item()`/`bool(x)`/`int(x)`, or a
  Python `if`/`while` on a 0-d device-array value, inside the driver's hot
  iteration loop — that forces a host sync every iteration. Flag hard-coded
  `float64` instead of reading dtype from the inputs, and unnecessary
  `asarray`/list round-trips.
- **Tests.** New or changed core behavior should ship with a corresponding
  test parametrized across backends (see the companion
  `.github/instructions/tests.instructions.md` for what that battery must
  cover).
