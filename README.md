# ipax

[![PyPI version](https://img.shields.io/pypi/v/ipax.svg)](https://pypi.org/project/ipax/)
[![CI](https://github.com/wahln/ipax/actions/workflows/ci.yml/badge.svg)](https://github.com/wahln/ipax/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/wahln/ipax/branch/main/graph/badge.svg)](https://codecov.io/gh/wahln/ipax)
[![Documentation](https://readthedocs.org/projects/ipax/badge/?version=latest)](https://ipax.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

> **Interior-Point on the Array API** — a pure-Python, [Python Array
> API](https://data-apis.org/array-api/latest/)–conformant primal–dual
> interior-point solver for **large-scale nonlinear constrained optimization**,
> tuned for the scale and structure of radiotherapy (RT) treatment planning.

`ipax` solves

```
min   f(x)
s.t.  c(x)  = 0           (equalities)
      g(x) ≤ 0            (inequalities; two-sided b_l ≤ ĝ(x) ≤ b_u supported)
      x_L ≤ x ≤ x_U       (bounds, possibly ±∞)
```

with a log-barrier interior-point method, an L-BFGS (Powell-damped) Lagrangian
Hessian by default, and an IPOPT-style filter line search. It runs unchanged on
**NumPy, PyTorch, JAX, and CuPy** (CPU/GPU) because the core never imports a
concrete array library — it infers the namespace from the arrays a problem
returns.

Design draws on **Wächter & Biegler (2006)** (IPOPT) for globalization and on
**Breedveld et al. (2017)** for the RT-tuned primal–dual formulation.

The solver surface is complete: equality/inequality/bound constraints, the
dense / matrix-free / sparse-direct solve routes, filter line search with
restoration, and multi-backend support. `ipax` is **beta** — the
algorithm is feature-complete and tested, and the road to a stable `1.0.0` is
about hardening the public API, docs, and verification rather than new features
(see [Status and roadmap](#status-and-roadmap)). **We are looking for adopters**
to try it on real workloads.

For more information check the [Documentation](https://ipax.readthedocs.io/).

## Install

```bash
# core + a backend (pick what you have)
pip install -e ".[numpy]"
pip install -e ".[torch]"

# full dev environment (NumPy + Torch + strict + sparse + tooling)
pip install -e ".[dev]"
```

CUDA is user-managed so that the CuPy wheel matches the installed toolkit.
Install the appropriate CuPy package and cuDSS runtime yourself; the
`sparse-cuda` extra installs only the nvmath bindings. Without cuDSS, the CuPy
adapter falls back to `cupyx.scipy.sparse.linalg.spsolve` and cannot report
inertia.

## Quick start

```python
import numpy as np
import ipax

Q = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
c = np.asarray([-1.0, -2.0], dtype=np.float64)
problem = ipax.QuadraticProblem(Q, c)

res = ipax.solve(
    problem,
    x0=np.asarray([0.0, 0.0], dtype=np.float64),
    options=ipax.Options(hessian="exact", linsolve="dense"),
)
print(res.status, res.x, res.objective)
```

Enable gradient-based problem scaling with a short `Options` value:

```python
res = ipax.solve(problem, x0, options=ipax.Options(scaling="gradient-based"))
```

For a custom threshold, use
`ipax.Options(scaling=ipax.ScalingOptions(method="gradient-based", max_gradient=50.0))`.

Termination comes in two families that share the same conditions: the objective
magnitude `f_tol` (`|f| ≤ f_tol`), the relative objective change
`f_rel_change_tol`, and the scaled KKT components `dual_inf_tol`,
`constr_viol_tol`, `compl_inf_tol` (`None` disables one). **Optimality** is
single-iteration and reports `Status.OPTIMAL` — the defaults are the classic
scaled-KKT test (each component ≤ `1e-8`):

```python
options = ipax.Options(
    optimality=ipax.OptimalityConditionOptions(
        dual_inf_tol=1e-4,
        compl_inf_tol=1e-4,  # loose optimality, but…
        constr_viol_tol=1e-8,  # …tight feasibility, in one step
    )
)
```

**Acceptable** is multi-iteration and reports `Status.ACCEPTABLE` (also a
success): the conditions must hold for `n_iter` consecutive iterations. It is
off by default and useful when a dual-infeasibility-dominated residual plateaus
but the objective and primal feasibility have settled:

```python
options = ipax.Options(
    acceptable=ipax.AcceptableStoppingOptions(
        dual_inf_tol=1.0,  # tolerate the stuck dual infeasibility
        constr_viol_tol=1e-6,
        f_tol=1e-7,  # relative objective change
        n_iter=5,
    )
)
```

The top-level `max_iter` and `max_time` (seconds) limits report `Status.MAX_ITER`
and `Status.MAX_TIME`.

The backend is whatever `x0` is: pass a `torch.Tensor` and the whole solve runs
in PyTorch. Derivatives you don't supply are filled by autodiff (if the backend
supports it) or finite differences; the Hessian defaults to L-BFGS.

See [`examples/`](./examples/) for short, runnable examples.

## Architecture at a glance

```
ipax/
  problem/   Problem ABC + derivative resolution (analytic → autodiff → finite-diff → L-BFGS)
  backend/   namespace resolution, LinearOperator hierarchy, per-backend sparse adapters
  linalg/    LinearSolver protocol: dense / matrix-free Krylov / sparse-direct
  ipm/       barrier, KKT assembly, filter line search, restoration, L-BFGS, driver
  testing/   analytic oracle problems shared by tests and benchmarks
tests/       contract batteries + unit/property/integration/backends/regression
examples/    minimal runnable examples for the current implementation surface
benchmarks/  RT-like generators, corpus, asv harness (separate from tests)
docs/        MkDocs site
```

The two protocols `LinearOperator` and `LinearSolver` are the architectural
core: every Jacobian/Hessian is an operator and every KKT solve goes through a
pluggable solver, so the same IPM loop runs dense, matrix-free, or sparse.

## Non-negotiable invariants

1. **No concrete array library in the core** — only in `backend/sparse/` and
   `problem/autodiff/` adapters.
2. **Stay inside the Array API standard** (main namespace + optional `linalg`).
3. **Linear algebra is injected**, never hard-wired into the IPM loop.
4. **Sparsity is an adapter concern** — the core emits index/value vectors only.
5. **No global mutable state.**

These are enforced in CI (`scripts/check_purity.py` + `array-api-strict`).

## Status and roadmap

`ipax` is **beta**. The solver surface is complete and tested across
NumPy and PyTorch, but the public API has not yet been hardened against real-world
use — expect refinements across the `0.x` series before it is frozen at `1.0.0`.

**We are looking for adopters.** If you have a large-scale constrained NLP —
radiotherapy planning or otherwise — and would like to try `ipax`, please
[open an issue](https://github.com/wahln/ipax/issues). Early feedback on the
`Problem` / `solve` API directly shapes the road to `1.0.0`.

### Road to 1.0.0

`1.0.0` is the point at which the public API carries a semver stability promise.
The day-to-day work stays **maintenance and measurement** (below), and the release
is gated on these exit criteria:

1. **API stability.** The public surface — `solve`, `Problem`, `Options`, `Result`,
   and the `LinearOperator` / `LinearSolver` protocols — holds stable across the
   `0.x` series, with any deprecations carried for at least one minor release.
2. **Published documentation.** The MkDocs site (API reference + guides) is built
   and published.
3. **GPU & sparse-direct verification.** The CuPy/cuDSS and Torch/JAX-CUDA sparse
   routes and the MUMPS inertia path are exercised on real hardware in CI, or
   documented as provisional until GPU CI exists.
4. **Adopter validation.** At least one external workload runs on `ipax`, with the
   feedback folded back into the API.
5. **Tracked baselines.** `asv` performance baselines and the QC accuracy sweep are
   tracked across releases, with cross-checks vs IPOPT / SciPy / OSQP green.

Near-term work is **maintenance and measurement** rather than new features:

- **Bug-fixing** — correctness on hard nonconvex / ill-conditioned problems; every
  fix ships with a regression test.
- **Performance** — drive the `benchmarks/` scaling and micro suites (`asv`); reduce
  per-iteration cost and memory across the dense / matrix-free / sparse routes.
- **Accuracy benchmarking** — grow the QC sweep and reference cross-checks
  (vs IPOPT / SciPy / OSQP); track scaled-KKT accuracy and robustness over time.

New algorithmic features beyond the current scope are out until discussed. See
[`CONTRIBUTING.md`](./CONTRIBUTING.md) and [`AGENTS.md`](./AGENTS.md).

## Development

```bash
pip install -e ".[dev]"

python scripts/check.py            # one command: format, lint, types, purity, multi-backend tests
python scripts/check.py --fast     # skip the slow test gate
python scripts/check.py --list     # show the gates and the raw commands they wrap
```

`scripts/check.py` is the single verification entrypoint and runs exactly what CI
runs. The individual tools (`pytest`, `ruff`, `mypy`, `scripts/check_purity.py`,
`asv`) can still be invoked directly — `--list` prints them.

### Agent-assisted development

This repo is set up for agentic coding. The import-purity boundary
(`scripts/check_purity.py`) is enforced both by a pre-commit hook and, for
[Claude Code](https://claude.com/claude-code), by a PostToolUse hook that fires the
moment an edit touches `ipax/`. Checked-in Claude Code config in `.claude/` adds an
**`invariant-auditor`** review subagent (diffs vs. the five invariants, math
conventions, and GPU/device-efficiency rules) plus `/verify` and `/tdd` commands.

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the short contributor guide and
[`AGENTS.md`](./AGENTS.md) for the canonical, detailed version.

## References

- Wächter & Biegler (2006), *On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming*, Math. Prog. 106(1):25–57. (IPOPT)
- Breedveld, van den Berg, Heijmen (2017), *An interior-point implementation
  developed and tuned for radiation therapy treatment planning*, COAP 68:209–242.
- Friedlander & Orban (2012), *A primal–dual regularized interior-point method for convex quadratic programs*, Math. Prog. Comp. 4:71–107

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
