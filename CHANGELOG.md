# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Fixed
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

[Unreleased]: https://github.com/wahln/ipax/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/wahln/ipax/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/wahln/ipax/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/wahln/ipax/releases/tag/v0.1.0
