# Benchmarks

Performance, scaling, robustness, and accuracy tracking — **separate from the
correctness tests**. These are *not* run per-PR; the full corpus runs nightly.

```
benchmarks/
  asv.conf.json     airspeed-velocity config (tracked perf over commits)
  generators/       synthetic RT-like block-sparse problem builders
  corpus/           standard sets: Hock–Schittkowski, CUTEst subset, Maros–Mészáros
  harness/          metrics, environment capture, result IO (JSON/parquet)
  runners/          asv benchmark classes + pytest-benchmark micro-benchmarks
  reports/          generated tables/plots
```

## Running

```bash
pip install -e ".[bench,numpy,torch,sparse-cpu]"

# Quality-control sweep (correctness / robustness / accuracy regression gate).
# Runs the curated corpus across a config matrix (hessian x linsolve x
# corrections x scaling) and every available backend, scoring each case against
# its known optimum. Writes <out>.json + <out>.md; exits non-zero if any case
# is not "correct". Runs as a gating CI job.
python -m benchmarks.runners.qc --out benchmarks/reports/qc

# Reference cross-check (advisory): solve the corpus with ipax and each available
# baseline (SciPy now; cyipopt/OSQP when installed) and compare. NumPy-only.
# Flags divergence but does not gate (the QC sweep already gates accuracy).
python -m benchmarks.runners.crosscheck --out benchmarks/reports/crosscheck

# Scaling & memory study: sweep n over the RT-like generator per solver route,
# measure wall-clock + peak memory (tracemalloc), fit the exponent p in cost~n^p.
# Heavier; run manually/nightly. (Dense memory ~n^2 vs matrix-free Krylov ~flat.)
python -m benchmarks.runners.scaling --sizes 500,1000,2000 --routes krylov,sparse,dense

asv run                      # tracked macro/scaling benchmarks
asv publish && asv preview   # browse results
pytest benchmarks/runners/micro --benchmark-only   # micro-benchmarks
```

The QC sweep is the quality core: `corpus/` holds backend-parametric
`BenchmarkProblem`s with known optima, `harness/` scores a `(problem, backend,
config)` cell into a `CaseResult` (status, iterations, scaled-KKT components,
constraint violation, accuracy vs `x*`, solve time, solver used), and
`runners/qc.py` drives the matrix and renders the report. A fast smoke test
(`tests/integration/test_benchmark_harness.py`) keeps it from bit-rotting.

## What is measured

Iterations to convergence; wall-clock split (assemble / factor / solve /
line-search); peak memory (matrix-free vs dense vs sparse); final scaled KKT
residual; robustness (success rate); accuracy vs reference optimum; empirical
scaling exponent in `n`; GPU-vs-CPU speedup; L-BFGS memory sweep; solver-strategy
comparison.

## Baselines

Pluggable reference solvers in `baselines/` (only those whose optional dependency
is installed are used): **SciPy** `trust-constr` and **IPOPT** via `cyipopt`
(general NLP), and **OSQP** (convex QP — skips nonlinear/non-quadratic problems).
CVXOPT / `qpax` slot in via the same `Baseline` protocol. The cross-check records
a baseline that cannot express a problem as "skipped".

## Plots

Matplotlib (in the `bench` extra) is optional and imported lazily. The scaling
runner writes `<out>.png` (time & memory vs `n`, log–log per route) and the QC
runner writes `<out>_iters.png` (mean iterations per config); pass `--no-plot` to
skip, and runs without matplotlib simply omit the figures.

## External standard sets

`corpus/external.py` exposes `cutest_problems()` (CUTEst via `pycutest`) and
`maros_meszaros_problems()` — install/download-gated, returning `[]` when their
toolchain/data is absent (never committed; not run in CI).
