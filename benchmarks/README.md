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

## Tall linear-inequality suite (asv)

`runners/bench_tall_linear_ineq.py` tracks the tall condensed path
(`m ≫ n`, `N = W + AᵀΣA + Σ_x`) that large sparse `linear_ineq` sets land on.
It solves `TallSparseQP` (`generators/`) with `linsolve="auto"` and
`hessian="lbfgs"` (the generator's analytic Hessian would keep the sparse
normal-equations route out of auto-selection), 10% range rows, and both
declarations of the same rows: `"linear_ineq"` (public API, lowered by the
core) and `"operator"` (hand-lowered `ineq_constraints` with a `CSROperator`
Jacobian). `TallLinearIneq` times one solve per cell (`time_solve`,
`peakmem_solve`); `TallLinearIneqTrack` shares one solve per cell through
`setup_cache` for `track_success`, `track_iterations`, `track_kkt_residual`
and `track_route`.

| cell (`n`, `m/n`, density, bandwidth) | expected route | what it pins |
|---|---|---|
| 500, 10, 0.01, random | 1 dense | fill probe runs, Gram fills, small-n dense rule |
| 500, 10, 0.2, random | 1 dense | density ≥ 5% skips the fill probe |
| 2000, 10, 0.01, random | 1 dense | fill-gate straddle (filled side) |
| 2000, 10, 0.01, banded 0.002 | 5 sparse-NE | fill-gate straddle (fill ≈ 0.4%) |
| 2000, 100, 0.01, banded 0.002 | 5 sparse-NE | 200 000 rows |
| 10000, 10, 0.01, banded 0.002 | 5 sparse-NE | only the fill gate keeps it off Krylov (3) |

`track_route` codes (append-only; from `Result.routes.linear_solver`'s leading
token and `Result.routes.kkt_form`): **1** dense/condensed, **2**
dense/augmented, **3** krylov/condensed, **4** sparse/augmented, **5**
sparse-NE/normal_equations. An unmapped route raises, and
`tests/integration/test_tall_linear_ineq_benchmark.py` fails when a solver
can emit a route the mapping lacks. asv needs `runners/` to be a package
(`runners/__init__.py`). No Krylov cell is tracked: a tall
random-row solve at `n = 10 000` (where auto picks Krylov) runs for minutes.

```bash
cd benchmarks
asv run --python=same --bench bench_tall_linear_ineq   # a few minutes on a laptop CPU
```

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
