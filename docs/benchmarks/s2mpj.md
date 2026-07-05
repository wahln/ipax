# S2MPJ / CUTEst benchmark

ipax is exercised over the **S2MPJ** translation of the CUTEst test collection — a
pure-Python rendering of the full Hock–Schittkowski and CUTEst problem sets (no
Fortran/SIF toolchain). The sweep is the broad-coverage complement to the curated
quality-control corpus: it measures convergence, robustness, and accuracy of every
solver route across ~1100 diverse problems.

The corpus is download-gated (S2MPJ carries no license, so it is not vendored) and
not part of per-PR CI; the loader and runner live in `benchmarks/` and the raw
JSON/Markdown reports are kept as artifacts. This page records the **latest tracked
run**.

## How a problem is scored

Scoring uses each problem's **own documented outcome**, parsed from the S2MPJ
source — not just "did the solver stop." Three machine-readable signals are taken
from every problem file:

- **`# LO SOLTN`** — the SIF author's documented solution objective (present on
  ~590 of the problems that ran here). A case is *correct* only if it converges
  **and** reaches this objective (relative+absolute tolerance `1e-4`), so a stall at
  a different local minimum is not counted as correct.
- **`Solution (infeasible)` / `Source: an infeasible problem`** — marks the
  deliberately-infeasible problems (e.g. `BURKEHAN`). For these, *detecting
  infeasibility is the correct outcome*, not a failure.
- **`pbclass`** — the CUTEst classification (used to separate the objective-free
  feasibility problems from the optimization problems below).

!!! note "`correct` is stricter than `converged`"
    Because `correct` requires matching the documented optimum, it is deliberately
    below the raw success (optimal + acceptable) count: the gap is problems that
    reached a KKT point at a *different* objective than the documented best-known
    value (typically a different local minimum on a nonconvex problem).

    The report therefore scores **two tiers**: `correct` (matched the documented
    outcome) and the weaker `converged` (reached a valid KKT point — a small scaled
    KKT residual at a success status, which already bounds primal infeasibility —
    regardless of *which* optimum). `correct` is a subset of `converged`, so the
    `converged` count credits genuine convergence to a different local optimum
    rather than reading it as a failure. In the per-case table a `≈` flag marks a
    converged-but-not-`correct` case and `⚠️` marks a non-converged one.

## Latest run

### System

| | |
| --- | --- |
| date | 2026-07-05 |
| CPU | 13th Gen Intel Core i9-13900HX (32 logical CPUs) |
| OS | Windows 11 (10.0.26200), AMD64 |
| Python | 3.14.6 |
| ipax | 0.3.0 (develop, post-0.3.0 fixes) |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| PyTorch | 2.12.0+cpu |
| sparse factorization | Feral LDLᵀ 0.11.0 (CPU) |

### Methodology

The full `{lbfgs, exact} × {dense, krylov, sparse}` matrix was swept, one
configuration per process (six in parallel, single-threaded BLAS), each process
additionally running **five problems concurrently** (`--jobs 5`) — the whole
six-config corpus completes in ~3.4 h wall. Gradient-based scaling (the solver
default), NumPy backend. Per-solve budget: `max_iter = 10000`, `max_time = 300 s`.
Each route is gated by a **per-route variable cap** (dense 2000, Krylov 10000,
sparse 25000) so a single full-corpus run stays tractable — three problems exceed
the dense cap and are reported as oversized for the two dense routes. The
**objective-free problems** (207 CUTEst feasibility / nonlinear-equation systems)
were run as `min 0` subject to the constraints (`--include-objective-free`).

S2MPJ's interpretive evaluation loop used to dominate sweep wall-time (~90% of a
solve); the corpus adapter now routes evaluations through a **precompiled
evaluator** (verified against the original S2MPJ methods per problem at build
time, with automatic fallback), so per-solve times measure the solver, not the
benchmark interface. Solves also use the (new default) **acceptable-level
termination** — IPOPT convention, `1e-6` held for 15 iterations — so a solve
whose achievable KKT floor sits just above the `1e-8` optimality tolerance stops
as `acceptable` instead of grinding to the budget cap.

### Results — full corpus

`correct` counts problems that matched the documented outcome (objective or
infeasibility); `converged` is the weaker tier (reached a valid KKT point, possibly
a different local optimum — a superset of `correct`). The status columns are the raw
terminal states.

| config | correct | converged | optimal | acceptable | infeasible | max_iter | max_time | unbounded | num.err | solve.err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lbfgs/dense`  | 710 / 1098 | 839 | 731 | 107 | 149 | 53 | 55  | 1  | 2  | 0 |
| `lbfgs/krylov` | 695 / 1101 | 822 | 726 | 95  | 143 | 37 | 95  | 1  | 4  | 0 |
| `lbfgs/sparse` | 705 / 1101 | 834 | 738 | 95  | 156 | 51 | 57  | 1  | 2  | 1 |
| `exact/dense`  | 715 / 1098 | 840 | 759 | 80  | 110 | 16 | 85  | 3  | 45 | 0 |
| `exact/krylov` | 619 / 1101 | 764 | 720 | 43  | 147 | 48 | 134 | 6  | 2  | 1 |
| `exact/sparse` | **740 / 1101** | **870** | **822** | 47 | 120 | 16 | 62 | 11 | 23 | 0 |

Solved-correct by **at least one route: 795 / 1101**; converged by at least one
route: **928 / 1101**.

### Results — optimization vs. feasibility problems

The 207 objective-free problems are feasibility / nonlinear-equation systems with
different semantics (success = found a feasible point; many are inconsistent and
correctly report infeasible), so they are reported separately rather than mixed into
the optimization rate. The optimization column shows `correct` (`converged`).

| config | optimization (894) | feasibility (207) |
| --- | --- | --- |
| `lbfgs/dense`  | 647 / 891* (776) | 63 / 207 |
| `lbfgs/krylov` | 633 / 894  (760) | 62 / 207 |
| `lbfgs/sparse` | 642 / 894  (771) | 63 / 207 |
| `exact/dense`  | 662 / 891* (786) | 53 / 207 |
| `exact/krylov` | 554 / 894  (699) | 65 / 207 |
| `exact/sparse` | 671 / 894  (801) | 69 / 207 |

<small>* dense routes ran 891 of the 894 optimization problems; three exceed the
dense variable cap.</small>

### Observations

- **`exact/sparse` is the strongest route** — most correct (740) and most
  optimal (822). Exact-Hessian Newton steps factored by the sparse-direct route
  (Feral LDLᵀ with inertia control) is the most robust combination here.
- **The L-BFGS/Krylov saddle failures are gone.** The previous run's dominant
  cluster — **187 `numerical_error`** on `lbfgs/krylov` — is down to **4**: a
  GMRES fallback when saddle MINRES fails, an adaptive (inexact-Newton) inner
  tolerance, and dual regularization for rank-deficient equality Jacobians
  attack exactly that cluster. `exact/krylov` still trails on speed
  (134 `max_time`) — the route is slow, not broken.
- **The `acceptable` tier does real work**: 43–107 cases per route stop at a
  valid near-optimal point (KKT ≤ 1e-6) instead of burning the 300 s budget —
  the main reason the sweep is faster and `max_time` counts dropped. The cost is
  a handful of ill-conditioned least-squares problems (PALMER `E`/`C` variants)
  that stop on a legitimate acceptable plateau which further grinding would have
  escaped to a deeper minimum.
- **`numerical_error` on the exact routes is now a fail-fast label, not a new
  failure mode**: those 45/23 rows (mostly `*NE` nonlinear-equation systems)
  previously wandered on heavily-regularized garbage steps into `infeasible` or
  `max_time`; with δ_c kept away from repairs that δ_w handles alone, the KKT
  solve fails outright and the row says so. Only 2 previously-correct problems
  moved into this bucket, while each exact route gained +12 correct overall.
- **The RT-typical `lbfgs/sparse`** route is solid (705 correct, 2
  `numerical_error`), validating the L-BFGS + sparse-direct path used for
  radiotherapy-scale problems.
- **Documented-infeasible detection works**: `BURKEHAN` and the rest of the
  documented-infeasible set are correctly scored as expected-infeasible across
  every route. The new `unbounded` status (diverging-iterates test) likewise
  labels genuinely unbounded problems (e.g. INDEF) honestly instead of
  `numerical_error`.
- Two isolated `solve_error` crashes remain open: LINSPANH (`lbfgs/sparse`) and
  HS9 (`exact/krylov`).

#### Changes since the 2026-07-02 baseline

Every route improved on `correct` — `lbfgs/krylov` most of all (638 → 695) —
and solved-correct by ≥1 route rose 790 → 795 (converged 920 → 928). The deltas
map to this round of fixes:

- **Adaptive Krylov inner tolerance** (Eisenstat–Walker forcing sequence) and a
  **GMRES fallback on saddle MINRES failures**: together they eliminate the
  L-BFGS/Krylov `numerical_error` cluster (187 → 4).
- **Two-phase dual regularization**: δ_c escalation for rank-deficient equality
  Jacobians is now a last resort after the pure-δ_w ladder is exhausted (and
  δ_w resets when it engages). This fixed a regression where δ_c contaminated
  legitimate δ_w repairs (HS61: cycling → optimal in 12 iterations) and turned
  the remaining hopeless solves into fail-fast `numerical_error` rows instead of
  `infeasible`/`max_time` wanderers.
- **Acceptable-level termination by default** (IPOPT convention): near-optimal
  grinders (e.g. PALMER1A: 300 s `max_time` at KKT 4.2e-8 → `acceptable` in
  1.2 s at the same point) now finish early, cutting per-config solve time by
  1–3 h.
- **`Status.UNBOUNDED`** via a diverging-iterates test labels unbounded problems
  honestly.
- **Benchmark interface rebuilt** (measurement change, not solver): a precompiled
  S2MPJ evaluator removes the ~90% interpretive-evaluation overhead (6–37×
  faster solves), and the runner fans problems out over worker processes
  (`--jobs`). Wall-clock for the full six-config sweep: ~4.5 h → ~3.4 h — with
  per-solve times now reflecting the solver itself.

## Reproducing

From a checkout with `IPAX_S2MPJ_DIR` pointing at an
[S2MPJ](https://github.com/GrattonToint/S2MPJ) clone:

```bash
# one configuration, full corpus, dataset-scored, recoverable, 5 problems at a time
python -m benchmarks.runners.s2mpj --all --config exact/sparse --jobs 5 \
    --include-objective-free --resume --max-iter 10000 --max-time 300 \
    --out benchmarks/reports/s2mpj_exact_sparse
```

Omit `--config` to sweep the whole matrix in one process. `--jobs N` runs N
problems concurrently in worker processes (pin BLAS threads, e.g.
`OMP_NUM_THREADS=1`, so per-solve timings stay comparable); reports are flushed
in sorted order after every problem, so they are deterministic and an
interrupted sweep never loses work. `--resume` keeps an existing report and
skips finished problems; `--exclude` steps past a problem that natively crashes
a backend (on a crashed worker pool the runner exits `2` and the `.inflight`
file names the candidate culprits).
