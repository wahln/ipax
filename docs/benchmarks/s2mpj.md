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
| date | 2026-07-10 |
| CPU | 13th Gen Intel Core i9-13900HX (32 logical CPUs) |
| OS | Windows 11 (10.0.26200), AMD64 |
| Python | 3.14.6 |
| ipax | 0.4.0 (develop, post-0.4.0 fixes) |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| PyTorch | 2.12.0+cpu |
| sparse factorization | Feral LDLᵀ 0.11.0 (CPU) |

### Methodology

The full `{lbfgs, exact} × {dense, krylov, sparse}` matrix was swept, one
configuration per process (six in parallel, single-threaded BLAS), each process
additionally running **four problems concurrently** (`--jobs 4`) — the whole
six-config corpus completes in ~2.8 h wall. Gradient-based scaling (the solver
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

| config | correct | converged | optimal | acceptable | infeasible | stalled | rest.failed | max_iter | max_time | unbounded | num.err | solve.err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lbfgs/dense`  | 720 / 1098 | 854 | 742 | 111 | 61 | 96 | 63 | 14 | 9  | 1 | 1  | 0 |
| `lbfgs/krylov` | 706 / 1101 | 840 | 733 | 106 | 61 | 81 | 55 | 9  | 48 | 1 | 7  | 0 |
| `lbfgs/sparse` | 724 / 1101 | 857 | 736 | 120 | 62 | 95 | 62 | 14 | 8  | 1 | 3  | 0 |
| `exact/dense`  | 720 / 1098 | 847 | 761 | 85  | 51 | 65 | 36 | 20 | 28 | 1 | 51 | 0 |
| `exact/krylov` | 671 / 1101 | 807 | 757 | 49  | 63 | 87 | 53 | 24 | 66 | 0 | 2  | 0 |
| `exact/sparse` | **762 / 1101** | **894** | **839** | 54 | 59 | 49 | 44 | 14 | 16 | 1 | 25 | 0 |

Solved-correct by **at least one route: 808 / 1101**; converged by at least one
route: **941 / 1101**.

### Results — optimization vs. feasibility problems

The 207 objective-free problems are feasibility / nonlinear-equation systems with
different semantics (success = found a feasible point; many are inconsistent and
correctly report infeasible), so they are reported separately rather than mixed into
the optimization rate. The optimization column shows `correct` (`converged`).

| config | optimization (894) | feasibility (207) |
| --- | --- | --- |
| `lbfgs/dense`  | 656 / 891* (790) | 64 / 207 |
| `lbfgs/krylov` | 643 / 894  (777) | 63 / 207 |
| `lbfgs/sparse` | 660 / 894  (793) | 64 / 207 |
| `exact/dense`  | 667 / 891* (793) | 53 / 207 |
| `exact/krylov` | 604 / 894  (740) | 67 / 207 |
| `exact/sparse` | 688 / 894  (820) | 74 / 207 |

<small>* dense routes ran 891 of the 894 optimization problems; three exceed the
dense variable cap.</small>

### Observations

- **`exact/sparse` is the strongest route** — most correct (762) and most
  optimal (839). Exact-Hessian Newton steps factored by the sparse-direct route
  (Feral LDLᵀ with inertia control) is the most robust combination here.
- **False infeasibility claims are largely gone.** The per-route `infeasible`
  counts dropped from 110–156 to 51–63: a local-infeasibility verdict now
  requires a *stationarity certificate* from the restoration phase (projected
  gradient ≈ 0, or no descent at the Levenberg–Marquardt ceiling), gets one
  x0-anchored second-chance probe before it is believed, and is vetoed when
  the run's own history already certified near-feasibility. Uncertified stalls
  end as the honest `restoration_failed`/`stalled` labels instead — the two
  new status columns — and dozens of the ex-`infeasible` rows now finish
  `optimal` outright (SNAKE, CATENARY, BT9, HS39, CRESC4, SSEBNLN, ALJAZZAF…).
- **Budget statuses are down sharply** (`max_iter` + `max_time`: 69–182 →
  23–90 per route). Two mechanisms: a run whose returned best iterate already
  satisfies the relaxed (acceptable-level) KKT tolerance now reports
  `acceptable` instead of `max_time`/`max_iter` (the DIAMON2DLS/DMN
  least-squares family — oscillating at KKT ~1e-7 without ever holding it for
  the acceptable window — is now scored *correct*), and the stall detector
  ends frozen-iterate grinds early.
- **`unbounded` is now a certified verdict**: it requires the objective to
  diverge below `−diverging_iterates_tol`, not just a large iterate norm.
  KOEBHELB — whose iterate wanders past 1e22 and then converges to f = 112 —
  flipped from `unbounded` to `optimal` on the exact routes, while genuinely
  unbounded problems (INDEF) are still detected.
- **The RT-typical `lbfgs/sparse`** route is solid (724 correct, 3
  `numerical_error`), validating the L-BFGS + sparse-direct path used for
  radiotherapy-scale problems.
- **Documented-infeasible detection works**: `BURKEHAN` and the rest of the
  documented-infeasible set are correctly scored as expected-infeasible across
  every route.
- **No crashes**: the `solve_error` column is zero everywhere for the first
  time (the LINSPANH and HS9 crashes of the previous run are gone).

#### Changes since the 2026-07-05 run

Every route improved on `correct` — `exact/krylov` most of all (619 → 671) —
and solved-correct by ≥1 route rose 795 → 808 (converged 928 → 941). The
deltas map to this round of fixes:

- **Restoration exit certificates + free-set steps**: only a stationarity-type
  restoration exit may claim local infeasibility (window/budget stalls resume
  or end as `restoration_failed`), the one-shot x0-anchored probe covers every
  believed claim, and bound-blocked variables are eliminated from the
  Gauss-Newton system (projected Newton on the free set) — restoring the GN
  rate on bound-active problems (DRUGDIS: θ 0.19 → 8e-4 per budget).
- **Budget exhaustion at a near-optimal iterate reports `acceptable`**,
  mirroring the stall/step-failure salvage (the best accepted iterate is
  already returned on failure statuses).
- **Two-signal unbounded detection** (iterate norm *and* diverged objective).
- **Mehrotra corrector step-length acceptance**: a corrected direction that
  collapses the predictor's boundary step falls back to the plain centered
  step, removing the HS71-class knife-edge (the QC corpus no longer excludes
  the corrector configs).
- **μ oracles + monotone default confirmed by paired A/B** (v10): adaptive /
  Breedveld / probing schedules are available opt-in; correctors now consume
  the oracle's μ instead of choosing it.
- Not visible in this CPU sweep but in the same batch: an opt-in **sparse
  condensed normal-equations route** for tall problems with localized
  Jacobians (32× faster than the dense route at n = 10k on banded QPs), a
  **density gate** on the tall dense-route auto-selection, and cuDSS-route
  fixes verified against a real GPU runtime.

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
