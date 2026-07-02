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
| date | 2026-07-02 |
| CPU | 13th Gen Intel Core i9-13900HX (32 logical CPUs) |
| OS | Windows 11 (10.0.26200), AMD64 |
| Python | 3.14.6 |
| ipax | 0.3.0 |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| PyTorch | 2.12.0+cpu |
| sparse factorization | Feral LDLᵀ 0.11.0 (CPU) |

### Methodology

The full `{lbfgs, exact} × {dense, krylov, sparse}` matrix was swept, one
configuration per process (six in parallel, single-threaded each), over the whole
corpus on the NumPy backend with gradient-based scaling (the solver default).
Per-solve budget: `max_iter = 10000`, `max_time = 300 s` (raised from the previous
`5000` / `120 s` for more signal on the slow-converging clusters). Each route is
gated by a **per-route variable cap** (dense 2000, Krylov 10000, sparse 25000) so a
single full-corpus run stays tractable — three problems exceed the dense cap and are
reported as oversized for the two dense routes. The **objective-free problems** (207
CUTEst feasibility / nonlinear-equation systems) were run as `min 0` subject to the
constraints (`--include-objective-free`).

### Results — full corpus

`correct` counts problems that matched the documented outcome (objective or
infeasibility); `converged` is the weaker tier (reached a valid KKT point, possibly
a different local optimum — a superset of `correct`). The status columns are the raw
terminal states.

| config | correct | converged | optimal | acceptable | infeasible | max_iter | max_time | num.err | solve.err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lbfgs/dense`  | 702 / 1098 | 825 | 788 | 36 | 147 | 30 | 94  | 3   | 0 |
| `lbfgs/krylov` | 638 / 1101 | 759 | 722 | 36 | 56  | 12 | 88  | 187 | 0 |
| `lbfgs/sparse` | 697 / 1101 | 819 | 792 | 26 | 156 | 31 | 93  | 3   | 0 |
| `exact/dense`  | 708 / 1098 | 828 | 810 | 17 | 142 | 25 | 104 | 0   | 0 |
| `exact/krylov` | 613 / 1101 | 748 | 741 | 6  | 147 | 45 | 158 | 4   | 0 |
| `exact/sparse` | **724 / 1101** | **848** | **837** | 10 | 140 | 22 | 89  | 3   | 0 |

Solved-correct by **at least one route: 790 / 1101**; converged by at least one
route: **920 / 1101**.

### Results — optimization vs. feasibility problems

The 207 objective-free problems are feasibility / nonlinear-equation systems with
different semantics (success = found a feasible point; many are inconsistent and
correctly report infeasible), so they are reported separately rather than mixed into
the optimization rate. The optimization column shows `correct` (`converged`).

| config | optimization (894) | feasibility (207) |
| --- | --- | --- |
| `lbfgs/dense`  | 639 / 891* (762) | 63 / 207 |
| `lbfgs/krylov` | 587 / 894  (708) | 51 / 207 |
| `lbfgs/sparse` | 634 / 894  (756) | 63 / 207 |
| `exact/dense`  | 654 / 891* (773) | 54 / 207 |
| `exact/krylov` | 558 / 894  (693) | 55 / 207 |
| `exact/sparse` | 655 / 894  (779) | 69 / 207 |

<small>* dense routes ran 891 of the 894 optimization problems; three exceed the
dense variable cap.</small>

### Observations

- **`exact/sparse` is the strongest route** — most correct (724) and most
  optimal (837). Exact-Hessian Newton steps factored by the sparse-direct route
  (Feral LDLᵀ with inertia control) is the most robust combination here.
- **The matrix-free Krylov route trails**, consistent with the documented
  [saddle-preconditioning limitation](../concepts/linalg.md). The attribution
  is now clear: `lbfgs/krylov` produces **187 `numerical_error`** results, but with
  the exact Hessian (`exact/krylov`) that collapses to **4** — so those errors come
  from the L-BFGS approximation interacting with the bordered saddle, not the
  subspace solver itself. The exact-Krylov failures instead shift to `max_time`
  (158) — slow, not broken. This cluster is the biggest remaining lever: it is
  addressed by a block/constraint saddle preconditioner (using the L-BFGS-aware
  Woodbury inverse for the (1,1) block), not yet the default.
- **The RT-typical `lbfgs/sparse`** route is solid (697 correct, only 3
  `numerical_error` and **no** `solve_error`), validating the L-BFGS + sparse-direct
  path used for radiotherapy-scale problems.
- **Documented-infeasible detection works**: `BURKEHAN` and the rest of the
  documented-infeasible set are correctly scored as expected-infeasible across every
  route.

#### Changes since the 2026-06-24 baseline

This run re-baselines the corpus after a round of robustness fixes (ipax 0.2.0 →
0.3.0). Every route improved on `correct` (+10 to +23; total solved-correct by ≥1
route 783 → 790), and the aggregate deltas map directly to the fixes:

- **`solve_error` eliminated** (14 → **0** corpus-wide): the sparse adapter now
  surfaces a non-finite KKT matrix as a recoverable `LinearSolveError` instead of
  crashing (`lbfgs/sparse` 8→0, `exact/sparse` 5→0, `lbfgs/dense` 1→0).
- **Fewer false `infeasible`** (down ~20 per route): feasibility restoration no
  longer reports a *feasible* iterate as locally infeasible, and resuming from the
  feasible point lets several problems reach the true optimum. Genuinely-infeasible
  problems are still detected (no `expected_infeasible` regressions).
- **`numerical_error` down** on the dense/sparse routes: guards against non-finite
  L-BFGS curvature pairs and a barrier objective, plus a line-search backtrack past
  a step whose gradient overflows (e.g. RAT42LS: `numerical_error` → `optimal`).
- **The `converged` tier** is now reported, crediting the ~120 problems per route
  that reach a valid KKT point at a different documented optimum.

## Reproducing

From a checkout with `IPAX_S2MPJ_DIR` pointing at an
[S2MPJ](https://github.com/GrattonToint/S2MPJ) clone:

```bash
# one configuration, full corpus, dataset-scored, recoverable
python -m benchmarks.runners.s2mpj --all --config exact/sparse \
    --include-objective-free --resume --max-iter 10000 --max-time 300 \
    --out benchmarks/reports/s2mpj_exact_sparse
```

Omit `--config` to sweep the whole matrix in one process. `--resume` keeps an
existing report and skips finished problems; `--exclude` steps past a problem that
natively crashes a backend (the report is persisted after every problem, so an
interrupted sweep never loses work).
