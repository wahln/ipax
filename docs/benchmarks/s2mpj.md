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

## Latest run

### System

| | |
| --- | --- |
| date | 2026-06-24 |
| CPU | 13th Gen Intel Core i9-13900HX (32 logical CPUs) |
| OS | Windows 11 (10.0.26200), AMD64 |
| Python | 3.14.6 |
| ipax | 0.2.0 |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| PyTorch | 2.12.0+cpu |
| sparse factorization | Feral LDLᵀ 0.11.0 (CPU) |

### Methodology

The full `{lbfgs, exact} × {dense, krylov, sparse}` matrix was swept, one
configuration per process (six in parallel, single-threaded each), over the whole
corpus on the NumPy backend with gradient-based scaling (the solver default).
Per-solve budget: `max_iter = 5000`, `max_time = 120 s`. Each route is gated by a
**per-route variable cap** (dense 2000, Krylov 10000, sparse 25000) so a single
full-corpus run stays tractable — three problems exceed the dense cap and are
reported as oversized for the two dense routes. The **objective-free problems** (207
CUTEst feasibility / nonlinear-equation systems) were run as `min 0` subject to the
constraints (`--include-objective-free`).

### Results — full corpus

`correct` counts problems that matched the documented outcome (objective or
infeasibility); the status columns are the raw terminal states.

| config | correct | optimal | acceptable | infeasible | max_iter | max_time | num.err | solve.err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lbfgs/dense`  | 691 / 1098 | 782 | 33 | 169 | 17 | 90  | 6   | 1 |
| `lbfgs/krylov` | 628 / 1101 | 711 | 35 | 69  | 12 | 84  | 190 | 0 |
| `lbfgs/sparse` | 683 / 1101 | 779 | 24 | 177 | 28 | 84  | 1   | 8 |
| `exact/dense`  | 685 / 1098 | 782 | 14 | 161 | 22 | 117 | 2   | 0 |
| `exact/krylov` | 600 / 1101 | 723 | 6  | 172 | 38 | 155 | 7   | 0 |
| `exact/sparse` | **711 / 1101** | **823** | 10 | 159 | 19 | 85  | 0   | 5 |

### Results — optimization vs. feasibility problems

The 207 objective-free problems are feasibility / nonlinear-equation systems with
different semantics (success = found a feasible point; many are inconsistent and
correctly report infeasible), so they are reported separately rather than mixed into
the optimization rate.

| config | optimization (894) | feasibility (207) |
| --- | --- | --- |
| `lbfgs/dense`  | 627 / 891* | 64 / 207 |
| `lbfgs/krylov` | 578 / 894  | 50 / 207 |
| `lbfgs/sparse` | 622 / 894  | 61 / 207 |
| `exact/dense`  | 632 / 891* | 53 / 207 |
| `exact/krylov` | 546 / 894  | 54 / 207 |
| `exact/sparse` | 644 / 894  | 67 / 207 |

<small>* dense routes ran 891 of the 894 optimization problems; three exceed the
dense variable cap.</small>

### Observations

- **`exact/sparse` is the strongest route** — most correct (711) and most
  optimal (823). Exact-Hessian Newton steps factored by the sparse-direct route
  (Feral LDLᵀ with inertia control) is the most robust combination here.
- **The matrix-free Krylov route trails**, consistent with the documented
  [saddle-preconditioning limitation](../concepts/linalg.md). The attribution
  is now clear: `lbfgs/krylov` produces **190 `numerical_error`** results, but with
  the exact Hessian (`exact/krylov`) that collapses to **7** — so those errors come
  from the L-BFGS approximation interacting with the bordered saddle, not the
  subspace solver itself. The exact-Krylov failures instead shift to `max_time`
  (155) — slow, not broken.
- **The RT-typical `lbfgs/sparse`** route is solid (683 correct, only 1
  `numerical_error`), validating the L-BFGS + sparse-direct path used for
  radiotherapy-scale problems.
- **Documented-infeasible detection works**: `BURKEHAN` is correctly scored as
  expected-infeasible across every route.

## Reproducing

From a checkout with `IPAX_S2MPJ_DIR` pointing at an
[S2MPJ](https://github.com/GrattonToint/S2MPJ) clone:

```bash
# one configuration, full corpus, dataset-scored, recoverable
python -m benchmarks.runners.s2mpj --all --config exact/sparse \
    --include-objective-free --resume --max-iter 5000 --max-time 120 \
    --out benchmarks/reports/s2mpj_exact_sparse
```

Omit `--config` to sweep the whole matrix in one process. `--resume` keeps an
existing report and skips finished problems; `--exclude` steps past a problem that
natively crashes a backend (the report is persisted after every problem, so an
interrupted sweep never loses work).
