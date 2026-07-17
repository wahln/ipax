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
| date | 2026-07-15 |
| CPU | 13th Gen Intel Core i9-13900HX (32 logical CPUs) |
| OS | Windows 11 (10.0.26200), AMD64 |
| Python | 3.14.6 |
| ipax | 0.6.1 + μ/line-search arc (develop @ `65d6a8e`) |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| PyTorch | 2.12.0+cpu |
| sparse factorization | Feral LDLᵀ (CPU) |

### Methodology

The full `{lbfgs, exact} × {dense, krylov, sparse}` matrix was swept in one run
(`--all`), **four problems concurrently** (`--jobs 4`) with single-threaded
BLAS. Gradient-based scaling (the solver default), NumPy backend. Per-solve
budget: `max_iter = 10000`, `max_time = 300 s`.
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
| `lbfgs/dense`  | 734 / 1098 | 871 | 762 | 108 | 58 | 86 | 63 | 13 | 7  | 1 | 0 | 0 |
| `lbfgs/krylov` | 716 / 1101 | 851 | 744 | 106 | 58 | 72 | 59 | 12 | 49 | 1 | 0 | 0 |
| `lbfgs/sparse` | 745 / 1101 | 882 | 755 | 126 | 59 | 78 | 62 | 15 | 5  | 1 | 0 | 0 |
| `exact/dense`  | 732 / 1098 | 859 | 776 | 82  | 58 | 69 | 60 | 24 | 28 | 1 | 0 | 0 |
| `exact/krylov` | 681 / 1101 | 824 | 773 | 50  | 58 | 73 | 54 | 24 | 69 | 0 | 0 | 0 |
| `exact/sparse` | **776 / 1101** | **909** | **858** | 50 | 59 | 46 | 58 | 16 | 12 | 1 | 1 | 0 |

Solved-correct by **at least one route: 809 / 1101**; converged by at least one
route: **942 / 1101**.

### Results — optimization vs. feasibility problems

The 207 objective-free problems are feasibility / nonlinear-equation systems with
different semantics (success = found a feasible point; many are inconsistent and
correctly report infeasible), so they are reported separately rather than mixed into
the optimization rate. The optimization column shows `correct` (`converged`).

| config | optimization (894) | feasibility (207) |
| --- | --- | --- |
| `lbfgs/dense`  | 670 / 891* (807) | 64 / 207 |
| `lbfgs/krylov` | 653 / 894  (788) | 63 / 207 |
| `lbfgs/sparse` | 681 / 894  (818) | 64 / 207 |
| `exact/dense`  | 679 / 891* (805) | 53 / 207 |
| `exact/krylov` | 614 / 894  (757) | 67 / 207 |
| `exact/sparse` | 702 / 894  (835) | 74 / 207 |

<small>* dense routes ran 891 of the 894 optimization problems; three exceed the
dense variable cap.</small>

### Observations

- **`exact/sparse` is the strongest route** — most correct (776) and most
  optimal (858). Exact-Hessian Newton steps factored by the sparse-direct route
  (Feral LDLᵀ with inertia control) is the most robust combination here.
- **`numerical_error` is essentially gone** (51 → 0 on `exact/dense`, and 0–1
  on every route). A Newton step the δ_w regularization ladder cannot complete
  now hands to feasibility restoration (Wächter & Biegler §3.1→§3.3) instead of
  a crash-like `numerical_error`: the objective-free nonlinear-equation / NLS
  cluster (`min 0` s.t. `r(x)=0`, where the equality multipliers diverge and no
  δ_w regularizes the runaway Hessian) now resolves to the honest
  `infeasible`/`restoration_failed`/`stalled` — the same verdict the L-BFGS
  routes already gave — and DEMBO7/KISSING recover to `optimal`.
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
- **The RT-typical `lbfgs/sparse`** route is solid (745 correct, 0
  `numerical_error`), validating the L-BFGS + sparse-direct path used for
  radiotherapy-scale problems.
- **Documented-infeasible detection works**: `BURKEHAN` and the rest of the
  documented-infeasible set are correctly scored as expected-infeasible across
  every route.
- **No crashes**: the `solve_error` column is zero everywhere for the first
  time (the LINSPANH and HS9 crashes of the previous run are gone).

#### Changes since the 2026-07-13 run

Every route improved on `correct`: net **+46** over the previous (2026-07-13)
baseline (4338 → 4384 across the six configs), with **every config positive** —
`exact/sparse` leads at 776 (was 769), `lbfgs/sparse` gained most (729 → 745).
Solved-correct by ≥1 route rose to 809 (was 807; converged 942). The deltas map
to the barrier-μ / line-search arc landed since the correctness-hardening
release:

- **The W&B filter is re-initialized whenever μ changes.** The filter's
  `(θ, φ_μ)` entries carry a barrier objective that is meaningful only for the
  μ they were recorded at; ipax previously kept one filter for the whole solve,
  so stale old-μ entries gated new-μ trials as a spurious rejection that
  tightened as μ shrank. Matching Wächter & Biegler / IPOPT (the filter is
  reset at every barrier update) removed that drag.
- **A repeated feasible-point re-center now raises μ instead of treadmilling.**
  When a line search fails at an already-feasible iterate and re-centering the
  barrier at the *same* μ fails again, the barrier parameter is now raised to
  the scale of the stall's KKT error (Nocedal, Wächter & Waltz §5.1) and the
  free-mode μ oracle is suspended until progress resumes — clearing several
  stalled power-flow (`ACOPP`/`ACOPR`) and least-squares runs.
- **The L-BFGS Hessian route gained inertia-guided δ_w.** The compact
  quasi-Newton middle-block signature now folds into the expected-inertia
  target (Haynsworth additivity), so a genuinely nonconvex L-BFGS-route problem
  gets the same inertia correction the sparse-direct route already had, instead
  of relying on Powell damping alone.
- **OET7 — a long-standing basin-flip limitation — is now `optimal`** on three
  routes. The one remaining non-blocking limitation is **AGG** — a feasible
  netlib LP the solver fails to converge (verified feasible vs HiGHS).

!!! note "Adaptive-μ line-search acceptance"
    Two acceptance heuristics from this arc target the adaptive-barrier /
    radiotherapy regime rather than the general corpus, and neither affects the
    table above:

    - **`LineSearchOptions.feasible_kkt_progress`** — disabled by default. As a
      default it cost 48 corpus flips: on unconstrained/bounds-only problems
      (θ ≡ 0, exactly its domain) accepting Armijo-failing but KKT-decreasing
      steps walked nonconvex least-squares runs into worse stationary points.
      Set it (e.g. `0.1`) to opt in.
    - **`LineSearchOptions.free_mode_acceptance`** — defaults to `"rigorous"`
      (the Wächter & Biegler gate in both regimes). The NWW §5
      `"obj-constr-filter"` weak test is opt-in, and in any case only reachable
      under a **non-monotone `mu_schedule`** (itself opt-in; the default
      schedule is `monotone`), so the default solver never uses it.

    Paired full-corpus A/Bs of the weak test against `"rigorous"` are
    **≈ neutral overall** — `quality` +6, `probing` ±0 — but with heavy two-way
    churn (~55 flips each way per arm). About half the regressions are
    *wrong-optimum*: the weak per-trial test drops the merit-function
    guardrail, so on basin-sensitive nonconvex problems (`WOMFLET`, `OET7`,
    `SPIRAL`, `READING5`, `ELATTAR`, `DISCS`) the solver reliably converges to
    a different, worse local optimum — the same set in both arms, so this is
    characterizable rather than run-to-run noise. `"rigorous"` is also the
    IPOPT-parity setting: released IPOPT never weakens its *per-trial* test
    either (its `(f, θ)` margin filter is an *iterate*-level progress check),
    and the mechanisms that make its free mode work — the filter reset on every
    μ change and the KKT-error monitor — are unconditional here.

    Enable the weak test (`free_mode_acceptance="obj-constr-filter"`) for
    central-path-following workloads (radiotherapy-style) where the rigorous
    gate grinds at near-feasible iterates: at θ ≈ 0 the eq. (19) switching
    condition degenerates, so every trial faces the full Armijo test and
    iterations can cost 10+ backtracks.

!!! note "Two Wächter & Biegler filter refinements (opt-in)"
    Both are faithful to the paper — and verified line-by-line against IPOPT's
    `FilterLSAcceptor` — yet both **lost** on the full corpus as defaults, so
    both are disabled by default and neither affects the table above. A paired
    sweep (2026-07-17) attributed the two independently; they sum exactly to the
    combined −14.

    - **`LineSearchOptions.gamma_alpha`** (γ_α, default `None`) switches the
      minimum step size from the flat `alpha_min_frac` to the **adaptive eq. (23)
      rule**, so a hopeless ray concedes to restoration as soon as no acceptable
      step remains rather than after a fixed 27 halvings. IPOPT applies this
      unconditionally with γ_α = `0.05`. Scored **−4**: conceding earlier is the
      point of the rule, but ipax's restoration phase is a weaker recovery than
      IPOPT's, so 7 problems went `optimal`/`acceptable` → `stalled` against 3
      recovered.
    - **`LineSearchOptions.ftype_requires_theta_min`** (default `False`) adds the
      **θ ≤ θ_min conjunct to the f-type test** (W&B Algorithm A, Step 4), so an
      infeasible iterate is judged by the eq. (20) sufficient-decrease test in θ
      *or* φ instead of by Armijo on φ. Scored **−10**: 10 problems reached a
      *different, worse* optimum (`ELATTAR`, `HS97`, `HS98`, `LUKVLE3` — the
      θ-branch admits steps Armijo refused, changing the basin) and 12 stalled
      (above θ_min almost every accepted step becomes θ-type and augments the
      filter, whose entries then choke later iterations).

    Caveat for whoever revisits this: ipax's `gamma_phi` (`1e-5`) and `eta_phi`
    (`1e-4`) differ from IPOPT's shipped `1e-8`/`1e-8`, so these structural
    changes have never been exercised alongside the constants they were designed
    against — a prerequisite to retrying them as defaults.

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
