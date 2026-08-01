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
| date | 2026-08-01 (v20) |
| CPU | 13th Gen Intel Core i9-13900HX (32 logical CPUs) |
| OS | Windows 11 (10.0.26200), AMD64 |
| Python | 3.14.6 |
| ipax | 0.9.0 + the terminal KKT certificate (develop @ `2e7f3a1`) |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| PyTorch | 2.12.0+cpu |
| sparse factorization | Feral LDLᵀ (CPU) |

### Methodology

The full `{lbfgs, exact} × {dense, krylov, sparse}` matrix was swept in one run
(`--all`), **six problems concurrently** (`--jobs 6`) with single-threaded
BLAS. Gradient-based scaling (the solver default), NumPy backend. Per-solve
budget: `max_iter = 10000`, `max_time = 300 s`.

!!! warning "`--jobs` moves one column"

    v20 ran at `--jobs 6`, matching the v19 baseline, so the delta below is
    jobs-clean. Concurrency cannot change which point a solve converges to, but
    `max_time` is a wall-clock verdict and machine load still moves it: v20's
    objective-free rows ran in a resumed second session against a lighter
    queue, so read `max_time` transitions on that subset as load, not code.
    Prefer iteration counts when comparing across runs with different load.
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

| config | correct | converged | optimal | acceptable | infeasible | stalled | rest.failed | max_iter | max_time | unbounded | num.err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lbfgs/dense`  | 774 / 1098 | 917 | 790 | 126 | 58 | 37 | 64 | 15 | 7  | 1 | 0 |
| `lbfgs/krylov` | 752 / 1101 | 894 | 769 | 124 | 58 | 26 | 61 | 12 | 50 | 1 | 0 |
| `lbfgs/sparse` | 780 / 1101 | 924 | 782 | 141 | 59 | 30 | 63 | 18 | 6  | 1 | 1 |
| `exact/dense`  | 768 / 1098 | 898 | 806 | 91  | 58 | 27 | 62 | 16 | 37 | 1 | 0 |
| `exact/krylov` | 706 / 1101 | 852 | 795 | 56  | 58 | 41 | 54 | 20 | 77 | 0 | 0 |
| `exact/sparse` | **797 / 1101** | **935** | **877** | 57 | 59 | 22 | 59 | 13 | 12 | 1 | 1 |

Solved-correct by **at least one route: 828 / 1101**; converged by at least one
route: **963 / 1101**.

### Changes since the v19 baseline

**+46 correct across the six configs (4531 → 4577), every config positive** —
46 fixed against **zero** broken, with zero linear-solver route changes.

| config | v19 | v20 | Δ |
| --- | --- | --- | --- |
| `lbfgs/dense`  | 764 | 774 | **+10** |
| `lbfgs/krylov` | 740 | 752 | **+12** |
| `lbfgs/sparse` | 772 | 780 | **+8** |
| `exact/dense`  | 762 | 768 | **+6** |
| `exact/krylov` | 700 | 706 | **+6** |
| `exact/sparse` | 793 | 797 | **+4** |

The gain is the **terminal KKT certificate**: a run ending
`stalled`/`max_iter`/`max_time` now re-judges its returned best iterate — with
repaired candidate multipliers where the recorded ones drifted — and reports
`acceptable` when the full scaled residual passes the acceptable-stopping
tolerances. Nearly every fixed row is the certificate signature
`stalled → acceptable`, in two families the per-iteration test structurally
missed: the objective-free **nonlinear-equation systems**, whose rank-deficient
`∇c` under-determines the multipliers so the recorded KKT is drifted-dual noise
(`NONSCOMPNE`, `VANDERM1`–`4`, `WATSONNE`, `DECONVBNE`, `CYCLOOCT`,
`DRCAVTY1`), and **least-squares valleys** whose best iterate sits inside the
acceptable band while the line search froze at a worse tail iterate (`WEEDS`,
`MGH09LS`, `ROSZMAN1LS`, `GAUSS1LS`/`3LS`, `HAHN1LS`, `FBRAIN3LS`, `LSC2LS`,
`HADAMALS`). `stalled` drops on every config (48 → 37 on `lbfgs/dense`,
28 → 22 on `exact/sparse`); the handful of `max_time → acceptable`/`optimal`
flips (`ZAMB2*`, `OET4`, `KISSING2`, `TARGUS`) are wall-clock luck under the
`--jobs` caveat, not the certificate.

All 85 objective drifts are on runs that were wall-clock-bounded (`max_time`)
in **both** sweeps — the iterate a budget cut lands on moves with machine load
— and 61 of the 85 are unscored. A terminal-only change cannot move a
trajectory, and the zero-drift result on every non-budget-bounded row confirms
it didn't.

### Results — optimization vs. feasibility problems

The 205 objective-free problems are feasibility / nonlinear-equation systems with
different semantics (success = found a feasible point; many are inconsistent and
correctly report infeasible), so they are reported separately rather than mixed into
the optimization rate. The optimization column shows `correct` (`converged`).

| config | optimization (896) | feasibility (205) |
| --- | --- | --- |
| `lbfgs/dense`  | 685 / 893* (828) | 89 / 205 |
| `lbfgs/krylov` | 665 / 896  (807) | 87 / 205 |
| `lbfgs/sparse` | 690 / 896  (834) | 90 / 205 |
| `exact/dense`  | 685 / 893* (815) | 83 / 205 |
| `exact/krylov` | 619 / 896  (765) | 87 / 205 |
| `exact/sparse` | 705 / 896  (843) | **92 / 205** |

<small>* dense routes ran 893 of the 896 optimization problems; three exceed the
dense variable cap.</small>

Both columns gain from the terminal certificate: the feasibility side because
the equation systems' recorded residuals were drifted-multiplier noise at
points that were in fact acceptably feasible (79–89 → 83–92), the optimization
side through the least-squares family. The split itself is derived by
inspecting each built problem for a constant objective (205/896).

### Observations

!!! note "`SparseOptions(kkt_route="auto")` verification (v18, 2026-07-20)"
    The sparse route's KKT form now defaults to `"auto"`. Its verification sweep
    isolates cleanly because the auto gate only engages where the
    normal-equations prerequisites hold: **`exact/sparse` changed route on 0 of
    1101 problems** (the fill probe is withheld for non-L-BFGS Hessians), so any
    `exact/*` movement in the A/B is provably unrelated to the change — a
    built-in control. It confirmed 4 of the 5 corpus flips as timing noise
    (all `max_time` transitions, from running the sweep at `--jobs 8` against a
    `--jobs 4` baseline; they net to ±0).

    On `lbfgs/sparse`, **22 problems** switched to the n×n condensation. An
    objective-level audit (not just the `correct` flag) splits them: **19 return
    the same objective** to ≤1e-8 relative, **1 improves** (`CRESC50` 1.061 →
    0.599, and 4× faster), and **2 land on a different, worse local optimum** —
    `ELATTAR` (0.1427 → 74.2) and `OET7` (4.45e-5 → 0.0872).

    On the 20 same-answer problems the route is **41.7s → 12.1s (3.44×
    aggregate**, median 1.79×, 13 faster / 7 slower — every slowdown sub-second).
    `OET6` (101 → 38 iterations) and `CRESC50` (2979 → 1223) are genuine
    conditioning wins at an unchanged-or-better objective, while well-scaled
    cases keep identical iteration counts (44 → 44, 71 → 71, 95 → 95).

    **`OET7` is a scoring caveat worth knowing.** Its 539 → 36 iteration drop is
    *not* a conditioning win: it is convergence to a ~2000× worse optimum, and
    the corpus metric scores it `correct` in both runs because `OET7` carries no
    reference objective, so any certified convergence counts. The corpus
    correctness delta therefore *understates* the basin cost of this change —
    2 basin changes, of which only `ELATTAR` (which has a reference objective) is
    visible in the ±count. Pass `kkt_route="augmented"` to restore the previous
    form.

- **`exact/sparse` is the strongest route** — most correct (797) and most
  optimal (877). Exact-Hessian Newton steps factored by the sparse-direct route
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
  22–97 per route). Two mechanisms: a run whose returned best iterate already
  satisfies the relaxed (acceptable-level) KKT tolerance now reports
  `acceptable` instead of `max_time`/`max_iter` (the DIAMON2DLS/DMN
  least-squares family — oscillating at KKT ~1e-7 without ever holding it for
  the acceptable window — is now scored *correct*), and the stall detector
  ends frozen-iterate grinds early. As of v20 the same re-judging extends to
  `stalled` exits, and a **terminal KKT certificate** additionally repairs
  drifted multipliers at the returned iterate before giving up on it — a
  genuinely active constraint's certificate fails, so it under-certifies at
  worst (see *Changes since the v19 baseline*).
- **`unbounded` is now a certified verdict**: it requires the objective to
  diverge below `−diverging_iterates_tol`, not just a large iterate norm.
  KOEBHELB — whose iterate wanders past 1e22 and then converges to f = 112 —
  flipped from `unbounded` to `optimal` on the exact routes, while genuinely
  unbounded problems (INDEF) are still detected.
- **The RT-typical `lbfgs/sparse`** route is solid (780 correct, a single
  `numerical_error`), validating the L-BFGS + sparse-direct path used for
  radiotherapy-scale problems.
- **Documented-infeasible detection works**: `BURKEHAN` and the rest of the
  documented-infeasible set are correctly scored as expected-infeasible across
  every route.
- **No crashes**: the `solve_error` column is zero everywhere for the first
  time (the LINSPANH and HS9 crashes of the previous run are gone).

#### Changes in the v19 baseline (2026-07-31), since v15

Retained for provenance: these are the deltas that produced the **v19** numbers
the section above is measured against, not the current run.

**+147 correct across the six configs (4384 → 4531), every config positive**
(157 fixed / 10 broken; nine of the ten breakages were `max_time`/`max_iter`
under a `--jobs` 4 → 6 change, the tenth an `ELATTAR` basin flip). The gain was
one coherent family: `stalled → optimal` on the CUTEst **nonlinear-equation
systems** (`BEALENE`, `BIGGS6NE`, `CYCLOOCF`, `CYCLOOCT`, `DEVGLA1NE`/`2NE`,
`HYDCAR6`, `METHANL8`, `ROBOT`, `SEMICON1`/`2`, `VARDIMNE`, …) — the class-A
feasibility fix: the feasible-point re-center guard required inequality
constraints to exist, so on equality-only systems a globalization failure at a
feasible iterate walked into a no-op restoration livelock. Three changes
together (guard on any constraints, restoration's own ℓ∞ measure, least-squares
equality-dual repair) cleared it on every route, not just the L-BFGS ones.
Earlier provenance (the v15 barrier-μ / line-search arc: per-μ filter reset,
re-center μ-escalation, L-BFGS inertia-guided δ_w) lives in the repository
history and the v15 report.

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

    The constants caveat was then closed by a full 2×2 (2026-07-18): ipax's
    `gamma_phi`/`eta_phi` (`1e-5`/`1e-4`) differ from IPOPT's shipped
    `1e-8`/`1e-8`, so the sweep was repeated under IPOPT's constants with the
    opt-ins off and on.

    For the record, **ipax's values are the ones Wächter & Biegler publish**
    (§2.4: `γ_θ = γ_φ = 1e-5`, `η_φ = 1e-4`, `δ = 1`, `s_θ = 1.1`, `s_φ = 2.3`,
    `γ_α = 0.05`, `θ_max = 1e4·max{1,θ(x₀)}`, `θ_min = 1e-4·max{1,θ(x₀)}`) —
    every constant in the filter line search matches the paper. IPOPT's shipped
    `1e-8`/`1e-8` is a later retuning that departs from its own paper. The paper
    adds that the values "have been chosen because they seem to produce overall
    good performance… but the most efficient choice … [is] usually problem
    dependent", which is exactly what the table below measures.

    | corpus correct (of 6600)  | opt-ins off     | opt-ins on | marginal |
    |---------------------------|-----------------|------------|----------|
    | ipax constants (default)  | **4384** (v15)  | 4370       | **−14**  |
    | IPOPT constants           | 4371            | 4373       | **+2**   |

    The interaction is the finding: the two structural refinements lose −14
    under ipax's constants but score +2 under the constants they were designed
    alongside — structure and acceptance margins are co-adapted, and the hybrid
    is what loses. No default changes: ipax's own co-adapted package (4384)
    beats every other cell, including full-IPOPT emulation (4373). The
    constants trade is also *characterizable*, not uniform: IPOPT's looser
    margins fix `AGG` (the long-open feasible-LP failure) on 3 of 6 configs
    plus `MINSURFO`/`BLOCKQP1`/`BATCH`, while breaking the basin-sensitive
    eigen/packing cluster (`EIGMAXA`, `EIGMINA`, `KISSING*`). Follow-up
    (2026-07-20) showed the `AGG` half of that is **not** really a margin
    effect — see the note below. Full-IPOPT emulation, when wanted
    (e.g. cross-solver comparisons):
    `LineSearchOptions(gamma_phi=1e-8, eta_phi=1e-8, gamma_alpha=0.05,
    ftype_requires_theta_min=True)`.

!!! note "`AGG` is a μ-schedule problem, not a margin problem (2026-07-20)"
    `AGG` — a netlib LP that is *feasible* (HiGHS cross-check) but which the
    default configuration ends as `restoration_failed` at an infeasible point
    (constraint violation 19.9, objective −3.11e7 vs the true −3.60e7) — was
    provisionally attributed to acceptance margins, because IPOPT's constants
    fix it. Isolating the two constants shows the effect is entirely `eta_phi`
    (the Armijo constant); `gamma_phi` alone changes nothing, not even the
    iteration count.

    That is *not* a roundoff artifact. Instrumenting the Armijo test records 64
    failures with shortfalls of 4.5e3 … 3.6e8 (median 1.2e6), against a
    scale-relative roundoff slack (IPOPT's `Compare_le`, `10·ε·|φ0|`) of only
    6.9e-8 — **0 of 64** failures are within it. The median shortfall instead
    matches the `eta_phi` 1e-4 → 1e-8 relaxation exactly, which means
    `φ_t ≈ φ0`: the step delivers essentially **no** barrier-objective change
    while `∇φᵀd` predicts a decrease of order 1e10 on an objective of 3e7. So
    `eta_phi=1e-8` "fixes" `AGG` only by relaxing Armijo into a near-vacuous
    non-increase test — it masks the real defect rather than addressing it.

    The real defect is the barrier schedule, and it is fixable with shipped
    options. `AGG` reaches the true optimum under
    **`mu_schedule="quality"` in 42 iterations** (cleanest), under
    `globalization="breedveld"` in 149, and under the `eta_phi` hack in 130;
    it fails identically with `scaling="none"`, so it is not a scaling issue
    either. The monotone schedule drives μ off the iterate's central path,
    producing the enormous-but-unrealizable predicted decrease; an adaptive μ
    oracle re-targets μ to the iterate and the problem solves cleanly.

    **Routing hint, not a default change.** `mu_schedule="quality"` stays
    opt-in: its corpus A/B is ≈ neutral (+6, with heavy two-way churn), and on
    the radiotherapy workload it is actively *worse* than the default (6–8 of 25
    TROTS Prostate_BT cases certified, against 11 for the default `monotone` and
    13 for `globalization="breedveld"`). But if a problem stalls or ends
    `restoration_failed` while its constraint violation stays large — the `AGG`
    signature, and an LP-like one — trying `mu_schedule="quality"` is cheap and
    is the first thing to reach for.

!!! note "Scale-aware slack initialization (`slack_init_scale`, opt-in, 2026-07-26)"
    `BarrierOptions.slack_init_scale` (default `0.0`) raises the flat slack floor
    (`1e-2`) to `max(1e-2, slack_init_scale·max|g(x₀)|)`. It targets a failure the
    IPOPT cross-check isolated on radiotherapy starts: on a deeply-infeasible
    point every violated-constraint slack is pinned at the flat floor, so the
    Newton direction drives it toward its infeasible target `s = −g < 0` and the
    fraction-to-boundary rule clips the primal step to ~`1e-3` for ~15 iterations
    (the "Phase-1 stall"). Scaling the floor to the constraint magnitude gives the
    slacks room and — via `y = μ_init/s` — starts the multipliers at a saner scale.
    On `Protons_01` (TROTS), `slack_init_scale=0.1` reaches feasibility at
    iteration ~20 (IPOPT parity) versus ~42 with the flat floor, and roughly
    halves the transient objective excursion (peak ~998 vs ~1917).

    It **stays opt-in**: the win is specific to deeply-infeasible, many-constraint
    starts. A full three-route A/B (`slack_init_scale=0.1`, full corpus, NumPy) is
    **net-neutral** — the general corpus is untouched by construction (the floor
    only bites where a constraint is violated/near-active at `x₀`), and on the 656
    already-correct `lbfgs/dense` cells the iteration count is a wash (median Δ0,
    55 fewer / 56 more).

    | correct (Δ vs flat floor) | `lbfgs/dense` | `lbfgs/krylov` | `lbfgs/sparse` |
    |---------------------------|---------------|----------------|----------------|
    | net                       | **+3** (+6/−3)| **−1** (+5/−6) | **−3** (+6/−9) |

    The gains are robust across all three routes (`HS59`, `HS97`, `HS98` reach the
    correct optimum; `SINROSNB` clears its budget) but so are two regressions
    (`HS116` → `restoration_failed`, `WOMFLET` → a different, worse optimum), with
    the remainder route-dependent basin/timing churn. Net-neutral-with-churn does
    not clear the bar to change the default; the radiotherapy win is available
    per-solve via the option. A value in `[0.05, 0.5]` matches the constraint
    scale on RT-sized problems.

## Reproducing

From a checkout with `IPAX_S2MPJ_DIR` pointing at an
[S2MPJ](https://github.com/GrattonToint/S2MPJ) clone:

```bash
# one configuration, full corpus, dataset-scored, recoverable, 5 problems at a time
python -m benchmarks.runners.s2mpj --all --config exact/sparse --jobs 5 \
    --include-objective-free --resume --max-iter 10000 --max-time 300 \
    --out benchmarks/reports/s2mpj_exact_sparse
```

A sweep is only meaningful against a baseline, so A/B two reports with:

```bash
python -m benchmarks.runners.compare \
    benchmarks/reports/s2mpj_v19.json benchmarks/reports/s2mpj_v20.json \
    --config lbfgs/sparse
```

It reports the correctness delta per configuration, the per-config count of
problems whose **linear-solver route changed** (a config with zero route changes
is a built-in control when A/B-ing linear-algebra work — anything moving there is
unrelated), and **objective drift**: problems whose objective moved materially,
flagged `[unscored]` when the problem carries no dataset reference objective.
Those are the ones the correctness count structurally *cannot* see — always read
them before accepting a delta.

!!! tip "Both sweeps should use the same `--jobs`"
    Problems near the `--max-time` cap flip status with machine load, so a sweep
    run at a different parallelism than its baseline injects `max_time` churn
    that is easily mistaken for a real effect.

Omit `--config` to sweep the whole matrix in one process. `--jobs N` runs N
problems concurrently in worker processes (pin BLAS threads, e.g.
`OMP_NUM_THREADS=1`, so per-solve timings stay comparable); reports are flushed
in sorted order after every problem, so they are deterministic and an
interrupted sweep never loses work. `--resume` keeps an existing report and
skips finished problems; `--exclude` steps past a problem that natively crashes
a backend (on a crashed worker pool the runner exits `2` and the `.inflight`
file names the candidate culprits).
