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
| date | 2026-09-06 (v31) |
| CPU | 13th Gen Intel Core i9-13900HX (32 logical CPUs) |
| OS | Windows 11 (10.0.26200), AMD64 |
| Python | 3.14.6 |
| ipax | 0.10.1 + the 0.11.0 performance/restoration arc (develop @ `761ac00`) |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| PyTorch | 2.12.0+cpu |
| sparse factorization | Feral LDLᵀ (CPU) |

### Methodology

The full `{lbfgs, exact} × {dense, krylov, sparse}` matrix was swept in one run
(`--all`), **ten problems concurrently** (`--jobs 10`) with single-threaded
BLAS. Gradient-based scaling (the solver default), NumPy backend. Per-solve
budget: `max_iter = 10000`, `max_time = 300 s`.

!!! warning "`max_time` verdicts are wall-clock, and concurrency moves them"

    Concurrency cannot change which point a solve converges to, but `max_time`
    is a wall-clock verdict, so a memory-bandwidth-heavy problem sharing the
    machine with nine others can cross the cap that it cleared before.
    **Prefer iteration counts when comparing across runs.**

    v22 ran at `--jobs 10` against v20's `--jobs 6`, and that difference owns
    three of its five regressions: `CHANDHEU` on the `exact/*` routes solves in
    **15 iterations either way**, but took 174 s in v20, 308–345 s in v22 — and
    ~100 s when re-run alone at the same commit. Its own `lbfgs/*` rows are the
    control: identical 15 iterations in both sweeps, 2.4–4× the wall time, so
    the cost is per-evaluation contention, not a trajectory change. Corpus-wide
    the machine was, if anything, *faster* in v22 (median `solve_time` ratio
    0.965 on rows with identical `n_iter`), which is what localises the effect
    to CHANDHEU's scheduling slot.

    The general check, worth reusing: compare `solve_time` only across rows
    with identical `n_iter` — same work — and treat the median as the machine
    factor before attributing anything to code.
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
| `lbfgs/dense`  | 779 / 1098 | 922 | 793 | 128 | 56 | 30 | 66 | 17 | 7  | 1 | 0 |
| `lbfgs/krylov` | 764 / 1101 | 906 | 776 | 129 | 56 | 18 | 63 | 13 | 45 | 1 | 0 |
| `lbfgs/sparse` | 785 / 1101 | 926 | 786 | 139 | 56 | 30 | 65 | 20 | 4  | 1 | 0 |
| `exact/dense`  | 777 / 1098 | 911 | 805 | 105 | 56 | 13 | 67 | 33 | 16 | 1 | 2 |
| `exact/krylov` | 713 / 1101 | 861 | 796 | 64  | 55 | 34 | 58 | 21 | 73 | 0 | 0 |
| `exact/sparse` | **798 / 1101** | **936** | **877** | 58 | 56 | 17 | 62 | 16 | 13 | 1 | 1 |

Solved-correct by **at least one route: 834 / 1101**; converged by at least one
route: **968 / 1101**.

!!! note "Reading the two budget columns together"

    `max_iter` and `max_time` are one quantity split by which cap a
    non-converging run happens to reach first, so they must be read as a pair.
    On `exact/dense` v20 → v22, `max_iter` rose 16 → 30 while `max_time` fell
    37 → 19: **13 of the 14 new `max_iter` rows were already incorrect in both
    sweeps** (`AVION2`, `CLNLBEAM`, `ERRINBAR`, `HS68`, `LUKVLE14`, `NCVXQP3`/
    `6`/`8`/`9`, `ORTHREGA`, `ORTHREGE`, `TRAINH`, `TRO6X2`) and had previously
    run out of *wall clock* at 3 000–9 000 iterations. Reaching 10 001
    iterations inside the same 300 s means they got faster, not worse. The one
    correct-affecting row is `COSHFUN`, covered below.

### Changes since the v22 baseline

**+38 correct across the six configs (4578 → 4616)** — 57 fixed against 19
broken, every config positive. The intervening work was the `0.11.0`
performance and restoration arc, landed and gated sweep by sweep (v23 … v31;
every default-affecting commit carries its own full-corpus gate):

| config | v22 | v27 | v29 | v30 | v31 | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| `lbfgs/dense`  | 774 | 773 | 779 | 779 | 779 | **+5** |
| `lbfgs/krylov` | 752 | 757 | 758 | 758 | 764 | **+12** |
| `lbfgs/sparse` | 780 | 782 | 785 | 785 | 785 | **+5** |
| `exact/dense`  | 770 | 766 | 774 | 776 | 777 | **+7** |
| `exact/krylov` | 706 | 711 | 712 | 712 | 713 | **+7** |
| `exact/sparse` | 796 | 797 | 798 | 798 | 798 | **+2** |

The one **route change** is deliberate and documented: on `lbfgs/krylov`
**387 problems** report `krylov (cg, pc=lbfgs-exact)` instead of
`pc=jacobi` — every bound-only system, where the condensed Woodbury inverse
*is* `N⁻¹` and the Krylov route now applies it directly (one apply plus a
true-residual check instead of ~30 CG iterations). No other config changed
route on any problem.

Sweep by sweep:

- **v23 → v24 (2026-08-31), +18.** The three L-BFGS bookkeeping commits
  (bound-only blocks stay on the structured Woodbury path; the compact factor
  stays in `S`/`Y` block form; opt-in quadratic-interpolation backtracking).
  Only knife-edge churn; the interpolation lever alone scored +11 but
  non-unanimously (−1 on the Hessian-agnostic `exact/*` configs, with
  reproducible worse-basin flips on HS97/HS98/OSBORNEB), so it stayed
  **opt-in** — v27 (4586) is the merged default state.
- **v27 → v29 (2026-09-03), +20 (31 fixed / 11 broken).** Restoration now
  probes for a saddle before certifying local infeasibility: on a
  symmetry-invariant subspace (POWERSUMNE, HADAMARD, CYCLOOCT) every
  Gauss-Newton direction stays in the subspace and the first-order
  certificate reported a saddle as infeasibility; a one-shot deterministic
  kick lets the damped loop continue. Plus the SOC factorization-reuse gate
  fix (ZAMB2/ZAMB2m10/ACOPP30/ACOPR30 back to their baseline trajectories).
  10 of the 11 broken rows were on the two *Krylov* main routes
  (DRUGDIS/DALLASS/NET1/SPECANNE → `max_time`): a fresh `δ_w = 0` SOC solve
  there is a full CG ladder per round.
- **v29 → v30 (2026-09-04), +2.** The fix for exactly that: the SOC reuse
  policy follows the solver kind (the `SolverKind.is_direct()` hook). Iterative
  routes re-solve the step's retained system for every correction (Wächter &
  Biegler 2006 eq. (26) verbatim) and never climb a ladder inside SOC; direct
  routes keep the fresh first correction at `δ_w = 0`. The direct routes are
  **trajectory-identical** to v29; `exact/krylov` traded 9 basin flips each
  way at 22 % less total wall (11.2 h → 8.7 h) with DRUGDIS/DALLASS/NET1/
  SPECANNE back from `max_time`.
- **v30 → v31 (2026-09-06), +8 (15 fixed / 7 broken).** Dense Cholesky reuse
  for the L-BFGS block with inequalities (`positive_definite_hint`). Zero
  route changes; `lbfgs/dense` — the config the change targets — has **zero
  correct flips** and a different iteration count on only 46 of 1098 rows.
  All 7 broken rows are Krylov churn (ORTH*/KISSING/HS111/PALMER3C/4C and the
  DJTL cap boundary). On rows with identical `n_iter` the wall ratio is 0.966
  overall, 0.875 on `lbfgs/dense` and 0.828 on `exact/dense`, against a ±10 %
  machine spread — consistent with the measured per-step speedup, not proof
  of it.
- **v32 (2026-09-16), `lbfgs/dense` only, −2 (1 fixed / 3 broken) — a
  confirming run, not a baseline.** The release-candidate commits after v31
  (PR #13) change dense-route numerics in one place only: a hinted Cholesky
  breakdown on the L-BFGS+inequality block is now bookkept
  (`DenseOptions.pd_hint_failure_limit`, the sticky `pd-hint->lu` marker),
  so this is the one config that could move. It ran on a machine **2.9×
  slower** on identical-`n_iter` rows (507 rows; the v28 throttled-machine
  factor), and all three broken rows are that: `max_time` at the same wall
  in a fraction of v31's iterations (`DMN15332LS` 9934 → 4757, `TWIRIMD1`
  239 → 131, `TAX13322` 520 → 476). The marker answers the question the run
  was for: **12 of 1098 problems** hit a hinted breakdown (`HS13`, `HS90`,
  `HS91`, `HS98`, `CRESC50`, `ELATTAR`, `LUKVLI12`, `OET7`, `POLAK6`, `S365`,
  `TAX13322`, `TFI1`). Six keep v31's iteration count exactly; the rest are
  the known knife-edge family, moved at round-off by later factorizations
  going straight to LU after the kill switch (`ELATTAR` now lands on the
  documented optimum, `HS98`/`HS13` converge in fewer iterations to the
  same value, `OET7` a worse unscored basin, `CRESC50` incorrect either way).
  No marked problem changed `correct`.

Two sweeps in the series are **not** in the table because they were not
baselines: **v28** (2026-09-03) ran on a throttled machine (2.87× slower on
identical-`n_iter` rows), so 21 of its 26 "regressions" were `max_time`
artefacts and v29 was re-run against v27; and the paired **v29 arm with
matrix-free restoration forced on every problem** scored −12/6600 — on a
corpus whose largest problem has n ≤ 4999 an `n×n` dense solve is cheaper than
a Krylov ladder per damping trial — which is why
`RestorationOptions(linear_solver="auto")` is size-gated (dense below 10 000
variables, Krylov above) and reproduces the dense results on the whole corpus.

Of the 19 rows broken over the whole span, 9 are on the two Krylov routes and
7 more are the known knife-edge family on the other routes (`DJTL`'s cap
boundary, `HS111`, `PALMER4C`, `ORTHREGA`, `CURLY20`, `DECONVB`/`DECONVBNE`);
the notable trajectory changes are `SPANHYD` (`optimal → stalled` on
`lbfgs/dense`, `acceptable → stalled` on `exact/krylov`), `CmRELOAD`
(`lbfgs/dense`, `optimal → stalled`, while it *recovered* from `max_time` on
`lbfgs/krylov`) and a different-basin `HALDMADS` on `exact/dense`. The fixed
side is broader than churn: `HYDCAR20` went `stalled/max_time → optimal` on
four routes, `CHANDHEU` cleared `max_time` on all three `exact/*` routes,
`VANDERM1`/`VANDERM2`, `SPECANNE`, `CORE2`, `DEMBO7`, `SPINOP` and the
`DMN*LS` least-squares family now certify, and `DISCS` left
`restoration_failed` for `acceptable`.

#### Changes in the v22 baseline (2026-08-07), since v20

Retained for provenance: **+1 correct across the six configs (4577 → 4578)** —
6 fixed against 5 broken, with **zero linear-solver route changes**. The
intervening work was the
`0.10.0` linear-algebra arc (symmetric rank-k Gram update on both the SciPy and
CuPy adapters, the fill-certified tall sparse-NE routing gate, and
opt-in-then-default mixed-precision Gram accumulation), none of which is
expected to move a float64 corpus — and it did not.

| config | v20 | v21 | v22 | Δ |
| --- | --- | --- | --- | --- |
| `lbfgs/dense`  | 774 | 774 | 774 | ±0 |
| `lbfgs/krylov` | 752 | 752 | 752 | ±0 |
| `lbfgs/sparse` | 780 | 780 | 780 | ±0 |
| `exact/dense`  | 768 | 771 | 770 | **+2** |
| `exact/krylov` | 706 | 707 | 706 | ±0 |
| `exact/sparse` | 797 | 797 | 796 | **−1** |

**Why mixed precision cannot move this corpus.** `gram_dtype="auto"` engages
only where an operator *declares* reduced-precision data through
`gram_accumulate_dtype_hint()`. Nothing in S2MPJ does — the corpus is float64
throughout, and the only producer that declares a hint is the TROTS loader — so
`_resolved_gram_dtype` returns `None` and the mixed branch is never entered.
Verified rather than argued: a 612-row probe across all six configs reported
zero rows on a reduced route.

All five regressions are **cap-boundary cases**, none a correctness change:

- **`CHANDHEU`** (`exact/dense`, `exact/krylov`, `exact/sparse`) — the `--jobs`
  contention case dissected above; re-running the same commit alone gives
  `optimal` in 15 iterations at ~100 s, faster than the v20 baseline row.
- **`COSHFUN`** (`exact/dense`) — converged in v20 at iteration **9974** of a
  10000 cap, i.e. 26 iterations of headroom. Any trajectory perturbation, down
  to a change in floating-point summation order, tips it over.
- **`CORE2`** (`lbfgs/dense`) — flips `optimal ↔ stalled` on **BLAS thread count
  alone**, at the baseline commit as well as at HEAD (optimal at 4 and 16
  threads, stalled at 1 and 8, iteration count swinging 40 ↔ 148). Attributing
  it to any commit is meaningless. Its stall lands at constraint violation
  6.7e3, which is a robustness cliff worth investigating on its own terms.

The six fixed rows are mostly the mirror image — `max_time → optimal/acceptable`
on `KISSING2`, `TWIRIMD1`, `ZAMB2m11`, `FBRAIN3LS` — plus two genuine
trajectory improvements: `VANDERM4` (`stalled` at 2437 iterations →
`acceptable` at 206) and `HALDMADS`, which now converges to the documented
optimum rather than a different local one.

#### Changes in the v20 baseline (2026-08-01), since v19

Retained for provenance: **+46 correct (4531 → 4577), every config positive** —
46 fixed against **zero** broken, zero route changes.

| config | v19 | v20 | Δ |
| --- | --- | --- | --- |
| `lbfgs/dense`  | 764 | 774 | **+10** |
| `lbfgs/krylov` | 740 | 752 | **+12** |
| `lbfgs/sparse` | 772 | 780 | **+8** |
| `exact/dense`  | 762 | 768 | **+6** |
| `exact/krylov` | 700 | 706 | **+6** |
| `exact/sparse` | 793 | 797 | **+4** |

The gain was the **terminal KKT certificate**: a run ending
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

| config | optimization (894) | feasibility (207) |
| --- | --- | --- |
| `lbfgs/dense`  | 687 / 891* (830) | 92 / 207 |
| `lbfgs/krylov` | 673 / 894  (815) | 91 / 207 |
| `lbfgs/sparse` | 691 / 894  (832) | 94 / 207 |
| `exact/dense`  | 689 / 891* (822) | 88 / 207 |
| `exact/krylov` | 624 / 894  (772) | 89 / 207 |
| `exact/sparse` | **705 / 894**  (843) | **93 / 207** |

<small>* dense routes ran 891 of the 894 optimization problems; three exceed the
dense variable cap.</small>

Both columns were lifted by the v20 terminal certificate: the feasibility side
because the equation systems' recorded residuals were drifted-multiplier noise
at points that were in fact acceptably feasible (79–89 → 83–92 at v20, 88–94
here), the optimization side through the least-squares family. The v29 saddle
probe added a few more feasibility rows (POWERSUMNE, HADAMARD, CYCLOOCT are
the poster children).

The split here uses the objective-type letter of each problem's CUTEst
classification (`pbclass`, recorded per row in the report; `N` = no objective),
which yields the same 207 objective-free problems the runner's
`--include-objective-free` gate admits. Earlier pages derived it by building
each problem and asking whether it has an objective (205 at v22, two problems
apart); if you re-derive it that way, do it by *constructing the problem*, not
by re-deriving the predicate: instantiating with an explicit `size=0` rather
than `None` silently builds a degenerate `n = 0` instance for every *scalable*
problem, which then has no objective groups and is miscounted as objective-free
(it inflates the count to 326, with `ARGLINA` collapsing from n=200 to n=0).

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

- **`exact/sparse` is the strongest route** — most correct (798) and most
  optimal (877). Exact-Hessian Newton steps factored by the sparse-direct route
  (Feral LDLᵀ with inertia control) is the most robust combination here.
- **`numerical_error` is essentially gone** (51 → 0–2 on `exact/dense`, and
  0–1 on every other route). A Newton step the δ_w regularization ladder cannot complete
  now hands to feasibility restoration (Wächter & Biegler §3.1→§3.3) instead of
  a crash-like `numerical_error`: the objective-free nonlinear-equation / NLS
  cluster (`min 0` s.t. `r(x)=0`, where the equality multipliers diverge and no
  δ_w regularizes the runaway Hessian) now resolves to the honest
  `infeasible`/`restoration_failed`/`stalled` — the same verdict the L-BFGS
  routes already gave — and DEMBO7/KISSING recover to `optimal`.
- **False infeasibility claims are largely gone.** The per-route `infeasible`
  counts dropped from 110–156 to 55–56: a local-infeasibility verdict now
  requires a *stationarity certificate* from the restoration phase (projected
  gradient ≈ 0, or no descent at the Levenberg–Marquardt ceiling), gets one
  x0-anchored second-chance probe before it is believed, and is vetoed when
  the run's own history already certified near-feasibility. Uncertified stalls
  end as the honest `restoration_failed`/`stalled` labels instead — the two
  new status columns — and dozens of the ex-`infeasible` rows now finish
  `optimal` outright (SNAKE, CATENARY, BT9, HS39, CRESC4, SSEBNLN, ALJAZZAF…).
- **Budget statuses are down sharply** (`max_iter` + `max_time`: 69–182 →
  24–94 per route). Two mechanisms: a run whose returned best iterate already
  satisfies the relaxed (acceptable-level) KKT tolerance now reports
  `acceptable` instead of `max_time`/`max_iter` (the DIAMON2DLS/DMN
  least-squares family — oscillating at KKT ~1e-7 without ever holding it for
  the acceptable window — is now scored *correct*), and the stall detector
  ends frozen-iterate grinds early. As of v20 the same re-judging extends to
  `stalled` exits, and a **terminal KKT certificate** additionally repairs
  drifted multipliers at the returned iterate before giving up on it — a
  genuinely active constraint's certificate fails, so it under-certifies at
  worst (see *Changes in the v20 baseline*).
- **`unbounded` is now a certified verdict**: it requires the objective to
  diverge below `−diverging_iterates_tol`, not just a large iterate norm.
  KOEBHELB — whose iterate wanders past 1e22 and then converges to f = 112 —
  flipped from `unbounded` to `optimal` on the exact routes, while genuinely
  unbounded problems (INDEF) are still detected.
- **The RT-typical `lbfgs/sparse`** route is solid (785 correct, zero
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

!!! note "Interpolated line-search backtracking (`backtrack_interpolation`, opt-in, 2026-08-31)"
    `LineSearchOptions.backtrack_interpolation` (default `False`) replaces the
    plain `α ← α/2` backtrack after a rejected trial with the safeguarded
    minimizer of the quadratic merit model through `φ(0)`, `φ'(0)` and the
    rejected `φ(α)` (Nocedal & Wright 2006, eq. (3.58)), clipped to
    `[0.1α, 0.5α]`. It targets an RT-scale finding: on a bound-only L-BFGS
    problem (n = 50k, `scalar1` seed) halving needs 3.3–4.1 objective
    evaluations per accepted step, each one a full dose-projection `D@x`;
    interpolation lands the trial in 2.0–2.4.

    Three commits landed together on 2026-08-31 (`6b5058a`/`eb46439`/
    `bdaff4e`, this lever being the last): the candidate-vs-baseline delta
    (develop @ `7673ac1`, 2026-08-26, vs `bdaff4e`; full corpus, NumPy,
    `max_iter=10000`, `max_time=300s`, incl. objective-free, `--jobs 8` with
    BLAS threads pinned to 1) is **+18/6600**, but that credits all three
    commits at once. Isolating *this lever alone* needs two more sweeps: the
    `exact/*` configs never touch the L-BFGS code the other two commits
    target, so the same `7673ac1`→`bdaff4e` comparison isolates it there
    directly; on `lbfgs/*`, a same-commit A/B at `bdaff4e`
    (`--backtrack-interpolation off` vs the default) isolates it with the
    other two commits' effects held fixed:

    | correct (Δ from this lever alone) | `exact/dense` | `exact/krylov` | `exact/sparse` | `lbfgs/dense` | `lbfgs/krylov` | `lbfgs/sparse` |
    |---|---|---|---|---|---|---|
    | net | **−1** (+8/−9) | **+5** (+16/−11) | **−5** (+7/−12) | **+9** (+15/−6) | **+4** (+12/−8) | **−1** (+12/−13) |

    `exact/*` nets **−1/3300**, including reproducible worse-basin flips on
    `HS97`/`HS98` (+30%) and `OSBORNEB` (+118%) — same `optimal` status both
    ways, a genuinely different local optimum — on 2 of the 3 exact routes
    (`exact/krylov` is unaffected on all three). `lbfgs/*` nets **+12/3300**
    (39 fixed, 27 broken) with the other two commits' own contribution
    (`6b5058a` + `eb46439` alone, `--backtrack-interpolation off` throughout)
    already at +7/3300 there. So this lever's own total contribution across
    the full corpus is **−1 + 12 = +11/6600** — real, but not the +18 the raw
    candidate delta suggests, and not unanimous.

    Per the same bar as `slack_init_scale` above (and `gamma_alpha`'s −4), a
    non-unanimous corpus result does not flip the default even though the net
    is positive; the L-BFGS-specific win is real and available per-solve via
    the option. `HS25` is the cleanest single-problem case: default `acceptable`
    at `32.83` (documented optimum `0`, measured at this sweep's `10000`/`300s`
    budget) versus `optimal` at `~1.4e-16` with the lever on, reproduced
    identically on all three `lbfgs/*` routes and already solved to `~1.4e-16`
    by every `exact/*` route regardless (see `benchmarks/routing_hints.py`).

## CuPy sparse-direct route (2026-09-28)

The first corpus run of the CuPy backend, made to validate two CuPy-only
changes that had only been checked on HS problems and RT-like runs: the cuDSS
adapter's **residual certificate** for perturbed-pivot factorizations
(`cf66a86`: when cuDSS reports `NPIVOTS > 0`, a solve column with
`‖b − A x‖∞ > 1e-2 ‖b‖∞` is rejected as a numerical failure so the δ_w/δ_c
ladder regularizes, together with a rejection of KKT matrices with inf/NaN
entries and the pivot-checked dense Cholesky `checked_cholesky`), and the dense
`cholesky_solve` through the factor's transpose view (`cfd9d71`).

**Setup.** `feat/trots-round-4` @ `db75a27`; RTX 4070 Laptop (8 GB), CuPy
14.1.1, cuDSS 0.7.1.6 through nvmath-python 0.9.0. The four configs that reach
the changed code (`lbfgs/dense`, `exact/dense`, `lbfgs/sparse`,
`exact/sparse`) over the whole corpus at the gate budget (`--max-iter 10000
--max-time 300 --include-objective-free`), on CuPy and — as the reference, same
commit, same flags — on NumPy (Feral LDLᵀ on the sparse configs). The NumPy arm
reproduces v31 to ±1 per config (780/776/784/798), so it is a sound reference.
The model is still evaluated on the host through the S2MPJ bridge, which the
runner now supports on CuPy (`--backends cupy`); `--cupy-certificates` counted,
per solve, how often each certificate fired and binned every certified relative
residual by decade. `SPECANNE` is excluded on CuPy: one `exact/dense`
iteration (9 variables, 5000 equality rows, a 5009×5009 dense saddle) ran for
over 85 minutes with the card's memory full, and `max_time` is only checked
between iterations. CuPy ran 2–4 problems at a time on one GPU, and
wall-clock times are **not** comparable: on rows with identical iteration
counts the median CuPy/NumPy time ratio is 5.8 (`exact/dense`) to 15.5
(`exact/sparse`). That is the host-bridged evaluation and kernel-launch cost
at CUTEst sizes, and it is why the `max_time` column grows.

| config | rows | correct NumPy → CuPy | fixed / broken | `max_time` NumPy → CuPy | rows with `NPIVOTS > 0` | perturbed-pivot solves | rejected by the certificate (rows) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lbfgs/dense`  | 1097 | 780 → 782 (**+2**)  | 9 / 7   | 5 → 16  | — | — | — |
| `exact/dense`  | 1097 | 775 → 766 (**−9**)  | 2 / 11  | 16 → 55 | — | — | — |
| `lbfgs/sparse` | 1100 | 783 → 762 (**−21**) | 6 / 27  | 6 → 33  | 893 (427k of 540k factorizations) | 695 428 | 99 784 (162) |
| `exact/sparse` | 1100 | 797 → 779 (**−18**) | 10 / 28 | 13 → 44 | 288 (111k of 565k) | 193 061 | 13 203 (72) |

The dense pivot check ran 4.4 M times on `exact/dense`, where 3.76 M of those
calls (755 problems) found the matrix not positive definite. That is the PD
probe doing its job: NumPy's Cholesky raises on the same inputs. On
`lbfgs/dense` it ran 20 k times (the hinted-reuse path) and found 36
breakdowns. The non-finite-matrix rejection fired on two problems
(`LUKVLE8`, 9 480 times, which ends in `numerical_error` on NumPy too, and
`RAT43`, 40 times).

**Where the certified residuals fall.** The residual of a perturbed-pivot solve
does not split into "consistent, ~1e-8" and "inconsistent, O(1)". The
distribution is continuous across the threshold:

| certified relative residual | < 1e-8 | [1e-8, 1e-4) | [1e-4, 1e-3) | [1e-3, 1e-2) | [1e-2, 1e-1) | [1e-1, 1) | [1, 1e3) | ≥ 1e3 | non-finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lbfgs/sparse` | 395 140 | 172 948 | 10 853 | 16 703 | *28 904* | *12 506* | *24 027* | *21 335* | *13 012* |
| `exact/sparse` | 128 402 | 24 777 | 15 132 | 11 547 | *1 139* | *705* | *7 107* | *3 374* | *878* |

(Italic columns are rejected.) About 28 k accepted solves lie within one decade
below `1e-2` and 30 k rejected ones within one decade above it.

**What the 100 correct-flips are.** Each flip was classed by what differed. A
*timing* flip is a CuPy `max_time` stop at fewer iterations than NumPy needed,
so the run was on its way and the wall clock is the difference. On the dense
configs, a *dense* flip is anything else. On the sparse configs,
*certificate fired* means at least one rejection in that solve, *pivots
accepted* means perturbed pivots with no rejection, and *no pivots* means
neither.

| config | timing | dense route | certificate fired | pivots accepted | no pivots |
| --- | --- | --- | --- | --- | --- |
| `lbfgs/dense`  | 0 / 1 | 9 / 6 | — | — | — |
| `exact/dense`  | 0 / 7 | 2 / 4 | — | — | — |
| `lbfgs/sparse` | 0 / 6 | — | 3 / 8  | 3 / 12 | 0 / 1 |
| `exact/sparse` | 0 / 4 | — | 2 / 11 | 5 / 10 | 3 / 3 |

(fixed / broken.) Every *certificate fired* row, plus the two timing rows
where it fired (`SPINLS`, `VANDERM3`), was then re-run alone on the idle GPU
three times: as-is, with the threshold at `1.0`, and with the certificate off
(a no-op). The 17 non-timing *pivots accepted* breaks were re-run as-is and
with a strict `1e-6`, and three `exact/dense` breaks were re-run with the
pivot check off. All re-runs used the same budget. One
confound first: the **CuPy sparse route is not run-to-run deterministic**
(see [Reproducibility on CUDA](../guide/backends.md#reproducibility-on-cuda)).
Identical re-runs moved `ACOPR14` `exact/sparse` from 25 to 38 iterations,
`YORKNET` from 2055 to 120 and `COSHFUN` from `max_time` to `optimal`. The
dense route reproduced the sweep exactly. So the evidence below is one or two
samples per row.

- **The certificate made no difference: 11 rows.** `DEMYMALO`, `GIGOMEZ1`,
  `HS106`, `MADSSCHJ`, `POLAK1`, `POLAK3`, `POLAK6`, `POWERSUMNE`, `SPIRAL`
  (both sparse configs) and `ORTHREGE` fail the same way in all three arms, so
  the failure comes from elsewhere. Most of them accept thousands of solves
  at residuals of 1e-4–1e-2 (see the pivots-accepted group below).
- **Run-to-run noise or timing: 7 rows.** `ARGTRIGLS`, `POWELLSQ`, `TRIGON1`,
  `DJTL` and `VANDERM3` (`lbfgs/sparse`) are correct in every re-run arm.
  `CURLY20` and `CURLY30` flip between arms with no pattern.
- **True positives: 5 rows.** With the certificate off, `WOMFLET`, `YORKNET`,
  `SPINLS`, `COSHFUN` (`exact/sparse`) and `HEART6LS` (`lbfgs/sparse`) stall or
  hit `max_time`, and with it they converge. Off, those runs accept solves
  with residuals from 0.2 (`HEART6LS`) up to 2e14 (`SPINLS`).
- **False positives: 3 rows.** `ACOPR30` and `HS109` (`lbfgs/sparse`) fail as-is
  but converge with the threshold at `1.0` or off; `ROTDISC` (`exact/sparse`)
  converges only with the certificate off, in NumPy's 21 iterations.
- **The `[1e-2, 1)` band cuts both ways.** Accepting it fixed `ACOPR30` and
  `HS109`, and took `COSHFUN` from 791 iterations to 17. Rejecting it was
  needed for `YORKNET`: at `1.0` it runs into `max_time`.
- **Accepted-but-inaccurate solves are the larger effect.** A strict `1e-6`
  turned **8 of the 17** re-run *pivots accepted* breaks correct: `CATENARY`,
  `MAKELA1`, `MAKELA2`, `MIFFLIN2`, `ROSENMMX` (`exact/sparse`, e.g. `MAKELA1`
  stalled at 291 iterations → optimal in 22), `GAUSS1LS`, `KIRBY2LS`,
  `PALMER2E` (`lbfgs/sparse`). It left 8 unchanged and made `SBRYBND` (wrong
  either way) worse. On these degenerate problems — the minimax family among
  them — cuDSS's static pivoting accepts Newton steps with residuals of
  1e-5–1e-2 at every iteration, and the IPM stalls where Feral's LDLᵀ does not.
- **Dense route: not the changed code.** `DJTL`, `ELATTAR` and `POWERSUMNE`
  (`exact/dense`) fail identically with the pivot check off. They are the
  known knife-edge rows, moved by GPU round-off, and `lbfgs/dense` nets +2.

**Verdict on `_PERTURBED_SOLVE_MAX_RESIDUAL = 1e-2`: keep it for now.** The
certificate is needed: turning it off loses 5 rows that converge with it. Moving
the threshold within `[1e-2, 1]` trades 2 rows (`ACOPR30`, `HS109`) against 1
(`YORKNET`), and on a non-deterministic route with one sample per row that is
not a basis for a default change. The CuPy-vs-NumPy gap on the sparse configs
is `−39` over the two, or `−29` without the timing flips. Only 3 of those rows
trace to the threshold's rejections. The larger part comes from solves the
certificate *accepts*: perturbed-pivot solves whose residual is too large for
a good Newton step, yet far below "inconsistent". A single threshold cannot
separate these from the benign ones, and the distribution above shows why.
For scale, IPOPT treats a KKT solve as singular when its residual ratio after
iterative refinement stays above `residual_ratio_singular = 1e-5`. The
principled next step is iterative refinement on perturbed-pivot
factorizations, then certifying the refined residual. The refinement can be
cuDSS's own refinement steps or one correction solve that reuses the residual
the certificate already forms. That needs a core change and its own corpus A/B
on both sparse configs, so it is out of scope for this benchmark-only pass.
(It has since been done; see the addendum below.)

The reports are `benchmarks/reports/s2mpj_v33_cupy.json` /
`s2mpj_v33_numpy.json`, with the per-solve certificate counts in
`s2mpj_v33_cupy_certs_all.jsonl` and the re-runs in
`s2mpj_v33_cupy_counterfactual.jsonl`. To reproduce the CuPy arm, add
`--backends cupy --cupy-certificates --config lbfgs/dense,exact/dense,lbfgs/sparse,exact/sparse`
to the gate command below, and run at most a few problems at a time. A
Windows GPU reset (TDR) during a three-process run killed every CUDA context
on the card, and the affected rows had to be discarded and re-run.

### Addendum (2026-09-29): iterative refinement on perturbed-pivot factorizations

**The change** (`fix/cudss-perturbed-refinement` @ `5c8f30c`). When cuDSS
reports `NPIVOTS > 0`, the adapter now iteratively refines each solve against
the unperturbed matrix, `x ← x + (A+E)⁻¹ (b − A x)`, before the certificate
judges it (Li & Demmel 2003 pair static pivoting with refinement; Higham 2002,
ch. 12). The stopping rules are IPOPT's (`PDFullSpaceSolver::Solve`): at most
10 steps, stop once IPOPT's residual ratio is at most `1e-10`, and stop at the
first step that does not reduce the certified ratio. Each column keeps its best
iterate, and the `1e-2` certificate is unchanged. cuDSS's own refinement
(`CUDSS_CONFIG_IR_N_STEPS`) was measured and not used, because the pinned
cuDSS 0.7 ignores `IR_TOL`. It therefore runs a fixed number of steps, and on
a `ROTDISC` KKT system those steps diverge: the relative residual goes from
1.3e-5 unrefined to 20 after five steps and 8e7 after ten. Factorizations
without perturbed pivots are untouched. A perturbed one pays one cuDSS SOLVE
phase, one SpMV and one host sync per refinement step taken.

Why the unrefined solves were bad: the systems behind the stalls are not ill
conditioned. A captured `MAKELA1` KKT matrix has condition number ≈ 4e2. Its
epigraph variable has a zero Hessian diagonal, which a 2×2 pivot would pair
off. cuDSS's static pivoting perturbs that entry instead, and the solve comes
back with a relative residual of 3e-4. One or two refinement steps bring it
to 1e-10–1e-16.

**Setup.** Same machine and runtime as above. The run covers only the rows
whose v33 CuPy solve met a perturbed pivot: 893 on `lbfgs/sparse` and 288 on
`exact/sparse`. The other rows never reach the changed code, so they keep
their v33 result. The run was serial on one CUDA process, at the gate budget
(`--max-iter 10000 --max-time 300`), with `--cupy-certificates`. The baseline
is the archived v33 CuPy report, not a re-run. Two arms:

- **refined**: refinement on, `1e-2` certificate (the new default), on all
  those rows;
- **strict**: refinement on, certificate at `1e-5` (IPOPT's
  `residual_ratio_singular`), via `--cudss-max-residual 1e-5`. This ran only
  on the 178 rows (136 + 42) where the refined arm accepted a refined solve
  with residual ≥ 1e-5. No other row can change under the tighter threshold.

Every flipped row was then re-run alone once. The first re-run pass of the
`lbfgs/sparse` flips overlapped with heavy CPU load from other jobs (per
iteration it was 3–6× slower, since the model is evaluated on the host), so
the rows it timed out on were re-run again on an idle machine. The v33
baseline itself ran 2–4 CuPy processes on the one GPU. A v33 row that hit
`max_time` after fewer iterations than the refined arm needed to converge is
therefore a *timing* flip: it says nothing about refinement.

| config | rows | correct v33 → refined | fixed / broken (sweep) | confirmed by lone re-run: fixed (of which timing) / broken | `max_time` v33 → refined | NumPy on the same rows |
| --- | --- | --- | --- | --- | --- | --- |
| `lbfgs/sparse` | 893 | 658 → 672 (**+14**) | 18 / 4 | 12 (4) / 2 | 31 → 19 | 678 |
| `exact/sparse` | 288 | 196 → 211 (**+15**) | 16 / 1 | 16 (1) / 1 | 17 → 9 | 212 |

| config | perturbed-pivot solves | refined (≥ 1 step) | mean steps per perturbed solve | hit the 10-step cap | rejected solves (rows), v33 → refined | rescued: rejected unrefined, accepted refined |
| --- | --- | --- | --- | --- | --- | --- |
| `lbfgs/sparse` | 1 220 368 | 300 047 (25 %) | 0.39 | 2 175 | 99 784 (162) → 96 463 (57) | 36 022 |
| `exact/sparse` | 166 771 | 33 104 (20 %) | 0.32 | 35 | 13 203 (72) → 12 634 (55) | 3 383 |

The rows the refinement fixes:

- **The whole minimax family on `exact/sparse`.** `DEMYMALO`, `GIGOMEZ1`,
  `MADSSCHJ`, `MAKELA1`, `MAKELA2`, `MIFFLIN2`, `POLAK1`, `POLAK3`, `POLAK6`,
  `ROSENMMX`, `ROTDISC`, `SPIRAL`, `SPINLS`, `VANDERM4` and `DRCAVTY2` now
  converge, most within a few iterations of NumPy. Before, `MAKELA1` stalled
  at 291 iterations and now converges in 20; `POLAK1` hit `max_time` at 2 925
  and now converges in 12. This covers four of the five `exact/sparse` rows
  the earlier strict-`1e-6` re-run fixed. The fifth, `CATENARY`, has a
  residual that refinement cannot reduce (see the strict arm below). It also
  fixes `ROTDISC`, a false positive that before converged only with the
  certificate off. And it fixes most of the rows listed above under "the
  certificate made no difference"; the exceptions are `HS106`, `POWERSUMNE`
  and `ORTHREGE`.
- **Eight refinement fixes on `lbfgs/sparse`**, confirmed alone: `ACOPR30` (a
  false positive of the certificate), `ARGTRIGLS`, `DMN37142LS`, `KIRBY2LS`,
  `SPIRAL`, `TAX13322`, `TRIGON1` and `VESUVIOLS`. The other four confirmed
  fixes are timing flips (`AIRPORT`, `DIAMON2DLS`, `POWELLSQ`, `VANDERM3`).
  Six more did not reproduce alone: `DECONVB`, `DMN37143LS`, `HS109`, `HS116`,
  `ORTHREGD` and `PALMER1E`.

The breaks: `ACOPP30` (`lbfgs/sparse`) now needs more than the 5 696
iterations the budget allows, where v33 converged in 3 243. `DJTL` fails in
both re-runs but fails on NumPy too; it is a known knife-edge row. `WOMFLET`
(`exact/sparse`) now converges in 15 iterations to the same local optimum NumPy
finds in 14, which is not the documented one. `MSS1` and `VANDERM1` did not
reproduce alone.

**Full-corpus effect.** The rows without perturbed pivots are unchanged, so
the gate totals move from `lbfgs/sparse` 762 → 776 and `exact/sparse`
779 → 794, against NumPy's 783 and 797. Counting only the confirmed flips
(net +10 and +15), the CuPy-vs-NumPy gap on these two configs goes from
`−39` to `−14`. What is left of it is mostly `max_time` rows, where CuPy's
per-iteration cost at CUTEst sizes decides the outcome.

**The strict arm (`1e-5`) does not help.** Among the refined solves, 95 299
on `lbfgs/sparse` and 2 705 on `exact/sparse` are still accepted with
residuals in `[1e-5, 1e-2)`. Refinement cannot reduce them; `CATENARY` is the
standard case, stagnating at 1.4e-5. Rejecting them more than doubles the
rejected solves on the rows the arm ran: 90 112 → 198 754 on the 136
`lbfgs/sparse` rows and 2 268 → 7 693 on the 42 `exact/sparse` rows. Every
rejection triggers another regularized refactorization. The `lbfgs/sparse`
solve time on those rows went from 124 to 185 minutes. Some of that may be
host load, since five of the six breaks ran 2–3× faster alone.

| config | rows | correct refined → strict | fixed / broken (sweep) | confirmed alone: fixed / broken |
| --- | --- | --- | --- | --- |
| `lbfgs/sparse` | 136 | 73 → 69 (−4) | 2 / 6 | 1 (`ORTHRGDM`) / 1 (`ORTHREGD`, whose refined-arm fix was itself unconfirmed) |
| `exact/sparse` | 42 | 21 → 22 (+1) | 2 / 1 | 1 (`CATENARY`) / 1 (`YORKNET`) |

Five of the six `lbfgs/sparse` breaks did not reproduce alone. Once the
re-runs are counted the strict arm is a wash: +2 / −2. It adds rejections,
and so δ-ladder refactorizations, on every row it touches.

**Verdict: ship the refinement as the default, and keep the `1e-2`
certificate.** Refinement removes the accepted-but-inaccurate class the v33
run identified. It fixes 16 `exact/sparse` rows for one break that lands on
NumPy's own optimum, and 12 `lbfgs/sparse` rows for 2 breaks. It costs 0.3–0.4
extra solves per perturbed-pivot solve and nothing on unperturbed ones. The
threshold stays where the v33 analysis put it: IPOPT's `1e-5` is not better
once refinement has run. The reports are
`benchmarks/reports/s2mpj_v34_cupy_{refine,strict}_{lbfgs,exact}.json`, with
the lone re-runs in `s2mpj_v34_cupy_{rerun,rerun2}_refine_*` and
`s2mpj_v34_cupy_rerun_strict_*`. The certificate counts, now binned both
before and after refinement, are in the matching `.certs.*.jsonl` files. To
reproduce an arm, add `--backends cupy --cupy-certificates --config
lbfgs/sparse` (or `exact/sparse`), plus `--cudss-max-residual 1e-5` for the
strict arm or `--cudss-refinement-steps 0` for the unrefined one, to the gate
command below. The report records the policy under `environment.cudss_policy`,
and `--resume` refuses to mix two policies.

### cuDSS matching and pivot options (2026-09-30)

**Question.** Refinement repairs perturbed-pivot solves after the fact. Can a
cuDSS configuration avoid the perturbed pivots in the first place, for
example the zero diagonal of the minimax epigraph variable that a 2×2 pivot
would pair off? The candidates are the pivot-structure parameters in
`cudssConfigParam_t`. Maximum weighted matching with scaling (Duff & Koster
2001) permutes large entries onto the diagonal, and static pivoting then has
fewer tiny pivots to perturb. A variant would qualify for a corpus A/B only if
it removed the perturbations at no more than about 1.3× the default
factorization time, both on these small systems and on a TROTS-scale
system.

**Setup.** Same machine and runtime as above (nvmath-python 0.9.0, cuDSS
0.7.1.6), `exp/cudss-matching` @ `321b16a`, no library change. Scratch
instrumentation captured real KKT systems (full CSR plus the first right-hand
side of each factorization) from CuPy solves:

- `exact/sparse`: `MAKELA1`, `POLAK1`, `MIFFLIN2` and `GIGOMEZ1` (minimax,
  n = 5–6), `CATENARY` (n = 19) and `ROTDISC` (n = 1986);
- `lbfgs/sparse`: `GAUSS1LS`, `KIRBY2LS`, `PALMER3E` (compact-border systems,
  n = 13–28) and `ACOPR30` (n = 264).

Every one of their factorizations, 30 in all (first, middle and last perturbed
factorization of each run), had `NPIVOTS > 0` under the default configuration.
Each system was factored under each configuration through the adapter's own
`CuDSSSparseSolver`, with the parameter set by `config_set` before the
analysis. The run recorded `NPIVOTS`, the unrefined relative residual, and
the refined residual and step count under the adapter's policy. It also read
the inertia back and compared it with a dense Bunch–Kaufman LDLᵀ of the same
matrix. Phase times were measured on the cuDSS `ANALYSIS` / `FACTORIZATION`
executes alone: warmed, device-synchronized, arms interleaved, median of 7.

**Which parameters exist and work.** The binding exposes `USE_MATCHING`,
`MATCHING_ALG` (there is no `MATCHING_TYPE`), `PIVOT_TYPE`,
`PIVOT_THRESHOLD`, `PIVOT_EPSILON`, `PIVOT_EPSILON_ALG`, `REORDERING_ALG`,
and also `DETERMINISTIC_MODE` (see below). The defaults read back as
`PIVOT_TYPE = PIVOT_COL`, `PIVOT_THRESHOLD = 1.0`, `PIVOT_EPSILON = -1`
(automatic), and matching off. `execute` returns `NOT_SUPPORTED` for
`PIVOT_EPSILON_ALG` ≥ 2, `REORDERING_ALG` 4 and 5, and `REORDERING_ALG` 1 or 2
combined with matching. Every other value was accepted on both the symmetric
(`LOWER` view) and the general route. On the `MAKELA1` probe, `MATCHING_ALG=5`
gave results identical to matching's default, and `MATCHING_ALG=1` left the
perturbed pivot in place. On 24 of
the 30 systems, `PIVOT_EPSILON = 1e-13` reproduced the default's `NPIVOTS`
and residual to the bit.

| variant | runs | factorizations with `NPIVOTS > 0` | median / max unrefined residual | max refined residual | refinement steps (sum) | inertia = Bunch–Kaufman | factorization time vs default | analysis time vs default |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | 30 | 30 | 2.1e-05 / 1.6e+01 | 5.3e-02 | 33 | 25 / 30 | 1.00× | 1.00× |
| `USE_MATCHING=1` | 27 (3 failed) | 3 | 4.8e-16 / 2.9e-10 | 2.9e-10 | 0 | **0 / 27** | 1.35× | 1.01× |
| `USE_MATCHING=1, MATCHING_ALG=4` | 30 | 10 | 3.1e-13 / 3.0e+03 | 3.2e-02 | 12 | 0 / 30 | 1.04× | 1.01× |
| `REORDERING_ALG=1` (`=2` identical) | 30 | 5 | 3.5e-14 / 1.4e-05 | 1.4e-05 | 0 | 0 / 30 | 1.68× | 0.14× |
| `REORDERING_ALG=3` | 30 | 24 | 3.7e-07 / 4.7e-02 | 4.7e-02 | 24 | 25 / 30 | 0.98× | 1.01× |
| `REORDERING_ALG=3` + matching | 27 (3 failed) | 9 | 7.4e-16 / 1.2e-08 | 1.1e-08 | 1 | 0 / 27 | 1.41× | 1.01× |
| `PIVOT_TYPE=PIVOT_NONE` | 30 | 28 | 9.6e-05 / 5.2e+01 | 4.1e-02 | 39 | 25 / 30 | 0.86× | 1.00× |
| `PIVOT_EPSILON=1e-6` | 30 | 30 | 4.8e-07 / 1.3e-03 | 4.1e-04 | 141 | 26 / 30 | 1.01× | 1.00× |
| `PIVOT_EPSILON=1e-8` | 30 | 30 | 2.4e-08 / 3.0e-02 | 2.6e-02 | 70 | 25 / 30 | 0.99× | 1.00× |
| `PIVOT_EPSILON=1e-10` | 30 | 30 | 4.5e-07 / 5.0e-02 | 3.3e-02 | 21 | 26 / 30 | 0.99× | 1.00× |

(`PIVOT_TYPE=PIVOT_ROW` and `PIVOT_THRESHOLD` ∈ {0, 0.01, 0.1} reproduced the
default exactly on the `MAKELA1` probe and are omitted. The three failed runs
are `CATENARY`'s: under matching, the solve of that structurally singular
system comes back non-finite, where the default factorization gives a finite
residual of 1.4e-5. The small-system times are launch-bound: 0.03–0.3 ms per
factorization.)

What the table shows:

- **Matching removes the perturbed pivots, but not with a symmetric
  factorization.** With `USE_MATCHING=1`, 24 of the 27 factorizations had no
  perturbed pivot, and every minimax and `ROTDISC` solve was accurate to
  about 1e-12 or better with no refinement. `ACOPR30`'s worst system, which
  refinement cannot repair (5.3e-2 after refinement, a rejected solve), came
  back at 2.9e-10. But cuDSS then returns `INERTIA = [0, 0]` on every
  system, and the signs of `DataParam.DIAG` no longer match the reference
  (`MAKELA1`: 1 positive and 4 negative, against 3 and 2). `LU_NNZ` grows
  1.3–3.4× (`ROTDISC`: 15 114 → 51 676), and `PERM_MATCHING` and `SCALE_ROW`
  become readable, where they return `NOT_SUPPORTED` without matching. The
  matched matrix is evidently factored as a nonsymmetric system, and no
  inertia comes out of it. `REORDERING_ALG` 1 and 2 behave the same way (no
  inertia) and cost 1.7× in factorization on these systems. The IPM's
  inertia correction on the sparse route (`_inertia_acceptable`, Wächter &
  Biegler 2006 §3.1) would read `(0, 0, n)` as a wrong inertia and reject
  every factorization, so neither option can be a drop-in policy. It would
  have to report no inertia and give up that correction. The
  symmetry-preserving alternative, a symmetric matching that selects 2×2
  pivots (Duff & Pralet 2005, as in MA57 and MUMPS), is not among cuDSS
  0.7's options.
- **The other parameters do not remove the perturbation.** `PIVOT_TYPE` and
  `PIVOT_THRESHOLD` only choose among pivots that cuDSS can reach without a
  2×2 block, and they leave the zero diagonal where it is. `REORDERING_ALG=3`
  keeps the symmetric factorization and its inertia. It removes the
  perturbation on only 6 of the 30 systems (the three `GIGOMEZ1` systems, two
  of `ACOPR30`'s and one of `KIRBY2LS`'s), and it raises `ROTDISC`'s
  perturbed pivots (114/158/272 → 179/179/349). A larger `PIVOT_EPSILON` makes the minimax solves more
  accurate before refinement (1e-8: median 1.1e-4 → 1.1e-8 on the
  `exact/sparse` systems). On the L-BFGS systems it makes them worse: it
  perturbs more pivots (39 → 93 in total), the median unrefined residual goes
  from 5.4e-10 to 1.3e-6, and the refinement steps go from 4 to 38. It
  changes where the error is, not how many pivots are perturbed.
- **Refinement already closes most of the gap.** Refinement took the default
  configuration's minimax and `ROTDISC` solves to 2e-9–3e-15 in 0–4 steps.
  Of the 30 systems, only five stay above 1e-8: two of `ACOPR30`'s (1.7e-7,
  and 5.3e-2 for the worst one) and the three `CATENARY` systems
  (4.8e-7–1.4e-5, stagnating in every configuration).
- **A perturbed factorization's inertia can be off.** The default configuration
  matched the reference inertia on 25 of 30 systems. It did not on
  `CATENARY`, where 7 zero pivots were reported positive; on `ACOPR30`'s worst
  system, where one pivot flipped (81/183 against 82/182); or on one
  `KIRBY2LS` system (13/12 against 15/10). Matching gives no inertia at all,
  so it cannot fix this.

**TROTS scale.** The sparse-route KKT systems of two TROTS cases were captured
the same way (exact Hessian, first factorization). `Prostate_BT_01`
(n = 6 393, nnz = 0.71 M) has no perturbed pivot. `Protons_01`
(n = 370 546, nnz = 41.5 M) perturbs one pivot. Medians of 5 and 3
interleaved samples:

| system | variant | `NPIVOTS` | unrefined → refined residual | inertia | `LU_NNZ` | factorization | analysis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Protons_01` | default | 1 | 4.1e-07 → 1.8e-13 (2 steps) | (1101, 369445, 0) | 21.5 M | 0.102 s | 4.77 s |
| `Protons_01` | `USE_MATCHING=1` | 0 | 5.0e-13 | none (`[0, 0]`) | 85.4 M | 0.362 s (**3.5×**) | 9.63 s (2.0×) |
| `Protons_01` | `REORDERING_ALG=3` | 2 | 9.7e-08 → 1.5e-13 (2 steps) | (1101, 369445, 0) | 22.0 M | 0.126 s (1.23×) | 2.10 s (0.44×) |
| `Prostate_BT_01` | default | 0 | 9.2e-15 | (55, 6338, 0) | 0.36 M | 1.36 ms | 42 ms |
| `Prostate_BT_01` | `USE_MATCHING=1` | 0 | 2.6e-15 | none | 1.36 M | 2.64 ms (1.95×) | 109 ms (2.6×) |
| `Prostate_BT_01` | `REORDERING_ALG=1` | 0 | 1.7e-15 | none | 0.88 M | 31.3 ms (23×) | 11 ms (0.25×) |

At scale, matching costs 3.5× per factorization, twice the analysis time, and
four times the factor memory, in exchange for a perturbation that two
refinement steps already repair to 1e-13.

**The solves the certificate rejects.** The one case where matching does
something refinement cannot is a solve that the certificate still rejects
after refinement. From seven v34 rows with rejections (`BROWNBS`, `INDEF`,
`HS109` on `lbfgs/sparse`; `JUNKTURN`, `LINSPANH`, `CSFI2`, `ANTWERP` on
`exact/sparse`), the first five rejected solves each were captured, 35 in
all. Every one of those matrices is numerically singular (condition numbers
1e16–1e30). Matching brought 20 of them to ≤ 1e-2 (all of `ANTWERP`,
`BROWNBS`, `CSFI2` and `HS109`), left `INDEF` and `LINSPANH` rejected, and
made `JUNKTURN` worse (≈ 1 → 4e1–7e3). `REORDERING_ALG=1` reached 30 of 35, and
`PIVOT_EPSILON=1e-8`, which keeps the symmetric factorization, reached 26.
On a numerically singular KKT matrix, though, the certificate's rejection
followed by δ_w/δ_c regularization is the response the method prescribes.
An "exact" solve of such a matrix is not evidently a better step, and the
v33 threshold study found flips in both directions in this band.

**Verdict: no configuration change; the corpus A/B was not run.** The only
parameters that remove the perturbed pivots (`USE_MATCHING`,
`REORDERING_ALG` 1/2) do it by giving up the symmetric indefinite
factorization. They lose the inertia the sparse route's correction depends
on. Every matching variant loses it, including `MATCHING_ALG=4`, which costs
only 1.04× but also leaves 10 of the 30 factorizations perturbed. The
default matching and `REORDERING_ALG` 1/2 also cost 1.35–1.7× on CUTEst-size
systems and 2–23× at TROTS scale, which misses the ≤ 1.3× bar. A policy built
on them would have to either run without inertia or pay for a second,
matched factorization on top of the one that supplies the inertia. Neither
is justified while refinement already brings most perturbed solves to the
1e-10 target, at 0.3–0.4 refinement steps per perturbed solve on average
(v34). The remainder is mostly the numerically singular solves discussed
above. `PIVOT_EPSILON` and `REORDERING_ALG=3` are free and inertia-preserving,
but they do not remove the perturbation. The one open lead is
`PIVOT_EPSILON` near 1e-8. It rescues 26 of the 35 rejected singular solves
and sharpens the minimax solves before refinement. After refinement,
though, it is worse: 11 of the 30 systems stay above 1e-8 instead of 5
(`ROTDISC` 3.6e-10 → 9.0e-7, `ACOPR30` 1.7e-7 → 6.7e-3, several L-BFGS
border systems). It would need its own corpus A/B on both sparse configs,
with that regression as the stated risk.

**Side finding: `DETERMINISTIC_MODE`.** The default cuDSS factorization is not
bitwise reproducible. Six fresh solvers on one captured system gave six
different solutions: `ROTDISC` differs by up to 1.4e2 in absolute value on a
perturbed system, and `ACOPR30` by 5.7e-4. That is the run-to-run
non-determinism of the CuPy sparse route documented above. With
`DETERMINISTIC_MODE = 1` all six were bitwise identical, with the same
inertia and `LU_NNZ`. The cost was 1.15–2× per factorization on the small
systems, 2.0× on `Protons_01` (0.102 → 0.208 s) and 3.0× on
`Prostate_BT_01`, with analysis unchanged. It is not enabled: whether a
reproducible sparse route is worth that factorization cost is a separate
decision.

The scratch scripts, captured CUTEst systems and result JSONs are in
`benchmarks/reports/cudss_matching_2026-09-30/` (local, not committed; the
TROTS captures, about 0.5 GB for `Protons_01`, were not kept there).

### Perturbed-pivot inertia and deterministic mode (2026-10-01)

**Two changes, one A/B.** First, `SparseOptions(deterministic=True)` now
switches cuDSS to `DETERMINISTIC_MODE` (opt-in; see
[Reproducible sparse factorizations](../guide/options.md#reproducible-sparse-factorizations)).
Second, the question this section answers: should the adapter keep reporting
cuDSS's inertia for a factorization with perturbed pivots? The matching study
above found that this inertia is wrong on 5 of the 30 captured systems. The
IPM's inertia check (`_inertia_acceptable`, Wächter & Biegler 2006 §3.1)
compares the reported inertia with the KKT target and escalates δ_w on a
mismatch. A wrong count therefore either hides a singular factor or triggers
a needless escalation.

**What cuDSS counts.** cuDSS's `DIAG` shows every perturbed pivot as exactly
`±1e-13` (the automatic `PIVOT_EPSILON`). The signs of `DIAG` match `INERTIA`
on all 30 systems. So cuDSS counts a perturbed pivot by the sign of the `±ε`
it put in, and the count is the inertia of `A + E`. Four candidate rules
were scored against a dense Bunch–Kaufman `LDLᵀ` (`scipy.linalg.ldl`, exact
zero test, as in the matching study):

1. *trust*: report cuDSS's count whatever `NPIVOTS` is (the current rule);
2. *withhold*: report `None` (skip the check) whenever `NPIVOTS > 0`, leaving
   the refined, certified solve as the only guard;
3. *zero*: count the `NPIVOTS` perturbed pivots as zero pivots;
4. *certified only*: report the count only if the refined solve passed the
   certificate.

| rule | reading = Bunch–Kaufman | wrong | withheld |
| --- | --- | --- | --- |
| 1 trust | 25 | 5 (`CATENARY` ×3: 7 zero pivots counted positive; `ACOPR30` #399: one sign flipped; `KIRBY2LS` #96: 13/12 against 15/10) | 0 |
| 2 withhold | 0 | 0 | 30 |
| 3 zero | 0 | 30 (Bunch–Kaufman finds no zero pivot on 27 systems; on `CATENARY` it finds 7, not 9–10) | 0 |
| 4 certified only | 25 | 4 | 1 (`ACOPR30` #399, rejected by the certificate) |

Rule 4 is a no-op inside the IPM. The driver solves before it checks the
inertia, and a solve the certificate rejects raises `LinearSolveError` before
the check is reached, so every inertia the driver reads already comes from a
certified solve. Rule 3 is wrong on every system: the minimax and `ROTDISC`
perturbations land on pivots that a 2×2 pivot would pair off, not on zero
eigenvalues. Neither went to the corpus.

The captured systems do not show what the IPM does with the reading. So each
inertia check was also recorded during one deterministic solve of each of the
ten captured problems, with the check's target and a Bunch–Kaufman inertia of
the factored matrix (at most 600 checks per problem). The reference verdict
is "accept iff Bunch–Kaufman equals the target":

| rule | perturbed checks: right accept / right reject / wrong accept / wrong reject |
| --- | --- |
| 1 trust | 1 436 / 0 / 25 / **618** |
| 2 withhold | 2 054 / 0 / 25 / 0 |
| 3 zero | 0 / 25 / 0 / **2 054** |

(334 unperturbed checks: 303 right accepts and 31 wrong rejects in every
rule. The 31 are `ACOPR30` factors with `NPIVOTS = 0` whose cuDSS inertia,
81/183, disagrees with Bunch–Kaufman's 82/182.) The 618 needless
escalations under *trust* are almost all on the L-BFGS problems (`KIRBY2LS`
545, `GAUSS1LS` 38, `PALMER3E` 19, `ACOPR30` 16). On these ten problems
Bunch–Kaufman equals the target at every check except `CATENARY`'s, where
the matrix is singular and every rule except *zero* accepts it. *Withhold*
was the one candidate worth a corpus run.

**Setup.** `fix/cudss-inertia-and-determinism` @ `58578ec`, frozen as a source
snapshot for the run. Same machine and runtime as above. The rows are v33's
perturbed-pivot rows (288 `exact/sparse`, 893 `lbfgs/sparse`), run serially at
the gate budget (`--max-iter 10000 --max-time 300 --include-objective-free`)
with `--cupy-certificates`. Every arm ran with `--cudss-deterministic`. A
scratch harness switched the rule by patching `inertia_or_none` and counted
each inertia check. The *withhold* arm ran in full on `exact/sparse`. On
`lbfgs/sparse` it ran only on the 103 rows where *trust* rejected at least one
perturbed factor on inertia, and the other 790 rows were carried over from
*trust*. On `exact/sparse`, the 185 rows without such a rejection never
changed correctness between the arms. Every flipped row was then re-run alone
in both arms.

| config | rows | correct trust → withhold | fixed / broken (sweep) | confirmed by lone re-run: fixed / broken | `max_time` | perturbed factors rejected on inertia | certificate-rejected solves | iterations, rows correct in both |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `exact/sparse` | 288 | 209 → 200 (**−9**) | 0 / 9 | 0 / 9 | 8 → 9 | 50 927 → 0 | 14 087 → 22 040 | 59 697 → 67 598 (194 rows) |
| `lbfgs/sparse` | 893 | 664 → 665 (+1) | 7 / 6 | 5 / 6 | 20 → 18 | 21 325 → 0 | 131 547 → 129 362 | 209 802 → 214 749 (656 rows) |

(Inertia checks under *trust*: 410 206 on `exact/sparse`, of which 34 % were
on perturbed factors; 644 452 on `lbfgs/sparse`, of which 82 % were.)

- **`exact/sparse`: every break is a nonconvex problem that needed the
  correction.** The breaks are `BIGGS6`, `CATMIX`, `DEMBO7`, `MODBEALE`,
  `RK23`, `YATP1CLS`, `YATP1LS`, `YATP2CLS` and `YATP2LS`. Under *trust* they
  converge in 10–310 iterations. Under *withhold* they stall, reach
  `max_iter`, reach `max_time`, or converge to another point. On these rows
  the perturbed factors that *trust* rejected were genuinely indefinite. For
  example, `YATP1LS` accepts 2 597 factors whose cuDSS inertia missed the
  target and runs to 10 001 iterations. All nine breaks reproduced alone,
  bitwise in iterations except the three `max_time` rows.
- **`lbfgs/sparse` is a wash.** *Withhold* fixes `KIRBY2LS` (stalled at 81
  → optimal in 104, the needless-escalation storm in the table above),
  `HS109`, `GAUSS2LS`, `VESUVIALS` and `PALMER4C`. It breaks `BROYDNBDLS`,
  `BRYBND`, `CURLY30`, `PALMER2E`, `PALMER3C` and `PALMER3E`. Two sweep
  fixes, `GASOIL` and `TAX13322`, did not reproduce alone (wall-clock and
  run-to-run noise).

**Verdict: keep today's rule.** *Withhold* is net −10 on the confirmed flips
(−9 / −1). It also adds 13 % iterations on the `exact/sparse` rows that both
arms solve. A perturbed factor's count is wrong often enough to cause
needless escalations on L-BFGS border systems. Where the factor really is
indefinite, though, the check is what saves the exact-Hessian route, and
cuDSS gives no way to tell the two cases apart. The adapter's
`inertia_or_none` docstring records this. The `CATENARY` blind spot (a
singular KKT system whose count matches the target) stays, and a GPU test
pins that reading, so a runtime that changes it is noticed.

**Deterministic mode: cost and what it does not cover.** Measured alone, both
modes interleaved, three repeats each (wall per factorization call, i.e. the
`factor` method including the host-side value update, device-synchronized):

| problem | iterations default / deterministic | distinct objectives in 3 runs | factorization call (median) | wall per iteration (median) |
| --- | --- | --- | --- | --- |
| `ROTDISC` `exact/sparse` | 22, 21, 24 / 22, 22, 22 | 3 / 1 | 16.7 → 17.1 ms (+2 %) | 50.7 → 53.9 ms |
| `ACOPR14` `exact/sparse` | 27, 36, 22 / 27, 27, 27 | 3 / 1 | 7.29 → 7.49 ms (+3 %) | 81 → 96 ms |

On these CUTEst-size systems a factorization call is launch-bound, so the
mode's 1.2–3× on the bare cuDSS phase becomes 2–3 % per call. Wall per
iteration also depends on which trajectory a run takes. The deterministic
`ACOPR14` trajectory needs 6.4 factorizations per iteration, against 4.3–5.8
on the default runs. With the mode on, the 22 flipped rows of the A/B,
re-run alone in both arms, reproduced their sweep iteration counts exactly.
The exceptions were the rows that stopped on `max_time` and one long run,
`GASOIL` (`lbfgs/sparse`). The mode does
not make the whole CuPy route reproducible, though. Two other CuPy kernels
accumulate with atomics, so their last bits vary from run to run (30
different results in 30 repeats each): the transposed SpMV `A.T @ y` behind
`SparseOperator.rmatvec`, and the scatter-add (`cupy.add.at`) behind the
canonical COO→CSR sum. Three processes running `AGG` (`exact/sparse`) with
the mode on diverged at the first right-hand side, before any factorization
differed. A deterministic transpose product (a cached CSR transpose) and a
deterministic duplicate sum would close that gap. Each changes the default
route's performance and would need its own measurement, so neither is part
of this change. (The next section closes the gap.)

A side observation: the `CATENARY` solve that the certificate accepts is
inconsistent. Its right-hand side is −100 on rows that are identically zero.
Measured against `‖b‖∞ ≈ 7e6`, though, that residual is 1.4e-5. A residual
relative to the largest entry of `b` cannot see an inconsistency in the
small entries.

The reports are `benchmarks/reports/s2mpj_v35_cupy_det_{trust,withhold}_{exact,lbfgs}.json`,
with the lone re-runs in `s2mpj_v35_cupy_det_rerun_*`, the per-case
inertia-check counts in the matching `.inertia.jsonl` files and the
certificate counts in `.certs.*.jsonl`. The capture study, the check study,
the A/B harness and the determinism probes are in the session scratchpad
(`sweep_cupy_inertia/`, local, not committed). To reproduce the *trust* arm,
add `--backends cupy --cupy-certificates --cudss-deterministic --config
exact/sparse` (or `lbfgs/sparse`) to the gate command below. The *withhold*
arm needs the scratch harness.

### Reproducible CuPy sparse route (2026-10-02)

**The gap the previous section left.** With `deterministic=True`, three
processes running `AGG` (`exact/sparse`) diverged at the very first
`rmatvec`: event 1 of the trace, the 452 × 163 Jacobian-transpose product of
the first right-hand side, before any factorization. Three CuPy kernels
outside cuDSS accumulate with atomics:

- the transposed SpMV/SpMM behind `rmatvec`/`rmatmat`/`gram_diagonal`;
- the forward CSR SpMV and SpMM of a matrix with long rows. On a 3000 × 20000
  matrix with 15 000-entry rows, the default SpMV and every SpMM algorithm
  (default, `CSR_ALG1/2/3`) gave 28–30 distinct results in 30 repeats;
- `cupy.add.at` in the COO → CSR duplicate sum, once a slot has three or more
  entries (30 distinct results in 30 repeats at about 15 entries per slot).

`CUSPARSE_SPMV_CSR_ALG2`, which the cuSPARSE documentation lists as bitwise
deterministic for the non-transposed product, gave 1 distinct result in every
case. The CuPy adapter now uses it for every vector product. A transposed
product runs forward on an explicit `Aᵀ`, cached per operator value array.
A problem's Jacobian is a new operator every iteration, so it pays one
`csr2csc` per iteration. The facade's operators on one pattern share a
transpose plan instead, so for them `Aᵀ` is a gather. A product with several
columns runs in an ordered kernel: one thread per output entry for rows of up
to 1024 entries, one warp per entry beyond that. The COO build sums each slot
in emission order. That is the SciPy adapter's `np.add.at` order, so the two
builds agree bit for bit. These kernels are on in every mode, for reasons
the [backends guide](../guide/backends.md#reproducibility-on-cuda) gives.

**Bisection on `AGG`.** Each pre-change kernel was patched back in a scratch
harness, with three processes per arm and `deterministic=True`. A trace of
hashes covered every COO build, product, factorization and solve:

| pre-change kernels restored | first divergence | result |
| --- | --- | --- |
| transposed product, forward product, COO sum | event 1 (first `rmatvec`) | 49, 48, 24 iterations |
| forward product, COO sum | none (2647 events) | 46 iterations ×3, identical |
| COO sum | none | identical |
| none | none | identical |

On `AGG` the transposed product was the only source. Its Jacobians have
short rows, and no slot sums three entries. The other two fixes are pinned
by the GPU tests (`tests/unit/test_cupy_sparse_determinism_gpu.py`) on
matrices that do trigger them.

**Corpus check.** The final code solved each of the five problems behind the
earlier flips three times in separate processes (`exact/sparse`,
`deterministic=True`). `YORKNET` and `COSHFUN` were capped at 150 iterations,
because a `max_time` stop depends on the wall clock. All five traces were
bitwise identical across the three processes:

| problem | result (×3) | traced events |
| --- | --- | --- |
| `AGG` | restoration failed, 46 it. | 2 647 |
| `ROTDISC` | optimal, 23 it. | 292 |
| `ACOPR14` | optimal, 50 it. | 1 260 |
| `YORKNET` | 150-iteration cap | 7 611 |
| `COSHFUN` | 150-iteration cap | 2 696 |

The cost was measured over whole solves: wall time per iteration and per
cuDSS factorization call. The table gives the median of 3 interleaved runs,
with the range in brackets, each run alone on the GPU. The arms are:

- *old*: the pre-change kernels patched back, default options;
- *default*: the new kernels, default options;
- *deterministic*: the new kernels, `deterministic=True`.

The CPU was loaded by another job throughout, so treat these as
provisional. Wall time per iteration also depends on the trajectory's
factorizations per iteration, which differ between runs that are not
reproducible.

| problem | iterations old / default / deterministic | ms per iteration old / default / deterministic | ms per factorization old / default / deterministic |
| --- | --- | --- | --- |
| `AGG` | 25, 47, 23 / 25, 25, 25 / 46 ×3 | 521 [368–791] / 740 [716–789] / 327 [306–384] | 387 / 459 / 279 |
| `ROTDISC` | 21 ×3 / 21, 22, 21 / 23 ×3 | 77 [71–112] / 104 [74–127] / 88 [84–157] | 81 / 109 / 72 |
| `ACOPR14` | 34, 26, 42 / 42, 31, 34 / 50 ×3 | 173 [96–174] / 166 [85–178] / 145 [90–165] | 32.4 / 31.3 / 35.0 |
| `YORKNET` | 150 ×3 (cap) | 199 [196–250] / 177 [166–198] / 291 [286–396] | 18.9 / 17.5 / 18.9 |
| `COSHFUN` | 150 ×3 (cap) | 102 [96–129] / 121 [88–149] / 94 [91–113] | 25.1 / 24.0 / 23.6 |

Neither the new kernels nor the mode shows a cost above this noise. Two
rows needed a closer look:

- **`YORKNET` per iteration.** Its deterministic trajectory makes 15
  factorizations per iteration, against 10 on the default runs, at the same
  cost per factorization. That is a property of the trajectory, not of the
  mode.
- **`AGG`, default arm.** It repeated one trajectory in all three runs (25
  iterations, restoration-heavy, the same final `x`). A default-mode run
  can repeat by chance; only the deterministic arm guarantees it.

**Micro-benchmarks.** These are the product kernels themselves, warmed,
device-synchronized, with interleaved arms. Each entry is the median of 7,
in ms.

| operation | S2MPJ-size KKT (n = 1 000, nnz 1.2e4) | mid (n = 5.5e4, nnz 6.5e5) | `Protons_01` KKT (n = 3.7e5, nnz 4.1e7) |
| --- | --- | --- | --- |
| `A @ x`: default → `ALG2` | 0.135 → 0.129 | 0.115 → 0.114 | 2.08 → 2.09 |
| `Aᵀ @ y`: transposed SpMV → cached `Aᵀ` | 0.132 → 0.114 | 0.125 → 0.126 | 3.41 → 2.12 |
| build `Aᵀ` per operator value array: `csr2csc` / plan gather | 0.48 / 0.14 | 0.27 / 0.067 | 22.7 / 5.2 |
| build a pattern's transpose plan (once per pattern and adapter) | 1.2 | 0.5 | 48 |
| `A @ X`, 8 columns: SpMM → ordered kernel | 0.162 → 0.082 | 0.447 → 0.328 | 14.4 → 17.3 (warp) |
| `Aᵀ @ Y`, 8 columns: SpMM → ordered kernel | 0.164 → 0.081 | 0.398 → 0.333 | 15.8 → 17.5 (warp) |
| COO values → CSR data: `add.at` → ordered sum | 0.117 → 0.055 | 0.066 → 0.047 | 39.4 → 19.3 |
| compile the COO map (once per pattern and adapter) | 3.1 → 3.9 | 1.5 → 2.0 | 158 → 241 |
| memory: cached `Aᵀ` / COO map old → new (MiB) | +0.1 / 0.2 → 0.1 | +7.7 / 8.5 → 8.1 | +476 / 482 → 479 |

The two ordered kernels were tuned on the same card. The thread kernel was
at or below SpMM on rows of up to 1024 entries, but 3–6× slower on rows of
15 000–52 000. The warp kernel was within 0.8–1.2× of SpMM on those long rows,
but up to 3× slower than the thread kernel on short ones.

**Memory, worst case.** The cached `Aᵀ` doubles a transposed operator's
storage, so the adapter caps it at 512 MiB per transpose. Above the cap an
operator keeps cuSPARSE's transposed product, which is the one
non-reproducible exception. The sparse Gram's `Aᵀ` is not capped, as before. The cap comes from the largest operator at hand,
`Head-and-Neck_01`'s 1.1e8-nonzero dose block on the CuPy dense route. These
are 4-iteration runs, with the pool peak read from a CuPy memory hook:

| arm | pool peak used / reserved (GiB) | run time |
| --- | --- | --- |
| before (no `Aᵀ`) | 6.51 / 7.31 | 54–56 s |
| cache the 1.24 GiB `Aᵀ` | 7.75 / 8.91 (past the 8 GiB card) | 68 s |
| transpose 2²⁴-entry row windows per call | 6.85 / 7.60 | 66 s |
| cap at 512 MiB (shipped) | 6.51 / 7.31 | 48 s |

Transposing row windows on every call cost about 7× a cuSPARSE transposed
SpMV (27.0 vs 3.8 ms on the `Protons_01` KKT). That route's device-tiled Gram
is not reproducible anyway, so the cap gives up nothing there.

**A CuPy fault found on the way.** `cupy.bincount` with `minlength ≈ 4e7`
raised `cudaErrorIllegalAddress` (CuPy 14.1.1, CUDA 12.9). The segment sum
finds its run boundaries with `searchsorted` instead.

To reproduce the trust arm of the previous section, which is now
reproducible run to run, add `--backends cupy --cupy-certificates
--cudss-deterministic` to the gate command below. The bisection harness, the
probes and the logs are in the session scratchpad (`repro_sparse/`, local,
not committed).

### Dense Cholesky through `dpotrf` (2026-10-03)

The NumPy dense route now factors `N` with SciPy's `dpotrf` on a single
cache-blocked copy instead of `numpy.linalg.cholesky`; see the
[TROTS Head-and-Neck_01 profile](trots.md#head-and-neck_01-speed-profile-2026-10-02),
lever 8. Both run `dpotrf('L')` on the same data. The change is expected
to be bit-identical, so it was checked with a bitwise A/B instead of the
gate sweep. Both dense configs (`lbfgs/dense`, `exact/dense`) ran over the
whole corpus (`--all --include-objective-free`), one arm with
`numpy.linalg.cholesky` and one with `dpotrf`. The budget was `--max-iter
500` with a 30–60 s cap, one BLAS thread per process, 8 processes. Every
solve's per-iteration objective and KKT error and its final `x` were
recorded:

| rows | iterations compared | identical, final `x` included | identical up to a `max_time` stop at a different iteration | diverged |
| ---: | ---: | ---: | ---: | ---: |
| 2196 | 164 748 | 2132 | 64 | **0** |

The 64 rows hit the time cap at different iterations, because the two
arms ran at different times on a shared machine. Their common prefixes are
identical. Because the trajectory does not change, verdicts do not
either, so the v33 baseline stands. The HN A/B (24 threads) was
bit-identical as well. The harness (`s2mpj_arm.py`,
`compare_bitwise.py`) and the reports are in the session scratchpad
(`sweep_potrf/`, local, not committed).

## Reproducing

From a checkout with `IPAX_S2MPJ_DIR` pointing at an
[S2MPJ](https://github.com/GrattonToint/S2MPJ) clone:

```bash
# one configuration, full corpus, dataset-scored, recoverable, 5 problems at a time
python -m benchmarks.runners.s2mpj --all --config exact/sparse --jobs 5 \
    --include-objective-free --resume --max-iter 10000 --max-time 300 \
    --out benchmarks/reports/s2mpj_exact_sparse
```

The full release gate is the whole matrix in one run. **Pass the budget
explicitly** — the runner defaults (`--max-iter 1000 --max-time 60`) are far
below what every tracked baseline was run at, and silently capping
long-but-converging solves manufactures a corpus-wide regression that looks
like a real one:

```bash
IPAX_S2MPJ_DIR=<checkout> PYTHONUTF8=1 OMP_NUM_THREADS=1 \
python -m benchmarks.runners.s2mpj --all --include-objective-free \
    --max-iter 10000 --max-time 300 --jobs 10 --resume \
    --out benchmarks/reports/s2mpj_vNN
```

`--include-objective-free` is load-bearing, not optional: without it the 205
feasibility problems are skipped and the totals are not comparable to any
baseline here. Any report's budget is recoverable from its own data — every
`max_iter` row's `n_iter` equals the cap plus one, and every `max_time` row's
`solve_time` sits at the cap — which is worth checking on a partial report
before committing hours to a run.

A sweep is only meaningful against a baseline, so A/B two reports with:

```bash
python -m benchmarks.runners.compare \
    benchmarks/reports/s2mpj_v20.json benchmarks/reports/s2mpj_v22.json \
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
