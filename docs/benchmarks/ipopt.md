# IPOPT cross-check

The [S2MPJ accuracy sweep](s2mpj.md) says *whether* ipax reaches each problem's
documented optimum. It cannot say **why** a failure happened — whether the
problem is hard, or ipax has a gap. This comparison answers that by running an
established solver from the same starting point and reporting a verdict per
problem:

| verdict | meaning |
| --- | --- |
| `agree` | both solvers solved it — validates ipax on that problem |
| `ipax-gap` | the reference solved it, ipax did not — **the actionable list** |
| `ipax-wins` | ipax solved it, the reference did not |
| `both-hard` | neither solved it — the difficulty is genuine, not ipax-specific |

A `*` suffix marks a problem the dataset documents no optimum for; those are
scored on solver *success* rather than against a reference objective.

The reference is IPOPT through the sparse-native
[`ipyopt`](https://gitlab.com/g-braeunlich/ipyopt) binding — it consumes the
constraint Jacobian as a COO pattern plus values, so unlike the SciPy-style
`cyipopt` path it does not densify and reaches the tall, sparse systems ipax
targets. Both are optional dev dependencies; this never runs in CI.

## Reading a verdict honestly

**Iteration counts are the comparable axis, never wall-clock.** ipax is pure
Python over an Array API abstraction; IPOPT is compiled. A time comparison
measures the implementation language, not the algorithm. Wall-clock is recorded
but is not a result.

**The two solvers are compared as each ships.** Their defaults are not the same,
and the differences move verdicts, so every report carries a Parameters section
naming them. Aligned by construction: the starting point, gradient-based NLP
scaling (`max_gradient=100` on both), a `1e-8` convergence tolerance with a
`1e-6` acceptable level, a limited-memory Hessian, and a filter line search.
Not aligned, as measured on this build rather than read from documentation:

| knob | ipax | IPOPT |
| --- | --- | --- |
| barrier schedule | `monotone` | `adaptive` — the build default, despite the docs |
| L-BFGS history | 10 | 6 |
| linear algebra | dense condensed normal equations | MUMPS augmented system + inertia |

Both were verified on `AGG`: with `mu_strategy` unset IPOPT takes 185 iterations,
exactly matching explicit `adaptive`, while explicit `monotone` takes 352; and
`limited_memory_max_history=10` moves it from 185 to 100. Re-run any gap with
`--ref-option mu_strategy=monotone --ref-option limited_memory_max_history=10`
to separate a defaults difference from a structural gap.

**A gap list is only worth its confound analysis.** Both solvers get the same
iteration and wall-clock budget by default (`--ref-max-iter` inherits
`--max-iter`); a reference with a larger budget manufactures gaps.

## Latest run (v5, 2026-08-01)

ipax `0.9.0` + the terminal KKT certificate (develop @ `2e7f3a1`) ·
Python 3.14.6 · Windows 11 · NumPy 2.4.6 · SciPy 1.17.1. Config `lbfgs/dense`,
1000 iterations / 60 s per solve on both sides, the accuracy sweep's dense
variable cap (2000). 1098 problems compared, 3 skipped as oversized, **every
one of them reaching both solvers**.

| verdict | v4 (2026-07-30) | v5 |
| --- | --- | --- |
| `agree` + `agree*` | 637 | **640** |
| `both-hard` + `both-hard*` | 310 | 307 |
| `ipax-wins` + `ipax-wins*` | 76 | **80** |
| `differ*` | 35 | 36 |
| **`ipax-gap` + `ipax-gap*`** | 40 | **35** |

### The confound-free backlog: 29 problems

Re-running the 35 gaps with the reference's parameters matched to ipax's
(`mu_strategy=monotone`, `limited_memory_max_history=10`, equal budget) clears
6 (`LEVYMONT`, `LEVYMONT8`, `NUFFIELD`, `SSEBNLN`, `SSINE`, `DMN37143LS`) and
leaves **29 that survive** — down from 34 in v4 and 47 in v3. `AGG` still
survives fully matched.

Classifying those 29 by what actually went wrong — using the constraint
violation at each solver's returned point, measured on the *raw* (unscaled)
constraints so both points are judged by one ruler:

| class | v4 | v5 | what it means |
| --- | --- | --- | --- |
| A. never reached feasibility | 9 | 8 | ipax stops at a violated point where IPOPT reaches ~1e-9 |
| B. reached it, would not certify it | 7 | **4** | same objective, feasible, but ipax reports `stalled`/`max_iter` |
| **C. worse objective, both feasible** | 9 | **8** | a genuine optimality gap — **now the largest class** |
| D. lower objective, both feasible | 4 | 5 | ipax's point is *better* — dataset or reference-basin question |
| E. out of budget | 5 | 4 | speed, not correctness |

!!! note "Class B was a certification bug, and the terminal certificate closed it"

    Class B — runs parked *at* the answer that would not say so — was diagnosed
    at the certificate level (2026-08-01): the per-iteration test evaluates the
    KKT residual with whatever multipliers the trajectory carries, but KKT
    satisfaction is an existence claim over them. On the zero-objective
    equation systems a rank-deficient `∇c` leaves the recorded residual as
    drifted-dual noise (`NONSCOMPNE`: reported KKT `6.8e-5` at a point whose
    least-squares multipliers give dual infeasibility exactly 0), and `WEEDS`
    returned a best iterate at `3.8e-7` — inside the acceptable band — while
    being judged on its frozen `6.2e-6` tail iterate.

    A run ending `stalled`/`max_iter`/`max_time` now re-judges the returned
    best iterate, with repaired candidate multipliers if needed, and reports
    `acceptable` when the full scaled residual passes. That certified `WEEDS`,
    `NONSCOMPNE` and `VANDERM1` at the same answers IPOPT reaches (now
    `agree`/`agree*`), and swept the wider equation-system and least-squares
    families in the [accuracy sweep](s2mpj.md) (+46/−0 across six configs).

    The **4 that remain are correctly uncertifiable** — measured with exact
    least-squares duals at both solvers' returned points, not guessed:
    `CURLY30` floors at `1.32e-6`, 32% above the acceptable band, at IPOPT's
    exact point (IPOPT needed 745 iterations to polish it); `POWELLBSLS` and
    `ORTHREGA` are parked at genuinely non-stationary points (`ORTHREGA`'s
    achievable dual infeasibility at ipax's point is `1.45e-2`, at IPOPT's
    `4.5e-9` — the measures agree, the trajectory is the gap); `VANDERM2` sits
    a hair above the feasibility band (`1.4e-6`).

!!! note "Class A: the 2026-07-29 repair closed two thirds; 8 remain"

    Class A was dominated by the CUTEst **nonlinear-equation systems** run as
    feasibility problems; the feasible-point barrier-repair fix closed 14 of
    23 (see the v4 report and changelog). The **8 that remain** are `AGG`,
    `ARTIF`, `COOLHANS`, `DISCS`, `HATFLDFLNE`, `HYDCAR20`, `LAKES`,
    `VANDERM3`. `COOLHANS` is the informative one: its restoration exits
    *stationary* rather than *feasible*, so the repaired guard never engages —
    a different sub-case of the same phase.

**Class C is now the largest class and the primary target**: `CRESC50`,
`GASOIL`, `HS59`, `HS98`, `NELSONLS`, `ORTHRGDS`, `PALMER1E`, `SINROSNB` —
both solvers land feasible and ipax settles for a worse objective. Cross-read
against the accuracy sweep's six routes, the class splits three ways rather
than one: `GASOIL` and `SINROSNB` reach the correct optimum under the sweep's
10× budget (convergence *speed*, not basin); `HS98`, `NELSONLS` and `ORTHRGDS`
are solved by the exact-Hessian routes but not the L-BFGS ones (an
L-BFGS-quality gap — IPOPT manages with limited memory); only `HS59`,
`PALMER1E` and `CRESC50` miss on every route (true basin/search gaps; `HS59`
is a known robust gain of the opt-in `slack_init_scale`).

Class D deserves its own honest reading: on `DIAMON2DLS`, `DMN15102LS` and
`GAUSS3LS` ipax's returned objective is **orders of magnitude better** than the
point IPOPT certified (`8916` vs `1.5e7`; `1244` vs `1.3e6`, the latter an
`exp`-underflow plateau where the gradient is exactly zero in float64) — these
are "gaps" only because ipax honestly reports `max_iter` instead of certifying.
`BT4`/`BT5` are long-standing dataset-value suspects.

Note that `ipax_infeasibility` in the report is the **raw** constraint violation,
while `Result.constraint_violation` is measured on the gradient-scaled problem.
The two legitimately differ (on `MISRA1C`, `0.016` scaled against `47.6` raw);
never quote them as the same quantity.

## Declaring a sparsity pattern for a solver that demands one

IPOPT wants the constraint Jacobian's sparsity declared **once**, up front, and
many CUTEst problems will not oblige: an entry whose value happens to be exactly
zero at a point is simply absent from the operator's COO triplets there, so the
structural pattern depends on where you look. The baseline therefore samples
several points and declares the **union**, letting the values callback store an
explicit zero wherever an entry is missing at the current point.

Sampling cannot *prove* coverage, so the callback verifies rather than assumes —
it scatters every point's nonzeros into the declared layout by `(row, column)`
and raises `BaselineUnsupported` if one falls outside. Two details that decide
whether this works at all:

- **Sample the interior, not the bounds.** Points clamped *onto* a bound are the
  worst place to look, because entries routinely vanish exactly there: `EIGMINA`
  loses a Jacobian entry when its first variable sits at its upper bound of 1,
  which is where every sample landed. Sample points are inset 1% into the box —
  also where an interior-point method's iterates actually live.
- **Never trust a small sample's agreement.** An earlier version declared a
  pattern "stable" when two sampled points matched and then passed the
  operator's values through verbatim. `EIGMINA` emits 5 nonzeros at both of
  those points and 6 once the iterates move, so IPOPT received an array of the
  wrong length and NumPy raised a shape error from inside the solve, where it
  could not be attributed to anything.

With both in place every problem in the corpus reaches the reference solver.

## Reproducing

```bash
# Full corpus, 12 workers. Pin BLAS threads so per-solve timings stay comparable.
IPAX_S2MPJ_DIR=/path/to/S2MPJ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m benchmarks.runners.s2mpj_baselines --all --jobs 12 \
    --out benchmarks/reports/s2mpj_baselines_v2 --max-iter 1000 --max-time 60

# The confound arm: re-run just the gaps with the reference matched to ipax.
python -m benchmarks.runners.s2mpj_baselines --names-file gaps.txt --jobs 10 \
    --ref-option mu_strategy=monotone --ref-option limited_memory_max_history=10 \
    --out benchmarks/reports/s2mpj_baselines_v2_matched
```

A run over this corpus is unattended for tens of minutes and will meet native
crashes, so it flushes both reports after every problem, `--resume` continues
from the partial report, and each worker records the problem it is running in a
pid-suffixed `.inflight` marker — a crash leaves at most `--jobs` named
candidates to `--exclude`, rather than the whole pending queue.
