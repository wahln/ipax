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

## Latest run (v4, 2026-07-30)

ipax `0.8.0` + the feasible-point barrier repair · Python 3.14.6 · Windows 11 ·
NumPy 2.4.6 · SciPy 1.17.1. Config `lbfgs/dense`, 1000 iterations / 60 s per
solve on both sides, the accuracy sweep's dense variable cap (2000). 1098
problems compared, 3 skipped as oversized, **every one of them reaching both
solvers**.

| verdict | v3 (2026-07-28) | v4 |
| --- | --- | --- |
| `agree` + `agree*` | 624 | **637** |
| `both-hard` + `both-hard*` | 322 | 310 |
| `ipax-wins` + `ipax-wins*` | 64 | **76** |
| `differ*` | 35 | 35 |
| **`ipax-gap` + `ipax-gap*`** | 53 | **40** |

### The confound-free backlog: 34 problems

Re-running the 40 gaps with the reference's parameters matched to ipax's
(`mu_strategy=monotone`, `limited_memory_max_history=10`, equal budget) clears
6 and leaves **34 that survive** — down from 47 in v3. The parameter asymmetry
accounts for about an eighth of the list, not the list: `AGG` in particular
survives fully matched, with IPOPT solving it in 324 iterations under ipax's own
settings.

Classifying those 34 by what actually went wrong — using the constraint
violation at each solver's returned point, measured on the *raw* (unscaled)
constraints so both points are judged by one ruler:

| class | v3 | v4 | what it means |
| --- | --- | --- | --- |
| A. never reached feasibility | 23 | **9** | ipax stops at a violated point where IPOPT reaches ~1e-9 |
| B. reached it, would not certify it | 8 | 7 | same objective, feasible, but ipax reports `stalled`/`max_iter` |
| **C. worse objective, both feasible** | 9 | **9** | a genuine optimality gap |
| D. lower objective, both feasible | 4 | 4 | ipax below the documented optimum — dataset or basin question |
| E. out of wall time | 3 | 5 | speed, not correctness |

!!! note "Class A was one coherent theme, and fixing it closed two thirds of it"

    Class A was dominated by the CUTEst **nonlinear-equation systems** run as
    feasibility problems, where ipax stopped at violations of `1e-3` to `6.7e3`
    while IPOPT drove the same systems to machine precision — the same
    *feasibility-phase* weakness the radiotherapy work isolated from the other
    direction (the Phase-1 stall behind `BarrierOptions.slack_init_scale`).

    The cause was a guard that skipped a no-op restoration only when the problem
    had *inequality* constraints, so it never fired on equality-only systems;
    see the changelog for the three changes that were needed together. That
    closed 14 of the 23 — `CORE2`, `CYCLOOCF`, `DRCAVTY1`/`2`/`3`, `DRUGDIS`,
    `EXTROSNBNE`, `HADAMARD`, `HYDCAR6`, `METHANL8`, `OPTCNTRL`, `ROBOT`,
    `SPANHYD`, `SPINOP` — most of them from a violation of order 1 to below
    `1e-9`.

    The **9 that remain** are `AGG`, `ARTIF`, `COOLHANS`, `DISCS`, `HATFLDFLNE`,
    `HYDCAR20`, `LAKES`, `VANDERM2`, `VANDERM3`. `COOLHANS` is the informative
    one: its restoration exits *stationary* rather than *feasible*, so the
    repaired guard never engages — a different sub-case of the same phase.

With class A cut to 9, **class C is now tied for the largest class**: 9 problems
where both solvers land feasible and ipax settles for a worse objective. That is
an optimality question rather than a feasibility one, so the corpus no longer
points primarily at the feasibility phase.

Class B remains the cheapest target: on 7 problems ipax is *at* the answer and
will not say so, which is a termination-criteria question rather than a search
one.

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
