# TROTS radiotherapy benchmark

[TROTS](http://www.trots.eu) (the Radiotherapy Optimisation Test Set,
Breedveld & Heijmen) is the corpus `ipax` is tuned toward: real
treatment-planning problems with `n ≈ 10³` fluence variables and up to several
`10⁵` per-voxel dose inequality rows, started from a warm start that fits the
tumour objective but is deeply infeasible on the organ-at-risk constraints.
The loader (`benchmarks/corpus/trots.py`) reproduces every case's published
reference objective at the dataset's `solutionX` to 1e-9..1e-13 relative
error, so the reference objectives below are trustworthy oracles.

This page records the measured **RT recipe** (2026-08-04, first patients of
each group plus a second patient per group for generalization; details and
provenance at the bottom).

## The recipe

```python
import ipax
from ipax.options import BarrierOptions

# Universal first move at RT scale:
opts = ipax.Options(
    max_iter=800,
    max_time=1800.0,  # realistic budget, see below
    barrier=BarrierOptions(slack_init_scale=0.1),
)

# Proton-signature cases additionally want the μ-raising stack:
opts_protons = ipax.Options(
    max_iter=800,
    max_time=1800.0,
    mu_schedule="quality",
    globalization="breedveld",
    barrier=BarrierOptions(slack_init_scale=0.1, kappa_centrality=1e-4),
)
```

Everything else stays at defaults (`hessian="lbfgs"`, `linsolve="auto"`,
which picks the dense condensed route at this scale — keep BLAS threading
on).

## Measured results

Budgets 30 min / 800 iterations per case (BT: 5 min), NumPy backend,
one process per case, threaded BLAS. "ref" is the dataset's published
reference objective for that patient.

**Universal arm — defaults + `slack_init_scale=0.1`:**

| case | ref obj | result | status | iters | KKT | time |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Liver_01 | −1.037070 | **−1.037070** | optimal | 211 | 9.1e-9 | 9 min |
| Liver_02 | −0.159868 | **−0.159864** | optimal | 285 | 8.6e-9 | 10 min |
| Prostate_CK_01 | 2.825806 | **2.825830** | acceptable | 254 | 3.6e-8 | 21 min |
| Prostate_CK_02 | 1.338323 | **1.338380** | optimal | 220 | 7.7e-9 | 26 min |
| Prostate_VMAT_101 | 21.8787 | 21.964 | max_time | 403 | 7.4e-5 | 30 min |
| Prostate_VMAT_102 | 20.2383 | 20.453 | max_time | 287 | 1.4e-4 | 30 min |
| Protons_01 | 32.6885 | 32.922 | stalled | 296 | 2.5e-5 | 21 min |
| Protons_02 | 42.9379 | 42.959 | max_time | 233 | 7.7e-6 | 30 min |
| Prostate_BT_01 | — | ≈0 | optimal | 15 | 1.1e-9 | <1 s |
| Prostate_BT_02 | — | −0.7527 | acceptable | 401 | 6.5e-7 | 9 s |

**Proton arm — the μ-raising stack on top of `slack_init_scale=0.1`:**

| case | ref obj | result | status | iters | KKT | feasible @ | peak \|obj\| |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Protons_01 | 32.6885 | **32.650** | acceptable | 171 | 1.1e-7 | 43 | 104 |
| Protons_02 | 42.9379 | **42.925** | acceptable | 149 | 1.5e-7 | 39 | 1113 |

Both proton solves certify *below* the published reference objective. For
context, stock IPOPT (limited-memory, adaptive μ) on Protons_01 reaches
feasibility at iteration 20 with a peak objective excursion of 62 and does not
converge the optimality tail within a 30-minute budget; ipax's stacked arm
reaches feasibility at 43 with a peak of 104 and certifies at KKT 1e-7.
(Compare **iteration counts**, never wall-clock, against compiled solvers.)

### Mixed-precision Gram (`gram_dtype="auto"`, the default)

The Gram accumulation is 80–93% of RT wall time, and much of the TROTS dose
data is float32 in the file — so the default reduces that accumulation exactly
where the data carries no float64 information, certifying every step by
iterative refinement against the exact operator (see
[the linalg concepts page](../concepts/linalg.md#mixed-precision-gram-accumulation-dense-route)).
What each case gets is a property of *its own data*, not a tuning choice:

| case | float32 share (by nnz) | engages | effect |
| --- | ---: | --- | --- |
| Prostate_VMAT_101 | 95.9% | yes, per block | **1.25×** end-to-end |
| Protons_01 | 0.07% | yes, ~nothing to reduce | +0.8% (noise) |
| Head-and-Neck_01 | 0% | no (correctly inert) | — |

Single-thread CPU, 16 iterations for VMAT and 21 for Protons, `auto` vs
`native`. VMAT is the mixed case the per-block reduction exists for: it is
96% float32 held back by a single float64 constraint matrix, so an
all-or-nothing rule reduced *nothing* there. Protons is the honest cost —
45 of its 369 445 rows are float32, so it engages for no real gain; the
objective is bit-identical because 99.93% of that Gram is still accumulated
exactly. VMAT's objective agrees with the native run to 8 significant figures
at identical KKT error (a trajectory difference within the refinement
certificate, not an accuracy loss).

**The speedup is CPU-side; do not read it as cross-platform.** On CUDA
(RTX 4070 Laptop, CuPy) the same VMAT reduction is a *wash* — interleaved
Gram timings with clocks verified stable give 0.96× for the per-block form and
1.12× for reducing everything, and the refinement overhead is negligible
(1.15 s across 7 iterations). At `n ≈ 2·10³` the Gram is bound by the
sparse→dense chunk expansion rather than the GEMM, so halving the accumulate
width changes little; the same signature as the syrk result (1.00× at
`n = 1080`, 1.80× at `n = 5000`). Correctness *is* verified on device: the
per-block filter fires (per-block and blanket results differ bitwise, with
per-block carrying the smaller error), the rank-k update leaves the Gram
exactly symmetric, and the objective matches the native run.

Whether the reduction pays on GPU at Head-and-Neck scale (`n ≈ 10⁴`) is
**unmeasured**: that case saturates an 8 GB card's memory pool and drives it
into clock throttling (2535 → 765 MHz), and repeat runs disagreed by more than
an order of magnitude in both the absolute times and the ratio. It needs
hardware with the headroom to answer it. Note also that `auto` would not engage
there in any case — Head-and-Neck's data is genuinely float64, so the
reduction would have to be user-forced.

**Device residency of the constant block.** The lowered `G z + h ≤ 0` block is
by far the largest array in an RT problem — 86 M nonzeros on
`Prostate_VMAT_101` — and the loader keeps exactly one device copy of it: the
constraint operator it hands the solver. Evaluating the linear constraint rows
through that operator, rather than mirroring `G` to the device a second time,
measures 3057 MiB in the CuPy pool instead of 4040 MiB on that case (the host
side drops the pre-lowering two-sided assembly for a comparable saving). What
remains is the Array-API COO triplets the core holds (24 B/nnz) plus the
adapter's device CSR (12 B/nnz) — invariant #4's cost, paid once. The
Head-and-Neck saturation above was measured before that saving.

Two measurement notes for anyone repeating this. The sparse adapters memoize
the Gram per `accumulate_dtype`, keyed on weight equality (it serves `δ_w`
retries within an iteration), so a timing loop reusing one weight vector
measures a cache hit rather than the kernel. And on a laptop GPU, arms run
back-to-back drift enough to invert the result — interleave the variants and
log the clocks.

## Why these levers

- **`slack_init_scale=0.1`** (the universal lever): the flat slack floor
  (`1e-2`) pins every violated constraint's slack near zero while TROTS
  constraint violations at the warm start are O(10–10³); the first Newton
  directions then drive those slacks at the boundary and fraction-to-boundary
  clips the primal step to `≈10⁻³` indefinitely. Scaling the floor to
  `max|g(x₀)|` fixes the slack — and, through `y = μ/s`, the dual — scale at
  init. Without it: Liver needed 329 iterations for a weaker certificate, CK
  livelocked at constraint violation 54, protons never reached feasibility.
- **μ-raising stack for protons only**: the proton cases are the extreme
  tall/infeasible regime (`m/n ≈ 340`, 33k violated rows) where escaping the
  boundary-clipped grind needs the barrier re-targeted upward
  (`mu_schedule="quality"`) and the excursion that causes needs both the
  Breedveld step controller and a relaxed centrality floor
  (`kappa_centrality=1e-4`) to stay bounded — the floor otherwise couples μ to
  the O(10²–10³) transient infeasibility and the barrier dominates for tens of
  iterations (peak excursion 1925 → 104 with the stack).

!!! warning "Do not blanket-apply the μ-raising stack"
    On Liver, CK, and VMAT, `mu_schedule="quality"` (or `"breedveld"`) drives
    the objective into a barrier-dominated excursion of 10⁴–10⁶ that does not
    recover within any practical budget — with or without the other levers.
    The stack is a *proton-signature* branch (deeply infeasible, extremely
    tall, default recipe stalls near the optimum after a long `α ≈ 10⁻³`
    grind), not a general RT setting.

## Bound-only fluence-map runs

Not every RT formulation carries per-voxel dose *constraints*: composite
plans often fold the dose criteria into the objective, leaving `min f(Dx)`
with only box bounds on the fluence — a different regime from the TROTS
tables above (no slacks, no Gram; the barrier acts on the bounds alone). The
recipe for that shape was measured on a synthetic study problem
(2026-08-28: piecewise least-squares `f(Dx)`, `0 ≤ x ≤ 10`, `n = 50 000`,
`D` 80k×50k sparse with 1.9·10⁷ nonzeros, `cond(DᵀD) ≈ 1.4·10⁵` — see
provenance below):

```python
from ipax.options import LBFGSOptions, LineSearchOptions

opts = ipax.Options(
    # mu_schedule="monotone" is the default — keep it (see numbers below)
    lbfgs=LBFGSOptions(seed_formula="scalar1", memory=20),  # 20–50
    line_search=LineSearchOptions(backtrack_interpolation=True),  # optional
)
```

Measured at `n = 50k` over 300 iterations (objective, lower is better;
SciPy L-BFGS-B references on the identical problem: `m=10` reaches 8.61,
`m=20` reaches 8.34):

| arm | objective @300 |
| --- | ---: |
| defaults (`direct` seed, monotone μ) | 13.4 |
| `scalar1` seed, memory 20, monotone μ | **8.396** |
| `scalar1`, memory 50 | ≈ converged by iteration 100 |
| `scalar1` m20, `mu_schedule="adaptive"` | 8.97 |
| `scalar1` m20, adaptive + Gondzio correctors | 21.6 |
| `scalar1` m20, `globalization="breedveld"` arms | 25.7–113.8 |

The mechanism: the default `direct` ξ seed is over-stiff on chained least
squares (the δ–γ misalignment factor — the same pathology as the
`GASOIL`/`NELSONLS` corpus signature), so steps are tiny despite cheap
iterations; `scalar1` un-freezes them, and the larger window pays because
`DᵀD`'s spectrum is genuinely ill-conditioned. The default **monotone μ
already wins outright** here — every adaptive/Breedveld arm is worse, and
Gondzio correctors cost +40% time for no objective gain. `scalar1`'s
under-stiff seed does buy more halving backtracks (3.5–3.9 objective
evaluations per iteration); the opt-in `backtrack_interpolation` cuts that
to 2.0–2.4 (measured 87 → 65 ms/iteration at memory 20).

Both the dense and the Krylov route hit the structured Woodbury solve on
this shape (the Krylov route reports `pc=lbfgs-exact`); either is fine, and
the per-iteration floor is the two dose-matrix passes (`D@x` + `Dᵀg`). To
avoid paying a third pass, share `D@x` between the objective and the
gradient — see the
[Problem guide](../guide/problems.md#sharing-work-between-the-objective-and-the-gradient).

The same recipe on GPU (CuPy, RTX 4070 Laptop 8 GB, measured 2026-09-01 at
`n = 50k` with interleaved order-alternated pairs, identical trajectories to
CPU): **11.9 ms/iteration** on the dense route and 12.7 on Krylov, vs ~56 on
this machine's threaded-BLAS NumPy — about **4.5×** end to end. The exact
Woodbury inverse is what makes the Krylov route viable on device at all:
with `exact_lbfgs_inverse=False` the Jacobi-CG arm pays ~27 latency-bound CG
iterations per solve plus the O(n·k²) L-BFGS diagonal and lands at
42.6 ms/iteration (3.4× slower). The remaining dense-vs-Krylov gap
(~0.5–0.8 ms) is the direct solve's verification round — the extra
operator matvec and its bookkeeping, *not* the host syncs: a follow-up
microbenchmark (2026-09-02) measured a CuPy scalar sync at ~14 µs on this
stack, so the GPU loop's non-compute time is per-op Python/launch
overhead (~11–16 µs across several hundred small array ops per
iteration), and batching the driver's scalar decision reads — shipped
anyway for the sync-expensive stacks — moved wall time by ~1% here.

!!! warning "This is a routing hint, not a better default"
    `seed_formula="scalar1"` as a corpus-wide default measured **net −140**
    on the full S2MPJ sweep (see the
    [routing-hints page](../guide/routing-hints.md)); it is decisive on this
    signature and harmful elsewhere. Reach for it when your problem matches
    the shape above.

**Provenance:** measured 2026-08-28 on `develop` (post-0.10.1), NumPy
backend, single desktop; per-arm objective trajectories at fixed iteration
budgets, interleaved runs. The study problem is synthetic (banded-random
`D`, piecewise-quadratic under/overdose penalties) but matches the
fluence-map shape in dimensionality, sparsity, and conditioning.

## Known limits

- **VMAT last mile**: both VMAT cases descend cleanly to within ~1 % of the
  reference but hold `α ≈ 5e-3` in the tail; certification needs a larger
  budget than 30 min.
- **Head-and-Neck scale** (`n ≈ 10⁴`, `m ≈ 10⁵`): ~45 s/iteration on 32
  threads — a practical-budget scale limit of the generic routes. The
  RT-specific dose-matrix kernels (graph tiling, custom mixed-precision tiled
  BLAS) that close the remaining per-iteration gap to Breedveld's tuned solver
  are deliberately out of scope (see `AGENTS.md`); the *generic*
  mixed-precision Gram accumulation is in, and engages by itself wherever the
  dose matrices are float32 in the file — this case is not one of them
  (`Head-and-Neck_01` is float64 throughout, so `gram_dtype="auto"` correctly
  does nothing for it).
- **Prostate_BT_02** shows the slack lever's basin effect: the plain default
  certifies `optimal` at objective ≈0 in 13 iterations, the recipe takes 401
  iterations (9 s) to a strictly better optimum at −0.75 — same signature as
  the lever's `HS59`/`HS98` corpus wins.

## Reproducing

```bash
IPAX_TROTS_DIR=/path/to/TROTS python -m benchmarks.runners.trots --solve \
    --cases Liver_01 --config lbfgs/dense \
    --slack-init-scale 0.1 --max-iter 800 --max-time 1800

# Proton stack:
IPAX_TROTS_DIR=/path/to/TROTS python -m benchmarks.runners.trots --solve \
    --cases Protons_01 --config lbfgs/dense \
    --slack-init-scale 0.1 --kappa-centrality 1e-4 \
    --mu-schedule quality --globalization breedveld \
    --max-iter 800 --max-time 1800
```

Run one case per process at scale (the big `.mat` files do not co-reside in
memory), and keep BLAS threads on — the dominant per-iteration cost (the
condensed Gram + Cholesky) parallelizes.

**Matrix cache.** Parsing the MATLAB v7.3 dose matrices dominates startup, and
every run re-paid it. Parsed matrices are now mirrored to `.npz` files in a
`.ipax_trots_cache/` directory beside the dataset, keyed by the source file's
size, mtime and a format version — so editing a `.mat` misses rather than
serving a stale matrix. Measured warm vs cold: `Prostate_CK_01` 7.3 s → 3.1 s,
`Prostate_VMAT_101` 12.5 s → 4.5 s, with the assembled constraint block
identical either way. The cache trades disk for time — roughly twice the
`.mat` size, since HDF5 stores the values gzip-compressed (~1.1 GB for those
two cases; values whose float64 widening is provably lossless are stored
narrow to halve that). Point `IPAX_TROTS_CACHE` at another directory to
relocate it, or set it to `off` to disable it.

**Provenance.** Measured 2026-08-04 on `develop` (post-0.9.0, including the
then-unreleased terminal-certificate change), NumPy backend, 32-core desktop,
threaded BLAS. Exploration budget 900 s/400 iters; validation budget
1800 s/800 iters. The per-arm iteration trajectories (feasibility iteration,
peak excursion, α/μ traces) were captured via `Result.history`.
