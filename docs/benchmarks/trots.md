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

**Breedveld-parity recipe (the runner's default).** Breedveld et al. 2017's
solver setup, lever by lever, on the loader's tiled Gram and grouped exact
Hessian:

```python
import ipax
from ipax.options import BarrierOptions
from benchmarks.corpus.trots import TROTSExactProblem, load_trots_file

problem = TROTSExactProblem(load_trots_file("Protons_01.mat"), xp, sparse=True)
x0 = problem.initial_point(strategy="breedveld")  # eqs. (23)–(24), floor 5
warm, mu0 = problem.breedveld_warm_start(x0)  # eqs. (26)–(28), μ0 eq. (10)
opts = ipax.Options(
    hessian="exact",
    linsolve="dense",
    corrections="mehrotra",  # predictor–corrector …
    mu_schedule="monotone",  # "probing" (μ from the predictor) on protons
    globalization="breedveld",  # ratio control, then the Markov filter
    barrier=BarrierOptions(mu_init=mu0),
    max_iter=300,
)
result = ipax.solve(problem, x0, options=opts, warm_start=warm)
```

`Liver_01`: 74 iterations / 80 s on NumPy to a certified `1e-8` point at the
reference objective (the reference solve: 36 iterations / 22 s), his
tolerance class at iteration 37. `Protons_01` with `mu_schedule="probing"`:
104 iterations, 60 s on NumPy / 27 s on CuPy plus a 4 s initialization (1 s
for the least-squares fluence, 3 s for the LSQR dual start; the reference:
79 iterations / 55 s); with the monotone default 227. The μ
schedule is the one lever that splits by site — see *Why these levers*.

**Previous L-BFGS recipes** (still the right choice for an L-BFGS solve:
`--config lbfgs/dense --init ipax --mu-schedule monotone --globalization filter
--corrections none --slack-init-scale 0.1` in the runner):

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

**Proton arm, revisited with the tiled Gram (2026-09-26) — exact Hessian +
Mehrotra:** with the per-iteration cost down to ~0.6 s (see *Tiled per-structure
Gram* below), `Protons_01` was run to convergence under four configurations,
all with `linsolve="dense"` and `slack_init_scale=0.1` at the default `1e-8`
tolerance, on an idle machine. `Protons_01` has no curved cost functions, so
`hessian="exact"` means `W ≡ 0` (no L-BFGS curvature model), at the same
per-iteration cost as L-BFGS; `corrections="mehrotra"` adds a second step solve
per iteration. Breedveld's reference solve (`Results/Protons_01_result.txt`,
optimality measure ~1e-3) takes 79 iterations / 55 s; the two "first at"
columns report the first iterate whose scaled KKT ∞-norm is ≤ 1e-4 and whose
objective is within 1 % of the published reference.

| hessian | corrections | status | objective | iters | s/iter | KKT ≤ 1e-4 first | within 1 % first |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| lbfgs | none | stalled | 32.956 | 287 | 0.64 | 122 | 249 |
| exact | none | acceptable | **32.650** | 358 | — | 144 | 234 |
| lbfgs | mehrotra | acceptable | **32.648** | 255 | — | 191 | 191 |
| exact | mehrotra | acceptable | **32.648** | 181 | 1.29 | 99 | 125 |

The L-BFGS stall (the objective freezes at 32.956 while the KKT error jumps
from 3e-5 to 25) is a globalization artefact, not an optimum: either the exact
Hessian or Mehrotra corrections removes it, and all three certified arms land
at 32.648, below the published reference. `exact` + `mehrotra` reaches the
reference's tolerance class at iteration 99 (Breedveld: 79) and a certified
`1e-8` point at 181 iterations / 234 s. Mehrotra saves 11 % of the iterations
under L-BFGS and 49 % under the exact Hessian — the paper's "−50 %" figure was
measured with exact Newton only. (The two "—" cells were timed while other
work shared the machine and are omitted.) Reproduce with
`--config exact/dense --init ipax --mu-schedule monotone --globalization filter
--corrections mehrotra --slack-init-scale 0.1`.

**Adding the adaptive barrier schedule (`mu_schedule="quality"`) on top of
`exact` + `mehrotra`** removes the slow tail after feasibility (from iteration
40 on, θ = 0 and the objective slid from 58 to 32.65 over 140 iterations under
the monotone schedule): 129 iterations to `acceptable` at 32.6476, the
reference's tolerance class reached at iteration **69** (Breedveld: 79) and 1 %
of the reference objective at 79. Wall on an idle machine, tiled Gram, the
float32 share gate folding Protons' 0.07 % float32 rows (route `dense`):
**NumPy 136 s (1.06 s/iter); CuPy on an RTX 4070 Laptop 38.8 s (0.30
s/iter)** — against the reference solve's 55 s on a 16-core 2017 Xeon. The
CuPy and NumPy trajectories agree through iteration 107 and certify the same
objective to 8 digits. Note the August finding still stands: the μ-raising
schedules help the proton cases and hurt Liver/CK/VMAT, so this is a per-case
recipe (`--config exact/dense --init ipax --mu-schedule quality --globalization
filter --corrections mehrotra --slack-init-scale 0.1`), not a default.

**Breedveld-parity recipe (2026-09-26) — the TROTS runner's default since
then, with monotone μ; the proton rows below use `--mu-schedule probing`.**
Breedveld's solver setup transcribed lever by lever: the exact
Hessian, the Mehrotra predictor–corrector with μ taken from the affine
predictor (`corrections="mehrotra"` + `mu_schedule="probing"` — the affine
solve doubles as the probe, so no extra solve per iteration), his
ratio-control step controller (`globalization="breedveld"`), and his §3.1
initialization: the dataset's least-squares fluence rescaled by eq. (24) and
floored at 5, LSQR multipliers from eq. (26) with `y₀`, `z₀`, `w₀` floored at
5 (eqs. 27–28) and `μ₀` from the duality gap (eq. 10) — reproduced at the
benchmark edge (`TROTSProblem.initial_point(strategy="breedveld")` +
`breedveld_warm_start()`) through the public warm-start API, no core change.
`Protons_01`, idle machine, tiled Gram, `linsolve="dense"`; "solve wall"
excludes the initialization, which the runner's reported solve time counts
in full: the least-squares fluence of eq. (23) — 1.0 s here, 3.5 s on
`Liver_01`, 18 s on `Prostate_VMAT_101` and 30 s on `Prostate_CK_01`, a
SciPy SpGEMM of the dataset's initialization matrices — and LSQR on the
host (3.4 s at the loader's `atol=btol=1e-6`, 24 s at `1e-8` for a
bit-identical start, because every LSQR multiplier is < 0.04 and the floor
sets all 369,445 of them to 5):

| backend | tol | status | objective | iters | solve wall | s/iter | KKT ≤ 1e-4 first | within 1 % first |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NumPy | 1e-8 | optimal | **32.6476** | 104 | 59.7 s | 0.57 | 57 | 65 |
| CuPy (RTX 4070 Laptop) | 1e-8 | optimal | **32.6476** | 104 | 27.4 s | 0.26 | 57 | 65 |
| NumPy | 1e-4 | optimal | 34.53 | 65 | 37.8 s | 0.58 | 64 | — |
| CuPy (RTX 4070 Laptop) | 1e-4 | optimal | 34.53 | 65 | 17.0 s | 0.26 | 64 | — |

The NumPy and CuPy trajectories are identical to the last iteration.
Isolating the levers (NumPy, `1e-8`; the two starred rows were timed with a
diagnostic callback in the loop):

| arm | iters | s/iter | KKT ≤ 1e-4 first | within 1 % first |
| --- | ---: | ---: | ---: | ---: |
| Breedveld-parity recipe (above) | 104 | 0.57 | 57 | 65 |
| … with ipax's start (`slack_init_scale=0.1`) instead of §3.1 | 108 | 0.64\* | 60 | 68 |
| … with Breedveld's eq. (12) μ rule (`mu_schedule="breedveld"`) instead of probing | 225 | 0.77\* | 128 | 116 |
| previous recipe: `quality` μ + filter line search + ipax's start | 133 | 1.06 | 69 | 79 |

Three things to know when reading this against the paper's 79 iterations /
55 s (16-core 2017 Xeon, initialization included):

- **The tolerances do not transcribe.** Breedveld stops on the ℓ2 norm of
  his eq. (8) residuals, which include `γ_z = x − μZ⁻¹e`: the bound
  complementarity residual *divided by the multiplier*. On our iterates that
  term sits at 1e3–1e4 even at the certified `1e-8` point (a 1e-4
  complementarity error over multipliers of 1e-10), so his `< 1e-4` cannot
  be used as a stopping rule here. ipax's scaled ∞-norm at `1e-4` is *looser*
  than his test — it stops at 34.53, 5.6 % above the published objective —
  so the honest comparison is the `1e-8` row: 104 iterations against his 79,
  and 27 s (+4 s initialization) on the GPU against his 55 s.
- **His eq. (12) μ rule is not what his results used.** Inside ipax it drops
  μ to the floor right after the first full step at iteration 20 (`σ(1) = 0`)
  and then crawls for 190 iterations at `α ≈ 0.02`. The paper's runs are the
  Mehrotra variant, whose μ comes from the predictor step (Supplementary
  material) — that is `mu_schedule="probing"`, and it is what makes the
  recipe: 104 iterations, and 1.9× less wall per iteration than the `quality`
  oracle (which pays an extra solve to pick σ).
- **The §3.1 dual start is nearly a no-op on `Protons_01`**: every LSQR
  multiplier is floored to 5, so it amounts to `y₀ = w₀ = 5e`, `μ₀ = 154`,
  and saves 4 iterations over ipax's start. It costs nothing to keep (3 s),
  and it is the paper's setup.

**`Liver_01` under the same setup (NumPy, `1e-8`, idle machine) — the μ
schedule splits by site.** The reference solve: 36 iterations / 22 s.

| arm | status | objective | iters | wall | s/iter | KKT ≤ 1e-4 first | within 1 % first |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Breedveld setup, `mu_schedule="monotone"` (the runner default) | optimal | **−1.03707** | 74 | 80 s | 1.09 | 37 | 55 |
| … `mu_schedule="probing"` | optimal | **−1.03707** | 430 | 481 s | 1.12 | 312 | 214 |
| … `probing` with ipax's start instead of §3.1 | optimal | **−1.03707** | 644 | 647 s | 1.01 | 271 | — |
| … Breedveld's eq. (12) μ rule (`mu_schedule="breedveld"`) | optimal | **−1.03707** | 600 | 663 s | 1.10 | 268 | 335 |
| previous universal L-BFGS recipe (`slack_init_scale=0.1`, monotone, filter) | optimal | **−1.03707** | 211 | 9 min | 2.6 | — | — |

Under probing the nonconvex LTCP terms drive the objective into repeated
barrier-dominated excursions (peaks of 1e6, restoration-like θ jumps every 30
iterations) — the August finding about μ-raising oracles on Liver/CK/VMAT,
now with Mehrotra's predictor-based μ in the same family (his own eq. (12)
rule behaves the same way: 600 iterations). Monotone μ reaches
his tolerance class at iteration 37 against his 36. The §3.1 start helps here
as well (644 → 430 under probing). Hence the runner default is Breedveld's
setup with **monotone μ**, and `--mu-schedule probing` is the proton lever
(`Protons_01`: 104 vs 227 iterations). The two remaining differences to the
paper's per-case iteration counts are the μ dynamics inside the Mehrotra
variant (his Supplementary material; not transcribed) and the tolerance
measure discussed above.



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
| Protons_01 | 0.07% | no — folded by the 5% share gate (below) | 0.91 → 0.82 s/iter |
| Head-and-Neck_01 | 0% | no (correctly inert) | — |

Single-thread CPU, 16 iterations for VMAT and 21 for Protons, `auto` vs
`native`. VMAT is the mixed case the per-block reduction exists for: it is
96% float32 held back by a single float64 constraint matrix, so an
all-or-nothing rule reduced *nothing* there. VMAT's objective agrees with the
native run to 8 significant figures at identical KKT error (a trajectory
difference within the refinement certificate, not an accuracy loss).

**Share gate for the float32 group.** Protons was the honest cost of the
per-block rule: 45 of its 369 445 rows (0.07% of the nonzeros) are float32,
so the hinted group reduced nothing worth having, yet it put every dense solve
on the mixed-precision route, whose refinement matvecs run over the *whole*
15.6 M-nonzero block (37 solves per 15 iterations). The loader therefore
splits off a float32 group only when it holds at least 5% of the linear
block's nonzeros; below that its rows fold into the float64 group without a
hint. Their values are exact float64 widenings, so every product stays
exact, but the fold is not otherwise invisible: the block becomes one group,
so the lowered row order (and the multiplier order in `y_ineq`) is the single
group's, summation orders change, and the float32 structures join the float64
group's tiled plan and gate statistic. `TROTSExactProblem` plans a folded
structure that is also curved under the same (unhinted) key, so the linear
block and the Hessian share one tile set.
Measured on `Protons_01` (`exact` + `mehrotra`, 15 iterations, NumPy):
0.91 s/iter with the hinted group vs 0.82 s/iter folded (the same as forcing
`gram_dtype="native"`), identical iterations. `Prostate_BT_02` (100% float32)
and VMAT (96%) keep their split. `TROTSProblem(...,
float32_group_min_share=...)` sets the threshold (a share equal to it
splits); `0` always splits, `problem.float32_nnz_share` reports the share and
`problem.float32_group_folded` the decision. The single-thread rows of the
table above were measured before the gate.

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

**Device residency of the constant block.** The linear `lo ≤ A z ≤ hi` block
(dose constraints + minimax links) is by far the largest array in an RT problem —
86 M nonzeros on `Prostate_VMAT_101`. Since 2026-09 the loader declares it
through the public `Problem.linear_ineq()` (as a `VStack` of the float32- and
float64-sourced row groups, so `gram_dtype="auto"` still reduces per group) and
ipax lowers it to one-sided rows at the triplet level; earlier the loader
hand-lowered it to `G z + h ≤ 0` and appended it to `ineq_constraints`, because
the `linear_ineq` lowering used to densify. The loader keeps only the host CSR
groups and builds the declared operator fresh per `linear_ineq()` call, so once
the lowering has consumed it the device holds one copy: the lowered operator's
Array-API COO triplets (24 B/nnz) plus the adapter's device CSR (12 B/nnz) —
invariant #4's cost, paid once, the same steady-state accounting as the
hand-lowered layout, which measured 3057 MiB in the CuPy pool on
`Prostate_VMAT_101` (4040 MiB before an earlier change stopped mirroring `G` a
second time). The migrated layout has **not** been re-measured on device, and
the one-time lowering transiently holds the declared and the lowered triplets
together. The Head-and-Neck saturation above was measured before both changes.
The lowered row order also differs from the old hand-lowering (per precision
group, finite-lower rows now precede finite-upper rows); the constraint set is
identical.

Two measurement notes for anyone repeating this. The sparse adapters memoize
the Gram per `accumulate_dtype`, keyed on weight equality (it serves `δ_w`
retries within an iteration), so a timing loop reusing one weight vector
measures a cache hit rather than the kernel. And on a laptop GPU, arms run
back-to-back drift enough to invert the result — interleave the variants and
log the clocks.

### Tiled per-structure Gram (loader-side, default)

The condensed dense route forms `N = W + AᵀΣA` once per iteration, and on
the proton cases the linear block's Gram *is* the iteration: 2.3 s of a 2.8 s
step on `Protons_01`. The generic SciPy adapter accumulates it by chunked
`syrk` over all `n` columns, yet a block of voxel rows only reaches the
beamlets that hit those voxels. Following Breedveld, van den Berg & Heijmen
(2017, *Comput. Optim. Appl.* 68:209–242, §4.2 eq. (48) and §5.2), the loader
now declares each precision group of the linear block as a
`TiledGramCSROperator` — a `CSROperator` subclass whose `gram` is a
per-structure tiled product. The rest of the operator (matvec family, COO,
row norms) is the plain one: same rows, same order, same bounds.

- **Per-structure plan (built once per `linear_ineq()` call).** Rows are
  reordered by reverse Cuthill–McKee on the bipartite graph
  `[[0, A], [Aᵀ, 0]]` and cut into tiles of 2048 rows. Each tile is compacted
  to its active columns (`Protons_01`: median width 300 of 1101).
- **Weight merging (eq. 48).** Row sets that copy the same structure matrix,
  such as a min/max constraint pair or a constraint reused as a minimax link,
  are multiplied once as `A_sᵀ diag(Σ w) A_s`. The minimax auxiliary column
  adds only rank-1 borders (`Aᵀw` matvecs). The core supports this through its
  subclass lowering. It keeps a `CSROperator` subclass intact and hands it
  per-row summed weights, which merges the lower and upper halves of a range row
  the same way.
- **Kernels.** Each dense tile is densified, scaled by `√w`, then passed to
  `dsyrk` (`ssyrk` for `gram_dtype` float32 requests, with float64
  accumulation). It accumulates into one triangle, and the result is mirrored
  once at the end, so the Gram is exactly symmetric. Tiles with negative or
  non-finite merged weights fall back to `Dᵀ(w∘D)`. A tile uses SciPy's sparse
  product when its `Σ nnz_row²` is at most 1% of `b·c²`: SpGEMM runs about
  100× below BLAS rate, and per-tile timings on four cases put the break-even
  there. An in-support *density* rule was tried first. Every cutoff was about
  optimal on some case and cost 2× on another; for example, 0.15 is good on
  Protons but a loss on CK.
- **Threads (§5.2).** Tiles are dealt by flops to 8 accumulation lanes, each
  with a private accumulator, served by a pool of `min(8, cores)` workers. The
  lane count depends only on the configuration and the data, so the result's
  last bits do not change with the machine. Only as many lanes are used as
  there is work for, within a 1 GiB accumulator budget, so `n ≈ 10⁴` runs
  sequentially. BLAS keeps its own threads. Setting `OPENBLAS_NUM_THREADS=1`
  made the pool **slower** (0.45 s vs 0.30 s), because the compacted scatter
  holds the GIL.
- **Gate.** A group is tiled only when the plan's arithmetic
  `Σ_r b_r·c_r²` is ≤ 50% of the plain `m·n²`. A lower bound
  (`Σ nnz_row²`) is checked first, which rejects dense groups before any RCM
  work. A gated-off group goes to the core as the plain `CSROperator`, exactly
  the untiled path. The gate is the same on every namespace; JAX and
  `array_api_strict` always take the plain path (see *Device namespaces*
  below for CuPy and Torch).

`TROTSProblem(..., tiled_gram=False)` restores the plain block for A/B runs.
Use it when a measurement is meant to track the *generic* core Gram on RT
data, because the default now times this loader kernel.
`tiled_gram=TiledGramConfig(...)` exposes the knobs (`block_rows`,
`flop_ratio_threshold`, `sparse_tile_flop_ratio`, `threads`, and the lane
heuristics).

**Measured** (2026-09-25, NumPy, 32-thread desktop, threaded OpenBLAS, fresh
random weights per call because both paths memoize on identical weights; min
of 3 per arm, one process per arm). This is the Gram of the whole lowered
`Protons_01` block (369 445 rows, `n = 1101`):

| arm | B | sparse tiles | threads | Gram wall |
| --- | ---: | --- | ---: | ---: |
| plain adapter | — | — | BLAS | 2.315 s |
| tiled | 1024 | off | 1 | 0.669 s |
| tiled | 1024 | ≤1% | 1 | 0.649 s |
| tiled | 1024 | off | 8 | 0.392 s |
| tiled | 1024 | ≤1% | 8 | 0.389 s |
| tiled | 2048 | off | 1 | 0.572 s |
| tiled | 2048 | ≤1% | 1 | 0.540 s |
| tiled | 2048 | off | 8 | 0.357 s |
| **tiled (default)** | **2048** | **≤1%** | **8** | **0.326 s** |

That is 7.1× on the Gram, agreeing with the plain path to 3e-16 relative and
exactly symmetric. The sequential phase split at the default tile size is
syrk 0.21 s (107 GFLOP/s on 22 GFLOP of compacted tiles), densify 0.10 s,
compacted scatter 0.08 s, sparse tiles 0.07 s, and √w scaling 0.03 s.

End to end (`hessian="lbfgs"`, `linsolve="dense"`,
`BarrierOptions(slack_init_scale=0.1)`, 35 iterations, one process per arm,
interleaved):

| case | plain s/iter | tiled s/iter | trajectory |
| --- | ---: | ---: | --- |
| Protons_01 | 2.77, 3.22 | 0.66, 0.70 | same path; objective agrees to 3e-10 |
| Prostate_VMAT_101 (10 it.) | 2.58, 2.57 | 2.55, 2.55 | same path; objective equal to 10 digits |

VMAT is the gate's check. Its float32 group (81% of rows, 64% dense) fails
the pre-gate and stays plain. Only the small float64 group, which holds the
smoothing pair and one minimax ring, is tiled, so the per-iteration cost is
unchanged. Per group, on the other cases (Gram wall, plain → tiled): Liver_01
float64 group 0.74 → 0.20 s, Prostate_CK_01 float64 group 1.08 → 0.58 s, and
Head-and-Neck_01 (`n = 9984`, one lane by the memory budget) 29.0 → 20.7 s.
Every group agrees with the plain Gram to ≤ 1e-15 relative.

Plan cost per `linear_ineq()` call:

| case | groups tiled | plan build | `linear_ineq()` tiled vs plain |
| --- | --- | ---: | ---: |
| Protons_01 | both (one group since the share gate) | 1.06 s | 2.1 s vs 1.7 s |
| Prostate_VMAT_101 | float64 only (float32 pre-gated) | 0.21 s | 4.2 s vs 5.1 s |
| Prostate_CK_01 | float64 only (float32 gated after planning: ratio 0.68) | 0.41 + 2.31 s | 5.3 s vs 3.4 s |
| Liver_01 | float64 only (float32 pre-gated) | 0.25 s | 2.1 s vs 2.3 s |
| Prostate_BT_02 | none (pre-gated, ratio 1.0) | 0 s | 0.02 s vs 0.02 s |
| Head-and-Neck_01 | its single float64 group | 6.19 s | 11.6 s vs 8.6 s |

The plan is rebuilt for each solve, since `linear_ineq()` builds its operator
fresh. Its cost is a few seconds against solves of minutes to hours. It holds
one permuted copy of each tiled structure's values and compact indices,
about 12 B/nnz (8 B/nnz for float32-sourced data), in addition to the operator
itself. Tiles are stored in float32 only when a structure's values are exact
float32 widenings (checked once per structure when it is planned, like the
matrix cache writer does), because the same tiles also serve exact float64
Gram and Hessian requests. A structure that is not float32-exact is stored in
float64 even under a float32 hint; the hint then only selects the reduced
`gram_dtype` accumulate, which rounds the tile at product time as the plain
adapter would.

**Device namespaces (CuPy / Torch).** The plan is host index/value data, so
the same plan drives a device kernel. On a CuPy or Torch namespace the loader
declares the same `TiledGramCSROperator`, and its `gram` runs in the
namespace on the operator's device:

- **Device tiles, materialized once.** Per structure, the RCM order and, per
  tile, a flat dense-scatter index `row·c + col` (int32) and the tile values
  in the plan's storage dtype, concatenated per structure, plus each tile's
  column set. That is about 12 B/nnz (8 B/nnz for float32-stored data), the
  host plan's footprint; no second copy of the block is kept beyond the
  operator's own. `Protons_01`: 195 MiB for 34 structures. A per-problem
  cache keyed by the plan's tile list shares one device copy between the
  operators `linear_ineq()` builds and the exact Hessian. On device
  namespaces the plan registry is always on, so plans survive
  `linear_ineq()` calls.
- **Kernel per tile.** Scatter the values into a zeroed `b×c` tile, scale the
  rows by `√w`, and form the `c×c` product. CuPy uses `cupy.cublas.syrk`
  (lower triangle, as the CuPy adapter does). Torch has no public `syrk`, so it
  uses `matmul` (`Dᵀ D`) at twice the flops. Tiles with signed or non-finite
  merged weights use `Dᵀ(w∘D)`. The weight test is one reduction per
  structure, read back once per Gram. The block is scatter-added at
  `ix_(cols, cols)`: `cupy.add.at` on CuPy (one kernel, 2.4× faster than
  fancy `+=`) and `index_put_(accumulate=True)` on Torch. The column sets are
  unique, so no index collides and the result is deterministic.
  The triangle is mirrored once, in blocked panels, so the result is exactly
  symmetric. There are no lanes: tiles run in plan order on the stream.
- **Sparse tiles.** On CuPy, exact Grams multiply them with cuSPARSE SpGEMM
  (`TiledGramConfig(device_sparse_tiles=False)` densifies them instead).
  float32 requests and Torch densify them. Measured on `Protons_01` (17
  sparse tiles): SpGEMM 0.216 vs densified 0.240 s in float64, densified
  0.056 vs SpGEMM 0.067 s in float32. Densifying is priced wrongly by the
  gate, which charges a sparse tile its sparse arithmetic `Σ nnz_row²`; a
  densified tile costs `b·c²` (twice that with Torch's `matmul`) and a `b×c`
  temporary, so on Torch and for float32 requests the gate statistic
  underestimates the device cost. `TiledGramConfig(device_densify_budget_bytes=...)`
  (default 256 MiB for the float64 `b×c` temporary) caps it: a sparse tile
  over the budget is never densified. CuPy multiplies it by SpGEMM at every
  precision, Torch by the host SciPy product of the plan's tile. No TROTS
  tile comes near the default (`2048 × n × 8 B` is 164 MB at `n = 10⁴`).
- **Rest of the operator.** The minimax borders come from one transpose
  product of the operator with the weights masked per aux segment. Matvecs
  and COO are the plain `CSROperator`'s, so the row order is the original
  one. `accumulate_dtype="float32"` runs `ssyrk`/float32 `matmul` per tile and
  accumulates in float64. The weight memo compares device arrays; the
  comparison is read back together with the weight flags.
- **Host synchronizations.** One host read per Gram call (the per-tile weight
  flags, with the memo comparison folded in when the memo slot is filled; a
  memo hit costs the same one read). Exact CuPy Grams add at least one sync
  per sparse tile inside cuSPARSE SpGEMM, which reads the product's nnz back
  to size its output; cupyx offers no way around it. The scaled operand is
  flagged canonical and the stored transpose is float64, so cupyx neither
  re-checks nor recasts them per call. Torch's over-budget sparse tiles add
  one weight download each (free on CPU tensors).
- **Host plan retained.** The host plan (tile CSRs, about 12 B/nnz) stays in
  memory next to the device copy: it keys the plan registry and the device
  cache, and it is what Torch's over-budget tiles multiply.

**Measured on the GPU** (2026-09-26, RTX 4070 Laptop, CuPy 14, fp64 at 1/64
rate; `Protons_01` lowered block, 369 445 × 1101). Gram wall, fresh weights,
one warm-up then 3 samples per arm, arms interleaved in one process,
`float32` forced at the operator level
(`gram(w, accumulate_dtype="float32")`, as `DenseOptions(gram_dtype="float32")`
requests it):

| arm | float64 min / median | float32 min / median |
| --- | ---: | ---: |
| plain CuPy adapter (dense-chunk `syrk`, block density 5.05%) | 2.907 / 2.907 s | 0.193 / 0.193 s |
| tiled, sparse tiles by SpGEMM (default) | **0.216 / 0.216 s** | 0.057 / 0.058 s |
| tiled, sparse tiles densified | 0.240 / 0.240 s | 0.055 / 0.056 s |

That is 13.5× in float64 and 3.4× in float32. The float64 tiled Gram
agrees with the plain one to 1.1e-14 relative. The float32 tiled Gram is
within 1.2e-11 of the float64 one, and the plain float32 Gram within 3.8e-6.
The tiled kernel accumulates the minimax borders in float64 and rounds only
the compacted tiles. A phase split of the float64 tiled Gram gives `dsyrk`
0.16 s, or 134 GFLOP/s on the 22 GFLOP of compacted tiles (counted as for
the CPU split above). That is on the order of this card's 1/64-rate fp64 peak. Densify,
scale, scatter and sparse tiles add 0.03 s, and weights, flags and borders
add the rest. On this card the float64 tiled Gram is compute-bound on fp64.

End to end (`hessian="lbfgs"`, `linsolve="dense"`,
`BarrierOptions(slack_init_scale=0.1)`, `max_iter=35`, which stops at
36 iterations; one process per arm, run plain, tiled, plain, tiled):

| arm | `gram_dtype` | s/iter | iterations | objective at stop | CuPy pool |
| --- | --- | ---: | ---: | --- | ---: |
| plain | auto (native: no float32 group) | 2.870, 2.868 | 36, 36 | 3126.43469170, 3126.43469190 | 3115 MiB |
| tiled | auto | **0.289, 0.288** | 36, 36 | 3126.43469183, 3126.43469178 | 1980 MiB |
| plain | float32 | 0.501 | 36 | 2599.914 | 3687 MiB |
| tiled | float32 | 0.170 | 36 | 3126.282 | 1980 MiB |

That is 9.9× per iteration at the default. The trajectory is the same, and
the objectives agree to 6e-11 relative, the same spread as between the two
plain runs. The laptop GPU iteration is now about 2.3× faster than the
CPU tiled iteration above (0.66–0.70 s). The forced float32 arms leave the native
trajectory and each other, so compare their per-iteration cost only. The
plain adapter's float32 Gram is 25× less accurate on this block (above), and
refinement certifies each step regardless.

**Torch.** The same code runs on Torch tensors. It is validated on CPU
tensors (Gram agrees to ≤ 1e-14, solves match the plain operator). There,
`Protons_01` takes 0.50 s tiled vs 2.43 s for the plain Torch-CPU Gram (the
routed SciPy adapter). This is slower than the NumPy kernel (0.32 s) because
it uses `matmul` instead of `syrk` and has no lanes. **Torch-CUDA is
unsupported and untested** (no CUDA build of Torch in the measuring
environment): the kernel places every array on the operator's device, but the
loader builds its operators with the namespace's default device and its
problem callbacks bridge through NumPy.

**The NumPy path is unchanged.** Re-measured in the same session (same
script, NumPy): plain 2.334 s, tiled 0.317 s (7.4×; 0.326 s in the table
above).

The kernel **does not** change the solver's trajectory beyond round-off. It
also does not help, and so is not used, where the dose data is dense in its
support, which covers the float32 VMAT/CK/Liver groups and brachytherapy. This
is RT-specific structure exploitation, so it stays in the loader (the
application edge). The `ipax` core only provides the generic `gram` hook it
plugs into (see the scope notes in `AGENTS.md`).

### Grouped exact Hessian (loader-side, `hessian="exact"`)

Every TROTS cost function is, in dose space, a diagonal plus at most a rank-1
term. In beamlet space that is
`∇²h = c1·Aᵀdiag(e2)A + c2·(Aᵀe1)(Aᵀe1)ᵀ` (van Haveren & Breedveld 2019,
*Phys. Med. Biol.* 64:16NT01, eqs. (8)–(9), with the corrigendum 65:079501).
LTCP, the smoothed DVH and the quadratic penalties are purely diagonal. gEUD
adds the rank-1 term, with a negative coefficient for `a > 1` when minimised.
The mean (gEUD with `a = 1`) has no curvature. The loader keeps its own DVH
second derivative, which is correct where the paper's Table B1 is misprinted.

`TROTSExactProblem.lagrangian_hessian` therefore returns a
`TROTSExactHessian` operator,
`H_x = Σ_k A_kᵀ diag(w_k) A_k + Q + U Vᵀ`, zero-padded for the minimax
variables:

- **One product per unique structure** (Breedveld et al. 2017, eq. (48)). The
  dose `d_k = A_k x + b` is computed once per structure, and the dose-space
  diagonals of all its terms are summed into one signed weight vector `w_k`.
  The objective terms carry `σ·factor·weight`, the constraint rows their
  multiplier times sign. `Prostate_CK_01` has two LTCP terms on one PTV
  matrix, so that product is formed once instead of twice.
- **Tiled kernels.** Each curved structure is planned once, at construction,
  with the linear block's machinery and its gate applied per structure. The
  weights are signed in general, because DVH curvature is indefinite and
  maximised terms flip sign, so tiles with a negative weight take the `gemm`
  fallback. A structure the gate rejects (the dense-ish CK PTV/PZ, for
  example) is multiplied in full-width dense row chunks on the same lanes,
  not by SciPy SpGEMM. Structures that also appear in the linear block share
  one tile copy through a per-problem plan registry.
- **Rank-1 terms.** The gEUD terms become `U = [A_kᵀu_t]`,
  `V = [mult_t·c2_t·A_kᵀu_t]` as an `ipax.backend.operators.LowRank`. No
  dense outer product is wrapped in a sparse matrix.
- **Quadratics.** The beamlet-space terms (`ctype 2` objectives and smoothing
  constraints) are symmetrised once. Each call only re-weights their sum.
- **Consumers.** `dense_matrix` (the condensed dense route) forms the tiled
  products, mirrors one triangle so the result is exactly symmetric, and is
  memoized per operator, so δ_w retries do not repeat it. `matvec`/`matmat`
  are exact float64 products (`Σ A_kᵀ(w_k ∘ A_k v) + Qv + U(Vᵀv)`) for Krylov
  and for the mixed-precision refinement. `diagonal` serves the Jacobi
  preconditioner. `to_coo` feeds the sparse-direct route through the plain
  SciPy product, grouped per structure.

**Namespaces.** The `TROTSExactHessian` operator is host-resident, so it is
returned as-is only on NumPy namespaces. On CuPy and Torch the curved
structures are planned as on NumPy and, when at least one is tiled,
`lagrangian_hessian` returns a `TROTSDeviceExactHessian`: `dense_matrix` runs
the tiled structures through the device tiled kernel (sharing the linear
block's device tiles), crosses the gated-off structures' SciPy products plus
`Q` once as a lower-triangle COO, adds `U Vᵀ` by a device matmul and mirrors
one triangle. Its `matvec` is exact float64 and device-side on CuPy (the
problem's device-mirrored structure matrices); on Torch every product is a
host round trip (download, SciPy products, upload onto the operator's
device). On JAX, or with `tiled_gram=False`, the
grouped Hessian is formed once per call as host sparse (one SciPy SpGEMM per
structure) and crossed to the namespace as a `CSCOperator`, like the per-term
path, so Krylov matvecs and the dense route still run on the namespace's own
arrays. `array_api_strict` has no sparse adapter to apply a `CSCOperator`, so
it keeps the host operator.
`hessian_stats["build_seconds"]` times `lagrangian_hessian` itself, which is
the whole sparse product on the per-term and crossed paths but only the
weights on the NumPy grouped path, whose products land in `dense_seconds`;
compare paths on the sum.

`TROTSExactProblem(..., grouped_hessian=False)` keeps the original per-term
path (one SciPy SpGEMM `Aᵀ diag(h) A` per term, a dense `np.outer` per gEUD
term) for parity tests. `tiled_gram=False` turns off tiling for both the
linear block and the Hessian; the Hessian is then grouped but uses dense row
chunks for every structure. The grouped and per-term Hessians agree to
≤ 1e-10 relative on Liver_01 and Prostate_CK_01 at the warm start with random
multipliers (integration test). `Prostate_BT_02` solves to the same status,
route and objective on `exact/dense`.

The cost functions also have guards at their singular points. The gEUD powers
with a negative exponent (`|d|^(a−2)` for `a < 2`) are evaluated at
`max(|d|, floor)`, so the mean no longer returns `0·∞ = nan` at zero dose. The
floor is 1e-12, raised for exponents below about −25 to `1e300^(−1/|e|)` so
that the floored power stays finite for every finite `a`. A negative exponent
with a voxel at zero dose makes `S = mean(d^a)` infinite; the gEUD is then 0
and the gradient and curvature are returned as 0. An all-zero structure dose
(`S = 0`) likewise returns a zero gradient and zero curvature instead of a
Python `ZeroDivisionError`. The LTCP exponent is clipped at 500; below the
clip the value is constant while the returned derivatives are not, so a
finite-difference check would flag that (never normal) region. Away from those
points the values are bitwise unchanged (tested).

**Measured** (2026-09-25, NumPy, development machine, one process per
arm, fresh multipliers per call). The machine ran about 3× slower than for
the tiled-Gram numbers above: the CPU reported 2.2 GHz, and `Protons_01`
`lbfgs/dense` took 1.94 s/iter against 0.66 s/iter there. The per-term
`Prostate_CK_01` Hessian took 48 s in a first probe and 286 s later that day
with unchanged code. Compare arms within this table only.

One `lagrangian_hessian(...)` plus `dense_matrix()` call, median of 3
(per-term CK: 2 calls):

| case | curved terms → structures | per-term | grouped | speed-up |
| --- | --- | ---: | ---: | ---: |
| Protons_01 | none (LP + minimax) | 1.3 ms | 3.8 ms | — (H ≡ 0) |
| Prostate_BT_02 | 7 → 4 | 69 ms | 17 ms | 4× |
| Liver_01 | 3 DVH + 15 quadratic → 3 | 6.5 s | 0.52 s | 12× |
| Prostate_CK_01 | 3 LTCP + 25 quadratic → 2 | 287–300 s | 1.27–1.47 s | ~210× |

End to end (`linsolve="dense"`, `BarrierOptions(slack_init_scale=0.1)`,
grouped Hessian; objective / scaled KKT error from `Result.history`):

| case | iters | `lbfgs` s/iter | `exact` s/iter | exact: Hessian share | exact: Gram share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Protons_01 | 60 | 1.94 | 3.01 | 0.5% (H ≡ 0) | 29% |
| Prostate_CK_01 | 10 | 9.39 | 9.80 | 11% | 35% |
| Liver_01 | 10 | 3.57 | 3.97 | 12% | 23% |

| Protons_01 | it. 20 obj / KKT | it. 40 obj / KKT | it. 60 obj / KKT |
| --- | --- | --- | --- |
| `lbfgs` | 303.1 / 8.0 | 843.8 / 9.8e-3 | 60.95 / 3.5e-4 |
| `exact` | 243.9 / 13.6 | 92.34 / 9.8e-4 | 47.05 / 3.7e-4 |

With the grouping, the exact Hessian costs about 10% of an iteration on the
curved cases. Before it, the Hessian was the whole CK iteration by two orders
of magnitude. `Protons_01` is a pure LP with minimax terms, so its exact
Hessian is identically zero and grouping cannot help there. Its extra
`exact` cost (3.0 vs 1.9 s/iter) comes from the solver's step-solve path with
a zero `W`, not from the Hessian (14 ms per iteration). It reaches a lower
objective at iteration 60 with the same KKT level.

A/B runs from the command line: `--per-term-hessian` selects the per-term
path on the `exact/*` configs, and `--plain-gram` passes `tiled_gram=False`:

```bash
IPAX_TROTS_DIR=/path/to/TROTS python -m benchmarks.runners.trots --solve \
    --cases Prostate_CK_01 --config exact/dense --init ipax \
    --mu-schedule monotone --globalization filter --corrections none \
    --slack-init-scale 0.1 --max-iter 10 [--per-term-hessian] [--plain-gram]
```

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
    The same holds for `"probing"` under the Breedveld-parity setup
    (`Liver_01`: 430 iterations vs 74 with monotone μ, see *Measured*
    *results*), which is why the runner's default μ schedule is monotone
    and `--mu-schedule probing` is a proton-only flag.

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
(~0.5–0.8 ms) is the direct solve's verification round: the extra
operator matvec and its bookkeeping, rather than its scalar host reads.
A follow-up microbenchmark (2026-09-02) measured a CuPy scalar sync at
about 14 µs on this stack.

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
  threads (measured before the loader's tiled Gram, which cuts that case's
  Gram from 29 s to 21 s) — a practical-budget scale limit of the generic
  routes. RT-specific dose-matrix kernels stay out of the `ipax` core (see
  `AGENTS.md`). The loader is the RT-specific layer: its tiled Gram above
  (including the float32 `ssyrk` tiles for float32-sourced data) is where that
  structure is exploited, while custom mixed-precision tiled BLAS stays out of
  scope *for the core*. The *generic*
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
# Breedveld-parity recipe = the runner defaults (exact/dense, mehrotra,
# breedveld globalization, monotone μ, --init breedveld, --tol 1e-8):
IPAX_TROTS_DIR=/path/to/TROTS python -m benchmarks.runners.trots --solve \
    --cases Liver_01 --max-iter 300 --max-time 900

# Proton cases: the same with Mehrotra's predictor-based μ:
IPAX_TROTS_DIR=/path/to/TROTS python -m benchmarks.runners.trots --solve \
    --cases Protons_01 --mu-schedule probing --max-iter 300 --max-time 900

# Previous universal L-BFGS recipe (the pre-2026-09-26 runner defaults +
# slack_init_scale=0.1):
IPAX_TROTS_DIR=/path/to/TROTS python -m benchmarks.runners.trots --solve \
    --cases Liver_01 --config lbfgs/dense --init ipax \
    --mu-schedule monotone --globalization filter --corrections none \
    --slack-init-scale 0.1 --max-iter 800 --max-time 1800

# Previous proton stack:
IPAX_TROTS_DIR=/path/to/TROTS python -m benchmarks.runners.trots --solve \
    --cases Protons_01 --config lbfgs/dense --init ipax \
    --slack-init-scale 0.1 --kappa-centrality 1e-4 \
    --mu-schedule quality --globalization breedveld --corrections none \
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
