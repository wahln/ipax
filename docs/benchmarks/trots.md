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
    mu_schedule="probing",  # Mehrotra's predictor-based μ …
    globalization="breedveld",  # ratio control, then the Markov filter
    barrier=BarrierOptions(mu_init=mu0, kappa_centrality=0.0),  # … floor off
    max_iter=300,
)
result = ipax.solve(problem, x0, options=opts, warm_start=warm)
```

`Protons_01`: 93 iterations, 24 s on CuPy including the LSQR dual start (the
reference solve: 79 iterations / 55 s); `Liver_01`: 57 iterations / 58 s on
CuPy (reference 36 / 22 s). The 2026-09-27 sweep below measured this recipe
against the monotone-μ variant on all seven TROTS sites; it wins every case.
Its times predate the runner counting the least-squares fluence of eq. (23)
as initialization, so add that solve to each row: since 2026-09-28 it costs
0.3 s on `Protons_01`, 0.1 s on `Liver_01`, 0.2 s on `Prostate_VMAT_101`,
0.3 s on `Prostate_CK_01` and 3.5 s on `Head-and-Neck_01` (n ≈ 10⁴). The
normal matrix is now accumulated per initialization matrix by the SciPy
adapter's dense-BLAS Gram and solved by Cholesky; the SciPy SpGEMM +
`todense` + LU it replaced took 1.3 / 5.6 / 19 / 35 / 22 s on the same
cases. The fluence agrees with the old formulation to ≤ 3.3e-11 relative
(max-norm; ≤ 7e-8 absolute), and no entry changes sides of the clip at 0.
Timings are best of two on a shared machine (provisional; the old path
varied by up to 3× between runs).

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
then; the rows below use `--mu-schedule probing` with ipax's centrality floor
still on (`kappa_centrality=1e-2`), the runner default until the sweep of
2026-09-27 turned the floor off.**
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
in full: the least-squares fluence of eq. (23) — 0.3 s here, 0.1–0.3 s on
`Liver_01`, `Prostate_VMAT_101` and `Prostate_CK_01` (a dense-BLAS Gram of
the dataset's initialization matrices since 2026-09-28; the SciPy SpGEMM it
replaced took 1–35 s, see above) — and LSQR on the
host (3.4 s at the loader's `atol=btol=1e-6`, 24 s at `1e-8` for a
bit-identical start, because every LSQR multiplier is < 0.04 and the floor
sets all 369,445 of them to 5). On CuPy the LSQR has since moved to the
GPU; see *Device LSQR* below:

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

**Device LSQR (CuPy, 2026-09-27).** On a CuPy namespace,
`breedveld_warm_start` now runs eq. (26) on the GPU. It uses `_lsqr` in
`benchmarks/corpus/trots.py`, a transcription of SciPy's Paige–Saunders LSQR
into the array namespace with the same recurrences, stopping tests and
`istop` codes. It runs on the problem's own lowered operator, so no NumPy
host twin is built. cupyx has no drop-in replacement: its `lsqr` is a direct
sparse-QR solve of a *square* CSR system. `lsqr="host"` keeps the SciPy path,
which Torch, JAX and `array_api_strict` always use.

The two paths give the same start to round-off. On `Protons_01` the relative
differences are 1e-14 in the slacks, 5e-16 in the bound multipliers and 2e-15
in `μ₀`, and `y₀` is on the floor in both. The unfloored LSQR iterate follows
SciPy's to 1e-12 over the first 10 iterations and then drifts, ending
8e-5 to 5e-4 apart in relative norm. That drift is LSQR's own sensitivity:
Golub–Kahan bidiagonalization loses orthogonality and amplifies round-off.
Perturbing SciPy's own matvec by 1e-16 relative moves its final iterate by
8e-5. Both runs stop on the same test at residual norms within 1 %.

`Protons_01`, RTX 4070 Laptop, fresh problem per sample. The CPU was at 100 %
load from other jobs, which inflates the host numbers (the idle figure is the
3.4 s above):

| arm | initialization | next 3 iterations | total |
| --- | ---: | ---: | ---: |
| host LSQR (SciPy on the host twin) | 14.4–16.7 s | 5.1–6.2 s | 19.2–23.1 s |
| device LSQR (`lsqr="auto"` on CuPy) | 3.5–4.1 s | 2.1 s | 5.5–6.2 s |

The device initialization includes about 3 s of tile-plan building for the
linear block. The solve reuses those plans, which is why its first iterations
get faster. Once the plans exist, the device initialization takes 0.8 s.

**`Liver_01` under the same setup (NumPy, `1e-8`, idle machine) — the μ
schedule splits by site.** The reference solve: 36 iterations / 22 s.

| arm | status | objective | iters | wall | s/iter | KKT ≤ 1e-4 first | within 1 % first |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Breedveld setup, `mu_schedule="monotone"` (the runner default until 2026-09-27) | optimal | **−1.03707** | 74 | 80 s | 1.09 | 37 | 55 |
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
as well (644 → 430 under probing). At this point the runner default became
Breedveld's setup with **monotone μ**, with `--mu-schedule probing` as the
proton lever (`Protons_01`: 104 vs 227 iterations) — superseded the next day
by the centrality-floor finding and the sweep below. The two remaining differences to the
paper's per-case iteration counts are the μ dynamics inside the Mehrotra
variant (his Supplementary material; not transcribed) and the tolerance
measure discussed above.

**Where the Mehrotra corrector's time goes (2026-09-27).** Stage-timed on
`Protons_01` (exact Hessian, dense route, Breedveld start, 40 iterations): the
corrector never rebuilds or refactors N. One tiled Gram (~0.32 s, 52–60 % of
the iteration) and one Cholesky are shared by the predictor and the corrector.
The corrector then pays exactly one Jᵀ·v for its right-hand side, one J·v for
the Δs/Δλ recovery (about 14 ms each over the 369,445 rows) and a back-solve
of the kept factor (under 1 ms). The rest was O(m) vector work — fraction-to-
boundary sweeps, target vectors, and a second evaluation of the barrier
directional derivative for the same step — now shared or skipped,
bit-identically, as is one Jᵀ pass per monotone-μ KKT test on every path. The
corrector's measured overhead is about 76 ms per iteration under monotone μ
(0.53 → 0.61 s); under `probing` the probe's own affine solve adds ~80 ms.
Re-timed on an idle machine after the reuse commit, the probing recipe on
`Protons_01` goes from 59.7 s to **56.0 s** on NumPy (0.54 s/iter) and from
27.4 s to **25.4 s** on CuPy (0.245 s/iter), at identical iteration counts.

**Breedveld's Mehrotra σ, his Gondzio variant, and the centrality floor
(2026-09-27).** With the supplement in hand (A.1–A.2), both of his
higher-order rules are available as opt-ins and neither moves the counts.
`BarrierOptions(probing_sigma="mehrotra")` is his σ = (μ_aff/μ)³ with μ the
measured gap of eq. (10) and no clip at σ ≤ 1: on `Protons_01`/`Liver_01` σ
exceeds 1 on one oracle call in ~250, so it is the `probing` rule with a
τ = 0.995 probe (103 / 441 iterations vs 104 / 430). Reading μ_prev as the
previous barrier *target* instead makes the rule μₖ₊₁ = μ_aff³/μₖ², which
multiplies any target/complementarity mismatch by −2 per iteration in log
space and diverges (tested: Protons > 293, Liver 736). His Gondzio loop
(`CorrectionsOptions(gondzio_variant="breedveld")`, A.2) is neutral on
`Protons_01` (103). What the same runs did locate is the Liver pathology
under probing: ipax's El-Bakry centrality floor `μ ≥ κ_cent·infeasibility`
(`kappa_centrality=1e-2`) binds on 36 of 37 oracle calls, the KKT-error
monitor suspends free mode from iteration 10, and the floor inside the
monotone re-entry then *raises* μ again and again (55, 12, 14, 3.2, 2.6, …) —
the 1e6 objective excursions above. With `kappa_centrality=0`:

| arm (Breedveld start, exact/dense, mehrotra, breedveld globalization, 1e-8) | Protons_01 | Liver_01 |
| --- | ---: | ---: |
| probing, default `kappa_centrality=1e-2` | 104 | 430 |
| probing, `kappa_centrality=0` | 95 (50.8 s NumPy, 23.2 s CuPy) | **57** (60.7 s NumPy) |
| probing + `probing_sigma="mehrotra"`, `kappa_centrality=0` | 95 | **50** |
| monotone (the runner default) | 227 | 74 |
| Breedveld's reference | 79 | 36 |

Both `kappa_centrality=0` rows certify the reference objectives.

**Corpus sweep (2026-09-27) — `probing` + `kappa_centrality=0` is now the
runner default.** Breedveld setup (exact Hessian, Mehrotra, ratio control,
§3.1 start, `1e-8`, budgets 800 iterations / 30 min, one process per case),
the previous default (monotone μ) against the candidate, two cases per site
plus the BT pair. CuPy where the case ran on the GPU before the device
exact-Hessian fix below; the VMAT and Head-and-Neck L-BFGS rows were run
when the exact arm's 30-minute cap on VMAT looked like a per-iteration cost
(it was not, see the exact rows and the note below). The "paper" column is
the supplement's Mehrotra run on the 2017 16-core machine.

| case | paper | monotone μ (previous default) | probing μ, κ_cent = 0 (default) |
| --- | ---: | --- | --- |
| Protons_01 | 79 it, 55 s | acceptable, 214 it, 53 s (CuPy) | acceptable, **93 it, 24 s** (CuPy) |
| Protons_02 | 65 it, 67 s | optimal, 154 it, 104 s (CuPy) | optimal, **80 it, 55 s** (CuPy) |
| Liver_01 | 36 it, 22 s | optimal, 74 it, 74 s (CuPy) | optimal, **57 it, 58 s** (CuPy) |
| Liver_02 | 43 it, 23 s | optimal, 72 it, 76 s (CuPy) | optimal, **39 it, 33 s** (CuPy) |
| Prostate_CK_01 | 46 it, 41 s | optimal, 87 it, 258 s (NumPy) | optimal, **39 it, 127 s** (NumPy) |
| Prostate_CK_02 | 39 it, 46 s | optimal, 90 it, 386 s (NumPy) | optimal, **48 it, 236 s** (NumPy) |
| Prostate_VMAT_101 (exact, NumPy) | 57 it, 52 s | max_time, 39 it (restoration from it. 28, KKT 2.25) | optimal, **60 it, 197 s** |
| Prostate_VMAT_102 (exact, NumPy) | 44 it, 50 s | — | optimal, **74 it, 317 s** |
| Prostate_VMAT_101 (L-BFGS) | 57 it, 52 s | acceptable, 635 it, 1844 s | acceptable, **404 it, 1080 s** |
| Prostate_VMAT_102 (L-BFGS) | 44 it, 50 s | max_time, 507 it, KKT 3.7e-5, +1.7 % obj | acceptable, **376 it, 1354 s** |
| Head-and-Neck_01 (L-BFGS) | 45 it, 279 s | max_time, 68 it, KKT 0.39 | max_time, 67 it, **KKT 0.09** |
| Prostate_BT_01 | — | optimal, 12 it, obj 3e-10 | optimal, 111 it, 4 s, **obj −0.780** |
| Prostate_BT_02 | — | optimal, 10 it, obj 2e-9 | optimal, 83 it, 3 s, **obj −0.753** |

Every objective matches the published one to 0.13 % or better except where
marked; on the BT pair the monotone arm certifies a poor stationary point at
objective 0 in a dozen iterations, while the candidate reaches the −0.78 /
−0.75 optima the August L-BFGS runs needed 400 iterations for — so the BT
"iteration regression" is a solution-quality win. Head-and-Neck (n = 9984)
does not converge in 30 minutes under either arm on this machine (27 s per
L-BFGS iteration on NumPy; the paper's 16 cores: 6 s); the candidate is
further along at the cap. The sweep also exposed two apparent per-iteration
problems. One was real and is fixed below: the exact Hessian on CuPy crossed
gated-off structures through the host (CK 33 s/iter, VMAT 51 s/iter → 0.96
and ~2 s/iter). The other was not a cost at all: the VMAT_101 exact arm's
50 s per iteration under monotone μ was restoration. The exact Hessian costs
~2.0 s per iteration on NumPy (VMAT_101; 2.7 s on VMAT_102), the same as
L-BFGS (2.0 / 2.5 s); the monotone arm climbs to θ ≈ 2e10 at iteration 2
and the objective to 9e6 by iteration 19, enters restoration at iteration
28 and never recovers (157–200 Levenberg–Marquardt steps per call, 150–430 s
each; the sweep row's final KKT 2.25 is iteration 28's). The default arm
does not restore at all. Restoration's own cost on these cases is now
mostly core-side (the LU solve of the SPD Gauss–Newton matrix, ~60 %): the
loader forms its active-set Grams from the ~100 active rows instead of all
81,279 (`ActiveRowGramCSROperator`; 946 → 631 s for the monotone arm's
first 31 iterations, identical trajectory). Reproduce: the runner defaults;
`--mu-schedule monotone --kappa-centrality 1e-2` for the previous arm.

**Head-and-Neck_01 profile (2026-09-28, provisional).** This is the
runner's default recipe on the only `n ≈ 10⁴` case measured so far
(`n = 9984`, 100,223 lowered linear rows, 104 M nonzeros in 49 tiles, all
float64). It adds the exact-Hessian arm the sweep above did not run, and
the first CuPy numbers. Setup: a 13th-gen i9-13900HX laptop (24 cores, 32
threads, threaded OpenBLAS, CPU load under 15 % before each run) and its
RTX 4070 Laptop GPU (8 GB, fp64 at 1/64 rate, CuPy 14, nothing else on
it). Every run was one process under 9 minutes. Each phase was timed
exclusively: a phase's time excludes the phases nested in it, and the GPU
was synchronized at every phase boundary. The per-iteration means cover
iterations 1 to the last one. Initialization is listed separately.

| per iteration (s) | NumPy exact | NumPy L-BFGS | CuPy exact, before → after | CuPy L-BFGS, before → after | paper (16-core Xeon) |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear-block Gram (tiled) | 19.66 | 19.52 | 67.33 → 10.51 | 67.37 → 8.53 | 4.07 ("make N", incl. Hessian) |
| other Gram rows (nonlinear constraints) | 0.34 | 0.37 | 1.29 → 1.51 | 1.34 → 0.84 | |
| exact Hessian, dense formation | 10.99 | — | 7.57 → 2.74 | — | |
| Cholesky factorization | 2.57 | 3.01 | 2.81 → 4.87 | 2.02 → 1.36 | 1.07 |
| back-solves | 1.88 | 2.08 | 3.97 → 4.80 | 3.84 → 2.69 | 0.44 |
| KKT assembly | 0.70 | 1.60 | 0.36 → 0.41 | 0.64 → 0.40 | |
| Mehrotra corrector, line search, evaluations, driver | 0.61 | 0.68 | 0.47 → 0.35 | 0.25 → 0.21 | 0.26 (functions, TQD, step length) |
| **total** | **36.75** | **27.24** | **83.8 → 25.2** | **75.5 → 14.0** | **6.2** (279 s / 45, incl. load and init) |
| iterations timed | 12 | 16 | 5 → 18 | 5 → 25 | 45 |

- **Initialization: 35–39 s, against the paper's 7.8 s.** It splits into
  the least-squares start (3.3 s), the linear block's tile plan (6.2–6.9 s,
  in the first `linear_ineq()` call), and the LSQR dual start (26.8 s on
  CuPy, 29.6 s through SciPy on NumPy). LSQR stops at its 200-iteration cap
  on this case ("LSQR stopped at its iteration limit"). The runner rows
  include this time. See "LSQR dual start across the corpus" below for the
  opt-in `lsqr_iter`/`lsqr_tol` override this motivated.
- **NumPy: the tiled Gram dominates.** It is one accumulation lane here,
  because the 1 GiB lane budget holds a single `n²` accumulator at
  `n = 9984`, and in the exact arm the Hessian's tiled products add 11 s.
  With `TiledGramConfig(lane_budget_bytes=4 << 30)`, which allows 5 lanes
  and about 3.2 GB more host memory, the Gram alone drops from 19.7–20.0 s
  to 10.4–12.2 s (3 fresh-weight samples each, results equal to 3·10⁻²⁰
  relative). The exact iteration drops from 36.8 to 24.4 s (Gram 11.1 s,
  Hessian 7.1 s), with the same trajectory. That is a lever, not the
  default — **verified 2026-09-28: the Gram is not bit-identical across
  `lane_budget_bytes`.** `TiledGramConfig`'s own docstring already says why:
  the lane *count* fixes the tile→lane assignment and with it the
  summation order, and float addition is not associative. Checked both on a
  small synthetic instance parametrized to hit 1/2/5/8 lanes through the
  same `lane_budget_bytes // (8·nx²)` arithmetic (`max|diff|` up to `9e-13`
  absolute — see `test_lane_budget_bytes_changes_the_gram_only_by_round_off`
  in `tests/unit/test_trots_tiled_gram.py`) and on `Head-and-Neck_01`'s real
  linear block at `1, 2, 4, 8` GiB (fresh weights each call; `max|diff|`
  `4.4·10⁻¹⁶`–`6.7·10⁻¹⁶` absolute against the 1 GiB Gram — consistent with
  the "3·10⁻²⁰ relative" already noted above). Because it is not
  bit-identical, `lane_budget_bytes`' default stays `1 << 30`; a size-aware
  default was considered (more lanes when the host has the memory) but is
  not shipped, since it would silently change every existing Gram's last
  bits on a bigger machine. `TiledGramConfig(lane_budget_bytes=4 << 30)` (or
  higher, memory permitting) remains an explicit, opt-in lever for this
  case. The back-solves are the core's `scipy.linalg.cho_solve` copying
  the C-ordered factor to Fortran order on every solve: 0.91 s per solve
  against 0.053 s on an F-ordered factor, measured standalone at
  `n = 9984`.
- **NumPy: the lower-to-upper mirror was a per-row Python loop.**
  `_mirror_lower` copied `out[j, j+1:] = out[j+1:, j]` in a loop over all
  `n = 9984` rows, once per Gram (and once per exact-Hessian dense
  formation). Replaced with the same cache-blocked panel scheme the device
  mirror already used (`_HOST_MIRROR_BLOCK = 512`: a `triu_indices` gather
  per diagonal block, one transposed-slice copy for the panel beside it) —
  a pure data copy, bitwise identical to the old loop's output for every
  size relative to the block (see
  `test_mirror_lower_matches_row_loop_reference`). Provisional, standalone
  at `n = 9984`, interleaved medians of 3 (shared machine): row loop
  0.41 s, blocked 0.39–0.41 s — a modest, noisy win on this run, well short
  of erasing the loop's cost outright; block sizes 128–2048 were within
  the same noise band, so 512 was kept as a reasonable middle value rather
  than a measured optimum. The fully vectorized alternatives
  (`np.tril(out) + np.tril(out, -1).T`, or a single whole-matrix
  `triu_indices` assignment) were 4–6× *slower* at this size — their
  temporaries are themselves gigabyte-scale at `n = 9984`.
- **CuPy, before:** 3 of the plan's 8 sparse tiles (`Σ nnz_row² / b·c² ≈
  4·10⁻³`, up to 2048 × 9684) took 37.7 s by SpGEMM against 3.1 s
  densified. One of them needs a 25 GB SpGEMM buffer. The CuPy pool grew
  to 39–40 GB through the driver's system-memory fallback.
- **CuPy, after:** the size-aware tile choice and the shared tile buffers
  (described under *Device namespaces* in the tiled-Gram section below).
  NumPy and CuPy follow the same trajectory in both arms: the
  objectives agree to 8·10⁻¹⁴ relative through iteration 9, and to
  5·10⁻¹⁰ at iteration 12 of the exact arm. The Gram alone now takes
  7.5 s, of which `syrk` is 7.24 s: 2.2 TFLOP of compacted tiles at about
  300 GFLOP/s, this card's fp64 ceiling. In the solve it takes 8.5–10.5 s.
  The whole problem still did not fit in 8 GB. The CuPy pool sat at
  12.4–13.9 GB, and with a 7.6 GB pool limit the start already failed
  converting the lowered block from COO to CSR. So the dense `n×n`
  factorization and the back-solves ran in fallback memory. Standalone at
  `n = 9984`, the Cholesky takes 1.2 s and a two-sided solve 0.02 s. With
  6 GB of ballast allocated first, they take 5.2 s and 2.5 s, which is the
  4.9 s and 4.8 s in the table. The next item removed that.
- **CuPy, device memory (2026-09-28, provisional).** The case now fits: 7
  iterations of the exact arm under a 7.5 GB pool limit, peak pool use
  6.51 GB (was 11.21 GB) and pool size 7.31 GB (was 13.56 GB), 11.2 s per
  iteration against 18.6–39.5 s before on the same laptop, and the
  initialization 19.5 s against 54 s. Both runs were measured under CPU load
  from other jobs, one process per arm, so the times are provisional. The
  trajectory is the same: objective and KKT error agree to 1.2·10⁻¹³
  relative through iteration 7, which is the CuPy run-to-run spread (two
  runs of the old code differ by 1.6·10⁻¹³: cuSPARSE's transpose products
  accumulate with atomics). In the solve the Cholesky now takes 1.19 s and a
  back-solve under 0.01 s, the standalone times. The attribution below walks
  the problem, driver and solver objects for live CuPy arrays (between
  iterations 2 and 3; `n² · 8` bytes = 760 MiB at `n = 9984`):

  | component | before | after | what changed |
  | --- | ---: | ---: | --- |
  | linear block, COO triplets (int64 rows, int64 columns, values) | 2530 MiB | — | a duplicate of the adapter's CSR: the loader's CuPy operator now holds only that CSR |
  | linear block, adapter CSR (values, int32 columns) | 1265 MiB | 1265 MiB | the only copy of the block used by the products |
  | device tiles of the linear block and the Hessian structures | 1256 MiB | 1256 MiB | the tiled Gram's own layout (RCM-permuted, compacted) |
  | device evaluation matrices (`_dev_mats`) | 186 MiB | 186 MiB | |
  | linear-block Gram memo | 760 MiB | 760 MiB | now released before the next Gram is accumulated beside it |
  | nonlinear-rows Gram memo (adapter) | 760 MiB | 760 MiB | |
  | exact Hessian dense memo | 760 MiB | 760 MiB | |
  | condensed matrix `N` | 760 MiB | 760 MiB | |
  | Cholesky factor | 760 MiB | 760 MiB | |
  | **pool in use between iterations** | **8.87 GB** | **6.40 GB** | |

  The peaks sat higher than that. The old start converted the lowered
  block's triplets to CSR on the device, which peaked at 8.84 GB; the block
  is now uploaded as that CSR (1.43 GB peak). The previous iteration's
  Hessian and nonlinear rows, with their two `n×n` memos, stayed alive
  through the next iteration's assembly: the driver's second-order-correction
  closure held them as default arguments. It is now released after the line
  search. Each back-solve copied the factor to Fortran order (two
  760 MiB copies per solve); the CuPy adapter now solves with the factor's
  transpose view, which is Fortran-ordered, and copies nothing. The last
  addition of the assembly (`N = block + Gram`) now adds into the assembly's
  own block in place. The largest remaining peak is the linear-block Gram
  itself (6.51 GB: its result and tile buffers next to the Hessian memo, the
  assembly block and the nonlinear-rows memo). The smaller cases follow the
  same trajectories on CuPy: `Protons_01` and `Liver_01` keep their
  iteration counts (93 and 57) and statuses, and their final objectives move
  by 2.8·10⁻⁹ and 2·10⁻¹⁶ relative; two runs of the old code already differ
  by 2.3·10⁻⁹ on `Protons_01`. Its peak pool use drops from 1.63 to
  0.70 GB. The L-BFGS arm also runs under the 7.5 GB limit (6 iterations,
  peak 7.06 GB, 8.8–10.1 s per iteration after the first, provisional; its
  peak phase was not attributed).

**Head-and-Neck_01 trajectory (2026-09-28).** The problem is convex
(`δ_w = 0` at every iteration), yet the runner defaults blow up on the
first step. Every LSQR multiplier of eq. (26) is below 2·10⁻⁵ in magnitude,
so the §3.1 floor of 5 sets all 100,246 inequality multipliers to 5 and
`μ0 = 665`: a barrier force of about 2.5·10⁸ × ‖∇f‖. ipax then takes
separate primal and dual step lengths (`α_p = 0.84`, `α_d = 0.27`), and the
corrected direction drops the PTV mean dose from 56 to 22 Gy, sending the
LTCP objective from 4.6 to 7.9·10⁸. The ratio control (eq. (36)) does not
reject the step. Two opt-in levers address this. The first is
`BreedveldOptions(common_step=True)` (runner `--common-step`): Breedveld's
own single step length `α = min(α_p, α_d)` (§2.1; the paper keeps the
common step for every problem with a nonlinear term and pairs the
primal/dual lengths separately only on purely linear problems such as the
proton cases — the option applies it everywhere). It couples only the
step that is taken; the corrector's probes keep separate lengths. The
second is a units-aware multiplier floor, `breedveld_warm_start(...,
dual_floor=1e-6)` (runner `--dual-floor 1e-6`), which gives `μ0 = 1.8·10⁻⁴`
(the slacks keep the floor of 5). Only the two together remove the
excursion.

The table below is the runner with its defaults plus the listed flags, on
NumPy with the default 1 GiB lane budget. It records iterations, not time
(the machine was shared). `f` is the TROTS objective of the iterate.

| Head-and-Neck_01 | f at it. 1 | largest f, it. 1–10 | f at it. 10 | f / KKT at it. 20 |
| --- | ---: | ---: | ---: | ---: |
| runner defaults (diagnosis harness) | 7.9·10⁸ | 7.9·10⁸ | — (it. 6: 7.3·10⁶) | — |
| `--dual-floor 1e-6` | 1.3·10¹⁰ | 1.3·10¹⁰ | — (it. 9: 1.0·10⁷) | — |
| `--common-step` | 11.6 | 2.7·10³ | 16.1 | — |
| `--common-step --dual-floor 1e-6` | 4.09 | 4.24 | 3.90 | 3.5575 / 3.8 |
| reference (paper) | | | | 3.5423 |

Both `--common-step` rows reproduce the diagnosis harness's trajectories
(the harness monkeypatched these two levers). They match to five digits
through iteration 10. From iteration 13 the runner row drifts in the
fourth digit (3.5575 against the harness's 3.5576 at iteration 20). The
harness used a 4 GiB lane budget, and the Gram's last bits depend on it
(see above). The run is not converged at iteration 20. The 23 `Sm2` quadratic smoothing constraints overshoot, which
is a separate open item. A variant that also couples the corrector's probes
reached a similar point at iteration 20 (f = 3.5786, KKT 1.6), but only
after an excursion to f = 95 at iteration 2, so it was not adopted.

The same flags on the two cases the runner defaults already converge are
the first data point for a sweep across all sites. The defaults are
unchanged: bit-identical final objective and KKT against the pre-change
code.

| case | runner defaults | `--common-step` | `--dual-floor 1e-6` | both |
| --- | --- | --- | --- | --- |
| Protons_01 | acceptable, 93 it, 32.647594 | acceptable, 116 it | optimal, 77 it | acceptable, 86 it |
| Liver_01 | optimal, 57 it, −1.0370696 | optimal, 46 it | optimal, 81 it | optimal, 118 it |

Every row certifies the same objective to at least 7 significant figures.
Reproduce with `--cases Head-and-Neck_01 --max-iter 20 --common-step
--dual-floor 1e-6`.

### Common step and dual floor across the corpus (2026-09-28)

The two-case spot check above was run to completion by coincidence (its
iteration counts already match the full run below); this section runs the
same three arms — `--common-step`, `--dual-floor 1e-6`, both together — to
their terminal status on all 11 corpus cases, at the runner's own budget
(`--max-iter 800 --max-time 1800`, Head-and-Neck `--max-time 3600`), to
decide whether either lever should become a runner default. NumPy, one
process per case, at most 3 concurrent solves and never two Head-and-Neck
solves at once (a second, GPU-bound agent shared the machine throughout, so
every wall-clock figure below is **provisional** — read iteration counts and
statuses as the evidence, not the seconds). The "defaults" column reuses the
2026-09-27 corpus-sweep table above unchanged (the NumPy default path is
fingerprint-identical since then), except the exact-Hessian rows for
`Prostate_VMAT_101`/`102` and `Head-and-Neck_01`, which that table listed
under L-BFGS: their exact-arm defaults come from the exact-Hessian rows
of the same table (`VMAT_101` 60 it, `VMAT_102` 74 it) and the
Head-and-Neck_01 profile section (does not converge).

| case | paper it | defaults (reused) | `--common-step` | `--dual-floor 1e-6` | both |
| --- | ---: | --- | --- | --- | --- |
| Protons_01 | 79 | acceptable, 93 it, 32.647594 (−0.125%) | acceptable, 116 it, 32.654422 (−0.104%) | optimal, 77 it, 32.654422 (−0.104%) | acceptable, 86 it, 32.654422 (−0.104%) |
| Protons_02 | 65 | optimal, 80 it | optimal, 96 it, 42.921087 (−0.039%) | optimal, 74 it, 42.921087 (−0.039%) | optimal, 90 it, 42.921087 (−0.039%) |
| Liver_01 | 36 | optimal, 57 it, −1.0370696 | optimal, 46 it, −1.0370702 (≈0%) | optimal, 81 it, −1.0370702 (≈0%) | optimal, 118 it, −1.0370702 (≈0%) |
| Liver_02 | 43 | optimal, 39 it | optimal, 49 it, −0.1598682 (≈0%) | optimal, 90 it, −0.1598682 (≈0%) | optimal, 123 it, −0.1598682 (≈0%) |
| Prostate_CK_01 | 46 | optimal, 39 it | optimal, 56 it, 2.8258065 (≈0%) | optimal, 121 it, 2.8258065 (≈0%) | optimal, 94 it, 2.8258065 (≈0%) |
| Prostate_CK_02 | 39 | optimal, 48 it | optimal, 53 it, 1.3383233 (≈0%) | optimal, 93 it, 1.3383233 (≈0%) | optimal, 64 it, 1.3383233 (≈0%) |
| Prostate_VMAT_101 (exact) | 57 | optimal, 60 it | optimal, 117 it, 21.878745 (≈0%) | optimal, 129 it, 21.878745 (≈0%) | optimal, 143 it, 21.878745 (≈0%) |
| Prostate_VMAT_102 (exact) | 44 | optimal, 74 it | optimal, 65 it, 20.238272 (≈0%) | **stalled, 49 it, 51.111085 (+152.5%)** | optimal, 99 it, 20.238272 (≈0%) |
| Head-and-Neck_01 (exact) | 45 | no convergence | optimal, 46 it, 3.5422668 (+0.00013%) | max_time, 53 it, 3.5422838 (+0.00061%), KKT 2.8e-3 | optimal, 47 it, 3.5422668 (+0.00013%) |
| Prostate_BT_01 | — | optimal, 111 it, −0.780 | optimal, 112 it, −0.780264 | optimal, 92 it, −0.780264 | optimal, 96 it, −0.780271 |
| Prostate_BT_02 | — | optimal, 83 it, −0.753 | optimal, 123 it, −0.752723 | optimal, 100 it, −0.752723 | optimal, 123 it, −0.752723 |

Percentages are relative to the dataset's published reference objective
(`benchmarks.corpus.trots_reference` has no entry for the `Prostate_BT_*`
cases, so those two report the raw objective as before). Every case matches
its reference to well under the 1e-3 relative flag except two: `Protons_01`
sits at its already-documented ≈0.1% "below reference" certification in
*every* arm including the reused defaults (a known property of this case,
not something either lever introduces — see the Breedveld-parity recipe
section above), and `Prostate_VMAT_102` under `--dual-floor 1e-6` alone,
which is a genuine failure, not a rounding artefact: `stalled` at a
constraint violation of 52.4 and an objective 152% above the reference.

**Per-lever tally, against the reused defaults column (the 10 non-Head-and-Neck
cases; Head-and-Neck is called out separately since its default does not
converge at all):**

- **`--common-step` alone:** reduces iterations on 2/10 (`Liver_01` 57→46,
  `Prostate_VMAT_102` 74→65), increases them on the other 8 — sharply on the
  proton cases (`Protons_01` 93→116, `Protons_02` 80→96) and `Prostate_VMAT_101`
  (60→117, +95%). No status changes on the 10. Objective moves beyond 1e-6
  relative to the default arm only on `Protons_01` (32.647594→32.654422,
  +2.1e-4 relative — still on the same "below reference" side, not a
  regression). On `Head-and-Neck_01` it is the lever that fixes the
  divergence outright: **optimal at 46 iterations**, within one iteration of
  the paper's 45.
- **`--dual-floor 1e-6` alone:** reduces iterations on 3/10 (`Protons_01`
  93→77, `Protons_02` 80→74, `Prostate_BT_01` 111→92), increases them on 6/10
  (up to `Prostate_CK_01` 39→121, +210%), and **changes the status on 1/10**:
  `Prostate_VMAT_102` goes from `optimal` (74 it) to `stalled` (49 it, KKT
  57.4) — a real regression, not noise, and disqualifying on its own for a
  default. On `Head-and-Neck_01` it does not fully recover either: `max_time`
  at 53 iterations, KKT 2.8e-3, objective within 0.0006% of the reference —
  close, but not a certified point within the budget.
- **Both together:** reduces iterations on 2/10 (`Protons_01` 93→86,
  `Prostate_BT_01` 111→96), increases them on the other 8, and recovers
  `Prostate_VMAT_102`'s stall back to `optimal` (99 it) — the pairing cures
  `--dual-floor`'s regression there, at roughly `--common-step`'s own
  iteration cost. On `Head-and-Neck_01` it converges (**optimal, 47
  iterations**), one more than `--common-step` alone and with an
  objective that agrees with it to 6 significant figures — pairing
  `--dual-floor` on top buys nothing further on this case.

**Protons (purely linear).** Breedveld's paper pairs separate primal/dual
step lengths on the proton sites and uses the common step only where a
problem has a nonlinear term (§2.1); `--common-step` applies it
unconditionally, so Protons is exactly the site his own algorithm would
*not* couple. The measurement matches that expectation: `--common-step`
costs the most iterations of the three levers on both proton cases (+23,
+16), while `--dual-floor` alone — a units fix unrelated to step coupling —
is the one lever that *helps* here (93→77, 80→74), consistent with every
LSQR multiplier being floored regardless of arm (see the LSQR section
below) and the floor of 5 being the wrong scale for gradients this small.

**Head-and-Neck (does it now converge?).** Yes, both `--common-step` alone
and `--common-step --dual-floor 1e-6` converge it to `optimal` within the
3600 s budget — 46 and 47 iterations, KKT 1e-9, objective within 0.0002% of
the reference either way, against the paper's 45 iterations. This updates
the trajectory section above: the 2026-09-28 diagnosis needed both levers to
avoid the barrier excursion within the first 20 iterations it examined, but
run to completion `--common-step` alone is sufficient — `--dual-floor`
changes nothing once the common step is applied (46 vs 47 iterations, same
objective to 6 figures) and does not by itself replace it (`max_time`, KKT
2.8e-3, not certified).

Re-timed on an idle machine (2026-09-29, the runner with `--common-step`,
default 1 GiB lane budget, `--max-iter 300 --max-time 3600`), the two
backends produce the same trajectory:

| backend | status | iterations | KKT | objective vs reference | wall | initialization | per iteration |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NumPy | optimal | 46 | 1.0e-9 | 4.5e-12 | 1786 s | 41 s | 37.9 s |
| CuPy (RTX 4070 Laptop, 8 GB) | optimal | 46 | 1.0e-9 | 4.5e-12 | 526 s | 13 s | 11.1 s |
| paper (16-core Xeon) | — | 45 | — | — | 279 s | 7.8 s | 6.2 s |

The GPU run fits the card after the device-memory work above; the NumPy
per-iteration cost is the single-lane Gram (a 2–4 GiB `lane_budget_bytes`
brings it to ~24 s, see the profile section).

**Recommendation: keep both opt-in.** Neither lever becomes a runner
default. `--common-step` costs iterations on 8 of the other 10 cases (up to
+95%) for a fix only `Head-and-Neck_01` needs; `--dual-floor` alone
regresses `Prostate_VMAT_102` from `optimal` to `stalled` at a 152%-wrong
objective, which rules it out as a default on correctness grounds alone. The
runner's existing recipe already reaches `optimal`/`acceptable` on 10 of 11
cases; `Head-and-Neck_01` is the one exception, and `--common-step` (alone,
without `--dual-floor`) is now the documented per-case override for it —
matching the page's existing pattern of proton- and site-specific lever
combinations rather than corpus-wide defaults.

**LSQR dual start across the corpus (2026-09-28, provisional).** The
Head-and-Neck_01 profile's initialization bullet above flags the LSQR dual
start (`breedveld_warm_start`, eq. (26)) as 26.8–29.6 s of the ~35–39 s
initialization, hitting its 200-iteration cap. Measured on the runner's
`init="breedveld"` primal start, `lsqr="host"`, same laptop, one process per
case, no other load:

| case | n | lowered rows | LSQR itn | istop | LSQR (s) | warm start (s) | y₀ > 5 | z₀ > 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Protons_01 | 1,101 | 369,445 | 63 | converged | 4.93 | 9.95 | 0 / 1,101 | 1,080 / 1,101 |
| Protons_02 | 2,083 | 419,447 | 91 | converged | 9.97 | 12.38 | 0 / 2,083 | 2,062 / 2,083 |
| Liver_01 | 1,124 | 121,975 | 200 | cap | 16.86 | 18.27 | 0 / 1,124 | 125 / 1,124 |
| Liver_02 | 967 | 119,826 | 64 | converged | 5.65 | 8.22 | 0 / 967 | 61 / 967 |
| Prostate_CK_01 | 2,266 | 89,195 | 200 | cap | 33.04 | 40.85 | 0 / 2,266 | 1,918 / 2,266 |
| Prostate_CK_02 | 2,733 | 87,017 | 200 | cap | 26.90 | 36.88 | 0 / 2,733 | 2,355 / 2,733 |
| Prostate_VMAT_101 | 1,976 | 81,279 | 200 | cap | 23.13 | 24.51 | 0 / 1,976 | 1,936 / 1,976 |
| Prostate_VMAT_102 | 2,259 | 99,911 | 180 | converged | 24.48 | 31.41 | 0 / 2,259 | 2,188 / 2,259 |
| Head-and-Neck_01 | 9,984 | 100,246 | 200 | cap | 33.56 | 46.97 | 0 / 9,984 | 4,790 / 9,984 |
| Prostate_BT_01 | 55 | 6,338 | 200 | cap | 0.10 | 0.11 | 0 / 55 | 55 / 55 |
| Prostate_BT_02 | 70 | 5,747 | 200 | cap | 0.10 | 0.11 | 0 / 70 | 70 / 70 |

`y₀ > 5` counts the LSQR multiplier components that clear the floor of 5
*before* the elementwise `max(y, 5)` (eq. (26)); it is **zero on every one
of the 11 cases** — every LSQR multiplier the corpus produces, converged or
capped at 200 iterations, is discarded by the floor, and `y0` is the
constant vector `5·1`. `z₀ > 5` (eq. (27), `z0 = max(∇f + Jᵀy0, 5)` on the
bounded entries) is *not* uniformly floored, but since `y0` is the same
constant `5·1` regardless of the exact (possibly unconverged) LSQR iterate,
`z0` and `μ0` (eq. (10)) do not depend on LSQR's numeric output either — only
on whether it stayed under the floor.

That raised an obvious question: given `y0` is always `5·1` here, can LSQR
be skipped or truncated? An early-exit was evaluated for a
**bit-identical** result — the same `WarmStart` arrays and `μ0` — against
three cheaper variants on all 11 cases, `lsqr="host"` fresh weights each run
(patching `scipy.sparse.linalg.lsqr` to return early):

| case | full (s) | skip (y=0), Δt | iter_lim=5, Δt | iter_lim=20, Δt |
| --- | ---: | ---: | ---: | ---: |
| Protons_01 | 7.82 | 2.47 s, bit-identical | 1.88 s, bit-identical | 2.31 s, bit-identical |
| Protons_02 | 11.66 | 2.99 s, bit-identical | 4.62 s, bit-identical | 5.61 s, bit-identical |
| Liver_01 | 15.41 | 0.94 s, bit-identical | 2.25 s, bit-identical | 3.54 s, bit-identical |
| Liver_02 | 4.23 | 0.59 s, bit-identical | 0.82 s, bit-identical | 1.98 s, bit-identical |
| Prostate_CK_01 | 26.79 | 4.15 s, bit-identical | 6.03 s, bit-identical | 8.10 s, bit-identical |
| Prostate_CK_02 | 34.22 | 5.67 s, bit-identical | 9.23 s, bit-identical | 14.62 s, bit-identical |
| Prostate_VMAT_101 | 35.82 | 1.50 s, bit-identical | 3.37 s, bit-identical | 6.18 s, bit-identical |
| Prostate_VMAT_102 | 42.83 | 9.72 s, bit-identical | 12.51 s, bit-identical | 15.00 s, bit-identical |
| Head-and-Neck_01 | 44.21 | 8.41 s, bit-identical | 11.79 s, bit-identical | 12.82 s, bit-identical |
| Prostate_BT_01 | 0.12 | 0.011 s, bit-identical | 0.013 s, bit-identical | 0.021 s, bit-identical |
| Prostate_BT_02 | 0.12 | 0.012 s, bit-identical | 0.014 s, bit-identical | 0.021 s, bit-identical |

Even **`iter_lim=0` (`y=0`, LSQR not run at all)** reproduces `warm.y_ineq`,
`warm.z_lower`, `warm.s` and `μ0` bit-for-bit against the full 200-iteration
run on **every one of the 11 corpus-sweep cases** — because `y0` was always
going to be floored to `5·1` regardless of the exact LSQR iterate.

**This is not shipped, and is not proposed as a default**, for the reason
the task set out to check: it is not provable ahead of time. The natural
candidate certificate — stop once the current iterate's `‖y_k‖∞ < 5` — is
*not* sufficient in general, because LSQR's iterate norm is monotonically
non-decreasing (Paige & Saunders 1982): a small early iterate proves nothing
about later ones, which could still cross the floor before convergence or
the cap. A synthetic counterexample confirms the general case is real, not
hypothetical: `tests/unit/test_trots_breedveld_start.py`'s existing
`_instance()` fixture is built to exercise *both* sides of every floor
(its docstring says so), and
`test_truncating_lsqr_is_not_safe_in_general` shows a 1-iteration early
exit there changes `y0` outright (`μ0` 818.95 → 706.52, more than 20 of 51
multipliers move) — i.e. the moment a problem's true dual multipliers are
not uniformly small, an early exit silently returns the wrong start. No
general, cheap, provably bit-identical shortcut was found; the corpus
result above is a property of TROTS's scaling (small gradients relative to
the floor of 5), not of the code. **Shipped as opt-in, not a new default:**
`breedveld_warm_start(..., lsqr_iter=, lsqr_tol=)` (and the runner's
matching `--lsqr-iter`/`--lsqr-tol`, `--init breedveld` only) let a caller
set an explicit, lower LSQR budget *after* confirming, as this measurement
did, that their corpus's multipliers land under the floor — the module
constants (200 iterations, `atol = btol = 1e-6`) stay the default because
the bit-identity above is a corpus fact, not something the code can
guarantee. Re-measured through the new parameter on two cases (NumPy,
`scaling="none"`, provisional, shared machine — full
`breedveld_warm_start` wall, not LSQR alone, so it also includes the
lowered-Jacobian setup):

| case | default (200 it) | `lsqr_iter=5` | `WarmStart`/`μ0` |
| --- | ---: | ---: | --- |
| Head-and-Neck_01 | 46.6 s | 7.2 s | bit-identical |
| Prostate_CK_01 | 25.3 s | 3.7 s | bit-identical |

Device Gram of the linear block alone (fresh weights, one process per arm,
because two copies of the problem overflow the card):

| arm | Gram wall |
| --- | ---: |
| before: 8 sparse tiles by SpGEMM (sum of per-tile timings; the whole-Gram arm ran out of memory in its 25 GB SpGEMM buffer) | 44.4 s (6.7 s dense + 37.7 s sparse tiles) |
| calibrated tile choice only (5 tiles sparse) | 16.8–19.2 s (4 samples) |
| + shared tile buffers and the lower-triangle scatter kernel | **7.5 s** (3 samples) |

Per-tile crossover. Each of the 8 sparse tiles was multiplied alone,
minimum of 3 runs:

| tile `b × c` | `Σ nnz_row² / b·c²` | SpGEMM | densified | faster |
| --- | ---: | ---: | ---: | --- |
| 4 smoothing tiles, 2048 × 2059–2082 | 5.3·10⁻⁶ | 0.5–0.6 ms | 40 ms | SpGEMM |
| 1785 × 1797 (the calibration tile) | 7.1·10⁻⁶ | 0.6 ms | 28 ms | SpGEMM |
| 2048 × 3609 | 4.0·10⁻³ | 2.35 s | 0.105 s | dense |
| 1250 × 8158 | 4.0·10⁻³ | 10.6 s | 1.10 s | dense |
| 2048 × 9684 | 4.1·10⁻³ | 24.8 s | 1.87 s | dense |

This table was measured before the shared tile buffers. Its "densified"
column therefore still includes each tile's own allocations, the F-order
copy for `syrk` and the `cupy.add.at` scatter. On the 41 dense tiles that
path reached 250–290 GFLOP/s, counting `b·c²`, over about 2000 columns, but
only 76 and 103 GFLOP/s on the two widest tiles above (8158 and 9684
columns). SpGEMM manages 3–7·10⁷ products per second, fewer on the big
tiles, which spill. The per-tile break-even `Σ nnz_row² / b·c²` is
1.8–4.2·10⁻⁴. Every tile lies at least 10× away from it, so any `r` in that
range picks the faster kernel on all 8. The calibration, which times the
1785 × 1797 tile, measured 3.4–3.7·10⁻⁴ with that probe's dense arm
allocating per call as above. With the production dense path, which the
committed probe times, it measures 1.8·10⁻⁴, and the Gram takes
7.47–7.49 s (3 samples). The 41 dense tiles took 6.7 s.

Convergence within the 9-minute runs, and the wall extrapolated to the
paper's 45 iterations at the measured rate (initialization included). This
is not a prediction of convergence:

| arm | reached | objective (reference 3.5423) / KKT | 45 iterations at this rate |
| --- | --- | --- | ---: |
| NumPy exact (runner) | 14 it., max_time, 523 s | best iterate: 3.64·10⁵ / 29.8 (iteration 9) | 1,600 s |
| NumPy exact (harness) | 12 it. | 4.5·10⁴ / 162 | 1,690 s (1,140 s with 5 lanes) |
| NumPy L-BFGS (harness) | 16 it. | 5.33 / 0.31 (best KKT 0.09, iteration 6) | 1,265 s |
| CuPy exact (runner) | 16 it., max_iter, 411 s | best iterate as NumPy | 1,090 s |
| CuPy exact (harness) | 18 it. | 3.79·10³ / 41.5 | 1,170 s |
| CuPy L-BFGS (harness) | 25 it. | 4.46 / 0.89 | 670 s |

Neither arm is on course for the paper's 45 iterations. The exact arm's
first step sends the objective to 7.9·10⁸, and it takes 18 iterations to
come back to 3.8·10³. L-BFGS gets within 26 % of the reference by
iteration 25, but with primal steps of 0.1–0.2 and a KKT error that rises
from 0.09 at iteration 6. The runners report the best iterate, which is why
the NumPy and CuPy exact rows agree to 10 digits. Reproduce with the
runner's defaults, `--cases Head-and-Neck_01 --max-iter 15 --max-time 480`,
adding `--backend cupy` or `--config lbfgs/dense`.



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
**unmeasured**: when this was measured, that case saturated an 8 GB card's
memory pool (it fits since the 2026-09-28 device-memory changes of the
Head-and-Neck profile; this arm was not re-run) and drove it
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
That accounting holds for the plain operator. Since 2026-09-28 the loader's
tiled operator on CuPy (`TiledGramCSROperator`, the default wherever a
group's plan passes the gate) holds only the adapter's device CSR, 12 B/nnz:
the host CSR is canonical, so it is uploaded as that CSR directly, and the
COO triplets are rebuilt on demand for the sparse routes. The lowering wraps
the loader's operator in a signed row selection and copies nothing. On
`Head-and-Neck_01` (104 M nonzeros) the block is 1.26 GB on the device, down
from 3.8 GB, and building it peaks at 1.43 GB instead of 8.84 GB (see the
Head-and-Neck profile).
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
  structure, read back once per Gram. The block's lower triangle is added at
  `ix_(cols, cols)`: on CuPy by one elementwise kernel over the `c×c` block
  (no index temporaries), on Torch by `index_put_(accumulate=True)`. The
  column sets are unique, so no index collides and the result is
  deterministic. The triangle is mirrored once, in blocked panels, so the
  result is exactly symmetric. There are no lanes: tiles run in plan order on
  the stream. On CuPy the densified tiles of one call share two buffers
  sized for the largest tile (the `b×c` tile and a C-ordered `c×c` `syrk`
  result).
  Until 2026-09-28 every tile allocated its own `b×c` tile, an F-ordered copy
  of it for `syrk` and a `c×c` result. The scatter used `cupy.add.at` with
  broadcast `ix_` indices, which materializes a `c×c` int64 index (715 MiB
  at `c = 9684`) and ran about 6× slower than the kernel. On
  `Head-and-Neck_01` that grew the CuPy pool about 5.8 GB per Gram,
  past the 8 GB card. On this CuPy 14 / cuBLAS build the lower triangle is
  bitwise the same either way (tested). The workspace path runs `dsyrk` in
  the transposed orientation on C-ordered memory, so the identity holds
  empirically; cuBLAS does not guarantee it.
- **Sparse tiles: a size-aware choice.** On CuPy, exact Grams may multiply
  the plan's sparse tiles with cuSPARSE SpGEMM; float32 requests and Torch
  densify them, and `TiledGramConfig(device_sparse_tiles=False)` densifies
  them on CuPy too. *Which* sparse tiles keep SpGEMM is decided per tile by
  the host rule's statistic with a device break-even: a tile stays sparse
  when `Σ nnz_row² ≤ r·b·c²`, where `r` is the ratio of the two kernels'
  throughputs on this device. The loader measures `r` once per problem and
  device, the first time a structure with sparse tiles is materialized. It
  multiplies the plan's sparse tile with the least dense work both ways
  (warm-up plus the minimum of 3 synchronized calls each), so
  `r = (Σ nnz_row²/t_SpGEMM) / (b·c²/t_dense)`. The linear block and the exact
  Hessian share the measurement, and it is logged at INFO. The host's 1%
  is far off on this GPU (see the Head-and-Neck profile above): measured
  `r` is 1.8·10⁻⁴ on `Head-and-Neck_01` and 2.1·10⁻² on `Protons_01`. A
  single probe measures a ratio of per-call rates, fixed costs included,
  so `r` is good to about 2×. That is enough on these cases, whose tiles
  lie at least 10× from the break-even. Because `r` is measured, a tile
  near the break-even can take the other kernel in another process, which
  changes the Gram's last bits.
  `TiledGramConfig(device_sparse_tile_flop_ratio=...)` pins it: `1.0` keeps
  every sparse tile sparse, as before the calibration, and `0.0` densifies
  them all. Over-budget tiles are unaffected (below). Measured on
  `Protons_01` (17 sparse tiles, all kept sparse by the calibration):
  calibrated 0.205–0.21 s, all-sparse (`1.0`) 0.205–0.21 s, all-densified
  0.223–0.229 s, 5 interleaved samples each. The committed probe measures
  the same Gram at 0.206–0.208 s. Earlier session: SpGEMM 0.216
  vs densified 0.240 s in float64, densified 0.056 vs SpGEMM 0.067 s in
  float32. Densifying is priced wrongly by the
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
  to size its output; cupyx offers no way around it. The one-time break-even
  calibration adds 8 stream synchronizations per problem, plus the four
  calibration SpGEMMs' own. The scaled operand is
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
returned as-is only on NumPy namespaces. On CuPy and Torch
`lagrangian_hessian` returns a `TROTSDeviceExactHessian` whenever the Hessian
has a curved or quadratic term. A curved structure the flop-ratio gate keeps
off is planned anyway, lazily on the first dense call (on a device its
alternative was a host SciPy SpGEMM per call: `Prostate_CK_01` spent 36 s per
iteration there, 33 s of a 33 s iteration; `Prostate_VMAT_101` 51 s), and
registered under the linear block's key so a shared structure keeps one tile
set whichever side plans it first. `dense_matrix` runs every structure
through the device tiled kernel, sums `Q` on the device from the constant
union pattern of the symmetrised quadratic terms (one upload of the
multipliers per call, deterministic scatter at unique lower-triangle
positions), adds `U Vᵀ` by a device matmul and mirrors one triangle; a
steady-state call uploads only the per-structure weights, the multipliers and
the n×r gEUD factors. After the fix (CuPy, RTX 4070 Laptop, Breedveld setup):
`Prostate_CK_01` 33 → 0.96 s/iter, `Prostate_VMAT_101` 51 → ~1.2–2.3 s/iter,
`Liver_01` 0.83 → 0.33 s/iter (its `Liver minus CTV` structure was gated off
too), `Protons_01` unchanged; results match the host operator to round-off.
`Head-and-Neck_01` (n = 9984) was slower on this laptop GPU than on NumPy
(84 s vs 37 s per exact iteration): the fp64 linear-block Gram took 67 s
per iteration. Three of its sparse tiles are far cheaper densified, and the
per-tile temporaries overflowed the 8 GB card. With the size-aware tile
choice and the shared tile buffers above it is 25 s per iteration, faster
than NumPy (see the Head-and-Neck profile above). The device operator's
`matvec` is exact float64
and device-side on CuPy (the
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

## Reference timings from the paper (supplement)

Breedveld et al. 2017 publish their per-problem results for the 120 TROTS
cases they solve (all but the 25 `Prostate_BT_*` brachytherapy cases) in the
paper's Electronic supplementary material 2: problem sizes, iterations and
seconds for their Default, Mehrotra and Gondzio methods, the tiling /
permutation / multiple-precision ablations, thread scaling, and a per-module
time breakdown. `benchmarks/corpus/data/breedveld2017/trots_reference.json`
holds that workbook with the values unchanged. The loader
`benchmarks/corpus/trots_reference.py` reads it
(`breedveld2017_reference(case)`, `breedveld2017_site_timings()`), and the
runner prints the paper's Mehrotra iterations and seconds next to each ipax
solve (`ref_iters` and `ref_time_s` in the JSON report). The dataset's
`Results/*_result.txt` files come from those Mehrotra runs. For all 120 cases,
their "Number of Iterations" and "Solve Time" equal the supplement's Mehrotra
columns.

Iterations / wall-clock seconds for the cases this page discusses. "n, m"
counts the variables and constraints as stored; the "reformulated" column is
the paper's solver problem after adding minimax auxiliaries and splitting
constraints into one-sided rows. That column equals the problem ipax solves
after lowering (Protons_01: 1,101 × 369,445, Liver_01: 1,124 × 121,975):

| case | n | m | reformulated n, m | Default | Mehrotra | Gondzio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Protons_01 | 1,080 | 87,637 | 1,101, 369,445 | 120 / 71.8 s | 79 / 55.1 s | 56 / 52.8 s |
| Liver_01 | 1,118 | 93,195 | 1,124, 121,975 | 86 / 42.3 s | 36 / 22.5 s | 32 / 25.1 s |
| Prostate_CK_01 | 2,260 | 45,728 | 2,266, 89,195 | 91 / 69.8 s | 46 / 40.9 s | 39 / 40.4 s |
| Prostate_VMAT_101 | 1,972 | 57,925 | 1,976, 81,279 | 85 / 72.2 s | 57 / 51.9 s | 44 / 48.3 s |
| Head-and-Neck_01 | 9,977 | 77,726 | 9,984, 100,246 | 76 / 461.2 s | 45 / 279.4 s | 44 / 287.8 s |

Where the time goes, per tumour site: each module's share of the
module-timed seconds (Mehrotra method, tiling + permutation + mixed
precision, summed over the site's cases; the supplement's `Image` sheet).
"Make N" forms the normal-equations matrix `N`, the paper's counterpart of
ipax's condensed Gram:

| site | Make N | Backsolves | Cholesky | Initialise | Permute/Tile | Functions & gradients | Make TQD | Steplength | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prostate CK | 67 % | 12 % | 2 % | 4 % | 5 % | 3 % | 4 % | 2 % | 1,668 s |
| Prostate VMAT | 69 % | 11 % | 2 % | 4 % | 8 % | 3 % | 1 % | 2 % | 2,148 s |
| Head-and-Neck | 70 % | 7 % | 14 % | 3 % | 1 % | 2 % | 2 % | 1 % | 3,038 s |
| Head-and-Neck Alt | 81 % | 5 % | 7 % | 2 % | 3 % | 1 % | 1 % | 1 % | 6,657 s |
| Protons | 68 % | 18 % | 1 % | 3 % | 1 % | 3 % | 1 % | 4 % | 1,226 s |
| Liver | 55 % | 14 % | 1 % | 4 % | 5 % | 3 % | 3 % | 16 % | 360 s |

Forming `N` dominates at every site, as the Gram does in ipax. The Cholesky
factorization only matters at Head-and-Neck scale (`n ≈ 10⁴`). The paper's
"Initialise" module takes 1.6 s on `Protons_01`.

The paper's seconds were measured on its 2017 hardware, a 16-core Xeon with
16 threads unless the thread-scaling columns say otherwise (see the paper's
§6.3). Read them as context
for iteration counts and time shares, not as a like-for-like wall-clock
target. The data is © 2017 The Author(s), CC BY 4.0: Breedveld, van den Berg
& Heijmen, *Comput. Optim. Appl.* 68(2), 209–242,
[doi:10.1007/s10589-017-9919-4](https://doi.org/10.1007/s10589-017-9919-4).
It was converted to JSON by `benchmarks/tools/convert_breedveld2017_esm.py`
without changing any value. The attribution and license are in
`benchmarks/corpus/data/breedveld2017/`. That directory is the one part of
the repository that is not Apache-2.0.

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
    Under the Breedveld-parity setup the culprit turned out to be the
    El-Bakry centrality floor rather than the oracle: `"probing"` with
    `kappa_centrality=0` converges `Liver_01` in 57 iterations (430 with the
    floor, 74 with monotone μ) and won the 2026-09-27 sweep on every site,
    so it is the runner default — see *Measured results*.

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
  routes. As of 2026-09-28, the laptop measurements in the Head-and-Neck
  profile above give 36.8 s (exact) and 27.2 s (L-BFGS) per iteration on
  NumPy, and 25.2 s and 14.0 s on an 8 GB GPU that the case overfilled;
  11.2 s (exact) once it fit the card (provisional), against the paper's
  6.2 s. RT-specific dose-matrix kernels stay out of the `ipax` core (see
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
# breedveld globalization, probing μ with --kappa-centrality 0,
# --init breedveld, --tol 1e-8); --backend cupy for the GPU:
IPAX_TROTS_DIR=/path/to/TROTS python -m benchmarks.runners.trots --solve \
    --cases Protons_01 --backend cupy --max-iter 300 --max-time 900

# The monotone-μ arm of the 2026-09-27 sweep (the previous runner default):
IPAX_TROTS_DIR=/path/to/TROTS python -m benchmarks.runners.trots --solve \
    --cases Liver_01 --mu-schedule monotone --kappa-centrality 1e-2 \
    --max-iter 300 --max-time 900

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
