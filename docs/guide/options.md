# Configuring the solver

All configuration goes through a single frozen
[`Options`](../reference.md#ipax.options.Options) object passed to `solve`:

```python
res = ipax.solve(problem, x0, options=ipax.Options(hessian="exact"))
```

`Options` is immutable and validated on construction (bad values raise
`ValueError` immediately rather than failing deep in the loop). The defaults are
tuned for RT-scale problems and solve most cases unchanged; this page covers the
knobs you are most likely to touch. Every field is documented in the
[API reference](../reference.md#ipax.options.Options).

## Termination

Six independent sources stop the solve, checked in priority order. The first
two share the same five conditions; `None` disables a condition.

| Condition | Meaning |
|---|---|
| `dual_inf_tol` | scaled dual-infeasibility component of the KKT residual |
| `constr_viol_tol` | scaled primal infeasibility (constraint violation) |
| `compl_inf_tol` | scaled complementarity component |
| `f_tol` | absolute objective magnitude `|f| ≤ f_tol` |
| `f_rel_change_tol` | relative objective change between iterations |

**Optimality** ([`OptimalityConditionOptions`](../reference.md#ipax.options.OptimalityConditionOptions))
is single-iteration and reports [`Status.OPTIMAL`](results.md#status). The
default reproduces the classic scaled-KKT test — every component ≤ `1e-8`:

```python
ipax.Options(
    optimality=ipax.OptimalityConditionOptions(
        dual_inf_tol=1e-4,
        compl_inf_tol=1e-4,  # loose optimality, but…
        constr_viol_tol=1e-8,  # …tight feasibility, in one step
    )
)
```

**Acceptable** ([`AcceptableStoppingOptions`](../reference.md#ipax.options.AcceptableStoppingOptions))
mirrors the same conditions but requires them to hold for `n_iter` *consecutive*
iterations, and reports [`Status.ACCEPTABLE`](results.md#status) (also a
success). It follows the **IPOPT convention and is on by default**
(`dual_inf_tol = constr_viol_tol = compl_inf_tol = 1e-6` — 1e2× the optimality
default — held for `n_iter = 15`), so a problem whose achievable KKT floor sits
between the acceptable and optimal tolerances (a degenerate optimum, an
ill-conditioned least-squares fit) reports `ACCEPTABLE` instead of grinding on
to `max_iter`/`max_time`. Loosen a tolerance further when a
dual-infeasibility-dominated residual plateaus early:

```python
ipax.Options(
    acceptable=ipax.AcceptableStoppingOptions(
        dual_inf_tol=1.0,  # tolerate the stuck dual infeasibility
        constr_viol_tol=1e-6,
        f_rel_change_tol=1e-7,
        n_iter=5,
    )
)
```

Set every tolerance to `None` to disable the mechanism entirely and always run
to `max_iter`/`max_time` on a stagnant residual.

**Limits.** `max_iter` (default `3000`) reports `Status.MAX_ITER`; `max_time`
in seconds (default `None`, disabled) reports `Status.MAX_TIME`. Both are
checked at iteration boundaries.

**Unbounded detection.** `diverging_iterates_tol` (default `1e20`) stops the
solve with [`Status.UNBOUNDED`](results.md#status) once `‖x‖∞` exceeds it while
the point stays feasible — an IPOPT-style diverging-iterates test that catches
an objective running away to `-∞` and reports it honestly instead of letting
the runaway iterate eventually overflow to a misleading `NUMERICAL_ERROR`. Set
to `None` to disable.

**Stall detection.** `max_stall_iter` (default `25`) stops the solve after that
many *consecutive* frozen iterations — no accepted step (α = 0) and a
bit-for-bit unchanged KKT error — reporting `Status.STALLED`, or
`Status.ACCEPTABLE` when the frozen iterate already sits within the relaxed
KKT tolerance. A frozen iterate cannot recover by repetition (identical state
reproduces the identical rejected direction), so this converts a
whole-budget burn into a fast, honest verdict. Genuine limit cycles — where
restoration keeps *moving* the iterate — change the KKT error and reset the
counter, so they still run to the ordinary budgets. Set to `None` to disable.

**Failure results carry the best iterate.** On a failure status (`STALLED`,
`MAX_ITER`, `MAX_TIME`, `INFEASIBLE`, `NUMERICAL_ERROR`), the result returns
the accepted iterate with the lowest scaled KKT error — not whatever state a
diverging final phase left behind — and the message names the iteration it
came from. Relatedly, a "locally infeasible" verdict is vetoed (reported as
`STALLED`) whenever an accepted iterate already reached the feasibility level
the verdict itself uses: a run that has visited feasibility cannot honestly
call the problem locally infeasible.

## Hessian

`hessian` selects how the Lagrangian Hessian is obtained:

| Value | Behavior |
|---|---|
| `"lbfgs"` *(default)* | compact, Powell-damped L-BFGS. Works on every backend and keeps the condensed system positive definite. |
| `"exact"` | use the problem's `lagrangian_hessian` (analytic). |
| `"autodiff-hvp"` | exact Hessian-vector products via backend autodiff (PyTorch/JAX). |

L-BFGS is the universal default because it needs no second derivatives and
makes the condensed Newton matrix positive definite, which removes the need for
an inertia oracle. Switch to `"exact"` or `"autodiff-hvp"` when accurate
curvature meaningfully cuts iterations (typically convex/structured problems).
Tune the memory and damping through
[`LBFGSOptions`](../reference.md#ipax.options.LBFGSOptions) (`memory`,
`powell_damping`, `initial_scaling`).

## Linear solver

`linsolve` chooses the KKT solve route; `"auto"` (default) picks one from the
problem size and backend capabilities.

| Value | Use |
|---|---|
| `"auto"` *(default)* | dense below ~10⁴ variables (if `xp.linalg` is available), otherwise Krylov. |
| `"dense"` | Cholesky/`solve` on the condensed matrix. Small, dense problems. |
| `"krylov"` | matrix-free CG/MINRES/GMRES. The scale default; never forms an `n×n` matrix. |
| `"sparse"` | per-backend sparse-direct factorization (assembles and factors the saddle). |

Picking a route does **not** require changing your `Problem` — it only controls
how the same operators are solved. The matrix-free route requires your
Jacobians/Hessian to be (or normalize to) operators with a `matvec`; the sparse
route additionally needs COO structure and a backend sparse adapter. See
[The linear-algebra layer](../concepts/linalg.md) and
[Backends & hardware](backends.md). Krylov tolerances and the preconditioner are
in [`KrylovOptions`](../reference.md#ipax.options.KrylovOptions).

!!! tip "Tall problems (`m ≫ n`), incl. radiotherapy scale: keep `\"auto\"`"
    For a very tall inequality system (`m ≫ n`) — the radiotherapy dose regime,
    `n≈10³`–`10⁴` with `m≈10⁵`–`10⁶` per-voxel dose constraints — `"auto"`'s
    dense condensed route is the right choice on a normally-threaded BLAS, and
    no override is needed. It forms the `n×n` normal-equations matrix
    `N = ∇gᵀΣ∇g` (the same condensation Breedveld 2017 uses for this problem
    class) and factors it by Cholesky; both the Gram accumulation (`O(m·n²)`)
    and the factorization parallelize well. Measured per iteration with a
    multi-threaded BLAS: a TROTS proton case (`n=1080`, `m=369445`) **3.3 s**,
    a head-and-neck case (`n=9977`, `m=100246`) **45 s** — dense at or ahead of
    every alternative at both scales.

    Two caveats. **(1) Keep BLAS threaded.** With BLAS pinned to one thread the
    ranking inverts and the sparse/matrix-free routes can look faster; that is
    an artifact of the throttle, not representative of a real solve — the dense
    route parallelizes far better than sparse `LDLᵀ` or (ill-conditioned,
    late-IPM) Krylov. **(2) Do not force `linsolve="sparse"` at large `n`.**
    When the `n×n` Gram fills in (dense/overlapping dose matrices, the
    head-and-neck case), the sparse route hands a near-dense `10⁴×10⁴` matrix
    to a *sparse* `LDLᵀ` — the slowest option. `linsolve="sparse"` only helps
    for **moderate `n` with a genuinely sparse Jacobian**, and even there its
    edge over the dense route is small once BLAS is threaded.

## Problem scaling

Badly scaled problems converge faster with gradient-based auto-scaling (IPOPT's
`nlp_scaling_method`). Enable it with the shorthand string:

```python
ipax.Options(scaling="gradient-based")
```

This rescales the objective and each constraint once at `x0` so their gradients
have an ∞-norm of at most `max_gradient` (default `100`); variables and bounds
are left unscaled. The returned `x`, objective, and multipliers are reported in
the **original** problem's units, while `kkt_error`/`constraint_violation`
remain the scaled-space metrics that drove convergence. For a custom threshold:

```python
ipax.Options(scaling=ipax.ScalingOptions(method="gradient-based", max_gradient=50.0))
```

## Higher-order corrections

`corrections` adds predictor–corrector steps that reuse the iteration's KKT
factorization for extra complementarity-target solves, often cutting the
iteration count on problems with active inequalities:

| Value | Scheme |
|---|---|
| `"none"` *(default)* | single Newton/centering direction. |
| `"mehrotra"` | Mehrotra (1992) predictor–corrector (one extra solve). |
| `"gondzio"` | Gondzio (1996) multiple centrality corrections (up to `gondzio_max_corrections` extra solves). |

```python
ipax.Options(corrections="mehrotra")
```

Corrections degenerate to the plain step on problems with no inequalities or
bounds, so they never hurt there. Tune Gondzio via
[`CorrectionsOptions`](../reference.md#ipax.options.CorrectionsOptions).

## Globalization

`globalization` selects the line-search strategy:

- `"filter"` *(default)* — IPOPT-style filter line search on
  $(\theta, \varphi_\mu)$ with second-order correction and a feasibility
  restoration phase. Robust on nonconvex problems.
- `"breedveld"` — a lighter Markov-filter + ratio-control step controller tuned
  for convex/RT-like problems ([`BreedveldOptions`](../reference.md#ipax.options.BreedveldOptions)).

!!! tip "Radiotherapy-scale planning: start from `slack_init_scale=0.1`"
    On large, deeply-infeasible-at-start dose-optimization problems (TROTS-scale:
    `n≈10³`, hundreds of thousands of dose constraints, a warm start that is
    objective-good but far outside the feasible region), the first lever is
    **not** the globalization mode but the
    [slack initialization](#slack-initialization): with the stock
    filter/monotone defaults plus `BarrierOptions(slack_init_scale=0.1)`, the
    TROTS Liver and Prostate-CK cases converge to their published reference
    objectives (`optimal`, KKT ≈ 1e-8) and VMAT gets within ~1 % of its
    reference in a 30-minute budget — where the flat slack floor left the same
    solves clipped at `α ≈ 10⁻³` or livelocked. See the
    [TROTS benchmark page](../benchmarks/trots.md) for the measured table and
    the per-signature routing (the proton cases additionally want a μ-raising
    schedule; the other groups punish one). Budget realistically: these cases
    need on the order of 10²–10³ iterations at seconds-per-iteration on a
    threaded BLAS (keep the default `linsolve="auto"`, above), so set
    `max_iter`/`max_time` accordingly rather than expecting the sub-second
    convergence of small problems. The `"breedveld"` controller remains a
    reasonable alternative step controller on convex RT problems (it was the
    strongest route across the small Prostate-BT cases), but at TROTS scale do
    **not** pair a μ-raising oracle (`"quality"`/`"breedveld"`) with it as a
    blanket setting — on three of the four at-scale TROTS groups that
    combination drives the objective into a large barrier-dominated excursion
    it cannot recover from within any practical budget.

The filter constants
([`LineSearchOptions`](../reference.md#ipax.options.LineSearchOptions)) and the
regularization escalation
([`RegularizationOptions`](../reference.md#ipax.options.RegularizationOptions))
are rarely-touched advanced knobs; the defaults follow Wächter & Biegler (2006)
and Friedlander & Orban (2012), with one deviation: a rejected trial step
backtracks by safeguarded quadratic interpolation of the merit model
(Nocedal & Wright 2006, eq. (3.58)) instead of plain halving —
`LineSearchOptions(backtrack_interpolation=False)` restores W&B's `α ← α/2`.

## Barrier μ schedule

`mu_schedule` selects the μ oracle — how the barrier parameter is driven:

- `"monotone"` *(default)* — Fiacco–McCormick: hold μ fixed until the barrier
  subproblem is solved to `κ_ε·μ`, then reduce it
  (Wächter & Biegler 2006, eq. (7)). The default was decided empirically: in
  the S2MPJ v10 paired A/B it beat probing on every solver/Hessian config
  (3837 vs 3770 correct overall) — an aggressive oracle can crash μ faster
  than the duals converge, stalling just above tolerance.
- `"probing"` — Mehrotra σ-rule: an affine (predictor) probe sets
  σ = (μ_aff/μ)³ (Mehrotra 1992; NWW 2009, eqs. (3.2)–(3.5)). Without a
  corrector this costs one extra KKT solve per iteration; with corrections
  active the affine solve is shared. Rescues 12–18 problems per config that
  monotone misses — worth trying when monotone stalls.
- `"adaptive"` — LOQO centrality rule: μ = σ·(average complementarity), where σ
  grows with the deviation of the smallest complementarity product from the
  average (Nocedal, Wächter & Waltz 2009, eq. (3.6)). Re-targeted every
  iteration; μ may increase but never above 0.8× the current complementarity.
- `"breedveld"` — duality-gap update μ = σ(α)·(average complementarity) with
  σ = ((α−1)/(α+10))² built from the last accepted steplength
  (Breedveld et al. 2017, eqs. (10)–(12)): full steps drive μ superlinearly to
  zero, blocked steps re-center. Pairs naturally with
  `globalization="breedveld"` on convex/RT-like problems.

```python
ipax.Options(mu_schedule="adaptive")
```

The oracle is orthogonal to [`corrections`](#higher-order-corrections): the
corrector improves the *step* toward whatever μ the oracle picked (Nocedal,
Wächter & Waltz 2009 — "the corrector is not part of the selection of the
barrier parameter"), so e.g. the LOQO oracle can steer Gondzio corrections.

The non-monotone oracles run in *free mode*, safeguarded twice:

- **KKT-error fallback** (NWW 2009 §5.1, Algorithm A): if the scaled KKT error
  fails to drop below `fallback_kappa` × (max of the last `fallback_window`+1
  free-mode values), the oracle is suspended and μ is handled monotonically —
  re-initialized at `fallback_mu_factor` × (average complementarity) — until
  the error recovers. Defaults follow the paper (κ = 0.9999, l_max = 5,
  factor 0.8); set `BarrierOptions(fallback="never")` for pure free mode.
  Iterates are never rolled back — the filter line search already globalizes
  each step, so the safeguard gates only the μ rule (matching IPOPT's
  `adaptive_mu_globalization="kkt-error"`).
- **Centrality floor**: μ never drops below
  `kappa_centrality` × max(dual, primal infeasibility) (default `1e-2`).
  El-Bakry et al. (1996)'s convergence theory requires the complementarity gap
  not to vanish faster than the KKT residual; without the floor an aggressive
  oracle can crush μ near a saddle while the iterate is far from stationary,
  pinning it to the boundary with no barrier left to re-center. The
  complementarity component is excluded from the floor, so superlinear μ
  decrease near a solution is unimpeded. `kappa_centrality=0.0` disables it.

### Slack initialization

`slack_init_scale` (default `0.0`) rescales the initial slack floor. By default a
violated inequality's slack starts at a fixed `1e-2`; on a deeply-infeasible start
with many violated or near-active constraints, those slacks are all pinned near
zero, so the first Newton direction drives them toward their infeasible target
`s = −g < 0` and the fraction-to-boundary rule clips the primal step to ~`1e-3`
for many iterations. Setting `slack_init_scale > 0` raises the floor to
`max(1e-2, slack_init_scale·max|g(x₀)|)`, giving the slacks — and, through
`y = μ_init/s`, the initial multipliers — a scale matched to the constraints
instead of a fixed constant.

```python
ipax.Options(barrier=BarrierOptions(slack_init_scale=0.1))
```

It is **opt-in** because the benefit is specific to badly-infeasible,
many-constraint starts; on the general corpus it is net-neutral, so the default
leaves the solver unchanged. Radiotherapy-scale problems are the motivating
case, and there the lever is decisive, not incremental: with `0.1` the TROTS
Liver and Prostate-CK cases go from grinding/livelocked to `optimal` at their
published reference objectives, VMAT reaches feasibility in ~120 instead of
~180 iterations, and the proton cases unclip from the `α ≈ 10⁻³` grind (see
the [TROTS benchmark page](../benchmarks/trots.md)). A value in `[0.05, 0.5]`
matches the constraint scale on such problems; `0.0` keeps the flat floor.

## Equality-multiplier repair

`equality_dual_repair` (default `None`, i.e. off) guards against equality
multipliers that diverge on a **rank-deficient constraint Jacobian**. When `∇c`
loses rank the multipliers are under-determined — the dual block is singular, so
`‖Δy‖` is limited only by the `delta_c` regularization and one step can inject
`‖c‖/δ_c ≈ 1e7` of pure null-space noise. The filter line search cannot detect
this, because it tests only the constraint violation `θ` and the barrier objective
`φ`: a step that improves feasibility while destroying stationarity passes at full
length. The damage compounds, since the multipliers feed the L-BFGS Lagrangian
curvature pairs and every subsequent step system.

The signature is a solve that reaches a feasible point and reports `stalled` with
a dual infeasibility orders of magnitude above tolerance. It is most visible on
feasibility problems (`min 0` subject to `c(x) = 0`), where the exact multipliers
are provably `y* = 0`: CUTEst `NONSCOMPNE` has Jacobian rank 24 of 25 at its
starting point and reaches `‖y‖_∞ = 1e8`.

Setting the option replaces `y_eq` with the least-squares estimate
`argmin_y ‖∇f + Aᵀy‖` whenever the current multipliers' stationarity residual
exceeds this factor times the estimate's:

```python
ipax.Options(regularization=RegularizationOptions(equality_dual_repair=1e10))
```

The factor must be `>= 1.0`, and it should be **large**. The gate is a ratio
against the residual the iterate can actually justify, not an absolute magnitude,
which is what leaves ordinary primal–dual coupling alone; it is also
self-limiting, since a repaired multiplier no longer trips the test. Repairing
eagerly is actively harmful — it overwrites legitimate coupling and, near
convergence, pins the dual residual at the repair's own accuracy, degrading
problems that would otherwise reach `optimal` down to `acceptable`.

It is **opt-in**, and the full-corpus sweep is the reason. Measured at `1e10`
across the whole S2MPJ corpus on the three L-BFGS routes (3300 solves), it is a
**net −1 correct**: 22 problems fixed against 23 broken (dense +2, Krylov −3,
sparse ±0).

The fixes are exactly the family it targets — `VANDERM1`, `VANDERM2`, `COOLHANS`,
`ARTIF`, `LAKES` and `HYDCAR20` reach `optimal`, five of them long-standing gaps
against IPOPT. The breakages are two separate effects worth knowing before you
enable it:

- **Cost.** Eleven of them are wall-clock timeouts. The repair runs a
  least-squares solve on every accepted step of every equality-constrained
  problem; iteration counts on problems solved in both arms move by only −0.5%,
  so this is per-iteration overhead, not slower convergence. The Krylov route
  suffers most.
- **Trajectory changes.** The rest cluster in the EIGEN family and `RES`, which
  converge to a *different eigenvalue* once the multipliers are reset. More
  broadly, 39 of 49 objective drifts are on problems with no documented optimum,
  so the headline count cannot see them, and they move in both directions
  (`LAKES` improves from `2.8e11` to `3.5e5`; `GASOIL` degrades from `1.08` to
  `12.0`).

Treat it as a targeted remedy rather than a general setting: enable it when a
solve reaches a feasible point and reports `stalled` with a large dual
infeasibility, or when the constraints are known to be redundant or
rank-deficient, and verify on your own problem.

!!! warning "On the Krylov route, pair it with the L-BFGS preconditioner"

    The entire corpus-level negative is the Krylov route (dense +2, sparse ±0,
    Krylov −3), and it is a **preconditioner mismatch**, not a cost of the repair
    itself. Repairing the multipliers to their correct value on a zero-objective
    problem leaves the Lagrangian Hessian genuinely near zero, so the condensed
    system degenerates and the default diagonal (Jacobi) preconditioner has
    nothing to work with — the per-iteration Krylov solve gets 4–11x more
    expensive (`DRCAVTY1`: 155 ms → 1688 ms) and problems that solved in seconds
    hit the time limit.

    `KrylovOptions(preconditioner="lbfgs")` removes it: `METHANL8` goes from
    `max_time` to `optimal` in 3 iterations, `HYDCAR6` from `max_time` to
    `optimal` in 4.

    ```python
    ipax.Options(
        linsolve="krylov",
        krylov=KrylovOptions(preconditioner="lbfgs"),
        regularization=RegularizationOptions(equality_dual_repair=1e10),
    )
    ```

## Verbosity

`verbose` (an integer `0`–`6`) opts in to a console handler with progressively
more detail. `0` is silent. See [Monitoring & diagnostics](diagnostics.md) for
the full ladder and for attaching your own logging handler instead.

```python
ipax.Options(verbose=2)  # result summary + per-iteration table
```

## Derivative resolution toggles

`enable_autodiff` and `enable_finite_diff` (both `True` by default) gate the
fallback chain. Set `enable_finite_diff=False` to force an error rather than
silently fall back to (slow, less accurate) finite differences when an analytic
or autodiff derivative is missing.
