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

Four independent sources stop the solve, checked in priority order. The first
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
        dual_inf_tol=1e-4, compl_inf_tol=1e-4,   # loose optimality, but…
        constr_viol_tol=1e-8,                     # …tight feasibility, in one step
    )
)
```

**Acceptable** ([`AcceptableStoppingOptions`](../reference.md#ipax.options.AcceptableStoppingOptions))
mirrors the same conditions but requires them to hold for `n_iter` *consecutive*
iterations, and reports [`Status.ACCEPTABLE`](results.md#status) (also a
success). It is **off by default** and is the right tool when a
dual-infeasibility-dominated residual plateaus while the objective and primal
feasibility have already settled:

```python
ipax.Options(
    acceptable=ipax.AcceptableStoppingOptions(
        dual_inf_tol=1.0,        # tolerate the stuck dual infeasibility
        constr_viol_tol=1e-6,
        f_rel_change_tol=1e-7,
        n_iter=5,
    )
)
```

**Limits.** `max_iter` (default `3000`) reports `Status.MAX_ITER`; `max_time`
in seconds (default `None`, disabled) reports `Status.MAX_TIME`. Both are
checked at iteration boundaries.

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

The barrier μ schedule (`mu_schedule`), the filter constants
([`LineSearchOptions`](../reference.md#ipax.options.LineSearchOptions)), and the
regularization escalation
([`RegularizationOptions`](../reference.md#ipax.options.RegularizationOptions))
are rarely-touched advanced knobs; the defaults follow Wächter & Biegler (2006)
and Friedlander & Orban (2012).

## Verbosity

`verbose` (an integer `0`–`6`) opts in to a console handler with progressively
more detail. `0` is silent. See [Monitoring & diagnostics](diagnostics.md) for
the full ladder and for attaching your own logging handler instead.

```python
ipax.Options(verbose=2)   # result summary + per-iteration table
```

## Derivative resolution toggles

`enable_autodiff` and `enable_finite_diff` (both `True` by default) gate the
fallback chain. Set `enable_finite_diff=False` to force an error rather than
silently fall back to (slow, less accurate) finite differences when an analytic
or autodiff derivative is missing.
