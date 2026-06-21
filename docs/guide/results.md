# Interpreting results

`solve` returns a frozen [`Result`](../reference.md#ipax.result.Result). This
page explains every field and how to check that a solve actually converged.

```python
res = ipax.solve(problem, x0)
if res.success:
    print(res.x, res.objective)
else:
    print("did not converge:", res.status, res.message)
```

## Did it converge?

Check `res.success` first — it is `True` only for
[`Status.OPTIMAL`](#status) and the explicitly enabled
[`Status.ACCEPTABLE`](#status). The `status` enum then tells you *which*
stopping condition fired, and `message` carries a short human-readable note.

```python
from ipax import Status

match res.status:
    case Status.OPTIMAL | Status.ACCEPTABLE:
        ...                      # res.x is usable
    case Status.MAX_ITER | Status.MAX_TIME:
        ...                      # ran out of budget; inspect res.kkt_error
    case _:
        ...                      # infeasible / numerical trouble / stopped
```

### Status

| Status | `success` | Meaning |
|---|:---:|---|
| `OPTIMAL` | ✅ | scaled-KKT optimality conditions met in one iteration |
| `ACCEPTABLE` | ✅ | acceptable conditions held for the configured consecutive iterations |
| `INFEASIBLE` | ❌ | restoration/feasibility detection concluded the problem is infeasible (or bounds with `x_L > x_U`) |
| `UNBOUNDED` | ❌ | objective appears unbounded below |
| `MAX_ITER` | ❌ | hit `Options.max_iter` |
| `MAX_TIME` | ❌ | hit `Options.max_time` |
| `RESTORATION_FAILED` | ❌ | feasibility restoration could not make progress |
| `NUMERICAL_ERROR` | ❌ | unrecoverable numerical failure in a step |
| `STOPPED` | ❌ | a [user callback](diagnostics.md#iteration-callbacks) requested a stop |

A non-success status is not always "wrong": `MAX_ITER` with a small
`kkt_error` often means the point is fine but the tolerance was strict. Always
look at the residuals before discarding a result.

## The solution and multipliers

| Field | Meaning |
|---|---|
| `x` | the primal solution, in the problem's backend/namespace |
| `objective` | `f(x)` at the solution, in the **original** problem's units |
| `y_eq` | equality multipliers (free sign), or `None` |
| `y_ineq` | inequality multipliers `≥ 0`, or `None` |
| `z_lower`, `z_upper` | bound multipliers `≥ 0` (lower/upper), or `None` |

A multiplier block is `None` when the corresponding constraints don't exist.
The Lagrangian sign convention matches the standard form `c(x)=0`, `g(x)≤0`,
`x_L≤x≤x_U`:

$$
\nabla f + \nabla c^\top y_{eq} + \nabla g^\top y_{ineq} - z_L + z_U = 0 .
$$

Inequality and bound multipliers are non-negative; a (near-)zero entry means
that constraint is inactive, a strictly positive entry means it is active and
gives its shadow price. With [scaling](options.md#problem-scaling) enabled the
multipliers are still reported in original units.

```python
active = res.y_ineq > 1e-6          # which inequalities bind at the solution
```

## Convergence diagnostics

| Field | Meaning |
|---|---|
| `kkt_error` | aggregate scaled KKT ∞-norm that drove termination |
| `dual_infeasibility` | scaled stationarity residual component |
| `primal_infeasibility` | scaled constraint-violation component |
| `complementarity` | scaled complementarity component |
| `constraint_violation` | scaled `‖(c, g+s)‖` feasibility measure |

`kkt_error` is the maximum of the three component residuals; comparing the three
tells you *which* condition is limiting. A solve that stalls at a large
`dual_infeasibility` but tiny `primal_infeasibility`, for example, is exactly the
case the [acceptable termination](options.md#termination) mode is designed for.
These remain **scaled-space** quantities when problem scaling is on.

## Performance and provenance

| Field | Meaning |
|---|---|
| `n_iter` | outer interior-point iterations taken |
| `solve_time` | total wall-clock seconds for the whole `solve` call |
| `linear_solver` | the solver route actually used, e.g. `"dense"`, `"krylov (cg, pc=jacobi)"`, `"sparse [Feral LDL^T (CPU)]"` |
| `derivative_sources` | how each derivative was obtained (see below) |

[`derivative_sources`](../reference.md#ipax.result.DerivativeSources) is worth a
look on a first run — it confirms whether your analytic derivatives were picked
up or whether the solver fell back to autodiff/finite-difference:

```python
print(res.derivative_sources)
# DerivativeSources(gradient='analytic', eq_jacobian='n/a',
#                   ineq_jacobian='finite-diff', hessian='lbfgs')
```

## Iteration history

`res.history` is a tuple of
[`IterationRecord`](../reference.md#ipax.result.IterationRecord)s, one per
iteration, holding the objective, `mu`, `theta` (constraint violation), the KKT
error and its components, step lengths, applied regularization, and per-row
timing splits. Use it to plot convergence or diagnose a slow solve:

```python
import matplotlib.pyplot as plt
plt.semilogy([r.kkt_error for r in res.history])
```

To react *during* the solve instead of after, pass a callback — see
[Monitoring & diagnostics](diagnostics.md#iteration-callbacks).
