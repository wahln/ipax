# The Problem interface

`ipax.Problem` is capability-graded: only `n_vars` and
`objective` are mandatory. Everything else is optional and resolved by the
derivative-precedence chain.

## Required

| Member | Meaning |
|---|---|
| `n_vars` | number of variables |
| `objective(x)` | scalar `f(x)` |

## Optional, with precedence

| Member | Resolution chain |
|---|---|
| `gradient(x)` | analytic → autodiff → finite-difference |
| `eq_constraints` / `eq_jacobian` | analytic → autodiff → finite-difference |
| `ineq_constraints` / `ineq_jacobian` | analytic → autodiff → finite-difference |
| `lagrangian_hessian(x, y_eq, y_ineq, σ)` | analytic → autodiff-HVP → **L-BFGS** |

## Declared separately

- **Bounds** `bounds() -> (x_L, x_U)` — handled directly by the barrier, not as
  constraints.
- **Linear constraints** `linear_eq()`, `linear_ineq()` — constant Jacobian,
  zero Hessian term, assembled once.

Returning a `LinearOperator` from any Jacobian/Hessian is the recommended path
for `1e5`-scale problems — it lets the same problem run dense, matrix-free, or
sparse.
