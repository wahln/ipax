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
| `lagrangian_hessian(x, y_eq, y_ineq, σ)` | analytic → **L-BFGS** (default `hessian="auto"`); autodiff-HVP only with `hessian="autodiff-hvp"` |

Overriding an optional method declares it provided; `provides()` returning a
set of method names overrides that inference (for delegating wrappers).
`lagrangian_hessian` is always called with four positional arguments; `σ` is
not 1 under the default gradient-based scaling and must scale the objective
term.

## Declared separately

- **Bounds** `bounds() -> (x_L, x_U)` — handled directly by the barrier, not as
  constraints.
- **Linear constraints** `linear_eq()`, `linear_ineq()` — constant Jacobian,
  zero Hessian term, assembled once.

Returning a `LinearOperator` from any Jacobian/Hessian is the recommended path
for `1e5`-scale problems — it lets the same problem run dense, matrix-free, or
sparse.
