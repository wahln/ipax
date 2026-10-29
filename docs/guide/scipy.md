# Coming from SciPy

If your model is already written for
[`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html),
you can run it on ipax without rewriting it.
[`minimize_ipax`](../reference.md#ipax.interfaces.scipy.minimize_ipax) accepts
the same arguments and returns an `OptimizeResult`. It works both as a
standalone function and as a custom `method`:

```python
import numpy as np
from scipy.optimize import NonlinearConstraint, minimize, rosen, rosen_der

import ipax
from ipax.interfaces.scipy import minimize_ipax

ring = NonlinearConstraint(lambda x: x @ x, 0.0, 1.5, jac=lambda x: 2 * x)

res = minimize(
    rosen, np.zeros(2), jac=rosen_der, constraints=ring, method=minimize_ipax
)
res = minimize_ipax(rosen, np.zeros(2), jac=rosen_der, constraints=ring)  # same
```

Install the optional dependencies with `pip install "ipax[scipy]"`. The adapter
lives in `ipax.interfaces`, so `import ipax` never imports SciPy.

## What is translated

| SciPy | ipax |
|---|---|
| `fun`, `jac`, `args` (`jac=True`: `fun` returns `(f, g)`) | objective and gradient. Without a callable `jac`, ipax uses finite differences |
| `Bounds` or `(min, max)` pairs (`None` = unbounded) | variable bounds |
| `LinearConstraint(A, lb, ub)`, dense or `scipy.sparse` | constant-data `linear_eq` / `linear_ineq`. Sparse `A` stays sparse |
| `NonlinearConstraint(fun, lb, ub, jac, hess)` | nonlinear equalities/inequalities |
| dict `{'type': 'eq' \| 'ineq', 'fun', 'jac', 'args'}` | the same; `'ineq'` means `fun(x) ≥ 0` |
| `hess` / `hessp` plus each constraint's `hess(x, v)` | exact Lagrangian Hessian |
| `tol` | the scaled KKT tolerances of [`OptimalityConditionOptions`](../reference.md#ipax.options.OptimalityConditionOptions) |
| `options={'maxiter': …, 'disp': …}` | `Options.max_iter`, `Options.verbose` |
| `callback(intermediate_result)`, `callback(xk, state)` or `callback(xk)`; raise `StopIteration` to stop (the `(xk, state)` form may also return `True`, as in trust-constr) | the per-iteration callback, also called at the starting point (iteration 0) as `trust-constr` does (status `"stopped"`) |

A constraint row with `lb == ub` becomes an equality. Each finite side of any
other row becomes a one-sided inequality.

**Hessians.** The exact Lagrangian Hessian is used only when the objective has
`hess` (or `hessp`) *and* every `NonlinearConstraint` has a callable
`hess(x, v)`. Otherwise ipax uses its L-BFGS approximation. If you passed an
objective Hessian that therefore goes unused, you get an `OptimizeWarning`. A
SciPy `HessianUpdateStrategy` (`BFGS()`, `SR1()`) or a finite-difference string
is also replaced by L-BFGS, with a warning.

**Other ipax options.** Pass any [`Options`](../reference.md#ipax.options.Options)
field by name, or a whole `Options` object as `ipax_options`:

```python
res = minimize(
    fun,
    x0,
    method=minimize_ipax,
    options={"maxiter": 500, "globalization": "breedveld"},
)
res = minimize(
    fun,
    x0,
    method=minimize_ipax,
    options={"ipax_options": ipax.Options(mu_schedule="adaptive")},
)
```

Unknown keys raise an `OptimizeWarning`, as SciPy's own methods do.

## Reading the result

Besides the usual `x`, `fun`, `success`, `status`, `message`, `nit`, `nfev`,
`njev` and `nhev` (plus `jac`, the objective gradient at `x`, when you gave a
gradient callable), the result carries:

- `v` and `constr`: one array per input constraint, in input order. The
  multipliers follow `trust-constr`'s sign convention, so `∇f + Σ J_kᵀ v_k +
  bound_multipliers = 0` at a KKT point. An active upper side is positive and
  an active lower side negative.
- `bound_multipliers`: `z_U − z_L`, one entry per variable.
- `constr_violation`: the largest violation in your units, bounds included.
- `ipax_status` and `ipax_result`: the [`Status`](../reference.md#ipax.result.Status)
  value and the full [`Result`](../reference.md#ipax.result.Result), with
  iteration history, routes and KKT residuals.

`status` is `0` for optimal and `1` for the iteration limit, as in most SciPy
methods. The other codes are listed in `ipax.interfaces.scipy.STATUS_CODES`.

The counters count calls of your functions: `nfev` objective evaluations, `njev`
gradient requests (with `jac=True` these share `fun`'s evaluation), and `nhev`
Hessian assemblies (with `hessp`, one per iteration, not one per product).

If some constraints have a `jac` and others do not, the missing ones are
differenced individually and `ipax_result.derivative_sources` still reports
the block as `"analytic"`; only when no constraint has a `jac` does ipax's own
fallback run and report `"finite-diff"`.

### Differences from trust-constr

The call is the same as for `method="trust-constr"`. The result differs in three
places:

- trust-constr appends the bound multipliers to `v` (and the bounded `x` to
  `constr`) as a last entry. Here `v` and `constr` have exactly one entry per
  constraint you passed; the bound multipliers are `bound_multipliers`.
- trust-constr's `jac` is the list of constraint Jacobians and the objective
  gradient is `grad`. Here `jac` is the objective gradient, as in BFGS and
  SLSQP.
- trust-constr's optimizer-specific fields (`tr_radius`, `barrier_parameter`,
  `cg_niter`, …) are absent; the corresponding ipax diagnostics are in
  `ipax_result`.

## Array backends

The adapter does not assume NumPy. It takes the Array-API namespace, device
and dtype from `x0`, and the solve and every array in the result stay in it. A
PyTorch, JAX or CuPy `x0` therefore runs on that backend and device. A missing
gradient or Jacobian comes from that backend's autodiff where it has one
(PyTorch, JAX), and from finite differences otherwise:

```python
import torch

x0 = torch.zeros(2, dtype=torch.float64)
res = minimize_ipax(lambda x: torch.sum((x - 1.0) ** 2), x0)  # autodiff gradient
type(res.x)  # torch.Tensor
```

Today this works when you call `minimize_ipax` directly. SciPy 1.17's
`minimize` still converts `x0` with `np.asarray` before it calls a custom
method, so through `minimize(..., method=minimize_ipax)` the arrays are NumPy.
SciPy is moving its optimizers to the Array API standard, and nothing in the
adapter relies on that conversion.

Some inputs are NumPy by construction and are converted into `x0`'s namespace,
device and dtype:

- `LinearConstraint` and `Bounds` data, which SciPy casts to float64 NumPy
  arrays. A float32 `x0` stays float32.
- `scipy.sparse` matrices, which still reach ipax as a sparse `CSROperator`.
- `lb` / `ub` of every constraint. These fix which rows are equalities and
  which inequalities, so they must be convertible to NumPy (scalars,
  sequences, NumPy or CPU arrays).

A `scipy.sparse.linalg.LinearOperator` Hessian only works with NumPy `x`.
Sparse input reaches the solver as a sparse operator, so it needs a backend
with a sparse adapter (NumPy/SciPy, PyTorch, CuPy and JAX have one;
array-api-strict does not). A `scipy.sparse` Jacobian from one constraint can be
combined with dense Jacobians from others only while those live on the host.

## Limits

A Hessian you leave out is approximated by L-BFGS. For matrix-free operators,
warm starts, or anything else beyond the `minimize` arguments, write a
[`Problem`](problems.md) and call `ipax.solve` directly. The translation table
above shows what each argument becomes there.

A few things are not supported:

- `keep_feasible` on constraints is not honored; you get a warning. Bounds are
  always kept strictly feasible.
- Constraint Jacobians that are `scipy.sparse.linalg.LinearOperator`s are
  rejected. Return a dense array or a sparse matrix instead.
