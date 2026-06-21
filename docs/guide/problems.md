# Defining a problem

Every solve starts from a [`Problem`](../reference.md#ipax.problem.base.Problem).
This page is the practical how-to; see [The Problem interface](../concepts/problem.md)
for the conceptual contract and the [API reference](../reference.md) for signatures.

You have two ways to define one:

- **Subclass `ipax.Problem`** — best when the model has structure (constraints,
  a structured Hessian) or you want to keep state on the object.
- **Use a ready-made adapter** — [`FunctionProblem`](../reference.md#ipax.problem.function.FunctionProblem)
  (wrap plain callables), [`QuadraticProblem`](../reference.md#ipax.problem.function.QuadraticProblem)
  (`½ xᵀQx + cᵀx`), or [`LinearProblem`](../reference.md#ipax.problem.function.LinearProblem)
  (`cᵀx`).

## The minimum

Only `n_vars` and `objective` are required. Everything else is optional.

```python
import numpy as np
import ipax

class Rosenbrock(ipax.Problem):
    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

res = ipax.solve(Rosenbrock(), x0=np.array([-1.2, 1.0]))
print(res.status, res.x)
```

Nothing in the class names a concrete array library: `x` arrives in whatever
backend `x0` carries, and the objective is written in backend-agnostic
operations. Pass a `torch.Tensor` as `x0` and the same class solves in PyTorch.
See [Backends & hardware](backends.md).

The same problem without subclassing:

```python
problem = ipax.FunctionProblem(
    n_vars=2,
    objective=lambda x: 100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
)
```

## Supplying derivatives

Override (or pass to `FunctionProblem`) any of `gradient`,
`eq_constraints`/`eq_jacobian`, `ineq_constraints`/`ineq_jacobian`, and
`lagrangian_hessian`. Anything you leave out is filled automatically by the
**derivative-precedence chain**:

| Quantity | Resolution order |
|---|---|
| gradient, Jacobians | analytic → autodiff → finite-difference |
| Lagrangian Hessian | analytic → autodiff-HVP → **L-BFGS** |

Autodiff is only attempted on backends that support it (PyTorch, JAX); on
NumPy/CuPy the chain falls through to finite differences. The Hessian default is
L-BFGS on every backend. Whatever was actually used is reported back on the
result:

```python
res = ipax.solve(problem, x0)
print(res.derivative_sources)
# DerivativeSources(gradient='finite-diff', ..., hessian='lbfgs')
```

Providing an analytic gradient is the single highest-value derivative to supply:
it removes `n+1` extra objective evaluations per finite-difference gradient and
is usually cheap to write.

!!! tip "Exact Hessian"
    Supply `lagrangian_hessian` and pass `Options(hessian="exact")` to use it.
    The default `hessian="lbfgs"` ignores an analytic Hessian; see
    [Configuring the solver](options.md#hessian).

## Constraints

`ipax` solves the standard form

$$
\min_x f(x)\quad\text{s.t.}\quad c(x)=0,\; g(x)\le 0,\; x_L\le x\le x_U .
$$

There are three kinds of constraint, declared independently. Define only the
ones you need; an undeclared block simply doesn't exist.

### Bounds

Return `(x_L, x_U)` from `bounds()`. Either side may be `None` (that side
unbounded) and individual entries may be `±inf`. Bounds are handled directly by
the log-barrier, not as general inequalities, so they are cheap.

```python
class Boxed(ipax.Problem):
    @property
    def n_vars(self): return 3
    def objective(self, x): ...
    def bounds(self):
        lo = np.zeros(3)            # x ≥ 0
        hi = np.array([np.inf, 5.0, np.inf])
        return (lo, hi)
```

### Nonlinear constraints

Implement the *value* method (`eq_constraints` / `ineq_constraints`); its
presence is what makes the constraint block exist. The matching Jacobian is
optional and filled by the precedence chain if omitted.

```python
class Constrained(ipax.Problem):
    @property
    def n_vars(self): return 2
    def objective(self, x): return x[0] ** 2 + x[1] ** 2

    def eq_constraints(self, x):           # c(x) = 0
        return x[0] + x[1] - 1.0           # returns a length-1 array-like

    def ineq_constraints(self, x):         # g(x) ≤ 0
        return x[0] ** 2 - x[1]            # i.e. x[1] ≥ x[0]²
```

Use the sign convention exactly: equalities are `c(x) = 0`, inequalities are
`g(x) ≤ 0`. Rewrite `h(x) ≥ 0` as `-h(x) ≤ 0`.

### Linear constraints

Constraints whose Jacobian is a constant matrix should be declared *separately*
through `linear_eq()` / `linear_ineq()`. They are assembled once, contribute no
Hessian term, and never get re-evaluated — a real performance lever at scale.

```python
def linear_eq(self):
    A = np.array([[1.0, 1.0, 1.0]])        # A x = b
    b = np.array([1.0])
    return (A, b)

def linear_ineq(self):
    A = np.array([[1.0, 0.0, 0.0]])
    lower = np.array([-np.inf])            # two-sided:  lower ≤ A x ≤ upper
    upper = np.array([2.0])
    return (A, lower, upper)
```

`A` may be a dense array **or** a [`LinearOperator`](../reference.md#ipax.backend.operators.LinearOperator),
so a large structured constraint matrix never has to be materialized.

## Returning operators instead of matrices

Any Jacobian or the Hessian may return a
[`LinearOperator`](../concepts/linalg.md) rather than a dense array. This is the
recommended path at `1e4`–`1e5` scale: the operator exposes a `matvec` (and
optionally `rmatvec`), so the matrix-free Krylov route never forms an `n×n`
object. The same `Problem` then runs dense, matrix-free, or sparse with no code
change — only the [linear-solver choice](options.md#linear-solver) differs.

```python
from ipax.backend.operators import MatrixFreeJacobian

def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
    n = self.n_vars
    return MatrixFreeJacobian((n, n), matvec=lambda v: self._hess_prod(x, v))
```

For sparse-direct factorization the operator additionally exposes COO structure;
the built-in `Dense`, `Diagonal`, and `Identity` operators already do, and the
`SparseOperator` adapter wraps a backend sparse matrix. See
[The linear-algebra layer](../concepts/linalg.md).

## Quadratic and linear programs

For a QP or LP, skip the subclass entirely:

```python
Q = np.array([[4.0, 1.0], [1.0, 3.0]])
c = np.array([-1.0, -2.0])
qp = ipax.QuadraticProblem(Q, c)          # min ½ xᵀQx + cᵀx, exact gradient & Hessian

lp = ipax.LinearProblem(
    c=np.array([1.0, 1.0]),
    bounds=(np.zeros(2), None),
    linear_eq=(np.array([[1.0, 2.0]]), np.array([3.0])),
)
```

`QuadraticProblem` reports an exact gradient (`Qx + c`) and Hessian (`Q`), so
`hessian="exact"` uses them directly. `Q` may be a `LinearOperator` for a
matrix-free quadratic.

## Next steps

- [Configuring the solver](options.md) — termination, Hessian mode, solver route.
- [Interpreting results](results.md) — multipliers, KKT residuals, status codes.
- Runnable end-to-end scripts live in [`examples/`](https://github.com/wahln/ipax/tree/main/examples).
