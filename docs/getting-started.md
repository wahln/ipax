# Getting started

## Install

```bash
pip install -e ".[numpy]"     # core + NumPy backend
pip install -e ".[dev]"       # full dev environment
```

CUDA dependencies are user-managed because the CuPy wheel must match the local
CUDA toolkit. Install the appropriate CuPy package and cuDSS runtime first, then
install the optional nvmath bindings:

```bash
# Example only: choose the CuPy package matching your installed CUDA toolkit.
pip install cupy-cuda12x
pip install -e ".[sparse-cuda]"
```

If nvmath or cuDSS cannot be loaded, the CuPy sparse adapter remains available
through its solve-only `cupyx.scipy.sparse.linalg.spsolve` fallback. Sparse KKT
inertia requires a working cuDSS runtime.

## Define a problem

Subclass `ipax.Problem`. Only `n_vars` and `objective` are required.

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

Switch backends by switching the array type — pass a `torch.Tensor` as `x0` and
the entire solve runs in PyTorch.

## Supplying derivatives

Override any of `gradient`, `eq_constraints`/`eq_jacobian`,
`ineq_constraints`/`ineq_jacobian`, `lagrangian_hessian`. Anything you omit is
resolved automatically (analytic → autodiff → finite-difference; Hessian →
autodiff-HVP → L-BFGS). The chosen source is reported in `res.derivative_sources`.

## Linear vs nonlinear constraints

Declare *linear* constraints separately via `linear_eq()` / `linear_ineq()`:
their Jacobian is constant and assembled once, and they contribute no Hessian
term — a real performance lever at scale.
