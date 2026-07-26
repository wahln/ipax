# Backends & hardware

`ipax` never imports a concrete array library in its core. It infers the
Array-API namespace from the arrays your `Problem` returns and the `x0` you pass
to `solve`, so **the backend is whatever `x0` is**:

```python
import torch

res = ipax.solve(problem, x0=torch.tensor([-1.2, 1.0], dtype=torch.float64))
# the entire solve runs in PyTorch; res.x is a torch.Tensor
```

No `Problem` code changes between backends, provided the objective and any
derivatives are written in backend-agnostic Array-API operations (use `x`'s
namespace, not `np.*`). For a backend-agnostic helper inside a problem method:

```python
from ipax.backend.namespace import array_namespace


def objective(self, x):
    xp = array_namespace(x)
    return xp.sum(x**2)
```

## Supported backends

| Backend | Device | Autodiff | Sparse-direct | Install |
|---|---|:---:|:---:|---|
| **NumPy** | CPU | — (finite-diff) | SciPy (Feral LDLᵀ / SuperLU) | `pip install -e ".[numpy]"` |
| **PyTorch** | CPU / CUDA | ✅ | via device routing | `pip install -e ".[torch]"` |
| **JAX** | CPU / GPU | ✅ | via device routing | `pip install -e ".[jax]"` |
| **CuPy** | CUDA | — (finite-diff) | cuDSS (nvmath) / `spsolve` | user-managed CUDA wheel |

"Autodiff" means the gradient/Jacobian/Hessian-vector fallback chain can use the
backend's automatic differentiation; without it the chain falls through to
finite differences. The Hessian default (L-BFGS) works on every backend
regardless.

## Float precision

`ipax` defaults to `float64` and never hard-codes a dtype — it reads dtype from
your inputs. Create `x0` (and any problem data) as `float64` for full accuracy:
a `float32` start propagates `float32` throughout and limits the achievable KKT
tolerance. On PyTorch this means `dtype=torch.float64`; note PyTorch's global
default is `float32`.

## GPU

Run on GPU by placing `x0` (and the arrays your problem returns) on the device:

```python
x0 = torch.tensor([...], dtype=torch.float64, device="cuda")
res = ipax.solve(problem, x0)  # solve executes on the GPU
```

The dense and matrix-free Krylov routes are pure Array-API and run on whatever
device the arrays live on. The sparse-direct route on CUDA uses NVIDIA **cuDSS**
(via `nvmath-python`); without a cuDSS runtime the CuPy adapter falls back to
`cupyx.scipy.sparse.linalg.spsolve` and cannot report inertia. CUDA itself is
user-managed so the CuPy wheel matches your toolkit — see
[Getting started](../getting-started.md#install).

## Choosing a solver route per backend

The [`linsolve`](options.md#linear-solver) option is independent of the backend,
but not every route is available everywhere:

- **dense** / **krylov** — pure Array-API; available on all backends that expose
  `xp.linalg` (dense) or just `matvec` (krylov).
- **sparse** — needs a backend sparse adapter. SciPy (CPU) and CuPy/cuDSS (CUDA)
  own concrete factorizations; **Torch and JAX reuse them by device**. The
  `DeviceRoutingSparseAdapter` reads the COO buffer's device and dispatches
  host arrays to SciPy and CUDA arrays to cuDSS — zero-copy via DLPack where the
  libraries allow — handing results back in your namespace. Routing is by
  *device*, not library name.

So a Torch-CPU problem factors through Feral/SuperLU and a JAX-GPU problem
through cuDSS, with no change to your code. See
[The linear-algebra layer](../concepts/linalg.md#sparsity-as-an-adapter-concern).

## Checking what a backend supports

`capabilities` probes a namespace for the optional features `ipax` cares about
(its `linalg` functions, sparse adapter, autodiff, devices):

```python
import numpy as np
from ipax.backend.namespace import array_namespace, capabilities

xp = array_namespace(np.empty(0))
caps = capabilities(xp)
print(caps.has_sparse_adapter, caps.supports_autodiff)
```

This is the same probe the `"auto"` solver selection uses internally.
