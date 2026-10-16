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

cuDSS uses *static pivoting*: a pivot too small to divide by is replaced by a
tiny `±ε`, so the factorization is of a slightly perturbed matrix. When cuDSS
reports such perturbed pivots, the adapter iteratively refines each solve
against the unperturbed matrix (IPOPT's stopping rules, at most 10 steps) and
then rejects a solve whose relative residual still exceeds `1e-2` as singular,
which hands it to the regularization ladder. Factorizations without perturbed
pivots are neither refined nor checked. Both numbers are
[`SparseOptions` fields](options.md#sparse-direct-solves-on-perturbed-pivots)
(`perturbed_pivot_refinement_steps`, `perturbed_pivot_max_residual`). The measured effect on the S2MPJ
corpus is in the
[CuPy sparse-direct benchmark](../benchmarks/s2mpj.md#cupy-sparse-direct-route-2026-09-28).

The inertia that cuDSS reports for such a factorization is that of the
perturbed matrix: it counts each perturbed pivot by the sign of its `±ε`.
That count can be wrong for the actual matrix. On a singular `CATENARY` KKT
system it reports zero pivots as positive, and on some L-BFGS systems a few
signs come out flipped. The adapter still reports it, and the IPM's inertia
check still acts on it. Withholding it whenever pivots were perturbed was
tried on the corpus and lost rows: on the exact-Hessian route those rejected
factors were genuinely indefinite, and the δ_w correction was needed. See
[the A/B](../benchmarks/s2mpj.md#perturbed-pivot-inertia-and-deterministic-mode-2026-10-01).

### Reproducibility on CUDA

On the CPU backends (NumPy/SciPy, Torch-CPU) every route is deterministic:
identical inputs and options reproduce identical iteration counts and
iterates run to run. On CUDA, the CuPy sparse-direct route is reproducible
too when `SparseOptions(deterministic=True)` is set: separate processes
solving the same problem with the same options follow the same trajectory
bit for bit, through every product, factorization and solve (see
[the check](../benchmarks/s2mpj.md#reproducible-cupy-sparse-route-2026-10-02)).
Without the option, cuDSS still factors in a run-dependent order.

This matters because an interior-point method amplifies last-bit
differences into different step-length and acceptance decisions. Before the
change, identical CuPy sparse-route re-runs of the full corpus moved one
problem from 25 to 38 iterations, another from 2055 to 120, and flipped a
third from a `max_time` stop to `optimal` convergence. See the [CuPy
sparse-direct route
section](../benchmarks/s2mpj.md#cupy-sparse-direct-route-2026-09-28) of the
S2MPJ benchmark report.

**The cuDSS factorization: `deterministic=True`.**
`SparseOptions(deterministic=True)` (see
[Reproducible sparse factorizations](options.md#reproducible-sparse-factorizations))
switches cuDSS to its `DETERMINISTIC_MODE`. Repeated factorizations of one
matrix are then bitwise identical. Without it, six fresh solvers on one
captured `ROTDISC` KKT system gave six different solutions, up to 1.4e2
apart. In isolation the mode cost 1.2–3× per cuDSS factorization phase (2× on
the `Protons_01` TROTS system), measured in the
[matching micro-study](../benchmarks/s2mpj.md#cudss-matching-and-pivot-options-2026-09-30).
Measured over whole S2MPJ solves, where small systems are launch-bound, it
added 2–3 % per factorization call. The default stays off.

**The adapter's own kernels: reproducible in every mode.** CuPy's
`csr @ x` always uses cuSPARSE's default algorithms. In three places the IPM
reaches every iteration, those accumulate with atomics, and each one returned
a different last-bit result on almost every call (30 distinct results in 30
repeats). The CuPy `SparseOperator` and the adapter's COO build now avoid all
three:

| operation | before | now |
| --- | --- | --- |
| `A.T @ y` (`rmatvec`, `rmatmat`, `gram_diagonal`), e.g. the Jacobian-transpose products in every right-hand side | cuSPARSE's transposed SpMV/SpMM, which uses atomics in every algorithm | a forward product on an explicit `Aᵀ` built on the first transposed product. Building it is a permutation, with no arithmetic. The operator keeps it for its values. |
| `A @ x` where `A` has long rows | the default CSR SpMV, which splits long rows across thread blocks | `CUSPARSE_SPMV_CSR_ALG2`, which cuSPARSE documents as bitwise deterministic |
| `A @ X` with several columns | cuSPARSE SpMM. None of its algorithms was reproducible on long rows. | an ordered kernel: one thread per output entry when the longest row has at most 1024 entries, one warp per entry beyond that |
| COO → CSR duplicate sum | `cupy.add.at` (atomics) | an ordered segment sum in emission order. It matches the SciPy adapter's build bit for bit. |

These kernels are the default, not an option, for two reasons:

- **The option could not reach them.** The Jacobian operators are built
  outside the sparse solver, so `SparseOptions` never reaches them.
- **They cost little or nothing where the IPM spends its time.** These
  figures are from an RTX 4070 Laptop, float64, as interleaved medians of 7.
  - **Forward SpMV:** unchanged (2.08 → 2.09 ms on the 4.1e7-nonzero
    `Protons_01` KKT).
  - **Transposed SpMV:** at most as slow, and faster on large systems: 3.41 →
    2.12 ms on that KKT, once `Aᵀ` exists.
  - **Eight-column products:** 18–50 % faster on systems with short rows,
    11–20 % slower on that KKT's 52 000-entry rows.
  - **COO → CSR value pass:** twice as fast (39.4 → 19.3 ms on 4.2e7
    triplets).
  - **Per-operator costs:** building `Aᵀ` costs as much as 3–11 transposed
    products, once per operator value array (22.7 ms on that KKT). A
    problem's Jacobian is a new operator every iteration, so its `Aᵀ` is
    rebuilt every iteration. Operators that one adapter builds on one
    pattern share a transpose plan, and from the second one on `Aᵀ` is a
    gather (5.2 ms). Compiling the COO map, once per pattern and adapter,
    takes 27–53 % longer.

On five S2MPJ problems, the wall time per factorization call changed by less
than the spread of three runs on a loaded machine.

**Memory.** The cached `Aᵀ` is a second copy of an operator's values and
indices. Only an operator that is transposed keeps one. KKT matrices built
for the factorization are never transposed, so they pay nothing. Operators
that one adapter builds on one sparsity pattern share the pattern's index
arrays for `Aᵀ` and add only a value array each. The COO map's order and
run-boundary arrays replace its old inverse map, so the map takes no more
memory than before. The cap is `_TRANSPOSE_CACHE_MAX_BYTES` = 512 MiB per
transpose:

- **Within the cap.** That covers every S2MPJ operator and an operator of the
  `Protons_01` KKT's size (476 MiB).
- **Above the cap.** An operator builds no `Aᵀ` for its transposed products
  and uses cuSPARSE's transposed product, which is the one non-reproducible
  exception. The sparse-Gram strategies (an SpGEMM against `Aᵀ`) still build
  and keep it whatever the cap, as they did before, and the transposed
  products then use that copy.

The worst case on an 8 GB card is `Head-and-Neck_01`'s 1.1e8-nonzero dose
block on the CuPy dense route (4 iterations):

- Caching its 1.24 GiB transpose raised the pool peak from 6.51 GiB to
  7.75 GiB used, with 8.91 GiB reserved, which is past the card. The run time
  went from 54 to 68 s.
- Transposing row windows on every call instead kept the peak at 6.85 GiB,
  but the run still took 66 s.
- With the cap, the peak stays at the old 6.51 GiB.

That route's device-tiled Gram is not reproducible anyway (see below).

**What is still not covered.**

- **The guarantee holds on one machine only.** Results are reproducible on
  the same GPU with the same driver, CUDA, CuPy and cuDSS versions. cuBLAS
  promises bitwise reproducibility only on GPUs of the same architecture and
  SM count, and its last bits have been seen to depend on the process's
  allocation history. Another cuDSS version may factor in a different order.
- **Some code is outside the adapter.** The benchmark harness's TROTS loader
  assembles its device-tiled Gram outside the adapter, so it is not covered.
  The sparse Gram's SpGEMM is cuSPARSE's default algorithm, which has been
  observed to be reproducible but is not documented as such.
- **A `max_time` stop depends on the wall clock.** Two runs that hit it stop
  at different iterations of the same trajectory.

For a CuPy-sparse A/B comparison, set `deterministic=True` in every arm. A
single run per row is then representative of that row.

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
