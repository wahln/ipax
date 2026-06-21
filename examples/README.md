# Examples

These are small executable snapshots of the implementation surface. They are not
benchmarks and they are intentionally short enough to read from top to bottom.

Run them from the repository root:

```powershell
.\.venv\Scripts\python.exe examples\unconstrained_quadratic.py
.\.venv\Scripts\python.exe examples\bound_and_inequality.py
```

Most examples use NumPy as the concrete backend at the edge of the application;
`autodiff.py` uses PyTorch / JAX to exercise the autodiff path. Either way the
solver core infers the namespace from the input arrays and remains free of
concrete array-library imports.

Current examples:

- `unconstrained_quadratic.py` shows the dense exact-Hessian fast path through
  `QuadraticProblem`.
- `bound_and_inequality.py` shows a minimal custom `Problem` with a simple
  active inequality and prints the associated multiplier.
- `equality_constrained.py` / `nonconvex_hs.py` show the filter line-search
  with linear and nonconvex equality constraints.
- `lbfgs_finite_diff.py` solves with only an objective callable on NumPy: the
  resolver fills the gradient by **finite differences** and the Hessian by
  Powell-damped **L-BFGS**.
- `autodiff.py` is the same problem on **PyTorch / JAX**, where the gradient
  resolves to **autodiff** and `hessian="autodiff-hvp"` feeds exact
  Hessian-vector products of the Lagrangian (skips backends that aren't
  installed).
- `matrix_free_krylov.py` solves a bounded QP with a matrix-free Hessian
  operator through the conjugate-gradient (`linsolve="krylov"`) route, so no
  matrix is ever formed or factored.
- `radiotherapy_sparse.py` is a radiotherapy-like fluence-map problem:
  minimize `½‖D x − d_pres‖²` for a sparse SciPy dose-influence matrix `D`
  (voxels × beamlets) with non-negativity bounds `x ≥ 0`. Gradient-only, so the
  Hessian is L-BFGS. Size is tunable via `RT_VOXELS` / `RT_BEAMLETS` /
  `RT_DENSITY`.
