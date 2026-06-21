# ipax

**Interior-Point on the Array API** — a pure-Python, Array-API-conformant
primal–dual interior-point solver for large-scale nonlinear constrained
optimization, tuned for radiotherapy treatment-planning scale.

```
min   f(x)
s.t.  c(x)  = 0
      g(x) ≤ 0
      x_L ≤ x ≤ x_U
```

- **Backend-agnostic** — runs unchanged on NumPy, PyTorch, JAX, CuPy (CPU/GPU);
  the backend is inferred from the arrays your problem returns.
- **Capability-graded problem interface** — supply only the objective; gradients,
  Jacobians, and the Hessian are filled by autodiff or finite differences, with
  L-BFGS as the universal Hessian default.
- **Pluggable linear algebra** — dense, matrix-free Krylov, and per-backend
  sparse-direct solve routes behind one protocol.
- **IPOPT-style filter line search** with feasibility restoration; a Breedveld
  step controller is available as an alternative.

Start with [Getting started](getting-started.md), or read
[Architecture & invariants](architecture.md) to understand the design rules.
