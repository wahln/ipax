"""Minimal example: **true autodiff** derivative resolution (PyTorch / JAX).

The objective is a plain callable with no gradient and no Hessian. On an
autodiff-capable backend the resolver binds the gradient to reverse-mode autodiff
(``derivative_sources.gradient == 'autodiff'``) instead of finite differences,
and ``hessian='autodiff-hvp'`` feeds exact Hessian-vector products of the
Lagrangian to the solver — no matrix ever formed.

Runs on whichever of PyTorch / JAX is importable; skips the rest. For the
no-autodiff (finite-difference) counterpart on NumPy, see
``lbfgs_finite_diff.py``.

HS1 Rosenbrock: ``min 100 (x2 - x1^2)^2 + (1 - x1)^2``; optimum (1, 1), f = 0.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import array_api_compat as xpc

import ipax

LINSOLVE_MODE = "sparse"


def rosenbrock(x: Any) -> Any:
    return 100.0 * (x[1] - x[0] * x[0]) ** 2 + (1.0 - x[0]) ** 2


def _run(backend: str, x0: Any) -> None:
    problem = ipax.FunctionProblem(2, rosenbrock)

    xp = xpc.array_namespace(x0)

    # L-BFGS Hessian, gradient by autodiff.
    lbfgs = ipax.solve(
        problem,
        x0,
        options=ipax.Options(hessian="lbfgs", linsolve=LINSOLVE_MODE, verbose=1),
    )
    # Exact Hessian-vector products of the Lagrangian by double-backprop.
    try:
        hvp = ipax.solve(
            problem,
            x0,
            options=ipax.Options(
                hessian="autodiff-hvp", linsolve=LINSOLVE_MODE, verbose=1
            ),
        )
    except (RuntimeError, NotImplementedError) as e:
        print(
            f"skipping autodiff-hvp solve for {backend} (not supported by the "
            f"dispatched sparse adapter): {e}"
        )
        hvp = ipax.Result(
            ipax.Status.NUMERICAL_ERROR,
            xp.full_like(x0, float("nan")),
            float("nan"),
            kkt_error=float("nan"),
            n_iter=0,
            derivative_sources=ipax.DerivativeSources(gradient=None, hessian=None),
        )

    print(f"=== backend: {backend} ===")
    print(f"  x (lbfgs):       {lbfgs.x}")
    print(f"  x (autodiff-hvp):{hvp.x}")
    print("  expected_x:      [1. 1.]")
    print(
        f"  gradient source: {lbfgs.derivative_sources.gradient}  (expected: autodiff)"
    )
    print(
        f"  hessian sources: lbfgs={lbfgs.derivative_sources.hessian!r}, "
        f"hvp={hvp.derivative_sources.hessian!r}"
    )
    print(f"  iterations:      lbfgs={lbfgs.n_iter}, autodiff-hvp={hvp.n_iter}")
    print(f"  kkt_error:       lbfgs={lbfgs.kkt_error:.3e}, hvp={hvp.kkt_error:.3e}")


def main() -> None:
    ran = False

    if find_spec("torch") is not None:
        import torch  # application-edge backend import (allowed in examples)

        _run("torch", torch.asarray([-1.2, 1.0], dtype=torch.float64))
        ran = True

    if find_spec("jax") is not None:
        import jax  # application-edge backend import (allowed in examples)
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)  # match the float64 default
        _run("jax", jnp.asarray([-1.2, 1.0], dtype=jnp.float64))
        ran = True

    if not ran:
        print(
            "Neither PyTorch nor JAX is importable; install one of them "
            "(`pip install ipax[torch]` or `ipax[jax]`) to run this example. "
            "See lbfgs_finite_diff.py for the NumPy / finite-difference path."
        )


if __name__ == "__main__":
    main()
