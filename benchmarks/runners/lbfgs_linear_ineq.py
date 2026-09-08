"""A/B affine-inequality kernels and constrained L-BFGS Newton solves.

Run as ``python -m benchmarks.runners.lbfgs_linear_ineq``. The before classes
reproduce the two benchmarked operator methods and the dense solver's LU-only
L-BFGS path from c19f336; everything else is shared. Each route step is one
factorization and two right-hand sides (the predictor/corrector pattern).
Timings include synchronization at sample boundaries, exclude first-use warmup,
and describe fixed Newton systems, not end-to-end IPM speedups.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.runners.lbfgs_bound_only import median_ms
from ipax.backend.namespace import array_namespace
from ipax.backend.operators import COOOperator, Dense, Diagonal, LinearOperator
from ipax.ipm.hessian import LBFGSOperator
from ipax.ipm.kkt import build_condensed_operator
from ipax.linalg.dense import DenseSolver
from ipax.linalg.krylov import KrylovSolver
from ipax.linalg.regularize import RegularizationState
from ipax.linalg.sparse import SparseDirectSolver
from ipax.options import KrylovOptions, LBFGSOptions
from ipax.problem.scaling import _RowScaled
from ipax.testing.backends import import_namespace

if TYPE_CHECKING:
    from collections.abc import Callable

    from ipax.linalg.solver import LinearSolver
    from ipax.typing import Array


class _BeforeDense(Dense):
    def gram_diagonal(self, weights: Array) -> Array:
        xp = array_namespace(self._A, weights)
        return xp.sum(xp.expand_dims(weights, axis=1) * self._A * self._A, axis=0)

    def row_gram_diagonal(self, weights: Array) -> Array:
        xp = array_namespace(self._A, weights)
        return xp.sum(self._A * self._A * xp.expand_dims(weights, axis=0), axis=1)


class _BeforeScaled(_RowScaled):
    rmatmat = LinearOperator.rmatmat


class _BeforeDenseSolver(DenseSolver):
    # c19f336: no Cholesky reuse for the (PD-by-construction) L-BFGS block.
    def _keep_pd_hinted_factor(self, matrix: Array, xp: Any, cholesky: Any) -> None:
        del matrix, xp, cholesky


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="numpy,torch,cupy")
    parser.add_argument("--sizes", default="128,512")
    parser.add_argument("--kernel-rows", type=int, default=10000)
    parser.add_argument("--kernel-cols", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument(
        "--out", default="benchmarks/reports/lbfgs_linear_ineq_timings.json"
    )
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    devices = {}
    for backend in args.backends.split(","):
        xp = import_namespace(backend)

        def sync() -> None:
            pass

        if backend == "torch":
            import torch

            torch.set_num_threads(1)
            devices[backend] = f"CPU; torch {torch.__version__}; one thread"
        elif backend == "cupy":
            import cupy

            sync = cupy.cuda.get_current_stream().synchronize
            devices[backend] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        else:
            devices[backend] = f"CPU; numpy {np.__version__}"

        rng = np.random.default_rng(42)
        A = xp.asarray(
            rng.standard_normal((args.kernel_rows, args.kernel_cols)), dtype=xp.float64
        )
        weights = xp.asarray(rng.uniform(0.1, 2.0, args.kernel_rows), dtype=A.dtype)
        V = xp.asarray(rng.standard_normal((args.kernel_rows, 8)), dtype=A.dtype)
        scales = xp.full_like(weights, 0.5)
        for version, dense_type, scaled_type in (
            ("before", _BeforeDense, _BeforeScaled),
            ("after", Dense, _RowScaled),
        ):
            jac = dense_type(A)
            scaled = scaled_type(jac, scales)
            for phase, fn in (
                (
                    "gram-diagonal",
                    lambda jac=jac, weights=weights: jac.gram_diagonal(weights),
                ),
                ("scaled-adjoint-8rhs", lambda scaled=scaled, V=V: scaled.rmatmat(V)),
            ):
                rows.append(
                    {
                        "backend": backend,
                        "version": version,
                        "phase": phase,
                        "m": args.kernel_rows,
                        "n": args.kernel_cols,
                        "ms": median_ms(fn, sync, args.repeats, args.calls),
                    }
                )
        del A, V, weights, jac, scaled, fn

        for n in map(int, filter(None, args.sizes.split(","))):
            m = 2 * n
            # Identical mathematical matrices in dense and sparse storage. Each
            # row couples two adjacent columns; the Gram has localized fill.
            ri = np.repeat(np.arange(m), 2)
            ci = np.column_stack((np.arange(m) % n, (np.arange(m) + 1) % n)).reshape(-1)
            av = rng.uniform(0.5, 1.5, 2 * m)
            host = np.zeros((m, n))
            host[ri, ci] = av
            A = xp.asarray(host, dtype=xp.float64)
            sparse = COOOperator(
                xp.asarray(ri), xp.asarray(ci), xp.asarray(av, dtype=A.dtype), (m, n)
            )
            rhs = xp.asarray(rng.standard_normal(n), dtype=A.dtype)
            rhs2 = xp.asarray(rng.standard_normal(n), dtype=A.dtype)
            W = LBFGSOperator(n, LBFGSOptions(memory=5))
            curvature = xp.asarray(rng.uniform(1.0, 4.0, n), dtype=A.dtype)
            for _ in range(5):
                delta = xp.asarray(rng.standard_normal(n), dtype=A.dtype)
                W.update(delta, curvature * delta)
            bounds = Diagonal(0.1 + curvature)
            scale = xp.full((m,), 0.5, dtype=A.dtype)
            for spread in (1.0, 1e4):
                sigma = Diagonal(
                    xp.asarray(np.geomspace(1.0, spread, m), dtype=A.dtype)
                )
                for route in (
                    "dense",
                    "krylov-jacobi",
                    "krylov-lbfgs",
                    "sparse",
                    "sparse-NE",
                ):
                    for version, dense_type, scaled_type, dense_solver in (
                        ("before", _BeforeDense, _BeforeScaled, _BeforeDenseSolver),
                        ("after", Dense, _RowScaled, DenseSolver),
                    ):
                        jac = scaled_type(
                            sparse if route.startswith("sparse") else dense_type(A),
                            scale,
                        )
                        if route == "dense":
                            solver = dense_solver()
                        elif route.startswith("krylov"):
                            solver = KrylovSolver(
                                KrylovOptions(
                                    preconditioner=route.split("-")[1],
                                    adaptive_tol=False,
                                    rtol=1e-10,
                                )
                            )
                        else:
                            solver = SparseDirectSolver(
                                form="normal_equations"
                                if route == "sparse-NE"
                                else "augmented"
                            )

                        def build(
                            W: LBFGSOperator = W,
                            bounds: Diagonal = bounds,
                            sigma: Diagonal = sigma,
                            jac: LinearOperator = jac,
                        ) -> LinearOperator:
                            return build_condensed_operator(
                                W, bounds, sigma, jac, RegularizationState(delta_w=1e-3)
                            )

                        def step(
                            solver: LinearSolver = solver,
                            build: Callable[[], LinearOperator] = build,
                            rhs: Array = rhs,
                            rhs2: Array = rhs2,
                        ) -> Array:
                            solver.factor(build())
                            solver.solve(rhs2)
                            return solver.solve(rhs)

                        solution = step()
                        residual = float(
                            xp.max(xp.abs(build().matvec(solution) - rhs))
                        ) / float(xp.max(xp.abs(rhs)))
                        if not np.isfinite(residual) or residual > 1e-7:
                            raise AssertionError(
                                f"{backend} {route}: residual {residual}"
                            )
                        rows.append(
                            {
                                "backend": backend,
                                "version": version,
                                "phase": "factor-solve",
                                "route": route,
                                "n": n,
                                "m": m,
                                "sigma_spread": spread,
                                "ms": median_ms(step, sync, args.repeats, args.calls),
                                "residual": residual,
                                "iterations": getattr(solver, "last_iterations", None),
                                "solver": solver.describe(),
                            }
                        )
            print(f"{backend} n={n} complete", flush=True)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "baseline": "c19f3366ee9e56b784f849a43b8dd7e102f1c149",
                "platform": platform.platform(),
                "devices": devices,
                "repeats": args.repeats,
                "calls": args.calls,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
