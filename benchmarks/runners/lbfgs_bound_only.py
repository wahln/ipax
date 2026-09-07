"""A/B compact L-BFGS setup and solve timings against a repository revision.

Run with ``python -m benchmarks.runners.lbfgs_bound_only --backends numpy,torch,cupy``.
Measures fresh setup + apply and repeated apply, with GPU synchronization at
sample boundaries. This is a microbenchmark, not an end-to-end IPM speedup.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ipax.backend.operators import Dense, Diagonal
from ipax.ipm import kkt
from ipax.ipm.hessian import LBFGSOperator
from ipax.linalg.dense import DenseSolver
from ipax.linalg.krylov import KrylovSolver
from ipax.linalg.regularize import RegularizationState
from ipax.linalg.sparse import SparseDirectSolver
from ipax.options import KrylovOptions, LBFGSOptions
from ipax.testing.backends import import_namespace

if TYPE_CHECKING:
    from ipax.backend.operators import LinearOperator
    from ipax.linalg.solver import LinearSolver
    from ipax.typing import Array, Namespace


def baseline_module(ref: str) -> tuple[str, types.ModuleType]:
    """Load only the old KKT module; the shared Hessian implementation is unchanged."""
    revision = subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
    source = subprocess.check_output(["git", "show", f"{revision}:ipax/ipm/kkt.py"])
    module = types.ModuleType("ipax.ipm._benchmark_baseline_kkt")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{revision}:ipax/ipm/kkt.py", "exec"), module.__dict__)
    return revision, module


def median_ms(
    fn: Callable[[], Any], sync: Callable[[], None], repeats: int, calls: int
) -> float:
    fn()
    sync()
    samples = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        sync()
        samples.append(1000 * (time.perf_counter() - start) / calls)
    return statistics.median(samples)


def device_tolerance_signature(matrix: Array, xp: Namespace) -> tuple[int, int, int]:
    """Rejected experiment: fewer scalar reads, but extra small GPU kernels."""
    r = int(matrix.shape[0])
    sym = 0.5 * (matrix + xp.permute_dims(matrix, (1, 0)))
    eig = xp.linalg.eigvalsh(sym)
    scale = xp.max(xp.abs(eig))
    tol = xp.maximum(xp.ones_like(scale), scale) * (r * xp.finfo(eig.dtype).eps)
    pos = int(xp.sum(xp.astype(eig > tol, xp.int64)))
    neg = int(xp.sum(xp.astype(eig < -tol, xp.int64)))
    return pos, neg, r - pos - neg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-ref", default="9a073a8fb4cb41ad8928736abc28e4465e1667fa"
    )
    parser.add_argument("--backends", default="numpy,torch,cupy")
    parser.add_argument("--sizes", default="1000,10000,100000")
    parser.add_argument("--memory", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument(
        "--out", default="benchmarks/reports/lbfgs_bound_only_timings.json"
    )
    args = parser.parse_args()
    revision, old = baseline_module(args.baseline_ref)
    rows, devices = [], {}
    for backend in args.backends.split(","):
        xp = import_namespace(backend)

        def sync() -> None:
            return None

        if backend == "cupy":
            import cupy

            sync = cupy.cuda.get_current_stream().synchronize
            devices[backend] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        elif backend == "torch":
            import torch

            torch.set_num_threads(1)
            devices[backend] = f"CPU; torch {torch.__version__}; 1 thread"
        else:
            devices[backend] = f"CPU; numpy {np.__version__}"
        for n in map(int, args.sizes.split(",")):
            rng = np.random.default_rng(42)
            w = LBFGSOperator(n, LBFGSOptions(memory=args.memory))
            curvature = xp.asarray(1.0 + np.arange(n) % 7, dtype=xp.float64)
            for _ in range(args.memory):
                delta = xp.asarray(rng.standard_normal(n), dtype=xp.float64)
                w.update(delta, curvature * delta)
            rhs = xp.asarray(rng.standard_normal(n), dtype=xp.float64)
            empty = Diagonal(xp.zeros((0,), dtype=rhs.dtype))
            jac = Dense(xp.zeros((0, n), dtype=rhs.dtype))
            for unbounded in (False, True):
                sigma = Diagonal(xp.zeros_like(rhs) if unbounded else 0.1 + curvature)
                for route in ("dense", "krylov"):
                    for version, module in (("before", old), ("after", kkt)):

                        def build(
                            module: types.ModuleType = module,
                            w: LBFGSOperator = w,
                            sigma: Diagonal = sigma,
                            empty: Diagonal = empty,
                            jac: Dense = jac,
                            unbounded: bool = unbounded,
                        ) -> LinearOperator:
                            return module.build_condensed_operator(
                                w,
                                sigma,
                                empty,
                                jac,
                                RegularizationState(delta_w=1e-3),
                                sigma_x_zero=unbounded,
                            )

                        def apply(
                            op: LinearOperator, route: str = route, rhs: Array = rhs
                        ) -> Array:
                            if route == "dense":
                                return op.dense_structured_solve(rhs)
                            return op.lbfgs_inverse_apply()(rhs)

                        op = build()
                        solution = apply(op)
                        relative_residual = float(
                            xp.max(xp.abs(op.matvec(solution) - rhs))
                        ) / float(xp.max(xp.abs(rhs)))
                        if relative_residual > 1e-9:
                            raise AssertionError(
                                f"unexpected residual: {relative_residual}"
                            )
                        for phase, fn in (
                            ("fresh", lambda: apply(build())),
                            ("reuse", lambda op=op: apply(op)),
                        ):
                            ms = median_ms(fn, sync, args.repeats, args.calls)
                            row = {
                                "backend": backend,
                                "n": n,
                                "memory": args.memory,
                                "unbounded": unbounded,
                                "route": route,
                                "version": version,
                                "phase": phase,
                                "ms": ms,
                                "relative_residual": relative_residual,
                            }
                            rows.append(row)
                if not unbounded and n <= 10000:
                    # Actual solver entrypoints, with symbolic reuse across
                    # Newton steps. Krylov includes its true-residual check.
                    for route, solver_type in (
                        ("dense", DenseSolver),
                        ("krylov", lambda: KrylovSolver(KrylovOptions())),
                        ("sparse", SparseDirectSolver),
                    ):
                        solver = solver_type()

                        def step(
                            solver: LinearSolver = solver,
                            build: Callable[[], LinearOperator] = build,
                            rhs: Array = rhs,
                        ) -> Array:
                            solver.factor(build())
                            return solver.solve(rhs)

                        solution = step()
                        relative_residual = float(
                            xp.max(xp.abs(build().matvec(solution) - rhs))
                        ) / float(xp.max(xp.abs(rhs)))
                        if relative_residual > 1e-8:
                            raise AssertionError(f"route residual: {relative_residual}")
                        rows.append(
                            {
                                "backend": backend,
                                "n": n,
                                "memory": args.memory,
                                "route": route,
                                "version": "after",
                                "phase": "solver-factor-solve",
                                "ms": median_ms(step, sync, args.repeats, args.calls),
                                "relative_residual": relative_residual,
                                "solver": solver.describe(),
                            }
                        )
                print(f"{backend} n={n} unbounded={unbounded} complete", flush=True)
        # The sparse target-inertia helper is independent of n.
        matrix = w.compact_blocks()[3]
        for version, module in (("before", old), ("after", kkt)):
            ms = median_ms(
                lambda module=module, matrix=matrix: module._symmetric_signature(
                    matrix
                ),
                sync,
                args.repeats,
                args.calls,
            )
            rows.append(
                {
                    "backend": backend,
                    "route": "sparse-inertia",
                    "version": version,
                    "ms": ms,
                }
            )
    payload = {
        "baseline": revision,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "devices": devices,
        "repeats": args.repeats,
        "calls": args.calls,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
