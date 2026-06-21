"""Device-efficiency study runner (GPU profiling).

Profiles the IPM loop for **host<->device sync count** and **wall time per
iteration** (plus the GPU-vs-CPU time split on CuPy) across solver routes and
problem sizes on the synthetic RT-like generator. The headline metric is
``syncs/iter``: every ``float()``/``bool()`` on a 0-d device array in the driver
loop forces a host sync that serializes the GPU, so this is the number the
no-cost loop optimization (consolidating per-iteration scalar reads) drives down.

Run from the repository root on a CUDA machine::

    python -m benchmarks.runners.device_efficiency \
        --backends cupy --routes krylov,sparse --sizes 1000,10000,100000

GPU-gated: a requested backend that is not installed is skipped with a note, so
this is a **no-op on CI** (no CuPy) and real work on a local GPU. Exits non-zero
if any solve fails. For kernel-launch counts, wrap the same command in an external
profiler::

    nsys profile -o device python -m benchmarks.runners.device_efficiency --backends cupy ...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import ipax
from benchmarks.harness import (
    DeviceMetrics,
    capture_environment,
    format_device,
    measure_device_solve,
)
from ipax.backend.namespace import capabilities
from ipax.testing.backends import import_namespace
from ipax.testing.problems import make_rt_like_problem

_ROUTES: dict[str, ipax.Options] = {
    "dense": ipax.Options(hessian="exact", linsolve="dense"),
    "krylov": ipax.Options(hessian="exact", linsolve="krylov"),
    "sparse": ipax.Options(hessian="exact", linsolve="sparse"),
}


def run_device_efficiency(
    backends: list[str], routes: list[str], sizes: list[int]
) -> list[DeviceMetrics]:
    """Profile each installed backend over the route × size grid."""
    metrics: list[DeviceMetrics] = []
    for backend in backends:
        try:
            xp = import_namespace(backend)
        except ImportError:
            print(f"  skip backend {backend!r}: not installed")
            continue

        has_sparse = capabilities(xp).has_sparse_adapter
        for route in routes:
            if route == "sparse" and not has_sparse:
                print(f"  skip {backend}/sparse: no sparse adapter")
                continue
            options = _ROUTES[route]
            for n in sizes:
                problem = make_rt_like_problem(
                    xp, n, n_structures=6, density=0.1, seed=0
                )
                x0 = xp.full((n,), 0.01, dtype=xp.float64)
                metrics.append(
                    measure_device_solve(
                        problem, x0, options, backend=backend, route=route
                    )
                )
    return metrics


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ipax device-efficiency study")
    parser.add_argument(
        "--backends", default="cupy", help="comma-separated backends (default: cupy)"
    )
    parser.add_argument(
        "--routes", default="krylov,sparse", help="comma-separated routes"
    )
    parser.add_argument(
        "--sizes", default="1000,10000,100000", help="comma-separated n"
    )
    parser.add_argument("--out", default="benchmarks/reports/device")
    args = parser.parse_args(argv)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    sizes = _parse_int_list(args.sizes)
    unknown = [r for r in routes if r not in _ROUTES]
    if unknown:
        parser.error(f"unknown route(s): {', '.join(unknown)}")

    environment = capture_environment()
    metrics = run_device_efficiency(backends, routes, sizes)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"environment": environment, "metrics": [asdict(m) for m in metrics]}
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    out.with_suffix(".md").write_text(
        format_device(metrics, environment), encoding="utf-8"
    )

    failures = [m for m in metrics if not m.success]
    print(
        f"device-efficiency: {len(metrics) - len(failures)}/{len(metrics)} solved "
        f"-> {out.with_suffix('.json')}, {out.with_suffix('.md')}"
    )
    for m in failures:
        print(f"  FAIL {m.backend}/{m.route} n={m.n_vars}: kkt={m.kkt_error:.2e}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
