"""Scaling & memory study runner.

Sweeps the synthetic RT-like generator over problem size ``n`` for each solver
route (dense / matrix-free Krylov / sparse-direct), measuring wall-clock solve
time and peak memory (``tracemalloc``), and fits an empirical scaling exponent
``p`` in ``cost ∝ nᵖ``. This is the quantitative answer to "does it scale to
1e3–1e5 variables, and how does memory grow per route?"

Run from the repository root::

    python -m benchmarks.runners.scaling --sizes 500,1000,2000 --routes krylov,sparse,dense

Heavier than the QC sweep — run manually / nightly, not per-PR. Exits non-zero
if any solve fails (a robustness signal at scale).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import ipax
from benchmarks.generators import initial_point, make_rt_like_problem
from benchmarks.harness import (
    ScalingPoint,
    capture_environment,
    format_scaling,
    measure_solve,
)
from ipax.backend.namespace import capabilities
from ipax.testing.backends import import_namespace

_ROUTES: dict[str, ipax.Options] = {
    "dense": ipax.Options(hessian="exact", linsolve="dense"),
    "krylov": ipax.Options(hessian="exact", linsolve="krylov"),
    "sparse": ipax.Options(hessian="exact", linsolve="sparse"),
}


def run_scaling(routes: list[str], sizes: list[int]) -> list[ScalingPoint]:
    """Measure each requested route over the size sweep (RT-like, NumPy)."""
    has_sparse = capabilities(import_namespace("numpy")).has_sparse_adapter
    points: list[ScalingPoint] = []
    for route in routes:
        if route == "sparse" and not has_sparse:
            continue
        options = _ROUTES[route]
        for n in sizes:
            problem = make_rt_like_problem(n, n_structures=6, density=0.1, seed=0)
            x0 = initial_point(n)
            elapsed, peak, result = measure_solve(problem, x0, options)
            points.append(
                ScalingPoint(
                    route=route,
                    n_vars=n,
                    n_iter=result.n_iter,
                    success=result.success,
                    solve_time=elapsed,
                    peak_memory_mb=peak / 1e6,
                    kkt_error=result.kkt_error,
                )
            )
    return points


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ipax scaling & memory study")
    parser.add_argument("--sizes", default="500,1000,2000", help="comma-separated n")
    parser.add_argument(
        "--routes", default="krylov,sparse,dense", help="comma-separated routes"
    )
    parser.add_argument("--out", default="benchmarks/reports/scaling")
    parser.add_argument(
        "--no-plot", action="store_true", help="skip the matplotlib plot"
    )
    args = parser.parse_args(argv)

    sizes = _parse_int_list(args.sizes)
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    unknown = [r for r in routes if r not in _ROUTES]
    if unknown:
        parser.error(f"unknown route(s): {', '.join(unknown)}")

    environment = capture_environment()
    points = run_scaling(routes, sizes)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"environment": environment, "points": [asdict(p) for p in points]}
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    out.with_suffix(".md").write_text(
        format_scaling(points, environment), encoding="utf-8"
    )
    if not args.no_plot:
        from benchmarks.harness.plots import PlottingUnavailable, plot_scaling

        try:
            print(f"  plot -> {plot_scaling(points, out.with_suffix('.png'))}")
        except PlottingUnavailable as exc:
            print(f"  (no plot: {exc})")

    failures = [p for p in points if not p.success]
    print(
        f"scaling: {len(points) - len(failures)}/{len(points)} solved "
        f"-> {out.with_suffix('.json')}, {out.with_suffix('.md')}"
    )
    for p in failures:
        print(f"  FAIL {p.route} n={p.n_vars}: kkt={p.kkt_error:.2e}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
