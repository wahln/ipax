"""Optional report plots.

Matplotlib is imported **lazily** through :func:`_require_matplotlib`, so the
harness core stays dependency-free and the QC/cross-check sweeps run without it.
It ships in the ``bench`` extra. The non-interactive ``Agg`` backend is forced so
plots render headless (CI, no display).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.harness import CaseResult, ScalingPoint


class PlottingUnavailable(RuntimeError):
    """Matplotlib is not installed (``pip install 'ipax[bench]'``)."""


def _require_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # optional dependency
        raise PlottingUnavailable(
            "matplotlib is not installed; install the 'bench' extra"
        ) from exc
    return plt


def plot_scaling(points: list[ScalingPoint], path: Path) -> Path:
    """Log–log solve-time and peak-memory vs ``n``, one line per solver route."""
    plt = _require_matplotlib()
    by_route: dict[str, list[ScalingPoint]] = {}
    for p in points:
        if p.success:
            by_route.setdefault(p.route, []).append(p)

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(11, 4.2))
    for route, rows in by_route.items():
        rows = sorted(rows, key=lambda r: r.n_vars)
        ns = [r.n_vars for r in rows]
        ax_time.plot(ns, [r.solve_time for r in rows], marker="o", label=route)
        ax_mem.plot(ns, [r.peak_memory_mb for r in rows], marker="o", label=route)

    for axis, title, ylabel in (
        (ax_time, "solve time", "seconds"),
        (ax_mem, "peak memory", "MB"),
    ):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("variables n")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
        if by_route:
            axis.legend()

    fig.suptitle("ipax scaling (synthetic RT-like)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_qc_iterations(results: list[CaseResult], path: Path) -> Path:
    """Bar chart of mean outer iterations per configuration (solved cases)."""
    plt = _require_matplotlib()
    by_config: dict[str, list[int]] = {}
    for r in results:
        if r.success:
            by_config.setdefault(r.config, []).append(r.n_iter)

    configs = sorted(by_config)
    means = [sum(v) / len(v) for v in (by_config[c] for c in configs)]

    fig, axis = plt.subplots(figsize=(max(6, 0.7 * len(configs) + 2), 4))
    axis.bar(range(len(configs)), means, color="#4C78A8")
    axis.set_xticks(range(len(configs)))
    axis.set_xticklabels(configs, rotation=30, ha="right")
    axis.set_ylabel("mean outer iterations")
    axis.set_title("ipax QC: iterations by configuration")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


__all__ = ["PlottingUnavailable", "plot_qc_iterations", "plot_scaling"]
