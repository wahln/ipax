"""Smoke test for the device-efficiency profiling harness.

Runs on CPU (the namespace fixture) so the host-sync counter, ``DeviceMetrics``,
the runner, and the report formatter stay correct as the solver evolves. The
host-sync *count* is unavailable on NumPy (its ``ndarray`` scalar dunders cannot
be patched), available on Torch — both paths are exercised. Real GPU numbers come
from running ``benchmarks.runners.device_efficiency`` on CUDA hardware.
"""

from __future__ import annotations

import json

import ipax
from benchmarks.harness import (
    DeviceMetrics,
    capture_environment,
    format_device,
    measure_device_solve,
)
from benchmarks.runners.device_efficiency import main, run_device_efficiency
from ipax.testing.problems import make_rt_like_problem


def test_measure_device_solve_populates_metrics(namespace, backend_name):
    problem = make_rt_like_problem(namespace, 60, n_structures=4, density=0.2, seed=0)
    x0 = namespace.full((60,), 0.01, dtype=namespace.float64)

    m = measure_device_solve(
        problem,
        x0,
        ipax.Options(hessian="exact", linsolve="krylov"),
        backend=backend_name,
        route="krylov",
    )

    assert isinstance(m, DeviceMetrics)
    assert m.success
    assert m.n_iter > 0
    assert m.solve_time > 0.0
    assert m.time_per_iter > 0.0
    # NumPy's built-in ndarray cannot be patched -> count unavailable (None);
    # Torch's Tensor can, so the loop's scalar reads are counted (> 0).
    if backend_name == "numpy":
        assert m.host_syncs is None
    else:
        assert m.host_syncs is not None and m.host_syncs > 0
        assert m.syncs_per_iter is not None and m.syncs_per_iter > 0


def test_sync_counter_restores_dunders_after_window(namespace):
    """The counter must not leave the array type permanently patched."""
    problem = make_rt_like_problem(namespace, 40, n_structures=4, density=0.2, seed=0)
    x0 = namespace.full((40,), 0.01, dtype=namespace.float64)
    before = namespace.asarray([2.5])[0]
    assert float(before) == 2.5

    measure_device_solve(
        problem,
        x0,
        ipax.Options(hessian="exact", linsolve="dense"),
        backend="numpy",
        route="dense",
    )

    after = namespace.asarray([3.5])[0]
    assert float(after) == 3.5  # dunder behaves normally post-window


def test_runner_produces_report(tmp_path):
    metrics = run_device_efficiency(["numpy"], ["dense"], [40])
    assert metrics and all(m.success for m in metrics)

    env = capture_environment()
    md = format_device(metrics, env)
    assert "# ipax device-efficiency study" in md
    assert "syncs/iter" in md

    out = tmp_path / "device"
    rc = main(
        ["--backends", "numpy", "--routes", "dense", "--sizes", "40", "--out", str(out)]
    )
    assert rc == 0
    payload = json.loads(out.with_suffix(".json").read_text())
    assert payload["metrics"][0]["route"] == "dense"
    assert out.with_suffix(".md").exists()
