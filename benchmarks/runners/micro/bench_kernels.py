"""pytest-benchmark micro-benchmarks: matvec, dense KKT solve, one Newton step.

These isolate the per-kernel GPU cost (matmul, factor/solve) from the IPM loop
overhead measured by ``benchmarks.runners.device_efficiency``. Each benchmark is
parametrized over every installed backend (NumPy / Torch / CuPy), so the same
kernels are timed on CPU and — when CuPy is present — on the GPU.

Run with::

    pytest benchmarks/runners/micro --benchmark-only
    pytest benchmarks/runners/micro --benchmark-only -k cupy   # GPU kernels only

Each callable ends with a scalar read so a lazy/async backend (CuPy) is forced to
synchronize *inside* the timed region — otherwise the benchmark would clock only
kernel-launch latency, not the kernel.
"""

from __future__ import annotations

import pytest

from ipax.backend.operators import Dense
from ipax.linalg.dense import DenseSolver
from ipax.testing.backends import import_namespace


def _installed_backends() -> list[str]:
    names: list[str] = []
    for name in ("numpy", "torch", "cupy"):
        try:
            import_namespace(name)
        except ImportError:
            continue
        names.append(name)
    return names


@pytest.fixture(params=_installed_backends(), ids=str)
def xp(request: pytest.FixtureRequest):
    return import_namespace(request.param)


def _spd_matrix(xp, n: int):
    """A symmetric positive-definite ``n×n`` matrix (well-conditioned)."""
    m = xp.reshape(xp.linspace(-1.0, 1.0, n * n, dtype=xp.float64), (n, n))
    return xp.matmul(m, xp.matrix_transpose(m)) + n * xp.eye(n, dtype=xp.float64)


def test_matvec_throughput(benchmark, xp):
    """Dense operator matvec — the dominant matrix-free kernel."""
    n = 1024
    op = Dense(_spd_matrix(xp, n))
    v = xp.ones((n,), dtype=xp.float64)

    def run() -> float:
        return float(xp.sum(op.matvec(v)))

    benchmark(run)


def test_dense_kkt_solve(benchmark, xp):
    """Factor + solve of a dense SPD system — the condensed KKT linear solve."""
    n = 512
    operator = Dense(_spd_matrix(xp, n))
    rhs = xp.ones((n,), dtype=xp.float64)
    solver = DenseSolver()

    def run() -> float:
        solver.factor(operator)
        return float(xp.sum(solver.solve(rhs)))

    benchmark(run)


def test_one_newton_step(benchmark, xp):
    """One Newton-system solve: assemble RHS, factor, solve, recover (proxy).

    Times the linear-algebra heart of a single IPM iteration — a symmetric solve
    plus the surrounding vector arithmetic — without the convergence/line-search
    machinery, so a GPU regression here is a kernel regression, not loop overhead.
    """
    n = 512
    operator = Dense(_spd_matrix(xp, n))
    grad = xp.linspace(0.1, 1.0, n, dtype=xp.float64)
    sigma = xp.ones((n,), dtype=xp.float64)
    solver = DenseSolver()

    def run() -> float:
        rhs = -(grad + sigma)
        solver.factor(operator)
        dx = solver.solve(rhs)
        dz = sigma * dx  # representative recovery of an eliminated block
        return float(xp.max(xp.abs(dx)) + xp.max(xp.abs(dz)))

    benchmark(run)
