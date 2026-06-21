"""pytest-benchmark micro-benchmarks: matvec, factor, one Newton step (§9.4).

Run with ``pytest benchmarks/runners/micro --benchmark-only``.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="kernels")
def test_matvec_throughput(benchmark):
    raise NotImplementedError


@pytest.mark.skip(reason="kernels")
def test_one_newton_step(benchmark):
    raise NotImplementedError
