"""Regression: fraction-to-boundary must not host-sync per element (GPU perf).

``fraction_to_boundary`` originally looped over the array in Python, calling
``float(v[idx])``/``float(dv[idx])`` per element — O(n) host<->device syncs per
call, and it is called 6x per IPM iteration. On a CUDA backend this dominated
iteration time: a device-efficiency profile showed ~5,500 syncs/iter at n=1000
(scaling linearly with n). The rule is now vectorized to a single sync.

This guards the property directly: counting the device->host scalar
materializations during one call on a large vector must be O(1), not O(n). The
counter patches the array type's scalar dunders, which only works on backends
whose array type allows it (Torch yes, NumPy's built-in ndarray no), so the test
skips where counting is unavailable — the cost it guards is a GPU cost anyway.
"""

from __future__ import annotations

import pytest

from benchmarks.harness import _ScalarSyncCounter
from ipax.ipm.barrier import fraction_to_boundary
from tests._helpers import array


def test_fraction_to_boundary_is_constant_sync(namespace):
    n = 2000
    v = array(namespace, [1.0] * n)
    dv = array(namespace, [-0.5 if i % 2 else 0.5 for i in range(n)])

    with _ScalarSyncCounter(type(v)) as counter:
        alpha = fraction_to_boundary(v, dv, tau=0.95)

    if not counter.available:
        pytest.skip(f"scalar-sync counting unavailable for {type(v)!r}")

    # Vectorized form does a single float() on the reduced scalar; a constant
    # bound (independent of n) is the invariant. The old loop would be ~n here.
    assert counter.count <= 4, f"{counter.count} host syncs for n={n} (expected O(1))"
    assert 0.0 <= alpha <= 1.0
