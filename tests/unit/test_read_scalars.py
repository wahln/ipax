# Copyright 2026 Niklas Wahl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``read_scalars`` — one host transfer for several device scalars.

The Array API standard has no portable bulk device→host read: ``float(x)``
forces one sync *per scalar* (expensive under WDDM/CUDA), and ``to_device``
has no portable CPU device token. The gap-filler stacks the 0-d parts and
moves them once; every value must be bitwise what ``float()`` of the part
would have produced, because the driver's fused decision reads rely on
that for trajectory neutrality.
"""

from __future__ import annotations

import math

from ipax.backend import scalars
from tests._helpers import array


def test_read_scalars_matches_per_element_float(namespace):
    parts = [
        namespace.sum(array(namespace, [1.0, 2.25, -3.5])),
        namespace.max(array(namespace, [0.125, -7.75])),
        namespace.min(array(namespace, [9.0, 2.0e-13])),
    ]
    expected = [float(p) for p in parts]

    got = scalars.read_scalars(namespace, parts)

    assert got == expected  # bitwise, not approximately


def test_read_scalars_single_part(namespace):
    part = namespace.sum(array(namespace, [4.0, 0.5]))
    assert scalars.read_scalars(namespace, [part]) == [4.5]


def test_read_scalars_empty(namespace):
    assert scalars.read_scalars(namespace, []) == []


def test_read_scalars_fallback_matches_per_element(namespace, monkeypatch):
    # A namespace/device combination without a bulk host transfer (or a
    # transient transfer failure) must degrade to exact per-element reads —
    # pinned explicitly, not just via array-api-strict happening to reject
    # the "cpu" token.
    import array_api_compat

    def boom(x, device):
        raise RuntimeError("no route to host")

    monkeypatch.setattr(array_api_compat, "to_device", boom)
    parts = [
        namespace.sum(array(namespace, [1.0, 2.25, -3.5])),
        namespace.max(array(namespace, [0.125, -7.75])),
    ]
    assert scalars.read_scalars(namespace, parts) == [float(p) for p in parts]


def test_read_scalars_propagates_nonfinite(namespace):
    parts = [
        namespace.sum(array(namespace, [math.inf])),
        namespace.sum(array(namespace, [1.0])),
    ]
    got = scalars.read_scalars(namespace, parts)
    assert math.isinf(got[0]) and got[1] == 1.0


def test_read_scalars_mixed_bool_flag(namespace):
    # The L-BFGS update fuses a finiteness flag (cast to the float dtype)
    # with its curvature scalars; the flag must round-trip exactly.
    ok = namespace.astype(
        namespace.all(namespace.isfinite(array(namespace, [1.0, 2.0]))),
        array(namespace, [0.0]).dtype,
    )
    bad = namespace.astype(
        namespace.all(namespace.isfinite(array(namespace, [math.nan]))),
        array(namespace, [0.0]).dtype,
    )
    assert scalars.read_scalars(namespace, [ok, bad]) == [1.0, 0.0]
