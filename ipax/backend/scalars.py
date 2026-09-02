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

"""Bulk device→host scalar reads — a labeled Array API gap-filler.

Missing primitive: the standard offers no portable *bulk* device→host
read. ``float(x)`` materializes one scalar per call — a full host sync
each time on a GPU backend (particularly expensive under Windows/WDDM,
where a sync flushes the driver queue) — and ``Array.to_device`` exists
but the standard defines no portable CPU device token to pass it. So k
scalar decisions per iteration cost k syncs where one stacked transfer
would do.

:func:`read_scalars` fills the gap: stack the 0-d parts, move the stack
to the host once via ``array_api_compat.to_device(x, "cpu")``, and read
the values from the host copy. Every returned value is bitwise what
``float()`` of the corresponding part would have produced — the driver's
fused decision reads rely on that for trajectory neutrality — and any
namespace/device combination that cannot express the transfer falls back
to per-element ``float`` (CPU backends, where a read is free anyway).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ipax.typing import Array, Namespace


def read_scalars(xp: Namespace, parts: Sequence[Array]) -> list[float]:
    """Materialize several 0-d device scalars with (at most) one host sync.

    ``parts`` must be 0-d arrays of a common real floating dtype (cast any
    boolean flag with ``xp.astype`` first). Returns their Python-float
    values, bitwise identical to per-element ``float(part)``.

    The fallback below is *silent* by design: a namespace whose device
    model rejects the ``"cpu"`` token (array-api-strict) lands there on
    every call, and re-checking per call is deliberate — memoizing the
    verdict would be module-level mutable state (invariant #5), and the
    per-element reads are exact, so nothing is at stake but speed.
    """
    if not parts:
        return []
    if len(parts) == 1:
        return [float(parts[0])]
    from array_api_compat import to_device

    stacked = xp.stack(parts)
    try:
        host = to_device(stacked, "cpu")
        # ``tolist`` is not in the standard, but ``host`` is a concrete
        # host-resident array (NumPy, CPU torch.Tensor, ...) inside this
        # labeled adapter; it reads all elements without further syncs.
        values = host.tolist()
    except Exception:
        # No portable route to the host on this backend (or a transient
        # transfer failure): read element-wise — exact, just unbatched.
        return [float(p) for p in parts]
    return [float(v) for v in values]


__all__ = ["read_scalars"]
