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

"""``LBFGSOptions.damping_skip_ratio`` — skip strongly-contradicted pairs.

Powell damping keeps the update PD by blending ``y`` toward ``Bs``, which
*fabricates* positive curvature out of a pair that may actively contradict it.
On S2MPJ ``ORTHRGDS`` a handful of such pairs (``s·y / s·Bs`` down to −25)
redirect the whole run: damped, the solve burns 1000+ iterations at a worse
stationary point; with those pairs dropped it converges in ~20 to IPOPT's
objective. The full-corpus damp-vs-skip A/B refuted blanket skipping
(net −31 — mild indefiniteness is where damping genuinely helps), so the knob
is a *threshold*: a pair with ``s·y < −ratio · s·Bs`` is skipped, anything
milder is damped as before. ``None`` (default) preserves today's behavior
bit-for-bit.
"""

from __future__ import annotations

import pytest

from ipax.ipm.hessian import LBFGSOperator
from ipax.options import LBFGSOptions
from tests._helpers import array


def _probe(op, xp):
    v = array(xp, [1.0, 0.0])
    w = array(xp, [0.0, 1.0])
    return [float(x) for x in (*op.matvec(v), *op.matvec(w))]


def test_the_knob_is_off_by_default_and_validated():
    assert LBFGSOptions().damping_skip_ratio is None
    assert LBFGSOptions(damping_skip_ratio=1.0).damping_skip_ratio == 1.0
    with pytest.raises(ValueError, match="damping_skip_ratio"):
        LBFGSOptions(damping_skip_ratio=0.0)
    with pytest.raises(ValueError, match="damping_skip_ratio"):
        LBFGSOptions(damping_skip_ratio=-1.0)
    with pytest.raises(ValueError, match="damping_skip_ratio"):
        LBFGSOptions(damping_skip_ratio=float("nan"))


def test_strongly_contradicted_pair_is_skipped(namespace):
    # Fresh operator: B = I, so s·Bs = ‖s‖² = 1 and s·y = −2 < −1·1.
    op = LBFGSOperator(2, LBFGSOptions(damping_skip_ratio=1.0))
    before = _probe(op, namespace)

    op.update(array(namespace, [1.0, 0.0]), array(namespace, [-2.0, 0.0]))

    assert _probe(op, namespace) == before  # the pair left no trace


def test_mildly_indefinite_pair_is_still_damped(namespace):
    # s·y = −0.5 is above the −1·s·Bs threshold: today's Powell blend applies,
    # the operator changes, and it stays positive definite.
    op = LBFGSOperator(2, LBFGSOptions(damping_skip_ratio=1.0))
    before = _probe(op, namespace)

    op.update(array(namespace, [1.0, 0.0]), array(namespace, [-0.5, 0.0]))

    after = _probe(op, namespace)
    assert after != before
    s = array(namespace, [1.0, 0.0])
    assert float(namespace.sum(s * op.matvec(s))) > 0.0


def test_none_preserves_todays_damping(namespace):
    # The identical strongly-contradicted pair, default options: Powell
    # damping fabricates a PD update — the operator must change (that IS
    # today's behavior, which the default has to keep bit-for-bit).
    op = LBFGSOperator(2, LBFGSOptions())
    before = _probe(op, namespace)

    op.update(array(namespace, [1.0, 0.0]), array(namespace, [-2.0, 0.0]))

    assert _probe(op, namespace) != before
