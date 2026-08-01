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

"""``LBFGSOptions.seed_formula`` — the ξ initial-scaling estimate.

Both are standard Rayleigh-quotient seeds for the L-BFGS identity block, but
they diverge by exactly the δ–γ misalignment factor ``1/cos²∠(δ,γ)``
(Cauchy–Schwarz): ``"direct"`` (Nocedal & Wright eq. 7.20 inverted for the
direct Hessian, ``ξ = γᵀγ/δᵀγ``) is always ≥ ``"scalar1"`` (IPOPT's
``limited_memory_initialization``, ``ξ = δᵀγ/δᵀδ``). On badly-scaled least
squares the misalignment is catastrophic: S2MPJ ``NELSONLS`` runs the direct
seed at a median ``ξ ≈ 1e20`` against scalar1's ``6e3``, freezing every step;
``GASOIL`` goes from 508 stalled iterations to optimal in 25 (IPOPT parity)
under scalar1.
"""

from __future__ import annotations

import pytest

from ipax.ipm.hessian import LBFGSOperator
from ipax.options import LBFGSOptions
from tests._helpers import array


def _xi_seen_by_orthogonal_probe(op, xp):
    # For a single stored pair, B v = ξ·v for any v orthogonal to span{δ, γ}.
    v = array(xp, [0.0, 0.0, 1.0])
    return float(op.matvec(v)[2])


def test_default_and_validation():
    assert LBFGSOptions().seed_formula == "direct"
    assert LBFGSOptions(seed_formula="scalar1").seed_formula == "scalar1"
    with pytest.raises(ValueError, match="seed_formula"):
        LBFGSOptions(seed_formula="bb3")


def test_direct_seed_is_the_default_formula(namespace):
    op = LBFGSOperator(3, LBFGSOptions())
    op.update(array(namespace, [1.0, 0.0, 0.0]), array(namespace, [2.0, 1.0, 0.0]))

    # ξ = γᵀγ/δᵀγ = 5/2
    assert _xi_seen_by_orthogonal_probe(op, namespace) == pytest.approx(2.5)


def test_scalar1_seed_uses_the_ipopt_formula(namespace):
    op = LBFGSOperator(3, LBFGSOptions(seed_formula="scalar1"))
    op.update(array(namespace, [1.0, 0.0, 0.0]), array(namespace, [2.0, 1.0, 0.0]))

    # ξ = δᵀγ/δᵀδ = 2/1
    assert _xi_seen_by_orthogonal_probe(op, namespace) == pytest.approx(2.0)


def test_disabled_initial_scaling_wins_over_the_formula(namespace):
    op = LBFGSOperator(3, LBFGSOptions(initial_scaling=False, seed_formula="scalar1"))
    op.update(array(namespace, [1.0, 0.0, 0.0]), array(namespace, [2.0, 1.0, 0.0]))

    assert _xi_seen_by_orthogonal_probe(op, namespace) == pytest.approx(1.0)
