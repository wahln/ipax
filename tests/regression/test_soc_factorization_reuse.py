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

"""Regression: SOC reuses the step's factorization at unregularized steps.

At a ``δ_w = 0`` step the retained matrix is exactly Wächter & Biegler
2006's SOC choice (§2.4, eq. (26): "the same matrix as in (13)... to avoid
additional matrix factorizations"), yet ipax's SOC closure re-entered
``_solve_step`` — rebuilding the operator and re-running the δ_w ladder, a
fresh factorization per round. This pins the re-solve fast path on a
problem whose SOC iterations are all unregularized (HS71): no
``_solve_step`` re-entry during the line search, and the reuse helper
actually running there. (At ``δ_w > 0`` steps SOC deliberately keeps the
fresh δ_w = 0 solve — reusing the inflated matrix rerouted
ZAMB2/ACOPP30/TWIRIMD1 onto restoration-heavy trajectories, measured
2026-09-02 — so the in-search ladder stays legitimate on such iterations;
HS71 has none.)
"""

from __future__ import annotations

import pytest

from ipax import Options, Status, solve
from ipax.ipm.driver import IPMDriver
from ipax.ipm.filter_ls import FilterLineSearch
from ipax.testing.problems import HS71
from tests._helpers import array


@pytest.mark.parametrize("linsolve", ["dense", "krylov"])
def test_soc_solves_reuse_the_step_factorization(namespace, linsolve, monkeypatch):
    in_search = {"active": False}
    soc_invoked = {"n": 0}
    reused = {"n": 0}

    orig_reuse = IPMDriver._resolve_reused_factorization

    def counting_reuse(self, *args, **kwargs):
        if in_search["active"]:
            reused["n"] += 1
        return orig_reuse(self, *args, **kwargs)

    orig_search = FilterLineSearch.search

    def marked_search(self, *args, **kwargs):
        # Count SOC *invocations* (not acceptances): the guard below is
        # non-vacuous as long as the soc callback ran at all, even if a
        # future tuning change made its corrected points get rejected.
        inner_soc = kwargs.get("soc")
        if inner_soc is not None:

            def counting_soc(alpha, _inner=inner_soc):
                soc_invoked["n"] += 1
                return _inner(alpha)

            kwargs["soc"] = counting_soc
        in_search["active"] = True
        try:
            return orig_search(self, *args, **kwargs)
        finally:
            in_search["active"] = False

    orig_step = IPMDriver._solve_step

    def guarded_step(self, *args, **kwargs):
        # HS71's SOC iterations are all unregularized (δ_w = 0), so every SOC
        # solve must be a re-solve on the retained factorization — never a
        # rebuild through the δ_w ladder. (On a problem with δ_w > 0 SOC
        # iterations the ladder is legitimate; see the module docstring.)
        assert not in_search["active"], (
            "SOC re-entered _solve_step (fresh operator + δ_w ladder) at an "
            "unregularized step instead of reusing the step's factorization "
            "(W&B 2006 §2.4, eq. (26))"
        )
        return orig_step(self, *args, **kwargs)

    monkeypatch.setattr(FilterLineSearch, "search", marked_search)
    monkeypatch.setattr(IPMDriver, "_solve_step", guarded_step)
    monkeypatch.setattr(IPMDriver, "_resolve_reused_factorization", counting_reuse)

    result = solve(
        HS71(namespace),
        array(namespace, [1.0, 5.0, 5.0, 1.0]),
        options=Options(linsolve=linsolve),
    )

    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    # Non-vacuous: HS71 under defaults invokes SOC several times, so the
    # guard above actually saw SOC solves — and they took the reuse path.
    assert soc_invoked["n"] > 0
    assert reused["n"] > 0
