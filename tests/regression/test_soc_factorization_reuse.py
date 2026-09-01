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

"""Regression: SOC solves reuse the step's factorization (W&B 2006, §2.4).

Eq. (26) computes the corrected step with "the same matrix as in (13)" —
same ``δ_w``/``δ_c``, same factorization — explicitly "to avoid additional
matrix factorizations". ipax's SOC closure instead re-entered
``_solve_step``, which rebuilt the operator and re-ran the δ_w ladder from
zero: a fresh factorization per SOC round and, whenever the step had been
accepted at ``δ_w > 0``, a *different, less regularized* matrix than the
direction being corrected. The centrality correctors already reused the
factorization (``_solve_targets_reused_operator``); this pins that SOC does
too.
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
        # The only solver activity inside the line search is the SOC solve,
        # which must be a re-solve on the retained factorization — never a
        # rebuild through the δ_w ladder.
        assert not in_search["active"], (
            "SOC re-entered _solve_step (fresh operator + δ_w ladder) instead "
            "of reusing the step's factorization (W&B 2006 §2.4, eq. (26))"
        )
        return orig_step(self, *args, **kwargs)

    monkeypatch.setattr(FilterLineSearch, "search", marked_search)
    monkeypatch.setattr(IPMDriver, "_solve_step", guarded_step)

    result = solve(
        HS71(namespace),
        array(namespace, [1.0, 5.0, 5.0, 1.0]),
        options=Options(linsolve=linsolve),
    )

    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    # Non-vacuous: HS71 under defaults invokes SOC several times, so the
    # guard above actually saw SOC solves.
    assert soc_invoked["n"] > 0
