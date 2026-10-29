"""Regression: work arrays must be allocated on the device of the inputs.

pyRadPlan (2026-10, ipax 0.12.0): a bound-constrained fluence optimization with
``x0`` and the bounds as PyTorch tensors on ``cuda:0`` failed in the first
iteration with ``RuntimeError: Expected all tensors to be on the same device``.
The driver's bound masks — and ~90 further creation calls (``zeros``, ``ones``,
``full``, ``eye``, ``arange``) across the IPM, KKT assembly, operators and
linear solvers — passed no ``device=``, so they landed on the backend's
default device (the CPU) instead of the device the problem lives on.

``array-api-strict`` has logical devices and rejects mixing them, so building
every problem array on a non-default device reproduces the bug on any machine.
"""

from __future__ import annotations

import pytest

from ipax import Options, solve
from ipax.backend.namespace import device
from ipax.testing.problems import (
    HS71,
    BoundConstrainedQP,
    InfeasibleEqualities,
    Maratos,
    make_rt_like_problem,
)

xps = pytest.importorskip("array_api_strict")
_DEVICE = xps.Device("device1")
_CREATION = frozenset(
    {"asarray", "zeros", "ones", "full", "empty", "eye", "arange", "linspace"}
)


class _OnDevice:
    """``array_api_strict`` with creation functions defaulting to ``_DEVICE``.

    Problems built through it keep all their data on the non-default device,
    as a GPU user's problem would; ipax itself still sees the real namespace.
    """

    def __getattr__(self, name: str):
        attr = getattr(xps, name)
        if name not in _CREATION:
            return attr

        def create(*args, **kwargs):
            kwargs.setdefault("device", _DEVICE)
            return attr(*args, **kwargs)

        return create


_XP = _OnDevice()


def _rt_like():
    problem = make_rt_like_problem(_XP, 24)
    return problem, _XP.full((problem.n_vars,), 0.5, dtype=xps.float64)


_CASES = {
    "bound-qp": lambda: (BoundConstrainedQP(_XP), _XP.asarray([0.5, 0.5])),
    "rt-like": _rt_like,
    "hs71": lambda: (HS71(_XP), _XP.asarray([1.0, 5.0, 5.0, 1.0])),
    "maratos": lambda: (Maratos(_XP), _XP.asarray([0.7, 0.7])),
    "infeasible-restoration": lambda: (InfeasibleEqualities(_XP), _XP.asarray([0.3])),
}
_OPTIONS = {
    "default": Options(),
    "lbfgs-dense": Options(hessian="lbfgs", linsolve="dense"),
    "lbfgs-krylov": Options(hessian="lbfgs", linsolve="krylov"),
}


@pytest.mark.parametrize("options", list(_OPTIONS.values()), ids=list(_OPTIONS))
@pytest.mark.parametrize("case", list(_CASES.values()), ids=list(_CASES))
def test_solve_stays_on_the_input_device(case, options):
    problem, x0 = case()
    assert device(x0) == _DEVICE
    result = solve(problem, x0, options=options)
    assert device(result.x) == _DEVICE
