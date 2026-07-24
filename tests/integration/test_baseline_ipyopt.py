"""The sparse-native ipyopt (IPOPT) baseline.

Gated on ``ipyopt`` being installed (a compiled binding, not a CI dependency yet),
so this skips cleanly where it is absent. The baseline exists to give the S2MPJ /
TROTS accuracy sweeps an IPOPT column on the language-neutral axis (iteration
count + correctness); unlike the SciPy-style ``cyipopt`` path it consumes the
constraint Jacobian as a COO pattern, so it does not densify at RT scale.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("ipyopt")

from benchmarks.baselines import (
    BaselineUnsupported,
    IpyoptBaseline,
    available_baselines,
)
from benchmarks.runners.s2mpj_baselines import _objective_ok, _verdict
from ipax.backend.operators import LinearOperator
from ipax.problem.base import Problem
from ipax.testing.problems import HS71


def test_available_baselines_includes_ipyopt():
    assert "ipyopt" in [b.name for b in available_baselines()]


def test_ipyopt_solves_hs71_to_the_known_optimum():
    # HS71 (Hock–Schittkowski): a nonlinear objective with one equality and one
    # inequality constraint plus bounds — exercises every constraint-block path.
    # Documented optimum 17.0140173 at (1, 4.743, 3.821, 1.379).
    problem = HS71(np)
    result = IpyoptBaseline().solve(problem, np.array([1.0, 5.0, 5.0, 1.0]))

    assert result.name == "ipyopt"
    assert result.success
    assert math.isclose(result.objective, 17.0140173, rel_tol=1e-6)
    np.testing.assert_allclose(
        result.x, [1.0, 4.7429994, 3.8211503, 1.3794082], atol=1e-4
    )
    assert result.n_iter > 0  # IPOPT's own iteration count, not a wall-clock proxy


class _MatrixFreeIneq(Problem):
    """Minimal problem whose inequality Jacobian is matrix-free (no COO)."""

    @property
    def n_vars(self) -> int:
        return 2

    def objective(self, x):
        return float(x[0] ** 2 + x[1] ** 2)

    def gradient(self, x):
        return 2.0 * np.asarray(x)

    def ineq_constraints(self, x):
        return np.asarray([1.0 - x[0] - x[1]])  # x0 + x1 >= 1

    def ineq_jacobian(self, x):
        del x

        class _Free(LinearOperator):
            shape = (1, 2)

            def matvec(self, v):
                return np.asarray([-v[0] - v[1]])

            def rmatvec(self, v):
                return np.asarray([-v[0], -v[0]])

            def matmat(self, V):
                return (-V[0] - V[1]).reshape(1, -1)

        return _Free()


def test_ipyopt_rejects_matrix_free_jacobian():
    # No COO structure to hand IPOPT ⇒ BaselineUnsupported (the cross-check
    # records this as "skipped", exactly like the dense path's rejection).
    with pytest.raises(BaselineUnsupported):
        IpyoptBaseline().solve(_MatrixFreeIneq(), np.array([0.5, 0.5]))


# --- the S2MPJ comparison runner's verdict logic (no ipyopt/data needed) -----


def test_objective_ok_tolerances():
    assert _objective_ok(1.0, None) is None  # no documented optimum
    assert _objective_ok(17.0140173, 17.0140173) is True
    assert _objective_ok(17.1, 17.0140173) is False


def test_verdict_categories():
    assert _verdict(True, True) == "agree"
    assert _verdict(False, True) == "ipax-gap"  # reference solved, ipax did not
    assert _verdict(True, False) == "ipax-wins"
    assert _verdict(False, False) == "both-hard"
    assert _verdict(None, True) == "unscored"  # scored fallback handled by caller
