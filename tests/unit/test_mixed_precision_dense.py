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

"""Mixed-precision condensed Gram (fp32 accumulate) + fp64 iterative refinement.

``DenseOptions(gram_dtype="float32")`` accumulates the FLOP-dominant inequality
Gram term ``∇gᵀ Σ_s ∇g`` in float32 (the RT dose matrices are float32 in the
source data anyway) and restores working accuracy with fixed-precision
iterative refinement against the *exact* float64 operator matvec (Carson &
Higham 2018). The safety net: a refinement stall or a precision-caused PD
failure rebuilds the exact matrix and permanently switches the solver instance
back to native precision — the endgame regime where ``κ(N)·u₃₂ ≳ 1``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ipax.backend.operators import Dense, Diagonal, LinearOperator
from ipax.ipm.kkt import _CondensedOperator
from ipax.linalg.dense import DenseSolver
from ipax.linalg.solver import LinearSolveError
from ipax.options import DenseOptions


def _condensed(namespace, *, n=24, m=96, seed=0, scale=10.0):
    """A well-conditioned condensed system with real fp32-visible dynamic range."""
    xp = namespace
    rng = np.random.default_rng(seed)
    w_diag = rng.uniform(1.0, 2.0, size=n)
    jac = rng.standard_normal((m, n)) * scale
    sigma_x = rng.uniform(0.5, 1.5, size=n)
    sigma_s = rng.uniform(1e-3, 1.0, size=m)
    op = _CondensedOperator(
        Diagonal(xp.asarray(w_diag, dtype=xp.float64)),
        Diagonal(xp.asarray(sigma_x, dtype=xp.float64)),
        Diagonal(xp.asarray(sigma_s, dtype=xp.float64)),
        Dense(xp.asarray(jac, dtype=xp.float64)),
        0.0,
    )
    rhs = xp.asarray(rng.standard_normal(n), dtype=xp.float64)
    return op, rhs


def test_gram_dtype_option_is_validated():
    assert DenseOptions().gram_dtype == "native"
    assert DenseOptions(gram_dtype="float32").gram_dtype == "float32"
    with pytest.raises(ValueError, match="gram_dtype"):
        DenseOptions(gram_dtype="float16")
    with pytest.raises(ValueError, match="refine_max_iters"):
        DenseOptions(refine_max_iters=0)
    with pytest.raises(ValueError, match="refine_tol"):
        DenseOptions(refine_tol=0.0)
    with pytest.raises(ValueError, match="refine_stall_ratio"):
        DenseOptions(refine_stall_ratio=1.5)


def test_mixed_materialization_is_actually_reduced(namespace):
    # The mixed matrix must differ from the exact one at the float32 rounding
    # scale — proof the accumulation really ran in reduced precision — while
    # still being float64-typed and close.
    op, rhs = _condensed(namespace)
    exact = np.asarray(op.dense_matrix(rhs))
    mixed = np.asarray(op.dense_matrix_mixed(rhs, "float32"))
    assert mixed.dtype == np.float64
    rel = np.max(np.abs(mixed - exact)) / np.max(np.abs(exact))
    assert 1e-12 < rel < 1e-3


def test_mixed_solve_matches_native_after_refinement(namespace):
    op, rhs = _condensed(namespace)
    native = DenseSolver(DenseOptions())
    native.factor(op)
    x_native = np.asarray(native.solve(rhs))

    mixed = DenseSolver(DenseOptions(gram_dtype="float32"))
    mixed.factor(op)
    x_mixed = np.asarray(mixed.solve(rhs))

    np.testing.assert_allclose(x_mixed, x_native, rtol=1e-6, atol=1e-9)
    # The fp32-factor first solve is not fp64-accurate on its own: refinement
    # must have corrected it at least once.
    assert mixed.refine_iterations >= 1
    assert "float32" in mixed.describe()
    assert "float32" not in native.describe()


def test_refinement_meets_residual_tolerance(namespace):
    xp = namespace
    op, rhs = _condensed(namespace, seed=3)
    options = DenseOptions(gram_dtype="float32")
    solver = DenseSolver(options)
    solver.factor(op)
    x = solver.solve(rhs)

    residual = np.asarray(rhs - op.matvec(x))
    bnorm = float(np.max(np.abs(np.asarray(rhs))))
    # Modest slack over refine_tol: the loop exits at the first residual at or
    # below tol·‖b‖, so the achieved residual is bounded by it exactly; slack
    # covers the matvec evaluation rounding.
    assert float(np.max(np.abs(residual))) <= 10.0 * options.refine_tol * bnorm
    del xp


class _FakeMixedOperator(LinearOperator):
    """Exact SPD system whose 'mixed' materialization is deliberately wrong.

    ``dense_matrix`` returns the exact matrix; ``dense_matrix_mixed`` returns a
    perturbed (``err``-scaled, still SPD) copy, so refinement against the exact
    matvec stalls and the solver must rebuild + permanently disable mixed mode.
    """

    def __init__(self, xp, a: np.ndarray, err: float) -> None:
        self._xp = xp
        self._a = a
        self._err = err
        self.mixed_calls = 0
        self.exact_calls = 0

    @property
    def shape(self):
        return self._a.shape

    def matvec(self, v):
        return self._xp.asarray(self._a, dtype=self._xp.float64) @ v

    def matmat(self, V):
        return self._xp.asarray(self._a, dtype=self._xp.float64) @ V

    def primal_block(self):
        return self

    def dense_matrix(self, like=None):
        self.exact_calls += 1
        return self._xp.asarray(self._a, dtype=self._xp.float64)

    def dense_matrix_mixed(self, like, gram_dtype):
        assert gram_dtype == "float32"
        self.mixed_calls += 1
        rng = np.random.default_rng(0)
        g = rng.standard_normal(self._a.shape)
        # PSD perturbation: the mixed matrix stays PD (the guard passes) while
        # ‖A⁻¹E‖ ≫ 1, so refinement against the exact matvec must diverge.
        e = self._err * (g @ g.T)
        return self._xp.asarray(self._a + e, dtype=self._xp.float64)


def _spd(n=12, seed=5, cond=100.0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.logspace(0.0, math.log10(cond), n)
    return q @ np.diag(eigs) @ q.T


def test_refinement_stall_rebuilds_exact_and_disables_mixed(namespace):
    xp = namespace
    a = _spd()
    op = _FakeMixedOperator(xp, a, err=0.05)
    rng = np.random.default_rng(1)
    rhs = xp.asarray(rng.standard_normal(a.shape[0]), dtype=xp.float64)

    solver = DenseSolver(DenseOptions(gram_dtype="float32"))
    solver.factor(op)
    x = np.asarray(solver.solve(rhs))

    np.testing.assert_allclose(x, np.linalg.solve(a, np.asarray(rhs)), rtol=1e-8)
    assert op.mixed_calls == 1
    assert op.exact_calls >= 1
    # Permanently native from here on: a fresh factorization must not try
    # the mixed materialization again.
    solver.factor(op)
    np.asarray(solver.solve(rhs))
    assert op.mixed_calls == 1
    # Sticky honesty marker: the run used the reduced Gram before disabling.
    assert solver.describe() == "dense (gram=float32->native)"


def test_precision_caused_pd_failure_falls_back_to_exact(namespace):
    xp = namespace
    a = _spd()

    class _NonPDMixed(_FakeMixedOperator):
        def dense_matrix_mixed(self, like, gram_dtype):
            self.mixed_calls += 1
            # Indefinite by construction — a precision-noise PD failure.
            return self._xp.asarray(
                self._a - 2.0 * np.trace(self._a) * np.eye(self._a.shape[0]),
                dtype=self._xp.float64,
            )

    op = _NonPDMixed(xp, a, err=0.0)
    rng = np.random.default_rng(2)
    rhs = xp.asarray(rng.standard_normal(a.shape[0]), dtype=xp.float64)

    solver = DenseSolver(DenseOptions(gram_dtype="float32"))
    solver.factor(op)
    x = np.asarray(solver.solve(rhs))  # must NOT raise LinearSolveError

    np.testing.assert_allclose(x, np.linalg.solve(a, np.asarray(rhs)), rtol=1e-8)
    assert op.mixed_calls == 1
    solver.factor(op)
    np.asarray(solver.solve(rhs))
    assert op.mixed_calls == 1  # permanently disabled


def test_genuine_non_pd_still_raises(namespace):
    # When the *exact* matrix is also non-PD the failure is real: it must
    # surface as LinearSolveError (driving δ_w escalation) and must NOT
    # permanently disable mixed mode (precision was not the cause).
    xp = namespace
    a = _spd()
    indefinite = a - 2.0 * np.trace(a) * np.eye(a.shape[0])

    class _BothNonPD(_FakeMixedOperator):
        def dense_matrix(self, like=None):
            self.exact_calls += 1
            return self._xp.asarray(indefinite, dtype=self._xp.float64)

        def dense_matrix_mixed(self, like, gram_dtype):
            self.mixed_calls += 1
            return self._xp.asarray(indefinite, dtype=self._xp.float64)

    op = _BothNonPD(xp, a, err=0.0)
    rng = np.random.default_rng(4)
    rhs = xp.asarray(rng.standard_normal(a.shape[0]), dtype=xp.float64)

    solver = DenseSolver(DenseOptions(gram_dtype="float32"))
    solver.factor(op)
    with pytest.raises(LinearSolveError):
        solver.solve(rhs)
    # Mixed stays enabled: the next factorization (after δ_w escalation) may
    # try reduced precision again.
    op2 = _FakeMixedOperator(xp, a, err=0.0)
    solver.factor(op2)
    np.asarray(solver.solve(rhs))
    assert op2.mixed_calls == 1


def test_native_default_never_calls_mixed(namespace):
    xp = namespace
    a = _spd()
    op = _FakeMixedOperator(xp, a, err=0.05)
    rng = np.random.default_rng(6)
    rhs = xp.asarray(rng.standard_normal(a.shape[0]), dtype=xp.float64)

    solver = DenseSolver(DenseOptions())
    solver.factor(op)
    np.asarray(solver.solve(rhs))
    assert op.mixed_calls == 0
    assert solver.refine_iterations == 0


def test_masked_indefinite_exact_matrix_still_escalates(namespace):
    # The PD probe runs on the *approximate* matrix, so an indefinite exact N
    # whose mixed materialization is PD passes the guard. The stall detector
    # is the second half of the defense: refinement against the exact matvec
    # must stall (masking forces κ·u32 ≳ 1), the exact rebuild re-probes on
    # the exact matrix, and the failure surfaces as LinearSolveError (δ_w
    # escalation) — never a returned step.
    xp = namespace
    a = _spd()
    indefinite = a - 1.2 * np.trace(a) / a.shape[0] * np.eye(a.shape[0])

    class _MaskedIndefinite(_FakeMixedOperator):
        def matvec(self, v):
            return self._xp.asarray(indefinite, dtype=self._xp.float64) @ v

        def dense_matrix(self, like=None):
            self.exact_calls += 1
            return self._xp.asarray(indefinite, dtype=self._xp.float64)

        def dense_matrix_mixed(self, like, gram_dtype):
            self.mixed_calls += 1
            # PD "masking" of the indefinite exact block.
            return self._xp.asarray(a, dtype=self._xp.float64)

    op = _MaskedIndefinite(xp, a, err=0.0)
    rng = np.random.default_rng(8)
    rhs = xp.asarray(rng.standard_normal(a.shape[0]), dtype=xp.float64)

    solver = DenseSolver(DenseOptions(gram_dtype="float32"))
    solver.factor(op)
    with pytest.raises(LinearSolveError):
        solver.solve(rhs)
    assert op.exact_calls >= 1  # the exact re-probe ran


def test_pre_keyword_gram_operator_degrades_to_exact(namespace):
    # A third-party operator implementing gram(self, weights) without the
    # accumulate_dtype keyword must not break the mixed route: the TypeError
    # retry serves the exact Gram instead.
    xp = namespace
    rng = np.random.default_rng(9)
    jac = rng.standard_normal((30, 8))

    class _LegacyGram(LinearOperator):
        @property
        def shape(self):
            return jac.shape

        def matvec(self, v):
            return xp.asarray(jac, dtype=xp.float64) @ v

        def rmatvec(self, v):
            j = xp.asarray(jac, dtype=xp.float64)
            return xp.permute_dims(j, (1, 0)) @ v

        def gram(self, weights):  # pre-keyword signature, on purpose
            j = np.asarray(jac)
            w = np.asarray(weights)
            return xp.asarray(j.T @ (w[:, None] * j), dtype=xp.float64)

    n, m = 8, 30
    op = _CondensedOperator(
        Diagonal(xp.ones(n, dtype=xp.float64)),
        Diagonal(xp.ones(n, dtype=xp.float64)),
        Diagonal(xp.asarray(rng.uniform(0.1, 1.0, size=m), dtype=xp.float64)),
        _LegacyGram(),
        0.0,
    )
    like = xp.ones(n, dtype=xp.float64)
    mixed = np.asarray(op.dense_matrix_mixed(like, "float32"))
    exact = np.asarray(op.dense_matrix(like))
    np.testing.assert_array_equal(mixed, exact)  # degraded to the exact Gram


def test_sparse_fast_path_end_to_end_reduced():
    # The production path at RT scale: a sparse-structured (CSROperator)
    # inequality Jacobian routed through _sparse_gram -> adapter
    # gram(accumulate_dtype=...) -> the reduced dense accumulate. NumPy-only
    # (the SciPy adapter owns this path).
    import array_api_compat.numpy as xp
    import scipy.sparse as sp

    from ipax.backend.operators import CSROperator

    rng = np.random.default_rng(10)
    n, m = 10, 60
    jac = sp.csr_matrix(rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.6) * 1e3)
    jac.eliminate_zeros()
    jac.sort_indices()
    J = CSROperator(
        xp.asarray(np.asarray(jac.indptr, dtype=np.int64)),
        xp.asarray(np.asarray(jac.indices, dtype=np.int64)),
        xp.asarray(jac.data),
        (m, n),
    )
    op = _CondensedOperator(
        Diagonal(xp.asarray(rng.uniform(1.0, 2.0, size=n))),
        Diagonal(xp.asarray(rng.uniform(0.5, 1.5, size=n))),
        Diagonal(xp.asarray(rng.uniform(1e-3, 1.0, size=m))),
        J,
        0.0,
    )
    rhs = xp.asarray(rng.standard_normal(n))
    exact = np.asarray(op.dense_matrix(rhs))
    mixed = np.asarray(op.dense_matrix_mixed(rhs, "float32"))
    rel = np.max(np.abs(mixed - exact)) / np.max(np.abs(exact))
    assert 1e-12 < rel < 1e-3  # the adapter really accumulated in float32

    native = DenseSolver(DenseOptions())
    native.factor(op)
    x_native = np.asarray(native.solve(rhs))
    solver = DenseSolver(DenseOptions(gram_dtype="float32"))
    solver.factor(op)
    x_mixed = np.asarray(solver.solve(rhs))
    np.testing.assert_allclose(x_mixed, x_native, rtol=1e-6, atol=1e-9)
    assert solver.refine_iterations >= 1


def test_float32_working_dtype_does_not_burn_the_refinement_budget(namespace):
    # DenseOptions.refine_tol assumes float64; on a float32 working dtype the
    # tolerance is floored at a multiple of eps(float32) so refinement
    # converges instead of exhausting its budget and permanently disabling
    # the mixed route.
    xp = namespace
    rng = np.random.default_rng(11)
    n, m = 12, 40
    op = _CondensedOperator(
        Diagonal(xp.asarray(rng.uniform(1.0, 2.0, size=n), dtype=xp.float32)),
        Diagonal(xp.asarray(rng.uniform(0.5, 1.5, size=n), dtype=xp.float32)),
        Diagonal(xp.asarray(rng.uniform(0.1, 1.0, size=m), dtype=xp.float32)),
        Dense(xp.asarray(rng.standard_normal((m, n)), dtype=xp.float32)),
        0.0,
    )
    rhs = xp.asarray(rng.standard_normal(n), dtype=xp.float32)
    solver = DenseSolver(DenseOptions(gram_dtype="float32"))
    solver.factor(op)
    x = np.asarray(solver.solve(rhs))

    assert np.all(np.isfinite(x))
    assert not solver._mixed_disabled  # no spurious stall/exhaustion fallback


def test_equality_saddle_ignores_gram_dtype():
    # Documented limitation: equality-constrained saddle systems assemble
    # exactly — gram_dtype is a no-op there (and must not break the solve).
    import array_api_compat.numpy as xp

    from ipax import Options, Status, solve
    from ipax.testing.problems import EqualityConstrainedQP

    problem = EqualityConstrainedQP(xp)
    x0 = xp.zeros(problem.n_vars, dtype=xp.float64)
    result = solve(
        problem,
        x0,
        options=Options(
            hessian="exact",
            linsolve="dense",
            dense=DenseOptions(gram_dtype="float32"),
        ),
    )
    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    assert "float32" not in (result.linear_solver or "")


def test_gram_accumulate_dtype_forwards_through_wrappers(namespace):
    # VStack and the auto-scaling _RowScaled wrapper must forward the
    # accumulate_dtype request to the operators that actually form the Gram.
    from ipax.backend.operators import VStack
    from ipax.problem.scaling import _RowScaled

    xp = namespace
    seen: list[str | None] = []

    class _SpyGram(LinearOperator):
        def __init__(self, n, rows):
            self._n = n
            self._rows = rows

        @property
        def shape(self):
            return (self._rows, self._n)

        def matvec(self, v):
            raise NotImplementedError

        def gram(self, weights, *, accumulate_dtype=None):
            seen.append(accumulate_dtype)
            return xp.zeros((self._n, self._n), dtype=xp.float64)

    stacked = VStack((_SpyGram(4, 3), _SpyGram(4, 5)))
    scaled = _RowScaled(stacked, xp.ones(8, dtype=xp.float64))
    scaled.gram(xp.ones(8, dtype=xp.float64), accumulate_dtype="float32")
    assert seen == ["float32", "float32"]


def test_solve_hs35_with_mixed_gram_matches_reference(namespace):
    # End-to-end: the full IPM with the mixed-precision dense route converges
    # to the known optimum of an inequality QP on every backend.
    from ipax import Options, Status, solve
    from ipax.testing.problems import HS35

    xp = namespace
    problem = HS35(xp)
    x0 = xp.asarray([0.5, 0.5, 0.5], dtype=xp.float64)
    result = solve(
        problem,
        x0,
        options=Options(
            hessian="exact",
            linsolve="dense",
            dense=DenseOptions(gram_dtype="float32"),
        ),
    )
    assert result.status in (Status.OPTIMAL, Status.ACCEPTABLE)
    np.testing.assert_allclose(
        np.asarray(result.x), np.asarray(problem.known_solution()), atol=1e-6
    )
    # The mixed route really engaged (sticky "->native" also counts: it means
    # it ran reduced first and self-disabled later) — guards against the
    # option silently becoming a no-op.
    assert "float32" in (result.linear_solver or "")
