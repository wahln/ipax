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
    assert DenseOptions().gram_dtype == "auto"
    assert DenseOptions(gram_dtype="native").gram_dtype == "native"
    assert DenseOptions(gram_dtype="float32").gram_dtype == "float32"
    with pytest.raises(ValueError, match="gram_dtype"):
        DenseOptions(gram_dtype="float16")
    with pytest.raises(ValueError, match="refine_max_iters"):
        DenseOptions(refine_max_iters=0)
    with pytest.raises(ValueError, match="refine_tol"):
        DenseOptions(refine_tol=0.0)
    with pytest.raises(ValueError, match="refine_accept_tol"):
        DenseOptions(refine_accept_tol=1e-12)  # below refine_tol
    with pytest.raises(ValueError, match="refine_stall_ratio"):
        DenseOptions(refine_stall_ratio=1.5)
    with pytest.raises(ValueError, match="refine_failure_limit"):
        DenseOptions(refine_failure_limit=0)


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

    def dense_matrix_mixed(self, like, gram_dtype, *, hinted_only=False):
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


class _HopelessMixed(_FakeMixedOperator):
    """Mixed materialization that is SPD (passes the guard) but structurally
    wrong: ``tr(A)·I`` gives refinement a ~1−λ_min/tr(A) contraction (an
    immediate plateau at the stall ratio) with an O(1) best residual — every
    mixed solve must be rejected. ``broken = False`` returns the exact matrix
    instead (a certified success), for exercising the failure-streak reset."""

    broken = True

    def dense_matrix_mixed(self, like, gram_dtype, *, hinted_only=False):
        self.mixed_calls += 1
        if not self.broken:
            return self._xp.asarray(self._a, dtype=self._xp.float64)
        n = self._a.shape[0]
        return self._xp.asarray(
            float(np.trace(self._a)) * np.eye(n), dtype=self._xp.float64
        )


def test_refinement_stall_rebuilds_exact_and_counts_failures(namespace):
    # Each rejected solve answers from the exact rebuild (correct step), and
    # only refine_failure_limit CONSECUTIVE failures disable the mixed route
    # for good — conditioning along an IPM run is not monotone, so one hard
    # factorization must not forfeit the savings on every later one.
    xp = namespace
    a = _spd()
    op = _HopelessMixed(xp, a, err=0.0)
    rng = np.random.default_rng(1)
    rhs = xp.asarray(rng.standard_normal(a.shape[0]), dtype=xp.float64)

    solver = DenseSolver(DenseOptions(gram_dtype="float32", refine_failure_limit=3))
    for expected_mixed_calls in (1, 2, 3):
        solver.factor(op)
        x = np.asarray(solver.solve(rhs))
        np.testing.assert_allclose(x, np.linalg.solve(a, np.asarray(rhs)), rtol=1e-8)
        assert op.mixed_calls == expected_mixed_calls
        assert op.exact_calls >= expected_mixed_calls

    # Third consecutive failure hit the limit: permanently native from here.
    solver.factor(op)
    np.asarray(solver.solve(rhs))
    assert op.mixed_calls == 3
    # Sticky honesty marker: the run used the reduced Gram before disabling.
    assert solver.describe() == "dense (gram=float32->native)"


def test_budget_exhaustion_accepts_certified_residual(namespace):
    # A refinement that runs out of budget while already below the (looser)
    # refine_accept_tol certificate is a *success*: the refined solve is
    # returned, no exact rebuild happens, and the failure counter stays 0.
    xp = namespace
    a = _spd()
    op = _FakeMixedOperator(xp, a, err=0.005)  # ρ ≈ 0.2: converging, slowly
    rng = np.random.default_rng(3)
    rhs = xp.asarray(rng.standard_normal(a.shape[0]), dtype=xp.float64)

    solver = DenseSolver(
        DenseOptions(
            gram_dtype="float32",
            refine_tol=1e-15,  # unreachable in the budget, on purpose
            refine_accept_tol=0.05,
            refine_max_iters=4,
        )
    )
    solver.factor(op)
    x = solver.solve(rhs)

    residual = np.asarray(rhs - op.matvec(x))
    bnorm = float(np.max(np.abs(np.asarray(rhs))))
    assert float(np.max(np.abs(residual))) <= 0.05 * bnorm
    assert op.exact_calls == 0  # accepted: no exact rebuild
    assert solver._mixed_failures == 0
    solver.factor(op)
    solver.solve(rhs)
    assert op.mixed_calls == 2  # still on the mixed route


def test_failure_counter_resets_on_success(namespace):
    xp = namespace
    a = _spd()
    op = _HopelessMixed(xp, a, err=0.0)  # broken: every refine fails
    rng = np.random.default_rng(4)
    rhs = xp.asarray(rng.standard_normal(a.shape[0]), dtype=xp.float64)

    solver = DenseSolver(DenseOptions(gram_dtype="float32", refine_failure_limit=2))
    solver.factor(op)
    solver.solve(rhs)
    assert solver._mixed_failures == 1 and not solver._mixed_disabled

    op.broken = False  # mixed == exact: refinement certifies immediately
    solver.factor(op)
    solver.solve(rhs)
    assert solver._mixed_failures == 0  # success resets the streak

    op.broken = True
    solver.factor(op)
    solver.solve(rhs)
    assert solver._mixed_failures == 1 and not solver._mixed_disabled
    solver.factor(op)
    solver.solve(rhs)
    assert solver._mixed_disabled  # two consecutive failures = the limit


def test_precision_caused_pd_failure_falls_back_to_exact(namespace):
    xp = namespace
    a = _spd()

    class _NonPDMixed(_FakeMixedOperator):
        def dense_matrix_mixed(self, like, gram_dtype, *, hinted_only=False):
            self.mixed_calls += 1
            # Indefinite by construction — a precision-noise PD failure.
            return self._xp.asarray(
                self._a - 2.0 * np.trace(self._a) * np.eye(self._a.shape[0]),
                dtype=self._xp.float64,
            )

    op = _NonPDMixed(xp, a, err=0.0)
    rng = np.random.default_rng(2)
    rhs = xp.asarray(rng.standard_normal(a.shape[0]), dtype=xp.float64)

    solver = DenseSolver(DenseOptions(gram_dtype="float32", refine_failure_limit=1))
    solver.factor(op)
    x = np.asarray(solver.solve(rhs))  # must NOT raise LinearSolveError

    np.testing.assert_allclose(x, np.linalg.solve(a, np.asarray(rhs)), rtol=1e-8)
    assert op.mixed_calls == 1
    solver.factor(op)
    np.asarray(solver.solve(rhs))
    assert op.mixed_calls == 1  # limit 1: disabled by the single PD mismatch


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

        def dense_matrix_mixed(self, like, gram_dtype, *, hinted_only=False):
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

        def dense_matrix_mixed(self, like, gram_dtype, *, hinted_only=False):
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


def test_refine_tolerance_floors_on_the_working_dtype(namespace):
    # refine_tol's default assumes float64. The narrower-than-working guard
    # keeps float32 solves off the mixed route entirely, so this floor is a
    # backstop for any future candidate narrower than a float32 working dtype
    # — pinned directly on _refine so it cannot rot: with an exact factor, a
    # float32 rhs must certify at iteration 0 rather than chase an
    # unreachable 1e-10.
    xp = namespace
    a = _spd(n=8)
    op = _FakeMixedOperator(xp, a, err=0.0)

    class _F32(_FakeMixedOperator):
        def matvec(self, v):
            return self._xp.asarray(self._a, dtype=self._xp.float32) @ v

    op = _F32(xp, a.astype(np.float32).astype(np.float64), err=0.0)
    rng = np.random.default_rng(11)
    rhs = xp.asarray(rng.standard_normal(a.shape[0]), dtype=xp.float32)

    solver = DenseSolver(DenseOptions(gram_dtype="float32"))
    solver.factor(op)
    solver._matrix = xp.asarray(op._a, dtype=xp.float32)
    solver._mixed_engaged = True
    x = xp.linalg.solve(solver._matrix, rhs)

    refined = solver._refine(x, rhs, xp)
    assert refined is not None  # floored tolerance is reachable
    assert solver.refine_iterations <= 1


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


def test_gram_accumulate_dtype_hint_protocol(namespace):
    # The hint is how gram_dtype="auto" discovers that the constraint data
    # carries only float32 information: storage-level metadata, forwarded by
    # every wrapper. Absent evidence, the answer is None (native).
    from ipax.backend.operators import CSROperator, VStack
    from ipax.problem.scaling import _RowScaled

    xp = namespace
    dense64 = Dense(xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float64))
    dense32 = Dense(xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32))
    assert dense64.gram_accumulate_dtype_hint() is None
    assert dense32.gram_accumulate_dtype_hint() == "float32"

    indptr = xp.asarray([0, 1, 2])
    indices = xp.asarray([0, 1])
    values64 = xp.asarray([1.0, 2.0], dtype=xp.float64)
    csr64 = CSROperator(indptr, indices, values64, (2, 2))
    assert csr64.gram_accumulate_dtype_hint() is None
    # Declared source precision: fp64 storage whose values are exact upcasts
    # of float32 data (the TROTS situation) hints without storing fp32.
    hinted = CSROperator(
        indptr,
        indices,
        values64,
        (2, 2),
        values_dtype_hint="float32",
    )
    assert hinted.gram_accumulate_dtype_hint() == "float32"
    csr32 = CSROperator(
        indptr, indices, xp.asarray([1.0, 2.0], dtype=xp.float32), (2, 2)
    )
    assert csr32.gram_accumulate_dtype_hint() == "float32"

    # A stack hints when ANY block does, because it honors the request
    # per block (see test_stack_reduces_only_the_hinted_blocks); row scaling
    # forwards its inner operator's hint unchanged.
    assert VStack((hinted, csr32)).gram_accumulate_dtype_hint() == "float32"
    assert VStack((csr64, hinted)).gram_accumulate_dtype_hint() == "float32"
    assert VStack((csr64, csr64)).gram_accumulate_dtype_hint() is None
    d = xp.ones(2, dtype=xp.float64)
    assert _RowScaled(hinted, d).gram_accumulate_dtype_hint() == "float32"
    assert _RowScaled(csr64, d).gram_accumulate_dtype_hint() is None

    # The condensed operator forwards its inequality Jacobian's hint.
    op = _CondensedOperator(
        Diagonal(xp.ones(2, dtype=xp.float64)),
        Diagonal(xp.ones(2, dtype=xp.float64)),
        Diagonal(xp.ones(2, dtype=xp.float64)),
        hinted,
        0.0,
    )
    assert op.gram_accumulate_dtype_hint() == "float32"


def _spy_stack(xp, n=4):
    """A VStack of a float32-hinting block and a plain float64 block, each
    recording the ``accumulate_dtype`` it is actually asked for."""
    from ipax.backend.operators import VStack

    seen: dict[str, object] = {}

    class _Block(LinearOperator):
        def __init__(self, label, rows, hint):
            self._label, self._rows, self._hint = label, rows, hint

        @property
        def shape(self):
            return (self._rows, n)

        def matvec(self, v):
            # Consistent zero operator: matches the zero Gram below, so the
            # condensed block is well defined and refinement can certify.
            return xp.zeros((self._rows,), dtype=xp.float64)

        def rmatvec(self, v):
            return xp.zeros((n,), dtype=xp.float64)

        def gram_accumulate_dtype_hint(self):
            return self._hint

        def gram(self, weights, *, accumulate_dtype=None, hinted_only=False):
            if (
                hinted_only
                and accumulate_dtype is not None
                and self.gram_accumulate_dtype_hint() != accumulate_dtype
            ):
                accumulate_dtype = None
            seen[self._label] = accumulate_dtype
            return xp.zeros((n, n), dtype=xp.float64)

    stack = VStack((_Block("f32", 3, "float32"), _Block("f64", 5, None)))
    return stack, seen


def test_stack_reduces_only_the_hinted_blocks(namespace):
    # The VMAT shape: a block assembled from mixed-precision sources. The
    # float32-sourced block accumulates reduced; the genuinely-float64 block
    # must stay exact rather than being silently reduced along with it.
    xp = namespace
    stack, seen = _spy_stack(xp)

    stack.gram(
        xp.ones(8, dtype=xp.float64), accumulate_dtype="float32", hinted_only=True
    )
    assert seen == {"f32": "float32", "f64": None}

    # A user-FORCED reduction still applies everywhere (hinted_only=False) —
    # that is what makes gram_dtype="float32" meaningful on data that carries
    # no float32 hint at all (e.g. TROTS Protons, genuinely float64).
    seen.clear()
    stack.gram(xp.ones(8, dtype=xp.float64), accumulate_dtype="float32")
    assert seen == {"f32": "float32", "f64": "float32"}


def test_row_scaling_forwards_per_block_request(namespace):
    from ipax.problem.scaling import _RowScaled

    xp = namespace
    stack, seen = _spy_stack(xp)
    scaled = _RowScaled(stack, xp.ones(8, dtype=xp.float64))

    scaled.gram(
        xp.ones(8, dtype=xp.float64), accumulate_dtype="float32", hinted_only=True
    )

    assert seen == {"f32": "float32", "f64": None}
    assert scaled.gram_accumulate_dtype_hint() == "float32"


def test_auto_uses_per_block_and_forced_uses_everything(namespace):
    # End-to-end through the condensed operator: "auto" must request the
    # per-block form, an explicit "float32" the blanket form.
    xp = namespace
    stack, seen = _spy_stack(xp)
    op = _CondensedOperator(
        Diagonal(xp.ones(4, dtype=xp.float64)),
        Diagonal(xp.ones(4, dtype=xp.float64)),
        Diagonal(xp.ones(8, dtype=xp.float64)),
        stack,
        0.0,
    )
    like = xp.ones(4, dtype=xp.float64)

    solver = DenseSolver(DenseOptions())  # auto
    solver.factor(op)
    solver.solve(like)
    assert seen == {"f32": "float32", "f64": None}

    seen.clear()
    forced = DenseSolver(DenseOptions(gram_dtype="float32"))
    forced.factor(op)
    forced.solve(like)
    assert seen == {"f32": "float32", "f64": "float32"}


def test_auto_engages_only_on_reduced_hint(namespace):
    # Default options (gram_dtype="auto"): a hinted operator engages the
    # mixed route; an unhinted fp64 operator is a strict no-op (no mixed
    # materialization, no refinement) — auto must never regress pure-fp64
    # problems.
    xp = namespace
    a = _spd()
    rng = np.random.default_rng(12)
    rhs = xp.asarray(rng.standard_normal(a.shape[0]), dtype=xp.float64)

    class _HintedExact(_FakeMixedOperator):
        def gram_accumulate_dtype_hint(self):
            return "float32"

        def dense_matrix_mixed(self, like, gram_dtype, *, hinted_only=False):
            assert gram_dtype == "float32"  # auto resolves to a concrete dtype
            self.mixed_calls += 1
            return self._xp.asarray(self._a, dtype=self._xp.float64)

    hinted = _HintedExact(xp, a, err=0.0)
    solver = DenseSolver(DenseOptions())  # auto
    solver.factor(hinted)
    x = np.asarray(solver.solve(rhs))
    np.testing.assert_allclose(x, np.linalg.solve(a, np.asarray(rhs)), rtol=1e-8)
    assert hinted.mixed_calls == 1
    assert "float32" in solver.describe()

    plain = _FakeMixedOperator(xp, a, err=0.05)  # no hint ⇒ native
    solver2 = DenseSolver(DenseOptions())
    solver2.factor(plain)
    np.asarray(solver2.solve(rhs))
    assert plain.mixed_calls == 0
    assert solver2.refine_iterations == 0
    assert solver2.describe() == "dense"


def test_reduction_is_skipped_when_working_precision_is_not_wider(namespace):
    # "Prefer the precision the input comes in" cuts both ways: on a float32
    # WORKING dtype, accumulating the Gram in float32 is not mixed precision —
    # it is the native arithmetic plus a pointless refinement pass. Both the
    # auto hint and an explicit float32 request must resolve to native.
    xp = namespace
    rng = np.random.default_rng(14)
    n, m = 10, 40

    class _Hinted(Dense):
        def gram_accumulate_dtype_hint(self):
            return "float32"

    op = _CondensedOperator(
        Diagonal(xp.asarray(rng.uniform(1.0, 2.0, size=n), dtype=xp.float32)),
        Diagonal(xp.asarray(rng.uniform(0.5, 1.5, size=n), dtype=xp.float32)),
        Diagonal(xp.asarray(rng.uniform(0.1, 1.0, size=m), dtype=xp.float32)),
        _Hinted(xp.asarray(rng.standard_normal((m, n)), dtype=xp.float32)),
        0.0,
    )
    rhs = xp.asarray(rng.standard_normal(n), dtype=xp.float32)

    for options in (DenseOptions(), DenseOptions(gram_dtype="float32")):
        solver = DenseSolver(options)
        solver.factor(op)
        x = solver.solve(rhs)
        assert np.all(np.isfinite(np.asarray(x)))
        assert solver.describe() == "dense"  # native: no reduction, no refine
        assert solver.refine_iterations == 0

    # The same operator family in float64 *does* reduce (the hint is real).
    op64 = _CondensedOperator(
        Diagonal(xp.asarray(rng.uniform(1.0, 2.0, size=n), dtype=xp.float64)),
        Diagonal(xp.asarray(rng.uniform(0.5, 1.5, size=n), dtype=xp.float64)),
        Diagonal(xp.asarray(rng.uniform(0.1, 1.0, size=m), dtype=xp.float64)),
        _Hinted(xp.asarray(rng.standard_normal((m, n)), dtype=xp.float64)),
        0.0,
    )
    solver = DenseSolver(DenseOptions())
    solver.factor(op64)
    solver.solve(xp.asarray(rng.standard_normal(n), dtype=xp.float64))
    assert "float32" in solver.describe()


def test_auto_is_native_on_fp64_condensed_system(namespace):
    # The realistic no-op check: a fully-fp64 condensed operator under the
    # auto default materializes the exact matrix and never refines.
    op, rhs = _condensed(namespace)
    solver = DenseSolver(DenseOptions())
    solver.factor(op)
    x = np.asarray(solver.solve(rhs))

    native = DenseSolver(DenseOptions(gram_dtype="native"))
    native.factor(op)
    np.testing.assert_array_equal(x, np.asarray(native.solve(rhs)))
    assert solver.refine_iterations == 0
    assert solver.describe() == "dense"


def test_auto_end_to_end_with_hinted_sparse_jacobian():
    # The TROTS shape: fp64-stored CSR values that are exact float32 upcasts,
    # declared via values_dtype_hint — default options engage the reduced
    # accumulate and refinement certifies the solve.
    import array_api_compat.numpy as xp
    import scipy.sparse as sp

    from ipax.backend.operators import CSROperator

    rng = np.random.default_rng(13)
    n, m = 10, 60
    dense = rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.6) * 1e3
    jac = sp.csr_matrix(dense.astype(np.float32).astype(np.float64))
    jac.eliminate_zeros()
    jac.sort_indices()
    J = CSROperator(
        xp.asarray(np.asarray(jac.indptr, dtype=np.int64)),
        xp.asarray(np.asarray(jac.indices, dtype=np.int64)),
        xp.asarray(jac.data),
        (m, n),
        values_dtype_hint="float32",
    )
    op = _CondensedOperator(
        Diagonal(xp.asarray(rng.uniform(1.0, 2.0, size=n))),
        Diagonal(xp.asarray(rng.uniform(0.5, 1.5, size=n))),
        Diagonal(xp.asarray(rng.uniform(1e-3, 1.0, size=m))),
        J,
        0.0,
    )
    rhs = xp.asarray(rng.standard_normal(n))

    native = DenseSolver(DenseOptions(gram_dtype="native"))
    native.factor(op)
    x_native = np.asarray(native.solve(rhs))

    solver = DenseSolver(DenseOptions())  # auto
    solver.factor(op)
    x_auto = np.asarray(solver.solve(rhs))

    np.testing.assert_allclose(x_auto, x_native, rtol=1e-6, atol=1e-9)
    assert "float32" in solver.describe()
    # The materialized matrix really came from the reduced accumulate.
    exact = np.asarray(op.dense_matrix(rhs))
    mixed = np.asarray(op.dense_matrix_mixed(rhs, "float32"))
    rel = np.max(np.abs(mixed - exact)) / np.max(np.abs(exact))
    assert 1e-12 < rel < 1e-3


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

        def gram(self, weights, *, accumulate_dtype=None, hinted_only=False):
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
