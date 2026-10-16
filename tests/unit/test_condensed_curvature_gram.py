"""The optional fused curvature + inequality-Gram hook of the condensed block.

``LinearOperator.dense_matrix_with_gram(jac, weights)`` lets the Hessian
operator ``W`` form ``W + jacᵀ diag(weights) jac`` itself. A curvature that is
a weighted Gram ``Bᵀ diag(h) B`` over rows it shares with the inequality
Jacobian can then be multiplied once with the summed weights — Breedveld et
al. 2017, eq. (48), applied across ``W`` and ``∇gᵀ Σ_s ∇g`` — instead of in
two passes. The condensed operator consults the hook only when ``W`` offers
it; every other ``W`` keeps the two-pass assembly bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from ipax.backend.namespace import array_namespace
from ipax.backend.operators import (
    Dense,
    Diagonal,
    LinearOperator,
    forward_gram,
)
from ipax.ipm.kkt import build_condensed_operator, build_saddle_operator
from ipax.linalg.regularize import RegularizationState
from tests._helpers import assert_allclose, to_numpy, transpose
from tests.contracts.test_operator_contract import LinearOperatorContract


class _GramJacobian(Dense):
    """A dense Jacobian with a ``gram`` (the condensed operator's fast path)."""

    def __init__(self, A):
        super().__init__(A)
        self.gram_calls = 0

    def gram(self, weights, *, accumulate_dtype=None, hinted_only=False):
        del accumulate_dtype, hinted_only
        self.gram_calls += 1
        xp = array_namespace(self._A)
        return xp.matmul(transpose(xp, self._A), weights[:, None] * self._A)


class _SharedRowCurvature(Dense):
    """``W = Bᵀ diag(h) B + R`` that fuses the Gram of ``J = [B; C]``.

    The fused form multiplies ``B`` once with ``h + w_B`` (eq. (48) across
    the two terms) and adds ``Cᵀ diag(w_C) C`` — a different summation from
    ``W + Jᵀ diag(w) J``, equal to round-off. ``decline`` makes every call
    refuse (``NotImplementedError``), which is always a valid answer.
    """

    def __init__(self, B, C, h, R, *, decline=False):
        xp = array_namespace(B)
        super().__init__(xp.matmul(transpose(xp, B), h[:, None] * B) + R)
        self.B, self.C, self.h, self.R = B, C, h, R
        self.decline = decline
        self.partner = None  # the Jacobian [B; C] whose layout it knows
        self.calls = []
        self.memo = None

    def dense_matrix_with_gram(
        self, jac, weights, *, accumulate_dtype=None, hinted_only=False
    ):
        self.calls.append((jac, weights, accumulate_dtype, hinted_only))
        if self.decline or jac is not self.partner:
            raise NotImplementedError("declined")
        xp = array_namespace(self.B)
        p = int(self.B.shape[0])
        wb = self.h + weights[:p]
        wc = weights[p:]
        fused = (
            xp.matmul(transpose(xp, self.B), wb[:, None] * self.B)
            + xp.matmul(transpose(xp, self.C), wc[:, None] * self.C)
            + self.R
        )
        self.memo = fused  # returned by reference: the core must not mutate it
        return fused


def _blocks(xp, n=6, p=5, q=4, seed=3):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((p, n))
    C = rng.standard_normal((q, n))
    h = rng.uniform(-0.5, 2.0, p)  # signed, like DVH curvature
    r = rng.standard_normal((n, n))
    R = r @ r.T + n * np.eye(n)
    sigma_x = rng.uniform(0.1, 1.0, n)
    sigma_s = rng.uniform(0.1, 3.0, p + q)
    return tuple(xp.asarray(a) for a in (B, C, h, R, sigma_x, sigma_s))


def _pair(xp, *, delta_w=0.0, dense_sigma_x=False, decline=False):
    """The fused-capable operator and the plain two-pass one, same data."""
    B, C, h, R, sx, ss = _blocks(xp)
    fused_w = _SharedRowCurvature(B, C, h, R, decline=decline)
    plain_w = Dense(fused_w.dense_matrix())
    jac = _GramJacobian(xp.concat((B, C), axis=0))
    fused_w.partner = jac
    sigma_x = (
        Dense(xp.asarray(np.diag(to_numpy(sx)))) if dense_sigma_x else Diagonal(sx)
    )
    sigma_s = Diagonal(ss)
    reg = RegularizationState(delta_w=delta_w)
    fused = build_condensed_operator(fused_w, sigma_x, sigma_s, jac, reg)
    plain = build_condensed_operator(plain_w, sigma_x, sigma_s, jac, reg)
    return fused, plain, fused_w, jac, ss


@pytest.mark.parametrize("dense_sigma_x", [False, True])
@pytest.mark.parametrize("delta_w", [0.0, 1e-4])
def test_fused_block_matches_the_two_pass_block(namespace, delta_w, dense_sigma_x):
    xp = namespace
    fused, plain, fused_w, jac, ss = _pair(
        xp, delta_w=delta_w, dense_sigma_x=dense_sigma_x
    )
    actual = fused.dense_matrix()
    expected = plain.dense_matrix()
    assert_allclose(xp, actual, expected, rtol=1e-12, atol=1e-12)
    # Consulted once, with the solver's own Jacobian and the Σ_s weights.
    assert len(fused_w.calls) == 1
    got_jac, got_w, acc, hinted = fused_w.calls[0]
    assert got_jac is jac
    assert bool(xp.all(got_w == ss))
    assert (acc, hinted) == (None, False)
    # The fused path never asked the Jacobian for its Gram (the hook did the
    # whole product); the two-pass one did.
    assert jac.gram_calls == 1


def test_fused_block_is_the_matvec_operator(namespace):
    # The dense block and the operator's (exact) matvec describe one matrix:
    # what DenseSolver's refinement relies on.
    xp = namespace
    fused, _, _, _, _ = _pair(xp, delta_w=1e-3)
    n = fused.shape[0]
    probe = fused.matmat(xp.eye(n, dtype=xp.float64))
    assert_allclose(xp, fused.dense_matrix(), probe, rtol=1e-12, atol=1e-12)


def test_declining_hook_falls_back_bit_identically(namespace):
    xp = namespace
    fused, plain, fused_w, _, _ = _pair(xp, delta_w=1e-4, decline=True)
    actual = fused.dense_matrix()
    assert len(fused_w.calls) == 1
    assert bool(xp.all(actual == plain.dense_matrix()))


def test_operators_without_the_hook_decline_by_default(namespace):
    xp = namespace
    B, C, h, R, _, ss = _blocks(xp)
    jac = _GramJacobian(xp.concat((B, C), axis=0))
    for op in (Dense(R), Diagonal(h)):
        with pytest.raises(NotImplementedError):
            op.dense_matrix_with_gram(jac, ss)
    assert LinearOperator.dense_matrix_with_gram is Dense.dense_matrix_with_gram


def test_hook_not_consulted_without_inequality_rows(namespace):
    xp = namespace
    B, C, h, R, sx, _ = _blocks(xp)
    fused_w = _SharedRowCurvature(B, C, h, R)
    empty = Dense(xp.zeros((0, int(R.shape[0])), dtype=R.dtype))
    op = build_condensed_operator(
        fused_w,
        Diagonal(sx),
        Diagonal(xp.zeros((0,), dtype=R.dtype)),
        empty,
        RegularizationState(delta_w=0.0),
    )
    op.dense_matrix()
    assert fused_w.calls == []


def test_hook_not_consulted_with_a_non_diagonal_sigma_s(namespace):
    # A non-diagonal Σ_s has no weight vector to hand over: the two-pass
    # (densified) assembly runs, exactly as before.
    xp = namespace
    B, C, h, R, sx, ss = _blocks(xp)
    fused_w = _SharedRowCurvature(B, C, h, R)
    jac = _GramJacobian(xp.concat((B, C), axis=0))
    fused_w.partner = jac
    sigma_s = Dense(xp.asarray(np.diag(to_numpy(ss))))
    op = build_condensed_operator(
        fused_w, Diagonal(sx), sigma_s, jac, RegularizationState(delta_w=0.0)
    )
    plain = build_condensed_operator(
        Dense(fused_w.dense_matrix()),
        Diagonal(sx),
        sigma_s,
        jac,
        RegularizationState(delta_w=0.0),
    )
    assert bool(xp.all(op.dense_matrix() == plain.dense_matrix()))
    assert fused_w.calls == []


@pytest.mark.parametrize("hinted_only", [False, True])
def test_mixed_precision_request_is_forwarded_to_the_hook(namespace, hinted_only):
    xp = namespace
    fused, _, fused_w, _, _ = _pair(xp)
    like = xp.zeros((fused.shape[0],), dtype=xp.float64)
    fused.dense_matrix_mixed(like, "float32", hinted_only=hinted_only)
    assert [(c[2], c[3]) for c in fused_w.calls] == [("float32", hinted_only)]


def test_regularization_shifts_the_fused_block_by_delta_w(namespace):
    # The δ_w ladder rebuilds the condensed operator around the same W and
    # Jacobian; δ_w (and Σ_x) are added after the fused block is formed.
    xp = namespace
    base, _, _, _, _ = _pair(xp, delta_w=0.0)
    for delta_w in (1e-6, 1e-2, 1e2):
        shifted, _, _, _, _ = _pair(xp, delta_w=delta_w)
        n = base.shape[0]
        diff = shifted.dense_matrix() - base.dense_matrix()
        assert_allclose(
            xp,
            diff,
            delta_w * xp.eye(n, dtype=xp.float64),
            rtol=1e-12,
            atol=1e-12 * max(1.0, delta_w),
        )


def test_hook_result_is_treated_read_only(namespace):
    # The hook may return a memo by reference (the TROTS loader does): the
    # Σ_x and δ_w terms must be added out of place.
    xp = namespace
    fused, _, fused_w, _, _ = _pair(xp, delta_w=1e-3, dense_sigma_x=True)
    first = fused.dense_matrix()
    memo = fused_w.memo
    before = xp.asarray(memo, copy=True)
    second = fused.dense_matrix()
    assert first is not memo and second is not memo
    assert bool(xp.all(memo == before))


def test_saddle_dense_matrix_goes_through_the_fused_block(namespace):
    # Equality saddles border the condensed block: the hook serves them too.
    xp = namespace
    fused, plain, fused_w, _, _ = _pair(xp, delta_w=1e-4)
    rng = np.random.default_rng(11)
    eq = Dense(xp.asarray(rng.standard_normal((2, fused.shape[0]))))
    s_fused = build_saddle_operator(fused, eq, 1e-8)
    s_plain = build_saddle_operator(plain, eq, 1e-8)
    assert_allclose(
        xp, s_fused.dense_matrix(), s_plain.dense_matrix(), rtol=1e-12, atol=1e-12
    )
    assert len(fused_w.calls) == 1


# --------------------------------------------------------------------------- #
# End to end: lowering, scaling and the δ_w ladder in between
# --------------------------------------------------------------------------- #
class _GenericFusedHessian(Dense):
    """A constant Hessian whose hook goes through the solver's Jacobian chain.

    ``jac`` arrives as the solver sees it — the lowered, row-scaled, stacked
    inequality block — so an operator that cannot read that layout itself
    delegates the Gram to ``jac`` (whose wrappers translate the weights) and
    adds its own part. The reference fusion every implementation must match.
    """

    def __init__(self, A, log):
        super().__init__(A)
        self.log = log

    def dense_matrix_with_gram(
        self, jac, weights, *, accumulate_dtype=None, hinted_only=False
    ):
        self.log.append(type(jac).__name__)
        try:
            gram = forward_gram(
                jac, weights, accumulate_dtype=accumulate_dtype, hinted_only=hinted_only
            )
        except NotImplementedError:  # no Gram of its own: densify (tests only)
            xp = array_namespace(weights)
            J = jac.dense_matrix(weights)
            gram = xp.matmul(transpose(xp, J), weights[:, None] * J)
        return self.dense_matrix() + gram


class TestGenericFusedHessianContract(LinearOperatorContract):
    """The operator battery over a fusing implementation (any Jacobian)."""

    def make_operator(self, namespace):
        return _GenericFusedHessian(self.make_dense(namespace), [])

    def make_dense(self, namespace):
        return namespace.asarray(
            [[4.0, 1.0, 0.5], [1.0, 3.0, -0.25], [0.5, -0.25, 2.0]],
            dtype=namespace.float64,
        )


def _qp(xp, *, fused, log):
    import ipax

    rng = np.random.default_rng(21)
    n, m = 8, 30
    r = rng.standard_normal((n, n))
    Q = xp.asarray(r @ r.T + np.eye(n))
    c = xp.asarray(rng.standard_normal(n))
    A = xp.asarray(rng.standard_normal((m, n)))
    lo = xp.asarray(np.where(np.arange(m) % 3 == 0, -1.0, -np.inf))
    hi = xp.asarray(np.where(np.arange(m) % 3 == 0, np.inf, 1.0))

    class _QP(ipax.Problem):
        has_analytic_hessian = True

        @property
        def n_vars(self):
            return n

        def objective(self, x):
            return 0.5 * xp.sum(x * xp.matmul(Q, x)) + xp.sum(c * x)

        def gradient(self, x):
            return xp.matmul(Q, x) + c

        def linear_ineq(self):
            return (_GramJacobian(A), lo, hi)

        def lagrangian_hessian(self, x, y_eq, y_ineq, sigma=1.0):
            del x, y_eq, y_ineq
            H = sigma * Q
            return _GenericFusedHessian(H, log) if fused else Dense(H)

    return _QP()


def test_solve_through_the_hook_matches_the_two_pass_solve(namespace):
    from ipax import Options, solve

    xp = namespace
    opts = Options(hessian="exact", linsolve="dense", max_iter=100)
    x0 = xp.zeros((8,), dtype=xp.float64)
    log: list[str] = []
    r_fused = solve(_qp(xp, fused=True, log=log), x0, options=opts)
    r_plain = solve(_qp(xp, fused=False, log=[]), x0, options=opts)
    assert log, "the dense route never consulted the hook"
    assert r_fused.status == r_plain.status
    assert abs(r_fused.n_iter - r_plain.n_iter) <= 2  # round-off level only
    assert abs(r_fused.objective - r_plain.objective) <= 1e-10 * max(
        1.0, abs(r_plain.objective)
    )
