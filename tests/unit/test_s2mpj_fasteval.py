"""Unit tests for the precompiled S2MPJ evaluator (no checkout required).

``FastS2MPJEval`` re-implements ``s2mpjlib``'s interpretive ``evalgrsum`` loop
(per-element ``eval()`` string dispatch + ``lil_matrix`` row assembly) as a
precompiled structure walk. These tests drive it with a hand-computable fake
CUTEst-like instance covering the structural features the corpus exercises:
weighted and unweighted elements, repeated elemental variables, non-TRIVIAL and
TRIVIAL group functions, group scaling, linear terms, and group constants —
plus the verify-at-build fallback that keeps a semantic mismatch from ever
corrupting benchmark scores.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import scipy.sparse as sp

import ipax.testing.backends as backends
from benchmarks.corpus._s2mpj_fast import FastS2MPJEval, fast_evaluator
from benchmarks.corpus.s2mpj import _S2MPJProblem

# -- reference semantics (naive dense re-implementation of evalgrsum) ---------


def _ref_group_value(inst, ig: int, x: np.ndarray, want_grad: bool):
    """Naive dense evaluation of one S2MPJ group: value (and gradient)."""
    n = len(x)
    fin = 0.0
    gin = np.zeros(n)
    gconst = getattr(inst, "gconst", None)
    if gconst is not None and ig < len(gconst) and gconst[ig] is not None:
        fin = -float(gconst[ig])
    A = getattr(inst, "A", None)
    if A is not None and ig < A.shape[0]:
        row = np.asarray(A.tocsr()[ig, :].todense()).ravel()
        fin += float(row @ x[: len(row)])
        gin[: len(row)] += row
    grelt = getattr(inst, "grelt", None)
    if grelt is not None and ig < len(grelt) and grelt[ig] is not None:
        grelw = getattr(inst, "grelw", None)
        weights = grelw[ig] if grelw is not None and ig < len(grelw) else None
        for pos, iel in enumerate(grelt[ig]):
            fn = getattr(inst, inst.elftype[iel])
            idx = np.asarray(list(inst.elvar[iel]), dtype=int)
            w = 1.0 if weights is None else float(weights[pos])
            if want_grad:
                fiel, giel = fn(inst, 2, x[idx].reshape(-1, 1), iel)
                np.add.at(gin, idx, w * np.asarray(giel, dtype=float).ravel())
            else:
                fiel = fn(inst, 1, x[idx].reshape(-1, 1), iel)
            fin += w * float(fiel)
    gname = None
    grftype = getattr(inst, "grftype", None)
    if grftype is not None and ig < len(grftype) and grftype[ig] is not None:
        gname = grftype[ig]
    gsc = 1.0
    gscale = getattr(inst, "gscale", None)
    if gscale is not None and ig < len(gscale) and gscale[ig] is not None:
        v = float(gscale[ig])
        if abs(v) > 1e-15:
            gsc = v
    if gname is not None and gname != "TRIVIAL":
        gfn = getattr(inst, gname)
        if want_grad:
            fin, grada = gfn(inst, 2, fin, ig)
            gin = grada * gin
        else:
            fin = gfn(inst, 1, fin, ig)
    if want_grad:
        return fin / gsc, gin / gsc
    return fin / gsc


def _ref_fx(inst, x):
    return sum(_ref_group_value(inst, int(ig), x, False) for ig in inst.objgrps)


def _ref_fgx(inst, x):
    f, g = 0.0, np.zeros(len(x))
    for ig in inst.objgrps:
        fi, gi = _ref_group_value(inst, int(ig), x, True)
        f += fi
        g += gi
    return f, g.reshape(-1, 1)


def _ref_cx(inst, x):
    x = np.asarray(x, dtype=float).ravel()
    return np.asarray(
        [_ref_group_value(inst, int(ig), x, False) for ig in inst.congrps]
    ).reshape(-1, 1)


def _ref_cJx(inst, x):
    x = np.asarray(x, dtype=float).ravel()
    vals, rows = [], []
    for ig in inst.congrps:
        fi, gi = _ref_group_value(inst, int(ig), x, True)
        vals.append(fi)
        rows.append(gi)
    return np.asarray(vals).reshape(-1, 1), sp.lil_matrix(np.asarray(rows))


# -- the fake instance ---------------------------------------------------------


def _ePROD(self, nargout, x, iel):
    """Element f = u*v over the two elemental variables."""
    u, v = float(x[0, 0]), float(x[1, 0])
    if nargout == 1:
        return u * v
    return u * v, np.array([[v], [u]])


def _gSQR(self, nargout, fin, ig):
    """Group function f(a) = a**2."""
    if nargout == 1:
        return fin**2
    return fin**2, 2.0 * fin


class _StructuredInstance:
    """Hand-computable fake with the structural features the corpus uses.

    Group 0 (objective): weighted ePROD element over vars (0, 2), gSQR group
    function, gscale 2. Group 1 (constraint): unweighted ePROD over the
    *repeated* var (1, 1) — exercises scatter-add — TRIVIAL group. Group 2
    (constraint): pure linear, gscale 4. ``grelw`` is deliberately shorter
    than the group list (S2MPJ pads it lazily).
    """

    n = 3
    m = 2
    name = "FAKE"
    objgrps = np.array([0])
    congrps = np.array([1, 2])
    x0 = np.array([[1.0], [2.0], [3.0]])
    xlower = np.full((3, 1), -np.inf)
    xupper = np.full((3, 1), np.inf)
    clower = np.array([[0.0], [-np.inf]])
    cupper = np.array([[0.0], [2.0]])
    elftype: ClassVar = ["ePROD", "ePROD"]
    elvar: ClassVar = [np.array([0, 2]), np.array([1, 1])]
    grelt: ClassVar = [np.array([0]), np.array([1]), None]
    grelw: ClassVar = [[2.0], None]
    grftype: ClassVar = ["gSQR", None, "TRIVIAL"]
    gconst = np.array([0.5, 1.0, 0.0])
    gscale = np.array([2.0, None, 4.0], dtype=object)

    ePROD = staticmethod(_ePROD)
    gSQR = staticmethod(_gSQR)

    def __init__(self):
        self.A = sp.lil_matrix(
            np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [3.0, 0.0, 0.0]])
        )
        self.cx_calls = 0
        self.cJx_calls = 0

    def getglobs(self):
        pass

    # original-method stand-ins (the naive reference implementation)
    def fx(self, x):
        return _ref_fx(self, np.asarray(x, dtype=float).ravel())

    def fgx(self, x):
        return _ref_fgx(self, np.asarray(x, dtype=float).ravel())

    def cx(self, x):
        self.cx_calls += 1
        return _ref_cx(self, x)

    def cJx(self, x):
        self.cJx_calls += 1
        return _ref_cJx(self, x)


_X = np.array([1.0, 2.0, 3.0])
# Hand-computed at x = (1, 2, 3) — see class docstring for the structure:
#   group 0: fin = (5 - 0.5) + 2*(1*3) = 10.5; gSQR, gscale 2 → f = 10.5²/2
#   group 1: fin = (5 - 1) + 2*2 = 8;   TRIVIAL          → c0 = 8
#   group 2: fin = 3;                    TRIVIAL, gscale 4 → c1 = 0.75
_F_EXPECTED = 55.125
_G_EXPECTED = np.array([73.5, 21.0, 21.0])
_C_EXPECTED = np.array([8.0, 0.75])
_J_EXPECTED = np.array([[0.0, 5.0, 1.0], [0.75, 0.0, 0.0]])


def test_fast_eval_matches_hand_computed_values():
    fast = FastS2MPJEval(_StructuredInstance())

    assert fast.fx(_X) == pytest.approx(_F_EXPECTED, rel=1e-14)
    f, g = fast.fgx(_X)
    assert f == pytest.approx(_F_EXPECTED, rel=1e-14)
    np.testing.assert_allclose(g.ravel(), _G_EXPECTED, rtol=1e-14)
    np.testing.assert_allclose(fast.cx(_X).ravel(), _C_EXPECTED, rtol=1e-14)
    c, J = fast.cJx(_X)
    np.testing.assert_allclose(c.ravel(), _C_EXPECTED, rtol=1e-14)
    np.testing.assert_allclose(J.toarray(), _J_EXPECTED, rtol=1e-14)


def test_fast_eval_matches_reference_at_random_points():
    inst = _StructuredInstance()
    fast = FastS2MPJEval(inst)
    rng = np.random.default_rng(7)
    for _ in range(5):
        x = rng.standard_normal(3)
        f_ref, g_ref = _ref_fgx(inst, x)
        f, g = fast.fgx(x)
        assert f == pytest.approx(f_ref, rel=1e-12)
        np.testing.assert_allclose(g.ravel(), g_ref.ravel(), rtol=1e-12)
        c_ref, J_ref = _ref_cJx(inst, x)
        c, J = fast.cJx(x)
        np.testing.assert_allclose(c.ravel(), c_ref.ravel(), rtol=1e-12)
        np.testing.assert_allclose(J.toarray(), J_ref.toarray(), rtol=1e-12)


class _QuadraticOnlyInstance:
    """Objective is only the explicit quadratic term ``H`` (no objective groups)."""

    n = 2
    m = 0
    name = "QUAD"
    objgrps = np.array([], dtype=int)
    congrps = np.array([], dtype=int)
    x0 = np.array([[1.0], [2.0]])

    def __init__(self):
        self.H = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 4.0]]))

    def getglobs(self):
        pass


def test_fast_eval_handles_explicit_quadratic_objective():
    fast = FastS2MPJEval(_QuadraticOnlyInstance())
    x = np.array([1.0, 2.0])
    # 0.5 xᵀHx = 0.5(2·1 + 4·4) = 9
    assert fast.fx(x) == pytest.approx(9.0, rel=1e-14)
    f, g = fast.fgx(x)
    assert f == pytest.approx(9.0, rel=1e-14)
    np.testing.assert_allclose(g.ravel(), [2.0, 8.0], rtol=1e-14)


def test_fast_eval_rejects_partial_derivative_levels():
    inst = _StructuredInstance()
    inst.conderlvl = [1]  # constraints supply first derivatives only
    with pytest.raises(NotImplementedError):
        FastS2MPJEval(inst)


# -- verify-at-build integration with the bridge -------------------------------


def test_fast_evaluator_verifies_and_is_cached_on_the_instance():
    inst = _StructuredInstance()
    fast = fast_evaluator(inst)
    assert fast is not None
    # Cached: a second problem build (same shared lru instance) skips re-verify.
    assert fast_evaluator(inst) is fast


def test_fast_evaluator_falls_back_on_mismatch():
    inst = _StructuredInstance()

    def lying_cx(x):  # original disagrees with the declared structure
        return _ref_cx(inst, x) + 1.0

    inst.cx = lying_cx
    assert fast_evaluator(inst) is None


def test_fast_evaluator_falls_back_when_unsupported():
    inst = _StructuredInstance()
    inst.conderlvl = [1]
    assert fast_evaluator(inst) is None


def test_bridge_uses_fast_eval_without_touching_original_methods():
    xp = backends.import_namespace("numpy")
    inst = _StructuredInstance()
    problem = _S2MPJProblem(inst, xp)
    inst.cx_calls = inst.cJx_calls = 0  # discard the verification calls

    x = xp.asarray(_X)
    c_eq = problem.eq_constraints(x)
    c_ineq = problem.ineq_constraints(x)
    _jac = problem.ineq_jacobian(x)
    assert inst.cx_calls == 0 and inst.cJx_calls == 0

    # Values still correct: group 1 is the equality (clower == cupper == 0),
    # group 2 the upper side c ≤ 2.
    np.testing.assert_allclose(np.asarray(c_eq), [8.0], rtol=1e-14)
    np.testing.assert_allclose(np.asarray(c_ineq), [0.75 - 2.0], rtol=1e-14)


def test_bridge_falls_back_to_original_methods_on_mismatch():
    xp = backends.import_namespace("numpy")
    inst = _StructuredInstance()
    true_cx = inst.cx

    def lying_cx(x):  # structure-derived fast path must NOT win here
        return true_cx(x) + 1.0

    inst.cx = lying_cx
    problem = _S2MPJProblem(inst, xp)
    x = xp.asarray(_X)
    # The bridge must serve the (authoritative) original methods.
    np.testing.assert_allclose(np.asarray(problem.eq_constraints(x)), [9.0])


def test_cjx_seeds_the_value_memo():
    # cJx returns c(x) alongside the Jacobian; a same-point value request must
    # be served from the memo instead of a second full constraint evaluation.
    xp = backends.import_namespace("numpy")
    inst = _StructuredInstance()
    inst.conderlvl = [1]  # force the original-method path to make counting real
    problem = _S2MPJProblem(inst, xp)
    inst.cx_calls = inst.cJx_calls = 0

    x = xp.asarray(_X)
    problem.eq_jacobian(x)
    problem.ineq_jacobian(x)
    problem.eq_constraints(x)
    problem.ineq_constraints(x)
    assert inst.cJx_calls == 1  # memoized across the eq/ineq split
    assert inst.cx_calls == 0  # served from the cJx-seeded memo
