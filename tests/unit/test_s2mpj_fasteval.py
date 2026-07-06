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
from benchmarks.corpus.s2mpj import _S2MPJExactProblem, _S2MPJProblem

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


def _ref_group_full(inst, ig: int, x: np.ndarray):
    """Naive dense evaluation of one group: value, gradient, and Hessian.

    Mirrors ``evalgrsum``'s ``nargout=3`` semantics: the group Hessian is
    ``(Hessa·gin·ginᵀ + grada·Hin)/gsc`` for a non-TRIVIAL group function and
    ``Hin/gsc`` for TRIVIAL, with element Hessians scatter-added scalar-by-scalar
    (so repeated elemental variables accumulate).
    """
    n = len(x)
    fin = 0.0
    gin = np.zeros(n)
    Hin = np.zeros((n, n))
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
            fiel, giel, Hiel = fn(inst, 3, x[idx].reshape(-1, 1), iel)
            fin += w * float(fiel)
            giel = np.asarray(giel, dtype=float).ravel()
            Hiel = np.asarray(Hiel, dtype=float)
            for ir, ii in enumerate(idx):
                gin[ii] += w * giel[ir]
                for jr, jj in enumerate(idx):
                    Hin[ii, jj] += w * Hiel[ir, jr]
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
        fa, grada, Hessa = gfn(inst, 3, fin, ig)
        return (
            fa / gsc,
            grada * gin / gsc,
            (Hessa * np.outer(gin, gin) + grada * Hin) / gsc,
        )
    return fin / gsc, gin / gsc, Hin / gsc


def _ref_LgHxy(inst, x, y):
    """Naive dense Lagrangian value/gradient/Hessian: ``L = f + Σ y_i c_i``."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    L = 0.0
    g = np.zeros(n)
    H = np.zeros((n, n))
    Hq = getattr(inst, "H", None)
    if Hq is not None:
        Hxv = np.asarray(Hq @ x).ravel()
        L += 0.5 * float(x @ Hxv)
        g += Hxv
        H += np.asarray(Hq.todense())
    for ig in inst.objgrps:
        v, gr, Hg = _ref_group_full(inst, int(ig), x)
        L, g, H = L + v, g + gr, H + Hg
    for k, ig in enumerate(inst.congrps):
        v, gr, Hg = _ref_group_full(inst, int(ig), x)
        L, g, H = L + y[k] * v, g + y[k] * gr, H + y[k] * Hg
    return L, g, H


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
    if nargout == 2:
        return u * v, np.array([[v], [u]])
    return u * v, np.array([[v], [u]]), np.array([[0.0, 1.0], [1.0, 0.0]])


def _gSQR(self, nargout, fin, ig):
    """Group function f(a) = a**2."""
    if nargout == 1:
        return fin**2
    if nargout == 2:
        return fin**2, 2.0 * fin
    return fin**2, 2.0 * fin, 2.0


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
        self.LgHxy_calls = 0

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

    def LgHxy(self, x, y):
        self.LgHxy_calls += 1
        L, g, H = _ref_LgHxy(self, x, y)
        return L, g.reshape(-1, 1), sp.lil_matrix(H)


_X = np.array([1.0, 2.0, 3.0])
# Hand-computed at x = (1, 2, 3) — see class docstring for the structure:
#   group 0: fin = (5 - 0.5) + 2*(1*3) = 10.5; gSQR, gscale 2 → f = 10.5²/2
#   group 1: fin = (5 - 1) + 2*2 = 8;   TRIVIAL          → c0 = 8
#   group 2: fin = 3;                    TRIVIAL, gscale 4 → c1 = 0.75
_F_EXPECTED = 55.125
_G_EXPECTED = np.array([73.5, 21.0, 21.0])
_C_EXPECTED = np.array([8.0, 0.75])
_J_EXPECTED = np.array([[0.0, 5.0, 1.0], [0.75, 0.0, 0.0]])
# Lagrangian Hessian at x = (1, 2, 3) with Y = (1, 1) — per-group Hessians:
#   group 0: gin = [7, 2, 2], Hin[0,2] = Hin[2,0] = 2 (weighted ePROD), gSQR with
#     grada = 21, Hessa = 2, gscale 2 → outer(gin, gin) + 10.5·Hin
#   group 1: unweighted ePROD on the repeated var (1, 1) → Hin[1,1] = 2, TRIVIAL
#   group 2: pure linear → zero Hessian
_HL_EXPECTED = np.array([[49.0, 14.0, 35.0], [14.0, 6.0, 4.0], [35.0, 4.0, 4.0]])
_L_EXPECTED = 55.125 + 8.0 + 0.75
_LG_EXPECTED = np.array([74.25, 26.0, 22.0])


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


def test_cjx_results_are_independent_across_calls():
    # The Jacobian's CSR structure is precompiled and shared across calls; the
    # *values* must not be — an earlier result must survive later evaluations.
    inst = _StructuredInstance()
    fast = FastS2MPJEval(inst)
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([-2.0, 0.5, 4.0])
    _c1, j1 = fast.cJx(x1)
    frozen = j1.toarray().copy()
    fast.cJx(x2)
    np.testing.assert_array_equal(j1.toarray(), frozen)


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


# -- per-elftype batched element evaluation ------------------------------------


def _eWSQ(self, nargout, *args):
    # Generator-style source: EV_[i,0] scalar indexing, np.zeros(dim) gradient
    # allocation, per-element parameters via self.elpar[iel_] — the patterns
    # the batch transformer must rewrite. f = p·u²·v.
    import numpy as np

    EV_ = args[0]
    iel_ = args[1]
    f_ = self.elpar[iel_][0] * EV_[0, 0] ** 2 * EV_[1, 0]
    if nargout > 1:
        try:
            dim = len(IV_)  # noqa: F821  (generated-code idiom)
        except Exception:
            dim = len(EV_)
        g_ = np.zeros(dim)
        g_[0] = 2.0 * self.elpar[iel_][0] * EV_[0, 0] * EV_[1, 0]
        g_[1] = self.elpar[iel_][0] * EV_[0, 0] ** 2
    if nargout == 1:
        return f_
    elif nargout == 2:
        return f_, g_


def _gPSQ(self, nargout, *args):
    # Generator-style group function: f = p·a² with p = self.grpar[igr_][0].
    GVAR_ = args[0]
    igr_ = args[1]
    f_ = self.grpar[igr_][0] * GVAR_ * GVAR_
    if nargout > 1:
        g_ = 2.0 * self.grpar[igr_][0] * GVAR_
    if nargout == 1:
        return f_
    elif nargout == 2:
        return f_, g_


def _gBR(self, nargout, *args):
    # Data-dependent branch: must stay on the per-row path.
    GVAR_ = args[0]
    igr_ = args[1]  # noqa: F841
    if GVAR_ > 0:
        f_, g_ = GVAR_**2, 2.0 * GVAR_
    else:
        f_, g_ = -(GVAR_**2), -2.0 * GVAR_
    if nargout == 1:
        return f_
    elif nargout == 2:
        return f_, g_


def _eBRANCH(self, nargout, *args):
    # Data-dependent branch: must be rejected by the batch transformer and
    # keep evaluating per element. f = |u|·u (branchy form).
    import numpy as np

    EV_ = args[0]
    iel_ = args[1]  # noqa: F841
    if EV_[0, 0] > 0:
        f_ = EV_[0, 0] ** 2
    else:
        f_ = -(EV_[0, 0] ** 2)
    if nargout > 1:
        g_ = np.zeros(1)
        g_[0] = 2.0 * abs(EV_[0, 0])
    if nargout == 1:
        return f_
    elif nargout == 2:
        return f_, g_


class _BatchInstance:
    """Generator-style fake: batchable eWSQ (weighted, unweighted, repeated
    elemental variables, per-element parameters) mixed with the unbatchable
    branchy eBRANCH; group functions cover TRIVIAL, batchable gPSQ (twice,
    with per-group grpar) and the branchy gBR; constraint rows cover an
    equality, a two-sided range, an upper side, and a lower side."""

    n = 4
    m = 4
    name = "BATCHFAKE"
    objgrps = np.array([0])
    congrps = np.array([1, 2, 3, 4])
    x0 = np.array([[1.0], [2.0], [3.0], [4.0]])
    clower = np.array([[0.0], [-1.0], [-np.inf], [2.0]])
    cupper = np.array([[0.0], [10.0], [5.0], [np.inf]])
    elftype: ClassVar = ["eWSQ", "eWSQ", "eWSQ", "eWSQ", "eBRANCH"]
    elvar: ClassVar = [
        np.array([0, 1]),
        np.array([2, 3]),
        np.array([2, 2]),  # repeated elemental variable: scatter must accumulate
        np.array([1, 3]),
        np.array([0]),
    ]
    elpar: ClassVar = [[2.0], [0.5], [1.5], [1.0], [7.0]]
    grelt: ClassVar = [np.array([0]), np.array([1, 2]), np.array([3, 4]), None, None]
    grelw: ClassVar = [[3.0], None, [2.0, 1.0]]
    grftype: ClassVar = ["gSQR", None, "gPSQ", "gPSQ", "gBR"]
    grpar: ClassVar = [None, None, [3.0], [0.5], None]
    gconst = np.array([0.2, 0.0, 0.0, 1.0, 0.0])
    gscale = np.array([2.0, None, 4.0, None, 2.0], dtype=object)

    eWSQ = staticmethod(_eWSQ)
    eBRANCH = staticmethod(_eBRANCH)
    gSQR = staticmethod(_gSQR)
    gPSQ = staticmethod(_gPSQ)
    gBR = staticmethod(_gBR)

    def __init__(self):
        self.A = sp.lil_matrix(
            np.array(
                [
                    [1.0, 0.0, 2.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [3.0, 0.0, 0.0, 0.5],
                    [0.0, 2.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 1.0],
                ]
            )
        )

    def getglobs(self):
        pass

    def fx(self, x):
        return _ref_fx(self, np.asarray(x, dtype=float).ravel())

    def fgx(self, x):
        return _ref_fgx(self, np.asarray(x, dtype=float).ravel())

    def cx(self, x):
        return _ref_cx(self, x)

    def cJx(self, x):
        return _ref_cJx(self, x)


def test_batchable_type_is_batched_and_branchy_type_is_not():
    fast = FastS2MPJEval(_BatchInstance())
    assert "eWSQ" in fast.batched_elftypes
    assert "eBRANCH" not in fast.batched_elftypes


def test_batched_evaluation_matches_reference():
    inst = _BatchInstance()
    fast = FastS2MPJEval(inst)
    rng = np.random.default_rng(23)
    points = [np.asarray(inst.x0, dtype=float).ravel()] + [
        rng.standard_normal(4) for _ in range(4)
    ]
    for x in points:
        f_ref, g_ref = _ref_fgx(inst, x)
        assert fast.fx(x) == pytest.approx(_ref_fx(inst, x), rel=1e-12)
        f, g = fast.fgx(x)
        assert f == pytest.approx(f_ref, rel=1e-12)
        np.testing.assert_allclose(g.ravel(), g_ref.ravel(), rtol=1e-12, atol=1e-14)
        c_ref, J_ref = _ref_cJx(inst, x)
        np.testing.assert_allclose(
            fast.cx(x).ravel(), c_ref.ravel(), rtol=1e-12, atol=1e-14
        )
        c, J = fast.cJx(x)
        np.testing.assert_allclose(c.ravel(), c_ref.ravel(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(J.toarray(), J_ref.toarray(), rtol=1e-12, atol=1e-14)


def test_batched_instance_passes_verification():
    fast = fast_evaluator(_BatchInstance())
    assert fast is not None
    assert "eWSQ" in fast.batched_elftypes


def test_nontrivial_group_functions_batch_by_gftype():
    fast = FastS2MPJEval(_BatchInstance())
    assert "gPSQ" in fast.batched_gftypes
    assert "gBR" not in fast.batched_gftypes


def test_bridge_jacobian_split_matches_reference():
    # eq/lo/up row selection (incl. the two-sided range appearing in both
    # ineq blocks) must reproduce the sign-lowered rows of the full Jacobian.
    xp = backends.import_namespace("numpy")
    inst = _BatchInstance()
    x = np.array([0.7, -1.3, 2.1, 0.4])
    _c, J = _ref_cJx(inst, x)
    J = np.asarray(J.todense())
    for sparse in (False, True):
        problem = _S2MPJProblem(_BatchInstance(), xp, sparse=sparse)
        xv = xp.asarray(x)
        eq = problem.eq_jacobian(xv)
        ineq = problem.ineq_jacobian(xv)
        if sparse:
            eq = eq.scipy_matrix.todense()
            ineq = ineq.scipy_matrix.todense()
        # Row 0 is the equality; lower sides (rows 1, 3) then upper (1, 2).
        np.testing.assert_allclose(np.asarray(eq), J[[0]], rtol=1e-12)
        expected = np.vstack((-J[[1, 3]], J[[1, 2]]))
        np.testing.assert_allclose(np.asarray(ineq), expected, rtol=1e-12)


def test_gradient_and_objective_share_one_fgx_evaluation():
    xp = backends.import_namespace("numpy")
    problem = _S2MPJProblem(_StructuredInstance(), xp)
    fgx_calls, fx_calls = [], []
    orig_fgx, orig_fx = problem._eval_fgx, problem._eval_fx
    problem._eval_fgx = lambda x_np: (fgx_calls.append(1), orig_fgx(x_np))[1]
    problem._eval_fx = lambda x_np: (fx_calls.append(1), orig_fx(x_np))[1]

    x = xp.asarray(_X)
    g = problem.gradient(x)
    f = problem.objective(x)  # same point: must reuse the memoized fgx value
    assert len(fgx_calls) == 1 and len(fx_calls) == 0
    np.testing.assert_allclose(np.asarray(g).ravel(), _G_EXPECTED, rtol=1e-13)
    assert float(np.asarray(f)) == pytest.approx(_F_EXPECTED, rel=1e-13)


def test_lying_batch_source_is_caught_by_type_verification():
    # A type whose vectorized evaluation would disagree with the per-element
    # original must be demoted to the per-element path, not batched. Simulate
    # by giving the class a batch-hostile numeric idiom: float() coercion.
    inst = _StructuredInstance()
    fast = FastS2MPJEval(inst)
    # _ePROD uses float(x[0,0]) → raises on a batch column, so it must not batch.
    assert "ePROD" not in fast.batched_elftypes
    # And the per-element path still serves correct values (hand-computed).
    np.testing.assert_allclose(fast.cx(_X).ravel(), _C_EXPECTED, rtol=1e-14)


# -- the precompiled Lagrangian Hessian ----------------------------------------


def test_fast_lghxy_matches_hand_computed_values():
    fast = FastS2MPJEval(_StructuredInstance())

    L, g, H = fast.LgHxy(_X, np.array([1.0, 1.0]))
    assert L == pytest.approx(_L_EXPECTED, rel=1e-14)
    np.testing.assert_allclose(g.ravel(), _LG_EXPECTED, rtol=1e-14)
    np.testing.assert_allclose(H.toarray(), _HL_EXPECTED, rtol=1e-14)


def test_fast_lghxy_matches_reference_at_random_points():
    inst = _StructuredInstance()
    fast = FastS2MPJEval(inst)
    rng = np.random.default_rng(11)
    for _ in range(5):
        x = rng.standard_normal(3)
        y = rng.standard_normal(2)
        L_ref, g_ref, H_ref = _ref_LgHxy(inst, x, y)
        L, g, H = fast.LgHxy(x, y)
        assert L == pytest.approx(L_ref, rel=1e-12)
        np.testing.assert_allclose(g.ravel(), g_ref.ravel(), rtol=1e-12)
        np.testing.assert_allclose(H.toarray(), H_ref, rtol=1e-12, atol=1e-14)


def test_fast_lghxy_handles_explicit_quadratic_objective():
    fast = FastS2MPJEval(_QuadraticOnlyInstance())
    x = np.array([1.0, 2.0])
    L, g, H = fast.LgHxy(x, np.zeros((0, 1)))
    assert L == pytest.approx(9.0, rel=1e-14)
    np.testing.assert_allclose(g.ravel(), [2.0, 8.0], rtol=1e-14)
    np.testing.assert_allclose(H.toarray(), [[2.0, 0.0], [0.0, 4.0]], rtol=1e-14)


def test_fast_evaluator_verifies_the_hessian():
    fast = fast_evaluator(_StructuredInstance())
    assert fast is not None
    assert fast.lghxy_ok is True


def test_hessian_mismatch_disables_only_the_hessian_path():
    inst = _StructuredInstance()
    true_lghxy = inst.LgHxy

    def lying_lghxy(x, y):  # original disagrees on the Hessian only
        L, g, H = true_lghxy(x, y)
        return L, g, sp.lil_matrix(H.toarray() + 1.0)

    inst.LgHxy = lying_lghxy
    fast = fast_evaluator(inst)
    # The base evaluator (fx/fgx/cx/cJx) survives; only the Hessian falls back.
    assert fast is not None
    assert fast.lghxy_ok is False


def test_exact_bridge_uses_fast_hessian_without_touching_original():
    xp = backends.import_namespace("numpy")
    inst = _StructuredInstance()
    problem = _S2MPJExactProblem(inst, xp)
    inst.LgHxy_calls = 0  # discard the verification calls

    x = xp.asarray(_X)
    hess = problem.lagrangian_hessian(x, xp.asarray([1.0]), xp.asarray([1.0]))
    assert inst.LgHxy_calls == 0
    # ipax multipliers map onto Y = (y_eq[0], y_ineq[0]) = (1, 1) — group 1 is
    # the equality, group 2 the (upper-side) inequality with a zero Hessian.
    np.testing.assert_allclose(np.asarray(hess), _HL_EXPECTED, rtol=1e-14)


def test_exact_bridge_honors_sigma_with_fast_hessian():
    xp = backends.import_namespace("numpy")
    inst = _StructuredInstance()
    problem = _S2MPJExactProblem(inst, xp)
    inst.LgHxy_calls = 0

    sigma = 0.5
    hess = problem.lagrangian_hessian(
        xp.asarray(_X), xp.asarray([1.0]), xp.asarray([1.0]), sigma
    )
    assert inst.LgHxy_calls == 0
    # σ·∇²f + Σ Y·∇²c with the objective part scaled: group 0 is the objective.
    _, _, h_obj = _ref_group_full(inst, 0, _X)
    _, _, h_eq = _ref_group_full(inst, 1, _X)
    np.testing.assert_allclose(np.asarray(hess), sigma * h_obj + h_eq, rtol=1e-13)


def test_exact_bridge_honors_nonpositive_sigma_with_fast_hessian():
    # σ ≤ 0 takes the general two-call form: LgHxy(Y) + (σ−1)·LgHxy(0).
    xp = backends.import_namespace("numpy")
    inst = _StructuredInstance()
    problem = _S2MPJExactProblem(inst, xp)
    inst.LgHxy_calls = 0

    hess = problem.lagrangian_hessian(
        xp.asarray(_X), xp.asarray([1.0]), xp.asarray([1.0]), 0.0
    )
    assert inst.LgHxy_calls == 0
    # σ = 0 zeroes the objective curvature: only the equality group remains.
    _, _, h_eq = _ref_group_full(inst, 1, _X)
    np.testing.assert_allclose(np.asarray(hess), h_eq, rtol=1e-13, atol=1e-14)


def test_exact_bridge_falls_back_to_original_hessian_on_mismatch():
    xp = backends.import_namespace("numpy")
    inst = _StructuredInstance()
    true_lghxy = inst.LgHxy

    def lying_lghxy(x, y):  # the (authoritative) original must win
        L, g, H = true_lghxy(x, y)
        return L, g, sp.lil_matrix(H.toarray() + 1.0)

    inst.LgHxy = lying_lghxy
    problem = _S2MPJExactProblem(inst, xp)
    hess = problem.lagrangian_hessian(
        xp.asarray(_X), xp.asarray([1.0]), xp.asarray([1.0])
    )
    np.testing.assert_allclose(np.asarray(hess), _HL_EXPECTED + 1.0, rtol=1e-14)


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
