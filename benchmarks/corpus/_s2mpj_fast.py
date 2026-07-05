"""Precompiled S2MPJ evaluator (benchmark-side, NumPy/SciPy allowed).

S2MPJ's generic ``CUTEst_problem.evalgrsum`` re-interprets the group-partially-
separable problem structure on **every** evaluation: each element and group
function is dispatched through ``eval('self.'+name+'(...)')`` (a string compile
per call), the Jacobian is assembled row-by-row into a ``lil_matrix`` (with each
group's dense gradient converted to ``lil`` just to write one row), and the
linear term is sliced out of the sparse ``A`` one group at a time. Profiling the
S2MPJ sweep shows ~90% of solve wall-time in that interpretive loop — tiny
problems like ACOPP14 (n=38) cost ~10 ms per constraint evaluation.

:class:`FastS2MPJEval` compiles the same structure **once per instance**: element
and group functions are resolved to callables via ``getattr``, all linear terms
become one vectorized ``A @ x − gconst`` product, and the Jacobian is built as
COO triplets on each group's precomputed sparsity support. The element/group
functions themselves — the actual math S2MPJ generated from the SIF file — are
untouched, so values match the original to floating-point roundoff (summation
order differs). Measured: 8–130× per ``cx`` call, 6–37× end-to-end solves.

:func:`fast_evaluator` is the safe entry point: it builds the fast evaluator and
**verifies it against the original methods** at the start point (and a perturbed
point) before use, returning ``None`` — meaning "use the original methods" — on
any unsupported feature, construction error, or numerical mismatch. A corpus
oddity can therefore never silently corrupt benchmark scores. The result is
cached on the instance, so the per-config problem rebuilds share one
verification.

Scope: values, gradients, and Jacobians (``fx/fgx/cx/cJx``, ``nargout ≤ 2``).
The Lagrangian-Hessian path (``LgHxy``) keeps the original implementation.
Problems that declare *partial* derivative levels (``objderlvl``/``conderlvl``
< 2, where the original substitutes NaNs) are rejected at construction and fall
back to the original methods.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

# Element entry: (element function, elemental variable indices, weight or None,
# element index). Group entry: (elements, group function or None for TRIVIAL).
_Element = tuple[Any, np.ndarray, float | None, int]
_Group = tuple[list[_Element], Any]

# Agreement gates for verify-at-build: a genuine reimplementation bug is orders
# of magnitude larger than the summation-order roundoff these absorb.
_VERIFY_RTOL = 1e-7
_VERIFY_ATOL = 1e-9

_FAST_ATTR = "_ipax_fast_eval"
_UNSET = object()


class FastS2MPJEval:
    """Precompiled ``fx/fgx/cx/cJx`` for one S2MPJ ``CUTEst_problem`` instance.

    Raises ``NotImplementedError`` (or any construction error) when the instance
    uses a feature outside the replicated ``evalgrsum`` semantics; callers should
    then use the original methods (see :func:`fast_evaluator`).
    """

    def __init__(self, instance: Any) -> None:
        self.inst = instance
        getglobs = getattr(instance, "getglobs", None)
        if getglobs is not None:  # set global element/group parameters once
            getglobs()
        n = int(instance.n)
        self.n = n

        # Partial derivative levels make evalgrsum emit NaN rows; keep those on
        # the original path rather than replicating the NaN bookkeeping.
        if int(getattr(instance, "objderlvl", 2)) < 2:
            raise NotImplementedError("partial objective derivatives")
        conderlvl = getattr(instance, "conderlvl", None)
        if conderlvl is not None and any(int(d) < 2 for d in conderlvl):
            raise NotImplementedError("partial constraint derivatives")

        self.objgrps = np.asarray(getattr(instance, "objgrps", ()), dtype=int).reshape(
            -1
        )
        self.congrps = np.asarray(getattr(instance, "congrps", ()), dtype=int).reshape(
            -1
        )
        n_groups = int(max([*self.objgrps, *self.congrps, -1])) + 1

        self.gconst = self._dense_per_group(
            getattr(instance, "gconst", None), n_groups, default=0.0
        )
        gscale = self._dense_per_group(
            getattr(instance, "gscale", None), n_groups, default=1.0
        )
        # evalgrsum treats |scale| ≤ 1e-15 as "no scaling"
        gscale[np.abs(gscale) <= 1.0e-15] = 1.0
        self.gscale = gscale

        # Linear term: one padded CSR so every group's row is A[ig].
        if hasattr(instance, "A"):
            A = sp.csr_matrix(instance.A, dtype=float)
            if A.shape[1] > n:  # evalgrsum requires sA2 ≤ n (gin[:sA2] scatter)
                raise NotImplementedError("linear term wider than n")
            A.resize((max(A.shape[0], n_groups), n))
        else:
            A = sp.csr_matrix((n_groups, n))
        self.A = A

        self.obj_groups = [self._compile_group(int(ig)) for ig in self.objgrps]
        self.con_groups = [self._compile_group(int(ig)) for ig in self.congrps]

        # Per-constraint-group Jacobian support: A-row indices ∪ element vars.
        self.con_support: list[np.ndarray] = []
        for k, ig in enumerate(self.congrps):
            cols = set(A.indices[A.indptr[ig] : A.indptr[ig + 1]].tolist())
            for _fn, idx, _w, _iel in self.con_groups[k][0]:
                cols.update(idx.tolist())
            self.con_support.append(
                np.fromiter(sorted(cols), dtype=int, count=len(cols))
            )

        self.H = getattr(instance, "H", None)
        self._has_objective = len(self.objgrps) > 0 or self.H is not None

    @staticmethod
    def _dense_per_group(values: Any, n_groups: int, *, default: float) -> np.ndarray:
        """Densify a lazily-padded per-group S2MPJ array (None entries → default)."""
        out = np.full(n_groups, default)
        if values is None:
            return out
        for ig in range(min(len(values), n_groups)):
            v = values[ig]
            if v is not None:
                out[ig] = float(np.asarray(v).reshape(-1)[0])
        return out

    def _compile_group(self, ig: int) -> _Group:
        """Resolve one group's element list and group function to callables."""
        inst = self.inst
        elements: list[_Element] = []
        grelt = getattr(inst, "grelt", None)
        if grelt is not None and ig < len(grelt) and grelt[ig] is not None:
            grelw = getattr(inst, "grelw", None)
            weights = (
                grelw[ig]
                if grelw is not None and ig < len(grelw) and grelw[ig] is not None
                else None
            )
            for pos, iel in enumerate(grelt[ig]):
                iel = int(iel)
                fn = getattr(inst, inst.elftype[iel])
                idx = np.asarray([int(iv) for iv in inst.elvar[iel]], dtype=int)
                w = None if weights is None else float(np.asarray(weights[pos]))
                elements.append((fn, idx, w, iel))
        gname = "TRIVIAL"
        grftype = getattr(inst, "grftype", None)
        if grftype is not None and ig < len(grftype) and grftype[ig] is not None:
            gname = grftype[ig]
        gfn = None if gname == "TRIVIAL" else getattr(inst, gname)
        return elements, gfn

    # -- constraints --------------------------------------------------------

    def cx(self, x: Any) -> np.ndarray:
        """Constraint values, shape ``(m, 1)`` (mirrors the original ``cx``)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        xc = x.reshape(-1, 1)
        fin = (self.A @ x)[self.congrps] - self.gconst[self.congrps]
        out = np.empty((len(self.congrps), 1))
        for k, (elements, gfn) in enumerate(self.con_groups):
            f = fin[k]
            for fn, idx, w, iel in elements:
                fiel = fn(self.inst, 1, xc[idx], iel)
                f = f + (w * fiel if w is not None else fiel)
            ig = int(self.congrps[k])
            if gfn is not None:
                f = gfn(self.inst, 1, f, ig)
            out[k, 0] = float(np.asarray(f).reshape(-1)[0]) / self.gscale[ig]
        return out

    def cJx(self, x: Any) -> tuple[np.ndarray, sp.csr_matrix]:
        """Constraint values and Jacobian ``(c, J)`` with ``J`` in CSR."""
        x = np.asarray(x, dtype=float).reshape(-1)
        xc = x.reshape(-1, 1)
        m = len(self.congrps)
        fin = (self.A @ x)[self.congrps] - self.gconst[self.congrps]
        A = self.A
        cvals = np.empty((m, 1))
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        for k, (elements, gfn) in enumerate(self.con_groups):
            ig = int(self.congrps[k])
            support = self.con_support[k]
            gin = np.zeros(len(support))
            a0, a1 = A.indptr[ig], A.indptr[ig + 1]
            if a1 > a0:
                gin[np.searchsorted(support, A.indices[a0:a1])] += A.data[a0:a1]
            f = fin[k]
            for fn, idx, w, iel in elements:
                fiel, giel = fn(self.inst, 2, xc[idx], iel)
                giel = np.asarray(giel, dtype=float).reshape(-1)
                pos = np.searchsorted(support, idx)
                if w is not None:
                    f = f + w * fiel
                    np.add.at(gin, pos, w * giel)
                else:
                    f = f + fiel
                    np.add.at(gin, pos, giel)
            if gfn is not None:
                f, grada = gfn(self.inst, 2, f, ig)
                gin = grada * gin
            gsc = self.gscale[ig]
            cvals[k, 0] = float(np.asarray(f).reshape(-1)[0]) / gsc
            if len(support):
                rows.append(np.full(len(support), k))
                cols.append(support)
                vals.append(np.asarray(gin, dtype=float).reshape(-1) / gsc)
        if rows:
            J = sp.csr_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                shape=(m, self.n),
            )
        else:
            J = sp.csr_matrix((m, self.n))
        return cvals, J

    # -- objective ----------------------------------------------------------

    def fx(self, x: Any) -> float:
        return self._obj(x, want_grad=False)  # type: ignore[return-value]

    def fgx(self, x: Any) -> tuple[float, np.ndarray]:
        return self._obj(x, want_grad=True)  # type: ignore[return-value]

    def _obj(self, x: Any, *, want_grad: bool) -> float | tuple[float, np.ndarray]:
        if not self._has_objective:
            raise ValueError("S2MPJ problem has no objective function")
        x = np.asarray(x, dtype=float).reshape(-1)
        xc = x.reshape(-1, 1)
        n = self.n
        fx = 0.0
        gx = np.zeros(n) if want_grad else None
        if self.H is not None:
            Hx = np.asarray(self.H @ x).reshape(-1)
            fx += 0.5 * float(x @ Hx)
            if want_grad:
                gx += Hx
        fin = (
            (self.A @ x)[self.objgrps] - self.gconst[self.objgrps]
            if len(self.objgrps)
            else np.zeros(0)
        )
        A = self.A
        for k, (elements, gfn) in enumerate(self.obj_groups):
            ig = int(self.objgrps[k])
            f = fin[k]
            if want_grad:
                gin = np.zeros(n)
                a0, a1 = A.indptr[ig], A.indptr[ig + 1]
                if a1 > a0:
                    gin[A.indices[a0:a1]] += A.data[a0:a1]
            for fn, idx, w, iel in elements:
                if want_grad:
                    fiel, giel = fn(self.inst, 2, xc[idx], iel)
                    giel = np.asarray(giel, dtype=float).reshape(-1)
                    if w is not None:
                        f = f + w * fiel
                        np.add.at(gin, idx, w * giel)
                    else:
                        f = f + fiel
                        np.add.at(gin, idx, giel)
                else:
                    fiel = fn(self.inst, 1, xc[idx], iel)
                    f = f + (w * fiel if w is not None else fiel)
            if gfn is not None:
                if want_grad:
                    f, grada = gfn(self.inst, 2, f, ig)
                    gin = grada * gin
                else:
                    f = gfn(self.inst, 1, f, ig)
            gsc = self.gscale[ig]
            fx += float(np.asarray(f).reshape(-1)[0]) / gsc
            if want_grad:
                gx += gin / gsc
        if want_grad:
            return fx, gx.reshape(-1, 1)
        return fx


# -- verify-at-build entry point ----------------------------------------------


def fast_evaluator(instance: Any) -> FastS2MPJEval | None:
    """The verified fast evaluator for ``instance``, or ``None`` to use the original.

    Builds :class:`FastS2MPJEval` and compares every replicated method against
    the instance's original one at the start point and a perturbed point. Any
    construction failure, evaluation mismatch, or inability to verify at least
    one method yields ``None``. The outcome is cached on the instance so the
    per-config problem rebuilds (which share the lru-cached instance) verify
    only once.
    """
    cached = getattr(instance, _FAST_ATTR, _UNSET)
    if cached is not _UNSET:
        return cached  # type: ignore[return-value]
    fast = _build_and_verify(instance)
    try:
        setattr(instance, _FAST_ATTR, fast)
    except Exception:  # exotic instance forbids attributes — just skip caching
        pass
    return fast


def _build_and_verify(instance: Any) -> FastS2MPJEval | None:
    try:
        fast = FastS2MPJEval(instance)
        x0 = np.asarray(instance.x0, dtype=float).reshape(-1)
    except Exception:
        return None
    # Two deterministic probe points: x0 itself and a relative perturbation
    # (x0 is often special — e.g. all zeros kills every linear term).
    probe = x0 + 1e-4 * (1.0 + np.abs(x0)) * np.cos(np.arange(x0.shape[0]))
    verified = 0
    for x in (x0, probe):
        agree = _agrees_at(instance, fast, x)
        if agree is False:
            return None
        verified += agree is True
    return fast if verified else None


def _agrees_at(instance: Any, fast: FastS2MPJEval, x: np.ndarray) -> bool | None:
    """Tri-state agreement at one point: True/False, or None when uncheckable.

    A raising *original* method skips that comparison (the original is the
    authority on exceptions, and the solve would fail there anyway); a raising
    or mismatching *fast* method where the original succeeded is a failure.
    """
    checked = 0
    with np.errstate(all="ignore"):
        if fast._has_objective:
            for ref_fn, fast_fn, is_pair in (
                (getattr(instance, "fx", None), fast.fx, False),
                (getattr(instance, "fgx", None), fast.fgx, True),
            ):
                if ref_fn is None:
                    continue
                try:
                    ref = ref_fn(x.copy())
                except Exception:
                    continue
                try:
                    new = fast_fn(x)
                except Exception:
                    return False
                if is_pair:
                    if not (_close(ref[0], new[0]) and _close(ref[1], new[1])):
                        return False
                elif not _close(ref, new):
                    return False
                checked += 1
        if len(fast.congrps):
            ref_cx = getattr(instance, "cx", None)
            if ref_cx is not None:
                try:
                    ref = ref_cx(x.copy())
                except Exception:
                    ref = None
                if ref is not None:
                    try:
                        if not _close(ref, fast.cx(x)):
                            return False
                    except Exception:
                        return False
                    checked += 1
            ref_cJx = getattr(instance, "cJx", None)
            if ref_cJx is not None:
                try:
                    c_ref, J_ref = ref_cJx(x.copy())
                except Exception:
                    c_ref = J_ref = None
                if c_ref is not None:
                    try:
                        c_new, J_new = fast.cJx(x)
                        if not (_close(c_ref, c_new) and _jac_close(J_ref, J_new)):
                            return False
                    except Exception:
                        return False
                    checked += 1
    return True if checked else None


def _close(a: Any, b: Any) -> bool:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=_VERIFY_RTOL, atol=_VERIFY_ATOL, equal_nan=True))


def _jac_close(ref: Any, new: Any) -> bool:
    ref = sp.csr_matrix(ref)
    new = sp.csr_matrix(new)
    if ref.shape != new.shape:
        return False
    diff = ref - new
    if diff.nnz == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(ref.data))) if ref.nnz else 1.0)
    return float(np.max(np.abs(diff.data))) <= _VERIFY_RTOL * scale


__all__ = ["FastS2MPJEval", "fast_evaluator"]
