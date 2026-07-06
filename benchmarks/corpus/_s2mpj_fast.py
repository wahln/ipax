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

Scope: values, gradients, Jacobians (``fx/fgx/cx/cJx``), and the Lagrangian
Hessian (``LgHxy``: value, gradient, and Hessian of ``f + Σ y_i c_i``). The
Hessian's COO structure is fixed per instance, so it is precompiled once and
each call only fills values (measured: ~3–70× per call vs the interpretive
``evalgrsum`` at ``nargout=3``). It is verified — and can fall back —
**independently** of the base methods: a Hessian-only mismatch or an oversized
structure (``_MAX_HESS_NNZ``) sets ``lghxy_ok=False`` and only ``LgHxy``
reverts to the original, keeping the verified fast ``fx/cx``. Problems that
declare *partial* derivative levels (``objderlvl``/``conderlvl`` < 2, where
the original substitutes NaNs) are rejected at construction and fall back to
the original methods.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import scipy.sparse as sp

# Element entry: (element function, elemental variable indices, weight or None,
# element index). Group entry: (elements, group function or None for TRIVIAL).
_Element = tuple[Any, np.ndarray, float | None, int]
_Group = tuple[list[_Element], Any]

# Jacobian element entry: an _Element plus the precomputed positions of its
# elemental variables inside the group's support — so no per-call
# ``searchsorted``.
_JElement = tuple[Any, np.ndarray, float | None, int, np.ndarray]


class _JacGroup(NamedTuple):
    """Precompiled per-constraint-group structure for ``cJx``.

    The Jacobian's sparsity is fixed: row ``k``'s columns are the group's
    sorted support (A-row indices ∪ element variables), so the CSR
    ``indices``/``indptr`` are built once and each call fills only the
    ``data`` segment starting at ``off``. The linear term's scatter positions
    (``a_pos``) and constant values (``a_data``) are precomputed too.
    """

    ig: int
    elements: list[_JElement]
    gfn: Any
    support: np.ndarray
    a_pos: np.ndarray
    a_data: np.ndarray
    off: int


class _JacStructure(NamedTuple):
    groups: list[_JacGroup]
    indices: np.ndarray
    indptr: np.ndarray
    nnz: int


# Hessian element entry: an _Element plus the precomputed positions of its
# elemental variables inside the group's full support (``gpos``, for the inner
# gradient) and inside its element-variable support (``hpos``, for the inner
# Hessian block) — so no per-call ``searchsorted``.
_HElement = tuple[Any, np.ndarray, float | None, int, np.ndarray, np.ndarray]


class _HessGroup(NamedTuple):
    """Precompiled per-group structure for the Lagrangian Hessian.

    ``k`` is the group's position in S2MPJ's constraint list (its multiplier
    index), or ``-1`` for an objective group (coefficient 1). The group's
    Hessian occupies up to two fixed COO blocks in the assembled triplet
    vector: the ``gsupp × gsupp`` outer-product block ``Hessa·gin·ginᵀ`` at
    ``off_outer`` (only for a non-TRIVIAL group function, whose curvature acts
    on the full inner gradient including the linear term) and the
    ``hsupp × hsupp`` element-curvature block ``grada·Hin`` at ``off_hloc``
    (only when the group has elements; ``hsupp`` spans the element variables).
    """

    ig: int
    k: int
    elements: list[_HElement]
    gfn: Any
    gsupp: np.ndarray
    a_pos: np.ndarray
    a_data: np.ndarray
    off_outer: int
    off_hloc: int
    hsize: int


class _HessStructure(NamedTuple):
    groups: list[_HessGroup]
    rows: np.ndarray
    cols: np.ndarray
    nnz: int


# Refuse to precompile a Hessian whose fixed COO structure would be enormous
# (a non-TRIVIAL group over a very wide support squares its size); the original
# interpretive path stays available.
_MAX_HESS_NNZ = 20_000_000

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

        # Fixed constraint-Jacobian structure (supports, scatter positions,
        # canonical CSR layout) — each cJx call only fills values.
        self._jac = self._compile_jacobian()

        self.H = getattr(instance, "H", None)
        self._has_objective = len(self.objgrps) > 0 or self.H is not None

        # Lagrangian-Hessian structure, compiled separately so an unsupported
        # feature only loses the Hessian path, never the fx/cx one. Whether it
        # is *used* is decided by verification (``lghxy_ok``, set by
        # :func:`fast_evaluator`).
        self.lghxy_ok = False
        try:
            self._hess: _HessStructure | None = self._compile_hessian()
        except Exception:
            self._hess = None

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

    def _compile_jacobian(self) -> _JacStructure:
        """Precompute the fixed CSR structure of the constraint Jacobian.

        Row ``k``'s columns are group ``k``'s sorted support (A-row indices ∪
        element variables), so ``indices``/``indptr`` are already in canonical
        CSR form; per element the scatter positions into the support are
        recorded once. Each :meth:`cJx` call then only fills the ``data``
        vector — no ``searchsorted``, no COO→CSR sort.
        """
        A = self.A
        groups: list[_JacGroup] = []
        indptr = np.zeros(len(self.congrps) + 1, dtype=int)
        parts: list[np.ndarray] = []
        nnz = 0
        for k, ig_ in enumerate(self.congrps):
            ig = int(ig_)
            elements, gfn = self.con_groups[k]
            a0, a1 = A.indptr[ig], A.indptr[ig + 1]
            a_idx = A.indices[a0:a1]
            cols = set(a_idx.tolist())
            for _fn, idx, _w, _iel in elements:
                cols.update(idx.tolist())
            support = np.fromiter(sorted(cols), dtype=int, count=len(cols))
            elements_j: list[_JElement] = [
                (fn, idx, w, iel, np.searchsorted(support, idx))
                for fn, idx, w, iel in elements
            ]
            groups.append(
                _JacGroup(
                    ig,
                    elements_j,
                    gfn,
                    support,
                    np.searchsorted(support, a_idx),
                    np.asarray(A.data[a0:a1], dtype=float),
                    nnz,
                )
            )
            parts.append(support)
            nnz += support.size
            indptr[k + 1] = nnz
        indices = np.concatenate(parts) if parts else np.zeros(0, dtype=int)
        return _JacStructure(groups, indices, indptr, nnz)

    def _compile_hessian(self) -> _HessStructure:
        """Precompute the fixed COO structure of the Lagrangian Hessian.

        Walks the objective and constraint groups once, recording each group's
        support, its elements' scatter positions, and the offsets of its (up to
        two) dense-block segments in the concatenated triplet vector — so each
        :meth:`LgHxy` call only fills values into a preallocated layout.
        """
        A = self.A
        groups: list[_HessGroup] = []
        rows_parts: list[np.ndarray] = []
        cols_parts: list[np.ndarray] = []
        nnz = 0
        specs = [(int(ig), -1, self.obj_groups[j]) for j, ig in enumerate(self.objgrps)]
        specs += [(int(ig), k, self.con_groups[k]) for k, ig in enumerate(self.congrps)]
        for ig, k, (elements, gfn) in specs:
            a0, a1 = A.indptr[ig], A.indptr[ig + 1]
            a_idx = A.indices[a0:a1]
            evars: set[int] = set()
            for _fn, idx, _w, _iel in elements:
                evars.update(idx.tolist())
            gcols = evars | set(a_idx.tolist())
            gsupp = np.fromiter(sorted(gcols), dtype=int, count=len(gcols))
            hsupp = np.fromiter(sorted(evars), dtype=int, count=len(evars))
            elements_h: list[_HElement] = [
                (
                    fn,
                    idx,
                    w,
                    iel,
                    np.searchsorted(gsupp, idx),
                    np.searchsorted(hsupp, idx),
                )
                for fn, idx, w, iel in elements
            ]
            # Size gate *before* materializing the blocks, so an enormous
            # group cannot transiently allocate its row/col arrays first.
            grow = (gsupp.size**2 if gfn is not None else 0) + hsupp.size**2
            if nnz + grow > _MAX_HESS_NNZ:
                raise NotImplementedError("Hessian COO structure too large")
            off_outer = -1
            if gfn is not None and gsupp.size:
                off_outer = nnz
                rows_parts.append(np.repeat(gsupp, gsupp.size))
                cols_parts.append(np.tile(gsupp, gsupp.size))
                nnz += gsupp.size * gsupp.size
            off_hloc = -1
            if hsupp.size:
                off_hloc = nnz
                rows_parts.append(np.repeat(hsupp, hsupp.size))
                cols_parts.append(np.tile(hsupp, hsupp.size))
                nnz += hsupp.size * hsupp.size
            groups.append(
                _HessGroup(
                    ig,
                    k,
                    elements_h,
                    gfn,
                    gsupp,
                    np.searchsorted(gsupp, a_idx),
                    np.asarray(A.data[a0:a1], dtype=float),
                    off_outer,
                    off_hloc,
                    int(hsupp.size),
                )
            )
        rows = np.concatenate(rows_parts) if rows_parts else np.zeros(0, dtype=int)
        cols = np.concatenate(cols_parts) if cols_parts else np.zeros(0, dtype=int)
        return _HessStructure(groups, rows, cols, nnz)

    # -- Lagrangian value / gradient / Hessian --------------------------------

    def LgHxy(self, x: Any, y: Any) -> tuple[float, np.ndarray, sp.csr_matrix]:
        """``(L, ∇L, ∇²L)`` of ``L = f + Σ_i y_i c_i`` (mirrors the original).

        One pass over all groups: objective groups contribute with coefficient
        1, constraint group ``k`` with ``y[k]`` (a zero multiplier skips the
        group entirely). Per group the Hessian is ``evalgrsum``'s
        ``(Hessa·gin·ginᵀ + grada·Hin)/gsc`` (``Hin/gsc`` for TRIVIAL), written
        into the precompiled COO layout; duplicate (i, j) entries across groups
        sum in the CSR conversion.
        """
        hs = self._hess
        if hs is None:
            raise NotImplementedError("no precompiled Hessian structure")
        x = np.asarray(x, dtype=float).reshape(-1)
        yv = np.asarray(y, dtype=float).reshape(-1)
        xc = x.reshape(-1, 1)
        n = self.n
        Lxy = 0.0
        gx = np.zeros(n)
        vals = np.zeros(hs.nnz)
        if self.H is not None:
            Hx = np.asarray(self.H @ x).reshape(-1)
            Lxy += 0.5 * float(x @ Hx)
            gx += Hx
        fin_all = self.A @ x
        for grp in hs.groups:
            coeff = 1.0 if grp.k < 0 else float(yv[grp.k])
            if coeff == 0.0:
                continue
            f: Any = fin_all[grp.ig] - self.gconst[grp.ig]
            gin = np.zeros(grp.gsupp.size)
            if grp.a_pos.size:
                gin[grp.a_pos] += grp.a_data  # CSR column indices are unique
            Hloc = np.zeros((grp.hsize, grp.hsize)) if grp.hsize else None
            for fn, idx, w, iel, gpos, hpos in grp.elements:
                fiel, giel, Hiel = fn(self.inst, 3, xc[idx], iel)
                giel = np.asarray(giel, dtype=float).reshape(-1)
                Hiel = np.asarray(Hiel, dtype=float)
                if w is not None:
                    f = f + w * fiel
                    np.add.at(gin, gpos, w * giel)
                    np.add.at(Hloc, (hpos[:, None], hpos[None, :]), w * Hiel)
                else:
                    f = f + fiel
                    np.add.at(gin, gpos, giel)
                    np.add.at(Hloc, (hpos[:, None], hpos[None, :]), Hiel)
            gsc = self.gscale[grp.ig]
            if grp.gfn is not None:
                fa, grada, Hessa = grp.gfn(self.inst, 3, f, grp.ig)
                grada = float(np.asarray(grada).reshape(-1)[0])
                Hessa = float(np.asarray(Hessa).reshape(-1)[0])
                fval = float(np.asarray(fa).reshape(-1)[0]) / gsc
                gvec = (grada / gsc) * gin
                if grp.off_outer >= 0:
                    seg = slice(grp.off_outer, grp.off_outer + gin.size * gin.size)
                    vals[seg] = (coeff * Hessa / gsc) * np.outer(gin, gin).ravel()
                if Hloc is not None:
                    seg = slice(grp.off_hloc, grp.off_hloc + grp.hsize * grp.hsize)
                    vals[seg] = (coeff * grada / gsc) * Hloc.ravel()
            else:
                fval = float(np.asarray(f).reshape(-1)[0]) / gsc
                gvec = gin / gsc
                if Hloc is not None:
                    seg = slice(grp.off_hloc, grp.off_hloc + grp.hsize * grp.hsize)
                    vals[seg] = (coeff / gsc) * Hloc.ravel()
            Lxy += coeff * fval
            gx[grp.gsupp] += coeff * gvec  # gsupp is sorted-unique: plain add
        Hout = sp.coo_matrix((vals, (hs.rows, hs.cols)), shape=(n, n)).tocsr()
        # The fixed dense-block layout stores zeros the original's value-sparse
        # assembly would not; drop them so sparse-direct consumers factor a
        # comparable nnz.
        Hout.eliminate_zeros()
        if self.H is not None:
            Hout = Hout + sp.csr_matrix(self.H, dtype=float)
        return Lxy, gx.reshape(-1, 1), Hout

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
        """Constraint values and Jacobian ``(c, J)`` with ``J`` in CSR.

        Fills the precompiled canonical CSR layout (``_compile_jacobian``); the
        returned matrices share the fixed ``indices``/``indptr`` arrays across
        calls, but each call allocates a fresh ``data`` vector.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        xc = x.reshape(-1, 1)
        m = len(self.congrps)
        fin = (self.A @ x)[self.congrps] - self.gconst[self.congrps]
        js = self._jac
        cvals = np.empty((m, 1))
        data = np.zeros(js.nnz)
        for k, grp in enumerate(js.groups):
            gin = np.zeros(grp.support.size)
            if grp.a_pos.size:
                gin[grp.a_pos] += grp.a_data  # CSR column indices are unique
            f = fin[k]
            for fn, idx, w, iel, pos in grp.elements:
                fiel, giel = fn(self.inst, 2, xc[idx], iel)
                giel = np.asarray(giel, dtype=float).reshape(-1)
                if w is not None:
                    f = f + w * fiel
                    np.add.at(gin, pos, w * giel)
                else:
                    f = f + fiel
                    np.add.at(gin, pos, giel)
            if grp.gfn is not None:
                f, grada = grp.gfn(self.inst, 2, f, grp.ig)
                gin = grada * gin
            gsc = self.gscale[grp.ig]
            cvals[k, 0] = float(np.asarray(f).reshape(-1)[0]) / gsc
            if grp.support.size:
                data[grp.off : grp.off + grp.support.size] = (
                    np.asarray(gin, dtype=float).reshape(-1) / gsc
                )
        J = sp.csr_matrix((data, js.indices, js.indptr), shape=(m, self.n))
        # The structure *is* canonical (sorted-unique columns per row); saying
        # so stops scipy from ever running its in-place sort/dedup mutators on
        # the shared ``indices``/``indptr`` template.
        J.has_sorted_indices = True
        J.has_canonical_format = True
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
    # The Hessian path is gated separately: a mismatch (or missing structure)
    # only disables ``LgHxy``, the verified base methods stay in use.
    hess_ok: bool | None = None if fast._hess is not None else False
    m = len(fast.congrps)
    for shift, x in enumerate((x0, probe)):
        agree = _agrees_at(instance, fast, x)
        if agree is False:
            return None
        verified += agree is True
        if hess_ok is not False:
            # Deterministic sign-varied multipliers (shifted per probe point);
            # all-equal or all-zero y would mask per-group mix-ups.
            y = (np.cos(np.arange(m) + shift) + 0.25).reshape(-1, 1)
            hess_agree = _hess_agrees_at(instance, fast, x, y)
            if hess_agree is not None:
                hess_ok = hess_agree
    fast.lghxy_ok = bool(hess_ok)
    return fast if verified else None


def _hess_agrees_at(
    instance: Any, fast: FastS2MPJEval, x: np.ndarray, y: np.ndarray
) -> bool | None:
    """Tri-state ``LgHxy`` agreement at one point (same policy as base methods)."""
    ref_fn = getattr(instance, "LgHxy", None)
    if ref_fn is None:
        return None
    with np.errstate(all="ignore"):
        try:
            l_ref, g_ref, h_ref = ref_fn(x.copy().reshape(-1, 1), y.copy())
        except Exception:
            return None  # the original is the authority on exceptions
        try:
            l_new, g_new, h_new = fast.LgHxy(x, y)
        except Exception:
            return False
        return bool(
            _close(l_ref, l_new) and _close(g_ref, g_new) and _jac_close(h_ref, h_new)
        )


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
