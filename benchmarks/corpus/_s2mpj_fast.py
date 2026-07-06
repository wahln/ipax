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
become one pre-filled CSR ``data`` template, and each side's (constraint /
objective) Jacobian sparsity is a fixed canonical CSR layout that calls only
fill. On top of that, same-elftype elements are evaluated **in one vectorized
call per type**: the machine-generated element functions are scalar-coded, so a
mechanical AST rewrite (see :class:`_BatchRewriter`) vectorizes them across a
trailing batch axis, and precomputed scatter slots turn the per-element
gradient writes into a handful of ``np.add.at`` calls. Each batched type is
numerically verified against its per-element original at build time; a type
that cannot be transformed or does not agree stays on the per-element path
(and the whole-evaluator verification below still guards the composition).
Values match the original to floating-point roundoff (summation order
differs). Measured vs the original ``evalgrsum``: ~25–250× per ``cx`` call;
vs the pre-batching evaluator: ~4–25× per ``cx``/``cJx``, ~2× per ``fgx``
(support-based objective gradient), halving an element-heavy L-BFGS solve
end-to-end. On a 96-problem corpus sample, 158/158 elftypes (277k element
occurrences) batch successfully.

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

import ast
import inspect
import textwrap
from typing import Any, NamedTuple

import numpy as np
import scipy.sparse as sp

# Element entry: (element function, elemental variable indices, weight or None,
# element index). Group entry: (elements, group function or None for TRIVIAL).
_Element = tuple[Any, np.ndarray, float | None, int]
_Group = tuple[list[_Element], Any]

# Leftover (non-batched) element entry: an _Element plus its row index and the
# precomputed global positions of its elemental variables in the side's CSR
# ``data`` vector — so no per-call ``searchsorted``.
_LeftElement = tuple[Any, np.ndarray, float | None, int, int, np.ndarray]


class _BatchType(NamedTuple):
    """One elftype's elements, evaluated in a single vectorized call.

    ``fn`` is the AST-transformed variant of the generated element function
    (see :func:`_batched_fn`): it takes the elemental variables of all
    ``N`` same-type elements column-stacked as ``EV_ (k, N)`` (plus the
    pre-gathered per-element parameters ``elpar (p, N)``) and returns the
    values ``(N,)`` and gradients ``(k, N)`` in one shot. ``gather`` indexes
    ``x`` into that stack; ``slots`` are the elements' global positions in the
    side's CSR ``data`` vector; ``rows`` their row indices for the value
    accumulation; ``w`` the group-element weights (1 where unweighted).
    """

    name: str
    fn: Any
    gather: np.ndarray
    elpar: np.ndarray | None
    w: np.ndarray
    rows: np.ndarray
    slots: np.ndarray


class _GfnBatch(NamedTuple):
    """One gftype's non-TRIVIAL group functions, applied in one call.

    ``fn`` maps the rows' inner values ``(N,)`` (plus pre-gathered per-group
    parameters) to ``(fa, grada)`` vectors; ``rows`` are the row indices,
    ``seg_idx`` the concatenated positions of those rows' CSR data segments
    (scaled by ``repeat(grada, sizes)``).
    """

    name: str
    fn: Any
    rows: np.ndarray
    grpar: np.ndarray | None
    seg_idx: np.ndarray
    sizes: np.ndarray


class _Side(NamedTuple):
    """Precompiled structure for one side (constraint rows or objective rows).

    The sparsity is fixed: row ``k``'s columns are its group's sorted support
    (A-row indices ∪ element variables), so the canonical CSR
    ``indices``/``indptr`` are built once; ``base`` is the ``data`` template
    pre-filled with the constant linear-term values. Elements are split into
    per-elftype vectorized batches (``batched``, verified at build) and a
    per-element ``leftover`` list; the non-TRIVIAL group functions likewise
    into per-gftype batches (``gfn_batched``) and a per-row ``gfns`` list of
    ``(row, gfn, ig, off, size)``.
    """

    rows_grps: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    nnz: int
    base: np.ndarray
    gsc_rows: np.ndarray
    gsc_data: np.ndarray
    batched: list[_BatchType]
    leftover: list[_LeftElement]
    gfn_batched: list[_GfnBatch]
    gfns: list[tuple[int, Any, int, int, int]]


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
    """Fixed COO layout of the Lagrangian Hessian plus its precomputed CSR
    reduction: ``order`` sorts the triplet vector row-major, ``starts`` marks
    each distinct (i, j) run, so per call the CSR data is one gather +
    ``np.add.reduceat`` — no COO sort."""

    groups: list[_HessGroup]
    rows: np.ndarray
    cols: np.ndarray
    nnz: int
    order: np.ndarray
    starts: np.ndarray
    csr_indices: np.ndarray
    csr_indptr: np.ndarray


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


# -- per-elftype batch compilation ---------------------------------------------
#
# S2MPJ's generated element functions are scalar-coded (``EV_[i,0]`` indexing,
# ``g_ = np.zeros(dim)``, per-element parameters via ``self.elpar[iel_]``), so
# same-type elements cannot be batched by simply stacking inputs. Because the
# code is machine-generated from a handful of templates, a *mechanical* AST
# rewrite vectorizes it across elements:
#
#   EV_ becomes (k, N) column-stacked  →  ``EV_[i, 0]``        → ``EV_[i]``
#   per-element parameters             →  ``self.elpar[iel_]`` → ``ELPAR_`` (p, N)
#   gradient allocation                →  ``np.zeros(dim)``    → ``np.zeros((dim, NB_))``
#   internal-variable reduction        →  ``to_scalar(e)``     → row vector (exec binding)
#
# Everything else (elementwise arithmetic, the U_ internal-variable transform,
# the ``try: dim = len(IV_)`` idiom) already broadcasts over the trailing batch
# axis. Group functions follow the same templates with ``GVAR_``/``igr_``/
# ``self.grpar`` (and a scalar batch variable, so no subscript rewrite fires).
# Functions with data-dependent control flow (or any construct outside the
# templates) are rejected and stay on the per-element/per-row path; every
# accepted type is additionally *verified numerically* against the original
# per-element evaluation at build time, so a semantically wrong transform can
# never serve values.


class _TransformSpec(NamedTuple):
    batch_var: str  # the per-call input that gains the batch axis
    index_var: str  # the per-element/-group index argument
    par_attr: str  # the self attribute of per-index parameters
    const_attr: str  # the problem-level constants attribute (kept as-is)
    nb_expr: str  # expression for the batch size NB_


_ELEMENT_SPEC = _TransformSpec("EV_", "iel_", "elpar", "efpar", "EV_.shape[1]")
_GROUP_SPEC = _TransformSpec("GVAR_", "igr_", "grpar", "gfpar", "GVAR_.shape[0]")

_BATCH_FN_CACHE: dict[tuple[type, str], Any] = {}


class _BatchRewriter(ast.NodeTransformer):
    """Vectorize one generated element/group function; ``ok=False`` on any
    pattern outside the known machine-generated templates."""

    def __init__(self, spec: _TransformSpec) -> None:
        self.spec = spec
        self.ok = True
        self._depth = 0

    def _reject(self) -> None:
        self.ok = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if self._depth:  # nested function: outside the templates
            self._reject()
            return node
        self._depth += 1
        self.generic_visit(node)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        # self.<par_attr>[<index_var>] -> PARS_ (rewrite before descending, so
        # the attribute guard below never sees the sanctioned access).
        v = node.value
        if (
            isinstance(v, ast.Attribute)
            and isinstance(v.value, ast.Name)
            and v.value.id == "self"
            and v.attr == self.spec.par_attr
            and isinstance(node.slice, ast.Name)
            and node.slice.id == self.spec.index_var
        ):
            return ast.copy_location(ast.Name(id="PARS_", ctx=node.ctx), node)
        self.generic_visit(node)
        # EV_[i, 0] -> EV_[i] (the batch axis replaces the column axis).
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == self.spec.batch_var
            and isinstance(node.slice, ast.Tuple)
            and len(node.slice.elts) == 2
            and all(isinstance(e, ast.Constant) for e in node.slice.elts)
        ):
            first, second = node.slice.elts
            if second.value == 0:  # type: ignore[attr-defined]
                node.slice = first
            else:  # EV_[i, j≠0]: outside the generated pattern
                self._reject()
        return node

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        # Only problem-level constants may be read off self (per-index
        # parameter accesses were rewritten above; anything else is outside
        # the templates).
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr != self.spec.const_attr
        ):
            self._reject()
        self.generic_visit(node)
        return node

    def visit_If(self, node: ast.If) -> Any:
        # Only the structural ``nargout`` dispatch may branch; a data-dependent
        # branch cannot be vectorized.
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if names - {"nargout"}:
            self._reject()
            return node
        self.generic_visit(node)
        return node

    def visit_For(self, node: ast.For) -> Any:
        self._reject()
        return node

    def visit_While(self, node: ast.While) -> Any:
        self._reject()
        return node

    def visit_Call(self, node: ast.Call) -> Any:
        self.generic_visit(node)
        # np.zeros(dim) / np.ones(dim) -> np.zeros((dim, NB_)): the gradient
        # work vector gains the batch axis. Tuple shapes (U_, H_) stay as-is.
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr in ("zeros", "ones")
            and len(node.args) == 1
            and not node.keywords
            and not isinstance(node.args[0], ast.Tuple)
        ):
            node.args[0] = ast.Tuple(
                elts=[node.args[0], ast.Name(id="NB_", ctx=ast.Load())],
                ctx=ast.Load(),
            )
        return node


def _batch_to_scalar(value: Any) -> Any:
    """Batch-mode ``to_scalar``: a (1, N) reduction row becomes (N,)."""
    return np.asarray(value).reshape(-1)


def _batched_fn(cls: type, name: str, spec: _TransformSpec) -> Any:
    """The vectorized variant of element/group function ``name``, or None.

    Cached per (class, name): the transform is source-level, so instances of
    the same problem class share it. Returns ``None`` when the source is
    unavailable or uses a construct outside the generated templates; numeric
    verification against the per-element/-row original happens at the *call
    site* (``_compile_side``), which demotes a mismatching type.
    """
    key = (cls, name)
    if key in _BATCH_FN_CACHE:
        return _BATCH_FN_CACHE[key]
    fn = None
    try:
        source = textwrap.dedent(inspect.getsource(getattr(cls, name)))
        tree = ast.parse(source)
        fdef = tree.body[0]
        assert isinstance(fdef, ast.FunctionDef)
        fdef.decorator_list = []  # drop @staticmethod: exec'd standalone
        rewriter = _BatchRewriter(spec)
        tree = rewriter.visit(tree)
        # Any surviving index-var *read* means a per-element dependence the
        # parameter rewrite did not cover.
        index_reads = any(
            isinstance(n, ast.Name)
            and n.id == spec.index_var
            and isinstance(n.ctx, ast.Load)
            for n in ast.walk(fdef)
        )
        # The batch bindings slot in right after the generated prologue
        # (``EV_ = args[0]``, ``iel_ = args[1]`` / the GVAR_/igr_ twins).
        anchor = next(
            (
                i
                for i, stmt in enumerate(fdef.body)
                if isinstance(stmt, ast.Assign)
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == spec.index_var
            ),
            None,
        )
        if rewriter.ok and not index_reads and anchor is not None:
            inject = ast.parse(f"PARS_ = args[2]\nNB_ = {spec.nb_expr}").body
            fdef.body[anchor + 1 : anchor + 1] = inject
            ast.fix_missing_locations(tree)
            namespace: dict[str, Any] = {"np": np, "to_scalar": _batch_to_scalar}
            exec(  # compiling our own transform of the generated source
                compile(tree, f"<batched {cls.__name__}.{name}>", "exec"), namespace
            )
            fn = namespace[fdef.name]  # source name (may be an alias of ``name``)
    except Exception:
        fn = None
    _BATCH_FN_CACHE[key] = fn
    return fn


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

        # Fixed CSR-layout structure per side (constraint rows / objective
        # rows): supports, scatter positions, linear-term template, and the
        # per-elftype vectorized batches (each verified numerically against
        # the per-element original — a failing type stays per-element).
        self._con = self._compile_side(self.congrps, self.con_groups)
        self._objside = self._compile_side(self.objgrps, self.obj_groups)
        self.batched_elftypes = frozenset(
            bt.name for side in (self._con, self._objside) for bt in side.batched
        )
        self.batched_gftypes = frozenset(
            gb.name for side in (self._con, self._objside) for gb in side.gfn_batched
        )

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

    def _compile_side(self, grps: np.ndarray, groups: list[_Group]) -> _Side:
        """Precompute one side's fixed CSR structure and element batches.

        Row ``k``'s columns are group ``k``'s sorted support (A-row indices ∪
        element variables), so ``indices``/``indptr`` are already in canonical
        CSR form; the constant linear-term values are pre-filled into the
        ``base`` template. Elements are grouped by elftype and each type is
        batch-compiled (:func:`_batched_element_fn`) and *numerically verified*
        against its per-element original at two probe points — a type that
        cannot be transformed or does not agree stays on the per-element
        ``leftover`` path.
        """
        A = self.A
        indptr = np.zeros(len(grps) + 1, dtype=int)
        parts: list[np.ndarray] = []
        gfns: list[tuple[int, Any, int, int, int]] = []
        # (elftype, fn, idx, w, iel, row, slots) per element occurrence
        entries: list[
            tuple[str, Any, np.ndarray, float | None, int, int, np.ndarray]
        ] = []
        a_fill: list[tuple[np.ndarray, np.ndarray]] = []
        nnz = 0
        for k, ig_ in enumerate(grps):
            ig = int(ig_)
            elements, gfn = groups[k]
            a0, a1 = A.indptr[ig], A.indptr[ig + 1]
            a_idx = A.indices[a0:a1]
            cols = set(a_idx.tolist())
            for _fn, idx, _w, _iel in elements:
                cols.update(idx.tolist())
            support = np.fromiter(sorted(cols), dtype=int, count=len(cols))
            for fn, idx, w, iel in elements:
                slots = nnz + np.searchsorted(support, idx)
                entries.append((self.inst.elftype[iel], fn, idx, w, iel, k, slots))
            a_fill.append(
                (
                    nnz + np.searchsorted(support, a_idx),
                    np.asarray(A.data[a0:a1], dtype=float),
                )
            )
            if gfn is not None:
                gfns.append((k, gfn, ig, nnz, int(support.size)))
            parts.append(support)
            nnz += support.size
            indptr[k + 1] = nnz
        indices = np.concatenate(parts) if parts else np.zeros(0, dtype=int)
        base = np.zeros(nnz)
        for pos, vals in a_fill:
            base[pos] = vals
        grps_arr = np.asarray(grps, dtype=int).reshape(-1)
        gsc_rows = self.gscale[grps_arr]
        gsc_data = np.repeat(gsc_rows, np.diff(indptr))
        batched, leftover = self._batch_entries(entries)
        gfn_batched, gfns_left = self._batch_gfns(gfns)
        return _Side(
            grps_arr,
            indices,
            indptr,
            nnz,
            base,
            gsc_rows,
            gsc_data,
            batched,
            leftover,
            gfn_batched,
            gfns_left,
        )

    def _batch_entries(
        self,
        entries: list[tuple[str, Any, np.ndarray, float | None, int, int, np.ndarray]],
    ) -> tuple[list[_BatchType], list[_LeftElement]]:
        """Split element occurrences into verified batches and leftovers."""
        by_type: dict[str, list[Any]] = {}
        for entry in entries:
            by_type.setdefault(entry[0], []).append(entry)
        batched: list[_BatchType] = []
        leftover: list[_LeftElement] = []
        for name, group in by_type.items():
            bt = self._try_batch_type(name, group)
            if bt is not None:
                batched.append(bt)
            else:
                leftover.extend(
                    (fn, idx, w, iel, row, slots)
                    for (_nm, fn, idx, w, iel, row, slots) in group
                )
        return batched, leftover

    def _try_batch_type(self, name: str, group: list[Any]) -> _BatchType | None:
        """Build and verify one elftype's batch, or ``None`` to stay per-element."""
        fn = _batched_fn(type(self.inst), name, _ELEMENT_SPEC)
        k = int(group[0][2].size)
        if fn is None or k == 0 or any(e[2].size != k for e in group):
            return None
        try:
            elpar = self._gather_pars("elpar", [e[4] for e in group])
        except Exception:
            return None
        bt = _BatchType(
            name,
            fn,
            np.stack([e[2] for e in group], axis=1),
            elpar,
            np.array([1.0 if e[3] is None else e[3] for e in group]),
            np.array([e[5] for e in group], dtype=int),
            np.stack([e[6] for e in group], axis=1),
        )
        return bt if self._verify_batch(bt, group) else None

    def _batch_gfns(
        self, gfns: list[tuple[int, Any, int, int, int]]
    ) -> tuple[list[_GfnBatch], list[tuple[int, Any, int, int, int]]]:
        """Split non-TRIVIAL group functions into verified per-gftype batches."""
        grftype = getattr(self.inst, "grftype", None)
        by_name: dict[str, list[tuple[int, Any, int, int, int]]] = {}
        left: list[tuple[int, Any, int, int, int]] = []
        for row in gfns:
            ig = row[2]
            gname = (
                grftype[ig]
                if grftype is not None and ig < len(grftype) and grftype[ig] is not None
                else None
            )
            if gname is None:  # cannot name the type: keep per-row
                left.append(row)
            else:
                by_name.setdefault(gname, []).append(row)
        batched: list[_GfnBatch] = []
        for gname, rows in by_name.items():
            gb = self._try_batch_gfn(gname, rows)
            if gb is not None:
                batched.append(gb)
            else:
                left.extend(rows)
        return batched, left

    def _try_batch_gfn(
        self, gname: str, rows: list[tuple[int, Any, int, int, int]]
    ) -> _GfnBatch | None:
        """Build and verify one gftype's batch, or ``None`` to stay per-row."""
        fn = _batched_fn(type(self.inst), gname, _GROUP_SPEC)
        if fn is None:
            return None
        try:
            grpar = self._gather_pars("grpar", [r[2] for r in rows])
        except Exception:
            return None
        sizes = np.array([r[4] for r in rows], dtype=int)
        seg_idx = (
            np.concatenate([np.arange(r[3], r[3] + r[4]) for r in rows])
            if int(sizes.sum())
            else np.zeros(0, dtype=int)
        )
        gb = _GfnBatch(
            gname, fn, np.array([r[0] for r in rows], dtype=int), grpar, seg_idx, sizes
        )
        return gb if self._verify_gfn_batch(gb, rows) else None

    def _gather_pars(self, attr: str, idxs: list[int]) -> np.ndarray | None:
        """Column-stack per-element/-group parameters, ``None`` when absent.

        Raises on ragged/partially-missing parameters (→ type not batchable).
        """
        pars = getattr(self.inst, attr, None)
        rows = [
            None
            if pars is None or i >= len(pars) or pars[i] is None
            else np.asarray(pars[i], dtype=float).reshape(-1)
            for i in idxs
        ]
        if all(r is None for r in rows):
            return None
        if any(r is None for r in rows):
            raise ValueError("partially missing parameters")
        return np.stack(rows, axis=1)  # type: ignore[arg-type]

    def _verify_batch(self, bt: _BatchType, group: list[Any]) -> bool:
        """Batched (f, g) must reproduce the per-element original at 2 probes."""
        x0 = np.asarray(self.inst.x0, dtype=float).reshape(-1)
        probe = x0 + 1e-4 * (1.0 + np.abs(x0)) * np.cos(np.arange(x0.shape[0]))
        n_batch = bt.rows.size
        for x in (x0, probe):
            xc = x.reshape(-1, 1)
            with np.errstate(all="ignore"):
                try:
                    fb, gb = bt.fn(self.inst, 2, x[bt.gather], None, bt.elpar)
                    fb = np.asarray(fb, dtype=float).reshape(-1)
                    gb = np.asarray(gb, dtype=float)
                except Exception:
                    return False
                if fb.shape != (n_batch,) or gb.shape != bt.slots.shape:
                    return False
                for j, (_nm, fn, idx, _w, iel, _row, _slots) in enumerate(group):
                    try:
                        fr, gr = fn(self.inst, 2, xc[idx], iel)
                    except Exception:
                        # The original raises here: keep its exact semantics
                        # (including the raise) via the per-element path.
                        return False
                    if not (_close(fr, fb[j]) and _close(gr, gb[:, j])):
                        return False
        return True

    def _verify_gfn_batch(
        self, gb: _GfnBatch, rows: list[tuple[int, Any, int, int, int]]
    ) -> bool:
        """Batched (fa, grada) must reproduce the per-row original at 2 probes.

        Synthetic inner-value probes suffice: the transform is mechanical, so
        any structural error diverges everywhere (and domain-limited functions
        produce the same NaNs on both sides). The whole-evaluator verification
        additionally exercises the batched composition at real points.
        """
        n_batch = gb.rows.size
        for shift in (0.3, -0.4):
            v = np.cos(np.arange(n_batch) * 1.3) + shift
            with np.errstate(all="ignore"):
                try:
                    fb, gvb = gb.fn(self.inst, 2, v, None, gb.grpar)
                    fb = np.asarray(fb, dtype=float).reshape(-1)
                    gvb = np.asarray(gvb, dtype=float).reshape(-1)
                except Exception:
                    return False
                if fb.shape != (n_batch,) or gvb.shape != (n_batch,):
                    return False
                for j, (_row, gfn, ig, _off, _size) in enumerate(rows):
                    try:
                        fr, gr = gfn(self.inst, 2, v[j], ig)
                    except Exception:
                        return False
                    if not (_close(fr, fb[j]) and _close(gr, gvb[j])):
                        return False
        return True

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
        # Precompute the COO→CSR reduction (sort order + duplicate-run starts),
        # so assembling the Hessian per call is gather + reduceat.
        if nnz:
            order = np.lexsort((cols, rows))
            r_s, c_s = rows[order], cols[order]
            new_run = np.ones(nnz, dtype=bool)
            new_run[1:] = (r_s[1:] != r_s[:-1]) | (c_s[1:] != c_s[:-1])
            starts = np.flatnonzero(new_run)
            csr_indices = c_s[starts]
            counts = np.zeros(self.n + 1, dtype=int)
            np.add.at(counts, r_s[starts] + 1, 1)
            csr_indptr = np.cumsum(counts)
        else:
            order = starts = csr_indices = np.zeros(0, dtype=int)
            csr_indptr = np.zeros(self.n + 1, dtype=int)
        return _HessStructure(
            groups, rows, cols, nnz, order, starts, csr_indices, csr_indptr
        )

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
        csr_data = np.add.reduceat(vals[hs.order], hs.starts) if hs.nnz else np.zeros(0)
        # Fresh copies of the precomputed structure: eliminate_zeros() below
        # mutates in place, and the template must survive across calls.
        Hout = sp.csr_matrix(
            (csr_data, hs.csr_indices.copy(), hs.csr_indptr.copy()), shape=(n, n)
        )
        Hout.has_sorted_indices = True
        Hout.has_canonical_format = True
        # The fixed dense-block layout stores zeros the original's value-sparse
        # assembly would not; drop them so sparse-direct consumers factor a
        # comparable nnz.
        Hout.eliminate_zeros()
        if self.H is not None:
            Hout = Hout + sp.csr_matrix(self.H, dtype=float)
        return Lxy, gx.reshape(-1, 1), Hout

    # -- shared per-side evaluation ------------------------------------------

    def _eval_side(
        self, side: _Side, x: np.ndarray, want_grad: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Row values (scaled) and CSR ``data`` (scaled) for one side at ``x``.

        One pass: linear term via the pre-filled ``base`` template, batched
        types in one vectorized call each, leftover elements per element, then
        the non-TRIVIAL group functions (whose ``grada`` scales the row's
        contiguous data segment) and the group scaling.
        """
        inst = self.inst
        fin = (self.A @ x)[side.rows_grps] - self.gconst[side.rows_grps]
        data = side.base.copy() if want_grad else None
        for bt in side.batched:
            EV = x[bt.gather]
            if want_grad:
                fb, gb = bt.fn(inst, 2, EV, None, bt.elpar)
                np.add.at(data, bt.slots, bt.w * np.asarray(gb, dtype=float))
            else:
                fb = bt.fn(inst, 1, EV, None, bt.elpar)
            np.add.at(fin, bt.rows, bt.w * np.asarray(fb, dtype=float).reshape(-1))
        if side.leftover:
            xc = x.reshape(-1, 1)
            for fn, idx, w, iel, row, slots in side.leftover:
                if want_grad:
                    fiel, giel = fn(inst, 2, xc[idx], iel)
                    giel = np.asarray(giel, dtype=float).reshape(-1)
                    np.add.at(data, slots, giel if w is None else w * giel)
                else:
                    fiel = fn(inst, 1, xc[idx], iel)
                fv = float(np.asarray(fiel).reshape(-1)[0])
                fin[row] += fv if w is None else w * fv
        for gb in side.gfn_batched:
            vals = fin[gb.rows]
            if want_grad:
                fa, ga = gb.fn(inst, 2, vals, None, gb.grpar)
                data[gb.seg_idx] *= np.repeat(
                    np.asarray(ga, dtype=float).reshape(-1), gb.sizes
                )
            else:
                fa = gb.fn(inst, 1, vals, None, gb.grpar)
            fin[gb.rows] = np.asarray(fa, dtype=float).reshape(-1)
        for row, gfn, ig, off, size in side.gfns:
            if want_grad:
                fa, grada = gfn(inst, 2, fin[row], ig)
                data[off : off + size] *= float(np.asarray(grada).reshape(-1)[0])
            else:
                fa = gfn(inst, 1, fin[row], ig)
            fin[row] = float(np.asarray(fa).reshape(-1)[0])
        fin = fin / side.gsc_rows
        if want_grad:
            data /= side.gsc_data
        return fin, data

    # -- constraints --------------------------------------------------------

    def cx(self, x: Any) -> np.ndarray:
        """Constraint values, shape ``(m, 1)`` (mirrors the original ``cx``)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        fin, _data = self._eval_side(self._con, x, want_grad=False)
        return fin.reshape(-1, 1)

    def cJx(self, x: Any) -> tuple[np.ndarray, sp.csr_matrix]:
        """Constraint values and Jacobian ``(c, J)`` with ``J`` in CSR.

        Fills the precompiled canonical CSR layout (``_compile_side``); the
        returned matrices share the fixed ``indices``/``indptr`` arrays across
        calls, but each call allocates a fresh ``data`` vector.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        side = self._con
        fin, data = self._eval_side(side, x, want_grad=True)
        J = sp.csr_matrix(
            (data, side.indices, side.indptr),
            shape=(side.rows_grps.size, self.n),
        )
        # The structure *is* canonical (sorted-unique columns per row); saying
        # so stops scipy from ever running its in-place sort/dedup mutators on
        # the shared ``indices``/``indptr`` template.
        J.has_sorted_indices = True
        J.has_canonical_format = True
        return fin.reshape(-1, 1), J

    # -- objective ----------------------------------------------------------

    def fx(self, x: Any) -> float:
        return self._obj(x, want_grad=False)  # type: ignore[return-value]

    def fgx(self, x: Any) -> tuple[float, np.ndarray]:
        return self._obj(x, want_grad=True)  # type: ignore[return-value]

    def _obj(self, x: Any, *, want_grad: bool) -> float | tuple[float, np.ndarray]:
        if not self._has_objective:
            raise ValueError("S2MPJ problem has no objective function")
        x = np.asarray(x, dtype=float).reshape(-1)
        side = self._objside
        fin, data = self._eval_side(side, x, want_grad)
        fx = float(np.sum(fin))
        gx = None
        if want_grad:
            # The objective "rows" collapse to one gradient: scatter the scaled
            # per-row data through the shared column indices (support-based —
            # no dense n-vector per group).
            gx = np.zeros(self.n)
            np.add.at(gx, side.indices, data)
        if self.H is not None:
            Hx = np.asarray(self.H @ x).reshape(-1)
            fx += 0.5 * float(x @ Hx)
            if want_grad:
                gx += Hx
        if want_grad:
            return fx, gx.reshape(-1, 1)  # type: ignore[union-attr]
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
