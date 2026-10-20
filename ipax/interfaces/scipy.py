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

"""SciPy-compatible front end: :func:`minimize_ipax`.

An application-edge adapter (it imports NumPy and SciPy, like the
``backend/sparse/`` adapters) that accepts the ``scipy.optimize.minimize``
calling convention, translates it onto a :class:`~ipax.FunctionProblem`, runs
:func:`ipax.solve`, and returns a :class:`scipy.optimize.OptimizeResult`.
``minimize`` hands a callable ``method`` the raw, unstandardized ``bounds`` and
``constraints`` plus ``tol`` inside ``options`` (``scipy/optimize/_minimize.py``,
the ``"_custom"`` branch), so this one function serves both entry points::

    from scipy.optimize import minimize
    from ipax.interfaces.scipy import minimize_ipax

    res = minimize(fun, x0, jac=grad, constraints=cons, method=minimize_ipax)
    res = minimize_ipax(fun, x0, jac=grad, constraints=cons)  # same thing

Array backends
--------------
The adapter takes the Array-API namespace, device and dtype from ``x0`` and
keeps every per-iteration value in it, so a PyTorch / JAX / CuPy ``x0`` runs on
that backend (with its autodiff for missing gradients). SciPy 1.17's
``minimize`` still converts ``x0`` with ``np.asarray`` before calling a custom
method, so through ``minimize`` the arrays are NumPy today; as SciPy moves to
the Array API standard nothing here assumes they are. Inputs that are NumPy by
construction — the data ``LinearConstraint`` / ``Bounds`` coerce to float64
NumPy arrays, ``scipy.sparse`` matrices, ``scipy.sparse.linalg.LinearOperator``
— are converted into ``x0``'s namespace, device and dtype (sparse data once per
evaluation, still as a :class:`~ipax.CSROperator`; ``LinearOperator`` Hessians
work with NumPy ``x`` only; sparse input needs a backend with a sparse
adapter). Constraint bounds ``lb`` / ``ub`` must be
host-convertible (scalars, sequences, NumPy or CPU arrays): they fix the row
structure, which is planned once on the host.

Translation rules
-----------------
* A constraint row with ``lb == ub`` becomes an equality ``c(x) − lb = 0``;
  otherwise each finite side becomes a one-sided inequality ``c(x) − ub ≤ 0``
  and/or ``lb − c(x) ≤ 0``. A dict ``{'type': 'ineq'}`` means ``fun(x) ≥ 0``
  (SciPy's convention), i.e. ``lb = 0, ub = ∞``.
* ``LinearConstraint`` data goes to the constant-data ``linear_eq`` /
  ``linear_ineq`` blocks (assembled once, no Hessian term); sparse ``A`` is
  passed as a :class:`~ipax.CSROperator`, never densified.
* An exact Lagrangian Hessian is used only when the objective has ``hess``
  (or ``hessp``) **and** every ``NonlinearConstraint`` has a callable
  ``hess(x, v)``; otherwise ipax's L-BFGS approximation is used.
  Quasi-Newton ``HessianUpdateStrategy`` objects and finite-difference strings
  are replaced by L-BFGS with an :class:`~scipy.optimize.OptimizeWarning`.

Multipliers are reported with ``trust-constr``'s sign convention: for each
constraint ``res.v[k]`` satisfies ``∇f + Σ_k J_kᵀ v_k + bound_multipliers = 0``
at a KKT point, so an active upper side is positive and an active lower side
negative.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any

try:
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import HessianUpdateStrategy, OptimizeResult, OptimizeWarning
    from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
except ImportError as exc:  # pragma: no cover - exercised only without SciPy
    raise ImportError(
        "ipax.interfaces.scipy needs NumPy and SciPy; install them with "
        "`pip install 'ipax[scipy]'`"
    ) from exc

from ipax.backend.namespace import array_namespace, device
from ipax.backend.operators import CSROperator, LinearOperator, MatrixFreeJacobian
from ipax.options import Options
from ipax.problem.finitediff import jacobian_fd
from ipax.problem.function import FunctionProblem
from ipax.result import IterationInfo, Result
from ipax.solve import solve

STATUS_CODES: MappingProxyType[str, int] = MappingProxyType(
    {
        "optimal": 0,
        "max_iter": 1,
        "infeasible": 2,
        "unbounded": 3,
        "stalled": 4,
        "restoration_failed": 5,
        "numerical_error": 6,
        "stopped": 7,
        "max_time": 8,
        "acceptable": 9,
    }
)
"""``OptimizeResult.status`` code for each :class:`ipax.Status` value.

``0`` is strict optimality and ``1`` the iteration limit, as in most SciPy
methods; ``success`` is true for ``"optimal"`` and ``"acceptable"``.
"""

_FD_STRINGS = frozenset({"2-point", "3-point", "cs"})
_OPTION_FIELDS = frozenset(f.name for f in dataclasses.fields(Options))


# ---------------------------------------------------------------------------
# The array space of x0, and evaluation bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Space:
    """Namespace, device and dtype of ``x0``; every per-iteration value lives here."""

    xp: Any
    device: Any
    dtype: Any
    array_type: Any  # type(x0)

    def asarray(self, value: Any) -> Any:
        # Values already in the space pass through untouched, so autodiff never
        # depends on how a backend's asarray treats a tracked tensor.
        if (
            type(value) is self.array_type
            and value.dtype == self.dtype
            and device(value) == self.device
        ):
            return value
        return self.xp.asarray(value, dtype=self.dtype, device=self.device)

    def index(self, rows: Any) -> Any:
        """Move host-planned row indices onto the device (once, at setup)."""
        return self.xp.asarray(np.asarray(rows, dtype=np.int64), device=self.device)

    def zeros(self, shape: tuple[int, ...]) -> Any:
        return self.xp.zeros(shape, dtype=self.dtype, device=self.device)

    def rows(self, matrix: Any, host_rows: Any, device_rows: Any) -> Any:
        """Row selection of a dense (device) or ``scipy.sparse`` (host) matrix."""
        if sp.issparse(matrix):
            return matrix[host_rows]
        return self.xp.take(matrix, device_rows, axis=0)

    def vstack(self, blocks: list[Any]) -> Any:
        if any(sp.issparse(b) for b in blocks):
            # scipy.sparse lives on the host; dense companions join it there.
            return sp.vstack(
                [b if sp.issparse(b) else sp.csr_array(_on_host(b)) for b in blocks],
                format="csr",
            )
        return self.xp.concat(blocks, axis=0)

    def csr_operator(
        self, matrix: Any, *, symmetric: bool | None = None
    ) -> CSROperator:
        csr = sp.csr_array(matrix)
        csr.sum_duplicates()
        csr.sort_indices()
        return CSROperator(
            self.index(csr.indptr),
            self.index(csr.indices),
            self.asarray(csr.data),
            (int(csr.shape[0]), int(csr.shape[1])),
            symmetric=symmetric,
        )

    def matrix(self, matrix: Any) -> Any:
        """What ipax receives: a device array, or a CSROperator for sparse input."""
        return self.csr_operator(matrix) if sp.issparse(matrix) else matrix


def _on_host(block: Any) -> Any:
    try:
        return np.asarray(block)
    except (TypeError, RuntimeError) as exc:  # CuPy / CUDA tensors refuse
        raise TypeError(
            "a scipy.sparse constraint Jacobian can only be stacked with dense "
            "Jacobians of other constraints that live on the host; return "
            "scipy.sparse from every constraint, or device arrays from every one"
        ) from exc


@dataclass
class _Counts:
    """Per-call evaluation counters reported as ``nfev``/``njev``/``nhev``."""

    nfev: int = 0
    njev: int = 0
    nhev: int = 0


class _LastPoint:
    """Single-entry cache keyed on the *identity* of ``x``.

    Constraint values and Jacobians are requested separately for the equality
    and inequality blocks at one iterate, with the same array object. Keying on
    identity rather than value is what keeps autodiff correct — the Torch/JAX
    adapters evaluate on a fresh tracked copy of the iterate, which must not be
    served the untracked result cached for an equal-valued array — and it costs
    no device synchronization. ipax never updates an iterate in place. The last
    key and value stay referenced until the next call (possibly an autodiff
    tracer or graph); they are never served again, only kept alive.
    """

    def __init__(self, compute: Callable[[Any], Any]) -> None:
        self._compute = compute
        self._x: Any = None
        self._value: Any = None

    def __call__(self, x: Any) -> Any:
        if x is not self._x:
            self._value = self._compute(x)
            self._x = x
        return self._value


# ---------------------------------------------------------------------------
# Constraint normalization (row structure planned on the host)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Nonlinear:
    fun: Callable[[Any], Any]  # x ↦ 1-D array in the space
    jac: Callable[[Any], Any] | None  # x ↦ 2-D array in the space, or scipy.sparse
    hess: Callable[[Any, Any], Any] | None
    lb: Any  # host float64
    ub: Any
    lb_dev: Any  # in the space
    ub_dev: Any


@dataclass(frozen=True)
class _Linear:
    A: Any  # host: dense float64 ndarray (LinearConstraint's cast) or scipy CSR
    lb: Any  # host float64
    ub: Any
    lb_dev: Any
    ub_dev: Any
    matvec: Callable[[Any], Any]  # x ↦ A x in the space


def _host_bounds(value: Any, m: int) -> Any:
    return np.broadcast_to(np.asarray(value, dtype=np.float64), (m,)).copy()


def _warn_keep_feasible(con: Any) -> None:
    if np.any(np.asarray(getattr(con, "keep_feasible", False))):
        warnings.warn(
            "keep_feasible is not supported for constraints: the interior-point "
            "iterates keep only the bounds strictly feasible",
            OptimizeWarning,
            # → _from_*/_normalize_constraints/_translate/minimize_ipax → caller
            stacklevel=6,
        )


def _as_vector(space: _Space, raw: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def fun(x: Any) -> Any:
        return space.xp.reshape(space.asarray(raw(x)), (-1,))

    return fun


def _as_jacobian(space: _Space, value: Any) -> Any:
    if sp.issparse(value):
        return sp.csr_array(value)
    if isinstance(value, ScipyLinearOperator):
        raise TypeError(
            "LinearOperator constraint Jacobians are not supported; return a "
            "dense array or a scipy.sparse matrix"
        )
    jac = space.asarray(value)
    return space.xp.reshape(jac, (1, -1)) if jac.ndim < 2 else jac


def _nonlinear(
    space: _Space, fun: Any, jac: Any, hess: Any, lb: Any, ub: Any, x0: Any
) -> _Nonlinear:
    vector = _as_vector(space, fun)
    jacobian = None
    if jac is not None:

        def jacobian(x: Any) -> Any:
            return _as_jacobian(space, jac(x))

    m = int(vector(x0).shape[0])
    lb_host, ub_host = _host_bounds(lb, m), _host_bounds(ub, m)
    return _Nonlinear(
        vector,
        jacobian,
        hess,
        lb_host,
        ub_host,
        space.asarray(lb_host),
        space.asarray(ub_host),
    )


def _from_dict(space: _Space, con: dict[str, Any], x0: Any) -> _Nonlinear:
    kind = con.get("type")
    if kind not in ("eq", "ineq"):
        raise ValueError(f"constraint type must be 'eq' or 'ineq', got {kind!r}")
    args = con.get("args", ())
    raw_fun, raw_jac = con["fun"], con.get("jac")
    jac = (lambda x: raw_jac(x, *args)) if callable(raw_jac) else None
    # SciPy: 'ineq' means fun(x) ≥ 0.
    ub = 0.0 if kind == "eq" else np.inf
    return _nonlinear(space, lambda x: raw_fun(x, *args), jac, None, 0.0, ub, x0)


def _from_nonlinear(space: _Space, con: Any, x0: Any) -> _Nonlinear:
    _warn_keep_feasible(con)
    jac = con.jac if callable(con.jac) else None
    hess = con.hess if callable(con.hess) else None
    return _nonlinear(space, con.fun, jac, hess, con.lb, con.ub, x0)


def _from_linear(space: _Space, con: Any, n: int) -> _Linear:
    _warn_keep_feasible(con)
    if sp.issparse(con.A):
        A: Any = sp.csr_array(con.A)
    else:
        A = np.atleast_2d(np.asarray(con.A, dtype=np.float64))
    if A.shape[1] != n:
        raise ValueError(f"LinearConstraint.A has {A.shape[1]} columns, expected {n}")
    m = int(A.shape[0])
    lb, ub = _host_bounds(con.lb, m), _host_bounds(con.ub, m)
    matvec: Callable[[Any], Any]
    if sp.issparse(A):
        matvec = space.csr_operator(A).matvec
    else:
        A_dev = space.asarray(A)
        matvec = partial(_dense_matvec, A_dev)
    return _Linear(A, lb, ub, space.asarray(lb), space.asarray(ub), matvec)


def _dense_matvec(A: Any, x: Any) -> Any:
    return A @ x


def _normalize_constraints(
    space: _Space, constraints: Any, x0: Any
) -> list[_Nonlinear | _Linear]:
    if constraints is None:
        return []
    if isinstance(constraints, dict) or not isinstance(constraints, Sequence):
        constraints = [constraints]
    out: list[_Nonlinear | _Linear] = []
    for con in constraints:
        if isinstance(con, dict):
            out.append(_from_dict(space, con, x0))
        elif hasattr(con, "A"):  # LinearConstraint (duck-typed)
            out.append(_from_linear(space, con, int(x0.shape[0])))
        elif hasattr(con, "fun"):  # NonlinearConstraint (duck-typed)
            out.append(_from_nonlinear(space, con, x0))
        else:
            raise TypeError(f"unsupported constraint type {type(con).__name__}")
    for spec in out:
        if np.any(spec.lb > spec.ub):
            raise ValueError("constraint has lb > ub")
        if np.any((spec.lb == spec.ub) & ~np.isfinite(spec.lb)):
            raise ValueError("constraint row has lb == ub == ±inf (infinite equality)")
    return out


def _normalize_bounds(space: _Space, bounds: Any, n: int) -> tuple[Any, Any]:
    if bounds is None:
        return None, None
    if hasattr(bounds, "lb") and hasattr(bounds, "ub"):  # scipy.optimize.Bounds
        try:
            lb = _host_bounds(bounds.lb, n)
            ub = _host_bounds(bounds.ub, n)
        except ValueError as exc:
            raise ValueError(f"bounds do not match the {n} variables") from exc
    else:
        pairs = list(bounds)
        if len(pairs) != n:
            raise ValueError(f"bounds must have {n} (min, max) pairs, got {len(pairs)}")
        lb = np.array([-np.inf if lo is None else lo for lo, _ in pairs], dtype=float)
        ub = np.array([np.inf if hi is None else hi for _, hi in pairs], dtype=float)
    # minimize() validates this itself only for its built-in methods.
    if np.any(lb > ub):
        raise ValueError("bounds have lb > ub")
    return (
        None if np.all(np.isneginf(lb)) else space.asarray(lb),
        None if np.all(np.isposinf(ub)) else space.asarray(ub),
    )


def _row_split(lb: Any, ub: Any) -> tuple[Any, Any, Any]:
    """Equality, finite-upper and finite-lower row indices of ``lb ≤ c ≤ ub``."""
    eq = lb == ub
    return (
        np.flatnonzero(eq),
        np.flatnonzero(~eq & np.isfinite(ub)),
        np.flatnonzero(~eq & np.isfinite(lb)),
    )


# ---------------------------------------------------------------------------
# Multipliers: one-sided rows back to one trust-constr-signed v per row
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MultiplierPlan:
    """Sync- and scatter-free map from stacked one-sided multipliers to ``v``.

    With ``w = [0, sign₁·block₁, sign₂·block₂, …]`` every source row is
    ``v = take(w, first) + take(w, second)``: an equality row or a one-sided
    row reads one entry, a row finite on both sides reads its upper and its
    lower multiplier, a free row reads the leading zero twice. (Indexed
    assignment is not guaranteed by the Array API standard; two gathers are.)
    """

    m: int
    first: Any  # device int64 indices into w
    second: Any
    signs: tuple[float, ...]
    sizes: tuple[int, ...]

    def apply(self, space: _Space, blocks: list[Any]) -> Any:
        xp = space.xp
        parts = [space.zeros((1,))]
        parts += [sign * block for sign, block in zip(self.signs, blocks, strict=True)]
        w = xp.concat(parts)
        return xp.take(w, self.first, axis=0) + xp.take(w, self.second, axis=0)


def _plan(
    space: _Space, m: int, groups: list[tuple[Any, float, str]]
) -> _MultiplierPlan:
    """``groups`` = ``(rows, sign, "first" | "second")`` in the order of ``w``."""
    first = np.zeros(m, dtype=np.int64)
    second = np.zeros(m, dtype=np.int64)
    offset = 1
    for rows, _, slot in groups:
        target = first if slot == "first" else second
        target[rows] = offset + np.arange(rows.size)
        offset += rows.size
    return _MultiplierPlan(
        m,
        space.index(first),
        space.index(second),
        tuple(sign for _, sign, _ in groups),
        tuple(int(rows.size) for rows, _, _ in groups),
    )


@dataclass(frozen=True)
class _Layout:
    """Row structure of the translated problem (fixed by ``lb`` / ``ub``)."""

    # Nonlinear block: stacked rows split into equality / upper / lower rows.
    nl: tuple[Any, Any, Any]
    nl_dev: tuple[Any, Any, Any]
    nl_plan: _MultiplierPlan
    # Linear block: equality rows, then the lowered rows of the rest
    # (finite-lower rows in source order, then finite-upper rows; see
    # Problem.linear_ineq), which is ipax's Result.y_ineq layout.
    lin_eq: Any
    lin_ineq: Any
    lin_plan: _MultiplierPlan

    @property
    def n_eq_nl(self) -> int:
        return int(self.nl[0].size)

    @property
    def n_ineq_nl(self) -> int:
        return int(self.nl[1].size + self.nl[2].size)


def _layout(space: _Space, specs: list[_Nonlinear | _Linear]) -> _Layout:
    nonlinear = [s for s in specs if isinstance(s, _Nonlinear)]
    linear = [s for s in specs if isinstance(s, _Linear)]
    empty = np.zeros(0)
    lb_nl = np.concatenate([s.lb for s in nonlinear]) if nonlinear else empty
    ub_nl = np.concatenate([s.ub for s in nonlinear]) if nonlinear else empty
    eq, up, lo = _row_split(lb_nl, ub_nl)
    # ipax's W = σ∇²f + Σ y_eq·∇²c + Σ y_ineq·∇²g: the split g = c − ub carries
    # +∇c and g = lb − c carries −∇c, so v = y (eq), λ_upper − λ_lower (ineq).
    nl_plan = _plan(
        space,
        int(lb_nl.shape[0]),
        [(eq, 1.0, "first"), (up, 1.0, "first"), (lo, -1.0, "second")],
    )
    lb_lin = np.concatenate([s.lb for s in linear]) if linear else empty
    ub_lin = np.concatenate([s.ub for s in linear]) if linear else empty
    lin_eq = np.flatnonzero(lb_lin == ub_lin)
    lin_ineq = np.flatnonzero(lb_lin != ub_lin)
    lower = lin_ineq[np.isfinite(lb_lin[lin_ineq])]
    upper = lin_ineq[np.isfinite(ub_lin[lin_ineq])]
    lin_plan = _plan(
        space,
        int(lb_lin.shape[0]),
        [(lin_eq, 1.0, "first"), (lower, -1.0, "first"), (upper, 1.0, "second")],
    )
    return _Layout(
        (eq, up, lo),
        (space.index(eq), space.index(up), space.index(lo)),
        nl_plan,
        lin_eq,
        lin_ineq,
        lin_plan,
    )


def _split_blocks(values: Any, sizes: tuple[int, ...]) -> list[Any]:
    out, start = [], 0
    for size in sizes:
        out.append(values[start : start + size])
        start += size
    return out


# ---------------------------------------------------------------------------
# Hessian terms
# ---------------------------------------------------------------------------


def _as_hessian(space: _Space, value: Any) -> Any:
    """Keep sparse matrices and operators; any other array-like (SciPy accepts
    e.g. nested lists) becomes a dense array in the space."""
    if (
        sp.issparse(value)
        or isinstance(value, ScipyLinearOperator)
        or isinstance(value, LinearOperator)
    ):
        return value
    return space.asarray(value)


def _scale(term: Any, sigma: Any) -> Any:
    if sp.issparse(term) or isinstance(term, ScipyLinearOperator):
        return float(sigma) * term
    if isinstance(term, LinearOperator):  # ipax operators have no scalar product
        op = term

        def scaled(p: Any) -> Any:
            return sigma * op.matvec(p)

        return MatrixFreeJacobian(op.shape, scaled, scaled)
    return sigma * term


def _sum_hessians(space: _Space, terms: list[Any], n: int) -> Any:
    """Sum dense / sparse / operator Hessian terms into what ipax receives."""
    if not terms:
        return space.zeros((n, n))
    if any(isinstance(t, ScipyLinearOperator | LinearOperator) for t in terms):
        # Sparse terms become device operators, so the products stay on device
        # (a SciPy LinearOperator term still needs NumPy p).
        ops = [space.csr_operator(t) if sp.issparse(t) else t for t in terms]

        def apply(t: Any, p: Any) -> Any:
            return t.matvec(p) if isinstance(t, LinearOperator) else t @ p

        def matvec(p: Any) -> Any:
            total = apply(ops[0], p)
            for t in ops[1:]:
                total = total + apply(t, p)
            return total

        return MatrixFreeJacobian((n, n), matvec, matvec)
    if all(sp.issparse(t) for t in terms):
        return space.csr_operator(sum(terms[1:], terms[0]), symmetric=True)
    dense = [space.asarray(t.toarray()) if sp.issparse(t) else t for t in terms]
    total = dense[0]
    for t in dense[1:]:
        total = total + t
    return total


# ---------------------------------------------------------------------------
# Translation onto FunctionProblem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Translation:
    """The translated problem plus what is needed to map the result back."""

    problem: FunctionProblem
    specs: tuple[_Nonlinear | _Linear, ...]
    counts: _Counts
    gradient: Callable[[Any], Any] | None
    space: _Space
    layout: _Layout


def _objective_callables(
    space: _Space,
    fun: Callable[..., Any],
    jac: Any,
    args: tuple[Any, ...],
    counts: _Counts,
) -> tuple[Callable[[Any], Any], Callable[[Any], Any] | None]:
    xp = space.xp

    def value_of(raw: Any) -> Any:
        return xp.reshape(space.asarray(raw), ())

    def grad_of(raw: Any, x: Any) -> Any:
        return xp.reshape(space.asarray(raw), x.shape)

    if jac is True:  # fun returns (f, g); minimize itself pre-wraps this case.

        def both(x: Any) -> Any:
            counts.nfev += 1
            return fun(x, *args)

        cached = _LastPoint(both)

        def shared_objective(x: Any) -> Any:
            return value_of(cached(x)[0])

        def shared_gradient(x: Any) -> Any:
            counts.njev += 1
            return grad_of(cached(x)[1], x)

        return shared_objective, shared_gradient

    def objective(x: Any) -> Any:
        counts.nfev += 1
        return value_of(fun(x, *args))

    if not callable(jac):  # None or a finite-difference string → ipax fallback
        return objective, None

    def gradient(x: Any) -> Any:
        counts.njev += 1
        return grad_of(jac(x, *args), x)

    return objective, gradient


def _objective_hessian(
    space: _Space, hess: Any, hessp: Any, args: tuple[Any, ...], n: int, counts: _Counts
) -> Callable[[Any, Any], Any] | None:
    """``(x, σ) ↦ σ∇²f(x)`` as a matrix or operator, or ``None`` if unavailable."""
    if callable(hess):

        def scaled(x: Any, sigma: Any) -> Any:
            counts.nhev += 1
            return _scale(_as_hessian(space, hess(x, *args)), sigma)

        return scaled
    if hess is not None:
        if isinstance(hess, HessianUpdateStrategy) or hess in _FD_STRINGS:
            warnings.warn(
                f"hess={hess!r} is replaced by ipax's L-BFGS approximation",
                OptimizeWarning,
                stacklevel=4,
            )
            return None
        raise TypeError(f"unsupported hess {hess!r}")
    if callable(hessp):

        def operator(x: Any, sigma: Any) -> Any:
            counts.nhev += 1
            x_fixed = x  # iterates are never updated in place

            def matvec(p: Any) -> Any:
                return sigma * space.asarray(hessp(x_fixed, p, *args))

            return MatrixFreeJacobian((n, n), matvec, matvec)

        return operator
    return None


def _translate(
    fun: Callable[..., Any],
    x0: Any,
    args: tuple[Any, ...] = (),
    *,
    jac: Any = None,
    hess: Any = None,
    hessp: Any = None,
    bounds: Any = None,
    constraints: Any = (),
) -> _Translation:
    """Translate a ``minimize``-style call onto a :class:`FunctionProblem`."""
    xp = array_namespace(x0)
    space = _Space(xp, device(x0), x0.dtype, type(x0))
    n = int(x0.shape[0])
    counts = _Counts()
    objective, gradient = _objective_callables(space, fun, jac, args, counts)
    lower, upper = _normalize_bounds(space, bounds, n)
    specs = _normalize_constraints(space, constraints, x0)
    layout = _layout(space, specs)
    nonlinear = [s for s in specs if isinstance(s, _Nonlinear)]
    linear = [s for s in specs if isinstance(s, _Linear)]

    kwargs: dict[str, Any] = {}

    # ---- nonlinear block: one stacked c(x), split by row kind ----
    if nonlinear:
        lb = xp.concat([s.lb_dev for s in nonlinear])
        ub = xp.concat([s.ub_dev for s in nonlinear])
        (eq, up, lo), (eq_d, up_d, lo_d) = layout.nl, layout.nl_dev
        values = _LastPoint(lambda x: xp.concat([s.fun(x) for s in nonlinear]))
        # No Jacobian anywhere → ipax's own fallback (honestly reported as
        # finite-diff). Only some missing → difference just those, so the
        # supplied Jacobians are not thrown away.
        has_jac = any(s.jac is not None for s in nonlinear)
        if has_jac and not all(s.jac is not None for s in nonlinear):
            warnings.warn(
                "some constraints have no callable jac; using central differences "
                "for those (supply jac for accuracy and speed)",
                OptimizeWarning,
                stacklevel=3,
            )
        pieces = [
            s.jac if s.jac is not None else partial(jacobian_fd, s.fun)
            for s in nonlinear
        ]
        jacobian = _LastPoint(lambda x: space.vstack([p(x) for p in pieces]))

        if eq.size:
            lb_eq = xp.take(lb, eq_d)
            kwargs["eq_constraints"] = lambda x: xp.take(values(x), eq_d) - lb_eq
            if has_jac:
                kwargs["eq_jacobian"] = lambda x: space.matrix(
                    space.rows(jacobian(x), eq, eq_d)
                )
        if up.size or lo.size:
            ub_up, lb_lo = xp.take(ub, up_d), xp.take(lb, lo_d)
            kwargs["ineq_constraints"] = lambda x: xp.concat(
                [xp.take(values(x), up_d) - ub_up, lb_lo - xp.take(values(x), lo_d)]
            )
            if has_jac:
                kwargs["ineq_jacobian"] = lambda x: space.matrix(
                    space.vstack(
                        [
                            space.rows(jacobian(x), up, up_d),
                            -space.rows(jacobian(x), lo, lo_d),
                        ]
                    )
                )

    # ---- linear block: constant data, equalities split off ----
    if linear:
        A = (
            sp.vstack([sp.csr_array(s.A) for s in linear], format="csr")
            if any(sp.issparse(s.A) for s in linear)
            else np.vstack([s.A for s in linear])
        )
        lb_lin = np.concatenate([s.lb for s in linear])
        ub_lin = np.concatenate([s.ub for s in linear])

        def data(rows: Any) -> Any:
            block = A[rows]
            return (
                space.csr_operator(block)
                if sp.issparse(block)
                else space.asarray(block)
            )

        if layout.lin_eq.size:
            rows = layout.lin_eq
            kwargs["linear_eq"] = (data(rows), space.asarray(lb_lin[rows]))
        if layout.lin_ineq.size:
            rows = layout.lin_ineq
            kwargs["linear_ineq"] = (
                data(rows),
                space.asarray(lb_lin[rows]),
                space.asarray(ub_lin[rows]),
            )

    # ---- exact Lagrangian Hessian, when every piece is available ----
    objective_hessian = _objective_hessian(space, hess, hessp, args, n, counts)
    if objective_hessian is not None:
        if all(s.hess is not None for s in nonlinear):
            kwargs["lagrangian_hessian"] = _lagrangian_hessian(
                space, objective_hessian, nonlinear, layout, n
            )
        else:
            warnings.warn(
                "a constraint has no callable hess(x, v) (dict constraints never "
                "do); using ipax's L-BFGS approximation of the whole Lagrangian "
                "Hessian",
                OptimizeWarning,
                stacklevel=3,
            )

    problem = FunctionProblem(
        n,
        objective,
        gradient=gradient,
        bounds=(lower, upper),
        **kwargs,
    )
    return _Translation(problem, tuple(specs), counts, gradient, space, layout)


def _nonlinear_multipliers(
    space: _Space, layout: _Layout, y_eq: Any, y_ineq: Any
) -> Any:
    """trust-constr-signed ``v`` per stacked nonlinear row.

    ``y_eq`` / ``y_ineq`` start with the nonlinear rows (equality rows, then
    upper rows, then lower rows, as stacked by :func:`_translate`).
    """
    n_eq, n_up, n_lo = (int(r.size) for r in layout.nl)
    blocks = [y_eq[:n_eq], y_ineq[:n_up], y_ineq[n_up : n_up + n_lo]]
    return layout.nl_plan.apply(space, blocks)


def _lagrangian_hessian(
    space: _Space,
    objective_hessian: Callable[[Any, Any], Any],
    nonlinear: list[_Nonlinear],
    layout: _Layout,
    n: int,
) -> Callable[[Any, Any, Any, Any], Any]:
    offsets = np.cumsum([0] + [int(s.lb.shape[0]) for s in nonlinear]).tolist()

    def lagrangian_hessian(x: Any, y_eq: Any, y_ineq: Any, sigma: Any) -> Any:
        # The driver passes only the nonlinear multipliers, so y maps back onto
        # the stacked rows exactly as in the result.
        v = _nonlinear_multipliers(space, layout, y_eq, y_ineq)
        skip = isinstance(sigma, int | float) and sigma == 0.0
        terms = [] if skip else [objective_hessian(x, sigma)]
        for k, spec in enumerate(nonlinear):
            assert spec.hess is not None
            terms.append(
                _as_hessian(space, spec.hess(x, v[offsets[k] : offsets[k + 1]]))
            )
        return space.matrix(_sum_hessians(space, terms, n))

    return lagrangian_hessian


# ---------------------------------------------------------------------------
# Options, callback, result
# ---------------------------------------------------------------------------


def _build_options(tol: float | None, options: dict[str, Any]) -> Options:
    opts = dict(options)
    base = opts.pop("ipax_options", None)
    if base is None:
        base = Options()
    elif not isinstance(base, Options):
        raise TypeError("options['ipax_options'] must be an ipax.Options instance")
    option_tol = opts.pop("tol", None)
    tol = option_tol if tol is None else tol
    changes = {key: opts.pop(key) for key in list(opts) if key in _OPTION_FIELDS}
    if "maxiter" in opts:
        changes["max_iter"] = int(opts.pop("maxiter"))
    if "disp" in opts:
        disp = opts.pop("disp")
        if disp:
            changes["verbose"] = 1 if disp is True else int(disp)
    if tol is not None:
        changes["optimality"] = dataclasses.replace(
            changes.get("optimality", base.optimality),
            dual_inf_tol=tol,
            constr_viol_tol=tol,
            compl_inf_tol=tol,
        )
    if opts:
        warnings.warn(
            f"Unknown solver options: {', '.join(sorted(opts))}",
            OptimizeWarning,
            stacklevel=3,
        )
    return dataclasses.replace(base, **changes)


def _wrap_callback(
    callback: Callable[..., Any] | None, xp: Any
) -> Callable[[IterationInfo], bool] | None:
    if callback is None:
        return None
    # SciPy's styles: callback(intermediate_result) (by that exact name),
    # trust-constr's callback(xk, state) — recognized by a *required* second
    # positional parameter, so callback(xk, history=[]) stays legacy — and the
    # legacy callback(xk).
    try:
        params = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        params = MappingProxyType({})
    new_style = set(params) == {"intermediate_result"}
    positional = [
        p
        for p in params.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    with_state = (
        not new_style
        and len(positional) >= 2
        and positional[1].default is inspect.Parameter.empty
    )

    def ipax_callback(info: IterationInfo) -> bool:
        x = xp.asarray(info.x, copy=True)
        state = OptimizeResult(
            x=x, fun=info.record.objective, nit=info.record.iteration
        )
        try:
            if new_style:
                callback(intermediate_result=state)
            elif with_state:
                # trust-constr ends the solve when this form returns True; the
                # other forms' return values are ignored, as in SciPy.
                return bool(callback(x, state))
            else:
                callback(x)
        except StopIteration:  # SciPy's way of ending a solve from a callback
            return True
        return False

    return ipax_callback


def _padded(space: _Space, values: Any, length: int) -> Any:
    """``values[:length]`` zero-padded to ``length``; zeros when ``None``."""
    if values is None:
        return space.zeros((length,))
    head = values[:length]
    missing = length - int(head.shape[0])
    return space.xp.concat([head, space.zeros((missing,))]) if missing else head


def _per_constraint_multipliers(result: Result, translation: _Translation) -> list[Any]:
    space, layout = translation.space, translation.layout
    # Result layout: y_eq = [nonlinear eq | linear eq]; y_ineq = [nonlinear
    # upper | nonlinear lower | lowered linear rows].
    n_eq_nl, n_ineq_nl = layout.n_eq_nl, layout.n_ineq_nl
    lin_sizes = layout.lin_plan.sizes  # (eq, lowered lower, lowered upper)
    # An exit before the loop (e.g. infeasible start data) carries no
    # multipliers; report zeros of the right length.
    y_eq = _padded(space, result.y_eq, n_eq_nl + lin_sizes[0])
    y_ineq = _padded(space, result.y_ineq, n_ineq_nl + lin_sizes[1] + lin_sizes[2])

    v_nl = _nonlinear_multipliers(space, layout, y_eq, y_ineq)
    lowered = y_ineq[n_ineq_nl:]
    v_lin = layout.lin_plan.apply(
        space, [y_eq[n_eq_nl:], *_split_blocks(lowered, lin_sizes[1:])]
    )

    out: list[Any] = []
    i_nl = i_lin = 0
    for spec in translation.specs:
        m = int(spec.lb.shape[0])
        if isinstance(spec, _Nonlinear):
            out.append(v_nl[i_nl : i_nl + m])
            i_nl += m
        else:
            out.append(v_lin[i_lin : i_lin + m])
            i_lin += m
    return out


def _to_optimize_result(result: Result, translation: _Translation) -> OptimizeResult:
    space = translation.space
    n = translation.problem.n_vars
    xp = space.xp
    x = result.x
    constr = [
        spec.fun(x) if isinstance(spec, _Nonlinear) else spec.matvec(x)
        for spec in translation.specs
    ]
    violation = 0.0
    for spec, c in zip(translation.specs, constr, strict=True):
        if c.shape[0]:
            excess = xp.max(xp.maximum(spec.lb_dev - c, c - spec.ub_dev))
            violation = max(violation, float(excess))
    lower, upper = translation.problem.bounds()
    if lower is not None:
        violation = max(violation, float(xp.max(lower - x)))
    if upper is not None:
        violation = max(violation, float(xp.max(x - upper)))

    z_lower = space.zeros((n,)) if result.z_lower is None else result.z_lower
    z_upper = space.zeros((n,)) if result.z_upper is None else result.z_upper
    # Evaluated before the counters are read, so njev includes this call.
    gradient = None if translation.gradient is None else translation.gradient(x)
    fields: dict[str, Any] = {
        "x": x,
        "fun": result.objective,
        "success": result.success,
        "status": STATUS_CODES[result.status.value],
        "message": result.message or result.status.value,
        # SciPy's nit counts iterations performed; Result.n_iter counts history
        # rows, which include the starting point (iteration 0).
        "nit": result.history[-1].iteration if result.history else 0,
        "nfev": translation.counts.nfev,
        "njev": translation.counts.njev,
        "nhev": translation.counts.nhev,
        "v": _per_constraint_multipliers(result, translation),
        "constr": constr,
        "constr_violation": max(violation, 0.0),
        "bound_multipliers": z_upper - z_lower,
        "ipax_status": result.status.value,
        "ipax_result": result,
    }
    if gradient is not None:
        fields["jac"] = gradient
    return OptimizeResult(fields)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _as_start(x0: Any) -> Any:
    """``x0`` as a 1-D floating array in its own namespace (SciPy's rules)."""
    try:
        xp = array_namespace(x0)
    except TypeError:  # a list, tuple or scalar: NumPy, as minimize does
        x0 = np.asarray(x0)
        xp = array_namespace(x0)
    if x0.ndim == 0:
        x0 = xp.reshape(x0, (1,))
    if x0.ndim != 1:
        raise ValueError("'x0' must only have one dimension.")
    if xp.isdtype(x0.dtype, ("bool", "integral")):
        x0 = xp.astype(x0, xp.float64)
    return x0


def minimize_ipax(
    fun: Callable[..., Any],
    x0: Any,
    args: Any = (),
    *,
    jac: Any = None,
    hess: Any = None,
    hessp: Any = None,
    bounds: Any = None,
    constraints: Any = (),
    tol: float | None = None,
    callback: Callable[..., Any] | None = None,
    options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> OptimizeResult:
    """Minimize ``fun`` with ipax using the ``scipy.optimize.minimize`` signature.

    Usable directly or as ``scipy.optimize.minimize(..., method=minimize_ipax)``
    (``minimize`` forwards its ``options`` as keyword arguments).

    Parameters
    ----------
    fun, x0, args
        As in :func:`scipy.optimize.minimize`: ``fun(x, *args) -> float`` and a
        one-dimensional starting point (integer input is promoted to float).
        ``x0`` may be any Array-API array; the solve and every array in the
        result use its namespace, device and dtype (a list or scalar is taken
        as NumPy, as SciPy does). See the module notes.
    jac
        Callable gradient ``jac(x, *args)``, ``True`` if ``fun`` returns
        ``(f, g)``, or ``None`` / a finite-difference string, in which case
        ipax derives the gradient (autodiff on backends that have it,
        finite differences otherwise).
    hess, hessp
        Callable objective Hessian ``hess(x, *args)`` (dense, sparse, or a
        ``scipy.sparse.linalg.LinearOperator``) or Hessian-vector product
        ``hessp(x, p, *args)``. Used only together with a callable
        ``hess(x, v)`` on every ``NonlinearConstraint``; otherwise ipax's
        L-BFGS approximation is used (see the module notes).
    bounds
        A :class:`scipy.optimize.Bounds` or a sequence of ``(min, max)`` pairs
        (``None`` for unbounded). Interior-point iterates always stay strictly
        inside the bounds, so ``keep_feasible`` is implied.
    constraints
        A ``LinearConstraint``, ``NonlinearConstraint`` or SciPy constraint
        dict, or a sequence of them (mixed kinds allowed).
    tol
        Sets the scaled KKT tolerances (``dual_inf_tol``, ``constr_viol_tol``,
        ``compl_inf_tol``) of :class:`~ipax.options.OptimalityConditionOptions`.
    callback
        ``callback(intermediate_result: OptimizeResult)`` (detected by that
        exact parameter name), trust-constr's ``callback(xk, state)``
        (detected by a required second positional parameter), or legacy
        ``callback(xk)``, called once per iteration — starting at ``x0`` —
        with a copy of the iterate. Raise ``StopIteration`` to end the solve
        (status ``"stopped"``); the ``(xk, state)`` form may also return
        ``True``, as in trust-constr. Other return values are ignored.
    options, **kwargs
        Solver options, merged (keyword arguments win). ``maxiter`` →
        ``Options.max_iter``; ``disp`` → ``Options.verbose``; ``tol`` as above;
        ``ipax_options`` an :class:`~ipax.Options` to start from; and any
        :class:`~ipax.Options` field name (e.g. ``globalization="breedveld"``)
        overrides that field. Unknown keys raise an ``OptimizeWarning``.

    Returns
    -------
    scipy.optimize.OptimizeResult
        ``x, fun, jac`` (when a gradient callable was given), ``success``,
        ``status`` (see :data:`STATUS_CODES`), ``message``, ``nit``, ``nfev``,
        ``njev``, ``nhev``; ``v`` and ``constr`` (per input constraint, in
        input order; trust-constr sign convention), ``constr_violation`` (max
        violation in the user's units, bounds included), ``bound_multipliers``
        (``z_U − z_L``), ``ipax_status`` (the :class:`ipax.Status` value), and
        ``ipax_result`` (the full :class:`ipax.Result`).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import minimize, rosen, rosen_der
    >>> from ipax.interfaces.scipy import minimize_ipax
    >>> res = minimize(rosen, np.array([-1.2, 1.0]), jac=rosen_der,
    ...                method=minimize_ipax)
    >>> bool(res.success), np.round(res.x, 6).tolist()
    (True, [1.0, 1.0])
    """
    x0 = _as_start(x0)
    if not isinstance(args, tuple):
        args = (args,)
    ipax_options = _build_options(tol, {**(options or {}), **kwargs})

    translation = _translate(
        fun,
        x0,
        args,
        jac=jac,
        hess=hess,
        hessp=hessp,
        bounds=bounds,
        constraints=constraints,
    )
    result = solve(
        translation.problem,
        x0,
        options=ipax_options,
        callback=_wrap_callback(callback, translation.space.xp),
    )
    return _to_optimize_result(result, translation)


__all__ = ["STATUS_CODES", "minimize_ipax"]
