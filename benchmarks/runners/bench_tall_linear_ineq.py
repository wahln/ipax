"""asv benchmarks: many linear inequality rows on the tall condensed path.

Adopters with large sparse linear inequality sets (``m ≫ n``) declared through
:meth:`ipax.Problem.linear_ineq` land on the condensed normal-equations system
``N = W + AᵀΣA + Σ_x`` (Breedveld 2017, eq. (18)). ``linsolve="auto"`` picks
the route for it in ``ipax/linalg/solver.py``: the fill-certified sparse
normal-equations route (``m ≥ 10n`` and Gram fill ≤ 1%, probed only below 5%
row density), the dense condensed route (``n < 10 000``, or tall with dense-ish
rows below ``n = 20 000``), and Krylov otherwise. :class:`TallLinearIneq`
straddles those boundaries with the seeded :class:`~benchmarks.generators.TallSparseQP`
and measures each cell under both declarations — the public ``linear_ineq``
path and the hand-lowered operator-Jacobian path — so a regression in the
lowering shows up as a gap between otherwise identical cells.

``track_route`` reports the selected route as a small integer (see
:data:`ROUTE_CODES`), so a routing-boundary change is a tracked value flip.

The Hessian is L-BFGS (``hessian="lbfgs"``), the project's large-scale default:
the generator also supplies an analytic Hessian, and with it the auto heuristic
never offers the sparse normal-equations route (it folds only a
diagonal-plus-low-rank ``W``), so the stock ``hessian="auto"`` would hide the
boundary this suite exists to track. Everything else is stock options.

The grid is the explicit :data:`TRACKED_CELLS` list (asv expands ``params``
into a Cartesian product; ``setup`` skips every other combination). It is
pruned so the whole file runs in well under 15 minutes on a laptop CPU.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, ClassVar

import numpy as np

import ipax
from benchmarks.generators import TallSparseQP, make_tall_sparse_problem

if TYPE_CHECKING:
    from ipax.result import Routes

DECLARATIONS: tuple[str, ...] = ("operator", "linear_ineq")

# Route codes for ``track_route``, keyed by (solver family, KKT form). The
# family is the leading token of ``Result.routes.linear_solver`` (everything
# before the first space or bracket, so markers such as ``dense (pd-hint->lu)``
# or the dispatched sparse engine do not change the code); the KKT form is
# ``Result.routes.kkt_form``. Codes are append-only: never renumber, or the
# tracked history becomes unreadable.
ROUTE_CODES: dict[tuple[str, str], int] = {
    ("dense", "condensed"): 1,  # dense condensed normal equations (Cholesky/LU)
    ("dense", "augmented"): 2,  # dense indefinite augmented (pivoted LDLᵀ)
    ("krylov", "condensed"): 3,  # matrix-free condensed (CG/MINRES)
    ("sparse", "augmented"): 4,  # sparse-direct augmented (bordered LDLᵀ)
    ("sparse-NE", "normal_equations"): 5,  # sparse-direct condensed Gram
}

# Two-sided (range) row fraction for every cell: exercises the range-row
# lowering without changing which rows are active at the solution.
TWO_SIDED = 0.1
# Banded cells: row windows of ``BANDED·n`` columns keep the Gram fill near
# ``2·BANDED`` (≈ 0.4%), comfortably under the 1% sparse-NE gate.
BANDED = 0.002


def route_code_of(linear_solver: str, kkt_form: str) -> int:
    """Decode a ``(Routes.linear_solver, Routes.kkt_form)`` pair; raise if unmapped."""
    family = linear_solver.split(" ", 1)[0].split("[", 1)[0]
    try:
        return ROUTE_CODES[(family, kkt_form)]
    except KeyError:
        raise ValueError(
            f"unmapped route {linear_solver!r} / kkt_form={kkt_form!r}: "
            "add it to ROUTE_CODES (append a new code, never renumber)"
        ) from None


def route_code(routes: Routes | None) -> int:
    """The :data:`ROUTE_CODES` value for a finished solve's ``Result.routes``."""
    if routes is None:
        raise ValueError("unmapped route: Result.routes is None (no route resolved)")
    return route_code_of(routes.linear_solver, routes.kkt_form)


# The tracked cells, as (n_vars, row_excess, density, bandwidth); both
# declarations run for each. asv takes the Cartesian product of ``params``;
# every other combination is skipped in ``setup`` (asv's NotImplementedError
# convention). Expected routes (see ``ROUTE_CODES``) and why each cell is here:
TRACKED_CELLS: tuple[tuple[int, int, float, float | None], ...] = (
    # n < 10 000, random rows: the fill probe runs (density < 5%), finds a
    # filling Gram, and the small-n dense rule takes it → 1 (dense).
    (500, 10, 0.01, None),
    # Dense-ish rows (≥ 5%): the fill probe is skipped entirely → 1 (dense).
    (500, 10, 0.2, None),
    # The fill-gate straddle at a fixed shape: the same n, m, density with
    # random rows (Gram fill ≈ 0.85 → 1, dense) and banded rows
    # (fill ≈ 0.4% → 5, sparse-NE).
    (2_000, 10, 0.01, None),
    (2_000, 10, 0.01, BANDED),
    # m/n = 100 on the sparse-NE route (200 000 rows) → 5.
    (2_000, 100, 0.01, BANDED),
    # Past the small-n dense cutoff (n ≥ 10 000) only the fill gate keeps
    # sparse rows off Krylov → 5; a gate regression flips it to 3 (and its
    # time to the timeout). Random-row cells at this n *are* Krylov cells and
    # run for minutes (late-IPM CG counts), so none is tracked.
    (10_000, 10, 0.01, BANDED),
)

_OPTIONS = ipax.Options(hessian="lbfgs", linsolve="auto")


def _problem_size(n_vars: int, row_excess: int) -> tuple[int, int]:
    """``(n, m)`` for a cell (a seam so the smoke test can shrink a cell)."""
    return n_vars, row_excess * n_vars


def skip_reason(
    n_vars: int,
    row_excess: int,
    density: float,
    bandwidth: float | None,
    declaration: str,
) -> str | None:
    """Why a grid cell is not run, or ``None`` when it is measured."""
    if declaration not in DECLARATIONS:
        return f"unknown declaration {declaration!r}"
    if (n_vars, row_excess, density, bandwidth) not in TRACKED_CELLS:
        return "not a tracked cell (see TRACKED_CELLS)"
    return None


def build_cell(
    n_vars: int,
    row_excess: int,
    density: float,
    bandwidth: float | None,
    declaration: str,
) -> tuple[TallSparseQP, np.ndarray]:
    """The seeded problem and cold start for one cell."""
    n, m = _problem_size(n_vars, row_excess)
    problem = make_tall_sparse_problem(
        n,
        m,
        density=density,
        seed=0,
        bandwidth=bandwidth,
        declaration=declaration,  # type: ignore[arg-type]
        two_sided=TWO_SIDED,
    )
    return problem, np.full(n, 0.5)


def solve_cell(*cell: object) -> dict[str, object]:
    """Solve one cell and keep the tracked scalars (picklable, for setup_cache)."""
    problem, x0 = build_cell(*cell)  # type: ignore[arg-type]
    result = ipax.solve(problem, x0, options=_OPTIONS)
    routes = result.routes
    return {
        "success": 1.0 if result.success else 0.0,
        "n_iter": int(result.n_iter),
        "kkt_error": float(result.kkt_error),
        "linear_solver": None if routes is None else routes.linear_solver,
        "kkt_form": None if routes is None else routes.kkt_form,
    }


def tracked_cells() -> list[tuple[object, ...]]:
    """Every measured (param tuple) cell, in ``params`` order."""
    return [
        cell
        for cell in itertools.product(*_PARAMS)
        if skip_reason(*cell) is None  # type: ignore[arg-type]
    ]


_PARAMS: list[list] = [
    [500, 2_000, 10_000],  # n_vars
    [10, 100],  # row excess m/n
    [0.01, 0.2],  # row density (nnz per row / n)
    [None, BANDED],  # bandwidth (None → random columns, the Gram fills)
    list(DECLARATIONS),
]
_PARAM_NAMES = ["n_vars", "row_excess", "density", "bandwidth", "declaration"]


class TallLinearIneq:
    """Wall time and peak memory of one full solve per tracked cell."""

    params: ClassVar = _PARAMS
    param_names: ClassVar = _PARAM_NAMES
    # One solve per sample: cells run seconds to a minute, so asv's default
    # repeat/rounds would multiply the suite's wall time for no gain.
    number = 1
    repeat = 1
    rounds = 1
    warmup_time = 0.0
    timeout = 300.0

    def setup(self, *cell: object) -> None:
        reason = skip_reason(*cell)  # type: ignore[arg-type]
        if reason is not None:
            raise NotImplementedError(reason)
        self.problem, self.x0 = build_cell(*cell)  # type: ignore[arg-type]

    def time_solve(self, *cell: object) -> None:
        ipax.solve(self.problem, self.x0, options=_OPTIONS)

    def peakmem_solve(self, *cell: object) -> None:
        ipax.solve(self.problem, self.x0, options=_OPTIONS)


class TallLinearIneqTrack:
    """Success, iterations, KKT residual and route code per tracked cell.

    The four trackers share one solve per cell through ``setup_cache`` (run
    once per class, results passed to every method as ``cache``) instead of
    four solves each; a separate class so a failure here never costs the
    timings above.
    """

    params: ClassVar = _PARAMS
    param_names: ClassVar = _PARAM_NAMES
    timeout = 300.0

    def setup_cache(self) -> dict[tuple[object, ...], dict[str, object]]:
        return {cell: solve_cell(*cell) for cell in tracked_cells()}

    setup_cache.timeout = 1_200.0  # type: ignore[attr-defined]

    def setup(self, cache: dict, *cell: object) -> None:
        reason = skip_reason(*cell)  # type: ignore[arg-type]
        if reason is not None:
            raise NotImplementedError(reason)
        self.summary = cache[cell]

    def track_success(self, cache: dict, *cell: object) -> float:
        return float(self.summary["success"])

    def track_iterations(self, cache: dict, *cell: object) -> int:
        return int(self.summary["n_iter"])

    def track_kkt_residual(self, cache: dict, *cell: object) -> float:
        return float(self.summary["kkt_error"])

    def track_route(self, cache: dict, *cell: object) -> int:
        linear_solver = self.summary["linear_solver"]
        if linear_solver is None:
            raise ValueError(
                "unmapped route: Result.routes is None (no route resolved)"
            )
        return route_code_of(str(linear_solver), str(self.summary["kkt_form"]))

    track_route.unit = "route code"  # type: ignore[attr-defined]
