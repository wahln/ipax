"""asv macro/scaling benchmarks: full solves and n-sweeps (§9.4).

asv discovers ``time_*`` / ``peakmem_*`` / ``track_*`` methods on classes in this
directory. ``SolveScaling`` drives the matrix-free Krylov route over the
synthetic block-structured RT-like generator, so wall-clock and peak memory scale
with ``n`` **without** ever materializing an ``n×n`` KKT matrix.
"""

from __future__ import annotations

from typing import ClassVar

import ipax
from benchmarks.generators import initial_point, make_rt_like_problem


class SolveScaling:
    """Wall-clock and memory vs problem size ``n`` (matrix-free Krylov route)."""

    params: ClassVar = [1_000, 10_000, 100_000]
    param_names: ClassVar = ["n_vars"]

    def setup(self, n_vars: int) -> None:
        self.problem = make_rt_like_problem(n_vars, n_structures=8, density=0.2, seed=0)
        self.x0 = initial_point(n_vars)
        # Matrix-free Krylov with the exact (structured, matrix-free) Hessian.
        self.options = ipax.Options(hessian="exact", linsolve="krylov")

    def time_full_solve(self, n_vars: int) -> None:
        ipax.solve(self.problem, self.x0, options=self.options)

    def peakmem_full_solve(self, n_vars: int) -> None:
        ipax.solve(self.problem, self.x0, options=self.options)

    def track_success(self, n_vars: int) -> float:
        result = ipax.solve(self.problem, self.x0, options=self.options)
        return 1.0 if result.success else 0.0

    def track_iterations(self, n_vars: int) -> int:
        result = ipax.solve(self.problem, self.x0, options=self.options)
        return result.n_iter

    def track_kkt_residual(self, n_vars: int) -> float:
        result = ipax.solve(self.problem, self.x0, options=self.options)
        return result.kkt_error
