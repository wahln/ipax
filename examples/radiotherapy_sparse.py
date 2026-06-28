"""Radiotherapy-like example: sparse dose-influence least squares.

A stripped-down fluence-map optimization (FMO) problem, the canonical workload
``ipax`` is tuned for (Breedveld et al. 2017):

- a sparse **dose-influence matrix** ``D`` maps beamlet weights ``x`` (the
  optimization variable) to a per-voxel dose ``d = D x``. ``D`` has ``V`` rows
  (voxels) and ``B`` columns (beamlets) and is very sparse — each beamlet only
  irradiates a small tube of voxels;
- the beamlet weights are physically non-negative, so ``x ≥ 0`` (upper bound
  ``+∞``);
- the objective drives the delivered dose toward a prescription ``d_pres``::

      f(x) = ½ ‖D x − d_pres‖²,    ∇f(x) = Dᵀ (D x − d_pres).

Only the objective and its gradient are supplied, so the resolver fills the
Lagrangian Hessian with Powell-damped **L-BFGS** (the true Hessian here is the
constant ``DᵀD``, never formed). The backend is NumPy and ``D`` is a SciPy CSC
matrix used at the application edge; the solver core still sees only Array-API
arrays via the ``D @ x`` / ``Dᵀ @ r`` products.

**Scale.** The clinical target is roughly ``V ≈ 1e7`` voxels and ``B ≈ 1e4``
beamlets at ~0.1 % density. The dense reference solver factorizes a ``B × B``
matrix every iteration (``O(B³)``), so this script defaults to a smaller, fast
instance; the sparse ``D`` products are already ``O(nnz)`` and backend-agnostic,
and the full beamlet count wants the matrix-free Krylov route.
Override the size with the env vars ``RT_VOXELS``, ``RT_BEAMLETS``, ``RT_DENSITY``.
"""

from __future__ import annotations

import os

FORCE_NUMPY = False
try:
    import cupy as np
    import cupyx.scipy.sparse as sp
except ImportError:
    import numpy as np
    import scipy.sparse as sp

if FORCE_NUMPY:
    import numpy as np
    import scipy.sparse as sp

import ipax  # noqa: E402

SOLVER = "sparse"
HESSIAN = "lbfgs"

# Runnable defaults. Clinical scale (see module docstring) is ~1e7 × 1e4 @ 1e-3.
N_VOXELS = int(os.environ.get("RT_VOXELS", 1_000_000))
N_BEAMLETS = int(os.environ.get("RT_BEAMLETS", 1000))
DENSITY = float(os.environ.get("RT_DENSITY", 0.001))

PRECISION = np.float64


def build_influence_matrix(
    n_voxels: int, n_beamlets: int, density: float, rng: np.random.Generator
) -> sp.csc_matrix:
    """Random non-negative sparse dose-influence matrix ``D`` (V × B, CSC)."""
    return sp.random(
        n_voxels,
        n_beamlets,
        density=density,
        format="csc",
        random_state=int(rng.integers(0, 2**31 - 1)),
        dtype=PRECISION,
    )


def main() -> None:
    rng = np.random.default_rng(0)
    D = build_influence_matrix(N_VOXELS, N_BEAMLETS, DENSITY, rng)
    print(
        f"D: {D.shape[0]} voxels x {D.shape[1]} beamlets, "
        f"nnz={D.nnz} ({100 * D.nnz / (D.shape[0] * D.shape[1]):.3g}% dense)"
    )

    # A realizable prescription: pick a ground-truth non-negative fluence and
    # let d_pres = D x_true, so the constrained optimum is x_true with f* = 0.
    # Zeroing ~40 % of the beamlets makes the x ≥ 0 bound genuinely active at
    # the optimum (those weights sit on the boundary).
    x_true = rng.random(N_BEAMLETS, dtype=PRECISION)
    x_true[rng.random(N_BEAMLETS, dtype=PRECISION) < 0.4] = 0.0
    d_pres = D @ x_true

    def objective(x: np.ndarray) -> np.ndarray:
        residual = D @ x - d_pres
        return 0.5 * np.dot(residual, residual)

    def gradient(x: np.ndarray) -> np.ndarray:
        return D.T @ (D @ x - d_pres)

    problem = ipax.FunctionProblem(
        N_BEAMLETS,
        objective,
        gradient=gradient,
        bounds=(np.zeros(N_BEAMLETS, dtype=PRECISION), None),  # x ≥ 0
    )

    x0 = np.ones(N_BEAMLETS, dtype=PRECISION)
    result = ipax.solve(
        problem,
        x0,
        options=ipax.Options(
            hessian=HESSIAN,
            linsolve=SOLVER,
            mu_schedule="adaptive",
            optimality=ipax.OptimalityConditionOptions(
                dual_inf_tol=1e-6, constr_viol_tol=1e-6, compl_inf_tol=1e-6
            ),
            max_iter=500,
            verbose=2,
            corrections=ipax.CorrectionsOptions(
                method="gondzio", gondzio_max_corrections=3
            ),
        ),
    )

    dose = D @ result.x
    print(
        "dose RMSE vs prescription: "
        f"{float(np.sqrt(np.mean((dose - d_pres) ** 2))):.3e}"
    )


if __name__ == "__main__":
    main()
