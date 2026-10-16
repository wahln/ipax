"""Benchmark-side instrumentation of CuPy's numerical-failure certificates.

CuPy and cuDSS report numerical failure *silently*, so ipax certifies their
results itself (``fix(cupy): classify CuPy's silent linear-algebra failures``):

- the cuDSS adapter (:class:`ipax.backend.sparse.cupy.CuDSSSparseSolver`)
  reads ``NPIVOTS`` once per factorization and, when cuDSS perturbed pivots,
  certifies every solve by its relative residual
  (``_PERTURBED_SOLVE_MAX_RESIDUAL``), raising ``LinearSolveError`` otherwise;
- the dense solver runs a pivot-checked Cholesky
  (:func:`ipax.linalg.solver.checked_cholesky`) where the backend's Cholesky
  breakdown is silent.

Neither path logs, so a corpus sweep cannot see how often they fire. This
module counts them from the outside: :class:`CertificateCounter` is a context
manager that wraps those methods for the duration of one solve (restoring them
on exit) and records

- ``factorizations`` — cuDSS factorizations whose certificate was settled
  (i.e. that reached a solve), ``perturbed_factorizations`` of them with
  ``NPIVOTS > 0`` and the largest ``NPIVOTS`` seen, ``nonfinite_factorizations``
  rejected for inf/NaN entries;
- ``perturbed_solves`` certified and ``rejected_solves`` among them, with the
  certified relative residual ``max_j ‖b_j − A x_j‖∞ / ‖b_j‖∞`` of each binned
  by decade (accepted and rejected separately) plus its extreme values — the
  margin the ``1e-2`` threshold actually has on the corpus;
- ``dense_checked_cholesky`` calls and ``dense_cholesky_rejections`` (pivot
  check tripped).

The residual is recomputed here (one extra SpMV + host sync, only on perturbed
factorizations), so instrumented timings are not comparable with an
uninstrumented run; the solve itself is untouched. Nothing in ``ipax/`` is
modified: the wrappers call the original methods and re-raise their verdicts.
"""

from __future__ import annotations

import math
from typing import Any

_DECADES = range(-18, 3)  # residual-ratio bins 1e-18 … 1e2 (+ under/overflow)


def _decade(ratio: float) -> str:
    """Decade bin label for a relative residual (``"<1e-18"``, ``"1e-3"``, …)."""
    if not math.isfinite(ratio):
        return "nonfinite"
    if ratio <= 0.0:
        return "0"
    exponent = math.floor(math.log10(ratio))
    if exponent < _DECADES.start:
        return f"<1e{_DECADES.start}"
    if exponent >= _DECADES.stop:
        return f">=1e{_DECADES.stop}"
    return f"1e{exponent}"


class CertificateCounter:
    """Count cuDSS perturbed-pivot certificates and dense Cholesky pivot checks.

    Use as ``with CertificateCounter() as counter: ...`` around one solve and
    read :meth:`summary` afterwards. Patches class/module attributes, so it is
    meant for a single-threaded worker process (the sweep's layout).
    """

    def __init__(self) -> None:
        self.factorizations = 0
        self.perturbed_factorizations = 0
        self.max_npivots = 0
        self.nonfinite_factorizations = 0
        self.perturbed_solves = 0
        self.rejected_solves = 0
        self.accepted_bins: dict[str, int] = {}
        self.rejected_bins: dict[str, int] = {}
        self.max_accepted_ratio: float | None = None
        self.min_rejected_ratio: float | None = None
        self.dense_checked_cholesky = 0
        self.dense_cholesky_rejections = 0
        self._restore: list[tuple[Any, str, Any]] = []

    # -- patching -----------------------------------------------------------
    def __enter__(self) -> CertificateCounter:
        import cupy

        import ipax.linalg.dense as dense_module
        from ipax.backend.sparse.cupy import CuDSSSparseSolver
        from ipax.linalg.solver import LinearSolveError

        counter = self
        orig_certify_factor = CuDSSSparseSolver._certify_factor
        orig_certify_solve = CuDSSSparseSolver._certify_perturbed_solve
        orig_checked = dense_module.checked_cholesky

        def certify_factor(solver: Any) -> int:
            first = solver._npivots is None
            try:
                npivots = orig_certify_factor(solver)
            except LinearSolveError:
                if first:
                    counter.nonfinite_factorizations += 1
                raise
            if first:
                counter.factorizations += 1
                if npivots > 0:
                    counter.perturbed_factorizations += 1
                    counter.max_npivots = max(counter.max_npivots, int(npivots))
            return int(npivots)

        def certify_solve(solver: Any, b: Any, x: Any) -> None:
            residual = b - solver._residual_matrix @ x
            r_norm = cupy.max(cupy.abs(residual), axis=0)
            b_norm = cupy.max(cupy.abs(b), axis=0)
            ratios = cupy.where(
                b_norm > 0, r_norm / cupy.where(b_norm > 0, b_norm, 1.0), r_norm
            )
            ratio = float(cupy.max(ratios))
            if not math.isfinite(ratio) or bool(cupy.any(~cupy.isfinite(ratios))):
                ratio = math.inf
            counter.perturbed_solves += 1
            try:
                orig_certify_solve(solver, b, x)
            except LinearSolveError:
                counter.rejected_solves += 1
                label = _decade(ratio)
                counter.rejected_bins[label] = counter.rejected_bins.get(label, 0) + 1
                if (
                    counter.min_rejected_ratio is None
                    or ratio < counter.min_rejected_ratio
                ):
                    counter.min_rejected_ratio = ratio
                raise
            label = _decade(ratio)
            counter.accepted_bins[label] = counter.accepted_bins.get(label, 0) + 1
            if counter.max_accepted_ratio is None or ratio > counter.max_accepted_ratio:
                counter.max_accepted_ratio = ratio

        def checked_cholesky(xp: Any, matrix: Any) -> Any:
            counter.dense_checked_cholesky += 1
            try:
                return orig_checked(xp, matrix)
            except LinearSolveError:
                counter.dense_cholesky_rejections += 1
                raise

        for owner, name, replacement in (
            (CuDSSSparseSolver, "_certify_factor", certify_factor),
            (CuDSSSparseSolver, "_certify_perturbed_solve", certify_solve),
            (dense_module, "checked_cholesky", checked_cholesky),
        ):
            self._restore.append((owner, name, owner.__dict__[name]))
            setattr(owner, name, replacement)
        return self

    def __exit__(self, *exc: object) -> None:
        while self._restore:
            owner, name, original = self._restore.pop()
            setattr(owner, name, original)

    # -- reporting ----------------------------------------------------------
    def summary(self) -> dict[str, object]:
        """JSON-ready counts for one solve."""
        return {
            "factorizations": self.factorizations,
            "perturbed_factorizations": self.perturbed_factorizations,
            "max_npivots": self.max_npivots,
            "nonfinite_factorizations": self.nonfinite_factorizations,
            "perturbed_solves": self.perturbed_solves,
            "rejected_solves": self.rejected_solves,
            "accepted_bins": dict(sorted(self.accepted_bins.items())),
            "rejected_bins": dict(sorted(self.rejected_bins.items())),
            "max_accepted_ratio": self.max_accepted_ratio,
            "min_rejected_ratio": self.min_rejected_ratio,
            "dense_checked_cholesky": self.dense_checked_cholesky,
            "dense_cholesky_rejections": self.dense_cholesky_rejections,
        }


__all__ = ["CertificateCounter"]
