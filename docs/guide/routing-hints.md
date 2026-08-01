# Routing hints

ipax ships several opt-in levers that are **corpus-neutral as defaults but
decisive on a signature**. Each was kept off the default path because its
full-corpus A/B showed two-way churn (what fixes one problem family perturbs
another), yet on the failure pattern it was built for the win is large,
measured, and reproducible. This page is the signature → lever map; the
per-problem measurements live in the curated registry
`benchmarks/routing_hints.py`, and the S2MPJ sweep report prints the known
win next to any default-configuration row that misses one of those problems
(the *Routing hints* section — absent when nothing applies).

## The signature → lever map

| you observe | try | why it works |
| --- | --- | --- |
| `stalled`/`restoration_failed` with a **large constraint violation**, LP-like problem | `mu_schedule="quality"` | the monotone μ schedule left the central path; an adaptive oracle re-targets μ to the iterate (netlib `AGG`: `restoration_failed` → optimal in 42 iterations) |
| certified convergence to a **visibly worse objective** than a reference | `mu_schedule="quality"` | same mechanism at the optimality end (`PALMER1E`: wrong basin 0.114 → the documented 8.35e-4) |
| wrong basin / stall on a problem whose **constraints are badly scaled at x₀** | `BarrierOptions(slack_init_scale=0.1)` | the flat slack floor pins violated slacks at 1e-2 and starts the duals at μ/s; scaling the floor to the constraint magnitude fixes both (`HS59`, `HS98`, `SINROSNB`; also the radiotherapy Phase-1 stall — feasibility at IPOPT parity on TROTS `Protons_01`) |
| an **L-BFGS run grinding at a worse objective** on a nonconvex problem while the exact-Hessian routes solve it | `LBFGSOptions(damping_skip_ratio=1.0)` | Powell damping fabricates positive curvature out of pairs that strongly contradict it (`δᵀγ/δᵀBδ` down to −25 on `ORTHRGDS`); the threshold skips those pairs and damps only mild indefiniteness — `ORTHRGDS` reaches IPOPT's objective in 19 iterations |
| an **L-BFGS run frozen at a KKT plateau** (steps microscopic, KKT barely moving) | `LBFGSOptions(seed_formula="scalar1")` | the default ξ seed `γᵀγ/δᵀγ` exceeds IPOPT's `δᵀγ/δᵀδ` by the δ–γ misalignment factor `1/cos²∠`, which badly-scaled least squares drives to ~1e15 — an over-stiff seed freezes the step (`GASOIL`: 508 stalled iterations → optimal in 25, IPOPT parity; `NELSONLS` runs the default seed at ξ ≈ 1e20) |
| deeply infeasible start grinds with **α clipped by fraction-to-boundary** (radiotherapy-scale) | `slack_init_scale` ∈ [0.05, 0.5], or a μ-raising schedule (`quality`/`breedveld`) | see the [S2MPJ notes](../benchmarks/s2mpj.md) and the RT routing guidance in the backends guide |

## How to read a hint

A hint is a *routing* suggestion, not a better default: every one of these
levers lost or tied its full-corpus A/B as a default (the sweeps are recorded
in the [S2MPJ benchmark page](../benchmarks/s2mpj.md) and the changelog), so
switching one on globally trades your problem family for another. Reach for
the lever when your solve matches the signature; verify against the metrics
the registry records.

## Keeping the registry honest

Entries carry the measured budget and the default result they beat. Two
rules:

- **Only measured wins.** An entry is added from a recorded run (report or
  probe), never from expectation.
- **Prune on default changes.** If a fix makes the default configuration
  solve a hinted problem, the sweep report stops printing the hint
  automatically (it only annotates rows the default *missed*) — but the entry
  should still be removed once the fix ships.
