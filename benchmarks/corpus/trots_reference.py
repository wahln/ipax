"""Breedveld et al. 2017's published TROTS timings (reference data, CC BY 4.0).

Reads ``benchmarks/corpus/data/breedveld2017/trots_reference.json``, the
conversion of "Electronic supplementary material 2" of

    Breedveld, S., van den Berg, B. & Heijmen, B. (2017). "An interior-point
    implementation developed and tuned for radiation therapy treatment
    planning." Computational Optimization and Applications 68(2), 209-242.
    https://doi.org/10.1007/s10589-017-9919-4

(CC BY 4.0; see that directory's ``README.md`` and ``LICENSE``). The table
covers the paper's 120 TROTS problems: every case except the 25
``Prostate_BT_*`` brachytherapy cases, which the paper does not solve.

The ``Results/<case>_result.txt`` files of the TROTS dataset come from the
paper's Mehrotra runs. For all 120 cases their "Number of Iterations" and
"Solve Time" equal the JSON's ``mehrotra`` iterations and seconds (checked by
``tests/integration/test_trots_reference_dataset.py``).

The paper's times are wall-clock seconds on its 2017 machine (a 16-core Xeon,
16 threads unless a thread count is given; §6.3 of the paper). They are
context for ipax's timings, not targets on other hardware.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

_DATA = Path(__file__).resolve().parent / "data" / "breedveld2017"
_JSON = _DATA / "trots_reference.json"


@dataclass(frozen=True)
class BreedveldRun:
    """Iterations and wall-clock seconds of one of the paper's solver runs."""

    iterations: int
    seconds: float


@dataclass(frozen=True)
class BreedveldReference:
    """The paper's supplement row for one TROTS case.

    ``default`` / ``mehrotra`` / ``gondzio`` are the three interior-point
    methods (paper §3.3) in the optimal setup (tiling + permutation +
    multiple-precision arithmetic, 16 threads). ``variants`` holds the
    Mehrotra method with tiling / permutation / MPA toggled, keyed
    ``"no_tile_no_perm_no_mpa"``, ``"tile_no_perm_no_mpa"``,
    ``"no_tile_no_perm_mpa"``, ``"tile_perm_no_mpa"`` and ``"tile_perm_mpa"``
    (the last equals ``mehrotra``). ``threads`` maps ``"optimal"`` /
    ``"naive"`` to runs at 1, 2, 4, 8 and 16 threads; ``init_ratio`` holds the
    initialisation / ratio-control ablation (``"init_ratio"``,
    ``"no_init_ratio"``, ``"init_no_ratio"``, ``"no_init_no_ratio"``);
    ``breakdown_s`` is the Mehrotra run's time per module in seconds.
    ``n`` / ``m`` are the problem sizes before, ``n_final`` / ``m_final``
    after the solver's reformulation.
    """

    case: str
    label: str
    site: str
    index: int
    n: int
    m: int
    n_final: int
    m_final: int
    quadratic_constraints: int
    convex_constraints: int
    nonconvex_constraints: int
    default: BreedveldRun
    mehrotra: BreedveldRun
    gondzio: BreedveldRun
    variants: Mapping[str, BreedveldRun]
    threads: Mapping[str, Mapping[int, BreedveldRun]]
    init_ratio: Mapping[str, BreedveldRun]
    breakdown_s: Mapping[str, float]


def _run(raw: Mapping[str, Any]) -> BreedveldRun:
    return BreedveldRun(iterations=int(raw["iterations"]), seconds=float(raw["time_s"]))


def _runs(raw: Mapping[str, Any]) -> Mapping[str, BreedveldRun]:
    return MappingProxyType({key: _run(value) for key, value in raw.items()})


def _record(raw: Mapping[str, Any]) -> BreedveldReference:
    size, nonlinear, methods = raw["size"], raw["nonlinear_constraints"], raw["methods"]
    return BreedveldReference(
        case=raw["case"],
        label=raw["label"],
        site=raw["site"],
        index=int(raw["index"]),
        n=int(size["n_initial"]),
        m=int(size["m_initial"]),
        n_final=int(size["n_final"]),
        m_final=int(size["m_final"]),
        quadratic_constraints=int(nonlinear["quadratic"]),
        convex_constraints=int(nonlinear["convex"]),
        nonconvex_constraints=int(nonlinear["nonconvex"]),
        default=_run(methods["default"]),
        mehrotra=_run(methods["mehrotra"]),
        gondzio=_run(methods["gondzio"]),
        variants=_runs(raw["variants"]),
        threads=MappingProxyType(
            {
                setup: MappingProxyType(
                    {int(t): _run(run) for t, run in by_threads.items()}
                )
                for setup, by_threads in raw["scalability"].items()
            }
        ),
        init_ratio=_runs(raw["init_ratio"]),
        breakdown_s=MappingProxyType(
            {key: float(value) for key, value in raw["breakdown_s"].items()}
        ),
    )


@functools.lru_cache(maxsize=1)
def _document() -> Mapping[str, Any]:
    with open(_JSON, encoding="utf-8") as fh:
        doc: Mapping[str, Any] = json.load(fh)
    return doc


@functools.lru_cache(maxsize=1)
def breedveld2017_references() -> Mapping[str, BreedveldReference]:
    """Every case of the paper's supplement, keyed by TROTS case name."""
    return MappingProxyType(
        {raw["case"]: _record(raw) for raw in _document()["problems"]}
    )


def breedveld2017_reference(case: str) -> BreedveldReference | None:
    """The paper's supplement row for TROTS ``case`` (``None`` if absent).

    ``case`` is the dataset's file stem, e.g. ``"Prostate_CK_01"`` or
    ``"Head-and-Neck-Alt_03"``; a ``"trots/"`` prefix (the corpus problem
    name) is accepted. The ``Prostate_BT_*`` cases are not in the paper.
    """
    return breedveld2017_references().get(case.removeprefix("trots/"))


def breedveld2017_site_timings() -> dict[str, dict[str, float]]:
    """Seconds per solver module and tumour site (the supplement's ``Image``).

    Keys: site (``"Prostate CK"``, ``"Prostate VMAT"``, ``"Head-and-Neck"``,
    ``"Head-and-Neck Alt"``, ``"Protons"``, ``"Liver"``) → module
    (``permute_tile``, ``initialise``, ``cholesky``, ``backsolves``,
    ``functions_and_gradients``, ``make_tqd``, ``steplength``, ``make_n``) →
    seconds summed over the site's problems (Mehrotra method, optimal setup).
    A fresh copy on every call.
    """
    return {
        site: dict(modules)
        for site, modules in _document()["site_module_timings_s"].items()
    }


__all__ = [
    "BreedveldReference",
    "BreedveldRun",
    "breedveld2017_reference",
    "breedveld2017_references",
    "breedveld2017_site_timings",
]
