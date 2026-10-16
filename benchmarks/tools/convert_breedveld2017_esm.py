"""Convert Breedveld et al. 2017's timing supplement (ESM 2) to JSON.

One-shot conversion of "Electronic supplementary material 2"
(``10589_2017_9919_MOESM2_ESM.xlsx``) of

    Breedveld, S., van den Berg, B. & Heijmen, B. (2017). "An interior-point
    implementation developed and tuned for radiation therapy treatment
    planning." Computational Optimization and Applications 68(2), 209-242.
    https://doi.org/10.1007/s10589-017-9919-4

published open access under CC BY 4.0. The same workbook ships with the TROTS
dataset as ``Results/Supplementary_Table.xlsx`` (byte-identical). The script
writes ``benchmarks/corpus/data/breedveld2017/trots_reference.json``, read by
:mod:`benchmarks.corpus.trots_reference`.

The workbook is parsed with the standard library only (``zipfile`` +
``xml.etree``); every value is the cell's stored value, converted to a JSON
number without rounding (a float round-trips to the same IEEE double). The
column layout of each sheet is asserted against the header text before any
value is read, so a different workbook fails loudly instead of being
mis-decoded.

Columns of the ``TROTS Overview`` sheet (header rows 1–5, problems in rows
6–125, one per TROTS case; row 126 and rows 133–144 are column sums and
per-site sums, which are not converted). All runs use 16 threads unless a
thread count is given; "optimal" = tiling + permutation + multiple-precision
arithmetic (MPA), "naive" = none of the three:

========  ==================================================================
A         index (0–119)
B         problem label, e.g. ``Prostate CK 01``
C–E       iterations, Default / Mehrotra / Gondzio method (optimal setup)
G–I       time (s), Default / Mehrotra / Gondzio
K–O       iterations of the Mehrotra method for five tiling / permutation /
          MPA variants: K no tile, no perm, no MPA · L tile, no perm, no MPA
          · M no tile, no perm, MPA · N tile, perm, no MPA · O tile, perm,
          MPA (O is a formula ``=D``)
P–T       time (s) of the same five variants (T is ``=H``)
V–Z       iterations, optimal setup, 1 / 2 / 4 / 8 / 16 threads (Z ``=D``)
AA–AE     iterations, naive setup, 1 / 2 / 4 / 8 / 16 threads (AE ``=K``)
AF–AJ     time (s), optimal setup, 1–16 threads (AJ ``=H``)
AK–AO     time (s), naive setup, 1–16 threads (AO ``=P``)
AP–AW     speed-up factors ``1->2 … 1->16`` (formulas ``AF/AG`` …; derived,
          not converted)
AY–BB     iterations: initialise + ratio control (``=D``) / no initialise +
          ratio / initialise + no ratio / no initialise + no ratio
BC–BF     time (s) of the same four (BC ``=H``)
BH–BQ     per-module time breakdown (s) of the Mehrotra run: load data from
          disk, permute/tile, initialise, Cholesky, backsolves, functions &
          gradients, make TQD, steplength, make N, and make N in the naive
          setup
BR        total measured excluding the load (``=SUM(BI:BP)``; derived, not
          converted)
BS        total runtime (``=H``)
BU–CS     time per iteration (formulas ``G/C`` …; derived,  # codespell:ignore bu
          not converted)
========  ==================================================================

``Problem Complexity`` (rows 4–123): index, label, initial ``n`` and ``m``,
final ``n`` and ``m`` (after the solver's reformulation, e.g. minimax
auxiliaries), and the numbers of quadratic / convex / nonconvex nonlinear
constraints. ``Image`` (rows 2–7): per tumour site, the seconds per module
summed over the site's problems (formula references into the per-site sum rows
of ``TROTS Overview``).

Run from the repository root::

    python benchmarks/tools/convert_breedveld2017_esm.py \\
        path/to/10589_2017_9919_MOESM2_ESM.xlsx
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_RELS_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_DOC_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

_OUT = (
    Path(__file__).resolve().parents[1]
    / "corpus"
    / "data"
    / "breedveld2017"
    / "trots_reference.json"
)

_FIRST_ROW, _LAST_ROW = 6, 125  # TROTS Overview problem rows
_COMPLEXITY_FIRST, _COMPLEXITY_LAST = 4, 123  # Problem Complexity problem rows

_THREADS = ("1", "2", "4", "8", "16")
_THREAD_HEADERS = ("1 thread", "2 threads", "4 threads", "8 threads", "16 threads")

# The five tiling / permutation / MPA variants of columns K–O and P–T, with the
# header text of rows 3 / 4 / 5 each column must carry.
_VARIANTS = (
    ("no_tile_no_perm_no_mpa", ("No Tile", "No Perm", "No MPA")),
    ("tile_no_perm_no_mpa", ("Tile", "No Perm", "No MPA")),
    ("no_tile_no_perm_mpa", ("No Tile", "No Perm", "MPA")),
    ("tile_perm_no_mpa", ("Tile", "Perm", "No MPA")),
    ("tile_perm_mpa", ("Tile", "Perm", "MPA")),
)
# Columns AY–BB / BC–BF: header rows 4 / 5.
_INIT_RATIO = (
    ("init_ratio", ("Initialise", "Ratio")),
    ("no_init_ratio", ("No Init", "Ratio")),
    ("init_no_ratio", ("Init", "No Ratio")),
    ("no_init_no_ratio", ("No Init", "No Ratio")),
)
# Columns BH–BQ: key, header rows 4 / 5 (a module name wraps onto two rows).
_BREAKDOWN = (
    ("load_data_from_disk", ("Load data", "from disk")),
    ("permute_tile", (None, "Permute/Tile")),
    ("initialise", (None, "Initialise")),
    ("cholesky", (None, "Cholesky")),
    ("backsolves", (None, "Backsolves")),
    ("functions_and_gradients", ("Functions &", "Gradients")),
    ("make_tqd", (None, "Make TQD")),
    ("steplength", (None, "Steplength")),
    ("make_n", (None, "Make N")),
    ("make_n_naive", ("Naive", "Make N")),
)
_TOTAL_RUNTIME_COL = "BS"
_IMAGE_MODULES = (
    ("permute_tile", "Permute/Tile"),
    ("initialise", "Initialise"),
    ("cholesky", "Cholesky"),
    ("backsolves", "Backsolves"),
    ("functions_and_gradients", "Functions &\nGradients"),
    ("make_tqd", "Make TQD"),
    ("steplength", "Steplength"),
    ("make_n", "Make N"),
)

_SITES = (
    "Prostate CK",
    "Prostate VMAT",
    "Head-and-Neck Alt",  # before "Head-and-Neck": the longer prefix wins
    "Head-and-Neck",
    "Protons",
    "Liver",
)


# --------------------------------------------------------------------------- #
# Minimal xlsx reader
# --------------------------------------------------------------------------- #
def _col_index(col: str) -> int:
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx


def _col_name(idx: int) -> str:
    name = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def _col_range(first: str, last: str) -> list[str]:
    return [_col_name(i) for i in range(_col_index(first), _col_index(last) + 1)]


class _Workbook:
    """Cell values of an xlsx workbook, keyed by sheet name and ``A1`` ref."""

    def __init__(self, path: Path) -> None:
        with zipfile.ZipFile(path) as zf:
            strings = self._shared_strings(zf)
            targets = self._sheet_targets(zf)
            self.sheets: dict[str, dict[str, str]] = {
                name: self._read_sheet(zf, target, strings)
                for name, target in targets.items()
            }

    @staticmethod
    def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
        if "xl/sharedStrings.xml" not in zf.namelist():
            return []
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        return [
            "".join(t.text or "" for t in si.iter(f"{_NS}t"))
            for si in root.findall(f"{_NS}si")
        ]

    @staticmethod
    def _sheet_targets(zf: zipfile.ZipFile) -> dict[str, str]:
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        by_id = {
            r.get("Id"): r.get("Target") for r in rels.iter(f"{_RELS_NS}Relationship")
        }
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        out: dict[str, str] = {}
        for sheet in wb.iter(f"{_NS}sheet"):
            target = by_id[sheet.get(f"{_DOC_REL_NS}id")]
            assert target is not None
            target = target.lstrip("/")
            out[str(sheet.get("name"))] = (
                target if target.startswith("xl/") else f"xl/{target}"
            )
        return out

    @staticmethod
    def _read_sheet(
        zf: zipfile.ZipFile, target: str, strings: list[str]
    ) -> dict[str, str]:
        root = ET.fromstring(zf.read(target))
        cells: dict[str, str] = {}
        for c in root.iter(f"{_NS}c"):
            ref = str(c.get("r"))
            kind = c.get("t")
            if kind == "inlineStr":
                cells[ref] = "".join(t.text or "" for t in c.iter(f"{_NS}t"))
                continue
            v = c.find(f"{_NS}v")
            if v is None or v.text is None:
                continue
            cells[ref] = strings[int(v.text)] if kind == "s" else v.text
        return cells


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def _expect(sheet: dict[str, str], ref: str, text: str) -> None:
    got = sheet.get(ref)
    if got != text:
        raise ValueError(f"header {ref}: expected {text!r}, found {got!r}")


def _int(sheet: dict[str, str], ref: str) -> int:
    raw = sheet[ref]
    value = float(raw)
    if not value.is_integer():
        raise ValueError(f"{ref}: expected an integer, found {raw!r}")
    return int(value)


def _float(sheet: dict[str, str], ref: str) -> float:
    return float(sheet[ref])


def _check_overview_headers(s: dict[str, str]) -> None:
    for ref, text in {
        "C1": "Different interior-point approaches",
        "K1": "Permutation/Tiling/Multiple Precision Arithmetic",
        "V1": "Scalability",
        "AY1": "Initialisation and ratio control",
        "BH1": "Breakdown",
        "C2": "Iterations",
        "G2": "Time (s)",
        "K2": "Iterations",
        "P2": "Time (s)",
        "V2": "Iterations",
        "AF2": "Time (s)",
        "AY2": "Iterations",
        "BC2": "Time (s)",
        "A3": "Index",
        "B3": "Problem",
        "V4": "Optimal (Tile/Perm/MPA)",
        "AA4": "Naive (No Tile/Perm/MPA)",
        "AF4": "Optimal (Tile/Perm/MPA)",
        "AK4": "Naive (No Tile/Perm/MPA)",
        "BS4": "Total",
        "BS5": "runtime",
    }.items():
        _expect(s, ref, text)
    for cols in (("C", "D", "E"), ("G", "H", "I")):
        for col, method in zip(cols, ("Default", "Mehrotra", "Gondzio"), strict=True):
            _expect(s, f"{col}5", method)
    for first in ("K", "P"):
        for col, (_, heads) in zip(
            _col_range(first, _col_name(_col_index(first) + 4)), _VARIANTS, strict=True
        ):
            for row, text in zip((3, 4, 5), heads, strict=True):
                _expect(s, f"{col}{row}", text)
    for first in ("V", "AA", "AF", "AK"):
        cols = _col_range(first, _col_name(_col_index(first) + 4))
        for col, text in zip(cols, _THREAD_HEADERS, strict=True):
            _expect(s, f"{col}5", text)
    for first in ("AY", "BC"):
        cols = _col_range(first, _col_name(_col_index(first) + 3))
        for col, (_, heads) in zip(cols, _INIT_RATIO, strict=True):
            for row, text in zip((4, 5), heads, strict=True):
                _expect(s, f"{col}{row}", text)
    for col, (_, heads) in zip(_col_range("BH", "BQ"), _BREAKDOWN, strict=True):
        for row, text in zip((4, 5), heads, strict=True):
            if text is not None:
                _expect(s, f"{col}{row}", text)


def _site_of(label: str) -> str:
    for site in _SITES:
        if label.startswith(site + " "):
            return site
    raise ValueError(f"no tumour site for problem label {label!r}")


def case_name(label: str) -> str:
    """The TROTS case (``.mat`` file stem) for a paper problem label.

    The paper writes ``Head-and-Neck Alt 01`` where the dataset file is
    ``Head-and-Neck-Alt_01``; every other label only swaps spaces for
    underscores (``Prostate VMAT 101`` → ``Prostate_VMAT_101``).
    """
    return label.replace("Head-and-Neck Alt", "Head-and-Neck-Alt").replace(" ", "_")


def _overview_record(s: dict[str, str], row: int) -> dict[str, Any]:
    r = str(row)

    def it(col: str) -> int:
        return _int(s, col + r)

    def sec(col: str) -> float:
        return _float(s, col + r)

    methods = {
        name: {"iterations": it(ic), "time_s": sec(tc)}
        for name, ic, tc in (
            ("default", "C", "G"),
            ("mehrotra", "D", "H"),
            ("gondzio", "E", "I"),
        )
    }
    variants = {
        key: {"iterations": it(ic), "time_s": sec(tc)}
        for (key, _), ic, tc in zip(
            _VARIANTS, _col_range("K", "O"), _col_range("P", "T"), strict=True
        )
    }
    scalability = {
        setup: {
            threads: {"iterations": it(ic), "time_s": sec(tc)}
            for threads, ic, tc in zip(
                _THREADS, _col_range(i0, i1), _col_range(t0, t1), strict=True
            )
        }
        for setup, i0, i1, t0, t1 in (
            ("optimal", "V", "Z", "AF", "AJ"),
            ("naive", "AA", "AE", "AK", "AO"),
        )
    }
    init_ratio = {
        key: {"iterations": it(ic), "time_s": sec(tc)}
        for (key, _), ic, tc in zip(
            _INIT_RATIO, _col_range("AY", "BB"), _col_range("BC", "BF"), strict=True
        )
    }
    breakdown = {
        key: sec(col)
        for (key, _), col in zip(_BREAKDOWN, _col_range("BH", "BQ"), strict=True)
    }
    breakdown["total_runtime"] = sec(_TOTAL_RUNTIME_COL)
    label = s["B" + r]
    return {
        "index": it("A"),
        "label": label,
        "case": case_name(label),
        "site": _site_of(label),
        "methods": methods,
        "variants": variants,
        "scalability": scalability,
        "init_ratio": init_ratio,
        "breakdown_s": breakdown,
    }


def _complexity(s: dict[str, str]) -> dict[str, dict[str, Any]]:
    for ref, text in {
        "A1": "Index",
        "B1": "Problem",
        "C2": "Initial",
        "E2": "Final",
        "G2": "Nonlinear constraints",
        "C3": "n",
        "D3": "m",
        "E3": "n",
        "F3": "m",
        "G3": "Quadratic",
        "H3": "Convex",
        "I3": "Nonconvex",
    }.items():
        _expect(s, ref, text)
    out: dict[str, dict[str, Any]] = {}
    for row in range(_COMPLEXITY_FIRST, _COMPLEXITY_LAST + 1):
        r = str(row)
        out[s["B" + r]] = {
            "index": _int(s, "A" + r),
            "size": {
                "n_initial": _int(s, "C" + r),
                "m_initial": _int(s, "D" + r),
                "n_final": _int(s, "E" + r),
                "m_final": _int(s, "F" + r),
            },
            "nonlinear_constraints": {
                "quadratic": _int(s, "G" + r),
                "convex": _int(s, "H" + r),
                "nonconvex": _int(s, "I" + r),
            },
        }
    return out


def _site_modules(s: dict[str, str]) -> dict[str, dict[str, float]]:
    for (_, text), col in zip(_IMAGE_MODULES, _col_range("B", "I"), strict=True):
        _expect(s, f"{col}1", text)
    out: dict[str, dict[str, float]] = {}
    for row in range(2, 2 + len(_SITES)):
        r = str(row)
        out[s["A" + r]] = {
            key: _float(s, col + r)
            for (key, _), col in zip(_IMAGE_MODULES, _col_range("B", "I"), strict=True)
        }
    if set(out) != set(_SITES):
        raise ValueError(f"Image sheet sites {sorted(out)} != {sorted(_SITES)}")
    return out


def convert(path: Path) -> dict[str, Any]:
    """Decode the workbook at ``path`` into the JSON document."""
    wb = _Workbook(path)
    overview = wb.sheets["TROTS Overview"]
    _check_overview_headers(overview)
    complexity = _complexity(wb.sheets["Problem Complexity"])
    problems = []
    for row in range(_FIRST_ROW, _LAST_ROW + 1):
        rec = _overview_record(overview, row)
        extra = complexity[rec["label"]]
        if extra["index"] != rec["index"]:
            raise ValueError(f"{rec['label']}: index mismatch between sheets")
        rec["size"] = extra["size"]
        rec["nonlinear_constraints"] = extra["nonlinear_constraints"]
        problems.append(rec)
    if len(problems) != 120 or len({p["case"] for p in problems}) != 120:
        raise ValueError("expected 120 distinct problems")
    return {
        "source": {
            "citation": (
                "Breedveld, S., van den Berg, B. & Heijmen, B. (2017). An "
                "interior-point implementation developed and tuned for "
                "radiation therapy treatment planning. Computational "
                "Optimization and Applications 68(2), 209-242."
            ),
            "doi": "10.1007/s10589-017-9919-4",
            "supplement": "Electronic supplementary material 2 "
            "(10589_2017_9919_MOESM2_ESM.xlsx)",
            "license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
            "converted_by": "ipax benchmarks/tools/convert_breedveld2017_esm.py",
            "converted_on": datetime.date.today().isoformat(),
            "note": (
                "Values are the workbook's stored cell values, unmodified. "
                "Formula-derived columns (speed-up factors, time per "
                "iteration, sums) are not converted. Runs use 16 threads "
                "unless a thread count is given; 'optimal' = tiling + "
                "permutation + multiple-precision arithmetic (MPA), 'naive' "
                "= none of them. 'variants', 'scalability', 'init_ratio' and "
                "'breakdown_s' are Mehrotra-method runs; 'variants."
                "tile_perm_mpa', 'scalability.optimal.16' and "
                "'init_ratio.init_ratio' are the same run as "
                "'methods.mehrotra'."
            ),
        },
        "units": {
            "iterations": "interior-point iterations",
            "time_s": "seconds (wall clock)",
            "breakdown_s": "seconds",
            "site_module_timings_s": "seconds, summed over the site's problems",
            "size": "n = variables, m = constraints",
        },
        "problems": problems,
        "site_module_timings_s": _site_modules(wb.sheets["Image"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("xlsx", type=Path, help="path to the ESM 2 workbook")
    parser.add_argument("--out", type=Path, default=_OUT, help="output JSON path")
    args = parser.parse_args(argv)
    doc = convert(args.xlsx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    # Compact the innermost {"iterations": .., "time_s": ..} objects to one line.
    text = re.sub(
        r'\{\n\s*"iterations": (\d+),\n\s*"time_s": ([^\n]+)\n\s*\}',
        r'{"iterations": \1, "time_s": \2}',
        text,
    )
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(doc['problems'])} problems to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
