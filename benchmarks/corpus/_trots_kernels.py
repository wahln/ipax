"""Compiled host kernels of the TROTS loader (benchmark-side, optional Numba).

The TROTS loader's tiled Gram (Breedveld et al. 2017, eq. (48), §5.2) adds
each compacted tile product ``g`` (``c×c``, the lower triangle of
``tileᵀ diag(w) tile``) into the ``n×n`` accumulator at the tile's active
columns. In NumPy that is ``acc[ix_(cols, cols)] += g``: a gathered
temporary, an add and a scatter, with per-element fancy-index overhead —
about 2.7 s of Head-and-Neck_01's 20 s NumPy iteration (docs/benchmarks/
trots.md, "Head-and-Neck_01 speed profile"). :func:`scatter_lower_rows` is
the same per-entry addition as a plain loop, compiled by Numba when it is
installed (:func:`compiled_scatter_lower_rows`): no temporaries, the GIL
released, so the loader's existing thread pool runs it in row chunks.

This module never imports Numba at module level: the kernel is plain
Python, so CI (which has no Numba) can run it directly on small blocks to
pin its semantics, and the loader falls back to its NumPy scatter when
:func:`compiled_scatter_lower_rows` returns None. Nothing here is reachable
from ``ipax/`` (invariant #1 governs the library; this is an application-edge
adapter, see AGENTS.md "Scope guardrails").
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

_LOG = logging.getLogger(__name__)

#: Rows of ``g`` handled together: for each column ``j`` the inner loop
#: reads a contiguous run of an F-ordered ``syrk`` product (8 float64 = one
#: 64-byte cache line) and writes one entry in each of ``ROW_BLOCK``
#: accumulator rows, which consecutive ``j`` (sorted ``cols``) revisit.
#: Measured at n = 9984, c = 9700, 8 threads (2026-10-02): 4 / 8 / 16 rows
#: took 35.6 / 28.0 / 27.4 ms against 34.3 ms for the plain row loop and
#: 86.1 ms for the NumPy scatter.
ROW_BLOCK = 8


def scatter_lower_rows(acc: Any, cols: Any, g: Any, r0: int, r1: int) -> None:
    """``acc[cols[i], cols[j]] += g[i, j]`` for ``r0 <= i < r1``, ``j <= i``.

    The lower triangle (diagonal included) of rows ``[r0, r1)`` of the
    ``c×c`` block ``g``, added at the strictly increasing ``cols`` — so it
    lands in the lower triangle of ``acc``. Each such entry receives exactly
    one IEEE addition ``acc + g`` (a float32 ``g`` entry widened to float64
    first, exactly), the same operation as NumPy's
    ``acc[ix_(cols, cols)] += g`` on that entry: the lower triangle is bitwise
    the NumPy scatter's, whatever the row ranges, their order, or the
    thread that runs them. Entries of ``g`` above the diagonal are never
    read (the loader's final mirror overwrites ``acc``'s upper triangle).

    ``cols`` must be strictly increasing and inside ``acc`` (every loader
    producer's is: ``np.unique`` of a tile's column indices, or ``arange``).
    Unlike NumPy, the compiled kernel does not check bounds, and repeated
    columns would add twice where ``ix_ +=`` keeps one write.
    """
    for i0 in range(r0, r1, ROW_BLOCK):
        i1 = min(i0 + ROW_BLOCK, r1)
        # Columns left of the row block: a full ROW_BLOCK-row panel.
        for j in range(i0):
            cj = cols[j]
            for i in range(i0, i1):
                acc[cols[i], cj] += g[i, j]
        # The block's own lower triangle.
        for i in range(i0, i1):
            ci = cols[i]
            for j in range(i0, i + 1):
                acc[ci, cols[j]] += g[i, j]


def compiled_scatter_lower_rows() -> Callable[..., None] | None:
    """:func:`scatter_lower_rows` compiled by Numba and warmed up, or None
    when Numba is not importable (the loader then keeps its NumPy scatter).

    ``nogil=True`` so the loader's thread pool runs it concurrently;
    ``parallel=False`` so Numba starts no threading layer of its own (the
    pool's ``TiledGramConfig.scatter_threads`` is the one thread knob, and
    the multi-lane Gram calls it from its lanes); no ``fastmath`` (the
    additions stay IEEE, hence bitwise the NumPy scatter's).
    ``cache=True`` keeps the machine code in ``__pycache__`` next to this
    file, so only a machine's first process pays the compilation. The
    warm-up compiles (or loads) the specializations the loader produces —
    a contiguous or sub-block accumulator, float64 or float32 products,
    F-ordered (``syrk``) or C-ordered (``gemm``) — so no Gram pays for it.
    Callers memoize the result (it costs the Numba import, ~0.5 s).
    Measured on the i9-13900HX of docs/benchmarks/trots.md: 1.8 s for the
    first compilation of the eight specializations, 0.7 s to load them from
    the cache afterwards (plus the import).

    A Numba that imports but fails to compile (an llvmlite or cache error)
    is logged as a warning and also returns None: the NumPy scatter gives
    the same bits, only slower.
    """
    try:
        import numba
    except ImportError:
        _LOG.debug("numba not importable: the TROTS loader keeps its NumPy scatter")
        return None
    import numpy as np

    try:
        kernel = numba.njit(nogil=True, cache=True)(scatter_lower_rows)
        cols = np.arange(3, dtype=np.int64)
        whole = np.zeros((4, 4), dtype=np.float64)
        for acc in (whole[:3, :3].copy(), whole[:3, :3]):  # C, and a sub-block view
            for dtype in (np.float64, np.float32):
                g = np.zeros((3, 3), dtype=dtype)
                kernel(acc, cols, g, 0, 3)
                kernel(acc, cols, np.asfortranarray(g), 0, 3)
    except Exception:  # any compiler failure: fall back to NumPy
        _LOG.warning(
            "numba failed to compile the TROTS scatter kernel; using NumPy",
            exc_info=True,
        )
        return None
    return kernel
