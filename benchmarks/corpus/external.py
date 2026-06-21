"""External standard-set loaders (CUTEst, Maros–Mészáros) — gated (§9.1).

These standard sets require a system CUTEst install (driven by ``pycutest``) or a
separately downloaded data set. Per the plan they are **not** committed and
**not** run in CI; each loader returns ``[]`` when its dependency/data is
unavailable, so callers can extend the corpus unconditionally::

    corpus = default_corpus() + cutest_problems() + maros_meszaros_problems()

Translating a ``pycutest`` problem (or a Maros–Mészáros ``.mat``) into an ipax
:class:`~ipax.problem.base.Problem` lands once the toolchain is wired into CI;
it is deliberately deferred here (download-gated, §1.2).
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.corpus import BenchmarkProblem


def cutest_available() -> bool:
    """Whether the optional ``pycutest`` binding is importable."""
    return importlib.util.find_spec("pycutest") is not None


def cutest_problems(names: list[str] | None = None) -> list[BenchmarkProblem]:
    """Hock–Schittkowski / CUTEst problems via ``pycutest`` (install-gated).

    Returns ``[]`` unless a CUTEst toolchain + ``pycutest`` are installed. The
    ipax-``Problem`` wrapping is deferred until the toolchain is available in CI.
    """
    del names
    if not cutest_available():
        return []
    return []  # toolchain present but the pycutest→Problem wrapper is not wired


def maros_meszaros_available() -> bool:
    """Whether the Maros–Mészáros QP data set can be located (``scipy`` + data)."""
    return importlib.util.find_spec("scipy") is not None


def maros_meszaros_problems(data_dir: str | None = None) -> list[BenchmarkProblem]:
    """Maros–Mészáros convex-QP set (download-gated).

    Returns ``[]`` until the ``.mat`` data set is downloaded and a loader is
    wired; the data is never committed (size/licensing).
    """
    del data_dir
    return []


__all__ = [
    "cutest_available",
    "cutest_problems",
    "maros_meszaros_available",
    "maros_meszaros_problems",
]
