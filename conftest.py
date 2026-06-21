"""Root pytest configuration: skip doctest collection of backend adapters whose
concrete library is not installed.

The suite runs with ``--doctest-modules`` over the ``ipax`` package (see
``[tool.pytest.ini_options]`` in ``pyproject.toml``), so pytest imports *every*
module to collect docstring examples. The backend adapters under
``ipax/problem/autodiff/`` and ``ipax/backend/sparse/`` are the one place allowed
to import a concrete array library at module top level (invariant #1 carve-out),
which makes them unimportable when that optional backend is absent — the normal
case in CI (e.g. the ``test`` job installs neither JAX nor CuPy, and CuPy is never
installed by CI at all). Without this, collection errors fail the whole job even
though the runtime dispatch already imports these adapters lazily and tolerates
their absence.

Each adapter is ignored only when one of its top-level concrete imports cannot be
located, so the doctests still run wherever the backend *is* installed.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

# Adapter module (relative to this file) -> the concrete imports it performs at
# module top level. Adapters that import their backend lazily (inside functions)
# are safe to collect unconditionally and are intentionally absent here.
_ADAPTER_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "ipax/problem/autodiff/torch.py": ("torch",),
    "ipax/problem/autodiff/jax.py": ("jax",),
    "ipax/backend/sparse/cupy.py": ("cupy",),
    "ipax/backend/sparse/numpy_scipy.py": ("numpy", "scipy"),
}


def _missing(module: str) -> bool:
    try:
        return find_spec(module) is None
    except ModuleNotFoundError:
        # A parent package is itself missing.
        return True


_HERE = Path(__file__).parent

collect_ignore = [
    str(_HERE / path)
    for path, modules in _ADAPTER_REQUIREMENTS.items()
    if any(_missing(module) for module in modules)
]
