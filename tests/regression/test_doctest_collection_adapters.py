"""Regression: CI doctest collection must tolerate absent optional backends (0.1.1).

The suite runs ``pytest --doctest-modules`` over the ``ipax`` package, so pytest
imports every module to collect docstring examples. The backend adapters under
``ipax/problem/autodiff`` and ``ipax/backend/sparse`` are the one place allowed to
import a concrete array library at module top level (invariant #1 carve-out), so
they cannot be imported when that optional backend is missing — which broke every
CI ``test`` job (neither JAX nor CuPy is installed there). The root
``conftest.py`` skips those adapters during collection.

This guards the skip map: any adapter that imports a concrete library at *module
top level* must be registered in ``conftest._ADAPTER_REQUIREMENTS`` (with at least
the libraries it imports), and any adapter that does not must be absent from it.
A new top-level-importing adapter therefore cannot silently reintroduce the
failure.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER_DIRS = ("ipax/problem/autodiff", "ipax/backend/sparse")
# Mirrors scripts/check_purity.BANNED_TOP_LEVEL plus the sparse-direct natives.
_CONCRETE_LIBS = {
    "numpy",
    "scipy",
    "torch",
    "jax",
    "jaxlib",
    "cupy",
    "cupyx",
    "nvmath",
    "feral",
}
# Modules that ship in the same wheel as another, so guarding the anchor package
# in conftest is enough (e.g. ``cupyx`` is installed iff ``cupy`` is).
_IMPORT_ANCHOR = {"cupyx": "cupy", "jaxlib": "jax"}


def _load_conftest_requirements() -> dict[str, tuple[str, ...]]:
    path = _REPO_ROOT / "conftest.py"
    spec = importlib.util.spec_from_file_location("ipax_root_conftest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module._ADAPTER_REQUIREMENTS)


def _top_level_concrete_imports(py: Path) -> set[str]:
    tree = ast.parse(py.read_text(encoding="utf-8"))
    libs: set[str] = set()
    for node in tree.body:  # module-level statements only; lazy imports are safe
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top in _CONCRETE_LIBS:
                    libs.add(_IMPORT_ANCHOR.get(top, top))
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top = (node.module or "").split(".", 1)[0]
            if top in _CONCRETE_LIBS:
                libs.add(_IMPORT_ANCHOR.get(top, top))
    return libs


def _adapter_modules() -> list[Path]:
    files: list[Path] = []
    for rel_dir in _ADAPTER_DIRS:
        files.extend(sorted((_REPO_ROOT / rel_dir).glob("*.py")))
    return files


@pytest.mark.parametrize(
    "module_path",
    _adapter_modules(),
    ids=lambda p: p.relative_to(_REPO_ROOT).as_posix(),
)
def test_top_level_adapter_imports_registered_for_collection_skip(
    module_path: Path,
) -> None:
    rel = module_path.relative_to(_REPO_ROOT).as_posix()
    libs = _top_level_concrete_imports(module_path)
    requirements = _load_conftest_requirements()

    if not libs:
        assert rel not in requirements, (
            f"{rel} imports no concrete library at module top level, so it is "
            "always importable and must not be skipped during collection."
        )
        return

    assert rel in requirements, (
        f"{rel} imports {sorted(libs)} at module top level but is not registered "
        "in conftest._ADAPTER_REQUIREMENTS; --doctest-modules will fail to collect "
        "it whenever that backend is absent (see the 0.1.1 CI fix)."
    )
    assert set(requirements[rel]) >= libs, (
        f"{rel} imports {sorted(libs)} but conftest only guards "
        f"{sorted(requirements[rel])}; collection will still fail when an "
        "unguarded backend is absent."
    )
