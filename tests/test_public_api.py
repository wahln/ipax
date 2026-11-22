"""The API reference, the top-level exports and the stability policy agree.

``docs/stability.md`` defines the public API as ``ipax.__all__`` plus the
documented entries of a few modules; ``docs/reference.md`` documents it.
This keeps the three in step: every reference entry imports, every top-level
export is documented, and every documented name outside ``ipax.__all__``
lives in a module the policy names.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

import ipax

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference.md"
POLICY = ROOT / "docs" / "stability.md"

# Modules whose documented entries are public at their own path
# (docs/stability.md, "What is public").
PUBLIC_MODULES = {
    "ipax.interfaces.scipy",
    "ipax.backend.operators",
    "ipax.linalg.solver",
    "ipax.backend.namespace",
}


def _entries() -> list[str]:
    text = REFERENCE.read_text(encoding="utf-8")
    return re.findall(r"^::: (\S+)", text, flags=re.MULTILINE)


def _resolve(path: str) -> object:
    module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


@pytest.mark.parametrize("path", _entries())
def test_every_reference_entry_imports(path):
    _resolve(path)


@pytest.mark.parametrize("name", sorted(ipax.__all__))
def test_every_top_level_export_is_in_the_reference(name):
    documented = {id(_resolve(path)) for path in _entries()}
    assert id(getattr(ipax, name)) in documented, (
        f"ipax.{name} missing from reference.md"
    )


@pytest.mark.parametrize("path", _entries())
def test_entries_outside_the_top_level_live_in_a_public_module(path):
    obj = _resolve(path)
    exported = {id(getattr(ipax, name)) for name in ipax.__all__}
    module_name = path.rpartition(".")[0]
    assert id(obj) in exported or module_name in PUBLIC_MODULES, (
        f"{path} is documented but neither exported from ipax nor in a public "
        "module of docs/stability.md"
    )


def test_policy_page_names_every_public_module():
    text = POLICY.read_text(encoding="utf-8")
    for module in PUBLIC_MODULES:
        assert f"`{module}" in text, module
