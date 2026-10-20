#!/usr/bin/env python
"""Import-purity gate (invariants #1 and #4).

Fails if a concrete array/sparse library is imported anywhere under
``ipax`` outside the explicitly allowed adapter directories. Pure stdlib
(``ast`` + ``pathlib``) so it never itself trips the boundary.

Usage::

    python scripts/check_purity.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

BANNED_TOP_LEVEL = {"numpy", "torch", "cupy", "cupyx", "jax", "jaxlib", "scipy"}

# Directories (relative to the ipax package) where adapter imports are allowed.
ALLOWED_PREFIXES = (
    "backend/sparse/",
    "backend/dense/",
    "problem/autodiff/",
    # Front ends in another library's calling convention (e.g. SciPy's
    # ``minimize``); never imported by ``ipax`` itself.
    "interfaces/",
)

# Front ends that only applications import; see _banned_imports.
INTERFACES_PACKAGE = "ipax.interfaces"

CORE_ROOT = Path(__file__).resolve().parents[1] / "ipax"


def _is_allowed(rel_path: str) -> bool:
    return any(rel_path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _is_interfaces(module: str) -> bool:
    return module == INTERFACES_PACKAGE or module.startswith(INTERFACES_PACKAGE + ".")


def _banned_imports(tree: ast.AST, rel_path: str) -> list[tuple[int, str]]:
    """Concrete-library imports outside the adapters, plus any import of the
    ``ipax.interfaces`` front ends from outside ``interfaces/`` — those may
    import SciPy only because nothing else in ``ipax`` imports them."""
    check_libs = not _is_allowed(rel_path)
    check_interfaces = not rel_path.startswith("interfaces/")
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if (check_libs and top in BANNED_TOP_LEVEL) or (
                    check_interfaces and _is_interfaces(alias.name)
                ):
                    found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".", 1)[0]
            # Relative imports name no top-level library, but may still reach
            # the front ends (``from .interfaces.scipy import …``).
            resolved = _resolve_relative(module, node.level, rel_path)
            names = [resolved, *(f"{resolved}.{alias.name}" for alias in node.names)]
            if (check_libs and node.level == 0 and top in BANNED_TOP_LEVEL) or (
                check_interfaces and any(_is_interfaces(name) for name in names)
            ):
                found.append((node.lineno, "." * node.level + module))
    return found


def _resolve_relative(module: str, level: int, rel_path: str) -> str:
    """Absolute module name of ``from <.*level><module> import …`` in ``rel_path``."""
    if level == 0:
        return module
    package = ["ipax", *rel_path.split("/")[:-1]]  # the importing module's package
    base = package[: len(package) - (level - 1)]
    return ".".join([*base, module] if module else base)


def main() -> int:
    violations: list[str] = []
    for path in sorted(CORE_ROOT.rglob("*.py")):
        rel = path.relative_to(CORE_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, name in _banned_imports(tree, rel):
            violations.append(f"  ipax/{rel}:{lineno}: imports '{name}'")

    if violations:
        print(
            "Import-purity gate FAILED — concrete library (or ipax.interfaces) "
            "imports in the core:"
        )
        print("\n".join(violations))
        print(
            "\nMove backend-specific code into an adapter "
            f"({', '.join(ALLOWED_PREFIXES)}); only applications import "
            f"{INTERFACES_PACKAGE}."
        )
        return 1

    print("Import-purity gate passed: core is free of concrete array-library imports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
