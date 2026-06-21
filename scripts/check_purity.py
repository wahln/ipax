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
    "problem/autodiff/",
)

CORE_ROOT = Path(__file__).resolve().parents[1] / "ipax"


def _is_allowed(rel_path: str) -> bool:
    return any(rel_path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _banned_imports(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top in BANNED_TOP_LEVEL:
                    found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".", 1)[0]
            if top in BANNED_TOP_LEVEL:
                found.append((node.lineno, node.module or ""))
    return found


def main() -> int:
    violations: list[str] = []
    for path in sorted(CORE_ROOT.rglob("*.py")):
        rel = path.relative_to(CORE_ROOT).as_posix()
        if _is_allowed(rel):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, name in _banned_imports(tree):
            violations.append(f"  ipax/{rel}:{lineno}: imports '{name}'")

    if violations:
        print("Import-purity gate FAILED — concrete library imports in the core:")
        print("\n".join(violations))
        print(
            "\nMove backend-specific code into an adapter "
            f"({', '.join(ALLOWED_PREFIXES)})."
        )
        return 1

    print("Import-purity gate passed: core is free of concrete array-library imports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
