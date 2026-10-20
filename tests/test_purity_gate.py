"""Static hygiene gate: banned concrete-library imports in the core (§8.1.6).

Fails if ``numpy`` / ``torch`` / ``cupy`` / ``jax`` / ``scipy`` are imported
anywhere under ``ipax`` outside the allowed adapter directories
(``backend/sparse/``, ``backend/dense/``, ``problem/autodiff/``,
``interfaces/``). Mirrors ``scripts/check_purity.py``
so the boundary is enforced both in CI and as a normal test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_no_banned_imports_in_core():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_purity.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _load_purity_script():
    import importlib.util

    path = REPO_ROOT / "scripts" / "check_purity.py"
    spec = importlib.util.spec_from_file_location("check_purity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_core_may_not_import_the_interfaces_front_ends():
    # ipax/interfaces/ may import SciPy because nothing in the core imports it;
    # a core import of ipax.interfaces would pull SciPy into `import ipax`.
    import ast

    purity = _load_purity_script()
    for source in (
        "from ipax.interfaces.scipy import minimize_ipax",
        "import ipax.interfaces.scipy",
        "from ipax.interfaces import scipy",
    ):
        found = purity._banned_imports(ast.parse(source), "solve.py")
        assert found, source
    allowed = purity._banned_imports(
        ast.parse("from ipax.interfaces.scipy import STATUS_CODES"),
        "interfaces/other.py",
    )
    assert not allowed


def test_import_ipax_does_not_load_scipy_or_the_interfaces():
    code = (
        "import sys, ipax; "
        "bad = [m for m in ('scipy', 'ipax.interfaces') if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_relative_imports_of_the_interfaces_are_caught_too():
    import ast

    purity = _load_purity_script()
    for rel_path, source in (
        ("solve.py", "from .interfaces.scipy import minimize_ipax"),
        ("solve.py", "from . import interfaces"),
        ("ipm/driver.py", "from ..interfaces import scipy"),
        ("ipm/driver.py", "from ..interfaces.scipy import STATUS_CODES"),
        ("__init__.py", "from .interfaces.scipy import minimize_ipax"),
    ):
        assert purity._banned_imports(ast.parse(source), rel_path), (rel_path, source)
    for rel_path, source in (
        ("ipm/driver.py", "from .interfaces import x"),  # ipax.ipm.interfaces
        ("solve.py", "from .options import Options"),
        ("interfaces/other.py", "from .scipy import minimize_ipax"),
    ):
        assert not purity._banned_imports(ast.parse(source), rel_path), (
            rel_path,
            source,
        )
