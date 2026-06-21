"""Static hygiene gate: banned concrete-library imports in the core (§8.1.6).

Fails if ``numpy`` / ``torch`` / ``cupy`` / ``jax`` / ``scipy`` are imported
anywhere under ``ipax`` outside the allowed adapter directories
(``backend/sparse/``, ``problem/autodiff/``). Mirrors ``scripts/check_purity.py``
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
