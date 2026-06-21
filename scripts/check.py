#!/usr/bin/env python
"""Single verification entrypoint for ipax (mirrors the CI gates).

Runs the same gate set CI runs, from one command, so "is my change done?" is
one invocation instead of reconstructing the ruff/mypy/pytest/purity flags from
``.github/workflows/ci.yml``, ``.pre-commit-config.yaml`` and ``AGENTS.md``.

Usage::

    python scripts/check.py                 # all gates
    python scripts/check.py lint types      # only the named gates
    python scripts/check.py --fast          # skip the (slow) multi-backend test gate
    python scripts/check.py --list          # show available gates

Gates run in a fixed order (``format → lint → types → purity → test``); each maps
to the exact command CI runs. Every selected gate runs even if an earlier one
fails, then a summary is printed and the script exits non-zero if any failed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Each gate is (argv, extra-env). The multi-backend test gate must exercise
# NumPy + Torch + the array-api-strict purity backend, exactly as CI does
# (see .github/workflows/ci.yml, the `test` job).
GATES: dict[str, tuple[list[str], dict[str, str]]] = {
    "format": (["ruff", "format", "--check", "."], {}),
    "lint": (["ruff", "check", "."], {}),
    "types": (["mypy", "ipax"], {}),
    "purity": ([sys.executable, "scripts/check_purity.py"], {}),
    "test": (
        [sys.executable, "-m", "pytest", "-q", "-n", "auto"],
        {"IPAX_BACKENDS": "numpy,torch,array_api_strict"},
    ),
}

# Fixed run order regardless of the order gates are named on the command line.
ORDER = ["format", "lint", "types", "purity", "test"]


def run_gate(name: str) -> bool:
    cmd, env_overlay = GATES[name]
    # ruff/mypy are console scripts from the dev env; give a friendly error
    # rather than a raw FileNotFoundError if the lint extra is not installed.
    if shutil.which(cmd[0]) is None and cmd[0] != sys.executable:
        print(
            f"=== {name}: SKIPPED — '{cmd[0]}' not found (pip install -e '.[dev]') ==="
        )
        return False
    env = os.environ.copy()
    env.update(env_overlay)
    print(f"\n=== {name}: {' '.join(cmd)} ===", flush=True)
    return subprocess.run(cmd, cwd=REPO, env=env).returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "gates", nargs="*", choices=[*GATES, []], help="gates to run (default: all)"
    )
    parser.add_argument(
        "--fast", action="store_true", help="skip the multi-backend test gate"
    )
    parser.add_argument(
        "--list", action="store_true", help="list available gates and exit"
    )
    args = parser.parse_args(argv)

    if args.list:
        for name in ORDER:
            print(f"{name:8} {' '.join(GATES[name][0])}")
        return 0

    selected = [g for g in ORDER if g in (args.gates or ORDER)]
    if args.fast:
        selected = [g for g in selected if g != "test"]

    results = {name: run_gate(name) for name in selected}

    print("\n=== summary ===")
    for name in selected:
        print(f"  {'ok  ' if results[name] else 'FAIL'}  {name}")
    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} gate(s) failed: {', '.join(failed)}")
        return 1
    print("\nall gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
