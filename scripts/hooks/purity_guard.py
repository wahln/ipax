#!/usr/bin/env python
"""Claude Code PostToolUse hook: enforce import-purity right after a core edit.

The pre-commit hook already runs ``check_purity.py`` at commit time, but an agent
makes many edits before committing. This hook moves the *same* gate to the moment
an ``Edit``/``Write``/``MultiEdit`` touches ``ipax/`` — so an invariant-#1/#4
violation (a concrete array/sparse import in the core) is fed back to the agent
immediately, before it builds on the broken assumption.

Reads the Claude Code hook payload on stdin. If the edited file is under
``ipax/`` it runs the purity gate and, on violation, exits 2 so Claude Code
surfaces the message to the agent. Any other situation is a silent no-op (exit
0), so edits to tests/docs/examples are never slowed down.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path")
    if not file_path:
        return 0

    try:
        rel = Path(file_path).resolve().relative_to(REPO)
    except ValueError:
        return 0  # edit outside the repo — not our concern

    if rel.parts[:1] != ("ipax",):
        return 0  # only the core is subject to the purity gate

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_purity.py")],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Exit code 2 → Claude Code feeds stderr back to the agent to self-correct.
        sys.stderr.write(result.stdout + result.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
