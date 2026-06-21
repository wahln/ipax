---
description: Run the ipax verification gate (scripts/check.py) and summarize.
---

Run the project's single verification entrypoint and report the result.

Run `python scripts/check.py $ARGUMENTS` — no arguments runs all gates (format, lint,
types, purity, multi-backend tests); pass `--fast` to skip the slow test gate, or gate
names (e.g. `lint types`) to scope it.

Then summarize concisely: which gates passed/failed, and for each failure the
`file:line` and root cause. Do **not** fix anything unless I ask — just report. If a tool
is reported missing, give me the install command.
