---
description: Drive the ipax release workflow (rc branch -> PR -> tag) for a version.
argument-hint: X.Y.Z
---

Drive a release of version **$ARGUMENTS** following the **Releasing** section in
`AGENTS.md` — that section is the source of truth; read it first and keep this command
in sync with it if it changes.

This workflow has human/async gates (CI, code review, PR merge). **Stop at each gate**
and report status — do not poll indefinitely or fabricate approval. Resume when I tell
you the gate has passed.

Phase A — open the RC (do now):

1. From an up-to-date `develop`, create branch `rc/v$ARGUMENTS`.
2. Push it and open a PR with **base `main`, head `rc/v$ARGUMENTS`**.
   **Do not bump the version or finalize the changelog yet.**
3. Report the PR URL and **stop**: waiting for CI + code review.

Phase B — review loop (when I report review feedback):

4. Address the review comments. Stage only the files you changed (**never `git add -A`**),
   commit, push. Re-summarize and **stop** until the PR is approved.

Phase C — release bump (only once I confirm the PR is approved):

5. Finalize `CHANGELOG.md`: add `## [$ARGUMENTS] - <today>` (move the `[Unreleased]`
   items into it) and update the compare links at the bottom.
6. Bump `ipax.__version__` to `$ARGUMENTS` (the single version source; pyproject derives it).
7. Run `python scripts/check.py` and `pre-commit run kacl-verify --files CHANGELOG.md`.
   Commit, push, and **stop**: waiting for CI to go green and the PR to be merged.

Phase D — tag & sync (once I confirm the PR is merged):

8. `git switch main && git pull --ff-only origin main`; confirm `__version__` is `$ARGUMENTS`.
9. Create the annotated tag: `git tag -a v$ARGUMENTS main` with a short summary drawn
   from the `[$ARGUMENTS]` changelog section.
10. Merge `main` into `develop` (`--no-ff`).
11. Push both: `git push origin develop && git push origin v$ARGUMENTS`. The tag push
    triggers `release.yml` (PyPI + GitHub release) — report that it has started.

If `$ARGUMENTS` is empty or not `X.Y.Z`, ask me for the version before doing anything.
Stop and ask if any step deviates from `AGENTS.md` rather than improvising.
