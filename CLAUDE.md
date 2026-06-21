# CLAUDE.md

The canonical instructions for this repository live in **[AGENTS.md](./AGENTS.md)**.
They are imported below so Claude Code loads them automatically:

@AGENTS.md

---

## Claude-specific notes

- Treat `AGENTS.md` as the single source of truth. If guidance here ever conflicts
  with it, `AGENTS.md` wins — and update one of the two files so they agree again.
- Before non-trivial work, read the math conventions and KKT reduction in
  `AGENTS.md`, and the source papers they cite (full citations in the AGENTS.md
  References section).
- The five **non-negotiable invariants** in `AGENTS.md` (no concrete array library
  in the core, stay inside the Array API standard, injected linear algebra, sparsity
  as an adapter concern, no global mutable state) are hard constraints. If a task
  seems to need breaking one, say so and propose options instead of working around
  it silently.
- When you touch numerics, add or run the **multi-backend** tests (NumPy **and**
  PyTorch at minimum) and the derivative-check harness before declaring done.
- Cite the relevant paper/equation in code comments when implementing an algorithmic
  step (e.g. `# Wächter & Biegler 2006, eq. (19)` / `# Breedveld 2017, eq. (18)`).
- Consult the source papers (full citations with DOIs in the AGENTS.md References
  section) rather than guessing at the algorithm details.
