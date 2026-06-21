---
description: Implement a change test-first, multi-backend, respecting the invariants.
---

Implement the following using the project's mandated TDD loop: $ARGUMENTS

Follow this order strictly:

1. **Red.** Write the failing test(s) first in the right `tests/` subdirectory,
   parametrized over the namespace fixture (NumPy + Torch + array-api-strict minimum).
   For a new protocol implementation, wire in the matching `tests/contracts/` battery.
   Run them and confirm they fail for the right reason.
2. **Green.** Write the minimal implementation to pass. Hold the five invariants (no
   concrete array library in the core, stay inside the Array API standard, injected
   linear algebra, sparsity as an adapter concern, no global mutable state). Cite the
   paper/eq for any algorithmic step.
3. **Verify.** Run the relevant tests on NumPy **and** Torch; run the derivative-check
   harness if derivatives changed. Then `python scripts/check.py --fast`, and finally the
   full `python scripts/check.py` before declaring done.
4. If you fixed a bug, add a regression test for it.

Stop and ask me if any step appears to require breaking an invariant — propose options
instead of working around it.
