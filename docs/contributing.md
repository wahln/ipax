# Contributing

The canonical guide is [`AGENTS.md`](https://github.com/wahln/ipax/blob/main/AGENTS.md).
Highlights:

- **Test-driven, multi-backend.** Write the failing tests first; a change is not
  done until the relevant tests pass on **both** NumPy and PyTorch. Add a
  regression test with every bug fix.
- **Respect the invariants.** No concrete array imports in the core; stay inside
  the Array API standard; inject linear algebra; sparsity is an adapter concern;
  no global mutable state.
- **Cite the math.** When implementing an algorithmic step, cite the paper and
  equation in a comment (e.g. `# Wächter & Biegler 2006, eq. (19)`).

## Local checks

```bash
pip install -e ".[dev]"
pre-commit install

pytest -q
IPAX_BACKENDS=numpy,torch,array_api_strict pytest -q
ruff check . && ruff format --check . && mypy ipax
python scripts/check_purity.py
```
