"""The SciPy adapter reuses a compiled COO→CSC map across fixed-pattern factors.

In an interior-point solve the KKT pattern is fixed and only values move; the
adapter must canonicalize once (per ``pattern_signature``) and then recompute
only the value array, never re-sorting. These run on NumPy/SciPy in CI.
"""

from __future__ import annotations

import pytest

scipy_sparse = pytest.importorskip("scipy.sparse")

from ipax.backend.sparse.numpy_scipy import SciPySparseAdapter  # noqa: E402
from ipax.testing.backends import import_namespace  # noqa: E402
from tests._helpers import array, assert_allclose  # noqa: E402

pytestmark = pytest.mark.sparse


def _numpy_namespace():
    try:
        return import_namespace("numpy")
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"NumPy namespace unavailable: {exc}")


def test_signature_reuse_recomputes_values_without_resorting(monkeypatch):
    xp = _numpy_namespace()
    adapter = SciPySparseAdapter()
    # (0,0) duplicated ⇒ summed on canonicalization; the pattern is symmetric.
    rows = xp.asarray([0, 0, 1, 1, 0])
    cols = xp.asarray([0, 1, 0, 1, 0])
    signature = ("stable-pattern", (2, 2))

    first = adapter.from_coo(
        rows,
        cols,
        array(xp, [2.0, 1.0, 1.0, 3.0, 0.5]),
        shape=(2, 2),
        pattern_signature=signature,
    )
    # Ban the sort that COO→CSC would otherwise run, proving the second build
    # reuses the compiled structure and only scatters new values.
    monkeypatch.setattr(
        scipy_sparse.csc_matrix,
        "sort_indices",
        lambda self: (_ for _ in ()).throw(AssertionError("re-sorted on reuse")),
    )
    second = adapter.from_coo(
        rows,
        cols,
        array(xp, [4.0, 1.0, 1.0, 5.0, 1.0]),
        shape=(2, 2),
        pattern_signature=signature,
    )

    # First matrix A = [[2.5, 1], [1, 3]]; second A = [[5, 1], [1, 5]].
    assert_allclose(xp, first.matvec(array(xp, [1.0, 0.0])), array(xp, [2.5, 1.0]))
    assert_allclose(xp, second.matvec(array(xp, [1.0, 0.0])), array(xp, [5.0, 1.0]))


def test_signature_reuse_solves_correctly_through_feral_or_superlu():
    xp = _numpy_namespace()
    adapter = SciPySparseAdapter()
    solver = adapter.solver()
    rows = xp.asarray([0, 0, 1, 1])
    cols = xp.asarray([0, 1, 0, 1])
    signature = ("kkt", (2, 2))

    solver.factor(
        adapter.from_coo(
            rows,
            cols,
            array(xp, [2.0, 1.0, 1.0, -3.0]),
            shape=(2, 2),
            pattern_signature=signature,
        )
    )
    # Same pattern, new values: A = [[5, 1], [1, -2]], det = -11.
    solver.factor(
        adapter.from_coo(
            rows,
            cols,
            array(xp, [5.0, 1.0, 1.0, -2.0]),
            shape=(2, 2),
            pattern_signature=signature,
        )
    )
    actual = solver.solve(array(xp, [1.0, 2.0]))
    assert_allclose(xp, actual, array(xp, [4.0 / 11.0, -9.0 / 11.0]))


def test_unsignatured_pattern_keeps_per_call_canonicalization():
    # Value-dependent patterns must NOT be cached: with pattern_signature=None the
    # adapter falls back to a from-scratch coo build every call (current behavior).
    xp = _numpy_namespace()
    adapter = SciPySparseAdapter()
    op = adapter.from_coo(
        xp.asarray([0, 1]), xp.asarray([0, 1]), array(xp, [2.0, 4.0]), shape=(2, 2)
    )
    assert_allclose(xp, op.matvec(array(xp, [1.0, 1.0])), array(xp, [2.0, 4.0]))


def test_signature_recompiles_when_structure_changes_under_same_key():
    # A grown pattern (e.g. the L-BFGS border adding columns) reuses the signature
    # key but has more triplets/shape; the adapter must recompile, not misapply the
    # stale map.
    xp = _numpy_namespace()
    adapter = SciPySparseAdapter()
    signature = ("growing",)

    small = adapter.from_coo(
        xp.asarray([0, 1]),
        xp.asarray([0, 1]),
        array(xp, [2.0, 3.0]),
        shape=(2, 2),
        pattern_signature=signature,
    )
    big = adapter.from_coo(
        xp.asarray([0, 1, 2]),
        xp.asarray([0, 1, 2]),
        array(xp, [2.0, 3.0, 4.0]),
        shape=(3, 3),
        pattern_signature=signature,
    )
    assert small.shape == (2, 2)
    assert big.shape == (3, 3)
    assert_allclose(
        xp, big.matvec(array(xp, [1.0, 1.0, 1.0])), array(xp, [2.0, 3.0, 4.0])
    )
