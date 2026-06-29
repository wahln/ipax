"""The sparse facade caches COO structure and recomputes only values (#2).

On a fixed ``coo_pattern_signature`` the ``SparseDirectSolver`` facade must reuse
the cached row/column vectors and call ``coo_values`` instead of rebuilding the
full triplet with ``to_coo`` — while a ``None`` signature (value-dependent
pattern) keeps rebuilding every factor. Driven on NumPy/SciPy in CI.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scipy.sparse")

from ipax.backend.operators import LinearOperator
from ipax.linalg.sparse import SparseDirectSolver
from ipax.testing.backends import import_namespace
from tests._helpers import array, assert_allclose

pytestmark = pytest.mark.sparse


class _CountingOperator(LinearOperator):
    """A fixed-pattern symmetric COO operator counting structure/value rebuilds."""

    def __init__(self, xp, values, *, signature):
        self._xp = xp
        self._values = values
        self._signature = signature
        self.to_coo_calls = 0
        self.coo_values_calls = 0
        # A = [[v0, v1], [v1, v2]] from a symmetric COO pattern.
        self._rows = xp.asarray([0, 0, 1, 1])
        self._cols = xp.asarray([0, 1, 0, 1])

    @property
    def shape(self):
        return 2, 2

    def matvec(self, v):  # pragma: no cover - direct route never calls matvec
        raise NotImplementedError

    def to_coo(self, like=None):
        del like
        self.to_coo_calls += 1
        return self._rows, self._cols, self._values, (2, 2)

    def coo_values(self, like=None):
        del like
        self.coo_values_calls += 1
        return self._values

    def coo_pattern_signature(self):
        return self._signature

    def symmetry_hint(self):
        return True


def _numpy_namespace():
    try:
        return import_namespace("numpy")
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"NumPy namespace unavailable: {exc}")


def test_facade_reuses_structure_and_recomputes_values_on_signature_hit():
    xp = _numpy_namespace()
    facade = SparseDirectSolver()

    first = _CountingOperator(xp, array(xp, [2.0, 1.0, 1.0, 3.0]), signature=("kkt",))
    facade.factor(first)
    # A = [[2,1],[1,3]] · x = [1,2] ⇒ x = [1/5, 3/5].
    assert_allclose(xp, facade.solve(array(xp, [1.0, 2.0])), array(xp, [0.2, 0.6]))

    second = _CountingOperator(xp, array(xp, [5.0, 1.0, 1.0, -2.0]), signature=("kkt",))
    facade.factor(second)
    # A = [[5,1],[1,-2]] · x = [1,2], det = -11 ⇒ x = [4/11, -9/11].
    assert_allclose(
        xp, facade.solve(array(xp, [1.0, 2.0])), array(xp, [4.0 / 11.0, -9.0 / 11.0])
    )

    # Second factor reused the cached structure: values only, no re-emitted triplet.
    assert first.to_coo_calls == 1
    assert second.to_coo_calls == 0
    assert second.coo_values_calls == 1


def test_facade_rebuilds_when_signature_is_none():
    xp = _numpy_namespace()
    facade = SparseDirectSolver()

    first = _CountingOperator(xp, array(xp, [2.0, 1.0, 1.0, 3.0]), signature=None)
    facade.factor(first)
    second = _CountingOperator(xp, array(xp, [4.0, 1.0, 1.0, 5.0]), signature=None)
    facade.factor(second)

    # No signature ⇒ no structure cache ⇒ every factor rebuilds the full triplet.
    assert second.to_coo_calls == 1
    assert second.coo_values_calls == 0


def test_facade_rebuilds_when_signature_changes():
    xp = _numpy_namespace()
    facade = SparseDirectSolver()

    facade.factor(_CountingOperator(xp, array(xp, [2.0, 1.0, 1.0, 3.0]), signature="a"))
    changed = _CountingOperator(xp, array(xp, [4.0, 1.0, 1.0, 5.0]), signature="b")
    facade.factor(changed)

    assert changed.to_coo_calls == 1
    assert changed.coo_values_calls == 0
