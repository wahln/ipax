"""Unit tests for the O(n) dense diagonal-add backend gap-filler.

``ipax.ipm.kkt._add_dense_diagonal`` used to build ``diag(diagonal)`` via
``xp.eye(n) * diagonal`` -- an extra O(n^2) identity allocation and elementwise
multiply on top of the O(n^2) copy already needed to return a new matrix. The
``get_dense_diagonal_adder`` gap-filler (NumPy, Torch) instead mutates a
writable O(n) diagonal view of a fresh copy. These tests pin: (1) the result is
bitwise identical to the old ``eye``-based formula (a vector diagonal and a
scalar ``delta * I`` term both), and (2) no O(n^2) identity matrix is
materialized on the adapters that have the fast path.
"""

from __future__ import annotations

import numpy as np
import pytest

from ipax.backend.dense import get_dense_diagonal_adder
from ipax.ipm.kkt import _add_dense_diagonal
from tests._helpers import array, to_numpy


def _reference(namespace, matrix, diagonal):
    n = int(matrix.shape[0])
    return matrix + namespace.eye(n, dtype=matrix.dtype) * diagonal


# --- bitwise identity, both call shapes (vector diagonal, scalar delta_w) ----


def test_add_diagonal_gap_filler_matches_eye_formula_vector(namespace):
    add_diagonal = get_dense_diagonal_adder(namespace)
    if add_diagonal is None:
        pytest.skip("backend has no O(n) diagonal-add primitive")
    matrix = array(namespace, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    diagonal = array(namespace, [0.5, -1.25, 3.0])

    fast = add_diagonal(matrix, diagonal)
    reference = _reference(namespace, matrix, diagonal)

    np.testing.assert_array_equal(to_numpy(fast), to_numpy(reference))


def test_add_diagonal_gap_filler_matches_eye_formula_scalar(namespace):
    add_diagonal = get_dense_diagonal_adder(namespace)
    if add_diagonal is None:
        pytest.skip("backend has no O(n) diagonal-add primitive")
    matrix = array(namespace, [[1.0, 2.0], [3.0, 4.0]])
    delta = 1e-6

    fast = add_diagonal(matrix, delta)
    reference = _reference(namespace, matrix, delta)

    np.testing.assert_array_equal(to_numpy(fast), to_numpy(reference))


# Regression test for double rounding: casting ``diagonal`` to ``matrix.dtype``
# unconditionally would convert the *default float64* ``np.asarray(delta)``
# down to float32 only at the very end, instead of rounding the Python float
# to float32 once like the reference ``eye`` formula does (a "weak" scalar
# under NEP 50 / Torch's own promotion rules). ``delta=0.1`` is not exactly
# representable in binary, so it reliably exposes the discrepancy; the
# smaller ``delta=1e-6`` used above happens not to, which is why this needs
# its own case.
def test_add_diagonal_gap_filler_matches_eye_formula_float32_scalar_double_rounding(
    namespace,
):
    add_diagonal = get_dense_diagonal_adder(namespace)
    if add_diagonal is None:
        pytest.skip("backend has no O(n) diagonal-add primitive")
    if not hasattr(namespace, "float32"):
        pytest.skip("backend has no float32")
    matrix = array(namespace, [[1.0, 2.0], [3.0, 4.0]])
    matrix = namespace.astype(matrix, namespace.float32)
    delta = 0.1  # not exactly representable in binary float; exposes rounding

    fast = add_diagonal(matrix, delta)
    reference = _reference(namespace, matrix, delta)

    assert fast.dtype == reference.dtype == matrix.dtype
    np.testing.assert_array_equal(to_numpy(fast), to_numpy(reference))


def test_add_diagonal_gap_filler_promotes_like_eye_formula_for_a_wider_diagonal(
    namespace,
):
    # A diagonal ARRAY (not a Python scalar) with wider precision than the
    # matrix is a "strong" type under NEP 50 / Torch's promotion rules and
    # promotes the whole sum -- unlike a same-valued Python scalar. An
    # unconditional cast to matrix.dtype would silently narrow this case.
    add_diagonal = get_dense_diagonal_adder(namespace)
    if add_diagonal is None:
        pytest.skip("backend has no O(n) diagonal-add primitive")
    if not (hasattr(namespace, "float32") and hasattr(namespace, "float64")):
        pytest.skip("backend needs both float32 and float64")
    matrix = namespace.astype(
        array(namespace, [[1.0, 2.0], [3.0, 4.0]]), namespace.float32
    )
    diagonal = namespace.astype(array(namespace, [0.1, 1.0 / 3.0]), namespace.float64)

    fast = add_diagonal(matrix, diagonal)
    reference = _reference(namespace, matrix, diagonal)

    assert fast.dtype == reference.dtype == namespace.float64
    np.testing.assert_array_equal(to_numpy(fast), to_numpy(reference))


def test_add_diagonal_gap_filler_does_not_mutate_the_input(namespace):
    add_diagonal = get_dense_diagonal_adder(namespace)
    if add_diagonal is None:
        pytest.skip("backend has no O(n) diagonal-add primitive")
    matrix = array(namespace, [[1.0, 2.0], [3.0, 4.0]])
    original = to_numpy(matrix).copy()

    add_diagonal(matrix, array(namespace, [10.0, 20.0]))

    np.testing.assert_array_equal(to_numpy(matrix), original)


# --- _add_dense_diagonal: the core call site, exercised through every backend


def test_kkt_add_dense_diagonal_matches_eye_formula_vector(namespace):
    matrix = array(namespace, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    diagonal = array(namespace, [0.5, -1.25, 3.0])

    actual = _add_dense_diagonal(matrix, diagonal)
    expected = _reference(namespace, matrix, diagonal)

    np.testing.assert_array_equal(to_numpy(actual), to_numpy(expected))


def test_kkt_add_dense_diagonal_matches_eye_formula_scalar_delta(namespace):
    matrix = array(namespace, [[4.0, 0.5], [0.5, 3.0]])
    delta = 1e-6

    actual = _add_dense_diagonal(matrix, delta)
    expected = _reference(namespace, matrix, delta)

    np.testing.assert_array_equal(to_numpy(actual), to_numpy(expected))


def test_kkt_add_dense_diagonal_promotes_for_a_float64_diagonal_on_float32_matrix(
    namespace,
):
    # The dtype-promotion case through the actual call site: a float64
    # diagonal vector against a float32 matrix must promote the whole result
    # to float64, matching the reference ``eye``-based formula exactly.
    if not (hasattr(namespace, "float32") and hasattr(namespace, "float64")):
        pytest.skip("backend needs both float32 and float64")
    matrix = namespace.astype(
        array(namespace, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        namespace.float32,
    )
    diagonal = namespace.astype(
        array(namespace, [0.1, 1.0 / 3.0, -2.5]), namespace.float64
    )

    actual = _add_dense_diagonal(matrix, diagonal)
    expected = _reference(namespace, matrix, diagonal)

    assert actual.dtype == expected.dtype == namespace.float64
    np.testing.assert_array_equal(to_numpy(actual), to_numpy(expected))


# --- no O(n²) identity materialization on the adapters with a fast path -----


def test_add_diagonal_numpy_never_builds_an_nxn_diagonal_matrix(monkeypatch):
    # Guards against a regression to the eye-based formula *or* an equally
    # O(n^2) rewrite via np.diag/np.diagflat (both build a full n x n matrix).
    numpy = pytest.importorskip("numpy")
    add_diagonal = get_dense_diagonal_adder(numpy)
    assert add_diagonal is not None

    def spy(*args, **kwargs):
        pytest.fail("add_diagonal materialized an n x n diagonal matrix")

    for name in ("eye", "diag", "diagflat"):
        monkeypatch.setattr(numpy, name, spy)
    n = 64
    matrix = numpy.arange(n * n, dtype=float).reshape(n, n)
    diagonal = numpy.ones(n)
    add_diagonal(matrix, diagonal)


def test_add_diagonal_torch_never_builds_an_nxn_diagonal_matrix(monkeypatch):
    torch = pytest.importorskip("torch")
    add_diagonal = get_dense_diagonal_adder(torch)
    assert add_diagonal is not None

    def spy(*args, **kwargs):
        pytest.fail("add_diagonal materialized an n x n diagonal matrix")

    for name in ("eye", "diag", "diagflat", "diag_embed"):
        monkeypatch.setattr(torch, name, spy)
    n = 64
    matrix = torch.arange(n * n, dtype=torch.float64).reshape(n, n)
    diagonal = torch.ones(n, dtype=torch.float64)
    add_diagonal(matrix, diagonal)
