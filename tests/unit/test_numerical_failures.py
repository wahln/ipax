"""The numerical-failure classifier behind the δ_w ladder.

``numerical_failure_types()`` is the exception set the KKT solve sites may
translate into :class:`LinearSolveError` (and so feed the IPM's regularization
ladder / step-failure restoration, Wächter & Biegler 2006 §3.1 → §3.3). It must
cover every backend's genuine linear-algebra failure and Python's arithmetic
errors, and nothing else: a ``TypeError`` from a user operator is a bug, not a
singular matrix, and has to propagate unchanged.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from ipax.backend.errors import backend_linalg_error_types
from ipax.linalg.solver import LinearSolveError, numerical_failure_types
from tests._helpers import array, backend_linalg_error


def test_backend_singular_solve_is_a_numerical_failure(namespace):
    exc = backend_linalg_error(namespace)
    assert isinstance(exc, numerical_failure_types())
    assert isinstance(exc, backend_linalg_error_types())


def test_backend_cholesky_breakdown_is_a_numerical_failure(namespace):
    indefinite = array(namespace, [[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(numerical_failure_types()):
        namespace.linalg.cholesky(indefinite)


@pytest.mark.parametrize(
    "exc",
    [
        LinearSolveError("classified"),
        FloatingPointError("overflow encountered"),
        ZeroDivisionError("float division by zero"),
        OverflowError("(34, 'Result too large')"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_ipax_and_arithmetic_failures_are_numerical(exc):
    assert isinstance(exc, numerical_failure_types())


@pytest.mark.parametrize(
    "exc",
    [
        TypeError("asarray() got an unexpected keyword argument 'device'"),
        AttributeError("'NoneType' object has no attribute 'shape'"),
        ValueError("shapes (3,) and (4,) not aligned"),
        RuntimeError("mat1 and mat2 shapes cannot be multiplied"),
        KeyError("rows"),
        IndexError("index 7 is out of bounds"),
        NotImplementedError("capability absent"),
        MemoryError(),
    ],
    ids=lambda e: type(e).__name__,
)
def test_programming_and_resource_errors_are_not_numerical(exc):
    assert not isinstance(exc, numerical_failure_types())


def test_classifier_never_imports_a_backend():
    # The lookup reads only libraries that are already loaded (an exception of
    # an unloaded library's type cannot exist), so classifying on a NumPy run
    # must not pay a multi-second Torch/CuPy/JAX import.
    code = (
        "import sys, numpy\n"
        "from ipax.linalg.solver import numerical_failure_types\n"
        "numerical_failure_types()\n"
        "print(sorted(m for m in ('torch', 'cupy', 'jax') if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == "[]"


# -- the gram() keyword-compat retry absorbs signature rejections only ------


def _raised(call) -> TypeError:
    try:
        call()
    except TypeError as exc:
        return exc
    raise AssertionError("expected a TypeError")


def _old_gram(weights):
    return weights


def _new_gram_with_a_bug(weights, *, accumulate_dtype=None, hinted_only=False):
    raise TypeError("asarray() got an unexpected keyword argument 'device'")


def _forwarding_wrapper(*args, **kwargs):
    return _old_gram(*args, **kwargs)


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        # Pre-keyword Python signature: rejected at the call boundary.
        (lambda: _old_gram(1.0, accumulate_dtype="float32"), True),
        # Compiled gram: "len() takes no keyword arguments" names no keyword,
        # but binding fails at the call boundary all the same.
        (lambda: len([1.0], accumulate_dtype="float32"), True),
        # A **kwargs wrapper over the old signature: one frame down, but the
        # message names the keyword.
        (lambda: _forwarding_wrapper(1.0, accumulate_dtype="float32"), True),
        # A Gram that accepted the keywords and then crashed: a bug.
        (lambda: _new_gram_with_a_bug(1.0, accumulate_dtype="float32"), False),
    ],
    ids=["python-signature", "builtin-signature", "kwargs-wrapper", "bug-inside"],
)
def test_gram_keyword_rejection_recognizer(call, expected):
    from ipax.backend.operators import _gram_rejects_keywords

    # Raised one frame below the recognizer's caller, exactly as in
    # ``forward_gram``: the lambda plays the ``op.gram(...)`` call site.
    exc = _raised(call)
    assert _gram_rejects_keywords(exc.with_traceback(exc.__traceback__.tb_next)) is (
        expected
    )
