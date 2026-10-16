"""Small Array-API test helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest


@contextmanager
def implemented(reason: str) -> Iterator[None]:
    """Treat not-yet-implemented stubs as xfails, while testing implemented code normally."""
    try:
        yield
    except NotImplementedError as exc:
        pytest.xfail(f"{reason}: {exc}")


def backend_linalg_error(xp: Any) -> BaseException:
    """The exception ``xp`` itself raises on an exactly singular ``linalg.solve``.

    A genuine backend numerical failure (``numpy.linalg.LinAlgError`` on
    NumPy/array-api-strict, ``torch.linalg.LinAlgError`` on Torch) to inject
    where a test needs one, without naming a concrete library. Skips on a
    backend that returns non-finite values instead of raising (JAX).
    """
    singular = array(xp, [[1.0, 1.0], [1.0, 1.0]])
    try:
        xp.linalg.solve(singular, array(xp, [1.0, 2.0]))
    except Exception as exc:  # whatever the backend raises
        return exc
    pytest.skip(f"{xp.__name__} returns non-finite values instead of raising")


def to_numpy(value: Any) -> Any:
    """A host NumPy array holding ``value`` (a backend array, or anything
    ``numpy.asarray`` accepts).

    ``numpy.asarray`` on a CuPy array raises ("Implicit conversion to a NumPy
    array is not allowed") — the device-to-host copy must be explicit — so
    tests that compare against a NumPy reference go through this instead.
    Everything else keeps the plain ``numpy.asarray`` conversion.
    """
    import numpy as np

    if hasattr(value, "__cuda_array_interface__") and hasattr(value, "get"):
        return np.asarray(value.get())  # CuPy: explicit device→host copy
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return np.asarray(value.detach().cpu())  # Torch, possibly on CUDA
    return np.asarray(value)


def float_dtype(xp: Any) -> Any:
    return getattr(xp, "float64", getattr(xp, "float32", None))


def array(xp: Any, value: object) -> Any:
    dtype = float_dtype(xp)
    if dtype is None:
        return xp.asarray(value)
    return xp.asarray(value, dtype=dtype)


def transpose(xp: Any, value: Any) -> Any:
    return xp.permute_dims(value, (1, 0))


def to_float(value: Any) -> float:
    return float(value)


def norm_inf(xp: Any, value: Any) -> float:
    return to_float(xp.max(xp.abs(value)))


def assert_allclose(
    xp: Any,
    actual: Any,
    expected: Any,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> None:
    diff = xp.abs(actual - expected)
    limit = atol + rtol * xp.abs(expected)
    assert bool(xp.all(diff <= limit)), (
        f"max abs diff {norm_inf(xp, diff)} exceeds tolerance"
    )


def assert_scalar_close(
    actual: Any,
    expected: float,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> None:
    actual_float = to_float(actual)
    assert abs(actual_float - expected) <= atol + rtol * abs(expected)


def central_gradient(
    xp: Any,
    f: Callable[[Any], Any],
    x: Any,
    *,
    step: float = 1e-6,
) -> Any:
    values = []
    n = int(x.shape[0])
    for idx in range(n):
        direction = array(xp, [1.0 if j == idx else 0.0 for j in range(n)])
        f_plus = f(x + step * direction)
        f_minus = f(x - step * direction)
        values.append((to_float(f_plus) - to_float(f_minus)) / (2.0 * step))
    return array(xp, values)
