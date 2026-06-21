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
