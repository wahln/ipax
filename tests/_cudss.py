"""Capability checks for optional real-cuDSS tests."""

from __future__ import annotations

import ctypes
from typing import Any

import pytest


def require_deterministic_cudss(cudss: Any) -> None:
    """Skip if the binding or runtime cannot set deterministic mode.

    Probe a separate config so the tests still verify that the adapter sets
    its own config correctly. Numerical failures and unrelated runtime errors
    must remain failures, even though the adapter may warn and fall back.
    """
    param = getattr(getattr(cudss, "ConfigParam", None), "DETERMINISTIC_MODE", None)
    if param is None:
        pytest.skip("cuDSS bindings lack DETERMINISTIC_MODE")
    config = cudss.config_create()
    try:
        value = ctypes.c_int32(1)
        try:
            cudss.config_set(
                config, int(param), ctypes.addressof(value), ctypes.sizeof(value)
            )
        except cudss.cuDSSError as exc:
            # CUDSS_STATUS_INVALID_VALUE / CUDSS_STATUS_NOT_SUPPORTED:
            # the loaded runtime does not accept this optional parameter.
            if getattr(exc, "status", None) not in (3, 4):
                raise
            pytest.skip(f"cuDSS runtime cannot set DETERMINISTIC_MODE: {exc}")
        except (TypeError, ValueError) as exc:
            pytest.skip(f"cuDSS binding cannot set DETERMINISTIC_MODE: {exc}")
    finally:
        cudss.config_destroy(config)
