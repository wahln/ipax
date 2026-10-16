"""An optional cuDSS mode must not make supported installations fail tests."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from tests._cudss import require_deterministic_cudss


class CuDSSError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"cuDSS status {status}")
        self.status = status


class Binding:
    ConfigParam = SimpleNamespace(DETERMINISTIC_MODE=25)
    cuDSSError = CuDSSError

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.created = 0
        self.destroyed = []
        self.values = []

    def config_create(self):
        self.created += 1
        return 17

    def config_set(self, config, param, address, size):
        assert (config, param, size) == (17, 25, 4)
        if self.error is not None:
            raise self.error
        self.values.append(ctypes.c_int32.from_address(address).value)

    def config_destroy(self, config):
        self.destroyed.append(config)


@pytest.mark.parametrize("params", [None, SimpleNamespace()])
def test_missing_determinism_parameter_skips_without_creating_config(params):
    binding = Binding()
    binding.ConfigParam = params
    with pytest.raises(pytest.skip.Exception, match="DETERMINISTIC_MODE"):
        require_deterministic_cudss(binding)
    assert binding.created == 0


@pytest.mark.parametrize(
    "error",
    [CuDSSError(3), CuDSSError(4), TypeError("old signature"), ValueError("bad enum")],
)
def test_unsupported_determinism_skips_and_releases_config(error):
    binding = Binding(error)
    with pytest.raises(pytest.skip.Exception, match="DETERMINISTIC_MODE"):
        require_deterministic_cudss(binding)
    assert binding.destroyed == [17]


def test_supported_determinism_is_enabled_and_config_is_released():
    binding = Binding()
    require_deterministic_cudss(binding)
    assert binding.values == [1]
    assert binding.destroyed == [17]


def test_unexpected_runtime_failure_is_not_hidden_as_a_skip():
    binding = Binding(CuDSSError(2))
    with pytest.raises(CuDSSError):
        require_deterministic_cudss(binding)
    assert binding.destroyed == [17]
