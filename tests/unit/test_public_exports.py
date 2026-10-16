"""Every options dataclass is reachable from the top-level package."""

from __future__ import annotations

import dataclasses
import inspect

import ipax
import ipax.options


def test_all_options_dataclasses_are_exported_from_ipax():
    option_classes = [
        name
        for name, obj in inspect.getmembers(ipax.options, inspect.isclass)
        if dataclasses.is_dataclass(obj)
        and obj.__module__ == "ipax.options"
        and name.endswith("Options")
    ]
    assert option_classes, "no options dataclasses found"
    for name in option_classes:
        assert name in ipax.__all__, f"{name} missing from ipax.__all__"
        assert getattr(ipax, name) is getattr(ipax.options, name)


def test_dunder_all_is_resolvable():
    for name in ipax.__all__:
        getattr(ipax, name)
