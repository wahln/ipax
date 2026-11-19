# Copyright 2026 Niklas Wahl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deprecation helpers: keep a renamed public name working, with a warning.

The public-API policy promises that a renamed or removed public name keeps
working for at least one minor release. User-facing names warn with
``FutureWarning``, which Python shows by default; names only extension authors
use (protocol hooks) warn with ``DeprecationWarning``. Every warning names the
user's line, not ipax's (:func:`ipax.errors.warn_external`).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Any

from ipax.errors import warn_external

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def warn_deprecated(
    old: str,
    new: str,
    *,
    since: str,
    remove_in: str,
    category: type[Warning] = FutureWarning,
) -> None:
    """Warn that ``old`` is deprecated in favour of ``new``."""
    warn_external(
        f"{old} is deprecated since ipax {since}; use {new} instead. "
        f"It will be removed in ipax {remove_in}.",
        category,
    )


def renamed_fields(
    *, since: str, remove_in: str, **old_to_new: str
) -> Callable[[type], type]:
    """Class decorator: accept and read deprecated names of dataclass fields.

    Apply it above ``@dataclass(frozen=True, slots=True, kw_only=True)``. The
    old keyword is mapped onto the new field and *wins* when both are passed:
    :func:`dataclasses.replace` re-passes every field under its current name,
    so ``replace(options, old_name=...)`` only takes effect this way. Reading
    ``instance.old_name`` returns the new field. Both warn.
    """

    def decorate(cls: type) -> type:
        names = {f.name for f in dataclasses.fields(cls)}
        for new in old_to_new.values():
            if new not in names:
                raise TypeError(f"{cls.__name__} has no field {new!r}")
        init = vars(cls)["__init__"]  # the @dataclass-generated one

        @functools.wraps(init)
        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            for old, new in old_to_new.items():
                if old in kwargs:
                    warn_deprecated(
                        f"{cls.__name__}({old}=...)",
                        f"{cls.__name__}({new}=...)",
                        since=since,
                        remove_in=remove_in,
                    )
                    kwargs[new] = kwargs.pop(old)
            init(self, *args, **kwargs)

        cls.__init__ = __init__  # type: ignore[misc]
        for old, new in old_to_new.items():
            setattr(
                cls, old, _deprecated_alias(cls.__name__, old, new, since, remove_in)
            )
        return cls

    return decorate


def _deprecated_alias(
    owner: str, old: str, new: str, since: str, remove_in: str
) -> property:
    def read(self: Any) -> Any:
        warn_deprecated(
            f"{owner}.{old}", f"{owner}.{new}", since=since, remove_in=remove_in
        )
        return getattr(self, new)

    return property(read)


def deprecated_module_attributes(
    module_name: str,
    old_to_new: Mapping[str, str],
    *,
    namespace: Mapping[str, Any],
    since: str,
    remove_in: str,
    category: type[Warning] = FutureWarning,
) -> Callable[[str], Any]:
    """Build a module ``__getattr__`` (PEP 562) serving deprecated names.

    Use as ``__getattr__ = deprecated_module_attributes(__name__, {...},
    namespace=globals(), ...)`` at the bottom of the module that owns the new
    names.
    """

    def __getattr__(name: str) -> Any:
        new = old_to_new.get(name)
        if new is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        warn_deprecated(
            f"{module_name}.{name}",
            f"{module_name}.{new}",
            since=since,
            remove_in=remove_in,
            category=category,
        )
        return namespace[new]

    return __getattr__
