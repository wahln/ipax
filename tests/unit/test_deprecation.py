"""The deprecation helpers in ``ipax._deprecation``.

They carry the Road-to-1.0 promise that a renamed public name keeps working,
with a warning, for at least one minor release. User-facing names warn with
``FutureWarning`` (shown by default); implementer-facing protocol names with
``DeprecationWarning``.
"""

from __future__ import annotations

import dataclasses
import pickle
import types
import warnings
from pathlib import Path

import pytest

from ipax._deprecation import (
    deprecated_module_attributes,
    renamed_fields,
    warn_deprecated,
)


def _from_here(record) -> bool:
    # samefile: the drive letter's case can differ on Windows.
    return all(Path(w.filename).samefile(__file__) for w in record)


def test_warn_deprecated_names_both_versions_and_points_at_the_caller():
    with pytest.warns(
        FutureWarning, match=r"old_name.*0\.14.*new_name.*0\.16"
    ) as record:
        warn_deprecated("old_name", "new_name", since="0.14", remove_in="0.16")
    assert _from_here(record)


def test_warn_deprecated_accepts_an_implementer_facing_category():
    with pytest.warns(DeprecationWarning):
        warn_deprecated(
            "hook",
            "other_hook",
            since="0.14",
            remove_in="0.16",
            category=DeprecationWarning,
        )


@renamed_fields(since="0.14", remove_in="0.16", old_value="value")
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Renamed:
    value: float = 1.0
    other: int = 0


def test_renamed_field_old_keyword_warns_and_builds_the_same_object():
    with pytest.warns(FutureWarning, match="old_value") as record:
        built = _Renamed(old_value=2.0, other=3)
    assert built == _Renamed(value=2.0, other=3)
    assert _from_here(record)


def test_renamed_field_new_keyword_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _Renamed(value=2.0).value == 2.0


def test_renamed_field_works_through_dataclasses_replace():
    # dataclasses.replace passes every current field by its new name, so the
    # old keyword must win; pyRadPlan edits Options this way.
    base = _Renamed(value=1.0, other=5)
    with pytest.warns(FutureWarning) as record:
        changed = dataclasses.replace(base, old_value=4.0)
    assert changed == _Renamed(value=4.0, other=5)
    assert _from_here(record)


def test_renamed_field_old_attribute_still_reads():
    built = _Renamed(value=7.0)
    with pytest.warns(FutureWarning, match="old_value"):
        assert built.old_value == 7.0


def test_renamed_class_still_pickles_and_hashes():
    built = _Renamed(value=3.0, other=1)
    assert pickle.loads(pickle.dumps(built)) == built
    assert hash(built) == hash(_Renamed(value=3.0, other=1))


def test_renamed_fields_rejects_an_unknown_target():
    with pytest.raises(TypeError, match="no field 'missing'"):
        renamed_fields(since="0.14", remove_in="0.16", old="missing")(_Renamed)


def test_deprecated_module_attributes_resolve_with_a_warning():
    module = types.ModuleType("fake")
    module.NewThing = object()
    module.__getattr__ = deprecated_module_attributes(
        "fake",
        {"OldThing": "NewThing"},
        namespace=vars(module),
        since="0.14",
        remove_in="0.16",
    )
    with pytest.warns(FutureWarning, match=r"fake\.OldThing.*fake\.NewThing"):
        assert module.OldThing is module.NewThing
    with pytest.raises(AttributeError, match="Missing"):
        _ = module.Missing
