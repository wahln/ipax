"""Construction contract of the public option and result dataclasses.

All of them are keyword-only, so fields can be added or reordered in a minor
release without breaking callers (API review D3). The classes holding arrays
(``Result``, ``IterationInfo``, ``WarmStart``) compare and hash by identity:
a field-wise ``==`` would compare arrays, and the generated ``__hash__``
raised ``TypeError``.
"""

from __future__ import annotations

import copy
import dataclasses
import pickle

import numpy as np
import pytest

import ipax

OPTION_CLASSES = [
    ipax.Options,
    ipax.OptimalityConditionOptions,
    ipax.AcceptableStoppingOptions,
    ipax.ScalingOptions,
    ipax.CorrectionsOptions,
    ipax.BarrierOptions,
    ipax.LineSearchOptions,
    ipax.RegularizationOptions,
    ipax.LBFGSOptions,
    ipax.KrylovOptions,
    ipax.RestorationOptions,
    ipax.SparseOptions,
    ipax.BreedveldOptions,
    ipax.DenseOptions,
]
RESULT_CLASSES = [
    ipax.KKTResiduals,
    ipax.IterationRecord,
    ipax.IterationInfo,
    ipax.DerivativeSources,
    ipax.Routes,
    ipax.Result,
    ipax.WarmStart,
]
IDENTITY_CLASSES = {ipax.Result, ipax.IterationInfo, ipax.WarmStart}


@pytest.mark.parametrize(
    "cls", OPTION_CLASSES + RESULT_CLASSES, ids=lambda c: c.__name__
)
def test_every_public_dataclass_is_keyword_only(cls):
    fields = [f for f in dataclasses.fields(cls) if f.init]
    assert fields, cls
    assert all(f.kw_only for f in fields), cls
    assert cls.__match_args__ == ()


@pytest.mark.parametrize("cls", OPTION_CLASSES, ids=lambda c: c.__name__)
def test_option_classes_reject_positional_arguments(cls):
    first = next(f for f in dataclasses.fields(cls) if f.init)
    with pytest.raises(TypeError):
        cls(getattr(cls(), first.name))


@pytest.mark.parametrize("cls", OPTION_CLASSES, ids=lambda c: c.__name__)
def test_option_classes_keep_value_semantics(cls):
    built = cls()
    assert built == cls()
    assert hash(built) == hash(cls())
    assert pickle.loads(pickle.dumps(built)) == built
    assert copy.deepcopy(built) == built
    assert dataclasses.replace(built) == built


def _solved() -> ipax.Result:
    problem = ipax.QuadraticProblem(np.diag([2.0, 3.0]), np.array([1.0, -1.0]))
    return ipax.solve(problem, np.zeros(2))


def test_results_compare_and_hash_by_identity():
    first, second = _solved(), _solved()
    assert first == first
    assert first != second  # equal contents, distinct solves
    assert len({first, second}) == 2  # hashable


def test_warm_start_compares_and_hashes_by_identity():
    result = _solved()
    first, second = (
        ipax.WarmStart.from_result(result),
        ipax.WarmStart.from_result(result),
    )
    assert first != second
    assert len({first, second}) == 2


@pytest.mark.parametrize(
    "cls", sorted(IDENTITY_CLASSES, key=lambda c: c.__name__), ids=lambda c: c.__name__
)
def test_identity_classes_have_no_generated_eq(cls):
    assert cls.__eq__ is object.__eq__
