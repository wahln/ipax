"""Every ``Options`` field is validated on construction (API review A2/A3/A8).

A misspelled selector used to run silently as the default (``hessian="bogus"``
as ``"auto"``), turn a safeguard off (``fallback="kkt_error"``), or fail deep
inside the loop (``mu_schedule="bogus"`` as a bare ``AssertionError``). Every
field is now checked against its annotation, and the numerical ones against
the range their source defines. Failures raise
:class:`ipax.ConfigurationError`, a ``ValueError``.
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import pytest

import ipax
import ipax.options as options_module

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


def _fields():
    for cls in OPTION_CLASSES:
        for f in dataclasses.fields(cls):
            if f.init:
                yield pytest.param(cls, f.name, id=f"{cls.__name__}.{f.name}")


def _literal_fields():
    for cls in OPTION_CLASSES:
        hints = typing.get_type_hints(cls, vars(options_module))
        for f in dataclasses.fields(cls):
            hint = hints[f.name]
            args = typing.get_args(hint) if typing.get_origin(hint) else ()
            literal = typing.get_origin(hint) is typing.Literal or any(
                typing.get_origin(a) is typing.Literal for a in args
            )
            if literal:
                yield pytest.param(cls, f.name, id=f"{cls.__name__}.{f.name}")


def test_configuration_error_is_a_value_error():
    with pytest.raises(ValueError):
        ipax.Options(max_iter=0)


@pytest.mark.parametrize(("cls", "name"), list(_fields()))
def test_every_field_rejects_a_value_of_the_wrong_type(cls, name):
    with pytest.raises(ipax.ConfigurationError, match=rf"{cls.__name__}\.{name}\b"):
        cls(**{name: object()})


@pytest.mark.parametrize(("cls", "name"), list(_literal_fields()))
def test_every_string_choice_rejects_an_unknown_value(cls, name):
    with pytest.raises(ipax.ConfigurationError, match=r"one of .*got 'bogus'"):
        cls(**{name: "bogus"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hessian": "bogus"},
        {"linsolve": "krylv"},
        {"globalization": "bogus"},
        {"mu_schedule": "bogus"},
        {"scaling": "gradient_based"},
        {"corrections": "Mehrotra"},
    ],
    ids=lambda kw: next(iter(kw)),
)
def test_misspelled_top_level_selectors_are_rejected(kwargs):
    with pytest.raises(ipax.ConfigurationError):
        ipax.Options(**kwargs)


def test_underscore_spelling_of_the_barrier_fallback_is_rejected_not_ignored():
    # "kkt_error" used to switch the NWW §5.1 safeguard off silently.
    with pytest.raises(ipax.ConfigurationError, match="'kkt-error'"):
        ipax.BarrierOptions(fallback="kkt_error")
    with pytest.raises(ipax.ConfigurationError, match="'obj-constr-filter'"):
        ipax.LineSearchOptions(free_mode_acceptance="obj_constr_filter")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_autodiff": "no"},
        {"enable_finite_diff": 1},
        {"max_iter": 5.5},
        {"max_iter": True},
        {"verbose": "x"},
        {"max_time": float("nan")},
        {"optimality": None},
        {"krylov": ipax.LBFGSOptions()},
    ],
    ids=lambda kw: f"{next(iter(kw))}={next(iter(kw.values()))!r}",
)
def test_wrong_types_are_rejected(kwargs):
    with pytest.raises(ipax.ConfigurationError):
        ipax.Options(**kwargs)


def test_numeric_fields_accept_integers_and_numpy_scalars():
    ipax.BarrierOptions(mu_init=1, tau_min=np.float64(0.9))
    ipax.Options(max_iter=np.int64(10), max_time=5)
    ipax.LBFGSOptions(memory=np.int32(5))


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (ipax.Options, {"verbose": -1}),
        # Wächter & Biegler 2006, eqs. (7)-(8): κ_μ ∈ (0,1), θ_μ ∈ (1,2), τ_min ∈ (0,1).
        (ipax.BarrierOptions, {"mu_init": 0.0}),
        (ipax.BarrierOptions, {"mu_init": -1.0}),
        (ipax.BarrierOptions, {"mu_init": float("inf")}),
        (ipax.BarrierOptions, {"mu_min": 0.0}),
        (ipax.BarrierOptions, {"kappa_mu": 0.0}),
        (ipax.BarrierOptions, {"kappa_mu": 1.0}),
        (ipax.BarrierOptions, {"theta_mu": 1.0}),
        (ipax.BarrierOptions, {"theta_mu": 2.0}),
        (ipax.BarrierOptions, {"tau_min": 0.0}),
        (ipax.BarrierOptions, {"tau_min": 1.0}),
        # Wächter & Biegler 2006, eqs. (18)-(20).
        (ipax.LineSearchOptions, {"max_soc": -1}),
        (ipax.LineSearchOptions, {"gamma_theta": 0.0}),
        (ipax.LineSearchOptions, {"gamma_theta": 1.0}),
        (ipax.LineSearchOptions, {"gamma_phi": 0.0}),
        (ipax.LineSearchOptions, {"gamma_phi": 1.0}),
        (ipax.LineSearchOptions, {"s_theta": 1.0}),
        (ipax.LineSearchOptions, {"s_phi": 0.5}),
        (ipax.LineSearchOptions, {"eta_phi": 0.0}),
        (ipax.LineSearchOptions, {"eta_phi": 0.5}),
        # IPOPT's ranges for the Hessian/Jacobian perturbations.
        (ipax.RegularizationOptions, {"delta_w_init": 0.0}),
        (ipax.RegularizationOptions, {"delta_w_init": 1.0, "delta_w_max": 0.5}),
        (ipax.RegularizationOptions, {"delta_w_factor": 1.0}),
        (ipax.RegularizationOptions, {"delta_c": -1e-8}),
        (ipax.RegularizationOptions, {"delta_c": 1.0, "delta_c_max": 0.5}),
        (ipax.RegularizationOptions, {"delta_c_trigger": 0.0}),
        (ipax.LBFGSOptions, {"memory": 0}),
        (ipax.KrylovOptions, {"max_iter": 0}),
        (ipax.BreedveldOptions, {"tau": 0.0}),
        (ipax.BreedveldOptions, {"tau": 1.0}),
        (ipax.BreedveldOptions, {"backtrack": 0.0}),
        (ipax.BreedveldOptions, {"backtrack": 1.0}),
        (ipax.BreedveldOptions, {"max_backtrack": 0}),
        (ipax.BreedveldOptions, {"armijo_c": 0.0}),
        (ipax.BreedveldOptions, {"armijo_c": 1.0}),
        (ipax.BreedveldOptions, {"ratio_limit": 0.0}),
    ],
    ids=lambda v: (
        v.__name__
        if isinstance(v, type)
        else ",".join(f"{k}={x}" for k, x in v.items())
    ),
)
def test_out_of_range_values_are_rejected(cls, kwargs):
    with pytest.raises(ipax.ConfigurationError, match=next(iter(kwargs))):
        cls(**kwargs)


@pytest.mark.parametrize("cls", OPTION_CLASSES, ids=lambda c: c.__name__)
def test_defaults_are_valid(cls):
    cls()


def test_shorthand_strings_still_normalize():
    options = ipax.Options(scaling="none", corrections="mehrotra")
    assert options.scaling == ipax.ScalingOptions(method="none")
    assert options.corrections == ipax.CorrectionsOptions(method="mehrotra")


def test_a_subclass_of_an_option_class_is_validated_too():
    @dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
    class MyBarrier(ipax.BarrierOptions):
        extra: int = 0

    MyBarrier(extra=3)
    with pytest.raises(ipax.ConfigurationError, match=r"MyBarrier\.extra"):
        MyBarrier(extra="x")
