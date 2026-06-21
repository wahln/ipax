"""Unit tests: options dataclasses are frozen and carry sane defaults."""

from __future__ import annotations

import dataclasses

import pytest

from ipax.options import (
    AcceptableStoppingOptions,
    KrylovOptions,
    OptimalityConditionOptions,
    Options,
)


def test_options_are_frozen():
    opts = Options()
    with pytest.raises(dataclasses.FrozenInstanceError):
        opts.max_iter = 5  # type: ignore[misc]


def test_default_hessian_is_lbfgs():
    assert Options().hessian == "lbfgs"


def test_default_optimality_matches_spec():
    # default ε_tol = 1e-8 on each KKT component.
    optimality = Options().optimality
    assert optimality.dual_inf_tol == 1e-8
    assert optimality.constr_viol_tol == 1e-8
    assert optimality.compl_inf_tol == 1e-8
    assert optimality.f_tol is None
    assert optimality.kkt_tol == 1e-8


def test_kkt_tol_is_the_smallest_enabled_residual_tolerance():
    optimality = OptimalityConditionOptions(
        dual_inf_tol=1e-6, constr_viol_tol=1e-9, compl_inf_tol=None
    )
    assert optimality.kkt_tol == 1e-9


def test_kkt_tol_falls_back_when_only_f_tol_is_set():
    optimality = OptimalityConditionOptions(
        f_tol=1e-8, dual_inf_tol=None, constr_viol_tol=None, compl_inf_tol=None
    )
    assert optimality.kkt_tol == 1e-8


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dual_inf_tol", -1.0),
        ("constr_viol_tol", 0.0),
        ("compl_inf_tol", -1.0),
        ("f_tol", -1.0),
        ("f_rel_change_tol", -1.0),
    ],
)
def test_condition_options_reject_invalid_tolerances(field, value):
    with pytest.raises(ValueError, match=field):
        OptimalityConditionOptions(**{field: value})
    with pytest.raises(ValueError, match=field):
        AcceptableStoppingOptions(**{field: value})


def test_objective_tolerances_accept_exact_zero():
    optimality = OptimalityConditionOptions(f_tol=0.0, f_rel_change_tol=0.0)
    assert optimality.f_tol == 0.0
    assert optimality.f_rel_change_tol == 0.0


def test_optimality_requires_at_least_one_condition():
    with pytest.raises(ValueError, match="at least one optimality condition"):
        OptimalityConditionOptions(
            f_tol=None,
            f_rel_change_tol=None,
            dual_inf_tol=None,
            constr_viol_tol=None,
            compl_inf_tol=None,
        )


def test_acceptable_stopping_is_disabled_by_default():
    acceptable = Options().acceptable

    assert acceptable.f_tol is None
    assert acceptable.f_rel_change_tol is None
    assert acceptable.dual_inf_tol is None
    assert acceptable.constr_viol_tol is None
    assert acceptable.compl_inf_tol is None


def test_acceptable_requires_positive_n_iter():
    with pytest.raises(ValueError, match="n_iter"):
        AcceptableStoppingOptions(dual_inf_tol=1e-3, n_iter=0)


@pytest.mark.parametrize(("field", "value"), [("max_time", 0.0), ("max_iter", 0)])
def test_options_reject_invalid_limits(field, value):
    with pytest.raises(ValueError, match=field):
        Options(**{field: value})


@pytest.mark.parametrize("method", ["cg", "minres", "gmres"])
def test_krylov_options_accept_supported_methods(method):
    assert KrylovOptions(method=method).method == method  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["bicgstab", "qmr"])
def test_krylov_options_reject_unsupported_methods(method):
    with pytest.raises(ValueError, match="Krylov method"):
        KrylovOptions(method=method)  # type: ignore[arg-type]


@pytest.mark.parametrize("preconditioner", ["none", "jacobi", "lbfgs"])
def test_krylov_options_accept_supported_preconditioners(preconditioner):
    opts = KrylovOptions(preconditioner=preconditioner)  # type: ignore[arg-type]
    assert opts.preconditioner == preconditioner


@pytest.mark.parametrize("preconditioner", ["ilu", "amg"])
def test_krylov_options_reject_unsupported_preconditioners(preconditioner):
    with pytest.raises(ValueError, match="Krylov preconditioner"):
        KrylovOptions(preconditioner=preconditioner)  # type: ignore[arg-type]


def test_krylov_options_reject_nonpositive_gmres_restart():
    with pytest.raises(ValueError, match="gmres_restart"):
        KrylovOptions(gmres_restart=0)
