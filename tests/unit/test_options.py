"""Unit tests: options dataclasses are frozen and carry sane defaults."""

from __future__ import annotations

import dataclasses

import pytest

from ipax.options import (
    AcceptableStoppingOptions,
    DenseOptions,
    KrylovOptions,
    LineSearchOptions,
    OptimalityConditionOptions,
    Options,
)


def test_options_are_frozen():
    opts = Options()
    with pytest.raises(dataclasses.FrozenInstanceError):
        opts.max_iter = 5  # type: ignore[misc]


def test_default_hessian_is_auto():
    assert Options().hessian == "auto"


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


def test_acceptable_stopping_defaults_to_the_ipopt_convention():
    # IPOPT enables acceptable-level termination by default (acceptable_tol =
    # 1e-6 = 1e2 x the optimality tol, acceptable_iter = 15). A disabled-by-
    # default gate lets a solve whose achievable KKT floor sits between 1e-8
    # and 1e-6 grind thousands of iterations into max_time at an essentially
    # optimal point (S2MPJ v6: PALMER1A, 2431 iterations at kkt 4.2e-8).
    acceptable = Options().acceptable

    assert acceptable.f_tol is None
    assert acceptable.f_rel_change_tol is None
    assert acceptable.dual_inf_tol == 1e-6
    assert acceptable.constr_viol_tol == 1e-6
    assert acceptable.compl_inf_tol == 1e-6
    assert acceptable.n_iter == 15


def test_acceptable_stopping_can_be_disabled():
    acceptable = AcceptableStoppingOptions(
        dual_inf_tol=None, constr_viol_tol=None, compl_inf_tol=None
    )
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


@pytest.mark.parametrize("preconditioner", ["none", "jacobi", "lbfgs", "auto"])
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


@pytest.mark.parametrize("ratio", [0.0, -0.5, 1.5])
def test_krylov_options_reject_out_of_range_auto_switch_ratio(ratio):
    with pytest.raises(ValueError, match="auto_switch_ratio"):
        KrylovOptions(auto_switch_ratio=ratio)


def test_krylov_options_accept_auto_switch_ratio_in_range():
    assert KrylovOptions(auto_switch_ratio=0.25).auto_switch_ratio == 0.25


def test_krylov_options_reject_nonpositive_adaptive_eta():
    with pytest.raises(ValueError, match="adaptive_eta"):
        KrylovOptions(adaptive_eta=0.0)


@pytest.mark.parametrize("cap", [1e-12, 2.0])  # <= rtol floor, and > 1
def test_krylov_options_reject_out_of_range_adaptive_rtol_max(cap):
    with pytest.raises(ValueError, match="adaptive_rtol_max"):
        KrylovOptions(rtol=1e-10, adaptive_rtol_max=cap)


def test_krylov_options_adaptive_defaults_on():
    opts = KrylovOptions()
    assert opts.adaptive_tol is True
    assert 0.0 < opts.adaptive_eta and opts.rtol < opts.adaptive_rtol_max <= 1.0


def test_dense_options_default_is_condensed():
    assert DenseOptions().kkt_route == "condensed"
    assert Options().dense.kkt_route == "condensed"


def test_dense_options_accepts_augmented():
    assert DenseOptions(kkt_route="augmented").kkt_route == "augmented"


@pytest.mark.parametrize("route", ["sparse", "bordered", ""])
def test_dense_options_rejects_unsupported_kkt_route(route):
    with pytest.raises(ValueError, match="kkt_route"):
        DenseOptions(kkt_route=route)  # type: ignore[arg-type]


def test_dense_options_augmented_max_size_default_and_validation():
    assert DenseOptions().augmented_max_size > 0
    assert DenseOptions(augmented_max_size=5).augmented_max_size == 5
    with pytest.raises(ValueError, match="augmented_max_size"):
        DenseOptions(augmented_max_size=0)


def test_line_search_feasible_kkt_progress_defaults_to_disabled():
    # Opt-in: as a default the rescue cost 48 S2MPJ corpus flips (nonconvex
    # θ ≡ 0 problems walking into worse stationary points).
    assert LineSearchOptions().feasible_kkt_progress is None
    assert LineSearchOptions(feasible_kkt_progress=0.1).feasible_kkt_progress == 0.1


@pytest.mark.parametrize(
    "value",
    [
        -0.1,  # (1 − γ) > 1 ⇒ would accept a KKT-error *increase*
        0.0,  # no decrease required — not a progress certificate
        1.0,  # requires e_t ≤ 0, which a norm never satisfies ⇒ never fires
        1.5,
        float("nan"),
        float("inf"),
    ],
)
def test_line_search_rejects_out_of_range_feasible_kkt_progress(value):
    # The rescue accepts when e_t ≤ (1 − γ)·e0, so only γ ∈ (0, 1) is a
    # meaningful decrease fraction; outside it the gate silently degenerates
    # into "always accept" or "never accept" instead of erroring.
    with pytest.raises(ValueError, match="feasible_kkt_progress"):
        LineSearchOptions(feasible_kkt_progress=value)


@pytest.mark.parametrize("field", ["free_filter_margin_fact", "free_filter_max_margin"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_line_search_rejects_invalid_free_filter_margins(field, value):
    with pytest.raises(ValueError, match=field):
        LineSearchOptions(**{field: value})
