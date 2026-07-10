"""Unit tests for the alternative Breedveld step controller."""

from __future__ import annotations

from ipax import Options, Status, solve
from ipax.ipm.breedveld_ls import BreedveldController
from ipax.ipm.driver import IPMDriver
from ipax.options import BarrierOptions, BreedveldOptions
from ipax.testing.problems import BoundConstrainedQP
from tests._helpers import array, assert_allclose


def test_markov_filter_accepts_barrier_armijo_decrease():
    controller = BreedveldController(BreedveldOptions())
    # φ decreases along a descent direction (dphi < 0) ⇒ accept.
    assert controller.markov_accept(
        theta_t=1.0, phi_t=0.5, theta0=1.0, phi0=1.0, dphi=-1.0, alpha=0.5
    )


def test_markov_filter_accepts_infeasibility_decrease():
    controller = BreedveldController(BreedveldOptions())
    # φ rises but the constraint violation drops ⇒ still acceptable.
    assert controller.markov_accept(
        theta_t=0.1, phi_t=10.0, theta0=1.0, phi0=1.0, dphi=1.0, alpha=0.5
    )


def test_markov_filter_rejects_when_both_worsen():
    controller = BreedveldController(BreedveldOptions())
    assert not controller.markov_accept(
        theta_t=2.0, phi_t=5.0, theta0=1.0, phi0=1.0, dphi=1.0, alpha=0.5
    )


def test_search_reports_trial_count_on_ratio_control_accept():
    # Ratio control (iteration < _RATIO_CONTROL_ITERS) accepts on its single
    # probe when the barrier objective doesn't inflate — one trial.
    controller = BreedveldController(BreedveldOptions())
    _alpha, restoration, n_trials = controller.search(
        alpha_max=1.0,
        theta0=1.0,
        phi0=1.0,
        dphi=-1.0,
        eval_point=lambda alpha: (0.5, 0.5),
        iteration=0,
    )
    assert not restoration
    assert n_trials == 1


def test_search_reports_trial_count_on_backtrack_accept():
    # Past the ratio-control window, each backtrack step counts as one trial.
    controller = BreedveldController(BreedveldOptions(backtrack=0.5))
    calls = {"n": 0}

    def eval_point(alpha):
        calls["n"] += 1
        # First two trials worsen both theta and phi (rejected by the Markov
        # filter); the third improves both (accepted).
        return (2.0, 2.0) if calls["n"] < 3 else (0.1, 0.5)

    _alpha, restoration, n_trials = controller.search(
        alpha_max=1.0,
        theta0=1.0,
        phi0=1.0,
        dphi=1.0,
        eval_point=eval_point,
        iteration=100,  # past _RATIO_CONTROL_ITERS
    )
    assert not restoration
    assert n_trials == 3


def test_breedveld_globalization_solves_bound_qp(namespace):
    problem = BoundConstrainedQP(namespace)
    result = solve(
        problem,
        array(namespace, [0.25, 0.75]),
        options=Options(hessian="exact", linsolve="dense", globalization="breedveld"),
    )
    assert result.status is Status.OPTIMAL
    assert_allclose(namespace, result.x, problem.known_solution(), rtol=1e-6, atol=1e-6)


def test_breedveld_globalization_uses_breedveld_tau(namespace, monkeypatch):
    captured: list[float] = []
    original = IPMDriver._alpha_primal

    def capture_tau(self, step, s, m, x_minus_l, u_minus_x, mask_l, mask_u, tau):
        captured.append(tau)
        return original(self, step, s, m, x_minus_l, u_minus_x, mask_l, mask_u, tau)

    monkeypatch.setattr(IPMDriver, "_alpha_primal", capture_tau)
    result = solve(
        BoundConstrainedQP(namespace),
        array(namespace, [0.25, 0.75]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            globalization="breedveld",
            # The probing default's affine probe legitimately measures the
            # maximal steplength at τ = 1, which would contaminate the captured
            # values; monotone keeps every _alpha_primal call a line-search one.
            mu_schedule="monotone",
            barrier=BarrierOptions(tau_min=0.99),
            breedveld=BreedveldOptions(tau=0.5),
        ),
    )

    assert result.status is Status.OPTIMAL
    assert captured
    assert all(tau == 0.5 for tau in captured)
