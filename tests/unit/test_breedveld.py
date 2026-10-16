"""Unit tests for the alternative Breedveld step controller."""

from __future__ import annotations

import dataclasses
import hashlib
import struct

import pytest

from ipax import Options, Status, solve
from ipax.ipm.breedveld_ls import BreedveldController
from ipax.ipm.corrections import MehrotraCorrector
from ipax.ipm.driver import IPMDriver
from ipax.options import BarrierOptions, BreedveldOptions
from ipax.testing.problems import HS21, HS35, HS43, HS71, BoundConstrainedQP
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
    result = controller.search(
        alpha_max=1.0,
        theta0=1.0,
        phi0=1.0,
        dphi=-1.0,
        eval_point=lambda alpha: (0.5, 0.5),
        iteration=0,
    )
    assert not result.restoration
    assert result.n_trials == 1


def test_search_reports_trial_count_on_backtrack_accept():
    # Past the ratio-control window, each backtrack step counts as one trial.
    controller = BreedveldController(BreedveldOptions(backtrack=0.5))
    calls = {"n": 0}

    def eval_point(alpha):
        calls["n"] += 1
        # First two trials worsen both theta and phi (rejected by the Markov
        # filter); the third improves both (accepted).
        return (2.0, 2.0) if calls["n"] < 3 else (0.1, 0.5)

    result = controller.search(
        alpha_max=1.0,
        theta0=1.0,
        phi0=1.0,
        dphi=1.0,
        eval_point=eval_point,
        iteration=100,  # past _RATIO_CONTROL_ITERS
    )
    assert not result.restoration
    assert result.n_trials == 3


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


# --------------------------------------------------------------------------- #
# BreedveldOptions.common_step: one step length for the primal and the dual
# update (Breedveld 2017 §2.1 / Algorithm 1). Opt-in; the default keeps ipax's
# separate primal and dual fraction-to-boundary lengths.
# --------------------------------------------------------------------------- #
def _runner_like(**breedveld: object) -> Options:
    """The TROTS runner's Breedveld setup (exact Hessian, Mehrotra, probing μ)."""
    return Options(
        hessian="exact",
        linsolve="dense",
        globalization="breedveld",
        corrections="mehrotra",
        mu_schedule="probing",
        barrier=BarrierOptions(kappa_centrality=0.0),
        breedveld=BreedveldOptions(**breedveld),  # type: ignore[arg-type]
    )


def _hs71(namespace):
    return HS71(namespace), array(namespace, [1.0, 5.0, 5.0, 1.0])


def _alphas(result) -> list[tuple[float, float]]:
    return [(r.alpha_primal, r.alpha_dual) for r in result.history]


def _digest(x) -> str:
    """sha256 of the float64 bytes of ``x`` (backend-agnostic, element-wise)."""
    values = [float(x[i]) for i in range(int(x.shape[0]))]
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


def test_common_step_is_off_by_default():
    assert BreedveldOptions().common_step is False
    assert Options().breedveld.common_step is False


def test_common_step_couples_the_primal_and_dual_step_lengths(namespace):
    # HS71 under the runner's Breedveld setup takes different primal and dual
    # fraction-to-boundary lengths by default; with ``common_step`` every
    # accepted step moves both by the same α.
    problem, x0 = _hs71(namespace)
    separate = solve(problem, x0, options=_runner_like())
    common = solve(problem, x0, options=_runner_like(common_step=True))

    assert any(ap != ad for ap, ad in _alphas(separate))
    assert common.status is Status.OPTIMAL
    assert all(ap == ad for ap, ad in _alphas(common))
    assert_allclose(namespace, common.x, separate.x, rtol=1e-6, atol=1e-6)


def test_common_step_hands_the_controller_the_smaller_length(monkeypatch, namespace):
    # α_max of the ratio control / Markov backtracking is min(α_p, α_d) of the
    # fraction-to-boundary rule (Breedveld 2017 §2.1). Monotone μ without a
    # corrector keeps every _alpha_primal/_alpha_dual call a line-search one.
    lengths: list[tuple[str, float]] = []
    maxima: list[float] = []
    orig_p, orig_d = IPMDriver._alpha_primal, IPMDriver._alpha_dual
    orig_search = BreedveldController.search

    def spy_p(self, *args, **kwargs):
        value = orig_p(self, *args, **kwargs)
        lengths.append(("p", value))
        return value

    def spy_d(self, *args, **kwargs):
        value = orig_d(self, *args, **kwargs)
        lengths.append(("d", value))
        return value

    def spy_search(self, **kwargs):
        maxima.append(kwargs["alpha_max"])
        return orig_search(self, **kwargs)

    monkeypatch.setattr(IPMDriver, "_alpha_primal", spy_p)
    monkeypatch.setattr(IPMDriver, "_alpha_dual", spy_d)
    monkeypatch.setattr(BreedveldController, "search", spy_search)
    problem, x0 = _hs71(namespace)
    options = dataclasses.replace(
        _runner_like(common_step=True), corrections="none", mu_schedule="monotone"
    )
    result = solve(problem, x0, options=options)

    assert result.status is Status.OPTIMAL
    kinds = [kind for kind, _ in lengths]
    assert kinds == ["p", "d"] * len(maxima) and maxima
    pairs = [(lengths[i][1], lengths[i + 1][1]) for i in range(0, len(lengths), 2)]
    assert all(
        amax == min(ap, ad) for amax, (ap, ad) in zip(maxima, pairs, strict=True)
    )
    assert any(ap != ad for ap, ad in pairs)  # the coupling actually binds


def test_common_step_backtracking_shrinks_both_updates(monkeypatch, namespace):
    # A backtracked α (ratio control / Markov filter) moves the duals by the
    # same shrunken length as the primal step, not by the dual boundary length.
    orig_search = BreedveldController.search
    returned: list[float] = []

    def half_step(self, **kwargs):
        result = orig_search(self, **kwargs)
        returned.append(0.5 * result.alpha)
        return dataclasses.replace(result, alpha=0.5 * result.alpha)

    monkeypatch.setattr(BreedveldController, "search", half_step)
    problem, x0 = _hs71(namespace)
    options = dataclasses.replace(_runner_like(common_step=True), max_iter=8)
    result = solve(problem, x0, options=options)

    steps = _alphas(result)[1:]  # record 0 is the start
    assert steps
    assert all(ap == ad for ap, ad in steps)
    assert [ap for ap, _ in steps] == returned[: len(steps)]


def test_common_step_leaves_the_corrector_probes_separate(monkeypatch, namespace):
    # ``common_step`` couples the step the driver takes, not the predictor-
    # corrector's step-length probes (the Head-and-Neck_01 diagnosis' lever;
    # coupling the probes too re-targets μ and was measured worse there). At
    # the shared start the corrector sees exactly the default's probe lengths.
    seen: dict[bool, list[tuple[float, float]]] = {False: [], True: []}
    orig = MehrotraCorrector.correct
    arm = {"common": False}

    def spy(self, context, mu_target):
        seen[arm["common"]].append(
            (
                context.alpha_primal(context.affine, tau=1.0),
                context.alpha_dual(context.affine, tau=1.0),
            )
        )
        return orig(self, context, mu_target)

    monkeypatch.setattr(MehrotraCorrector, "correct", spy)
    problem, x0 = _hs71(namespace)
    solve(problem, x0, options=_runner_like())
    arm["common"] = True
    solve(problem, x0, options=_runner_like(common_step=True))

    assert seen[True] and seen[True][0] == seen[False][0]
    assert any(ap != ad for ap, ad in seen[True])


def test_common_step_is_inert_under_the_filter_line_search(namespace):
    # The option belongs to the Breedveld controller: the IPOPT filter line
    # search keeps its separate primal and dual lengths (W&B 2006, eq. (15)).
    problem, x0 = _hs71(namespace)
    base = solve(problem, x0, options=Options())
    flagged = solve(
        problem, x0, options=Options(breedveld=BreedveldOptions(common_step=True))
    )
    assert flagged.n_iter == base.n_iter
    assert _alphas(flagged) == _alphas(base)
    assert _digest(flagged.x) == _digest(base.x)
    assert any(ap != ad for ap, ad in _alphas(flagged))


# Regression pin of the default path: status and iteration counts recorded at
# db75a27, before ``common_step`` existed (NumPy; the same on Torch and
# array-api-strict). Portable across BLAS builds, unlike a digest of x; the
# pre/post-change sha256 of x itself was compared out of tree (HS problems and
# the synthetic TROTS instance, 24 solves, identical) — see the commit message.
# The implicit-vs-explicit ``common_step=False`` digest comparison guards the
# option's default value, not the code path. A future algorithmic change that
# moves these counts should re-pin them, not blame this option.
_FINGERPRINTS = {
    ("HS71", "breedveld"): ("OPTIMAL", 10),
    ("HS43", "breedveld"): ("OPTIMAL", 10),
    ("HS35", "breedveld"): ("OPTIMAL", 6),
    ("HS21", "breedveld"): ("OPTIMAL", 6),
    ("HS71", "filter"): ("OPTIMAL", 9),
    ("HS43", "filter"): ("OPTIMAL", 11),
}
_STARTS = {
    "HS71": (HS71, [1.0, 5.0, 5.0, 1.0]),
    "HS43": (HS43, [0.0, 0.0, 0.0, 0.0]),
    "HS35": (HS35, [0.5, 0.5, 0.5]),
    "HS21": (HS21, [-1.0, -1.0]),
}


@pytest.mark.parametrize(("name", "mode"), sorted(_FINGERPRINTS))
def test_common_step_default_is_bit_identical(namespace, name, mode):
    cls, start = _STARTS[name]
    problem, x0 = cls(namespace), array(namespace, start)
    base = _runner_like() if mode == "breedveld" else Options()
    explicit = dataclasses.replace(base, breedveld=BreedveldOptions(common_step=False))

    implicit_result = solve(problem, x0, options=base)
    explicit_result = solve(problem, x0, options=explicit)

    status, n_iter = _FINGERPRINTS[(name, mode)]
    assert implicit_result.status.name == status
    assert implicit_result.n_iter == n_iter
    assert explicit_result.n_iter == n_iter
    assert _alphas(explicit_result) == _alphas(implicit_result)
    assert _digest(explicit_result.x) == _digest(implicit_result.x)


# --------------------------------------------------------------------------- #
# BreedveldOptions.max_soc: opt-in second-order correction for the Markov
# acceptance (Wächter & Biegler 2006, §2.4 / Algorithm A-5.5–A-5.9, adapted to
# Breedveld 2017 eqs. (34)–(35)). Off by default — Breedveld's own solver only
# backtracks.
# --------------------------------------------------------------------------- #
_PAST_RATIO = 100  # an iteration index past the ratio-control window


class _SocSpy:
    """A ``soc(α)`` stand-in recording its calls and returning a fixed point."""

    def __init__(self, point: tuple[float, float] | None) -> None:
        self.point = point
        self.calls: list[float] = []

    def __call__(self, alpha: float) -> tuple[float, float] | None:
        self.calls.append(alpha)
        return self.point


def _sequence(*points: tuple[float, float]):
    """``eval_point`` returning ``points`` in order (the last one repeats)."""
    state = {"i": 0}

    def eval_point(alpha: float) -> tuple[float, float]:
        i = min(state["i"], len(points) - 1)
        state["i"] += 1
        return points[i]

    return eval_point


def _soc_search(controller, eval_point, soc, *, iteration=_PAST_RATIO, **kwargs):
    arguments = {
        "alpha_max": 1.0,
        "theta0": 1.0,
        "phi0": 1.0,
        "dphi": -1.0,
        "eval_point": eval_point,
        "iteration": iteration,
        "soc": soc,
    }
    arguments.update(kwargs)
    return controller.search(**arguments)


def test_max_soc_is_off_by_default():
    assert BreedveldOptions().max_soc == 0
    assert Options().breedveld.max_soc == 0


@pytest.mark.parametrize("bad", [-1, -5])
def test_max_soc_rejects_negative_counts(bad):
    with pytest.raises(ValueError, match="max_soc"):
        BreedveldOptions(max_soc=bad)


@pytest.mark.parametrize("bad", [1.5, True])
def test_max_soc_rejects_non_integer_counts(bad):
    with pytest.raises(ValueError, match="max_soc"):
        BreedveldOptions(max_soc=bad)  # type: ignore[arg-type]


def test_search_result_defaults_to_no_soc():
    controller = BreedveldController(BreedveldOptions())
    result = _soc_search(controller, _sequence((0.5, 0.5)), None)
    assert result.alpha == 1.0
    assert not result.restoration
    assert result.n_trials == 1
    assert result.used_soc is False


def test_soc_not_invoked_when_max_soc_is_zero():
    # Rejected first trial that worsens θ: the SOC trigger, but the option is off.
    controller = BreedveldController(BreedveldOptions(backtrack=0.5))
    soc = _SocSpy((0.1, 0.1))
    result = _soc_search(controller, _sequence((2.0, 2.0), (0.1, 0.5)), soc)
    assert soc.calls == []
    assert result.used_soc is False
    assert result.alpha == 0.5
    assert result.n_trials == 2


def test_soc_not_invoked_when_first_trial_is_accepted():
    controller = BreedveldController(BreedveldOptions(max_soc=1))
    soc = _SocSpy((0.1, 0.1))
    # θ rises but φ passes Armijo ⇒ eq. (34) accepts, no correction needed.
    result = _soc_search(controller, _sequence((2.0, 0.5)), soc)
    assert soc.calls == []
    assert result.used_soc is False
    assert result.n_trials == 1


def test_soc_not_invoked_when_rejected_trial_reduced_infeasibility():
    # W&B A-5.5: θ_t < θ0 skips the correction even though the Markov test
    # rejects (θ fell, but by less than the eq. (35) fraction).
    controller = BreedveldController(BreedveldOptions(max_soc=1, backtrack=0.5))
    soc = _SocSpy((0.1, 0.1))
    result = _soc_search(controller, _sequence((0.99995, 2.0), (0.1, 0.5)), soc)
    assert soc.calls == []
    assert result.used_soc is False
    assert result.n_trials == 2


@pytest.mark.parametrize("theta_t", [1.0, 3.0])  # θ_t = θ0 and θ_t > θ0
def test_soc_invoked_on_rejected_first_trial_that_did_not_reduce_theta(theta_t):
    controller = BreedveldController(BreedveldOptions(max_soc=2, backtrack=0.5))
    soc = _SocSpy((0.1, 0.1))
    result = _soc_search(controller, _sequence((theta_t, 2.0)), soc)
    assert soc.calls == [1.0]
    assert result.used_soc is True


def test_soc_only_on_the_first_trial():
    controller = BreedveldController(BreedveldOptions(max_soc=1, backtrack=0.5))
    soc = _SocSpy(None)  # the correction fails (e.g. leaves the interior)
    # Three rejected trials, all worsening θ; the fourth is accepted.
    result = _soc_search(
        controller, _sequence((2.0, 2.0), (2.0, 2.0), (2.0, 2.0), (0.1, 0.5)), soc
    )
    assert soc.calls == [1.0]
    assert result.used_soc is False
    assert result.alpha == 0.125
    # A correction that returns no point evaluated nothing: not a trial.
    assert result.n_trials == 4


def test_soc_skipped_when_the_trial_is_feasible_to_the_threshold():
    # Round-off θ (linear constraints: θ_t ≥ θ0 ≈ 1e-15 at random) leaves the
    # correction nothing to correct; the driver passes its feasibility notion.
    controller = BreedveldController(BreedveldOptions(max_soc=1, backtrack=0.5))
    soc = _SocSpy((0.0, 0.1))
    result = _soc_search(
        controller,
        _sequence((2e-15, 2.0), (1e-15, 0.5)),
        soc,
        theta0=1e-15,
        soc_theta_min=1e-8,
    )
    assert soc.calls == []
    assert result.used_soc is False
    assert result.n_trials == 2
    # Above the threshold the same rejected trial is corrected.
    above = _SocSpy((0.0, 0.1))
    corrected = _soc_search(
        controller,
        _sequence((2e-6, 2.0)),
        above,
        theta0=1e-6,
        soc_theta_min=1e-8,
    )
    assert above.calls == [1.0]
    assert corrected.used_soc is True


def test_max_soc_accepts_integer_like_counts():
    import numpy as np

    assert BreedveldOptions(max_soc=np.int64(2)).max_soc == 2  # type: ignore[arg-type]


def test_soc_not_used_during_ratio_control_iterations():
    # Ratio control (eq. (36)) governs the first iterations; SOC is a Markov-
    # acceptance addition only.
    controller = BreedveldController(BreedveldOptions(max_soc=1, backtrack=0.5))
    soc = _SocSpy((0.1, 0.1))
    result = _soc_search(
        controller, _sequence((2.0, 1e9), (2.0, 2.0), (0.1, 0.5)), soc, iteration=0
    )
    assert soc.calls == []
    assert result.used_soc is False


def test_accepted_soc_point_returns_alpha_max_with_flag():
    controller = BreedveldController(BreedveldOptions(max_soc=1))
    soc = _SocSpy((0.1, 0.5))
    result = _soc_search(controller, _sequence((2.0, 2.0)), soc, alpha_max=0.7)
    assert soc.calls == [0.7]
    assert result.alpha == 0.7
    assert result.used_soc is True
    assert not result.restoration
    assert result.n_trials == 2  # the full step + the corrected trial


def test_soc_point_is_judged_by_the_markov_test_at_the_full_step():
    # The corrected point's Armijo test uses the original α (W&B A-5.7):
    # φ_c = 1 − 0.75·c·|dφ| passes Armijo at α = 0.5 but not at α = 1, and
    # θ_c = 2 > θ0 rules out the eq. (35) infeasibility branch.
    controller = BreedveldController(BreedveldOptions(max_soc=1, backtrack=0.5))
    borderline = (2.0, 1.0 - 0.75e-4)
    at_one = _soc_search(
        controller, _sequence((2.0, 2.0), (0.1, 0.1)), _SocSpy(borderline)
    )
    assert at_one.used_soc is False
    at_half = _soc_search(
        controller,
        _sequence((2.0, 2.0), (0.1, 0.1)),
        _SocSpy(borderline),
        alpha_max=0.5,
    )
    assert at_half.used_soc is True


def test_rejected_soc_point_falls_back_to_backtracking():
    controller = BreedveldController(BreedveldOptions(max_soc=1, backtrack=0.5))
    soc = _SocSpy((3.0, 3.0))  # the corrected point is worse still
    result = _soc_search(controller, _sequence((2.0, 2.0), (0.1, 0.5)), soc)
    assert soc.calls == [1.0]
    assert result.used_soc is False
    assert result.alpha == 0.5
    # full step + SOC evaluation + the accepted backtrack
    assert result.n_trials == 3


def test_failed_soc_keeps_the_restoration_hand_off():
    # A correction does not consume the backtracking budget: the α handed to
    # restoration is the one the plain search reaches.
    options = BreedveldOptions(max_soc=1, backtrack=0.5, max_backtrack=4)
    plain = BreedveldController(dataclasses.replace(options, max_soc=0))
    corrected = BreedveldController(options)
    base = _soc_search(plain, _sequence((2.0, 2.0)), None)
    result = _soc_search(corrected, _sequence((2.0, 2.0)), _SocSpy((3.0, 3.0)))
    assert base.restoration and result.restoration
    assert result.alpha == base.alpha
    assert result.n_trials == base.n_trials + 1
