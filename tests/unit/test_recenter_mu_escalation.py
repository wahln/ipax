"""Unit tests for the feasible-point re-center's barrier (μ) escalation.

A globalization failure at a *feasible* iterate re-centers the barrier state
(``recenter_slacks_duals``) instead of running restoration — the 0.6.1 HS101
fix. But when that re-center repeats, the solve is on a treadmill: the barrier
problem itself is what cannot be solved (observed on an RT fluence case: μ
pinned at its ε/10 floor, every re-center re-derived at the same μ, 480
iterations of 11–27-trial line searches). The escalation contract:

- the **first** feasible re-center keeps the current μ (HS101 behavior,
  A/B-validated +10/−0 — must not change);
- a **repeated** feasible re-center raises μ — the "iterate has outrun the
  barrier" response (cf. IPOPT's adaptive oracle raising μ freely; Nocedal,
  Wächter & Waltz 2009, §5.1 re-initialization) — meaningfully (≥10×) and
  capped at ``mu_init`` (never worse-centered than a cold start).
"""

from __future__ import annotations

from ipax.backend.namespace import array_namespace
from ipax.ipm.driver import IPMDriver, _RestorationState
from ipax.ipm.filter_ls import Filter
from ipax.options import Options
from ipax.result import IterationRecord
from tests._helpers import array


def _driver(namespace, opts):
    driver = IPMDriver.__new__(IPMDriver)  # bypass __init__: the feasible
    driver._xp = array_namespace(array(namespace, [0.0]))  # branch only needs
    driver._options = opts  # the namespace and the options
    return driver


def _stalled_record(mu):
    # kkt_error far above tolerance and inf components (defaults) so the
    # handler cannot take the "stall at an essentially optimal iterate" exit.
    return IterationRecord(7, 1.0, mu, 0.0, 1e-4, 1.0, 1.0, 0.0)


def _handle(driver, namespace, *, mu, rstate, record=None):
    """Call the handler at a feasible point (θ0 = 0, one inequality)."""
    xp = driver._xp
    g = array(namespace, [-1.0])  # strictly satisfied inequality
    return driver._handle_restoration(
        x=array(namespace, [0.5, 0.5]),
        s=array(namespace, [1.0]),
        y_ineq=array(namespace, [1.0]),
        y_eq=array(namespace, []),
        g=g,
        mu=mu,
        m=1,
        m_eq=0,
        mask_l=xp.asarray([True, True]),
        mask_u=xp.asarray([False, False]),
        lower_safe=array(namespace, [0.0, 0.0]),
        upper_safe=array(namespace, [1e20, 1e20]),
        theta0=0.0,
        theta_inf=0.0,
        phi0=1.0,
        record=record if record is not None else _stalled_record(mu),
        filt=Filter(),
        theta_best=0.0,
        x_restore_anchor=array(namespace, [0.5, 0.5]),
        rstate=rstate,
        it=7,
    )


def test_first_feasible_recenter_keeps_mu(namespace):
    # The HS101 fix: the first re-center repairs slacks/duals at the *current*
    # μ. No barrier escalation — outcome.mu is None (keep the loop's μ).
    opts = Options()
    driver = _driver(namespace, opts)
    rstate = _RestorationState()

    outcome = _handle(driver, namespace, mu=1e-9, rstate=rstate)

    assert outcome.resume
    assert outcome.s is not None and outcome.y_ineq is not None
    assert outcome.mu is None


def test_repeat_feasible_recenter_raises_mu(namespace):
    # The treadmill signature: a second feasible re-center in the same solve
    # means re-centering at the current μ already failed once — raise μ so the
    # next barrier problem is one the step/line search can actually solve.
    opts = Options()
    driver = _driver(namespace, opts)
    rstate = _RestorationState()
    mu = 1e-9

    first = _handle(driver, namespace, mu=mu, rstate=rstate)
    second = _handle(driver, namespace, mu=mu, rstate=rstate)

    assert first.mu is None
    assert second.resume
    assert second.mu is not None
    assert second.mu > mu  # strictly raised
    assert second.mu >= 10.0 * mu  # meaningfully raised, not epsilon
    assert second.mu <= opts.barrier.mu_init  # capped at the cold-start μ


def test_repeat_recenter_mu_capped_at_mu_init(namespace):
    # At μ already equal to mu_init there is no headroom: the re-center must
    # still resume, and any μ it returns must not exceed the cold-start value.
    opts = Options()
    driver = _driver(namespace, opts)
    rstate = _RestorationState()
    mu = opts.barrier.mu_init

    _handle(driver, namespace, mu=mu, rstate=rstate)
    second = _handle(
        driver, namespace, mu=mu, rstate=rstate, record=_stalled_record(mu)
    )

    assert second.resume
    assert second.mu is None or second.mu <= opts.barrier.mu_init
