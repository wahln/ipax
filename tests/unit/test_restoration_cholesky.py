"""Unit tests for the opt-in Cholesky factorization of the dense restoration solve.

``RestorationOptions.dense_factorization="cholesky"`` factors the damped
Gauss-Newton matrix ``Jᵀ Σ J + λ I`` (SPD for ``λ > 0``; Nocedal & Wright 2006,
§10.3) with ``xp.linalg.cholesky`` and back-substitutes it through the
``get_dense_cholesky_solve`` gap-filler (the Array-API ``linalg`` extension has
no triangular solve). Any Cholesky breakdown falls back to the default LU solve
for that damping trial; the default stays ``"lu"`` with unchanged arithmetic.
"""

from __future__ import annotations

import pytest

import ipax.ipm.restoration as restoration_mod
from ipax import Options, Status, solve
from ipax.backend.operators import as_operator
from ipax.ipm.restoration import RestorationExit, restore
from ipax.options import RestorationOptions
from ipax.testing.problems import HS6
from tests._helpers import array, assert_allclose


def _no_ineq(x):
    raise AssertionError("inequality callbacks should not be used here")


def _pure_standard_cholesky_solve(xp):
    """A pure-Array-API stand-in for the gap-filler (``solve`` on the factors).

    Pays an LU per triangle, so it is only a correctness device: it lets the
    Cholesky code path run on backends without a native triangular solve
    (array-api-strict), where the shipped lookup returns ``None``.
    """

    def cholesky_solve(factor, rhs):
        y = xp.linalg.solve(factor, rhs)
        return xp.linalg.solve(xp.permute_dims(factor, (1, 0)), y)

    return cholesky_solve


@pytest.fixture
def cholesky_backend(namespace, monkeypatch):
    """Guarantee a triangular-solve provider for ``namespace``.

    The shipped adapter where one exists (NumPy, Torch, CuPy); the pure-standard
    stand-in otherwise, so every backend exercises the Cholesky branch.
    """
    if restoration_mod._dense_cholesky_solver(namespace) is None:
        stand_in = _pure_standard_cholesky_solve(namespace)
        monkeypatch.setattr(
            restoration_mod, "_dense_cholesky_solver", lambda xp: stand_in
        )
    return namespace


def _hs6_kwargs(xp):
    """HS6 from a point off its constraint manifold (one nonlinear equality)."""
    problem = HS6(xp)
    x = array(xp, [2.0, 0.0])
    return {
        "xp": xp,
        "x": x,
        "s": xp.zeros((0,), dtype=x.dtype),
        "m": 0,
        "m_eq": 1,
        "eq_fn": problem.eq_constraints,
        "eq_jac_fn": lambda z: as_operator(problem.eq_jacobian(z)),
        "ineq_fn": _no_ineq,
        "ineq_jac_fn": _no_ineq,
        "mask_l": xp.zeros((2,), dtype=xp.bool),
        "mask_u": xp.zeros((2,), dtype=xp.bool),
        "lower_safe": xp.zeros((2,), dtype=x.dtype),
        "upper_safe": xp.zeros((2,), dtype=x.dtype),
        "tol": 1e-8,
    }


def _mixed_kwargs(xp):
    """Two nonlinear equalities, one violated inequality and an active bound.

    ``x0² + x1² = 4``, ``x2 x3 = 1``, ``x0 + x2 ≤ 0.5``, ``x1 ≥ 0.5`` from
    ``(3, -1, 2, 2)``: several LM iterations, a bound-blocked (reduced) system
    and an inequality Gram block — the dense route's full feature set.
    """
    x = array(xp, [3.0, -1.0, 2.0, 2.0])

    def eq_fn(z):
        return xp.stack((z[0] * z[0] + z[1] * z[1] - 4.0, z[2] * z[3] - 1.0))

    def eq_jac_fn(z):
        zero = xp.zeros((), dtype=z.dtype)
        rows = (
            xp.stack((2.0 * z[0], 2.0 * z[1], zero, zero)),
            xp.stack((zero, zero, z[3], z[2])),
        )
        return as_operator(xp.stack(rows))

    def ineq_fn(z):
        return xp.stack((z[0] + z[2] - 0.5,))

    def ineq_jac_fn(z):
        del z
        return as_operator(array(xp, [[1.0, 0.0, 1.0, 0.0]]))

    return {
        "xp": xp,
        "x": x,
        "s": xp.ones((1,), dtype=x.dtype),
        "m": 1,
        "m_eq": 2,
        "eq_fn": eq_fn,
        "eq_jac_fn": eq_jac_fn,
        "ineq_fn": ineq_fn,
        "ineq_jac_fn": ineq_jac_fn,
        "mask_l": xp.asarray([False, True, False, False], dtype=xp.bool),
        "mask_u": xp.zeros((4,), dtype=xp.bool),
        "lower_safe": array(xp, [0.0, 0.5, 0.0, 0.0]),
        "upper_safe": xp.zeros((4,), dtype=x.dtype),
        "tol": 1e-8,
    }


_CASES = {"hs6": _hs6_kwargs, "mixed": _mixed_kwargs}


def _spy(monkeypatch, xp, name):
    """Count calls to ``xp.linalg.<name>`` while delegating to the original."""
    calls = {"n": 0}
    original = getattr(xp.linalg, name)

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(xp.linalg, name, counting)
    return calls


# ---------------------------------------------------------------------------
# Option surface
# ---------------------------------------------------------------------------


def test_dense_factorization_defaults_to_lu():
    assert RestorationOptions().dense_factorization == "lu"
    assert RestorationOptions(dense_factorization="cholesky").dense_factorization == (
        "cholesky"
    )


@pytest.mark.parametrize("bad", ["LU", "ldl", "", None, 1])
def test_dense_factorization_is_validated(bad):
    with pytest.raises(ValueError, match="dense_factorization"):
        RestorationOptions(dense_factorization=bad)


def test_restore_rejects_an_unknown_dense_factorization(namespace):
    with pytest.raises(ValueError, match="dense_factorization"):
        restore(**_hs6_kwargs(namespace), dense_factorization="qr")


# ---------------------------------------------------------------------------
# Default path: unchanged arithmetic
# ---------------------------------------------------------------------------

# Reference restoration iterate produced by the LU path *before* the option
# existed (commit 2343566); identical on NumPy, Torch and array-api-strict.
# HS6 is deliberately not pinned: its rank-1 Gram makes the damped matrix
# ~1e10-conditioned, so its feasible end point moves by ~1e-6 with library
# round-off (NumPy vs Torch LU already differ) — it would pin the BLAS build,
# not the code. ``test_explicit_lu_is_the_default_bit_for_bit`` covers it.
_LU_REFERENCE = {
    "mixed": (
        [1.9364916717334204, 0.50000001, -1.4364916717334224, -0.6961404786936854],
        "feasible",
    ),
}


@pytest.mark.parametrize("case", sorted(_LU_REFERENCE))
def test_default_restoration_reproduces_the_lu_reference(namespace, monkeypatch, case):
    xp = namespace
    cholesky_calls = _spy(monkeypatch, xp, "cholesky")
    x_new, _, exit_reason = restore(**_CASES[case](xp))
    x_ref, exit_ref = _LU_REFERENCE[case]
    assert exit_reason.value == exit_ref
    assert_allclose(xp, x_new, array(xp, x_ref), rtol=1e-12, atol=1e-12)
    assert cholesky_calls["n"] == 0


@pytest.mark.parametrize("case", sorted(_CASES))
def test_explicit_lu_is_the_default_bit_for_bit(namespace, case):
    xp = namespace
    x_default, s_default, exit_default = restore(**_CASES[case](xp))
    x_lu, s_lu, exit_lu = restore(**_CASES[case](xp), dense_factorization="lu")
    assert exit_lu is exit_default
    assert bool(xp.all(x_lu == x_default))
    assert bool(xp.all(s_lu == s_default))


# ---------------------------------------------------------------------------
# Cholesky path
# ---------------------------------------------------------------------------


def test_cholesky_reaches_the_lu_feasible_point(cholesky_backend, monkeypatch):
    xp = cholesky_backend
    x_lu, s_lu, exit_lu = restore(**_mixed_kwargs(xp))
    cholesky_calls = _spy(monkeypatch, xp, "cholesky")
    x_ch, s_ch, exit_ch = restore(**_mixed_kwargs(xp), dense_factorization="cholesky")

    assert exit_lu is exit_ch is RestorationExit.FEASIBLE
    assert cholesky_calls["n"] > 0
    assert_allclose(xp, x_ch, x_lu, rtol=1e-8, atol=1e-8)
    assert_allclose(xp, s_ch, s_lu, rtol=1e-8, atol=1e-8)


def test_cholesky_on_a_rank_deficient_gram_stays_on_the_manifold(
    cholesky_backend, monkeypatch
):
    """HS6: one equality in two variables, so ``JᵀJ`` has rank 1.

    With ``λ = 1e-8`` the damped matrix has condition ~``‖J‖²/λ ≈ 1e10``, and
    the null-space component of the step is round-off in ``∇F`` amplified by
    ``1/λ``: LU and Cholesky (like NumPy and Torch LU, see ``_LU_REFERENCE``)
    drift apart along the feasible manifold by ~1e-6 over a few iterations.
    Both must still land on it, feasibly and close together.
    """
    xp = cholesky_backend
    problem = HS6(xp)
    x_lu, _, exit_lu = restore(**_hs6_kwargs(xp))
    cholesky_calls = _spy(monkeypatch, xp, "cholesky")
    x_ch, _, exit_ch = restore(**_hs6_kwargs(xp), dense_factorization="cholesky")

    assert exit_lu is exit_ch is RestorationExit.FEASIBLE
    assert cholesky_calls["n"] > 0
    assert float(xp.max(xp.abs(problem.eq_constraints(x_ch)))) <= 1e-8
    assert_allclose(xp, x_ch, x_lu, rtol=1e-5, atol=1e-5)


def test_cholesky_never_pays_an_lu_when_the_factor_succeeds(namespace, monkeypatch):
    """With a native triangular solve, a successful trial costs no LU at all."""
    xp = namespace
    if restoration_mod._dense_cholesky_solver(xp) is None:
        pytest.skip("no native triangular-solve adapter for this backend")
    lu_calls = _spy(monkeypatch, xp, "solve")
    cholesky_calls = _spy(monkeypatch, xp, "cholesky")
    _, _, exit_reason = restore(**_mixed_kwargs(xp), dense_factorization="cholesky")
    assert exit_reason is RestorationExit.FEASIBLE
    assert cholesky_calls["n"] > 0
    assert lu_calls["n"] == 0


def test_cholesky_without_a_triangular_solve_uses_lu(namespace, monkeypatch):
    """A backend without the gap-filler (JAX, array-api-strict) keeps the LU path."""
    xp = namespace
    monkeypatch.setattr(restoration_mod, "_dense_cholesky_solver", lambda xp: None)
    cholesky_calls = _spy(monkeypatch, xp, "cholesky")
    x_lu, _, exit_lu = restore(**_mixed_kwargs(xp))
    x_ch, _, exit_ch = restore(**_mixed_kwargs(xp), dense_factorization="cholesky")
    assert exit_ch is exit_lu
    assert bool(xp.all(x_ch == x_lu))
    assert cholesky_calls["n"] == 0


# ---------------------------------------------------------------------------
# Breakdown fallback: never a crash, never a silent NaN step
# ---------------------------------------------------------------------------


def _assert_fell_back_to_lu(xp, monkeypatch, case="mixed"):
    """Every trial fell back: the run must equal the LU run bit for bit."""
    x_lu, s_lu, exit_lu = restore(**_CASES[case](xp))
    lu_calls = _spy(monkeypatch, xp, "solve")
    x_ch, s_ch, exit_ch = restore(**_CASES[case](xp), dense_factorization="cholesky")
    assert exit_ch is exit_lu is RestorationExit.FEASIBLE
    assert bool(xp.all(xp.isfinite(x_ch)))
    assert bool(xp.all(x_ch == x_lu))
    assert bool(xp.all(s_ch == s_lu))
    assert lu_calls["n"] > 0


def test_cholesky_breakdown_falls_back_to_lu(cholesky_backend, monkeypatch):
    """A backend-raised breakdown (a genuinely indefinite input) → LU for that trial."""
    xp = cholesky_backend
    original = xp.linalg.cholesky
    raised = {"n": 0}

    def indefinite_cholesky(matrix, *args, **kwargs):
        # Factor −A instead of A: negative definite, so the backend itself
        # raises its own LinAlgError (numpy/strict/torch) for every trial.
        try:
            return original(-matrix, *args, **kwargs)
        except Exception:
            raised["n"] += 1
            raise

    monkeypatch.setattr(xp.linalg, "cholesky", indefinite_cholesky)
    # Every backend must fall back — a silent breakdown is classified too —
    # but only a raising one exercises the path this test is about.
    _assert_fell_back_to_lu(xp, monkeypatch)
    if raised["n"] == 0:
        pytest.skip(
            f"{xp.__name__} reports a non-PD Cholesky silently (a NaN pivot) "
            "instead of raising; that path is "
            "test_non_finite_cholesky_factor_falls_back_to_lu"
        )


def test_non_finite_cholesky_factor_falls_back_to_lu(cholesky_backend, monkeypatch):
    """A silent breakdown (NaN factor: NumPy on NaN input, CuPy, JAX) → LU."""
    xp = cholesky_backend

    def nan_cholesky(matrix, *args, **kwargs):
        del args, kwargs
        return xp.full(matrix.shape, float("nan"), dtype=matrix.dtype)

    monkeypatch.setattr(xp.linalg, "cholesky", nan_cholesky)
    _assert_fell_back_to_lu(xp, monkeypatch)


def test_zero_pivot_cholesky_factor_is_a_breakdown(namespace, monkeypatch):
    """A *finite* factor with a zero pivot is a breakdown too (CuPy, singular input).

    CuPy's default ``errstate`` returns ``L`` with ``l_kk = 0`` for a singular
    PSD matrix, and a reference-BLAS ``trsm`` skips the zero right-hand-side
    entries of bound-blocked rows, so the back-substitution can come out
    finite. Classified by the pivots (``cholesky_factor_ok``), the trial goes
    to LU like every other breakdown instead of taking that direction.
    """
    xp = namespace
    damped = array(xp, [[1.0, 0.0], [0.0, 0.0]])
    rhs = array(xp, [1.0, 0.0])
    monkeypatch.setattr(
        xp.linalg,
        "cholesky",
        lambda matrix, *a, **k: array(xp, [[1.0, 0.0], [0.0, 0.0]]),
    )

    def skipping_solve(factor, rhs):  # trsm that skips the zero entry
        del factor
        return rhs

    assert (
        restoration_mod._cholesky_lm_direction(xp, damped, rhs, skipping_solve) is None
    )


def test_non_finite_cholesky_solution_falls_back_to_lu(cholesky_backend, monkeypatch):
    """A finite factor whose back-substitution overflows → LU, not a NaN step."""
    xp = cholesky_backend

    def overflowing_solve(factor, rhs):
        del factor
        return xp.full(rhs.shape, float("inf"), dtype=rhs.dtype)

    monkeypatch.setattr(
        restoration_mod, "_dense_cholesky_solver", lambda xp: overflowing_solve
    )
    _assert_fell_back_to_lu(xp, monkeypatch)


def test_raising_cholesky_solve_falls_back_to_lu(cholesky_backend, monkeypatch):
    xp = cholesky_backend

    def failing_solve(factor, rhs):
        del factor, rhs
        raise ArithmeticError("synthetic back-substitution overflow")

    monkeypatch.setattr(
        restoration_mod, "_dense_cholesky_solver", lambda xp: failing_solve
    )
    _assert_fell_back_to_lu(xp, monkeypatch)


@pytest.mark.parametrize(
    "bug",
    [TypeError("unexpected keyword argument 'upper'"), MemoryError("host OOM")],
    ids=lambda e: type(e).__name__,
)
def test_non_numerical_cholesky_errors_propagate(cholesky_backend, monkeypatch, bug):
    """Only a numerical failure is a breakdown; a bug or OOM leaves ``restore``."""
    xp = cholesky_backend

    def raising(*args, **kwargs):
        del args, kwargs
        raise bug

    monkeypatch.setattr(xp.linalg, "cholesky", raising)
    with pytest.raises(type(bug)) as info:
        restore(**_mixed_kwargs(xp), dense_factorization="cholesky")
    assert info.value is bug


def test_singular_gauss_newton_system_survives_with_cholesky(cholesky_backend):
    """The HS7-class extreme-scale Jacobian (see ``test_restoration``) must not crash."""
    xp = cholesky_backend
    x = array(xp, [1.0, 1.0])
    jac = array(xp, [[4.0e100, -1.9e27]])
    kwargs = _hs6_kwargs(xp)
    kwargs.update(
        x=x,
        eq_fn=lambda z: array(xp, [1.0]),
        eq_jac_fn=lambda z: as_operator(jac),
    )
    x_new, _, exit_reason = restore(**kwargs, dense_factorization="cholesky")
    assert exit_reason.certifies_infeasibility
    assert bool(xp.all(xp.isfinite(x_new)))


# ---------------------------------------------------------------------------
# Driver plumbing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factorization", ["lu", "cholesky"])
def test_solve_passes_the_dense_factorization_to_restoration(
    namespace, monkeypatch, factorization
):
    import ipax.ipm.driver as driver_mod
    from tests.regression.test_restoration_tall_gram_block_not_densified import (
        _fail_step_solve_once,
        _TallLinearIneq,
    )

    xp = namespace
    _fail_step_solve_once(monkeypatch)
    seen: list[object] = []
    original = driver_mod.restore

    def recording(**kwargs):
        seen.append(kwargs.get("dense_factorization"))
        return original(**kwargs)

    monkeypatch.setattr(driver_mod, "restore", recording)
    result = solve(
        _TallLinearIneq(xp, []),
        array(xp, [4.0, 4.0, 4.0]),
        options=Options(
            hessian="exact",
            linsolve="dense",
            max_iter=100,
            restoration=RestorationOptions(
                linear_solver="dense", dense_factorization=factorization
            ),
        ),
    )
    assert seen, "the forced step failure never reached restoration"
    assert all(value == factorization for value in seen)
    assert result.status is Status.OPTIMAL
