"""Run-to-run reproducibility of the CuPy sparse adapter's own kernels (GPU-gated).

cuSPARSE's default algorithms accumulate with atomics in three places the
interior-point method reaches every iteration: the transposed product
``Aᵀy`` (``SparseOperator.rmatvec``/``rmatmat``/``gram_diagonal``), the
forward product of a matrix with long rows (the default CSR SpMV/SpMM split
such rows across thread blocks), and CuPy's ``cupy.add.at`` duplicate sum in
the COO→CSR build. Each returned a different last-bit result on almost every
call (30 distinct results in 30 repeats). These tests pin that every product
and every COO build of the adapter is now bitwise reproducible, and that the
results still agree with SciPy to round-off.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests._cudss import require_deterministic_cudss

cupy = pytest.importorskip("cupy")
sp = pytest.importorskip("scipy.sparse")

pytestmark = pytest.mark.gpu


def _cuda_available() -> bool:
    try:
        return bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


if not _cuda_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

import ipax.backend.sparse.cupy as cupy_adapter  # noqa: E402
from ipax.testing.backends import import_namespace  # noqa: E402

REPEATS = 30


def _distinct(fn) -> int:
    """Number of bitwise-distinct results over ``REPEATS`` calls."""
    seen = set()
    for _ in range(REPEATS):
        seen.add(cupy.asnumpy(cupy.asarray(fn())).tobytes())
    return len(seen)


def _wide_ranged(rng: np.random.Generator, size: int) -> np.ndarray:
    # Magnitudes over 12 decades: any reordering of a sum shows in the bits.
    return rng.standard_normal(size) * 10.0 ** rng.uniform(-6.0, 6.0, size)


def _operator(matrix: object):
    """The adapter operator of a host SciPy matrix, built from its COO."""
    coo = sp.coo_matrix(matrix)
    return cupy_adapter.CuPySparseAdapter().from_coo(
        cupy.asarray(coo.row),
        cupy.asarray(coo.col),
        cupy.asarray(coo.data),
        shape=coo.shape,
    )


def _tall_with_long_rows(seed: int = 0):
    """A 3000×20000 matrix: ~7 entries per column (a transposed product sums
    many terms into each output entry), plus 4 rows of 15000 entries (the
    default forward algorithms split such long rows across thread blocks)."""
    rng = np.random.default_rng(seed)
    a = sp.random(3000, 20000, density=0.0014, random_state=rng, format="lil")
    for row in (5, 700, 1500, 2999):
        a[row, rng.choice(20000, 15000, replace=False)] = 1.0
    a = a.tocsr()
    a.data = _wide_ranged(rng, a.nnz)
    return a, rng


def test_products_are_bitwise_reproducible_and_match_scipy():
    a, rng = _tall_with_long_rows()
    op = _operator(a)
    x = rng.standard_normal(a.shape[1])
    y = rng.standard_normal(a.shape[0])
    X = rng.standard_normal((a.shape[1], 5))
    Y = rng.standard_normal((a.shape[0], 5))
    w = rng.uniform(0.1, 10.0, a.shape[0])
    # Both multi-column kernels: ``matmat`` meets the 15000-entry rows (the
    # warp kernel), ``rmatmat`` the short rows of Aᵀ (the thread kernel).
    cases = {
        "matvec": (lambda: op.matvec(cupy.asarray(x)), a @ x),
        "rmatvec": (lambda: op.rmatvec(cupy.asarray(y)), a.T @ y),
        "matmat": (lambda: op.matmat(cupy.asarray(X)), a @ X),
        "rmatmat": (lambda: op.rmatmat(cupy.asarray(Y)), a.T @ Y),
        "gram_diagonal": (
            lambda: op.gram_diagonal(cupy.asarray(w)),
            np.asarray(a.multiply(a).T @ w).reshape(-1),
        ),
        "row_gram_diagonal": (
            lambda: op.row_gram_diagonal(cupy.asarray(x * x)),
            np.asarray(a.multiply(a) @ (x * x)).reshape(-1),
        ),
    }
    for name, (fn, reference) in cases.items():
        assert _distinct(fn) == 1, name
        actual = cupy.asnumpy(cupy.asarray(fn()))
        scale = float(np.max(np.abs(reference)))
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-13 * scale)


def _duplicated_triplets(seed: int = 1):
    """COO triplets with ~15 entries per slot, in a shuffled emission order."""
    rng = np.random.default_rng(seed)
    n_triplets, shape = 300_000, (500, 400)
    slots = rng.integers(0, shape[0] * shape[1] // 10, n_triplets)
    rows, cols = slots // shape[1], slots % shape[1]
    return rows, cols, _wide_ranged(rng, n_triplets), shape


@pytest.mark.parametrize("signature", [None, ("dups", 1)])
def test_coo_build_is_bitwise_reproducible(signature):
    rows, cols, values, shape = _duplicated_triplets()
    rows_d, cols_d, values_d = map(cupy.asarray, (rows, cols, values))
    adapter = cupy_adapter.CuPySparseAdapter()

    def build():
        op = adapter.from_coo(
            rows_d, cols_d, values_d, shape=shape, pattern_signature=signature
        )
        return op.cupy_matrix.data

    assert _distinct(build) == 1
    built = adapter.from_coo(
        rows_d, cols_d, values_d, shape=shape, pattern_signature=signature
    ).cupy_matrix
    # The sum runs in emission order, as NumPy's sequential ``np.add.at`` in
    # the SciPy adapter's compiled map does: the CPU and GPU builds agree to
    # the bit, not merely to round-off.
    from ipax.backend.sparse.numpy_scipy import SciPySparseAdapter

    host = SciPySparseAdapter().from_coo(
        rows, cols, values, shape=shape, pattern_signature=("dups", 1)
    )
    reference = host.scipy_matrix.tocsr()
    reference.sort_indices()
    np.testing.assert_array_equal(cupy.asnumpy(built.indptr), reference.indptr)
    np.testing.assert_array_equal(cupy.asnumpy(built.indices), reference.indices)
    assert cupy.asnumpy(built.data).tobytes() == reference.data.tobytes()


def test_sparse_gram_is_bitwise_reproducible(monkeypatch):
    # The SpGEMM Gram strategies (dense and COO results) multiply the cached
    # Aᵀ by diag(w)·A with cuSPARSE's default SpGEMM, observed reproducible.
    a, rng = _tall_with_long_rows(seed=4)
    a = a[:, :2000].tocsr()  # a Gram small enough to densify
    op = _operator(a)
    monkeypatch.setattr(cupy_adapter, "_GRAM_DENSE_MIN_DENSITY", 1.1)
    w = rng.uniform(0.1, 10.0, a.shape[0])
    reference = (a.T @ sp.diags(w) @ a).toarray()
    scale = float(np.max(np.abs(reference)))

    def gram():
        op._gram_memo.clear()  # recompute, not the memo
        return op.gram(cupy.asarray(w))

    def gram_coo():
        op._gram_coo_value = None
        return op.gram_coo(cupy.asarray(w))[2]

    assert _distinct(gram) == 1
    assert _distinct(gram_coo) == 1
    np.testing.assert_allclose(cupy.asnumpy(gram()), reference, atol=1e-13 * scale)


def test_pattern_transpose_plan_matches_csr2csc_bitwise():
    # Aᵀ via the shared pattern's permutation and via cuSPARSE's csr2csc are
    # the same arrays, so the two operator kinds give the same products.
    a, rng = _tall_with_long_rows(seed=5)
    coo = a.tocoo()
    adapter = cupy_adapter.CuPySparseAdapter()
    first, shared = (
        adapter.from_coo(
            cupy.asarray(coo.row),
            cupy.asarray(coo.col),
            cupy.asarray(coo.data),
            shape=a.shape,
            pattern_signature="plan",
        )
        for _ in range(2)
    )
    first._transpose_csr()  # the pattern's first Aᵀ: csr2csc
    own = cupy_adapter.SparseOperator(shared.cupy_matrix, import_namespace("cupy"))
    assert shared._structure is first._structure and own._structure is None
    t_plan, t_own = shared._transpose_csr(), own._transpose_csr()
    assert shared._structure._perm is not None  # the second: the plan's gather
    for name in ("indptr", "indices", "data"):
        assert cupy.array_equal(getattr(t_plan, name), getattr(t_own, name)), name
    y = cupy.asarray(rng.standard_normal(a.shape[0]))
    assert cupy.array_equal(shared.rmatvec(y), own.rmatvec(y))


def test_float32_products_are_reproducible():
    # The float32 instantiations of the kernels (data and x in float32).
    a, rng = _tall_with_long_rows(seed=8)
    a = a.astype(np.float32)
    op = _operator(a)
    X = rng.standard_normal((a.shape[1], 3)).astype(np.float32)
    Y = rng.standard_normal((a.shape[0], 3)).astype(np.float32)
    y = Y[:, 0].copy()
    cases = {
        "rmatvec": (lambda: op.rmatvec(cupy.asarray(y)), a.T @ y),
        "matmat": (lambda: op.matmat(cupy.asarray(X)), a @ X),
        "rmatmat": (lambda: op.rmatmat(cupy.asarray(Y)), a.T @ Y),
    }
    for name, (fn, reference) in cases.items():
        assert _distinct(fn) == 1, name
        actual = cupy.asnumpy(cupy.asarray(fn()))
        assert actual.dtype == np.float32, name
        scale = float(np.max(np.abs(reference)))
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-5 * scale)


def test_identity_probe_of_a_long_row_operator_is_exact():
    # Densifying by ``matmat(eye(n))`` (the dense routes' fallback) on rows of
    # 15000 entries takes the warp kernel: one launch, every entry exact.
    a, _ = _tall_with_long_rows(seed=6)
    a = a[:, :4000].tocsr()
    op = _operator(a)
    dense = cupy.asnumpy(op.matmat(cupy.eye(a.shape[1])))
    np.testing.assert_array_equal(dense, a.toarray())


def test_operator_over_the_cache_budget_keeps_no_transpose(monkeypatch):
    # The documented exception: above the budget the transposed products are
    # cuSPARSE's own, correct to round-off but not cached (and not
    # reproducible), so no second copy of a huge operator is held.
    monkeypatch.setattr(cupy_adapter, "_TRANSPOSE_CACHE_MAX_BYTES", 0)
    a, rng = _tall_with_long_rows(seed=7)
    op = _operator(a)
    y = rng.standard_normal(a.shape[0])
    reference = a.T @ y
    np.testing.assert_allclose(
        cupy.asnumpy(op.rmatvec(cupy.asarray(y))),
        reference,
        rtol=0,
        atol=1e-13 * float(np.max(np.abs(reference))),
    )
    assert op._transpose is None


def test_cached_transpose_follows_replaced_values():
    # The explicit Aᵀ behind rmatvec is cached per value array: replacing the
    # wrapped matrix's values must not serve the old transpose.
    a, rng = _tall_with_long_rows(seed=2)
    op = _operator(a)
    y = cupy.asarray(rng.standard_normal(a.shape[0]))
    before = cupy.asnumpy(op.rmatvec(y))
    np.testing.assert_allclose(before, a.T @ cupy.asnumpy(y), rtol=1e-12, atol=1e-6)
    new = _wide_ranged(rng, a.nnz)
    op.cupy_matrix.data = cupy.asarray(new)
    b = a.copy()
    b.data = new
    after = cupy.asnumpy(op.rmatvec(y))
    scale = float(np.max(np.abs(b.T @ cupy.asnumpy(y))))
    np.testing.assert_allclose(after, b.T @ cupy.asnumpy(y), rtol=0, atol=1e-13 * scale)
    gd = cupy.asnumpy(op.gram_diagonal(y * y))
    ref = np.asarray(b.multiply(b).T @ cupy.asnumpy(y * y)).reshape(-1)
    np.testing.assert_allclose(gd, ref, rtol=0, atol=1e-13 * float(np.max(ref)))


def test_operators_sharing_a_pattern_signature_keep_their_own_values():
    # One adapter, one signature, two value vectors: the compiled COO map is
    # shared, the products are each operator's own.
    a, rng = _tall_with_long_rows(seed=3)
    coo = a.tocoo()
    adapter = cupy_adapter.CuPySparseAdapter()
    rows, cols = cupy.asarray(coo.row), cupy.asarray(coo.col)
    y = rng.standard_normal(a.shape[0])
    for values in (coo.data, _wide_ranged(rng, coo.nnz)):
        op = adapter.from_coo(
            rows, cols, cupy.asarray(values), shape=a.shape, pattern_signature="p"
        )
        b = sp.csr_matrix((values, (coo.row, coo.col)), shape=a.shape)
        ref = b.T @ y
        np.testing.assert_allclose(
            cupy.asnumpy(op.rmatvec(cupy.asarray(y))),
            ref,
            rtol=0,
            atol=1e-13 * float(np.max(np.abs(ref))),
        )


# --- whole solves ------------------------------------------------------------

_SOLVE_SCRIPT = textwrap.dedent(
    """
    from tests.unit.test_cupy_sparse_determinism_gpu import _solve_fingerprint
    print("FINGERPRINT", _solve_fingerprint())
    """
)


class _CoupledDoubleWell:
    """A small nonconvex sparse NLP whose Jacobian has a dense column.

    ``min Σ (x_i² − 1)²/4 + t  s.t.  x_i² + x_{i+1}² − t ≤ 0``: every
    constraint involves the shared bound ``t`` (one dense Jacobian column,
    so ``Jᵀλ`` sums ``n − 1`` terms into one entry), the double wells make
    it nonconvex, and the Jacobian is emitted as COO with each entry split
    into three duplicate triplets (a duplicate sum of three terms depends on
    its order).
    """

    def __new__(cls, xp, n: int = 400):
        from ipax.backend.operators import COOOperator
        from ipax.problem.base import Problem

        m = n - 1
        idx = np.arange(m)
        parts = (0.3, 0.5, 1.2)
        rows = np.concatenate([idx] * 9)
        cols = np.concatenate([idx] * 3 + [idx + 1] * 3 + [np.full(m, n)] * 3)
        rows_d = xp.asarray(rows)
        cols_d = xp.asarray(cols)

        class Coupled(Problem):
            @property
            def n_vars(self) -> int:
                return n + 1

            def objective(self, z):
                x = z[:n]
                return xp.sum((x * x - 1.0) ** 2) / 4.0 + z[n]

            def gradient(self, z):
                x = z[:n]
                return xp.concat((x * (x * x - 1.0), xp.ones((1,), dtype=z.dtype)))

            def ineq_constraints(self, z):
                x = z[:n]
                return x[:-1] ** 2 + x[1:] ** 2 - z[n]

            def ineq_jacobian(self, z):
                x = z[:n]
                pieces = [f * src for src in (x[:-1], x[1:]) for f in parts]
                pieces += [-xp.ones((m,), dtype=z.dtype) * f / 2.0 for f in parts]
                values = xp.concat(pieces)
                return COOOperator(
                    rows_d, cols_d, values, (m, n + 1), pattern_key="coupled"
                )

        return Coupled()


def _solve_fingerprint(deterministic: bool = True) -> str:
    import ipax
    from ipax.options import SparseOptions

    xp = import_namespace("cupy")
    problem = _CoupledDoubleWell(xp)
    rng = np.random.default_rng(7)
    x0 = xp.asarray(np.concatenate([rng.uniform(-0.5, 0.5, 400), [5.0]]))
    result = ipax.solve(
        problem,
        x0,
        options=ipax.Options(
            linsolve="sparse",
            max_iter=60,
            sparse=SparseOptions(deterministic=deterministic),
        ),
    )
    digest = hashlib.sha256()
    digest.update(cupy.asnumpy(cupy.asarray(result.x)).tobytes())
    # Every history field except the two wall-clock timings.
    untimed = [
        dataclasses.replace(row, problem_time=0.0, step_solve_time=0.0)
        for row in result.history
    ]
    digest.update(repr(untimed).encode())
    digest.update(
        repr((result.status, result.n_iter, float(result.objective))).encode()
    )
    return digest.hexdigest()


def _require_deterministic_cudss() -> None:
    try:
        cudss = cupy_adapter._load_cudss()
    except ImportError as exc:
        pytest.skip(f"cuDSS runtime unavailable: {exc}")
    require_deterministic_cudss(cudss)


def test_deterministic_solve_repeats_bitwise_in_one_process():
    _require_deterministic_cudss()
    assert _solve_fingerprint() == _solve_fingerprint()


@pytest.mark.slow
def test_deterministic_solve_repeats_bitwise_across_processes():
    _require_deterministic_cudss()
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONUTF8="1")
    prints = []
    for _ in range(2):
        out = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", _SOLVE_SCRIPT],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        ).stdout
        prints.append([line for line in out.splitlines() if "FINGERPRINT" in line])
    assert prints[0] and prints[0] == prints[1]
