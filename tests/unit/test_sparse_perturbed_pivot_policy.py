"""``SparseOptions``' adapter policies reach the sparse adapters.

Adapters whose factorization uses static pivoting (the CUDA adapter) refine
and certify solves on a perturbed factor; their policy is exposed through
``SparseOptions.perturbed_pivot_refinement_steps`` /
``perturbed_pivot_max_residual``. The core forwards it to the adapter through
the optional, duck-typed ``configure_perturbed_pivots`` hook, which adapters
that never perturb pivots (SciPy/Feral) simply do not implement. These tests
pin that threading without a GPU; the CuPy-side hook is covered with the fake
cuDSS in ``test_cupy_sparse_adapter.py`` and on real cuDSS in
``test_cudss_refinement_gpu.py``.

``SparseOptions.deterministic`` (run-to-run reproducible factorizations where
the backend offers them) travels the same way through its own optional
``configure_deterministic(enabled=...)`` hook; the tests at the end of this
file pin it.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("scipy.sparse")

import ipax.backend.sparse as sparse_registry
from ipax.backend.namespace import Capabilities
from ipax.backend.operators import LinearOperator
from ipax.backend.sparse import _routing
from ipax.backend.sparse._routing import DeviceRoutingSparseAdapter
from ipax.linalg.solver import select_solver
from ipax.linalg.sparse import SparseDirectSolver
from ipax.options import Options, SparseOptions
from ipax.testing.backends import import_namespace
from tests._helpers import array, assert_allclose

pytestmark = pytest.mark.sparse


class _COOOperator(LinearOperator):
    """Symmetric 2×2 ``[[v0, v1], [v1, v2]]`` emitted as COO triplets."""

    def __init__(self, xp: Any, values: Any) -> None:
        self._values = values
        self._rows = xp.asarray([0, 0, 1, 1])
        self._cols = xp.asarray([0, 1, 0, 1])

    @property
    def shape(self) -> tuple[int, int]:
        return 2, 2

    def matvec(self, v: Any) -> Any:  # pragma: no cover - direct route only
        raise NotImplementedError

    def to_coo(self, like: Any = None) -> Any:
        del like
        return self._rows, self._cols, self._values, (2, 2)

    def coo_values(self, like: Any = None) -> Any:
        del like
        return self._values

    def coo_pattern_signature(self) -> object:
        return ("policy-test",)

    def symmetry_hint(self) -> bool:
        return True


class _RecordingAdapter:
    """Adapter stand-in recording the hook call and delegating to SciPy."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.events: list[tuple[str, object]] = []

    def configure_perturbed_pivots(
        self, *, refinement_steps: int | None, max_residual: float | None
    ) -> None:
        self.events.append(("configure", (refinement_steps, max_residual)))

    def configure_deterministic(self, *, enabled: bool) -> None:
        self.events.append(("deterministic", enabled))

    def from_coo(self, *args: Any, **kwargs: Any) -> Any:
        self.events.append(("from_coo", None))
        return self._inner.from_coo(*args, **kwargs)

    def solver(self, *, require_inertia: bool = False) -> Any:
        self.events.append(("solver", None))
        return self._inner.solver(require_inertia=require_inertia)


def _numpy():
    try:
        return import_namespace("numpy")
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"NumPy namespace unavailable: {exc}")


def _caps() -> Capabilities:
    return Capabilities(
        name="test",
        has_linalg=True,
        linalg_functions=frozenset({"cholesky", "solve"}),
        has_sparse_adapter=True,
        supports_autodiff=False,
        devices=("cpu",),
        default_float="float64",
    )


def _install_recorder(monkeypatch: pytest.MonkeyPatch) -> list[_RecordingAdapter]:
    from ipax.backend.sparse.numpy_scipy import SciPySparseAdapter

    made: list[_RecordingAdapter] = []

    def fake_get_sparse_adapter(xp: Any) -> _RecordingAdapter:
        del xp
        adapter = _RecordingAdapter(SciPySparseAdapter())
        made.append(adapter)
        return adapter

    monkeypatch.setattr(sparse_registry, "get_sparse_adapter", fake_get_sparse_adapter)
    return made


def _factor_and_solve(solver: SparseDirectSolver) -> None:
    xp = _numpy()
    solver.factor(_COOOperator(xp, array(xp, [2.0, 1.0, 1.0, 3.0])))
    # [[2,1],[1,3]] x = [1,2] ⇒ x = [1/5, 3/5].
    assert_allclose(xp, solver.solve(array(xp, [1.0, 2.0])), array(xp, [0.2, 0.6]))


@pytest.mark.parametrize("kkt_route", ["augmented", "normal_equations", "auto"])
def test_select_solver_threads_the_policy_to_the_adapter(monkeypatch, kkt_route):
    made = _install_recorder(monkeypatch)
    solver = select_solver(
        n_vars=10,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(
            linsolve="sparse",
            sparse=SparseOptions(
                kkt_route=kkt_route,
                perturbed_pivot_refinement_steps=3,
                perturbed_pivot_max_residual=0.25,
            ),
        ),
    )
    assert isinstance(solver, SparseDirectSolver)
    assert solver._perturbed_pivot_policy == (3, 0.25)
    if kkt_route == "normal_equations":
        return  # the 2×2 operator has no NE emission; the threading is the same
    _factor_and_solve(solver)
    (adapter,) = made
    # Configured once, before the adapter builds anything or makes a solver.
    assert adapter.events[0] == ("configure", (3, 0.25))
    assert [name for name, _ in adapter.events].count("configure") == 1


def test_auto_selected_normal_equations_route_carries_the_policy():
    # linsolve="auto" picking the sparse-NE form for a tall sparse-Gram
    # problem must honour Options.sparse too, not only linsolve="sparse".
    solver = select_solver(
        n_vars=15_000,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(
            linsolve="auto",
            sparse=SparseOptions(
                perturbed_pivot_refinement_steps=0, perturbed_pivot_max_residual=0.5
            ),
        ),
        m_ineq=300_000,
        ineq_gram_capable=lambda: True,
        ineq_density=lambda: 0.01,
        ineq_gram_fill=lambda: 0.002,
    )
    assert isinstance(solver, SparseDirectSolver)
    assert solver.form == "normal_equations"
    assert solver._perturbed_pivot_policy == (0, 0.5)


def test_default_facade_leaves_the_adapter_defaults_alone(monkeypatch):
    # A SparseDirectSolver built without a policy (benchmarks, user code) never
    # calls the hook: the adapter keeps its own defaults.
    made = _install_recorder(monkeypatch)
    _factor_and_solve(SparseDirectSolver())
    (adapter,) = made
    assert all(name != "configure" for name, _ in adapter.events)


def test_scipy_adapter_without_the_hook_ignores_the_policy():
    # The SciPy/Feral adapter never perturbs pivots and has no hook: a
    # non-default policy must be ignored without error on the real adapter.
    from ipax.backend.sparse.numpy_scipy import SciPySparseAdapter

    assert not hasattr(SciPySparseAdapter(), "configure_perturbed_pivots")
    solver = select_solver(
        n_vars=10,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(
            linsolve="sparse",
            sparse=SparseOptions(
                perturbed_pivot_refinement_steps=0, perturbed_pivot_max_residual=1e-6
            ),
        ),
    )
    _factor_and_solve(solver)


# --- the device-routing adapter (Torch/JAX) forwards the hook ----------------


class _FakeDeviceArray:
    """Minimal array stand-in exposing only a DLPack device kind."""

    def __init__(self, kind: int) -> None:
        self._kind = kind

    def __dlpack_device__(self) -> tuple[int, int]:
        return (self._kind, 0)


class _HookedDelegate:
    def __init__(self) -> None:
        self.policy: tuple[object, object] | None = None
        self.deterministic: bool | None = None

    def configure_deterministic(self, *, enabled: bool) -> None:
        self.deterministic = enabled

    def configure_perturbed_pivots(
        self, *, refinement_steps: int | None, max_residual: float | None
    ) -> None:
        self.policy = (refinement_steps, max_residual)

    def from_coo(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return "operator"


def test_routing_adapter_forwards_the_policy_to_a_new_cuda_delegate(monkeypatch):
    delegate = _HookedDelegate()
    monkeypatch.setattr(_routing, "_cuda_adapter", lambda: delegate)
    adapter = DeviceRoutingSparseAdapter()
    adapter.configure_perturbed_pivots(refinement_steps=4, max_residual=0.125)
    cuda = _FakeDeviceArray(2)
    adapter.from_coo(cuda, cuda, cuda, shape=(1, 1))
    assert delegate.policy == (4, 0.125)


def test_routing_adapter_forwards_the_policy_to_an_existing_delegate(monkeypatch):
    delegate = _HookedDelegate()
    monkeypatch.setattr(_routing, "_cuda_adapter", lambda: delegate)
    adapter = DeviceRoutingSparseAdapter()
    cuda = _FakeDeviceArray(2)
    adapter.from_coo(cuda, cuda, cuda, shape=(1, 1))
    assert delegate.policy is None
    adapter.configure_perturbed_pivots(refinement_steps=0, max_residual=0.5)
    assert delegate.policy == (0, 0.5)


def test_routing_adapter_cpu_delegate_ignores_the_policy():
    # The CPU delegate (SciPy) has no hook: the routed factor/solve works as
    # before under any policy.
    xp = _numpy()
    adapter = DeviceRoutingSparseAdapter()
    adapter.configure_perturbed_pivots(refinement_steps=0, max_residual=1e-6)
    K = adapter.from_coo(
        xp.asarray([0, 0, 1, 1]),
        xp.asarray([0, 1, 0, 1]),
        array(xp, [2.0, 1.0, 1.0, 3.0]),
        shape=(2, 2),
        symmetric=True,
    )
    solver = adapter.solver()
    solver.factor(K)
    assert_allclose(xp, solver.solve(array(xp, [1.0, 2.0])), array(xp, [0.2, 0.6]))


# --- the public options path on every backend with a sparse adapter ----------


def test_public_options_path_threads_the_policy_on_each_backend(namespace):
    # NumPy reaches the SciPy adapter (no hook: ignored); CuPy its own hooked
    # adapter; Torch reaches the
    # real DeviceRoutingSparseAdapter, which must remember the policy for its
    # delegates; array_api_strict has no sparse adapter at all.
    from ipax.backend.namespace import capabilities

    xp = namespace
    caps = capabilities(xp)
    if not caps.has_sparse_adapter:
        pytest.skip("no sparse adapter for this backend")
    solver = select_solver(
        n_vars=10,
        has_equalities=False,
        capabilities=caps,
        options=Options(
            linsolve="sparse",
            sparse=SparseOptions(
                perturbed_pivot_refinement_steps=0, perturbed_pivot_max_residual=1e-6
            ),
        ),
    )
    assert isinstance(solver, SparseDirectSolver)
    solver.factor(_COOOperator(xp, array(xp, [2.0, 1.0, 1.0, 3.0])))
    assert_allclose(xp, solver.solve(array(xp, [1.0, 2.0])), array(xp, [0.2, 0.6]))
    adapter = solver._adapter
    if isinstance(adapter, DeviceRoutingSparseAdapter):
        assert adapter._perturbed_pivots == (0, 1e-6)
    elif hasattr(adapter, "configure_perturbed_pivots"):
        # The CuPy adapter applies the policy to itself.
        assert (adapter._max_refinement_steps, adapter._perturbed_max_residual) == (
            0,
            1e-6,
        )
    else:
        assert not hasattr(adapter, "configure_perturbed_pivots")


# --- SparseOptions.deterministic: the configure_deterministic hook -----------


@pytest.mark.parametrize("kkt_route", ["augmented", "normal_equations", "auto"])
def test_select_solver_threads_determinism_to_the_adapter(monkeypatch, kkt_route):
    made = _install_recorder(monkeypatch)
    solver = select_solver(
        n_vars=10,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(
            linsolve="sparse",
            sparse=SparseOptions(kkt_route=kkt_route, deterministic=True),
        ),
    )
    assert isinstance(solver, SparseDirectSolver)
    assert solver._deterministic is True
    if kkt_route == "normal_equations":
        return  # the 2×2 operator has no NE emission; the threading is the same
    _factor_and_solve(solver)
    (adapter,) = made
    names = [name for name, _ in adapter.events]
    # Configured once, before the adapter builds anything or makes a solver.
    assert ("deterministic", True) in adapter.events
    assert names.count("deterministic") == 1
    assert names.index("deterministic") < names.index("from_coo")


def test_auto_selected_normal_equations_route_carries_determinism():
    solver = select_solver(
        n_vars=15_000,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(linsolve="auto", sparse=SparseOptions(deterministic=True)),
        m_ineq=300_000,
        ineq_gram_capable=lambda: True,
        ineq_density=lambda: 0.01,
        ineq_gram_fill=lambda: 0.002,
    )
    assert isinstance(solver, SparseDirectSolver)
    assert solver.form == "normal_equations"
    assert solver._deterministic is True


def test_default_options_never_call_the_determinism_hook(monkeypatch):
    # deterministic=False (the default) leaves the adapter in its own default
    # mode: the hook is not called at all, through the options path or not.
    made = _install_recorder(monkeypatch)
    solver = select_solver(
        n_vars=10,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(linsolve="sparse"),
    )
    _factor_and_solve(solver)
    _factor_and_solve(SparseDirectSolver())
    assert all(name != "deterministic" for a in made for name, _ in a.events)


def test_scipy_adapter_without_the_determinism_hook_ignores_it():
    # The SciPy/Feral factorizations are reproducible already; the adapter has
    # no hook and the option must be ignored without error.
    from ipax.backend.sparse.numpy_scipy import SciPySparseAdapter

    assert not hasattr(SciPySparseAdapter(), "configure_deterministic")
    solver = select_solver(
        n_vars=10,
        has_equalities=False,
        capabilities=_caps(),
        options=Options(linsolve="sparse", sparse=SparseOptions(deterministic=True)),
    )
    _factor_and_solve(solver)


def test_routing_adapter_forwards_determinism_to_a_new_cuda_delegate(monkeypatch):
    delegate = _HookedDelegate()
    monkeypatch.setattr(_routing, "_cuda_adapter", lambda: delegate)
    adapter = DeviceRoutingSparseAdapter()
    adapter.configure_deterministic(enabled=True)
    cuda = _FakeDeviceArray(2)
    adapter.from_coo(cuda, cuda, cuda, shape=(1, 1))
    assert delegate.deterministic is True
    # The perturbed-pivot policy was never configured: not forwarded.
    assert delegate.policy is None


def test_routing_adapter_forwards_determinism_to_an_existing_delegate(monkeypatch):
    delegate = _HookedDelegate()
    monkeypatch.setattr(_routing, "_cuda_adapter", lambda: delegate)
    adapter = DeviceRoutingSparseAdapter()
    cuda = _FakeDeviceArray(2)
    adapter.from_coo(cuda, cuda, cuda, shape=(1, 1))
    assert delegate.deterministic is None
    adapter.configure_deterministic(enabled=True)
    assert delegate.deterministic is True


def test_routing_adapter_cpu_delegate_ignores_determinism():
    xp = _numpy()
    adapter = DeviceRoutingSparseAdapter()
    adapter.configure_deterministic(enabled=True)
    K = adapter.from_coo(
        xp.asarray([0, 0, 1, 1]),
        xp.asarray([0, 1, 0, 1]),
        array(xp, [2.0, 1.0, 1.0, 3.0]),
        shape=(2, 2),
        symmetric=True,
    )
    solver = adapter.solver()
    solver.factor(K)
    assert_allclose(xp, solver.solve(array(xp, [1.0, 2.0])), array(xp, [0.2, 0.6]))


def test_public_options_path_threads_determinism_on_each_backend(namespace):
    from ipax.backend.namespace import capabilities

    xp = namespace
    caps = capabilities(xp)
    if not caps.has_sparse_adapter:
        pytest.skip("no sparse adapter for this backend")
    solver = select_solver(
        n_vars=10,
        has_equalities=False,
        capabilities=caps,
        options=Options(linsolve="sparse", sparse=SparseOptions(deterministic=True)),
    )
    assert isinstance(solver, SparseDirectSolver)
    solver.factor(_COOOperator(xp, array(xp, [2.0, 1.0, 1.0, 3.0])))
    assert_allclose(xp, solver.solve(array(xp, [1.0, 2.0])), array(xp, [0.2, 0.6]))
    adapter = solver._adapter
    if isinstance(adapter, DeviceRoutingSparseAdapter):
        assert adapter._deterministic is True
    elif hasattr(adapter, "configure_deterministic"):
        assert adapter._deterministic is True  # the CuPy adapter
    else:
        assert not hasattr(adapter, "configure_deterministic")


@pytest.mark.parametrize("value", ["no", 1, None])
def test_facade_rejects_a_non_bool_determinism_flag(value):
    # A truthy non-bool must not switch the mode on silently.
    with pytest.raises(ValueError, match="deterministic"):
        SparseDirectSolver(deterministic=value)
