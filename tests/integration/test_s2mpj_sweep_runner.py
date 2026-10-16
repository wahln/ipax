# Copyright 2026 Niklas Wahl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Crash-culprit identification in the S2MPJ accuracy sweep's worker pool.

A full-corpus sweep is unattended for hours and does meet native crashes (a
backend factorization on an overflowed model takes the whole worker process
down, with no chance to report). The only way to learn *which* problem did it is
a marker the worker itself wrote — the parent's pending-futures list cannot do
the job, because every problem is submitted up front, so it contains every
unstarted problem too and names the entire queue.

These tests stub the corpus, so they run without an S2MPJ checkout.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.runners import s2mpj as runner


class _StubCase:
    """A corpus case whose build refuses the problem (the `no_objective` path)."""

    name = "s2mpj/STUB"

    def build(self, xp):
        raise NotImplementedError("objective-free")


def _args(**over):
    """The positional arguments of :func:`_run_problem_cases`."""
    base = {
        "root": "/nowhere",
        "bare": "CRASHER",
        "backend": "numpy",
        "size": None,
        "feasibility": False,
        "configs": runner.default_configs(10, 5.0),
        "global_max_vars": 0,
        "max_build_seconds": 0.0,
    }
    base.update(over)
    return tuple(base.values())


def test_restoration_linear_solver_can_be_overridden_for_a_paired_sweep():
    configs = runner.default_configs(10, 5.0, restoration_linear_solver="krylov")
    assert configs
    assert all(
        options.restoration.linear_solver == "krylov" for _, options, _ in configs
    )


def test_worker_marks_the_problem_and_backend_it_is_running(monkeypatch, tmp_path):
    prefix = tmp_path / "sweep.inflight"
    seen: dict[str, str] = {}

    def _peek(*args, **kwargs):
        marker = next(tmp_path.glob("sweep.inflight.*"))
        seen["content"] = marker.read_text(encoding="utf-8")
        return [_StubCase()]

    monkeypatch.setattr(runner, "s2mpj_problems", _peek)
    rows, skip = runner._run_problem_cases(*_args(), inflight_prefix=str(prefix))

    assert skip == "no_objective" and rows == []
    # The backend matters: a resume is keyed on (backend, problem).
    assert seen["content"] == "numpy CRASHER"
    assert list(tmp_path.glob("sweep.inflight.*")) == []


def test_the_marker_is_cleared_even_when_the_worker_raises(monkeypatch, tmp_path):
    prefix = tmp_path / "sweep.inflight"

    def _boom(*args, **kwargs):
        raise RuntimeError("corpus exploded")

    monkeypatch.setattr(runner, "s2mpj_problems", _boom)
    with pytest.raises(RuntimeError, match="corpus exploded"):
        runner._run_problem_cases(*_args(), inflight_prefix=str(prefix))

    # Only a hard kill may leave a marker behind, or every crashed run would
    # accuse a problem that merely errored.
    assert list(tmp_path.glob("sweep.inflight.*")) == []


def test_no_prefix_means_no_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "s2mpj_problems", lambda *a, **k: [_StubCase()])
    runner._run_problem_cases(*_args())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("backend", "policy", "applied"),
    [
        ("cupy", (0, 1e-5), [(0, 1e-5)]),
        ("cupy", (None, None), []),
        ("numpy", (3, None), []),
    ],
)
def test_cudss_policy_override_reaches_only_cupy_workers(
    monkeypatch, backend, policy, applied
):
    # The A/B lever for the cuDSS perturbed-pivot policy is applied per worker
    # to the configs' options, only on the backend it concerns, and only when
    # asked for.
    seen: list[tuple[object, object]] = []

    def _record(configs, steps, res):
        seen.append((steps, res))
        return configs

    monkeypatch.setattr(runner, "import_namespace", lambda name: None)
    monkeypatch.setattr(runner, "_with_cudss_policy", _record)
    monkeypatch.setattr(runner, "s2mpj_problems", lambda *a, **k: [_StubCase()])
    runner._run_problem_cases(*_args(backend=backend), cudss_policy=policy)
    assert seen == applied


def test_cudss_policy_is_set_through_sparse_options():
    # The flags reach the adapter as SparseOptions fields (the public path),
    # not by patching adapter module state; None keeps the option default.
    from ipax.options import SparseOptions

    configs = runner.default_configs(10, 5.0)
    both = runner._with_cudss_policy(configs, 0, 1e-5)
    assert [label for label, _, _ in both] == [label for label, _, _ in configs]
    for (_, before, cap_before), (_, after, cap_after) in zip(
        configs, both, strict=True
    ):
        assert after.sparse.perturbed_pivot_refinement_steps == 0
        assert after.sparse.perturbed_pivot_max_residual == 1e-5
        assert after.sparse.kkt_route == before.sparse.kkt_route
        assert after.hessian == before.hessian and cap_after == cap_before

    (_, steps_only, _), *_ = runner._with_cudss_policy(configs, 3, None)
    assert steps_only.sparse.perturbed_pivot_refinement_steps == 3
    default = SparseOptions().perturbed_pivot_max_residual
    assert steps_only.sparse.perturbed_pivot_max_residual == default

    assert runner._with_cudss_policy(configs, None, None) is configs
    with pytest.raises(ValueError, match="perturbed_pivot_refinement_steps"):
        runner._with_cudss_policy(configs, -1, None)


def _fake_checkout(tmp_path):
    root = tmp_path / "s2mpj"
    (root / "python_problems").mkdir(parents=True)
    (root / "s2mpjlib.py").write_text("", encoding="utf-8")
    return str(root)


@pytest.mark.parametrize(
    "flag", [["--cudss-refinement-steps", "-1"], ["--cudss-max-residual", "inf"]]
)
def test_invalid_cudss_policy_flags_fail_the_parse(tmp_path, capsys, flag):
    # Rejected up front, not by the first worker after the sweep started.
    argv = ["--dir", _fake_checkout(tmp_path), "--names", "X", "--backends", "cupy"]
    with pytest.raises(SystemExit):
        runner.main([*argv, *flag, "--out", str(tmp_path / "r")])
    assert flag[0] in capsys.readouterr().err


def test_resume_refuses_to_mix_cudss_policies(tmp_path, capsys):
    # A report records the policy its CuPy rows ran with; resuming it under a
    # different one would silently mix two arms of an A/B.
    out = tmp_path / "r"
    env = {"cudss_policy": {"max_refinement_steps": 0, "perturbed_max_residual": None}}
    out.with_suffix(".json").write_text(
        json.dumps({"environment": env, "results": []}), encoding="utf-8"
    )
    argv = ["--dir", _fake_checkout(tmp_path), "--names", "X", "--backends", "cupy"]
    with pytest.raises(SystemExit):
        runner.main([*argv, "--resume", "--out", str(out)])
    assert "would mix cuDSS policies" in capsys.readouterr().err
