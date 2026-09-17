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
