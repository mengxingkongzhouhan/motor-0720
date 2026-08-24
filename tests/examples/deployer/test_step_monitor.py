# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from __future__ import annotations

import subprocess

import pytest
from lib.tui import step
from lib.tui.actions import _DeployActionsMixin

NAMESPACE = "mindie-motor"


def _kubectl_rows(*rows: tuple[str, str]) -> str:
    return "".join(f"{name} 1/1 {status} 0 5m\n" for name, status in rows)


def _fake_kubectl(monkeypatch, stdout: str, returncode: int = 0) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(step.subprocess, "run", fake_run)
    return calls


# ---------------------------------------------------------------------------
# is_engine_pod: engine naming follows engine_type / CRD name, never "vllm"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pod_name",
    [
        # multi_deployment, one Deployment per engine instance
        "vllm-p0-7d8f9b6c5-x2klm",
        "sglang-d1-7d8f9b6c5-x2klm",
        "mindie-server-p0-7d8f9b6c5-x2klm",
        "vllm-u0-7d8f9b6c5-x2klm",
        "vllm-e0-7d8f9b6c5-x2klm",
        # infer_service_set, one StatefulSet per engine instance
        "vllm-0-prefill-0",
        "sglang-1-decode-3",
        "mindie-server-0-union-0",
        "vllm-0-encode-0",
    ],
)
def test_is_engine_pod_accepts_every_engine_type_and_deploy_mode(pod_name):
    assert step.is_engine_pod(pod_name) is True


@pytest.mark.parametrize(
    "pod_name",
    [
        # multi_deployment control plane
        "mindie-motor-controller-7d8f9b6c5-x2klm",
        "mindie-motor-coordinator-7d8f9b6c5-x2klm",
        "mindie-motor-kv-store-7d8f9b6c5-x2klm",
        "mindie-motor-kv-conductor-7d8f9b6c5-x2klm",
        "mindie-motor-mf-store-7d8f9b6c5-x2klm",
        # infer_service_set control plane shares the engine naming scheme
        "vllm-0-controller-0",
        "vllm-0-coordinator-0",
        "vllm-0-kv-store-0",
        "vllm-0-kv-conductor-0",
    ],
)
def test_is_engine_pod_rejects_control_plane(pod_name):
    assert step.is_engine_pod(pod_name) is False


# ---------------------------------------------------------------------------
# Pod discovery
# ---------------------------------------------------------------------------


def test_shell_get_pod_returns_running_engine_pods_for_sglang(monkeypatch):
    _fake_kubectl(
        monkeypatch,
        _kubectl_rows(
            ("mindie-motor-controller-7d8f9b6c5-aaaaa", "Running"),
            ("mindie-motor-coordinator-7d8f9b6c5-bbbbb", "Running"),
            ("sglang-p0-7d8f9b6c5-ccccc", "Running"),
            ("sglang-d0-7d8f9b6c5-ddddd", "Pending"),
        ),
    )

    assert step.shell_get_pod(NAMESPACE) == ["sglang-p0-7d8f9b6c5-ccccc"]


def test_shell_get_pod_returns_running_engine_pods_for_infer_service_set(monkeypatch):
    _fake_kubectl(
        monkeypatch,
        _kubectl_rows(
            ("vllm-0-controller-0", "Running"),
            ("vllm-0-prefill-0", "Running"),
            ("vllm-0-decode-0", "Running"),
        ),
    )

    assert step.shell_get_pod(NAMESPACE) == ["vllm-0-prefill-0", "vllm-0-decode-0"]


def test_shell_get_pod_returns_none_when_kubectl_fails(monkeypatch):
    _fake_kubectl(monkeypatch, "", returncode=1)

    assert step.shell_get_pod(NAMESPACE) is None


def test_shell_get_pod_returns_none_when_kubectl_times_out(monkeypatch):
    def fake_run(cmd, **_kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=step.KUBECTL_QUERY_TIMEOUT)

    monkeypatch.setattr(step.subprocess, "run", fake_run)

    assert step.shell_get_pod(NAMESPACE) is None


def test_shell_get_engine_pods_keeps_status(monkeypatch):
    _fake_kubectl(
        monkeypatch,
        _kubectl_rows(
            ("vllm-p0-7d8f9b6c5-ccccc", "Running"),
            ("vllm-d0-7d8f9b6c5-ddddd", "ImagePullBackOff"),
        ),
    )

    assert step.shell_get_engine_pods(NAMESPACE) == [
        ("vllm-p0-7d8f9b6c5-ccccc", "Running"),
        ("vllm-d0-7d8f9b6c5-ddddd", "ImagePullBackOff"),
    ]


# ---------------------------------------------------------------------------
# Wait-loop diagnostics
# ---------------------------------------------------------------------------


def test_format_status_summary_counts_only_non_running():
    engine_pods = [
        ("vllm-p0-7d8f9b6c5-aaaaa", "Running"),
        ("vllm-d0-7d8f9b6c5-bbbbb", "Pending"),
        ("vllm-d1-7d8f9b6c5-ccccc", "Pending"),
        ("vllm-d2-7d8f9b6c5-ddddd", "ImagePullBackOff"),
    ]

    assert step.format_status_summary(engine_pods) == "ImagePullBackOff=1 Pending=2"


def test_print_wait_diagnosis_reports_missing_workload(monkeypatch, capsys):
    _fake_kubectl(monkeypatch, _kubectl_rows(("mindie-motor-controller-7d8f9b6c5-aaaaa", "Running")))

    step.print_wait_diagnosis(NAMESPACE, [], 32)

    output = capsys.readouterr().out
    assert "No engine pod exists" in output
    assert "mindie-motor-controller-7d8f9b6c5-aaaaa" in output
    assert f"kubectl -n {NAMESPACE} get events" in output


def test_print_wait_diagnosis_reports_stuck_engine_pods(monkeypatch, capsys):
    _fake_kubectl(monkeypatch, "")

    step.print_wait_diagnosis(NAMESPACE, [("vllm-p0-7d8f9b6c5-aaaaa", "Pending")], 2)

    output = capsys.readouterr().out
    assert "vllm-p0-7d8f9b6c5-aaaaa  Pending" in output
    assert "describe pod" in output


def test_start_monitor_stops_waiting_once_all_engine_pods_run(monkeypatch):
    polls = [
        [],
        [("vllm-p0-7d8f9b6c5-aaaaa", "Pending")],
        [("vllm-p0-7d8f9b6c5-aaaaa", "Running")],
    ]
    monitored: list[list[str]] = []

    monkeypatch.setattr(step.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(step, "shell_get_engine_pods", lambda _ns: polls.pop(0))
    monkeypatch.setattr(
        step.VLLMProgressMonitor,
        "start",
        lambda _self, list_pod, _ns: monitored.append(list_pod),
    )

    step.start_monitor(NAMESPACE, 1)

    assert monitored == [["vllm-p0-7d8f9b6c5-aaaaa"]]
    assert not polls


def test_start_monitor_diagnoses_a_stalled_wait(monkeypatch, capsys):
    monkeypatch.setattr(step.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(step, "shell_get_engine_pods", lambda _ns: [])
    monkeypatch.setattr(step, "shell_get_all_pods", lambda _ns: [])
    # Time only advances past the diagnosis threshold on the second poll
    clock = iter([0.0, 0.0, step.POD_WAIT_DIAGNOSE_AFTER, step.POD_WAIT_DIAGNOSE_AFTER])
    monkeypatch.setattr(step.time, "monotonic", lambda: next(clock))

    with pytest.raises(StopIteration):
        step.start_monitor(NAMESPACE, 1)

    assert "No engine pod exists" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Pod restart identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pod_name", "expected"),
    [
        ("vllm-p0-7d8f9b6c5-x2klm", "vllm-p0"),
        ("mindie-server-d1-7d8f9b6c5-x2klm", "mindie-server-d1"),
        # StatefulSet pod names are already stable across restarts
        ("vllm-0-prefill-1", "vllm-0-prefill-1"),
    ],
)
def test_pod_prefix_identifies_role_across_restarts(pod_name, expected):
    assert _DeployActionsMixin._pod_prefix(pod_name) == expected
