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
from lib import utils

APPLY_CMD = ["kubectl", "apply", "-f", "vllm_p0.yaml", "-n", "mindie-motor"]


def _fake_run(monkeypatch, returncode: int, stdout: str = "", stderr: str = ""):
    def fake_run(cmd, **_kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(utils.subprocess, "run", fake_run)


def test_exec_cmd_logs_output_on_success(monkeypatch, caplog):
    _fake_run(monkeypatch, 0, stdout="deployment.apps/vllm-p0 created")

    with caplog.at_level("INFO", logger=utils.logger.name):
        utils.exec_cmd(APPLY_CMD)

    assert "deployment.apps/vllm-p0 created" in caplog.text


def test_exec_cmd_raises_on_failure_with_stderr(monkeypatch):
    _fake_run(
        monkeypatch,
        1,
        stderr='error: resource mapping not found for kind "InferServiceSet"',
    )

    with pytest.raises(RuntimeError) as excinfo:
        utils.exec_cmd(APPLY_CMD)

    message = str(excinfo.value)
    assert "exit 1" in message
    assert "kubectl apply -f vllm_p0.yaml -n mindie-motor" in message
    assert "InferServiceSet" in message


def test_exec_cmd_falls_back_to_stdout_when_stderr_is_empty(monkeypatch):
    _fake_run(monkeypatch, 1, stdout="the server could not find the requested resource")

    with pytest.raises(RuntimeError, match="could not find the requested resource"):
        utils.exec_cmd(APPLY_CMD)


def test_safe_exec_cmd_propagates_kubectl_failure(monkeypatch):
    _fake_run(monkeypatch, 1, stderr="Forbidden")

    with pytest.raises(RuntimeError, match="Forbidden"):
        utils.safe_exec_cmd(APPLY_CMD)


def test_safe_exec_cmd_succeeds_when_kubectl_succeeds(monkeypatch):
    _fake_run(monkeypatch, 0, stdout="deployment.apps/vllm-p0 configured")

    utils.safe_exec_cmd(APPLY_CMD)
