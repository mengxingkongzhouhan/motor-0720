# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import subprocess

import pytest

import lib.constant as C
from lib.config_validator import validate_node_selectors, validate_pod_npu_against_hardware


def test_validate_pod_npu_accepts_a2_eight_cards():
    validate_pod_npu_against_hardware(
        {
            C.HARDWARE_TYPE: C.HARDWARE_TYPE_800I_A2,
            C.P_POD_NPU_NUM: 8,
            C.D_POD_NPU_NUM: 8,
        }
    )


def test_validate_pod_npu_rejects_a2_sixteen_cards():
    with pytest.raises(ValueError, match="p_pod_npu_num=16 exceeds cards per node \\(8\\)"):
        validate_pod_npu_against_hardware(
            {
                C.HARDWARE_TYPE: C.HARDWARE_TYPE_800I_A2,
                C.P_POD_NPU_NUM: 16,
                C.D_POD_NPU_NUM: 8,
            }
        )


def test_validate_pod_npu_accepts_a3_sixteen_cards():
    validate_pod_npu_against_hardware(
        {
            C.HARDWARE_TYPE: C.HARDWARE_TYPE_800I_A3,
            C.P_POD_NPU_NUM: 16,
            C.D_POD_NPU_NUM: 16,
        }
    )


def test_validate_pod_npu_rejects_hybrid_over_a2():
    with pytest.raises(ValueError, match="hybrid_pod_npu_num=16 exceeds cards per node \\(8\\)"):
        validate_pod_npu_against_hardware(
            {
                C.HARDWARE_TYPE: C.HARDWARE_TYPE_800I_A2,
                C.HYBRID_POD_NPU_NUM: 16,
            }
        )


def test_validate_pod_npu_skips_unknown_hardware():
    validate_pod_npu_against_hardware(
        {
            C.HARDWARE_TYPE: "850-Atlas-8p-8",
            C.P_POD_NPU_NUM: 64,
        }
    )


def test_validate_node_selectors_includes_custom_engine_override(monkeypatch):
    seen = []

    def fake_run(cmd, **_kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="node/slave29\n", stderr="")

    monkeypatch.setattr("lib.config_validator.subprocess.run", fake_run)
    monkeypatch.setattr(
        "lib.config_validator.get_accelerator_type_from_cluster",
        lambda _hardware: C.ACCELERATOR_TYPE_910B,
    )
    monkeypatch.setattr("lib.config_validator.shutil.which", lambda _name: "kubectl")

    validate_node_selectors(
        {
            C.HARDWARE_TYPE: C.HARDWARE_TYPE_800I_A2,
            C.PREFILL_NODE_SELECTOR: {"ai-worker": "slave27"},
        }
    )

    prefill_cmd = next(cmd for cmd in seen if any("ai-worker=slave27" in part for part in cmd))
    assert "accelerator=huawei-Ascend910" in ",".join(prefill_cmd)
    assert "accelerator-type=module-910b-8" in ",".join(prefill_cmd)


def test_validate_node_selectors_rejects_unmatched_custom_override(monkeypatch):
    monkeypatch.setattr(
        "lib.config_validator.subprocess.run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        "lib.config_validator.get_accelerator_type_from_cluster",
        lambda _hardware: C.ACCELERATOR_TYPE_910B,
    )
    monkeypatch.setattr("lib.config_validator.shutil.which", lambda _name: "kubectl")

    with pytest.raises(RuntimeError, match="ai-worker"):
        validate_node_selectors(
            {
                C.HARDWARE_TYPE: C.HARDWARE_TYPE_800I_A2,
                C.PREFILL_NODE_SELECTOR: {"ai-worker": "slave27"},
            }
        )
