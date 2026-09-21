# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import pytest

import lib.constant as C
from lib.config_validator import validate_pod_npu_against_hardware


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
