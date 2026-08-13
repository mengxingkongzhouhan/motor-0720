# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Tests for cluster-wide cumulative average request token length."""

import struct

from motor.common.resources.endpoint import AVG_REQUEST_TOKENS_WEIGHT, Workload
from motor.common.resources.instance import PDRole
from motor.common.resources.request_token_stats import (
    avg_request_tokens,
    record_request_token_length,
    reset_request_token_stats,
    set_avg_request_tokens,
)
from motor.coordinator.domain.workload_calculator import request_token_length
from motor.coordinator.scheduler.runtime.workload_shm.layout import (
    HEADER_FMT,
    HEADER_SIZE,
    MAGIC,
    SCHEMA_VERSION,
    WorkloadShmHeader,
    pack_header,
    unpack_header,
)


def setup_function() -> None:
    reset_request_token_stats()


def test_request_token_length_prefers_token_ids():
    class _Req:
        token_ids = [1, 2, 3]
        prompt_token_ids = [9]
        req_len = 999

    assert request_token_length(_Req()) == 3.0


def test_request_token_length_falls_back_to_byte_heuristic():
    class _Req:
        token_ids = None
        prompt_token_ids = []
        req_len = 8

    assert request_token_length(_Req()) == 2.0


def test_cumulative_average_is_global_not_per_endpoint():
    record_request_token_length(100)
    record_request_token_length(300)
    assert avg_request_tokens() == 200.0


def test_score_uses_cluster_wide_average():
    record_request_token_length(100)
    record_request_token_length(300)
    workload = Workload(active_tokens=10, active_kv_cache=20)
    expected = 10 + 20 * 0.3 + 200 * AVG_REQUEST_TOKENS_WEIGHT
    assert workload.calculate_workload_score(PDRole.ROLE_P) == expected


def test_idle_stats_keep_legacy_score():
    workload = Workload(active_tokens=10, active_kv_cache=20)
    assert workload.calculate_workload_score("prefill") == 10 + 20 * 0.3


def test_set_avg_snapshot_replaces_local_mean():
    record_request_token_length(10)
    set_avg_request_tokens(50)
    assert avg_request_tokens() == 50.0


def test_shm_header_round_trips_avg_request_tokens():
    assert struct.calcsize(HEADER_FMT) == HEADER_SIZE == 72
    packed = pack_header(
        WorkloadShmHeader(
            magic=MAGIC,
            schema_version=SCHEMA_VERSION,
            sequence=2,
            entry_count=0,
            max_entries=10,
            avg_request_tokens=123.5,
        )
    )
    header = unpack_header(memoryview(packed))
    assert header.avg_request_tokens == 123.5
