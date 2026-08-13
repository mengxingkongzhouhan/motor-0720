# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for Workload scoring, including per-endpoint average request tokens."""

import pytest

from motor.common.resources.endpoint import AVG_REQUEST_TOKENS_WEIGHT, Workload
from motor.common.resources.instance import PDRole


def test_avg_request_tokens_zero_when_idle():
    assert Workload().avg_request_tokens == 0.0
    assert Workload(active_tokens=100, num_requests=0).avg_request_tokens == 0.0


def test_avg_request_tokens_is_tokens_per_request():
    workload = Workload(active_tokens=1000, num_requests=4)
    assert workload.avg_request_tokens == 250.0


def test_prefill_score_includes_avg_request_tokens():
    workload = Workload(active_tokens=1000, active_kv_cache=200, num_requests=4)
    expected = 1000 + 200 * 0.3 + 250 * AVG_REQUEST_TOKENS_WEIGHT
    assert workload.calculate_workload_score(PDRole.ROLE_P) == expected


def test_decode_score_includes_avg_request_tokens():
    workload = Workload(active_tokens=1000, num_requests=4)
    expected = 1000 + 250 * AVG_REQUEST_TOKENS_WEIGHT
    assert workload.calculate_workload_score(PDRole.ROLE_D) == expected


def test_idle_score_matches_legacy_formula():
    workload = Workload(active_tokens=100, active_kv_cache=50)
    assert workload.calculate_workload_score("prefill") == 100 + 50 * 0.3
    assert workload.calculate_workload_score("decode") == 100


def test_longer_average_ranks_higher_than_many_short_requests():
    long_reqs = Workload(active_tokens=1000, num_requests=1)
    short_reqs = Workload(active_tokens=1000, num_requests=10)
    assert long_reqs.calculate_workload_score(PDRole.ROLE_D) > short_reqs.calculate_workload_score(PDRole.ROLE_D)


def test_iadd_accumulates_num_requests():
    total = Workload()
    total += Workload(active_tokens=10, num_requests=1)
    total += Workload(active_tokens=30, num_requests=1)
    assert total.active_tokens == 40
    assert total.num_requests == 2
    assert total.avg_request_tokens == 20


def test_calculate_workload_score_requires_role():
    with pytest.raises(ValueError, match="role is required"):
        Workload().calculate_workload_score(None)
