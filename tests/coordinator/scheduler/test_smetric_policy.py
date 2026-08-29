# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 license for more details.

"""Tests for SMetricPolicy: rank by prefill_cost with overlap_credit fixed at 1."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from motor.common.resources.instance import PDRole
from motor.config.coordinator import SchedulerType
from motor.coordinator.api_client.conductor_api_client import TENANT_ID, conductor_instance_id
from motor.coordinator.scheduler.policy.factory import create
from motor.coordinator.scheduler.policy.kv_cache_affinity import KvCacheAffinityPolicy
from motor.coordinator.scheduler.policy.smetric import (
    SMetricPolicy,
    SMetricPrefillCostTracker,
    _matched_tokens,
    _prefill_cost,
    _prompt_token_ids,
)
from motor.coordinator.scheduler.runtime.scheduler_client import (
    AsyncSchedulerClient,
    SchedulerClientConfig,
)
from motor.coordinator.scheduler.runtime.zmq_protocol import (
    SchedulerResponse,
    SchedulerResponseType,
)
from motor.coordinator.domain.scheduling_pin import select_endpoint_for_instance
from tests.coordinator.scheduler.conftest import MockInstanceProvider


def _endpoint(ep_id: int) -> Mock:
    """Endpoint stub with only ``id`` so load/workload access would fail if reused from KVA."""
    ep = Mock(spec=["id"])
    ep.id = ep_id
    return ep


def _instance(instance_id: int, endpoint_ids: tuple[int, ...], role: PDRole = PDRole.ROLE_P) -> Mock:
    inst = Mock()
    inst.id = instance_id
    inst.role = role
    eps = tuple(_endpoint(eid) for eid in endpoint_ids)
    inst.get_all_endpoints.return_value = eps
    return inst


def _req_info(token_count: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        req_data={},
        token_ids=list(range(token_count)),
        smetric_debug=None,
        kv_affinity_debug=None,
    )


def _conductor_tenant(*instances: Mock, matched: dict[tuple[int, int], int]) -> dict:
    tenant: dict = {}
    for inst in instances:
        dp = {}
        for ep in inst.get_all_endpoints():
            dp[f"{ep.id}"] = matched.get((inst.id, ep.id), 0)
        tenant[conductor_instance_id(inst)] = {"DP": dp}
    return {TENANT_ID: tenant}


class TestPrefillCostFormula:
    def test_overlap_credit_is_one(self):
        assert _prefill_cost(isl=100, matched_tokens=40) == 60

    def test_matched_capped_at_isl(self):
        assert _prefill_cost(isl=10, matched_tokens=99) == 0

    def test_zero_isl(self):
        assert _prefill_cost(isl=0, matched_tokens=5) == 0

    def test_negative_match_is_clamped(self):
        assert _prefill_cost(isl=100, matched_tokens=-10) == 100

    def test_dp_blocks_format_reads_matched_tokens(self):
        assert _matched_tokens({"npu_blocks": 2, "matched_tokens": 64}) == 64

    @patch("motor.coordinator.scheduler.policy.kv_cache_affinity.TokenizerManager")
    def test_empty_cached_tokens_are_retokenized(self, mock_tokenizer_manager):
        req_info = SimpleNamespace(token_ids=[], req_data={"prompt": "hello"})
        mock_tokenizer_manager.return_value.encode.return_value = [1, 2, 3]

        assert _prompt_token_ids(req_info) == [1, 2, 3]
        assert req_info.token_ids == [1, 2, 3]


class TestSMetricPrefillCostTracker:
    def test_first_request_ranks_when_ratio_above_half(self):
        tracker = SMetricPrefillCostTracker()
        assert tracker.use_smetric_rank(req_cost=60, isl=100) is True

    def test_first_request_uses_lb_when_ratio_at_most_half(self):
        tracker = SMetricPrefillCostTracker()
        assert tracker.use_smetric_rank(req_cost=50, isl=100) is False
        assert tracker.use_smetric_rank(req_cost=10, isl=100) is False

    def test_zero_isl_uses_lb(self):
        tracker = SMetricPrefillCostTracker()
        assert tracker.use_smetric_rank(req_cost=0, isl=0) is False

    def test_cost_above_average_uses_lb_even_if_ratio_high(self):
        tracker = SMetricPrefillCostTracker()
        tracker.record(10)
        tracker.record(10)
        assert tracker.snapshot()[0] == 10
        assert tracker.use_smetric_rank(req_cost=60, isl=100) is False

    def test_cost_at_average_still_checks_ratio(self):
        tracker = SMetricPrefillCostTracker()
        tracker.record(60)
        assert tracker.use_smetric_rank(req_cost=60, isl=100) is True
        assert tracker.use_smetric_rank(req_cost=40, isl=100) is False


class TestSMetricPolicyRanking:
    @patch("motor.coordinator.scheduler.policy.smetric.ConductorApiClient.query_conductor")
    def test_lowest_prefill_cost_wins(self, mock_query):
        inst_a = _instance(1, (10, 11))
        inst_b = _instance(2, (20, 21))
        req_info = _req_info(100)
        mock_query.return_value = _conductor_tenant(
            inst_a,
            inst_b,
            matched={(1, 10): 10, (1, 11): 90, (2, 20): 50, (2, 21): 0},
        )

        ranked = SMetricPolicy.select_endpoint_candidates_from_list([inst_a, inst_b], req_info, top_k=4)

        assert ranked is not None
        costs = [(inst.id, ep.id, int(cost)) for inst, ep, cost in ranked]
        assert costs[0] == (1, 11, 10)  # isl 100 - matched 90
        assert costs[1] == (2, 20, 50)
        assert costs[2] == (1, 10, 90)
        assert costs[3] == (2, 21, 100)

    @patch("motor.coordinator.scheduler.policy.smetric.ConductorApiClient.query_conductor")
    def test_logs_all_endpoint_match_lengths(self, mock_query, caplog):
        inst_a = _instance(1, (10, 11))
        inst_b = _instance(2, (20, 21))
        req_info = _req_info(100)
        req_info.req_id = "req-match-log"
        mock_query.return_value = _conductor_tenant(
            inst_a,
            inst_b,
            matched={(1, 10): 10, (1, 11): 90, (2, 20): 50, (2, 21): 0},
        )

        with caplog.at_level(logging.INFO):
            SMetricPolicy.select_endpoint_candidates_from_list([inst_a, inst_b], req_info)

        assert "smetric: req_id=req-match-log conductor_rsp=" in caplog.text
        assert "smetric: req_id=req-match-log isl=100 endpoint_matches=[" in caplog.text
        assert "1-10:10" in caplog.text
        assert "1-11:90" in caplog.text
        assert "2-20:50" in caplog.text
        assert "2-21:0" in caplog.text

    @patch("motor.coordinator.scheduler.policy.smetric.ConductorApiClient.query_conductor")
    def test_ignores_load_when_cheaper_endpoint_is_busier(self, mock_query):
        cheap = _instance(2, (20,))
        light = _instance(1, (10,))
        req_info = _req_info(80)
        mock_query.return_value = _conductor_tenant(
            light,
            cheap,
            matched={(1, 10): 0, (2, 20): 70},
        )

        selected = SMetricPolicy.select_endpoint_from_list([light, cheap], req_info)

        assert selected is not None
        instance, endpoint = selected
        assert instance.id == 2
        assert endpoint.id == 20
        assert req_info.smetric_debug[(2, 20)] == 10
        assert req_info.smetric_debug[(1, 10)] == 80

    @patch("motor.coordinator.scheduler.policy.smetric.ConductorApiClient.query_conductor")
    def test_stashes_smetric_debug_not_kv_affinity_debug(self, mock_query):
        inst = _instance(3, (7,))
        req_info = _req_info(20)
        mock_query.return_value = _conductor_tenant(inst, matched={(3, 7): 5})

        SMetricPolicy.select_endpoint_from_list([inst], req_info)

        assert req_info.smetric_debug == {(3, 7): 15}
        assert req_info.kv_affinity_debug is None

    @patch("motor.coordinator.scheduler.policy.smetric.ConductorApiClient.query_conductor")
    def test_ranks_dp_blocks_conductor_response(self, mock_query):
        inst = _instance(3, (7, 8))
        req_info = _req_info(100)
        mock_query.return_value = {
            TENANT_ID: {
                conductor_instance_id(inst): {
                    "DP": {
                        "7": {"npu_blocks": 2, "matched_tokens": 80},
                        "8": {"npu_blocks": 1, "matched_tokens": 20},
                    }
                }
            }
        }

        ranked = SMetricPolicy.select_endpoint_candidates_from_list([inst], req_info, top_k=2)

        assert ranked is not None
        assert [(ep.id, cost) for _instance, ep, cost in ranked] == [(7, 20.0), (8, 80.0)]

    @patch(
        "motor.coordinator.scheduler.policy.kv_cache_affinity.KvCacheAffinityPolicy.select_endpoint_candidates_from_list"
    )
    @patch("motor.coordinator.scheduler.policy.kv_cache_affinity.KvCacheAffinityPolicy.select_endpoint_from_list")
    @patch("motor.coordinator.scheduler.policy.kv_cache_affinity.KvCacheAffinityPolicy._collect_load_candidates")
    @patch("motor.coordinator.scheduler.policy.kv_cache_affinity.KvCacheAffinityPolicy._stash_affinity_debug")
    @patch("motor.coordinator.scheduler.policy.smetric.ConductorApiClient.query_conductor")
    def test_does_not_call_kv_affinity_ranking(
        self,
        mock_query,
        mock_stash,
        mock_collect,
        mock_select,
        mock_candidates,
    ):
        inst = _instance(1, (10,))
        req_info = _req_info(16)
        mock_query.return_value = _conductor_tenant(inst, matched={(1, 10): 4})

        result = SMetricPolicy.select_endpoint_from_list([inst], req_info)

        assert result is not None
        mock_candidates.assert_not_called()
        mock_select.assert_not_called()
        mock_collect.assert_not_called()
        mock_stash.assert_not_called()

    @patch("motor.coordinator.scheduler.policy.smetric.ConductorApiClient.query_conductor")
    def test_no_tenant_returns_none(self, mock_query):
        inst = _instance(1, (10,))
        mock_query.return_value = {}
        assert SMetricPolicy.select_endpoint_candidates_from_list([inst], _req_info()) is None

    def test_decode_falls_back_to_load_balance(self):
        policy = SMetricPolicy(MockInstanceProvider())
        instances = [_instance(1, (10,))]
        req_info = _req_info()
        with (
            patch.object(SMetricPolicy, "select_endpoint_from_list") as mock_smetric,
            patch(
                "motor.coordinator.scheduler.policy.load_balance.LoadBalancePolicy.select_endpoint_from_list",
                return_value=(instances[0], instances[0].get_all_endpoints()[0]),
            ) as mock_lb,
        ):
            selected = policy.select_instance_and_endpoint_from_list(instances, role=PDRole.ROLE_D, req_info=req_info)
        mock_smetric.assert_not_called()
        mock_lb.assert_called_once()
        assert selected[0].id == 1


class TestSMetricFactoryAndPin:
    def test_factory_creates_smetric_policy(self):
        policy = create(SchedulerType.SMETRIC, MockInstanceProvider())
        assert isinstance(policy, SMetricPolicy)

    def test_pinned_endpoint_does_not_invoke_kv_affinity(self):
        inst = Mock()
        inst.id = 1
        ep = Mock()
        ep.id = 10
        with (
            patch(
                "motor.coordinator.domain.scheduling_pin.LoadBalancePolicy.select_endpoint_from_instance",
                return_value=ep,
            ) as mock_lb,
            patch.object(KvCacheAffinityPolicy, "select_endpoint_from_list") as mock_kva,
        ):
            got = select_endpoint_for_instance(inst, scheduler_type="smetric")
        assert got is ep
        mock_lb.assert_called_once()
        mock_kva.assert_not_called()


def _build_smetric_client() -> AsyncSchedulerClient:
    return AsyncSchedulerClient(SchedulerClientConfig(scheduler_type="smetric"))


class TestSMetricClientDispatch:
    def test_role_p_uses_smetric_not_kv_affinity(self):
        client = _build_smetric_client()
        instance = _instance(1, (10,))
        endpoint = instance.get_all_endpoints()[0]
        req_info = _req_info()
        ranked = [(instance, endpoint, 3.0)]

        with (
            patch.object(
                SMetricPolicy,
                "select_endpoint_candidates_from_list",
                return_value=ranked,
            ) as mock_smetric,
            patch.object(
                KvCacheAffinityPolicy,
                "select_endpoint_candidates_from_list",
            ) as mock_kva,
            patch.object(client, "_select_endpoint_candidates_by_load_balance") as mock_lb,
        ):
            candidates, candidate_policy = client._select_endpoint_candidates_from_list_with_policy(
                [instance], PDRole.ROLE_P, req_info, top_k=1
            )

        assert candidates == ranked
        assert candidate_policy == "smetric"
        mock_smetric.assert_called_once()
        mock_kva.assert_not_called()
        mock_lb.assert_not_called()

    def test_role_d_skips_smetric(self):
        client = _build_smetric_client()
        instance = _instance(1, (10,))
        endpoint = instance.get_all_endpoints()[0]
        req_info = _req_info()
        lb_candidates = [(instance, endpoint, 0.42)]

        with (
            patch.object(SMetricPolicy, "select_endpoint_candidates_from_list") as mock_smetric,
            patch.object(
                client,
                "_select_endpoint_candidates_by_load_balance",
                return_value=lb_candidates,
            ) as mock_lb,
        ):
            candidates, candidate_policy = client._select_endpoint_candidates_from_list_with_policy(
                [instance], PDRole.ROLE_D, req_info, top_k=1
            )

        assert candidates == lb_candidates
        assert candidate_policy == "load_balance"
        mock_smetric.assert_not_called()
        mock_lb.assert_called_once_with([instance], PDRole.ROLE_D, 1)

    def test_smetric_none_falls_back_to_load_balance(self):
        client = _build_smetric_client()
        instance = _instance(1, (10,))
        endpoint = instance.get_all_endpoints()[0]
        req_info = _req_info()
        lb_candidates = [(instance, endpoint, 0.42)]

        with (
            patch.object(
                SMetricPolicy,
                "select_endpoint_candidates_from_list",
                return_value=None,
            ) as mock_smetric,
            patch.object(
                client,
                "_select_endpoint_candidates_by_load_balance",
                return_value=lb_candidates,
            ) as mock_lb,
        ):
            candidates, candidate_policy = client._select_endpoint_candidates_from_list_with_policy(
                [instance], PDRole.ROLE_P, req_info, top_k=1
            )

        assert candidates == lb_candidates
        assert candidate_policy == "load_balance"
        mock_smetric.assert_called_once()
        mock_lb.assert_called_once_with([instance], PDRole.ROLE_P, 1)

    def test_candidate_payload_reads_smetric_debug_not_affinity_tuple(self):
        from motor.coordinator.scheduler.runtime.scheduler_client import _candidate_endpoint_payload

        payload = _candidate_endpoint_payload(
            1,
            10,
            affinity_debug={(1, 10): (8, 1.0, 99)},
            smetric_debug={(1, 10): 12},
        )
        assert payload == {"instance_id": 1, "endpoint_id": 10, "prefill_cost": 12}

    @pytest.mark.asyncio
    async def test_select_and_allocate_forwards_all_costs_without_kva_scales(self):
        client = _build_smetric_client()
        inst = Mock()
        inst.id = 1
        ep = Mock()
        ep.id = 10
        req_info = SimpleNamespace(
            req_id="req-smetric",
            req_data={},
            req_len=10,
            token_ids=[1, 2, 3],
            smetric_debug={(1, 10): 3, (2, 20): 1},
            kv_affinity_debug=None,
        )
        captured: dict = {}

        async def fake_send(request):
            captured["data"] = request.data
            return SchedulerResponse(
                request_id=request.request_id,
                response_type=SchedulerResponseType.SUCCESS,
                data={"instance": None, "endpoint": None},
            )

        with patch.object(
            client,
            "_select_endpoint_candidates_with_policy",
            new=AsyncMock(return_value=([(inst, ep, 3.0)], "smetric")),
        ):
            client._transport.send_request = fake_send
            await client.select_and_allocate(PDRole.ROLE_P, req_info)

        data = captured["data"]
        assert data["candidate_policy"] == "smetric"
        assert "prefill_load_scale" not in data
        assert "load_weight" not in data
        costs = {(c["instance_id"], c["endpoint_id"]): c["prefill_cost"] for c in data["candidates"]}
        assert costs == {(1, 10): 3, (2, 20): 1}
        assert data["isl"] == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scheduler_type", ["smetric", "load_balance"])
    async def test_pinned_allocation_does_not_request_global_reranking(self, scheduler_type):
        """A scheduler policy must not move an allocation away from its pinned instance."""
        client = AsyncSchedulerClient(SchedulerClientConfig(scheduler_type=scheduler_type))
        inst = Mock()
        inst.id = 7
        ep = Mock()
        ep.id = 70
        req_info = SimpleNamespace(
            req_id="req-pinned",
            req_data={},
            req_len=10,
            token_ids=[1, 2, 3],
            smetric_debug=None,
            kv_affinity_debug=None,
        )
        captured: dict = {}

        async def fake_send(request):
            captured["data"] = request.data
            return SchedulerResponse(
                request_id=request.request_id,
                response_type=SchedulerResponseType.SUCCESS,
                data={"instance": None, "endpoint": None},
            )

        client.get_available_instances = AsyncMock(return_value={inst.id: inst})
        client._transport.send_request = fake_send
        with (
            patch(
                "motor.coordinator.scheduler.runtime.scheduler_client.resolve_pinned_instance",
                return_value=inst,
            ),
            patch(
                "motor.coordinator.scheduler.runtime.scheduler_client.select_endpoint_for_instance",
                return_value=ep,
            ),
        ):
            await client.select_and_allocate(PDRole.ROLE_P, req_info, target_instance_id=inst.id)

        assert captured["data"]["instance_id"] == inst.id
        assert captured["data"]["endpoint_id"] == ep.id
        assert captured["data"]["candidate_policy"] == "round_robin"
