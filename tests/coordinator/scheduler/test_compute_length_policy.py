# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 license for more details.

"""Tests for ComputeLengthPolicy and its scheduler-client wiring."""

from unittest.mock import Mock, patch

from motor.common.resources.endpoint import Endpoint, Workload
from motor.common.resources.instance import PDRole
from motor.coordinator.api_client.conductor_api_client import TENANT_ID
from motor.coordinator.scheduler.policy.compute_length import (
    ComputeLengthPolicy,
    endpoint_compute_load,
    instance_compute_load,
)
from motor.coordinator.scheduler.runtime.scheduler_client import (
    AsyncSchedulerClient,
    SchedulerClientConfig,
)
from motor.coordinator.scheduler.runtime.zmq_protocol import CANDIDATE_POLICY_COMPUTE_LENGTH
from tests.coordinator.scheduler.conftest import (
    MockInstanceProvider,
    create_mock_endpoint,
    create_mock_instance,
    create_mock_workload,
)


def _make_endpoint(ep_id: int, active_tokens: float = 0.0) -> Endpoint:
    return Endpoint(
        id=ep_id,
        ip="127.0.0.1",
        business_port="8000",
        workload=Workload(active_tokens=active_tokens),
    )


def _make_instance(instance_id, endpoints, gathered_tokens: float, role=PDRole.ROLE_P):
    inst = Mock()
    inst.id = instance_id
    inst.role = role
    inst.endpoints = {"group": {ep.id: ep for ep in endpoints}}
    inst.get_all_endpoints.return_value = tuple(endpoints)
    inst.gathered_workload = Workload(active_tokens=gathered_tokens)
    return inst


class TestComputeLengthSelection:
    def test_picks_min_instance_then_min_dp(self):
        """Instance 2 is lighter (20 vs 105); then DP 20 (10) beats DP 21 (10) by first-min, wait.

        Inst1 gathered=105 (eps 5+100), inst2 gathered=20 (eps 10+11). Hierarchical
        must pick inst2 then the lighter DP (10), not the globally lightest endpoint (5).
        """
        ep10 = _make_endpoint(10, 5.0)
        ep11 = _make_endpoint(11, 100.0)
        ep20 = _make_endpoint(20, 10.0)
        ep21 = _make_endpoint(21, 11.0)
        inst1 = _make_instance(1, [ep10, ep11], 105.0)
        inst2 = _make_instance(2, [ep20, ep21], 20.0)

        result = ComputeLengthPolicy.select_instance_then_endpoint([inst1, inst2], PDRole.ROLE_P)

        assert result is not None
        instance, endpoint, score = result
        assert instance.id == 2
        assert endpoint.id == 20
        assert score == 20.0

    def test_min_dp_inside_selected_instance(self):
        ep_heavy = _make_endpoint(0, 40.0)
        ep_light = _make_endpoint(1, 5.0)
        inst = _make_instance(7, [ep_heavy, ep_light], 45.0)

        result = ComputeLengthPolicy.select_instance_then_endpoint([inst], PDRole.ROLE_P)

        assert result is not None
        assert result[1].id == 1

    def test_tie_breaks_first_instance(self):
        ep_a = _make_endpoint(0, 3.0)
        ep_b = _make_endpoint(1, 3.0)
        inst_a = _make_instance(1, [ep_a], 3.0)
        inst_b = _make_instance(2, [ep_b], 3.0)

        result = ComputeLengthPolicy.select_instance_then_endpoint([inst_a, inst_b], PDRole.ROLE_P)

        assert result is not None
        assert result[0].id == 1

    def test_skips_blocked_instance(self):
        ep_a = _make_endpoint(0, 1.0)
        ep_b = _make_endpoint(1, 50.0)
        inst_a = _make_instance(1, [ep_a], 1.0)
        inst_b = _make_instance(2, [ep_b], 50.0)

        result = ComputeLengthPolicy.select_instance_then_endpoint(
            [inst_a, inst_b], PDRole.ROLE_P, is_blocked=lambda iid: iid == 1
        )

        assert result is not None
        assert result[0].id == 2

    def test_skips_excluded_pair_and_empty_instance(self):
        ep_a = _make_endpoint(0, 1.0)
        ep_b0 = _make_endpoint(0, 8.0)
        ep_b1 = _make_endpoint(1, 9.0)
        inst_a = _make_instance(1, [ep_a], 1.0)
        inst_b = _make_instance(2, [ep_b0, ep_b1], 17.0)

        result = ComputeLengthPolicy.select_instance_then_endpoint(
            [inst_a, inst_b],
            PDRole.ROLE_P,
            excluded_pairs={(1, 0)},
        )

        assert result is not None
        assert result[0].id == 2
        assert result[1].id == 0

    def test_empty_returns_none(self):
        assert ComputeLengthPolicy.select_instance_then_endpoint([], PDRole.ROLE_P) is None

    def test_instance_compute_load_falls_back_to_dp_sum(self):
        ep0 = _make_endpoint(0, 4.0)
        ep1 = _make_endpoint(1, 6.0)
        inst = _make_instance(1, [ep0, ep1], 0.0)
        inst.gathered_workload = None

        assert instance_compute_load(inst, PDRole.ROLE_P) == 10.0
        assert endpoint_compute_load(inst, ep1, PDRole.ROLE_P) == 6.0


class TestComputeLengthMaxMatch:
    @patch("motor.coordinator.scheduler.policy.compute_length.ConductorApiClient.query_conductor")
    @patch("motor.coordinator.scheduler.policy.kv_cache_affinity.TokenizerManager")
    def test_uses_global_max_match_not_per_dp(self, mock_tokenizer_manager, mock_query_conductor):
        """Heavier instance with a longer prefix must not win; stash uses the global max."""
        ep_heavy_hit = _make_endpoint(0, 80.0)
        ep_light = _make_endpoint(1, 10.0)
        inst_heavy = _make_instance("heavy", [ep_heavy_hit], 80.0)
        inst_light = _make_instance("light", [ep_light], 10.0)
        req_info = Mock()
        req_info.req_data = {"prompt": "hello"}
        req_info.token_ids = None
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(200))
        mock_tokenizer_manager.return_value = mock_tokenizer
        mock_query_conductor.return_value = {
            TENANT_ID: {
                "vllm-prefill-heavy": {"DP": {"0": 180}},
                "vllm-prefill-light": {"DP": {"1": 20}},
            }
        }

        ranked = ComputeLengthPolicy.select_endpoint_candidates_from_list(
            [inst_heavy, inst_light], req_info, role=PDRole.ROLE_P
        )

        assert ranked
        assert ranked[0][0].id == "light"
        assert ranked[0][1].id == 1
        assert req_info.max_matched_tokens == 180
        mock_query_conductor.assert_called_once()

    @patch("motor.coordinator.scheduler.policy.compute_length.ConductorApiClient.query_conductor")
    @patch("motor.coordinator.scheduler.policy.kv_cache_affinity.KvCacheAffinityPolicy._conductor_block_size")
    @patch("motor.coordinator.scheduler.policy.kv_cache_affinity.TokenizerManager")
    def test_short_prompt_skips_conductor(self, mock_tokenizer_manager, mock_block_size, mock_query):
        mock_block_size.return_value = 16
        ep = _make_endpoint(0, 1.0)
        inst = _make_instance(1, [ep], 1.0)
        req_info = Mock()
        req_info.req_data = {"prompt": "hi"}
        req_info.token_ids = None
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer_manager.return_value = mock_tokenizer

        matched = ComputeLengthPolicy.resolve_max_matched_tokens([inst], req_info)

        assert matched == 0
        assert req_info.max_matched_tokens == 0
        mock_query.assert_not_called()

    @patch("motor.coordinator.scheduler.policy.compute_length.ConductorApiClient.query_conductor")
    @patch("motor.coordinator.scheduler.policy.kv_cache_affinity.TokenizerManager")
    def test_no_tenant_stashes_zero_and_still_selects(self, mock_tokenizer_manager, mock_query_conductor):
        ep = _make_endpoint(0, 4.0)
        inst = _make_instance(3, [ep], 4.0)
        req_info = Mock()
        req_info.req_data = {"prompt": "hello"}
        req_info.token_ids = None
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(32))
        mock_tokenizer_manager.return_value = mock_tokenizer
        mock_query_conductor.return_value = {}

        result = ComputeLengthPolicy.select_endpoint_from_list([inst], req_info, role=PDRole.ROLE_P)

        assert result is not None
        assert result[1].id == 0
        assert req_info.max_matched_tokens == 0


class TestComputeLengthPolicyInstanceMethods:
    def test_select_instance_and_endpoint_via_provider(self):
        ep1 = create_mock_endpoint(1, workload=create_mock_workload(8.0))
        ep2 = create_mock_endpoint(2, workload=create_mock_workload(2.0))
        inst = create_mock_instance(
            instance_id=1,
            endpoints={"g": {1: ep1, 2: ep2}},
            gathered_workload=create_mock_workload(10.0),
        )
        policy = ComputeLengthPolicy(MockInstanceProvider({PDRole.ROLE_P: {1: inst}}))

        result = policy.select_instance_and_endpoint(PDRole.ROLE_P)

        assert result is not None
        assert result[0] is inst
        assert result[1] is ep2

    def test_select_instance_no_instances(self):
        policy = ComputeLengthPolicy(MockInstanceProvider())
        assert policy.select_instance_and_endpoint(PDRole.ROLE_P) is None


class TestComputeLengthClientWiring:
    def test_role_p_resolves_match_and_selects_hierarchically(self):
        client = AsyncSchedulerClient(SchedulerClientConfig(scheduler_type="compute_length"))
        instance = Mock()
        endpoint = Mock()
        req_info = Mock()
        selected = (instance, endpoint, 4.0)

        with (
            patch(
                "motor.coordinator.scheduler.runtime.scheduler_client."
                "ComputeLengthPolicy.resolve_max_matched_tokens"
            ) as mock_resolve,
            patch(
                "motor.coordinator.scheduler.runtime.scheduler_client."
                "ComputeLengthPolicy.select_instance_then_endpoint",
                return_value=selected,
            ) as mock_select,
            patch.object(client, "_select_endpoint_candidates_by_load_balance") as mock_lb,
        ):
            candidates, policy = client._select_endpoint_candidates_from_list_with_policy(
                [instance], PDRole.ROLE_P, req_info, top_k=1
            )

        assert candidates == [selected]
        assert policy == CANDIDATE_POLICY_COMPUTE_LENGTH
        mock_resolve.assert_called_once()
        mock_select.assert_called_once()
        mock_lb.assert_not_called()

    def test_role_d_skips_conductor_but_stays_hierarchical(self):
        client = AsyncSchedulerClient(SchedulerClientConfig(scheduler_type="compute_length"))
        instance = Mock()
        endpoint = Mock()
        req_info = Mock()
        selected = (instance, endpoint, 1.0)

        with (
            patch(
                "motor.coordinator.scheduler.runtime.scheduler_client."
                "ComputeLengthPolicy.resolve_max_matched_tokens"
            ) as mock_resolve,
            patch(
                "motor.coordinator.scheduler.runtime.scheduler_client."
                "ComputeLengthPolicy.select_instance_then_endpoint",
                return_value=selected,
            ),
        ):
            candidates, policy = client._select_endpoint_candidates_from_list_with_policy(
                [instance], PDRole.ROLE_D, req_info, top_k=1
            )

        assert candidates == [selected]
        assert policy == CANDIDATE_POLICY_COMPUTE_LENGTH
        mock_resolve.assert_not_called()

    def test_committed_workload_uses_global_max_match(self):
        client = AsyncSchedulerClient(SchedulerClientConfig(scheduler_type="compute_length"))
        instance = Mock()
        instance.id = 2
        endpoint = Mock()
        endpoint.id = 1
        demand = Workload(active_tokens=200.0)

        committed = client._committed_workload_for(
            PDRole.ROLE_P,
            CANDIDATE_POLICY_COMPUTE_LENGTH,
            instance,
            endpoint,
            demand,
            {(1, 0): 20.0, (2, 1): 180.0},
            isl=200.0,
        )

        assert committed.active_tokens == 20.0

    def test_committed_workload_falls_back_to_demand_without_isl(self):
        client = AsyncSchedulerClient(SchedulerClientConfig(scheduler_type="compute_length"))
        demand = Workload(active_tokens=42.0)

        committed = client._committed_workload_for(
            PDRole.ROLE_P,
            CANDIDATE_POLICY_COMPUTE_LENGTH,
            Mock(),
            Mock(),
            demand,
            {},
            isl=0.0,
        )

        assert committed is demand
