# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Tests for SMetricGatedPolicy: prefill_cost order, then first endpoint under both ledger averages."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from motor.common.resources.endpoint import Endpoint, EndpointStatus, Workload, WorkloadAction
from motor.common.resources.http_msg_spec import EventType
from motor.common.resources.instance import Instance, InsStatus, PDRole, ParallelConfig
from motor.config.coordinator import CoordinatorConfig, SchedulerType, SMetricGatedConfig
from motor.coordinator.api_client.conductor_api_client import TENANT_ID, conductor_instance_id
from motor.coordinator.domain import ScheduledResource
from motor.coordinator.domain.instance_manager import InstanceManager
from motor.coordinator.domain.workload_calculator import allocated_cpu_hit_blocks, allocated_prefill_cost
from motor.coordinator.scheduler import allocate_arbitration
from motor.coordinator.scheduler.allocate_arbitration import ArbitrationContext
from motor.coordinator.scheduler.policy.factory import create
from motor.coordinator.scheduler.policy.load_balance import LoadBalancePolicy
from motor.coordinator.scheduler.policy.smetric_gated import (
    PICK_ACTIVE_GATE,
    PICK_BOTH_GATES,
    PICK_MIN_LEDGER_PREFILL,
    GatedCandidate,
    SMetricGatedPolicy,
    _cpu_hit_blocks,
    pick_gated,
    sort_candidates,
)
from motor.coordinator.scheduler.runtime.scheduler_client import (
    AsyncSchedulerClient,
    SchedulerClientConfig,
)
from motor.coordinator.scheduler.runtime.zmq_protocol import (
    CANDIDATE_POLICY_SMETRIC_GATED,
    KNOWN_CANDIDATE_POLICIES,
)
from motor.coordinator.scheduler.scheduler import Scheduler
from motor.coordinator.domain.scheduling_pin import select_endpoint_for_instance
from motor.coordinator.router.workload import WorkloadActionHandler
from tests.coordinator.scheduler.conftest import MockInstanceProvider


def _endpoint(
    ep_id: int,
    active_tokens: float = 0.0,
    cpu_hit_blocks: float = 0.0,
    prefill_cost: float = 0.0,
) -> Endpoint:
    return Endpoint(
        id=ep_id,
        ip="10.0.0.1",
        business_port=f"80{ep_id}",
        status=EndpointStatus.NORMAL,
        workload=Workload(active_tokens=active_tokens, cpu_hit_blocks=cpu_hit_blocks, prefill_cost=prefill_cost),
    )


def _instance(instance_id: int, endpoints: list[Endpoint], role: PDRole = PDRole.ROLE_P) -> Instance:
    inst = Instance(
        job_name=f"{role.value}-{instance_id}",
        model_name="test_model",
        id=instance_id,
        role=role,
        status=InsStatus.ACTIVE,
        parallel_config=ParallelConfig(dp_size=max(1, len(endpoints))),
    )
    inst.add_endpoints(f"pod-{instance_id}", {idx: ep for idx, ep in enumerate(endpoints)})
    return inst


def _cand(
    ep_id: int,
    ledger_prefill: float,
    active: float = 0.0,
    cpu: float = 0.0,
    req_cost: float = 0.0,
    req_cpu: float = 0.0,
) -> GatedCandidate:
    """Standalone candidate: endpoint ledger (prefill, active, cpu) + this request's stamp values."""
    inst = _instance(ep_id, [_endpoint(ep_id, active_tokens=active, cpu_hit_blocks=cpu, prefill_cost=ledger_prefill)])
    return GatedCandidate(inst, inst.get_all_endpoints()[0], req_cost, req_cpu)


def _req_info(token_count: int = 100, req_id: str = "req-gated") -> SimpleNamespace:
    return SimpleNamespace(
        req_id=req_id,
        req_data={},
        req_len=token_count * 4,
        token_ids=list(range(token_count)),
        smetric_debug=None,
        smetric_gated_debug=None,
        kv_affinity_debug=None,
    )


def _conductor_tenant(*instances: Instance, dp: dict[tuple[int, int], dict]) -> dict:
    tenant: dict = {}
    for inst in instances:
        tenant[conductor_instance_id(inst)] = {
            "DP": {f"{ep.id}": dp.get((inst.id, ep.id), 0) for ep in inst.get_all_endpoints()}
        }
    return {TENANT_ID: tenant}


def _arbitration_context(
    instance_manager: InstanceManager,
    *,
    blocked: tuple[int, ...] = (),
    active_factor: float = 1.0,
    cpu_factor: float = 1.0,
) -> ArbitrationContext:
    blocked_set = set(blocked)
    return ArbitrationContext(
        get_available_instances=instance_manager.get_available_instances,
        is_instance_circuit_open=lambda instance_id: instance_id in blocked_set,
        endpoint_instance_score_weight=0.0,
        is_load_balance_scheduler=False,
        smetric_gated_active_factor=active_factor,
        smetric_gated_cpu_factor=cpu_factor,
    )


# ---------------------------------------------------------------------------
# Ledger field
# ---------------------------------------------------------------------------


class TestCpuHitLedger:
    def test_workload_iadd_accumulates_cpu_hits(self):
        w = Workload(active_tokens=1, prefill_cost=2, cpu_hit_blocks=3)
        w += Workload(active_tokens=1, prefill_cost=1, cpu_hit_blocks=4)
        assert (w.active_tokens, w.prefill_cost, w.cpu_hit_blocks) == (2, 3, 7)

    def test_default_is_zero(self):
        assert Workload().cpu_hit_blocks == 0
        assert Workload().prefill_cost == 0

    @pytest.mark.asyncio
    async def test_instance_manager_floors_negative_cpu_hits(self):
        config = CoordinatorConfig()
        im = InstanceManager(config)
        inst = _instance(1, [_endpoint(10)])
        await im.refresh_instances(EventType.ADD, [inst])
        await im.update_instance_workload(1, 10, Workload(cpu_hit_blocks=5))
        await im.update_instance_workload(1, 10, Workload(cpu_hit_blocks=-9))
        _, ledger = await im.get_endpoint_workload(1, 10)
        assert ledger.cpu_hit_blocks == 0

    @pytest.mark.asyncio
    async def test_router_release_negates_cpu_hits(self):
        request_mgr = Mock()
        request_mgr.get_req_workload = AsyncMock(
            return_value=Workload(active_tokens=10, prefill_cost=4, cpu_hit_blocks=3)
        )
        handler = WorkloadActionHandler(request_mgr)
        inst = _instance(1, [_endpoint(10)])
        resource = ScheduledResource(instance=inst, endpoint=inst.get_all_endpoints()[0])
        change, role = await handler.compute_and_update(resource, "req", WorkloadAction.RELEASE_TOKENS, _req_info())
        assert role == PDRole.ROLE_P
        assert (change.active_tokens, change.prefill_cost, change.cpu_hit_blocks) == (-10, -4, -3)


class TestConductorParsing:
    def test_cpu_blocks_from_dp_blocks(self):
        assert _cpu_hit_blocks({"npu_blocks": 2, "cpu_blocks": 5, "matched_tokens": 64}) == 5

    def test_legacy_int_match_has_no_cpu_hits(self):
        assert _cpu_hit_blocks(40) == 0

    def test_missing_or_invalid_cpu_blocks(self):
        assert _cpu_hit_blocks({"matched_tokens": 8}) == 0
        assert _cpu_hit_blocks({"cpu_blocks": "x"}) == 0
        assert _cpu_hit_blocks({"cpu_blocks": -3}) == 0


# ---------------------------------------------------------------------------
# Ordering and gating
# ---------------------------------------------------------------------------


class TestPickGated:
    def test_sorted_by_ledger_prefill_then_ids(self):
        ranked = sort_candidates(
            [_cand(3, ledger_prefill=50), _cand(1, ledger_prefill=10), _cand(2, ledger_prefill=10)]
        )
        assert [c.endpoint.id for c in ranked] == [1, 2, 3]

    def test_request_cost_does_not_affect_order(self):
        ranked = sort_candidates([_cand(1, ledger_prefill=40, req_cost=90), _cand(2, ledger_prefill=80, req_cost=1)])
        assert [c.endpoint.id for c in ranked] == [1, 2]

    def test_first_under_both_averages_wins(self):
        ranked = sort_candidates(
            [
                _cand(1, ledger_prefill=10, active=90, cpu=0),
                _cand(2, ledger_prefill=20, active=10, cpu=90),
                _cand(3, ledger_prefill=30, active=20, cpu=10),
                _cand(4, ledger_prefill=40, active=0, cpu=0),
            ]
        )
        chosen, reason, mean_active, mean_cpu = pick_gated(ranked)
        assert chosen.endpoint.id == 3
        assert reason == PICK_BOTH_GATES
        assert mean_active == 30 and mean_cpu == 25

    def test_gate_is_inclusive(self):
        ranked = sort_candidates(
            [
                _cand(1, ledger_prefill=1, active=20, cpu=20),
                _cand(2, ledger_prefill=2, active=10, cpu=10),
                _cand(3, ledger_prefill=3, active=30, cpu=30),
            ]
        )
        chosen, reason, _a, _c = pick_gated(ranked)
        assert chosen.endpoint.id == 1 and reason == PICK_BOTH_GATES

    def test_just_above_mean_is_rejected(self):
        ranked = sort_candidates(
            [
                _cand(1, ledger_prefill=1, active=21, cpu=20),
                _cand(2, ledger_prefill=2, active=9, cpu=10),
                _cand(3, ledger_prefill=3, active=30, cpu=30),
            ]
        )
        chosen, reason, _a, _c = pick_gated(ranked)
        assert chosen.endpoint.id == 2 and reason == PICK_BOTH_GATES

    def test_idle_cluster_passes_both_gates(self):
        ranked = sort_candidates([_cand(1, ledger_prefill=0), _cand(2, ledger_prefill=0), _cand(3, ledger_prefill=0)])
        chosen, reason, active_threshold, cpu_threshold = pick_gated(ranked)
        assert chosen.endpoint.id == 1 and reason == PICK_BOTH_GATES
        assert (active_threshold, cpu_threshold) == (0.0, 0.0)

    def test_fallback_active_gate_only(self):
        ranked = sort_candidates(
            [_cand(1, ledger_prefill=5, active=10, cpu=30), _cand(2, ledger_prefill=6, active=30, cpu=10)]
        )
        chosen, reason, _a, _c = pick_gated(ranked)
        assert chosen.endpoint.id == 1 and reason == PICK_ACTIVE_GATE

    def test_all_equal_load_passes_gates_and_takes_lowest_ledger_prefill(self):
        ranked = sort_candidates(
            [_cand(2, ledger_prefill=7, active=5, cpu=5), _cand(1, ledger_prefill=9, active=5, cpu=5)]
        )
        chosen, reason, _a, _c = pick_gated(ranked)
        assert chosen.endpoint.id == 2 and reason == PICK_BOTH_GATES

    def test_idle_cluster_takes_lowest_ledger_prefill(self):
        ranked = sort_candidates([_cand(1, ledger_prefill=50), _cand(2, ledger_prefill=5), _cand(3, ledger_prefill=20)])
        chosen, reason, _a, _c = pick_gated(ranked)
        assert chosen.endpoint.id == 2 and reason == PICK_BOTH_GATES

    def test_fallback_min_ledger_prefill_when_every_gate_fails(self):
        ranked = sort_candidates(
            [_cand(2, ledger_prefill=7, active=5, cpu=5), _cand(1, ledger_prefill=9, active=5, cpu=5)]
        )
        chosen, reason, _a, _c = pick_gated(ranked, 0.0, 0.0)
        assert chosen.endpoint.id == 2 and reason == PICK_MIN_LEDGER_PREFILL

    def test_empty(self):
        assert pick_gated([]) is None

    def test_thresholds_are_mean_times_factor(self):
        ranked = sort_candidates(
            [_cand(1, ledger_prefill=1, active=10, cpu=10), _cand(2, ledger_prefill=2, active=30, cpu=30)]
        )
        _c, _r, active_threshold, cpu_threshold = pick_gated(ranked, 1.5, 0.5)
        assert active_threshold == 20 * 1.5
        assert cpu_threshold == 20 * 0.5

    def test_factor_above_one_loosens_gate(self):
        ranked = sort_candidates(
            [
                _cand(1, ledger_prefill=1, active=22, cpu=0),
                _cand(2, ledger_prefill=2, active=8, cpu=0),
                _cand(3, ledger_prefill=3, active=30, cpu=0),
            ]
        )
        plain, reason_plain, _a, _c = pick_gated(ranked)
        loose, reason_loose, _a2, _c2 = pick_gated(ranked, active_tokens_mean_factor=1.2)
        assert plain.endpoint.id == 2 and reason_plain == PICK_BOTH_GATES
        assert loose.endpoint.id == 1 and reason_loose == PICK_BOTH_GATES

    def test_factor_below_one_tightens_gate(self):
        ranked = sort_candidates(
            [
                _cand(1, ledger_prefill=1, active=15, cpu=5),
                _cand(2, ledger_prefill=2, active=5, cpu=5),
                _cand(3, ledger_prefill=3, active=40, cpu=20),
            ]
        )
        plain, _r, _a, _c = pick_gated(ranked)
        tight, _r2, _a2, _c2 = pick_gated(ranked, active_tokens_mean_factor=0.5)
        assert plain.endpoint.id == 1
        assert tight.endpoint.id == 2

    def test_cpu_factor_only_affects_cpu_gate(self):
        ranked = sort_candidates(
            [
                _cand(1, ledger_prefill=1, active=5, cpu=15),
                _cand(2, ledger_prefill=2, active=5, cpu=5),
                _cand(3, ledger_prefill=3, active=50, cpu=10),
            ]
        )
        plain, reason_plain, _a, _c = pick_gated(ranked)
        loose, reason_loose, _a2, _c2 = pick_gated(ranked, cpu_hit_blocks_mean_factor=2.0)
        assert plain.endpoint.id == 2 and reason_plain == PICK_BOTH_GATES
        assert loose.endpoint.id == 1 and reason_loose == PICK_BOTH_GATES

    def test_negative_or_none_factor_normalized(self):
        ranked = sort_candidates(
            [_cand(1, ledger_prefill=1, active=10, cpu=10), _cand(2, ledger_prefill=2, active=30, cpu=30)]
        )
        _c, _r, active_threshold, cpu_threshold = pick_gated(ranked, -3.0, None)
        assert active_threshold == 0.0
        assert cpu_threshold == 20.0


# ---------------------------------------------------------------------------
# Policy (conductor scoring + registration)
# ---------------------------------------------------------------------------


class TestPolicy:
    @patch("motor.coordinator.scheduler.policy.smetric_gated.ConductorApiClient.query_conductor")
    def test_score_endpoints_reads_cost_and_cpu_hits_and_orders_by_ledger(self, mock_query):
        inst_a = _instance(1, [_endpoint(10, prefill_cost=300), _endpoint(11, prefill_cost=100)])
        inst_b = _instance(2, [_endpoint(20, prefill_cost=200)])
        req_info = _req_info(100)
        mock_query.return_value = _conductor_tenant(
            inst_a,
            inst_b,
            dp={
                (1, 10): {"npu_blocks": 1, "cpu_blocks": 4, "matched_tokens": 90},
                (1, 11): {"npu_blocks": 0, "cpu_blocks": 0, "matched_tokens": 0},
                (2, 20): 50,
            },
        )

        ranked = SMetricGatedPolicy.score_endpoints([inst_a, inst_b], req_info)

        assert [(c.endpoint.id, c.ledger_prefill_cost, c.prefill_cost, c.cpu_hit_blocks) for c in ranked] == [
            (11, 100.0, 100.0, 0.0),
            (20, 200.0, 50.0, 0.0),
            (10, 300.0, 10.0, 4.0),
        ]
        assert req_info.smetric_gated_debug == {(1, 11): (100.0, 0.0), (2, 20): (50.0, 0.0), (1, 10): (10.0, 4.0)}
        assert allocated_prefill_cost(req_info, 1, 10) == 10.0
        assert allocated_cpu_hit_blocks(req_info, 1, 10) == 4.0
        assert allocated_cpu_hit_blocks(req_info, 9, 9) == 0.0

    @patch("motor.coordinator.scheduler.policy.smetric_gated.ConductorApiClient.query_conductor")
    def test_worker_proposal_puts_gated_pick_first(self, mock_query):
        inst_a = _instance(1, [_endpoint(10, active_tokens=500, prefill_cost=10)])
        inst_b = _instance(2, [_endpoint(20, active_tokens=10, prefill_cost=80)])
        req_info = _req_info(100)
        mock_query.return_value = _conductor_tenant(inst_a, inst_b, dp={(1, 10): 90, (2, 20): 20})

        ranked = SMetricGatedPolicy.select_endpoint_candidates_from_list([inst_a, inst_b], req_info, top_k=2)

        assert [(ep.id, score) for _i, ep, score in ranked] == [(20, 80.0), (10, 10.0)]

    @patch("motor.coordinator.scheduler.policy.smetric_gated.ConductorApiClient.query_conductor")
    def test_no_tenant_returns_none(self, mock_query):
        mock_query.return_value = {}
        assert SMetricGatedPolicy.score_endpoints([_instance(1, [_endpoint(10)])], _req_info()) is None

    def test_decode_falls_back_to_load_balance(self):
        policy = SMetricGatedPolicy(MockInstanceProvider())
        inst = _instance(1, [_endpoint(10)], role=PDRole.ROLE_D)
        with (
            patch.object(SMetricGatedPolicy, "select_endpoint_from_list") as mock_gated,
            patch.object(
                LoadBalancePolicy, "select_endpoint_from_list", return_value=(inst, inst.get_all_endpoints()[0])
            ) as mock_lb,
        ):
            selected = policy.select_instance_and_endpoint_from_list([inst], role=PDRole.ROLE_D, req_info=_req_info())
        mock_gated.assert_not_called()
        mock_lb.assert_called_once()
        assert selected[0].id == 1

    def test_factory_and_protocol_registration(self):
        assert isinstance(create(SchedulerType.SMETRIC_GATED, MockInstanceProvider()), SMetricGatedPolicy)
        assert SchedulerType.from_string("smetric_gated") is SchedulerType.SMETRIC_GATED
        assert CANDIDATE_POLICY_SMETRIC_GATED in KNOWN_CANDIDATE_POLICIES

    def test_scheduler_pushes_mean_factors_from_config(self):
        config = CoordinatorConfig()
        config.scheduler_config.scheduler_type = SchedulerType.SMETRIC_GATED
        config.scheduler_config.smetric_gated.active_tokens_mean_factor = 1.5
        config.scheduler_config.smetric_gated.cpu_hit_blocks_mean_factor = 0.5
        policy = Scheduler(instance_provider=MockInstanceProvider(), config=config).get_scheduling_policy()
        assert isinstance(policy, SMetricGatedPolicy)
        assert policy.mean_factors == (1.5, 0.5)

    def test_json_config_sets_factors(self, tmp_path):
        cfg_path = tmp_path / "coordinator.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "scheduler_config": {
                        "scheduler_type": "smetric_gated",
                        "smetric_gated": {"active_tokens_mean_factor": 1.3, "cpu_hit_blocks_mean_factor": 0.8},
                    }
                }
            ),
            encoding="utf-8",
        )
        config = CoordinatorConfig.from_json(str(cfg_path))
        assert config.scheduler_config.scheduler_type is SchedulerType.SMETRIC_GATED
        assert config.scheduler_config.smetric_gated.active_tokens_mean_factor == 1.3
        assert config.scheduler_config.smetric_gated.cpu_hit_blocks_mean_factor == 0.8

    def test_in_process_selection_uses_factors(self):
        inst_a = _instance(1, [_endpoint(10, active_tokens=22, prefill_cost=1)])
        inst_b = _instance(2, [_endpoint(20, active_tokens=8, prefill_cost=2)])
        inst_c = _instance(3, [_endpoint(30, active_tokens=30, prefill_cost=3)])
        instances = [inst_a, inst_b, inst_c]

        def fake_score(insts, info):
            info.smetric_gated_debug = {}
            return sort_candidates([GatedCandidate(i, i.get_all_endpoints()[0], 0.0, 0.0) for i in insts])

        policy = SMetricGatedPolicy(MockInstanceProvider())
        with patch.object(SMetricGatedPolicy, "score_endpoints", side_effect=fake_score):
            plain = policy.select_instance_and_endpoint_from_list(instances, PDRole.ROLE_P, _req_info())
            policy.set_mean_factors(1.2, 1.0)
            loose = policy.select_instance_and_endpoint_from_list(instances, PDRole.ROLE_P, _req_info())
        assert plain[1].id == 20
        assert loose[1].id == 10

    def test_pinned_endpoint_uses_load_balance_within_instance(self):
        inst = Mock()
        inst.id = 1
        ep = Mock()
        ep.id = 10
        with patch.object(LoadBalancePolicy, "select_endpoint_from_instance", return_value=ep) as mock_lb:
            assert select_endpoint_for_instance(inst, scheduler_type="smetric_gated") is ep
        mock_lb.assert_called_once()

    @pytest.mark.asyncio
    async def test_instance_manager_stamps_both_ledger_fields(self):
        config = CoordinatorConfig()
        im = InstanceManager(config)
        inst = _instance(1, [_endpoint(10)])
        await im.refresh_instances(EventType.ADD, [inst])
        await im.update_instance_workload(
            1, 10, Workload(active_tokens=100.0, prefill_cost=60.0, cpu_hit_blocks=3.0)
        )
        _, ledger = await im.get_endpoint_workload(1, 10)
        assert (ledger.active_tokens, ledger.prefill_cost, ledger.cpu_hit_blocks) == (100.0, 60.0, 3.0)


# ---------------------------------------------------------------------------
# Worker client
# ---------------------------------------------------------------------------


def _client(active_factor: float = 1.0, cpu_factor: float = 1.0) -> AsyncSchedulerClient:
    return AsyncSchedulerClient(
        SchedulerClientConfig(
            scheduler_type="smetric_gated",
            smetric_gated=SMetricGatedConfig(
                active_tokens_mean_factor=active_factor,
                cpu_hit_blocks_mean_factor=cpu_factor,
            ),
        )
    )


class TestClientDispatch:
    def test_factors_from_config_reach_the_policy(self):
        client = _client(active_factor=1.7, cpu_factor=0.3)
        inst = _instance(1, [_endpoint(10)])
        with patch.object(
            SMetricGatedPolicy,
            "select_endpoint_candidates_from_list",
            return_value=[(inst, inst.get_all_endpoints()[0], 0.0)],
        ) as m:
            client._select_endpoint_candidates_from_list_with_policy([inst], PDRole.ROLE_P, _req_info(), top_k=1)
        kwargs = m.call_args.kwargs
        assert kwargs["active_tokens_mean_factor"] == 1.7
        assert kwargs["cpu_hit_blocks_mean_factor"] == 0.3

    def test_default_factors_when_config_absent(self):
        client = AsyncSchedulerClient(SchedulerClientConfig(scheduler_type="smetric_gated"))
        assert (client._smetric_gated_active_factor, client._smetric_gated_cpu_factor) == (1.0, 1.0)

    def test_prefill_uses_gated_policy(self):
        client = _client()
        inst = _instance(1, [_endpoint(10)])
        ranked = [(inst, inst.get_all_endpoints()[0], 3.0)]
        with (
            patch.object(SMetricGatedPolicy, "select_endpoint_candidates_from_list", return_value=ranked) as m,
            patch.object(client, "_select_endpoint_candidates_by_load_balance") as mock_lb,
        ):
            candidates, policy = client._select_endpoint_candidates_from_list_with_policy(
                [inst], PDRole.ROLE_P, _req_info(), top_k=1
            )
        assert candidates == ranked and policy == "smetric_gated"
        m.assert_called_once()
        mock_lb.assert_not_called()

    def test_decode_and_conductor_miss_fall_back_to_load_balance(self):
        client = _client()
        inst = _instance(1, [_endpoint(10)])
        lb = [(inst, inst.get_all_endpoints()[0], 0.5)]
        with (
            patch.object(SMetricGatedPolicy, "select_endpoint_candidates_from_list", return_value=None) as m,
            patch.object(client, "_select_endpoint_candidates_by_load_balance", return_value=lb) as mock_lb,
        ):
            _, p_policy = client._select_endpoint_candidates_from_list_with_policy(
                [inst], PDRole.ROLE_P, _req_info(), 1
            )
            _, d_policy = client._select_endpoint_candidates_from_list_with_policy(
                [inst], PDRole.ROLE_D, _req_info(), 1
            )
        assert p_policy == "load_balance" and d_policy == "load_balance"
        assert m.call_count == 1
        assert mock_lb.call_count == 2

    def test_committed_workload_stamps_both_fields(self):
        client = _client()
        inst = _instance(1, [_endpoint(10)])
        ep = inst.get_all_endpoints()[0]
        committed = client._committed_workload_for(
            PDRole.ROLE_P,
            CANDIDATE_POLICY_SMETRIC_GATED,
            inst,
            ep,
            Workload(active_tokens=100.0),
            {},
            100.0,
            prefill_cost_map={(1, 10): 60.0},
            cpu_hit_map={(1, 10): 3.0},
        )
        assert (committed.active_tokens, committed.prefill_cost, committed.cpu_hit_blocks) == (100.0, 60.0, 3.0)

    @pytest.mark.asyncio
    async def test_overlay_survives_shm_token_patch(self):
        client = _client()
        inst = _instance(1, [_endpoint(10)])
        await client._cache.replace_all(PDRole.ROLE_P, [inst])
        client._cache.apply_ledger_delta(1, 10, PDRole.ROLE_P, 40.0, 9.0)
        client._cache.patch_workload_from_shm(1, 10, PDRole.ROLE_P, 12.0)
        ep = client._cache._endpoint_map[(1, 10)]
        assert (ep.workload.active_tokens, ep.workload.prefill_cost, ep.workload.cpu_hit_blocks) == (12.0, 40.0, 9.0)


# ---------------------------------------------------------------------------
# Authoritative arbitration (replaces ALLOCATE_ONLY ZMQ tests from PR #16)
# ---------------------------------------------------------------------------


async def _pool(instances: list[Instance]) -> InstanceManager:
    config = CoordinatorConfig()
    config.scheduler_config.endpoint_instance_score_weight = 0.0
    im = InstanceManager(config)
    await im.refresh_instances(EventType.ADD, instances)
    return im


def _gated_quads(*items: tuple[int, int, float, float]) -> list[tuple[int, int, float, float]]:
    return list(items)


class TestArbitration:
    @pytest.mark.asyncio
    async def test_reorders_by_ledger_prefill_and_gates_on_fresh_ledger(self):
        inst_a = _instance(1, [_endpoint(10), _endpoint(11)])
        inst_b = _instance(2, [_endpoint(20)])
        im = await _pool([inst_a, inst_b])
        await im.update_instance_workload(1, 10, Workload(active_tokens=900, prefill_cost=30))
        await im.update_instance_workload(1, 11, Workload(cpu_hit_blocks=90, prefill_cost=60))
        await im.update_instance_workload(2, 20, Workload(active_tokens=100, cpu_hit_blocks=5, prefill_cost=90))
        ctx = _arbitration_context(im)

        selected = allocate_arbitration.select_authoritative_allocate_candidate(
            ctx,
            (1, 10),
            [(1, 10), (1, 11), (2, 20)],
            PDRole.ROLE_P,
            CANDIDATE_POLICY_SMETRIC_GATED,
            gated_candidates=_gated_quads((2, 20, 50.0, 6.0), (1, 10, 10.0, 1.0), (1, 11, 20.0, 2.0)),
        )

        assert selected is not None
        instance, endpoint, score = selected
        assert (instance.id, endpoint.id) == (2, 20)
        assert score == 90.0

    @pytest.mark.asyncio
    async def test_order_follows_ledger_not_request_cost(self):
        inst = _instance(1, [_endpoint(10), _endpoint(11)])
        im = await _pool([inst])
        await im.update_instance_workload(1, 10, Workload(prefill_cost=50))
        await im.update_instance_workload(1, 11, Workload(prefill_cost=5))
        ctx = _arbitration_context(im)

        selected = allocate_arbitration.select_authoritative_allocate_candidate(
            ctx,
            (1, 10),
            [(1, 10), (1, 11)],
            PDRole.ROLE_P,
            CANDIDATE_POLICY_SMETRIC_GATED,
            gated_candidates=_gated_quads((1, 10, 1.0, 0.0), (1, 11, 99.0, 0.0)),
        )

        assert selected is not None
        assert selected[1].id == 11
        assert selected[2] == 5.0

    @pytest.mark.asyncio
    async def test_equal_active_lets_cpu_gate_decide(self):
        inst = _instance(1, [_endpoint(10), _endpoint(11)])
        im = await _pool([inst])
        await im.update_instance_workload(1, 10, Workload(active_tokens=10, cpu_hit_blocks=50, prefill_cost=5))
        await im.update_instance_workload(1, 11, Workload(active_tokens=10, cpu_hit_blocks=0, prefill_cost=60))
        ctx = _arbitration_context(im)

        selected = allocate_arbitration.select_authoritative_allocate_candidate(
            ctx,
            (1, 10),
            [(1, 10), (1, 11)],
            PDRole.ROLE_P,
            CANDIDATE_POLICY_SMETRIC_GATED,
            gated_candidates=_gated_quads((1, 10, 5.0, 0.0), (1, 11, 60.0, 0.0)),
        )

        assert selected[1].id == 11

    @pytest.mark.asyncio
    async def test_mean_factors_change_the_pick(self):
        inst = _instance(1, [_endpoint(10), _endpoint(11), _endpoint(12)])
        im = await _pool([inst])
        for ep, active in {10: 22, 11: 8, 12: 30}.items():
            await im.update_instance_workload(1, ep, Workload(active_tokens=active, prefill_cost=ep))
        quads = _gated_quads((1, 10, 10.0, 0.0), (1, 11, 11.0, 0.0), (1, 12, 12.0, 0.0))

        strict = allocate_arbitration.select_authoritative_allocate_candidate(
            _arbitration_context(im),
            (1, 10),
            [(1, 10), (1, 11), (1, 12)],
            PDRole.ROLE_P,
            CANDIDATE_POLICY_SMETRIC_GATED,
            gated_candidates=quads,
        )
        assert strict[1].id == 11

        loose = allocate_arbitration.select_authoritative_allocate_candidate(
            _arbitration_context(im, active_factor=1.2),
            (1, 10),
            [(1, 10), (1, 11), (1, 12)],
            PDRole.ROLE_P,
            CANDIDATE_POLICY_SMETRIC_GATED,
            gated_candidates=quads,
        )
        assert loose[1].id == 10

    @pytest.mark.asyncio
    async def test_cpu_factor_on_arbitration(self):
        inst = _instance(1, [_endpoint(10), _endpoint(11), _endpoint(12)])
        im = await _pool([inst])
        for ep, (active, cpu) in {10: (5, 15), 11: (5, 5), 12: (50, 10)}.items():
            await im.update_instance_workload(
                1, ep, Workload(active_tokens=active, cpu_hit_blocks=cpu, prefill_cost=ep)
            )
        quads = _gated_quads((1, 10, 10.0, 0.0), (1, 11, 11.0, 0.0), (1, 12, 12.0, 0.0))

        plain = allocate_arbitration.select_authoritative_allocate_candidate(
            _arbitration_context(im),
            (1, 10),
            [(1, 10), (1, 11), (1, 12)],
            PDRole.ROLE_P,
            CANDIDATE_POLICY_SMETRIC_GATED,
            gated_candidates=quads,
        )
        assert plain[1].id == 11

        loose = allocate_arbitration.select_authoritative_allocate_candidate(
            _arbitration_context(im, cpu_factor=2.0),
            (1, 10),
            [(1, 10), (1, 11), (1, 12)],
            PDRole.ROLE_P,
            CANDIDATE_POLICY_SMETRIC_GATED,
            gated_candidates=quads,
        )
        assert loose[1].id == 10

    @pytest.mark.asyncio
    async def test_no_costs_validates_worker_candidate(self):
        inst = _instance(1, [_endpoint(10)])
        im = await _pool([inst])
        ctx = _arbitration_context(im)
        selected = allocate_arbitration.select_authoritative_allocate_candidate(
            ctx,
            (1, 10),
            [(1, 10)],
            PDRole.ROLE_P,
            CANDIDATE_POLICY_SMETRIC_GATED,
            gated_candidates=None,
        )
        assert selected is not None
        assert selected[1].id == 10

    @pytest.mark.asyncio
    async def test_circuit_open_and_unknown_endpoints_are_skipped(self):
        inst_a = _instance(1, [_endpoint(10)])
        inst_b = _instance(2, [_endpoint(20)])
        im = await _pool([inst_a, inst_b])
        ctx = _arbitration_context(im, blocked=(1,))

        selected = allocate_arbitration.select_authoritative_allocate_candidate(
            ctx,
            (1, 10),
            [(1, 10), (2, 20)],
            PDRole.ROLE_P,
            CANDIDATE_POLICY_SMETRIC_GATED,
            gated_candidates=_gated_quads((1, 10, 1.0, 0.0), (9, 99, 2.0, 0.0), (2, 20, 3.0, 0.0)),
        )

        assert selected[0].id == 2

    @pytest.mark.asyncio
    async def test_engine_type_constraint(self):
        inst_a = _instance(1, [_endpoint(10)])
        inst_a.engine_type = "vllm"
        inst_b = _instance(2, [_endpoint(20)])
        inst_b.engine_type = "sglang"
        im = await _pool([inst_a, inst_b])
        ctx = _arbitration_context(im)

        selected = allocate_arbitration.select_authoritative_allocate_candidate(
            ctx,
            (1, 10),
            [(1, 10), (2, 20)],
            PDRole.ROLE_P,
            CANDIDATE_POLICY_SMETRIC_GATED,
            gated_candidates=_gated_quads((1, 10, 1.0, 0.0), (2, 20, 3.0, 0.0)),
            required_engine_type="sglang",
        )

        assert selected[0].id == 2

    @pytest.mark.asyncio
    async def test_all_candidates_unavailable_returns_empty(self):
        inst = _instance(1, [_endpoint(10)], role=PDRole.ROLE_D)
        im = await _pool([inst])
        ctx = _arbitration_context(im)
        selected = allocate_arbitration.select_authoritative_allocate_candidate(
            ctx,
            (1, 10),
            [(1, 10)],
            PDRole.ROLE_P,
            CANDIDATE_POLICY_SMETRIC_GATED,
            gated_candidates=_gated_quads((1, 10, 1.0, 0.0)),
        )
        assert selected is None
