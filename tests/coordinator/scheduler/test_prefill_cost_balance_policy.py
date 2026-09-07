# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Tests for PrefillCostBalancePolicy: endpoint score = ledger prefill_cost + x * active_tokens."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from motor.common.resources.endpoint import Endpoint, EndpointStatus, Workload
from motor.common.resources.http_msg_spec import EventType
from motor.common.resources.instance import Instance, InsStatus, PDRole, ParallelConfig
from motor.config.coordinator import CoordinatorConfig, SchedulerType
from motor.coordinator.domain.instance_manager import InstanceManager
from motor.coordinator.scheduler.policy.factory import create
from motor.coordinator.scheduler.policy.load_balance import LoadBalancePolicy
from motor.coordinator.scheduler.policy.prefill_cost_balance import PrefillCostBalancePolicy
from motor.coordinator.scheduler.policy.smetric import SMetricPolicy
from motor.coordinator.scheduler.runtime.scheduler_client import (
    AsyncSchedulerClient,
    SchedulerClientConfig,
)
from motor.coordinator.scheduler.runtime.scheduler_server import _SchedulerRequestDispatcher
from motor.coordinator.scheduler.runtime.zmq_protocol import (
    CANDIDATE_POLICY_PREFILL_COST_BALANCE,
    KNOWN_CANDIDATE_POLICIES,
    SchedulerRequest,
    SchedulerRequestType,
    SchedulerResponse,
    SchedulerResponseType,
)
from motor.coordinator.scheduler.scheduler import Scheduler

# Imported after the scheduler package: scheduling_pin <-> scheduler.policy is a known import cycle.
from motor.coordinator.domain.scheduling_pin import select_endpoint_for_instance
from tests.coordinator.scheduler.conftest import MockInstanceProvider


def _endpoint(ep_id: int, prefill_cost: float = 0.0, active_tokens: float = 0.0) -> Endpoint:
    return Endpoint(
        id=ep_id,
        ip="10.0.0.1",
        business_port=f"80{ep_id}",
        status=EndpointStatus.NORMAL,
        workload=Workload(active_tokens=active_tokens, prefill_cost=prefill_cost),
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


def _req_info(token_count: int = 100, req_id: str = "req-pcb") -> SimpleNamespace:
    return SimpleNamespace(
        req_id=req_id,
        req_data={},
        req_len=token_count * 4,
        token_ids=list(range(token_count)),
        smetric_debug=None,
        kv_affinity_debug=None,
    )


class _DummyWorkloadWriter:
    def __init__(self, role_sequences: dict[PDRole, int] | None = None):
        self.sequence = 0
        self.instance_version = 1
        self._role_sequences = role_sequences
        self.writes: list[tuple[int, int]] = []

    def role_sequence(self, role: PDRole) -> int | None:
        if self._role_sequences is None:
            return None
        return self._role_sequences.get(role)

    def write_single_entry_sync(self, instance_id: int, endpoint_id: int) -> None:
        self.sequence += 2
        self.writes.append((instance_id, endpoint_id))

    def write_single_entry_from_workload(self, instance_id, endpoint_id, role, workload) -> None:
        self.write_single_entry_sync(instance_id, endpoint_id)


# ---------------------------------------------------------------------------
# Score formula
# ---------------------------------------------------------------------------


class TestEndpointScore:
    def test_formula_prefill_cost_plus_weighted_active_tokens(self):
        ep = _endpoint(1, prefill_cost=30, active_tokens=100)
        assert PrefillCostBalancePolicy.calculate_endpoint_score(ep, 0.5) == 30 + 0.5 * 100

    def test_default_weight_is_one(self):
        ep = _endpoint(1, prefill_cost=30, active_tokens=100)
        assert PrefillCostBalancePolicy.calculate_endpoint_score(ep) == 130

    def test_zero_weight_ranks_by_prefill_cost_only(self):
        ep = _endpoint(1, prefill_cost=30, active_tokens=100)
        assert PrefillCostBalancePolicy.calculate_endpoint_score(ep, 0.0) == 30

    def test_negative_weight_is_clamped_to_zero(self):
        ep = _endpoint(1, prefill_cost=30, active_tokens=100)
        assert PrefillCostBalancePolicy.calculate_endpoint_score(ep, -5.0) == 30

    def test_missing_prefill_cost_field_reads_as_zero(self):
        ep = Mock()
        ep.workload = SimpleNamespace(active_tokens=40)
        assert PrefillCostBalancePolicy.calculate_endpoint_score(ep, 2.0) == 80


class TestRanking:
    def test_weight_changes_the_winner(self):
        # A: little remaining prefill but many tokens; B: more prefill, fewer tokens.
        inst_a = _instance(1, [_endpoint(10, prefill_cost=10, active_tokens=200)])
        inst_b = _instance(2, [_endpoint(20, prefill_cost=60, active_tokens=20)])

        # x=0 -> prefill_cost only -> A (10 < 60)
        low_x = PrefillCostBalancePolicy.select_endpoint_from_list([inst_a, inst_b], active_tokens_weight=0.0)
        assert low_x[1].id == 10
        # x=1 -> A=210, B=80 -> B
        one_x = PrefillCostBalancePolicy.select_endpoint_from_list([inst_a, inst_b], active_tokens_weight=1.0)
        assert one_x[1].id == 20

    def test_top_k_is_sorted_ascending_with_scores(self):
        inst = _instance(
            1,
            [
                _endpoint(10, prefill_cost=50, active_tokens=10),
                _endpoint(11, prefill_cost=5, active_tokens=10),
                _endpoint(12, prefill_cost=20, active_tokens=10),
            ],
        )
        ranked = PrefillCostBalancePolicy.select_endpoint_candidates_from_list([inst], top_k=3, active_tokens_weight=1)
        assert [(ep.id, score) for _inst, ep, score in ranked] == [(11, 15.0), (12, 30.0), (10, 60.0)]

    def test_ties_follow_rotated_traversal_order(self):
        inst_a = _instance(1, [_endpoint(10)])
        inst_b = _instance(2, [_endpoint(20)])
        first = PrefillCostBalancePolicy.select_endpoint_from_list([inst_a, inst_b], start_index=0)
        second = PrefillCostBalancePolicy.select_endpoint_from_list([inst_a, inst_b], start_index=1)
        assert first[1].id == 10
        assert second[1].id == 20

    def test_blocked_instances_are_skipped(self):
        inst_a = _instance(1, [_endpoint(10, prefill_cost=0)])
        inst_b = _instance(2, [_endpoint(20, prefill_cost=99)])
        selected = PrefillCostBalancePolicy.select_endpoint_candidates_from_list(
            [inst_a, inst_b], is_blocked=lambda iid: iid == 1
        )
        assert [(i.id, ep.id) for i, ep, _s in selected] == [(2, 20)]

    def test_empty_and_zero_top_k(self):
        assert PrefillCostBalancePolicy.select_endpoint_candidates_from_list([]) == []
        assert PrefillCostBalancePolicy.select_endpoint_candidates_from_list([_instance(1, [_endpoint(1)])], 0) == []
        assert PrefillCostBalancePolicy.select_endpoint_from_list([]) is None


class TestPolicyInstance:
    def test_factory_registers_policy(self):
        policy = create(SchedulerType.PREFILL_COST_BALANCE, MockInstanceProvider())
        assert isinstance(policy, PrefillCostBalancePolicy)

    def test_scheduler_type_from_string(self):
        assert SchedulerType.from_string("prefill_cost_balance") is SchedulerType.PREFILL_COST_BALANCE

    def test_candidate_policy_is_known(self):
        assert CANDIDATE_POLICY_PREFILL_COST_BALANCE in KNOWN_CANDIDATE_POLICIES

    def test_scheduler_pushes_weight_from_config(self):
        config = CoordinatorConfig()
        config.scheduler_config.scheduler_type = SchedulerType.PREFILL_COST_BALANCE
        config.scheduler_config.prefill_cost_balance.active_tokens_weight = 0.25
        scheduler = Scheduler(instance_provider=MockInstanceProvider(), config=config)
        policy = scheduler.get_scheduling_policy()
        assert isinstance(policy, PrefillCostBalancePolicy)
        assert policy.active_tokens_weight == 0.25

    def test_select_from_provider_uses_weight(self):
        inst_a = _instance(1, [_endpoint(10, prefill_cost=10, active_tokens=200)])
        inst_b = _instance(2, [_endpoint(20, prefill_cost=60, active_tokens=20)])
        provider = MockInstanceProvider({PDRole.ROLE_P: {1: inst_a, 2: inst_b}})
        policy = PrefillCostBalancePolicy(provider)
        policy.set_active_tokens_weight(0.0)
        assert policy.select_instance_and_endpoint(PDRole.ROLE_P)[1].id == 10
        policy.set_active_tokens_weight(1.0)
        assert policy.select_instance_and_endpoint(PDRole.ROLE_P)[1].id == 20

    def test_from_list_collects_costs_for_prefill_but_ranks_by_ledger(self):
        # Conductor says endpoint 20 has the best prefix hit, but the ledger says 10 is emptier.
        inst_a = _instance(1, [_endpoint(10, prefill_cost=0, active_tokens=0)])
        inst_b = _instance(2, [_endpoint(20, prefill_cost=500, active_tokens=500)])
        req_info = _req_info(100)
        policy = PrefillCostBalancePolicy(MockInstanceProvider())

        def fake_smetric(instances, info, top_k=1):
            info.smetric_debug = {(1, 10): 100.0, (2, 20): 5.0}
            return [(inst_b, inst_b.get_all_endpoints()[0], 5.0)]

        with patch.object(SMetricPolicy, "select_endpoint_candidates_from_list", side_effect=fake_smetric) as m:
            selected = policy.select_instance_and_endpoint_from_list([inst_a, inst_b], PDRole.ROLE_P, req_info)

        m.assert_called_once()
        assert selected[1].id == 10
        assert req_info.smetric_debug == {(1, 10): 100.0, (2, 20): 5.0}

    def test_from_list_skips_conductor_for_decode(self):
        inst = _instance(1, [_endpoint(10)], role=PDRole.ROLE_D)
        policy = PrefillCostBalancePolicy(MockInstanceProvider())
        with patch.object(SMetricPolicy, "select_endpoint_candidates_from_list") as m:
            selected = policy.select_instance_and_endpoint_from_list([inst], PDRole.ROLE_D, _req_info())
        m.assert_not_called()
        assert selected[1].id == 10

    def test_collect_costs_swallows_conductor_errors(self):
        inst = _instance(1, [_endpoint(10)])
        req_info = _req_info()
        with patch.object(SMetricPolicy, "select_endpoint_candidates_from_list", side_effect=RuntimeError("down")):
            assert PrefillCostBalancePolicy.collect_request_prefill_costs([inst], req_info) is False
        assert req_info.smetric_debug is None

    def test_pinned_endpoint_uses_load_balance_within_instance(self):
        inst = Mock()
        inst.id = 1
        ep = Mock()
        ep.id = 10
        with patch.object(LoadBalancePolicy, "select_endpoint_from_instance", return_value=ep) as mock_lb:
            got = select_endpoint_for_instance(inst, scheduler_type="prefill_cost_balance")
        assert got is ep
        mock_lb.assert_called_once()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_weight(self):
        assert CoordinatorConfig().scheduler_config.prefill_cost_balance.active_tokens_weight == 1.0

    def test_json_sets_type_and_weight(self, tmp_path):
        cfg_path = tmp_path / "coordinator.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "scheduler_config": {
                        "scheduler_type": "prefill_cost_balance",
                        "prefill_cost_balance": {"active_tokens_weight": 0.3},
                    }
                }
            ),
            encoding="utf-8",
        )
        config = CoordinatorConfig.from_json(str(cfg_path))
        assert config.scheduler_config.scheduler_type is SchedulerType.PREFILL_COST_BALANCE
        assert config.scheduler_config.prefill_cost_balance.active_tokens_weight == 0.3

    def test_negative_weight_is_rejected(self):
        config = CoordinatorConfig()
        config.scheduler_config.prefill_cost_balance.active_tokens_weight = -1.0
        config._errors = []
        config._validate_positive_number(
            config.scheduler_config.prefill_cost_balance.active_tokens_weight,
            "prefill_cost_balance.active_tokens_weight",
            allow_zero=True,
        )
        assert any("prefill_cost_balance.active_tokens_weight" in err for err in config._errors)


# ---------------------------------------------------------------------------
# Worker (AsyncSchedulerClient)
# ---------------------------------------------------------------------------


def _client(weight: float = 1.0, client_index: int = 0, client_count: int = 1) -> AsyncSchedulerClient:
    config = CoordinatorConfig()
    config.scheduler_config.prefill_cost_balance.active_tokens_weight = weight
    return AsyncSchedulerClient(
        SchedulerClientConfig(
            scheduler_type="prefill_cost_balance",
            client_index=client_index,
            client_count=client_count,
            prefill_cost_balance=config.scheduler_config.prefill_cost_balance,
        )
    )


class TestClientDispatch:
    def test_prefill_ranks_by_ledger_and_collects_costs(self):
        client = _client(weight=1.0)
        inst_a = _instance(1, [_endpoint(10, prefill_cost=10, active_tokens=200)])
        inst_b = _instance(2, [_endpoint(20, prefill_cost=60, active_tokens=20)])
        req_info = _req_info()

        with (
            patch.object(PrefillCostBalancePolicy, "collect_request_prefill_costs", return_value=True) as mock_costs,
            patch.object(client, "_select_endpoint_candidates_by_load_balance") as mock_lb,
        ):
            candidates, candidate_policy = client._select_endpoint_candidates_from_list_with_policy(
                [inst_a, inst_b], PDRole.ROLE_P, req_info, top_k=1
            )

        assert candidate_policy == "prefill_cost_balance"
        assert [(i.id, ep.id, score) for i, ep, score in candidates] == [(2, 20, 80.0)]
        mock_costs.assert_called_once_with([inst_a, inst_b], req_info)
        mock_lb.assert_not_called()

    def test_weight_from_config_is_applied(self):
        client = _client(weight=0.0)
        inst_a = _instance(1, [_endpoint(10, prefill_cost=10, active_tokens=200)])
        inst_b = _instance(2, [_endpoint(20, prefill_cost=60, active_tokens=20)])

        with patch.object(PrefillCostBalancePolicy, "collect_request_prefill_costs", return_value=False):
            candidates, _ = client._select_endpoint_candidates_from_list_with_policy(
                [inst_a, inst_b], PDRole.ROLE_P, _req_info(), top_k=1
            )
        assert candidates[0][1].id == 10

    def test_decode_skips_conductor_but_still_ranks_by_ledger(self):
        client = _client(weight=1.0)
        inst_a = _instance(1, [_endpoint(10, active_tokens=50)], role=PDRole.ROLE_D)
        inst_b = _instance(2, [_endpoint(20, active_tokens=5)], role=PDRole.ROLE_D)

        with patch.object(PrefillCostBalancePolicy, "collect_request_prefill_costs") as mock_costs:
            candidates, candidate_policy = client._select_endpoint_candidates_from_list_with_policy(
                [inst_a, inst_b], PDRole.ROLE_D, _req_info(), top_k=1
            )
        mock_costs.assert_not_called()
        assert candidate_policy == "prefill_cost_balance"
        assert candidates[0][1].id == 20

    def test_circuit_broken_instance_is_skipped(self):
        client = _client()
        inst_a = _instance(1, [_endpoint(10, active_tokens=0)])
        inst_b = _instance(2, [_endpoint(20, active_tokens=100)])
        client._cb_blocked_instances.add(1)
        with patch.object(PrefillCostBalancePolicy, "collect_request_prefill_costs", return_value=False):
            candidates, _ = client._select_endpoint_candidates_from_list_with_policy(
                [inst_a, inst_b], PDRole.ROLE_P, _req_info(), top_k=1
            )
        assert candidates[0][0].id == 2

    @pytest.mark.asyncio
    async def test_allocate_forwards_every_endpoint_cost_and_isl(self):
        client = _client()
        inst = Mock()
        inst.id = 1
        ep = Mock()
        ep.id = 10
        req_info = _req_info(3)
        req_info.smetric_debug = {(1, 10): 3, (2, 20): 1}
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
            new=AsyncMock(return_value=([(inst, ep, 3.0)], CANDIDATE_POLICY_PREFILL_COST_BALANCE)),
        ):
            client._transport.send_request = fake_send
            await client.select_and_allocate(PDRole.ROLE_P, req_info)

        data = captured["data"]
        assert data["candidate_policy"] == "prefill_cost_balance"
        assert data["instance_id"] == 1 and data["endpoint_id"] == 10
        costs = {(c["instance_id"], c["endpoint_id"]): c["prefill_cost"] for c in data["candidates"]}
        assert costs == {(1, 10): 3, (2, 20): 1}
        assert data["isl"] == 3
        assert data["workload_active_tokens"] == 3.0
        assert "prefill_load_scale" not in data

    @pytest.mark.asyncio
    async def test_decode_allocate_does_not_forward_stale_prefill_costs(self):
        client = _client()
        inst = Mock()
        inst.id = 5
        ep = Mock()
        ep.id = 50
        req_info = _req_info(3)
        # Left over from this request's prefill leg.
        req_info.smetric_debug = {(1, 10): 3, (2, 20): 1}
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
            new=AsyncMock(return_value=([(inst, ep, 0.0)], CANDIDATE_POLICY_PREFILL_COST_BALANCE)),
        ):
            client._transport.send_request = fake_send
            await client.select_and_allocate(PDRole.ROLE_D, req_info)

        assert captured["data"]["candidates"] == [{"instance_id": 5, "endpoint_id": 50}]

    @pytest.mark.asyncio
    async def test_pinned_allocation_keeps_single_candidate(self):
        client = _client()
        inst = Mock()
        inst.id = 7
        ep = Mock()
        ep.id = 70
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
            await client.select_and_allocate(PDRole.ROLE_P, _req_info(), target_instance_id=inst.id)

        assert captured["data"]["candidate_policy"] == "round_robin"
        assert captured["data"]["instance_id"] == 7


# ---------------------------------------------------------------------------
# Scheduler server (authoritative arbitration + ledger stamp)
# ---------------------------------------------------------------------------


async def _dispatcher(weight: float, instances: list[Instance], writer=None):
    config = CoordinatorConfig()
    config.scheduler_config.scheduler_type = SchedulerType.PREFILL_COST_BALANCE
    config.scheduler_config.prefill_cost_balance.active_tokens_weight = weight
    instance_manager = InstanceManager(config)
    await instance_manager.refresh_instances(EventType.ADD, instances)
    scheduler = Scheduler(instance_provider=instance_manager, config=config)
    writer = writer or _DummyWorkloadWriter()
    dispatcher = _SchedulerRequestDispatcher(instance_manager, scheduler, config, workload_writer=writer)
    return dispatcher, instance_manager, writer


def _allocate_request(
    instance_id: int,
    endpoint_id: int,
    candidates: list[dict] | None = None,
    isl: float | None = 100.0,
    role: PDRole = PDRole.ROLE_P,
    **extra,
) -> SchedulerRequest:
    data = {
        "instance_id": instance_id,
        "endpoint_id": endpoint_id,
        "role": role.value,
        "req_id": "req-pcb",
        "workload_active_tokens": 100.0,
        "candidate_policy": CANDIDATE_POLICY_PREFILL_COST_BALANCE,
    }
    if candidates is not None:
        data["candidates"] = candidates
    if isl is not None:
        data["isl"] = isl
    data.update(extra)
    return SchedulerRequest(request_type=SchedulerRequestType.ALLOCATE_ONLY, request_id="alloc-pcb", data=data)


class TestServerArbitration:
    @pytest.mark.asyncio
    async def test_reranks_by_fresh_ledger_with_weight(self):
        # Worker proposes 1/10 (its view was stale); the ledger says 2/20 is cheaper at x=1.
        inst_a = _instance(1, [_endpoint(10)])
        inst_b = _instance(2, [_endpoint(20)])
        dispatcher, im, writer = await _dispatcher(1.0, [inst_a, inst_b])
        await im.update_instance_workload(1, 10, Workload(active_tokens=200, prefill_cost=10))
        await im.update_instance_workload(2, 20, Workload(active_tokens=20, prefill_cost=60))

        response = await dispatcher.dispatch(
            _allocate_request(
                1,
                10,
                candidates=[
                    {"instance_id": 1, "endpoint_id": 10, "prefill_cost": 40.0},
                    {"instance_id": 2, "endpoint_id": 20, "prefill_cost": 70.0},
                ],
            )
        )

        assert response.response_type == SchedulerResponseType.SUCCESS
        assert response.data["instance"]["id"] == 2
        assert response.data["endpoint"]["id"] == 20
        assert response.data["fast_path"] is False
        # score = prefill_cost + 1.0 * active_tokens on the ledger before this allocation
        assert response.data["selected_score"] == pytest.approx(60 + 20)
        # committed: demand tokens + this request's own remaining prefill on the chosen endpoint
        committed = response.data["committed_workload"]
        assert committed["active_tokens"] == 100.0
        assert committed["prefill_cost"] == 70.0
        _, ledger = await im.get_endpoint_workload(2, 20)
        assert ledger.active_tokens == 120.0
        assert ledger.prefill_cost == 130.0
        assert writer.writes == [(2, 20)]

    @pytest.mark.asyncio
    async def test_zero_weight_ranks_by_prefill_cost_only(self):
        inst_a = _instance(1, [_endpoint(10)])
        inst_b = _instance(2, [_endpoint(20)])
        dispatcher, im, _ = await _dispatcher(0.0, [inst_a, inst_b])
        await im.update_instance_workload(1, 10, Workload(active_tokens=1000, prefill_cost=5))
        await im.update_instance_workload(2, 20, Workload(active_tokens=1, prefill_cost=50))

        response = await dispatcher.dispatch(_allocate_request(2, 20, candidates=[]))

        assert response.data["instance"]["id"] == 1
        assert response.data["selected_score"] == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_missing_cost_stamps_full_isl(self):
        inst = _instance(1, [_endpoint(10)])
        dispatcher, im, _ = await _dispatcher(1.0, [inst])

        response = await dispatcher.dispatch(_allocate_request(1, 10, candidates=[], isl=77.0))

        assert response.data["committed_workload"]["prefill_cost"] == 77.0
        _, ledger = await im.get_endpoint_workload(1, 10)
        assert ledger.prefill_cost == 77.0

    @pytest.mark.asyncio
    async def test_missing_cost_and_isl_stamps_demand_tokens(self):
        inst = _instance(1, [_endpoint(10)])
        dispatcher, _im, _ = await _dispatcher(1.0, [inst])

        response = await dispatcher.dispatch(_allocate_request(1, 10, candidates=None, isl=None))

        assert response.data["committed_workload"]["prefill_cost"] == 100.0

    @pytest.mark.asyncio
    async def test_reported_zero_cost_is_kept(self):
        """A fully cached prompt has cost 0.0; it must not be replaced by the isl fallback."""
        inst = _instance(1, [_endpoint(10)])
        dispatcher, _im, _ = await _dispatcher(1.0, [inst])

        response = await dispatcher.dispatch(
            _allocate_request(1, 10, candidates=[{"instance_id": 1, "endpoint_id": 10, "prefill_cost": 0.0}])
        )

        assert response.data["committed_workload"]["prefill_cost"] == 0.0

    @pytest.mark.asyncio
    async def test_decode_role_does_not_stamp_prefill_cost(self):
        inst = _instance(1, [_endpoint(10)], role=PDRole.ROLE_D)
        dispatcher, _im, _ = await _dispatcher(1.0, [inst])

        response = await dispatcher.dispatch(_allocate_request(1, 10, candidates=[], role=PDRole.ROLE_D))

        assert response.data["instance"]["id"] == 1
        assert response.data["committed_workload"]["prefill_cost"] == 0.0

    @pytest.mark.asyncio
    async def test_fast_path_keeps_worker_top1_and_reports_policy_score(self):
        inst_a = _instance(1, [_endpoint(10)])
        inst_b = _instance(2, [_endpoint(20)])
        writer = _DummyWorkloadWriter(role_sequences={PDRole.ROLE_P: 4})
        dispatcher, im, _ = await _dispatcher(0.5, [inst_a, inst_b], writer=writer)
        await im.update_instance_workload(1, 10, Workload(active_tokens=100, prefill_cost=10))

        response = await dispatcher.dispatch(
            _allocate_request(
                1,
                10,
                candidates=[{"instance_id": 1, "endpoint_id": 10, "prefill_cost": 8.0}],
                role_workload_sequence=4,
                instance_version=1,
            )
        )

        assert response.data["fast_path"] is True
        assert response.data["instance"]["id"] == 1
        assert response.data["selected_score"] == pytest.approx(10 + 0.5 * 100)

    @pytest.mark.asyncio
    async def test_circuit_open_instance_is_skipped_in_rerank(self):
        inst_a = _instance(1, [_endpoint(10)])
        inst_b = _instance(2, [_endpoint(20)])
        dispatcher, im, _ = await _dispatcher(1.0, [inst_a, inst_b])
        await im.update_instance_workload(2, 20, Workload(active_tokens=500))
        dispatcher._cb_manager.is_open = lambda iid: iid == 1

        response = await dispatcher.dispatch(_allocate_request(1, 10, candidates=[]))

        assert response.data["instance"]["id"] == 2

    @pytest.mark.asyncio
    async def test_engine_type_constraint_is_honored(self):
        inst_a = _instance(1, [_endpoint(10)])
        inst_a.engine_type = "vllm"
        inst_b = _instance(2, [_endpoint(20)])
        inst_b.engine_type = "sglang"
        dispatcher, im, _ = await _dispatcher(1.0, [inst_a, inst_b])
        await im.update_instance_workload(2, 20, Workload(active_tokens=500))

        response = await dispatcher.dispatch(_allocate_request(1, 10, candidates=[], required_engine_type="sglang"))

        assert response.data["instance"]["id"] == 2

    @pytest.mark.asyncio
    async def test_no_endpoint_for_role_returns_empty(self):
        inst = _instance(1, [_endpoint(10)], role=PDRole.ROLE_D)
        dispatcher, _im, _ = await _dispatcher(1.0, [inst])

        response = await dispatcher.dispatch(_allocate_request(1, 10, candidates=[], role=PDRole.ROLE_P))

        assert response.response_type == SchedulerResponseType.SUCCESS
        assert response.data["instance"] is None
