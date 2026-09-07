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
from motor.coordinator.scheduler.runtime.scheduler_server import _SchedulerRequestDispatcher
from motor.coordinator.scheduler.runtime.zmq_protocol import (
    CANDIDATE_POLICY_SMETRIC_GATED,
    KNOWN_CANDIDATE_POLICIES,
    SchedulerRequest,
    SchedulerRequestType,
    SchedulerResponse,
    SchedulerResponseType,
)
from motor.coordinator.scheduler.scheduler import Scheduler

# Imported after the scheduler package: scheduling_pin <-> scheduler.policy is a known import cycle.
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


class _DummyWorkloadWriter:
    def __init__(self):
        self.sequence = 0
        self.instance_version = 1
        self.writes: list[tuple[int, int]] = []

    def role_sequence(self, role: PDRole) -> int | None:
        return None

    def write_single_entry_sync(self, instance_id: int, endpoint_id: int) -> None:
        self.sequence += 2
        self.writes.append((instance_id, endpoint_id))

    def write_single_entry_from_workload(self, instance_id, endpoint_id, role, workload) -> None:
        self.write_single_entry_sync(instance_id, endpoint_id)


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
        # ep2 has the best cache hit for this request (req_cost 1) but the heavier ledger.
        ranked = sort_candidates([_cand(1, ledger_prefill=40, req_cost=90), _cand(2, ledger_prefill=80, req_cost=1)])
        assert [c.endpoint.id for c in ranked] == [1, 2]

    def test_first_under_both_averages_wins(self):
        # Lowest ledger prefill (ep1) is hot on active tokens; ep2 is hot on cpu hits; ep3 is under both.
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
        # ep1 sits exactly on both averages -> accepted (<=), even though ep2 is further below.
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
        # All ledgers 0 -> means 0 -> every endpoint passes with <=; the head (lowest prefill) wins.
        ranked = sort_candidates([_cand(1, ledger_prefill=0), _cand(2, ledger_prefill=0), _cand(3, ledger_prefill=0)])
        chosen, reason, active_threshold, cpu_threshold = pick_gated(ranked)
        assert chosen.endpoint.id == 1 and reason == PICK_BOTH_GATES
        assert (active_threshold, cpu_threshold) == (0.0, 0.0)

    def test_fallback_active_gate_only(self):
        # Nobody is under both: ep1 under active but over cpu, ep2 the reverse.
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
        # Zero factors make both thresholds 0 while every ledger is positive: nothing passes.
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
        # ep1 (lowest ledger prefill) is 10% over the active mean: rejected at 1.0, accepted at 1.2.
        ranked = sort_candidates(
            [
                _cand(1, ledger_prefill=1, active=22, cpu=0),
                _cand(2, ledger_prefill=2, active=8, cpu=0),
                _cand(3, ledger_prefill=3, active=30, cpu=0),
            ]
        )
        plain, reason_plain, _a, _c = pick_gated(ranked)
        loose, reason_loose, _a2, _c2 = pick_gated(ranked, active_tokens_mean_factor=1.2)
        assert plain.endpoint.id == 2 and reason_plain == PICK_BOTH_GATES  # cpu all 0 -> cpu gate passes (<=)
        assert loose.endpoint.id == 1 and reason_loose == PICK_BOTH_GATES

    def test_factor_below_one_tightens_gate(self):
        # ep1 is under the plain mean (15 < 20) but not under 0.5 * mean (10); ep2 is.
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
        # Both under the active mean; ep1 over the cpu mean at 1.0 but under it at 2.0.
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
        assert active_threshold == 0.0  # negative -> 0: only an idle endpoint (0 <= 0) can pass
        assert cpu_threshold == 20.0  # None -> default 1.0


# ---------------------------------------------------------------------------
# Policy (conductor scoring + registration)
# ---------------------------------------------------------------------------


class TestPolicy:
    @patch("motor.coordinator.scheduler.policy.smetric_gated.ConductorApiClient.query_conductor")
    def test_score_endpoints_reads_cost_and_cpu_hits_and_orders_by_ledger(self, mock_query):
        # Ledger prefill: ep10=300, ep11=100, ep20=200 -> order 11, 20, 10 regardless of request cost.
        inst_a = _instance(1, [_endpoint(10, prefill_cost=300), _endpoint(11, prefill_cost=100)])
        inst_b = _instance(2, [_endpoint(20, prefill_cost=200)])
        req_info = _req_info(100)
        mock_query.return_value = _conductor_tenant(
            inst_a,
            inst_b,
            dp={
                (1, 10): {"npu_blocks": 1, "cpu_blocks": 4, "matched_tokens": 90},  # best hit, heaviest ledger
                (1, 11): {"npu_blocks": 0, "cpu_blocks": 0, "matched_tokens": 0},
                (2, 20): 50,  # legacy int match
            },
        )

        ranked = SMetricGatedPolicy.score_endpoints([inst_a, inst_b], req_info)

        assert [(c.endpoint.id, c.ledger_prefill_cost, c.prefill_cost, c.cpu_hit_blocks) for c in ranked] == [
            (11, 100.0, 100.0, 0.0),
            (20, 200.0, 50.0, 0.0),
            (10, 300.0, 10.0, 4.0),
        ]
        assert req_info.smetric_gated_debug == {(1, 11): (100.0, 0.0), (2, 20): (50.0, 0.0), (1, 10): (10.0, 4.0)}
        assert req_info.smetric_debug is None
        assert allocated_prefill_cost(req_info, 1, 10) == 10.0
        assert allocated_cpu_hit_blocks(req_info, 1, 10) == 4.0
        assert allocated_cpu_hit_blocks(req_info, 9, 9) == 0.0

    @patch("motor.coordinator.scheduler.policy.smetric_gated.ConductorApiClient.query_conductor")
    def test_worker_proposal_puts_gated_pick_first(self, mock_query):
        # ep10 has the lowest ledger prefill but is hot on active tokens (SHM view); ep20 passes.
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
        # ep10 first in ledger order, 10% over the active mean (22 vs 20): only passes with factor > 1.1.
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
    async def test_in_process_allocation_stamps_both_ledger_fields(self):
        config = CoordinatorConfig()
        config.scheduler_config.scheduler_type = SchedulerType.SMETRIC_GATED
        im = InstanceManager(config)
        inst = _instance(1, [_endpoint(10)])
        await im.refresh_instances(EventType.ADD, [inst])
        scheduler = Scheduler(instance_provider=im, config=config)
        req_info = _req_info(100)

        def fake_score(instances, info):
            info.smetric_gated_debug = {(1, 10): (60.0, 3.0)}
            return [GatedCandidate(instances[0], instances[0].get_all_endpoints()[0], 60.0, 3.0)]

        with patch.object(SMetricGatedPolicy, "score_endpoints", side_effect=fake_score):
            result = await scheduler.select_and_allocate(PDRole.ROLE_P, req_info)

        assert result is not None
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
        assert m.call_count == 1  # only the prefill leg queries the conductor
        assert mock_lb.call_count == 2

    @pytest.mark.asyncio
    async def test_allocate_forwards_cost_and_cpu_hits_for_every_endpoint(self):
        client = _client()
        inst = Mock()
        inst.id = 1
        ep = Mock()
        ep.id = 10
        req_info = _req_info(3)
        req_info.smetric_gated_debug = {(1, 10): (3.0, 2.0), (2, 20): (1.0, 7.0)}
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
            new=AsyncMock(return_value=([(inst, ep, 3.0)], CANDIDATE_POLICY_SMETRIC_GATED)),
        ):
            client._transport.send_request = fake_send
            await client.select_and_allocate(PDRole.ROLE_P, req_info)

        data = captured["data"]
        assert data["candidate_policy"] == "smetric_gated"
        assert data["isl"] == 3
        assert sorted(data["candidates"], key=lambda c: c["instance_id"]) == [
            {"instance_id": 1, "endpoint_id": 10, "prefill_cost": 3.0, "cpu_hit_blocks": 2.0},
            {"instance_id": 2, "endpoint_id": 20, "prefill_cost": 1.0, "cpu_hit_blocks": 7.0},
        ]

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
            patch("motor.coordinator.scheduler.runtime.scheduler_client.resolve_pinned_instance", return_value=inst),
            patch("motor.coordinator.scheduler.runtime.scheduler_client.select_endpoint_for_instance", return_value=ep),
        ):
            await client.select_and_allocate(PDRole.ROLE_P, _req_info(), target_instance_id=inst.id)

        assert captured["data"]["candidate_policy"] == "round_robin"
        assert captured["data"]["candidates"] == [{"instance_id": 7, "endpoint_id": 70}]


# ---------------------------------------------------------------------------
# Scheduler server arbitration
# ---------------------------------------------------------------------------


async def _dispatcher(instances: list[Instance], active_factor: float = 1.0, cpu_factor: float = 1.0):
    config = CoordinatorConfig()
    config.scheduler_config.scheduler_type = SchedulerType.SMETRIC_GATED
    config.scheduler_config.smetric_gated.active_tokens_mean_factor = active_factor
    config.scheduler_config.smetric_gated.cpu_hit_blocks_mean_factor = cpu_factor
    im = InstanceManager(config)
    await im.refresh_instances(EventType.ADD, instances)
    scheduler = Scheduler(instance_provider=im, config=config)
    writer = _DummyWorkloadWriter()
    return _SchedulerRequestDispatcher(im, scheduler, config, workload_writer=writer), im, writer


def _cand_payload(instance_id: int, endpoint_id: int, cost: float, cpu: float = 0.0) -> dict:
    return {"instance_id": instance_id, "endpoint_id": endpoint_id, "prefill_cost": cost, "cpu_hit_blocks": cpu}


def _allocate(instance_id: int, endpoint_id: int, candidates: list[dict] | None, **extra) -> SchedulerRequest:
    data = {
        "instance_id": instance_id,
        "endpoint_id": endpoint_id,
        "role": PDRole.ROLE_P.value,
        "req_id": "req-gated",
        "workload_active_tokens": 100.0,
        "isl": 100.0,
        "candidate_policy": CANDIDATE_POLICY_SMETRIC_GATED,
    }
    if candidates is not None:
        data["candidates"] = candidates
    data.update(extra)
    return SchedulerRequest(request_type=SchedulerRequestType.ALLOCATE_ONLY, request_id="alloc-gated", data=data)


class TestServerArbitration:
    @pytest.mark.asyncio
    async def test_reorders_by_ledger_prefill_and_gates_on_fresh_ledger(self):
        inst_a = _instance(1, [_endpoint(10), _endpoint(11)])
        inst_b = _instance(2, [_endpoint(20)])
        dispatcher, im, writer = await _dispatcher([inst_a, inst_b])
        # Ledger prefill order 10 (30) -> 11 (60) -> 20 (90); ep10 hot on active, ep11 hot on cpu,
        # ep20 under both averages.
        await im.update_instance_workload(1, 10, Workload(active_tokens=900, prefill_cost=30))
        await im.update_instance_workload(1, 11, Workload(cpu_hit_blocks=90, prefill_cost=60))
        await im.update_instance_workload(2, 20, Workload(active_tokens=100, cpu_hit_blocks=5, prefill_cost=90))

        # Worker proposed ep10; the request's own costs (ep20 the most expensive) must not reorder.
        response = await dispatcher.dispatch(
            _allocate(
                1,
                10,
                [
                    _cand_payload(2, 20, 50.0, cpu=6.0),
                    _cand_payload(1, 10, 10.0, cpu=1.0),
                    _cand_payload(1, 11, 20.0, cpu=2.0),
                ],
            )
        )

        assert response.response_type == SchedulerResponseType.SUCCESS
        assert response.data["instance"]["id"] == 2
        assert response.data["endpoint"]["id"] == 20
        assert response.data["fast_path"] is False
        assert response.data["selected_score"] == 90.0  # ledger prefill_cost of the committed endpoint
        committed = response.data["committed_workload"]
        assert (committed["active_tokens"], committed["prefill_cost"], committed["cpu_hit_blocks"]) == (
            100.0,
            50.0,
            6.0,
        )
        _, ledger = await im.get_endpoint_workload(2, 20)
        assert (ledger.active_tokens, ledger.prefill_cost, ledger.cpu_hit_blocks) == (200.0, 140.0, 11.0)
        assert writer.writes == [(2, 20)]

    @pytest.mark.asyncio
    async def test_order_follows_ledger_not_request_cost(self):
        inst = _instance(1, [_endpoint(10), _endpoint(11)])
        dispatcher, im, _ = await _dispatcher([inst])
        # Both idle on active/cpu (everything passes the <= gates), so the head of the ledger
        # order wins: ep11 (ledger 5) even though ep10 has the better cache hit.
        await im.update_instance_workload(1, 10, Workload(prefill_cost=50))
        await im.update_instance_workload(1, 11, Workload(prefill_cost=5))

        response = await dispatcher.dispatch(_allocate(1, 10, [_cand_payload(1, 10, 1.0), _cand_payload(1, 11, 99.0)]))

        assert response.data["endpoint"]["id"] == 11
        assert response.data["selected_score"] == 5.0
        assert response.data["committed_workload"]["prefill_cost"] == 99.0

    @pytest.mark.asyncio
    async def test_equal_active_lets_cpu_gate_decide(self):
        inst = _instance(1, [_endpoint(10), _endpoint(11)])
        dispatcher, im, _ = await _dispatcher([inst])
        await im.update_instance_workload(1, 10, Workload(active_tokens=10, cpu_hit_blocks=50, prefill_cost=5))
        await im.update_instance_workload(1, 11, Workload(active_tokens=10, cpu_hit_blocks=0, prefill_cost=60))

        response = await dispatcher.dispatch(_allocate(1, 10, [_cand_payload(1, 10, 5.0), _cand_payload(1, 11, 60.0)]))

        # active equal -> both pass the active gate (<=); ep10 is over the cpu mean (50 > 25), ep11 under.
        assert response.data["endpoint"]["id"] == 11

    @pytest.mark.asyncio
    async def test_cpu_gate_with_active_headroom_prefers_cold_cpu(self):
        inst = _instance(1, [_endpoint(10), _endpoint(11)])
        dispatcher, im, _ = await _dispatcher([inst])
        # ep10 first in ledger order but over the cpu average; ep11 is under both.
        await im.update_instance_workload(1, 10, Workload(active_tokens=10, cpu_hit_blocks=50, prefill_cost=1))
        await im.update_instance_workload(1, 11, Workload(active_tokens=5, cpu_hit_blocks=0, prefill_cost=2))

        response = await dispatcher.dispatch(_allocate(1, 10, [_cand_payload(1, 10, 5.0), _cand_payload(1, 11, 60.0)]))

        assert response.data["endpoint"]["id"] == 11

    @pytest.mark.asyncio
    async def test_mean_factors_change_the_server_pick(self):
        # ep10 first in ledger order, 10% over the active mean (22 of [22, 8, 30]): needs factor > 1.1.
        inst = _instance(1, [_endpoint(10), _endpoint(11), _endpoint(12)])
        ledger = {10: 22, 11: 8, 12: 30}
        candidates = [_cand_payload(1, ep, float(ep)) for ep in ledger]

        dispatcher, im, _ = await _dispatcher([inst])
        for ep, active in ledger.items():
            await im.update_instance_workload(1, ep, Workload(active_tokens=active, prefill_cost=ep))
        strict = await dispatcher.dispatch(_allocate(1, 10, candidates))
        assert strict.data["endpoint"]["id"] == 11

        dispatcher, im, _ = await _dispatcher([_instance(1, [_endpoint(10), _endpoint(11), _endpoint(12)])], 1.2, 1.0)
        for ep, active in ledger.items():
            await im.update_instance_workload(1, ep, Workload(active_tokens=active, prefill_cost=ep))
        loose = await dispatcher.dispatch(_allocate(1, 10, candidates))
        assert loose.data["endpoint"]["id"] == 10

    @pytest.mark.asyncio
    async def test_cpu_factor_on_server(self):
        # Both under the active mean; ep10 over the cpu mean at 1.0, under it at 2.0.
        def build():
            return _instance(1, [_endpoint(10), _endpoint(11), _endpoint(12)])

        ledger = {10: (5, 15), 11: (5, 5), 12: (50, 10)}
        candidates = [_cand_payload(1, ep, float(ep)) for ep in ledger]

        dispatcher, im, _ = await _dispatcher([build()])
        for ep, (active, cpu) in ledger.items():
            await im.update_instance_workload(
                1, ep, Workload(active_tokens=active, cpu_hit_blocks=cpu, prefill_cost=ep)
            )
        assert (await dispatcher.dispatch(_allocate(1, 10, candidates))).data["endpoint"]["id"] == 11

        dispatcher, im, _ = await _dispatcher([build()], 1.0, 2.0)
        for ep, (active, cpu) in ledger.items():
            await im.update_instance_workload(
                1, ep, Workload(active_tokens=active, cpu_hit_blocks=cpu, prefill_cost=ep)
            )
        assert (await dispatcher.dispatch(_allocate(1, 10, candidates))).data["endpoint"]["id"] == 10

    @pytest.mark.asyncio
    async def test_release_subtracts_cpu_hits_from_ledger(self):
        inst = _instance(1, [_endpoint(10)])
        dispatcher, im, _ = await _dispatcher([inst])
        response = await dispatcher.dispatch(_allocate(1, 10, [_cand_payload(1, 10, 40.0, cpu=9.0)]))
        committed = Workload.model_validate(response.data["committed_workload"])
        _, ledger = await im.get_endpoint_workload(1, 10)
        assert ledger.cpu_hit_blocks == 9.0

        release = SchedulerRequest(
            request_type=SchedulerRequestType.UPDATE_WORKLOAD,
            request_id="rel",
            data={
                "instance_id": 1,
                "endpoint_id": 10,
                "role": "prefill",
                "req_id": "req-gated",
                "workload_action": "Release_Tokens",
                "workload_change": Workload(
                    active_tokens=-committed.active_tokens,
                    prefill_cost=-committed.prefill_cost,
                    cpu_hit_blocks=-committed.cpu_hit_blocks,
                ).model_dump(mode="json"),
            },
        )
        await dispatcher.dispatch(release)
        _, ledger = await im.get_endpoint_workload(1, 10)
        assert (ledger.active_tokens, ledger.prefill_cost, ledger.cpu_hit_blocks) == (0.0, 0.0, 0.0)

    @pytest.mark.asyncio
    async def test_no_costs_validates_worker_candidate(self):
        inst = _instance(1, [_endpoint(10)])
        dispatcher, _im, _ = await _dispatcher([inst])
        response = await dispatcher.dispatch(_allocate(1, 10, None))
        assert response.data["endpoint"]["id"] == 10
        assert response.data["committed_workload"]["cpu_hit_blocks"] == 0.0

    @pytest.mark.asyncio
    async def test_circuit_open_and_unknown_endpoints_are_skipped(self):
        inst_a = _instance(1, [_endpoint(10)])
        inst_b = _instance(2, [_endpoint(20)])
        dispatcher, _im, _ = await _dispatcher([inst_a, inst_b])
        dispatcher._cb_manager.is_open = lambda iid: iid == 1

        response = await dispatcher.dispatch(
            _allocate(1, 10, [_cand_payload(1, 10, 1.0), _cand_payload(9, 99, 2.0), _cand_payload(2, 20, 3.0)])
        )

        assert response.data["instance"]["id"] == 2

    @pytest.mark.asyncio
    async def test_engine_type_constraint(self):
        inst_a = _instance(1, [_endpoint(10)])
        inst_a.engine_type = "vllm"
        inst_b = _instance(2, [_endpoint(20)])
        inst_b.engine_type = "sglang"
        dispatcher, _im, _ = await _dispatcher([inst_a, inst_b])

        response = await dispatcher.dispatch(
            _allocate(1, 10, [_cand_payload(1, 10, 1.0), _cand_payload(2, 20, 3.0)], required_engine_type="sglang")
        )

        assert response.data["instance"]["id"] == 2

    @pytest.mark.asyncio
    async def test_all_candidates_unavailable_returns_empty(self):
        inst = _instance(1, [_endpoint(10)], role=PDRole.ROLE_D)
        dispatcher, _im, _ = await _dispatcher([inst])
        response = await dispatcher.dispatch(_allocate(1, 10, [_cand_payload(1, 10, 1.0)]))
        assert response.data["instance"] is None
