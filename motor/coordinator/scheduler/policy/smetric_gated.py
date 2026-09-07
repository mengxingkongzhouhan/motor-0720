# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
SMetric-gated scheduling policy: rank endpoints by their ledger, gate on two ledger averages.

1. Sort the endpoints of the request's role by the ledger ``workload.prefill_cost`` ascending,
   i.e. the remaining prefill currently outstanding on each endpoint (sum over its in-flight
   requests of ``isl - matched_tokens``).
2. Walk that order and commit the first endpoint whose ledger is strictly below BOTH averages
   over the ranked endpoints: ``active_tokens < mean(active_tokens)`` and
   ``cpu_hit_blocks < mean(cpu_hit_blocks)``.

All three inputs are ledger fields, so the ranking itself needs no per-request affinity math.
The KV Conductor is queried once per request only to know what to ADD to the committed
endpoint's ledger: the request's own remaining prefill (SMetric cost model,
``max(0, isl - matched_tokens)``) and the CPU-tier KV blocks it would pull there
(``cpu_blocks``). RELEASE subtracts both again, so ``prefill_cost`` / ``cpu_hit_blocks`` track
the prefill compute and CPU->NPU KV transfer currently in flight per endpoint.

When no endpoint passes both gates the policy degrades in order: first endpoint passing the
``active_tokens`` gate alone, then the head of the list (lowest ledger prefill_cost).
"""

from __future__ import annotations

from dataclasses import dataclass

from motor.common.logger import get_logger
from motor.common.resources.endpoint import Endpoint
from motor.common.resources.instance import Instance, PDRole
from motor.coordinator.api_client.conductor_api_client import (
    TENANT_ID,
    ConductorApiClient,
    conductor_instance_id,
)
from motor.coordinator.domain import InstanceProvider
from motor.coordinator.models.constants import DEFAULT_REQUEST_ID
from motor.coordinator.models.request import RequestInfo
from motor.coordinator.scheduler.policy.base import BaseSchedulingPolicy, WorkloadLedgerMixin
from motor.coordinator.scheduler.policy.smetric import _matched_tokens, _prefill_cost, _prompt_token_ids

logger = get_logger(__name__)

# Roles that do prefill, i.e. whose allocations have a conductor cost and CPU hit count.
SMETRIC_GATED_ROLES = frozenset({PDRole.ROLE_P, PDRole.ROLE_U})

# How the final endpoint was chosen (returned for logging / tests).
PICK_BOTH_GATES = "both_gates"
PICK_ACTIVE_GATE = "active_gate"
PICK_MIN_LEDGER_PREFILL = "min_ledger_prefill"


@dataclass(frozen=True)
class GatedCandidate:
    """
    One endpoint of the request's role.

    ``prefill_cost`` / ``cpu_hit_blocks`` are what THIS request would add to the endpoint ledger
    if committed there (conductor-derived); they are stamped on allocation and never used for
    ordering. Ordering and gating read the endpoint's current ledger via ``endpoint.workload``.
    """

    instance: Instance
    endpoint: Endpoint
    prefill_cost: float
    cpu_hit_blocks: float

    @property
    def key(self) -> tuple[int, int]:
        return (self.instance.id, self.endpoint.id)

    @property
    def ledger_prefill_cost(self) -> float:
        return _ledger_value(self.endpoint, "prefill_cost")

    @property
    def ledger_active_tokens(self) -> float:
        return _ledger_value(self.endpoint, "active_tokens")

    @property
    def ledger_cpu_hit_blocks(self) -> float:
        return _ledger_value(self.endpoint, "cpu_hit_blocks")


def _cpu_hit_blocks(matched: object) -> float:
    """CPU-tier matched blocks from a DpBlocks conductor entry; 0 for legacy integer matches."""
    if not isinstance(matched, dict):
        return 0.0
    try:
        return max(0.0, float(matched.get("cpu_blocks", 0) or 0))
    except (TypeError, ValueError):
        return 0.0


def _ledger_value(endpoint: Endpoint, field: str) -> float:
    try:
        return max(0.0, float(getattr(endpoint.workload, field, 0.0) or 0.0))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def sort_candidates(candidates: list[GatedCandidate]) -> list[GatedCandidate]:
    """Lowest ledger ``workload.prefill_cost`` first, ties by (instance_id, endpoint_id)."""
    return sorted(candidates, key=lambda c: (c.ledger_prefill_cost, c.instance.id, c.endpoint.id))


def pick_gated(candidates: list[GatedCandidate]) -> tuple[GatedCandidate, str, float, float] | None:
    """
    Walk ``candidates`` (already in ledger prefill_cost order) and return the first one whose
    ledger is strictly below both averages, plus the pick reason and the two averages used.

    Averages are taken over the candidates' current ledgers (``endpoint.workload``), so the
    caller decides which view is authoritative (worker SHM cache vs scheduler ledger).
    Fallback order when nothing passes both gates: active_tokens gate only, then the head of the
    list (lowest ledger prefill_cost).
    """
    if not candidates:
        return None
    n = len(candidates)
    mean_active = sum(c.ledger_active_tokens for c in candidates) / n
    mean_cpu = sum(c.ledger_cpu_hit_blocks for c in candidates) / n
    active_only: GatedCandidate | None = None
    for cand in candidates:
        under_active = cand.ledger_active_tokens < mean_active
        under_cpu = cand.ledger_cpu_hit_blocks < mean_cpu
        if under_active and under_cpu:
            return (cand, PICK_BOTH_GATES, mean_active, mean_cpu)
        if under_active and active_only is None:
            active_only = cand
    if active_only is not None:
        return (active_only, PICK_ACTIVE_GATE, mean_active, mean_cpu)
    return (candidates[0], PICK_MIN_LEDGER_PREFILL, mean_active, mean_cpu)


def format_candidates(candidates: list[GatedCandidate]) -> str:
    """``ins-ep:ledger_prefill/active/cpu(+req_cost/+req_cpu)`` per candidate, for the selection log."""
    return " ".join(
        f"{c.instance.id}-{c.endpoint.id}:{c.ledger_prefill_cost:.0f}/{c.ledger_active_tokens:.0f}/"
        f"{c.ledger_cpu_hit_blocks:.0f}(+{c.prefill_cost:.0f}/+{c.cpu_hit_blocks:.0f})"
        for c in candidates
    )


class SMetricGatedPolicy(WorkloadLedgerMixin, BaseSchedulingPolicy):
    """
    Rank by ledger prefill_cost, commit the first endpoint under both ledger load averages.

    Workers run the conductor query (for the stamp values) and forward every endpoint with its
    request cost + cpu_blocks to the central Scheduler, which re-ranks and re-gates against its
    authoritative ledger before committing; the worker's own pick is only a proposal.
    """

    def __init__(self, instance_provider: InstanceProvider):
        super().__init__(instance_provider=instance_provider)
        logger.info("SMetricGatedPolicy started.")

    @staticmethod
    def score_endpoints(instances: list[Instance], req_info: RequestInfo) -> list[GatedCandidate] | None:
        """
        Conductor lookup: every endpoint as a ``GatedCandidate``, sorted by its ledger prefill_cost.

        The conductor only supplies the per-endpoint stamp values (request cost, cpu_blocks).
        ``None`` means it had no data for our instances (caller falls back). Also caches
        ``{(instance_id, endpoint_id): (prefill_cost, cpu_hit_blocks)}`` on
        ``req_info.smetric_gated_debug`` for the ALLOCATE payload and the ledger stamp.
        """
        encoded_ids = _prompt_token_ids(req_info)
        isl = len(encoded_ids)
        rsp = ConductorApiClient.query_conductor(instances, encoded_ids)
        req_id = getattr(req_info, "req_id", None) or DEFAULT_REQUEST_ID
        logger.debug("smetric_gated: req_id=%s conductor_rsp=%s", req_id, rsp)
        tenant = rsp.get(TENANT_ID, None) if isinstance(rsp, dict) else None
        if tenant is None:
            logger.warning(
                "smetric_gated: conductor query returned no tenant data (tenant_id=%s, instances=%d)",
                TENANT_ID,
                len(instances),
            )
            return None

        candidates: list[GatedCandidate] = []
        any_instance = False
        for instance in instances:
            instance_data = tenant.get(conductor_instance_id(instance), None)
            if instance_data is None:
                continue
            any_instance = True
            dp_map = instance_data.get("DP", {}) if isinstance(instance_data, dict) else {}
            for ep in instance.get_all_endpoints():
                matched = dp_map.get(f"{ep.id}", 0)
                candidates.append(
                    GatedCandidate(
                        instance=instance,
                        endpoint=ep,
                        prefill_cost=_prefill_cost(isl, _matched_tokens(matched)),
                        cpu_hit_blocks=_cpu_hit_blocks(matched),
                    )
                )
        if not any_instance:
            logger.warning("smetric_gated: no instance data")
            return None
        if not candidates:
            logger.warning("smetric_gated: no endpoint scored")
            return None

        ranked = sort_candidates(candidates)
        req_info.smetric_gated_debug = {c.key: (c.prefill_cost, c.cpu_hit_blocks) for c in ranked}
        logger.info(
            "smetric_gated: req_id=%s isl=%s ranked[ins-ep:ledger_prefill/active/cpu(+req_cost/+req_cpu)]=%s",
            req_id,
            isl,
            format_candidates(ranked),
        )
        return ranked

    @staticmethod
    def select_endpoint_candidates_from_list(
        instances: list[Instance],
        req_info: RequestInfo,
        top_k: int = 1,
    ) -> list[tuple[Instance, Endpoint, float]] | None:
        """
        Worker-side proposal: the gated pick first, then the rest in ledger prefill_cost order.

        The ledger view here is the worker's SHM cache (it carries prefill_cost and active_tokens
        but not ``cpu_hit_blocks``, so only the active_tokens gate can bite on the worker); the
        Scheduler re-ranks and re-gates on its own ledger.
        """
        ranked = SMetricGatedPolicy.score_endpoints(instances, req_info)
        if not ranked:
            return None
        picked = pick_gated(ranked)
        if picked is None:
            return None
        chosen, reason, mean_active, mean_cpu = picked
        logger.debug(
            "smetric_gated(worker): req_id=%s pick=%s-%s reason=%s mean_active=%.1f mean_cpu=%.1f",
            getattr(req_info, "req_id", None) or DEFAULT_REQUEST_ID,
            chosen.instance.id,
            chosen.endpoint.id,
            reason,
            mean_active,
            mean_cpu,
        )
        ordered = [chosen] + [c for c in ranked if c is not chosen]
        return [(c.instance, c.endpoint, c.ledger_prefill_cost) for c in ordered[: max(1, top_k)]]

    @staticmethod
    def select_endpoint_from_list(
        instances: list[Instance],
        req_info: RequestInfo,
    ) -> tuple[Instance, Endpoint] | None:
        ranked = SMetricGatedPolicy.select_endpoint_candidates_from_list(instances, req_info, top_k=1)
        if not ranked:
            return None
        instance, endpoint, _cost = ranked[0]
        return (instance, endpoint)

    def _select_instance(self, _: PDRole = None) -> Instance | None:
        return None

    def _select_endpoint(self, _: Instance) -> Endpoint | None:
        return None

    def select_instance_and_endpoint_from_list(
        self,
        instances: list[Instance],
        role: PDRole | None = None,
        req_info: RequestInfo | None = None,
    ):
        if role in SMETRIC_GATED_ROLES and req_info is not None:
            selected = SMetricGatedPolicy.select_endpoint_from_list(instances, req_info)
            if selected is not None:
                return selected
        from motor.coordinator.scheduler.policy.load_balance import LoadBalancePolicy

        return LoadBalancePolicy.select_endpoint_from_list(instances, role)
