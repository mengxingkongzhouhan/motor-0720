# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""SMetric scheduling policy: rank endpoints by prefill_cost only (lower is better)."""

from __future__ import annotations

import threading

from motor.common.logger import get_logger
from motor.common.resources.endpoint import Endpoint
from motor.common.resources.instance import Instance, PDRole
from motor.coordinator.api_client.conductor_api_client import (
    TENANT_ID,
    ConductorApiClient,
    conductor_instance_id,
)
from motor.coordinator.domain import InstanceProvider
from motor.coordinator.models.constants import DEFAULT_REQUEST_ID, OpenAIField
from motor.coordinator.models.request import RequestInfo
from motor.coordinator.scheduler.policy.base import (
    BaseSchedulingPolicy,
    WorkloadLedgerMixin,
)
from motor.coordinator.scheduler.tokenizer import TokenizerManager

logger = get_logger(__name__)

# SMetric discounts a cached prefix 1:1 against prompt length. Not configurable; not shared with
# kv_cache_affinity's overlap_credit knob.
_SMETRIC_OVERLAP_CREDIT = 1
_LOAD_FACTOR = 2
# Remaining-prefill / prompt-length gate. Above this, min-cost ranking is worth it; otherwise
# pick the endpoint whose ledger ``workload.prefill_cost`` is currently smallest.
_SMETRIC_COST_ISL_RATIO = 0.5


def _prompt_token_ids(req_info: RequestInfo) -> list[int]:
    """Tokenize the prompt for the conductor query and isl. Independent of KV-affinity helpers."""
    cached = getattr(req_info, "token_ids", None)
    if isinstance(cached, list) and cached:
        return cached
    encoded_ids: list[int] = []
    req_data = getattr(req_info, "req_data", None) or {}
    messages = req_data.get(OpenAIField.MESSAGES, None)
    tools = req_data.get(OpenAIField.TOOLS, None)
    if messages is not None:
        encoded_ids = TokenizerManager().apply_chat_template(messages, tools, req_data=req_data)
    else:
        prompt = req_data.get(OpenAIField.PROMPT, None)
        if prompt is not None:
            encoded_ids = TokenizerManager().encode(prompt)
    req_info.token_ids = encoded_ids
    return encoded_ids


def _prefill_cost(isl: int, matched_tokens: int) -> float:
    """Remaining prefill tokens with overlap_credit fixed at 1: max(0, isl - matched)."""
    matched = max(0, min(matched_tokens, isl)) if isl > 0 else 0
    return float(max(0, isl - _SMETRIC_OVERLAP_CREDIT * matched))


def _matched_tokens(matched: object) -> int:
    """Read both legacy integer and DpBlocks conductor match formats."""
    if isinstance(matched, dict):
        matched = matched.get("matched_tokens", 0)
    try:
        return max(0, int(matched or 0))
    except (TypeError, ValueError):
        return 0


class SMetricPrefillCostTracker:
    """
    Running average of the prefill_cost actually committed on each allocation.

    Owned by the central Scheduler (one process). Workers only report per-endpoint costs; they
    must not keep their own copy, or each Worker would count a different subset of traffic.
    """

    def __init__(self) -> None:
        self._avg = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._avg = 0.0
            self._count = 0

    def snapshot(self) -> tuple[float, int]:
        """Return ``(average, sample_count)``. Average is 0.0 when no samples."""
        with self._lock:
            return self._avg, self._count

    def record(self, req_cost: float) -> None:
        with self._lock:
            self._count += 1
            self._avg += (req_cost - self._avg) / self._count

    def use_smetric_rank(self, req_cost: float, isl: float) -> bool:
        """True when this request should pick the lowest-cost endpoint instead of dump."""

        with self._lock:
            count = self._count
            avg = self._avg
        if count > 0 and req_cost > avg * _LOAD_FACTOR:
            logger.info(
                "smetric: prefill_cost=%s > avg=%.3f (n=%d), using min ledger prefill_cost",
                req_cost,
                avg,
                count,
            )
            return False
        if isl <= 0:
            logger.info("smetric: isl=%s, using min ledger prefill_cost", isl)
            return False
        ratio = (isl - req_cost) / isl
        if ratio > _SMETRIC_COST_ISL_RATIO:
            return True
        logger.info(
            "smetric: prefill_cost/isl=%.3f <= %s (cost=%s isl=%s), using min ledger prefill_cost",
            ratio,
            _SMETRIC_COST_ISL_RATIO,
            req_cost,
            isl,
        )
        return False


class SMetricPolicy(WorkloadLedgerMixin, BaseSchedulingPolicy):
    """
    Score every reported endpoint by remaining prefill; the lowest cost is the SMetric candidate.

    ``prefill_cost = max(0, isl - matched_tokens)`` (overlap_credit is always 1). Workers compute
    and rank by these costs, then forward every endpoint cost plus the min-cost top-1 and ``isl``.
    The central Scheduler keeps a running average of the **allocated** endpoint's prefill_cost
    and uses that, plus the cached-prefix ratio ``(isl - cost) / isl``, to decide min-cost ranking
    vs picking the endpoint with the lowest current ``workload.prefill_cost``. When the gate keeps
    SMetric and the worker's workload view is fresh, the worker top-1 is committed as-is.

    Conductor lookup and ``smetric_debug`` are owned here; no other policy is called.
    """

    def __init__(self, instance_provider: InstanceProvider):
        super().__init__(instance_provider=instance_provider)
        logger.info("SMetricPolicy started.")

    @staticmethod
    def select_endpoint_candidates_from_list(
        instances: list[Instance],
        req_info: RequestInfo,
        top_k: int = 1,
    ) -> list[tuple[Instance, Endpoint, float]] | None:
        """
        Return up to ``top_k`` ``(instance, endpoint, prefill_cost)`` tuples, lowest cost first.

        ``None`` means no conductor data; the caller should fall back. Hybrid average / ratio
        gating is applied later by the central Scheduler, not here.
        """
        encoded_ids = _prompt_token_ids(req_info)
        isl = len(encoded_ids)
        rsp = ConductorApiClient.query_conductor(instances, encoded_ids)
        logger.info(
            "smetric: req_id=%s conductor_rsp=%s",
            getattr(req_info, "req_id", None) or DEFAULT_REQUEST_ID,
            rsp,
        )
        tenant = rsp.get(TENANT_ID, None) if isinstance(rsp, dict) else None
        if tenant is None:
            logger.warning(
                "smetric: conductor query returned no tenant data (tenant_id=%s, instances=%d)",
                TENANT_ID,
                len(instances),
            )
            return None

        scored: list[tuple[float, int, int, Instance, Endpoint]] = []
        matches: list[tuple[int, int, int]] = []
        any_instance = False
        for instance in instances:
            instance_data = tenant.get(conductor_instance_id(instance), None)
            if instance_data is None:
                continue
            any_instance = True
            dp_map = instance_data.get("DP", {}) if isinstance(instance_data, dict) else {}
            for ep in instance.get_all_endpoints():
                matched = dp_map.get(f"{ep.id}", 0)
                matched_tokens = _matched_tokens(matched)
                cost = _prefill_cost(isl, matched_tokens)
                matches.append((instance.id, ep.id, matched_tokens))
                scored.append((cost, instance.id, ep.id, instance, ep))

        if not any_instance:
            logger.warning("smetric: no instance data")
            return None
        if not scored:
            logger.warning("smetric: no endpoint selected")
            return None

        matches.sort()
        logger.info(
            "smetric: req_id=%s isl=%s endpoint_matches=[%s]",
            getattr(req_info, "req_id", None) or DEFAULT_REQUEST_ID,
            isl,
            ", ".join(f"{iid}-{eid}:{matched}" for iid, eid, matched in matches),
        )

        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        SMetricPolicy._stash_debug(req_info, scored)
        ranked = scored[: max(1, top_k)]
        top_cost, _iid, _eid, top_inst, top_ep = ranked[0]
        logger.debug(
            "select_endpoint(smetric): role=%s %s-%s prefill_cost=%s (top%d of %d)",
            top_inst.role,
            top_inst.id,
            top_ep.id,
            top_cost,
            len(ranked),
            len(scored),
        )
        return [(inst, ep, float(cost)) for (cost, _iid, _eid, inst, ep) in ranked]

    @staticmethod
    def select_endpoint_from_list(
        instances: list[Instance],
        req_info: RequestInfo,
    ) -> tuple[Instance, Endpoint] | None:
        ranked = SMetricPolicy.select_endpoint_candidates_from_list(instances, req_info, top_k=1)
        if not ranked:
            return None
        instance, endpoint, _score = ranked[0]
        return (instance, endpoint)

    @staticmethod
    def _stash_debug(
        req_info: RequestInfo | None,
        scored: list[tuple[float, int, int, Instance, Endpoint]],
    ) -> None:
        """Cache per-endpoint prefill_cost on ``req_info.smetric_debug``. Never fail selection."""
        if req_info is None:
            return
        req_info.smetric_debug = {
            (instance.id, ep.id): cost for (cost, _iid, _eid, instance, ep) in scored
        }

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
        if role in (PDRole.ROLE_P, PDRole.ROLE_U) and req_info is not None:
            return SMetricPolicy.select_endpoint_from_list(instances, req_info)
        return None
