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

from motor.common.logger import get_logger
from motor.common.resources.endpoint import Endpoint
from motor.common.resources.instance import Instance, PDRole
from motor.coordinator.api_client.conductor_api_client import (
    ConductorApiClient,
    TENANT_ID,
    conductor_instance_id,
)
from motor.coordinator.domain import InstanceProvider
from motor.coordinator.models.constants import OpenAIField
from motor.coordinator.models.request import RequestInfo
from motor.coordinator.scheduler.policy.base import BaseSchedulingPolicy, WorkloadLedgerMixin

logger = get_logger(__name__)

# SMetric discounts a cached prefix 1:1 against prompt length. Not configurable; not shared with
# kv_cache_affinity's overlap_credit knob.
_SMETRIC_OVERLAP_CREDIT = 1


def _prompt_token_ids(req_info: RequestInfo) -> list[int]:
    """Tokenize the prompt for the conductor query and isl. Independent of KV-affinity helpers."""
    cached = getattr(req_info, "token_ids", None)
    if isinstance(cached, list):
        return cached
    encoded_ids: list[int] = []
    req_data = getattr(req_info, "req_data", None) or {}
    messages = req_data.get(OpenAIField.MESSAGES, None)
    tools = req_data.get(OpenAIField.TOOLS, None)
    # TokenizerManager is the process-wide tokenizer singleton (lives next to KV affinity for
    # historical reasons). SMetric only uses it to encode the prompt; it does not call any
    # KvCacheAffinityPolicy ranking/stash helpers.
    from motor.coordinator.scheduler.policy.kv_cache_affinity import TokenizerManager

    if messages is not None:
        encoded_ids = TokenizerManager().apply_chat_template(messages, tools, req_data=req_data)
    else:
        prompt = req_data.get(OpenAIField.PROMPT, None)
        if prompt is not None:
            encoded_ids = TokenizerManager().encode(prompt)
    try:
        req_info.token_ids = encoded_ids
    except Exception as e:  # pragma: no cover - req_info may be immutable in some callers
        logger.debug("Could not cache token_ids on req_info: %s", e)
    return encoded_ids


def _prefill_cost(isl: int, matched_tokens: int) -> int:
    """Remaining prefill tokens with overlap_credit fixed at 1: max(0, isl - matched)."""
    matched = min(matched_tokens, isl) if isl > 0 else 0
    return max(0, isl - _SMETRIC_OVERLAP_CREDIT * matched)


class SMetricPolicy(WorkloadLedgerMixin, BaseSchedulingPolicy):
    """
    Rank every reported endpoint by SMetric prefill_cost; the lowest cost wins.

    ``prefill_cost = max(0, isl - matched_tokens)`` (overlap_credit is always 1). Load is not part
    of the score. Selection, conductor lookup, and the ``smetric_debug`` cache are owned by this
    class and do not call KvCacheAffinityPolicy.
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

        ``None`` means the caller should fall back (no conductor data). An empty-but-not-None list
        is not used; no endpoints after a successful tenant match logs and returns None.
        """
        encoded_ids = _prompt_token_ids(req_info)
        isl = len(encoded_ids)
        rsp = ConductorApiClient.query_conductor(instances, encoded_ids)
        tenant = rsp.get(TENANT_ID, None) if isinstance(rsp, dict) else None
        if tenant is None:
            logger.warning(
                "smetric: conductor query returned no tenant data (tenant_id=%s, instances=%d)",
                TENANT_ID,
                len(instances),
            )
            return None

        scored: list[tuple[int, int, int, Instance, Endpoint]] = []
        any_instance = False
        for instance in instances:
            instance_data = tenant.get(conductor_instance_id(instance), None)
            if instance_data is None:
                continue
            any_instance = True
            dp_map = instance_data.get("DP", {}) if isinstance(instance_data, dict) else {}
            for ep in instance.get_all_endpoints():
                matched = dp_map.get(f"{ep.id}", 0)
                try:
                    matched_tokens = int(matched)
                except (TypeError, ValueError):
                    matched_tokens = 0
                cost = _prefill_cost(isl, matched_tokens)
                scored.append((cost, instance.id, ep.id, instance, ep))

        if not any_instance:
            logger.warning("smetric: no instance data")
            return None
        if not scored:
            logger.warning("smetric: no endpoint selected")
            return None

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
        scored: list[tuple[int, int, int, Instance, Endpoint]],
    ) -> None:
        """Cache per-endpoint prefill_cost on ``req_info.smetric_debug``. Never fail selection."""
        if req_info is None:
            return
        try:
            req_info.smetric_debug = {
                (instance.id, ep.id): cost for (cost, _iid, _eid, instance, ep) in scored
            }
        except Exception as e:  # pragma: no cover
            logger.debug("Could not cache smetric_debug on req_info: %s", e)

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
            selected = SMetricPolicy.select_endpoint_from_list(instances, req_info)
            if selected is not None:
                return selected
        from motor.coordinator.scheduler.policy.load_balance import LoadBalancePolicy

        return LoadBalancePolicy.select_endpoint_from_list(instances, role)
