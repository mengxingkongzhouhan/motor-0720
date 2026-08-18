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
    ConductorApiClient,
    TENANT_ID,
    conductor_instance_id,
)
from motor.coordinator.domain import InstanceProvider
from motor.coordinator.models.constants import DEFAULT_REQUEST_ID, OpenAIField
from motor.coordinator.models.request import RequestInfo
from motor.coordinator.scheduler.policy.base import BaseSchedulingPolicy, WorkloadLedgerMixin

logger = get_logger(__name__)

# SMetric discounts a cached prefix 1:1 against prompt length. Not configurable; not shared with
# kv_cache_affinity's overlap_credit knob.
_SMETRIC_OVERLAP_CREDIT = 1
# Remaining-prefill / prompt-length gate. Above this, min-cost ranking is worth it; otherwise
# pick the endpoint whose ledger ``workload.prefill_cost`` is currently smallest.
_SMETRIC_COST_ISL_RATIO = 0.5


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


def _prefill_cost(isl: int, matched_tokens: int) -> float:
    """Remaining prefill tokens with overlap_credit fixed at 1: max(0, isl - matched)."""
    matched = min(matched_tokens, isl) if isl > 0 else 0
    return float(max(0, isl - _SMETRIC_OVERLAP_CREDIT * matched))


def _conductor_block_size() -> int:
    """KV block size used to turn conductor ``*_blocks`` into token counts. 0 if unknown."""
    try:
        bs = int(ConductorApiClient.coordinator_config.scheduler_config.kv_conductor_config.block_size)
        return bs if bs > 0 else 0
    except Exception as e:  # pragma: no cover - config shape guard
        logger.debug("Could not read conductor block_size: %s", e)
        return 0


def _as_nonneg_int(value: object) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _matched_hit_lengths(
    matched_raw: object,
    isl: int,
    block_size: int,
) -> tuple[int, int | None, int | None]:
    """
    Parse one conductor DP match into ``(matched, local_hit, remote_hit)``.

    * ``local_hit`` is NPU/HBM prefix tokens (``npu_blocks * block_size``).
    * ``remote_hit`` is CPU+Disk tokens (external KV store).
    * Scalar / ``matched_tokens``-only replies leave local/remote as None.
    """
    local_hit: int | None = None
    remote_hit: int | None = None
    matched = 0
    if isinstance(matched_raw, dict):
        has_blocks = any(key in matched_raw for key in ("npu_blocks", "cpu_blocks", "disk_blocks"))
        if has_blocks and block_size > 0:
            local_hit = _as_nonneg_int(matched_raw.get("npu_blocks")) * block_size
            remote_hit = (
                _as_nonneg_int(matched_raw.get("cpu_blocks")) + _as_nonneg_int(matched_raw.get("disk_blocks"))
            ) * block_size
            matched = _as_nonneg_int(matched_raw.get("matched_tokens")) or (local_hit + remote_hit)
        else:
            matched = _as_nonneg_int(matched_raw.get("matched_tokens"))
    else:
        matched = _as_nonneg_int(matched_raw)
    if isl > 0:
        matched = min(matched, isl)
        if local_hit is not None:
            local_hit = min(local_hit, isl)
        if remote_hit is not None:
            remote_hit = min(remote_hit, isl)
    return matched, local_hit, remote_hit


def _format_hit(value: int | None) -> str:
    return "-" if value is None else str(value)


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
        if count > 0 and req_cost > avg:
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
        ratio = req_cost / isl
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
    and uses that, plus ``cost/isl``, to decide min-cost ranking vs picking the endpoint with the
    lowest current ``workload.prefill_cost``. When the gate keeps SMetric and the worker's
    workload view is fresh, the worker top-1 is committed as-is.

    Conductor lookup and ``smetric_debug`` are owned here; KvCacheAffinityPolicy is not called.
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
            "smetric: req_id=%s isl=%s conductor_rsp=%s",
            getattr(req_info, "req_id", None) or DEFAULT_REQUEST_ID,
            isl,
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
        matches: list[tuple[int, int, int, int | None, int | None]] = []
        any_instance = False
        block_size = _conductor_block_size()
        for instance in instances:
            instance_data = tenant.get(conductor_instance_id(instance), None)
            if instance_data is None:
                continue
            any_instance = True
            dp_map = instance_data.get("DP", {}) if isinstance(instance_data, dict) else {}
            for ep in instance.get_all_endpoints():
                matched_tokens, local_hit, remote_hit = _matched_hit_lengths(
                    dp_map.get(f"{ep.id}", 0),
                    isl,
                    block_size,
                )
                cost = _prefill_cost(isl, matched_tokens)
                matches.append((instance.id, ep.id, matched_tokens, local_hit, remote_hit))
                scored.append((cost, instance.id, ep.id, instance, ep))

        if not any_instance:
            logger.warning("smetric: no instance data")
            return None
        if not scored:
            logger.warning("smetric: no endpoint selected")
            return None

        matches.sort()
        req_id = getattr(req_info, "req_id", None) or DEFAULT_REQUEST_ID
        logger.info(
            "smetric: req_id=%s isl=%s endpoint_matches=[%s]",
            req_id,
            isl,
            ", ".join(
                f"{iid}-{eid}:matched={matched}/local={_format_hit(local_hit)}/remote={_format_hit(remote_hit)}"
                for iid, eid, matched, local_hit, remote_hit in matches
            ),
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
        try:
            req_info.smetric_debug = {(instance.id, ep.id): cost for (cost, _iid, _eid, instance, ep) in scored}
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
