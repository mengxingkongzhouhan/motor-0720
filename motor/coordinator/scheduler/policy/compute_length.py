# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Hierarchical compute-length scheduling: min instance, then min DP.

In-flight load is the same ``active_tokens`` ledger used by KV affinity
(per-DP endpoint workload, instance ``gathered_workload`` as the sum).

The current request's increment is ``ISL - max_matched_tokens`` across *all*
reported DPs -- not a per-DP match. Selection never prefers a hotter prefix;
it only picks the lightest instance, then the lightest DP inside it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from motor.common.logger import get_logger
from motor.common.resources.endpoint import Endpoint
from motor.common.resources.instance import Instance, PDRole
from motor.coordinator.api_client.conductor_api_client import (
    TENANT_ID,
    ConductorApiClient,
    conductor_instance_id,
)
from motor.coordinator.domain import InstanceProvider
from motor.coordinator.models.request import RequestInfo
from motor.coordinator.scheduler.policy.base import BaseSchedulingPolicy
from motor.coordinator.scheduler.policy.kv_cache_affinity import KvCacheAffinityPolicy

logger = get_logger(__name__)


def _score_role(instance: Instance, role: PDRole | None) -> PDRole | str | None:
    return role if role is not None else instance.role


def instance_compute_load(instance: Instance, role: PDRole | None = None) -> float:
    """Instance-level running compute: gathered_workload, else the sum of DP ledgers."""
    score_role = _score_role(instance, role)
    gathered = getattr(instance, "gathered_workload", None)
    if gathered is not None:
        try:
            return float(gathered.calculate_workload_score(role=score_role))
        except Exception as exc:
            logger.warning(
                "Failed to read gathered compute load for instance %s: %s",
                getattr(instance, "id", None),
                exc,
            )
    total = 0.0
    for endpoint in instance.get_all_endpoints():
        try:
            total += float(endpoint.workload.calculate_workload_score(role=score_role))
        except Exception as exc:
            logger.warning(
                "Failed to read DP compute load for instance %s endpoint %s: %s",
                getattr(instance, "id", None),
                getattr(endpoint, "id", None),
                exc,
            )
    return total


def endpoint_compute_load(instance: Instance, endpoint: Endpoint, role: PDRole | None = None) -> float:
    """Per-DP running compute (``active_tokens``)."""
    return float(endpoint.workload.calculate_workload_score(role=_score_role(instance, role)))


class ComputeLengthPolicy(BaseSchedulingPolicy):
    """Pick the least-loaded instance, then the least-loaded DP; commit ISL - max match."""

    def __init__(self, instance_provider: InstanceProvider):
        super().__init__(instance_provider=instance_provider)
        logger.info("ComputeLengthPolicy started.")

    @staticmethod
    def select_instance_then_endpoint(
        instances: list[Instance] | Iterable[Instance],
        role: PDRole | None = None,
        *,
        is_blocked: Callable[[int], bool] | None = None,
        excluded_pairs: set[tuple[int, int]] | None = None,
    ) -> tuple[Instance, Endpoint, float] | None:
        """
        Hierarchical min-load pick: lightest instance, then lightest DP on that instance.

        ``is_blocked`` skips circuit-open instances. ``excluded_pairs`` drops
        ``(instance_id, endpoint_id)`` already rejected in this CAS round; an
        instance with no remaining DPs is skipped.
        """
        if not isinstance(instances, (list, tuple)):
            instances = list(instances)
        if not instances:
            return None

        best_instance: Instance | None = None
        best_instance_load = float("inf")
        for instance in instances:
            if is_blocked is not None and is_blocked(instance.id):
                continue
            remaining = [
                endpoint
                for endpoint in instance.get_all_endpoints()
                if excluded_pairs is None or (instance.id, endpoint.id) not in excluded_pairs
            ]
            if not remaining:
                continue
            try:
                load = instance_compute_load(instance, role)
            except Exception as exc:
                logger.warning(
                    "Failed to calculate instance compute load for instance %s: %s",
                    instance.id,
                    exc,
                )
                continue
            if load < best_instance_load:
                best_instance_load = load
                best_instance = instance
        if best_instance is None:
            return None

        best_endpoint: Endpoint | None = None
        best_endpoint_load = float("inf")
        for endpoint in best_instance.get_all_endpoints():
            if excluded_pairs is not None and (best_instance.id, endpoint.id) in excluded_pairs:
                continue
            try:
                load = endpoint_compute_load(best_instance, endpoint, role)
            except Exception as exc:
                logger.warning(
                    "Failed to calculate DP compute load for instance %s endpoint %s: %s",
                    best_instance.id,
                    endpoint.id,
                    exc,
                )
                continue
            if load < best_endpoint_load:
                best_endpoint_load = load
                best_endpoint = endpoint
        if best_endpoint is None:
            return None
        return (best_instance, best_endpoint, best_instance_load)

    @staticmethod
    def resolve_max_matched_tokens(
        instances: list[Instance],
        req_info: RequestInfo,
        w_npu: float = 1.0,
        w_cpu: float = 1.0,
        w_disk: float = 0.0,
    ) -> int:
        """
        Query conductor and return the global max prefix match (tokens).

        Per-DP hits are not used for routing -- only the maximum length is kept
        so the request's compute increment is ``ISL - max_matched``.
        """
        encoded_ids = KvCacheAffinityPolicy._ensure_token_ids(req_info)
        isl = len(encoded_ids)
        block_size = KvCacheAffinityPolicy._conductor_block_size()
        tenant: dict = {}
        if block_size <= 0 or isl >= block_size:
            rsp = ConductorApiClient.query_conductor(instances, encoded_ids)
            tenant = rsp.get(TENANT_ID) or {}
            if not tenant:
                logger.warning(
                    "compute_length: conductor query returned no tenant data (tenant_id=%s, instances=%d)",
                    TENANT_ID,
                    len(instances),
                )

        max_matched = 0
        for instance in instances:
            instance_data = tenant.get(conductor_instance_id(instance))
            if instance_data is None:
                continue
            dp_map = instance_data.get("DP", {})
            for endpoint in instance.get_all_endpoints():
                matched_raw = dp_map.get(f"{endpoint.id}", 0)
                matched = KvCacheAffinityPolicy._weighted_matched_tokens(
                    matched_raw, block_size, w_npu, w_cpu, w_disk
                )
                capped = min(matched, isl) if isl > 0 else 0
                if capped > max_matched:
                    max_matched = capped

        ComputeLengthPolicy._stash_max_matched(req_info, max_matched, isl)
        return max_matched

    @staticmethod
    def _stash_max_matched(req_info: RequestInfo | None, max_matched: int, isl: int) -> None:
        if req_info is None:
            return
        try:
            req_info.max_matched_tokens = max_matched
        except Exception as exc:  # pragma: no cover - req_info may be immutable in some callers
            logger.debug("Could not cache max_matched_tokens on req_info: %s", exc)
            return
        logger.debug(
            "compute_length max_matched=%s isl=%s compute=%s",
            max_matched,
            isl,
            max(0, isl - max_matched),
        )

    @staticmethod
    def select_endpoint_candidates_from_list(
        instances: list[Instance],
        req_info: RequestInfo | None = None,
        role: PDRole | None = None,
        top_k: int = 1,
        *,
        resolve_match: bool = True,
        is_blocked: Callable[[int], bool] | None = None,
        excluded_pairs: set[tuple[int, int]] | None = None,
        w_npu: float = 1.0,
        w_cpu: float = 1.0,
        w_disk: float = 0.0,
    ) -> list[tuple[Instance, Endpoint, float]]:
        """
        Optionally resolve the global max match, then rank by instance then DP load.

        ``top_k`` is accepted for API symmetry; hierarchical pick yields at most one
        ``(instance, endpoint, instance_load)`` tuple.
        """
        del top_k
        if resolve_match and req_info is not None:
            ComputeLengthPolicy.resolve_max_matched_tokens(
                instances, req_info, w_npu=w_npu, w_cpu=w_cpu, w_disk=w_disk
            )
        selected = ComputeLengthPolicy.select_instance_then_endpoint(
            instances,
            role,
            is_blocked=is_blocked,
            excluded_pairs=excluded_pairs,
        )
        if selected is None:
            return []
        return [selected]

    @staticmethod
    def select_endpoint_from_list(
        instances: list[Instance],
        req_info: RequestInfo | None = None,
        role: PDRole | None = None,
        *,
        resolve_match: bool = True,
    ) -> tuple[Instance, Endpoint] | None:
        ranked = ComputeLengthPolicy.select_endpoint_candidates_from_list(
            instances, req_info, role=role, resolve_match=resolve_match
        )
        if not ranked:
            return None
        instance, endpoint, _score = ranked[0]
        return (instance, endpoint)

    def select_instance_and_endpoint(self, role: PDRole = None):
        active_instances = self._instance_provider.get_available_instances(role)
        if not active_instances:
            logger.warning("No active instances available for scheduling")
            return None
        selected = ComputeLengthPolicy.select_instance_then_endpoint(active_instances.values(), role)
        if selected is None:
            return None
        return (selected[0], selected[1])

    def select_instance_and_endpoint_from_list(
        self,
        instances: list[Instance],
        role: PDRole | None = None,
        req_info=None,
    ):
        selected = ComputeLengthPolicy.select_endpoint_from_list(instances, req_info, role=role)
        return selected

    def _select_instance(self, role: PDRole = None) -> Instance | None:
        active_instances = self._instance_provider.get_available_instances(role)
        if not active_instances:
            logger.warning("No active instances available for scheduling")
            return None
        selected = ComputeLengthPolicy.select_instance_then_endpoint(active_instances.values(), role)
        return None if selected is None else selected[0]

    def _select_endpoint(self, instance: Instance) -> Endpoint | None:
        selected = ComputeLengthPolicy.select_instance_then_endpoint([instance], instance.role)
        return None if selected is None else selected[1]
