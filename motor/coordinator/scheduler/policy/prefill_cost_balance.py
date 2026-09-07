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
Prefill-cost load balance: score every endpoint by its committed workload ledger,
``prefill_cost + active_tokens_weight * active_tokens`` (lower is better).

``prefill_cost`` is the endpoint's outstanding remaining prefill (sum over in-flight requests of
``isl - matched_tokens``), ``active_tokens`` its outstanding compute load. Both come from the
scheduler ledger (``Endpoint.workload``), so ranking needs no per-request affinity math. The
Conductor is only consulted to stamp the *allocated* request's own remaining prefill on the ledger
(reusing the SMetric cost model), which is what keeps ``prefill_cost`` distinct from
``active_tokens`` over time.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable

from motor.common.logger import get_logger
from motor.common.resources.endpoint import Endpoint
from motor.common.resources.instance import Instance, PDRole
from motor.coordinator.domain import InstanceProvider
from motor.coordinator.models.request import RequestInfo
from motor.coordinator.scheduler.policy.base import BaseSchedulingPolicy, WorkloadLedgerMixin

logger = get_logger(__name__)

DEFAULT_ACTIVE_TOKENS_WEIGHT = 1.0

# Roles whose allocations carry a remaining-prefill cost worth stamping on the ledger.
PREFILL_COST_ROLES = frozenset({PDRole.ROLE_P, PDRole.ROLE_U})


def _ledger_value(workload, field: str) -> float:
    try:
        return max(0.0, float(getattr(workload, field, 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


class PrefillCostBalancePolicy(WorkloadLedgerMixin, BaseSchedulingPolicy):
    """
    Ledger-only load balance with a tunable trade-off between remaining prefill and total load.

    ``score(endpoint) = workload.prefill_cost + x * workload.active_tokens`` where ``x`` is
    ``SchedulerConfig.prefill_cost_balance.active_tokens_weight``. The lowest score wins; ties keep
    traversal order (rotated by ``start_index`` so equal-load choices spread across workers).
    """

    def __init__(self, instance_provider: InstanceProvider):
        super().__init__(instance_provider=instance_provider)
        self._active_tokens_weight = DEFAULT_ACTIVE_TOKENS_WEIGHT
        logger.info("PrefillCostBalancePolicy started.")

    def set_active_tokens_weight(self, weight: float) -> None:
        """Set ``x`` in ``prefill_cost + x * active_tokens``."""
        self._active_tokens_weight = max(0.0, float(weight))

    @property
    def active_tokens_weight(self) -> float:
        return self._active_tokens_weight

    @staticmethod
    def calculate_endpoint_score(
        endpoint: Endpoint,
        active_tokens_weight: float = DEFAULT_ACTIVE_TOKENS_WEIGHT,
    ) -> float:
        """``prefill_cost + active_tokens_weight * active_tokens`` from the endpoint ledger."""
        workload = endpoint.workload
        weight = max(0.0, float(active_tokens_weight))
        return _ledger_value(workload, "prefill_cost") + weight * _ledger_value(workload, "active_tokens")

    @staticmethod
    def select_endpoint_candidates_from_list(
        instances: list[Instance] | Iterable[Instance],
        top_k: int = 1,
        active_tokens_weight: float = DEFAULT_ACTIVE_TOKENS_WEIGHT,
        start_index: int = 0,
        *,
        is_blocked: Callable[[int], bool] | None = None,
    ) -> list[tuple[Instance, Endpoint, float]]:
        """
        Rank every endpoint of ``instances`` by ledger score; return the ``top_k`` lowest.

        ``start_index`` rotates traversal order and only affects ties. ``is_blocked`` drops
        circuit-broken instances.
        """
        if top_k <= 0:
            return []
        if not isinstance(instances, (list, tuple)):
            instances = list(instances)
        if not instances:
            return []
        n = len(instances)
        rotated = [instances[(start_index + i) % n] for i in range(n)]
        scored: list[tuple[float, int, Instance, Endpoint]] = []
        order = 0
        for instance in rotated:
            if is_blocked is not None and is_blocked(instance.id):
                continue
            for endpoint in instance.get_all_endpoints():
                try:
                    score = PrefillCostBalancePolicy.calculate_endpoint_score(endpoint, active_tokens_weight)
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(
                        "Failed to score endpoint instance_id=%s endpoint_id=%s: %s",
                        instance.id,
                        endpoint.id,
                        e,
                    )
                    continue
                scored.append((score, order, instance, endpoint))
                order += 1
        best = heapq.nsmallest(top_k, scored, key=lambda item: (item[0], item[1]))
        return [(instance, endpoint, score) for score, _order, instance, endpoint in best]

    @staticmethod
    def select_endpoint_from_list(
        instances: list[Instance] | Iterable[Instance],
        active_tokens_weight: float = DEFAULT_ACTIVE_TOKENS_WEIGHT,
        start_index: int = 0,
    ) -> tuple[Instance, Endpoint] | None:
        candidates = PrefillCostBalancePolicy.select_endpoint_candidates_from_list(
            instances,
            top_k=1,
            active_tokens_weight=active_tokens_weight,
            start_index=start_index,
        )
        if not candidates:
            return None
        instance, endpoint, _score = candidates[0]
        return (instance, endpoint)

    @staticmethod
    def collect_request_prefill_costs(instances: list[Instance], req_info: RequestInfo | None) -> bool:
        """
        Best-effort Conductor lookup that caches this request's remaining prefill per endpoint on
        ``req_info.smetric_debug`` (SMetric cost model: ``max(0, isl - matched_tokens)``).

        The cache is only used to stamp the committed endpoint's ``prefill_cost`` on the ledger;
        it never influences which endpoint this policy picks. Returns False when the Conductor
        had no data (the ledger stamp then falls back to the full prompt length).
        """
        if req_info is None or not instances:
            return False
        from motor.coordinator.scheduler.policy.smetric import SMetricPolicy

        try:
            ranked = SMetricPolicy.select_endpoint_candidates_from_list(instances, req_info, top_k=1)
        except Exception as e:
            logger.warning(
                "prefill_cost_balance: conductor cost lookup failed req_id=%s: %s",
                getattr(req_info, "req_id", None),
                e,
            )
            return False
        return bool(ranked) and isinstance(getattr(req_info, "smetric_debug", None), dict)

    def select_instance_and_endpoint(self, role: PDRole = None):
        active_instances = self._instance_provider.get_available_instances(role)
        if not active_instances:
            logger.warning("No active instances available for scheduling")
            return None
        return PrefillCostBalancePolicy.select_endpoint_from_list(
            active_instances.values(),
            active_tokens_weight=self._active_tokens_weight,
        )

    def select_instance_and_endpoint_from_list(
        self,
        instances: list[Instance],
        role: PDRole | None = None,
        req_info: RequestInfo | None = None,
    ):
        """Stamp-cost lookup for P/U, then rank the subset by the ledger score."""
        if role in PREFILL_COST_ROLES:
            PrefillCostBalancePolicy.collect_request_prefill_costs(instances, req_info)
        return PrefillCostBalancePolicy.select_endpoint_from_list(
            instances,
            active_tokens_weight=self._active_tokens_weight,
        )

    def _select_instance(self, role: PDRole = None) -> Instance | None:
        selected = self.select_instance_and_endpoint(role)
        return selected[0] if selected else None

    def _select_endpoint(self, instance: Instance) -> Endpoint | None:
        if not instance:
            return None
        selected = PrefillCostBalancePolicy.select_endpoint_from_list(
            [instance],
            active_tokens_weight=self._active_tokens_weight,
        )
        return selected[1] if selected else None
