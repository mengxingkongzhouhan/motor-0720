# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
ZMQ-free allocate arbitration for the coordinator scheduling hot path.

These functions are the single source of truth for "given a worker-proposed candidate (or
candidate set) and a fresh workload view, which (instance, endpoint) wins". Infer Worker
``select_and_allocate`` calls them on the CAS CHANGED slow path (same Python scorer, no ZMQ).
Scoring formulas live in ``motor/coordinator/scheduler/policy`` and are not duplicated here.
"""

from collections.abc import Callable
from dataclasses import dataclass

from motor.common.logger import get_logger
from motor.common.resources.endpoint import Endpoint
from motor.common.resources.instance import Instance, PDRole
from motor.coordinator.scheduler.policy.load_balance import LoadBalancePolicy
from motor.coordinator.scheduler.runtime.zmq_protocol import (
    CANDIDATE_POLICY_KV_CACHE_AFFINITY,
    CANDIDATE_POLICY_LOAD_BALANCE,
    CANDIDATE_POLICY_SMETRIC_GATED,
    KNOWN_CANDIDATE_POLICIES,
)
from motor.coordinator.scheduler.policy.smetric_gated import (
    GatedCandidate,
    format_candidates,
    pick_gated,
    sort_candidates,
)

logger = get_logger(__name__)


@dataclass
class ArbitrationContext:
    """
    Host-supplied dependencies the ZMQ-free arbitration needs.

    Infer Worker ``AsyncSchedulerClient`` supplies the local instance cache and circuit-breaker
    / SHM blocked view so the same selection logic runs without a ZMQ round-trip.
    """

    get_available_instances: Callable[[PDRole | None], dict[int, Instance]]
    is_instance_circuit_open: Callable[[int], bool]
    endpoint_instance_score_weight: float = 0.05
    is_load_balance_scheduler: bool = False
    smetric_gated_active_factor: float = 1.0
    smetric_gated_cpu_factor: float = 1.0


def matches_engine_type(instance: Instance, required_engine_type: str | None) -> bool:
    """Return True when the instance's engine_type matches the requirement (or none required)."""
    if not required_engine_type:
        return True
    return str(getattr(instance, "engine_type", "")).strip().lower() == required_engine_type


def matches_dispatch_capability(instance: Instance, required_dispatch_capability: str | None) -> bool:
    """Return True when the instance advertises the required dispatch capability (or none required)."""
    if not required_dispatch_capability:
        return True
    return required_dispatch_capability in (getattr(instance, "dispatch_capabilities", None) or [])


def find_available_instance_endpoint(
    ctx: ArbitrationContext,
    instance_id: int,
    endpoint_id: int,
) -> tuple[Instance, Endpoint] | None:
    """Find an available (instance, endpoint) pair across the E/P/D/U pools."""
    for role in (PDRole.ROLE_E, PDRole.ROLE_P, PDRole.ROLE_D, PDRole.ROLE_U):
        instance = ctx.get_available_instances(role).get(instance_id)
        if not instance:
            continue
        for pod_eps in (instance.endpoints or {}).values():
            for endpoint in (pod_eps or {}).values():
                if endpoint.id == endpoint_id:
                    return (instance, endpoint)
    return None


def select_valid_candidate(
    ctx: ArbitrationContext,
    candidate: tuple[int, int],
    role: PDRole,
    required_engine_type: str | None = None,
    required_dispatch_capability: str | None = None,
) -> tuple[Instance, Endpoint, float] | None:
    """
    Validate one proposed candidate against the fresh view and compute its current score.

    This is the fast path: when the worker selected from the exact current view, only the proposed
    endpoint is validated (in pool, right role, engine_type/capability match, circuit not open).
    """
    instance_id, endpoint_id = candidate
    if ctx.is_instance_circuit_open(instance_id):
        return None
    found = find_available_instance_endpoint(ctx, instance_id, endpoint_id)
    if found is None:
        return None
    instance, endpoint = found
    if not matches_engine_type(instance, required_engine_type):
        return None
    if not matches_dispatch_capability(instance, required_dispatch_capability):
        return None
    try:
        instance_role = PDRole(instance.role)
    except ValueError:
        instance_role = PDRole.ROLE_U
    if instance_role != role:
        return None
    try:
        score = LoadBalancePolicy.calculate_endpoint_score(
            instance,
            endpoint,
            role=role,
            instance_score_weight=ctx.endpoint_instance_score_weight,
        )
    except Exception as e:
        logger.warning(
            "Failed to score fast-path allocate candidate instance_id=%s endpoint_id=%s: %s",
            instance_id,
            endpoint_id,
            e,
        )
        return None
    return (instance, endpoint, score)


def select_global_load_balance_candidate(
    ctx: ArbitrationContext,
    role: PDRole,
    required_engine_type: str | None = None,
    excluded: set[tuple[int, int]] | None = None,
    required_dispatch_capability: str | None = None,
) -> tuple[Instance, Endpoint, float] | None:
    """Select the globally lowest-score endpoint for the role from the fresh pool.

    Circuit-broken endpoints are filtered so the authoritative re-scan never picks one that a local
    PUB cache may not yet know about. ``excluded`` drops pairs this CAS round already rejected.
    """
    instances = [
        instance
        for instance in ctx.get_available_instances(role).values()
        if matches_engine_type(instance, required_engine_type)
        and matches_dispatch_capability(instance, required_dispatch_capability)
    ]
    candidates = LoadBalancePolicy.select_endpoint_candidates_from_list(
        instances,
        role=role,
        top_k=1,
        instance_score_weight=ctx.endpoint_instance_score_weight,
        is_blocked=ctx.is_instance_circuit_open,
        excluded_pairs=excluded,
    )
    if not candidates:
        return None
    candidate = candidates[0]
    return (candidate.instance, candidate.endpoint, candidate.score)


def select_affinity_global(
    ctx: ArbitrationContext,
    affinity_candidates: list[tuple[int, int, float]],
    role: PDRole,
    prefill_load_scale: float | None,
    load_weight: float | None,
    required_engine_type: str | None = None,
    excluded: set[tuple[int, int]] | None = None,
    required_dispatch_capability: str | None = None,
) -> tuple[Instance, Endpoint, float] | None:
    """
    Global kv_cache_affinity unified selection over EVERY reported endpoint.

    For each candidate, recompute the unified cost with the fresh load:
    ``combined = prefill_load_scale * prefill_cost + load_weight * fresh_load``. Pick the minimum;
    ties prefer the lower prefill_cost (better affinity). The returned score is ``combined``.
    ``excluded`` drops pairs this CAS round already rejected.
    """
    pscale = prefill_load_scale if prefill_load_scale is not None else 1.0
    lweight = load_weight if load_weight is not None else 1.0
    best: tuple[Instance, Endpoint, float, float] | None = None  # (..., combined, prefill_cost)
    for instance_id, endpoint_id, prefill_cost in affinity_candidates:
        if excluded is not None and (instance_id, endpoint_id) in excluded:
            continue
        if ctx.is_instance_circuit_open(instance_id):
            continue
        found = find_available_instance_endpoint(ctx, instance_id, endpoint_id)
        if found is None:
            continue
        instance, endpoint = found
        if not matches_engine_type(instance, required_engine_type):
            continue
        if not matches_dispatch_capability(instance, required_dispatch_capability):
            continue
        try:
            instance_role = PDRole(instance.role)
        except ValueError:
            instance_role = PDRole.ROLE_U
        if instance_role != role:
            continue
        try:
            load = LoadBalancePolicy.calculate_endpoint_score(
                instance,
                endpoint,
                role=role,
                instance_score_weight=ctx.endpoint_instance_score_weight,
            )
        except Exception as e:
            logger.warning(
                "Failed to score affinity candidate instance_id=%s endpoint_id=%s: %s",
                instance_id,
                endpoint_id,
                e,
            )
            continue
        combined = pscale * prefill_cost + lweight * load
        if best is None:
            best = (instance, endpoint, combined, prefill_cost)
        elif combined < best[2] or (combined == best[2] and prefill_cost < best[3]):
            best = (instance, endpoint, combined, prefill_cost)
    if best is None:
        return None
    return (best[0], best[1], best[2])


def select_lowest_load_among_candidates(
    ctx: ArbitrationContext,
    candidates: list[tuple[int, int]],
    role: PDRole,
    required_engine_type: str | None = None,
    required_dispatch_capability: str | None = None,
) -> tuple[Instance, Endpoint, float] | None:
    """
    Among the affinity-ranked candidates, pick the lowest current endpoint score from the fresh
    ledger. The set is already the affinity top-k, so this spreads a burst by fresh load without
    breaking affinity. Ties keep the earliest (best-affinity) one.
    """
    best: tuple[Instance, Endpoint, float] | None = None
    for cand in candidates:
        if ctx.is_instance_circuit_open(cand[0]):
            continue
        found = find_available_instance_endpoint(ctx, *cand)
        if found is None:
            continue
        instance, endpoint = found
        if not matches_engine_type(instance, required_engine_type):
            continue
        if not matches_dispatch_capability(instance, required_dispatch_capability):
            continue
        try:
            instance_role = PDRole(instance.role)
        except ValueError:
            instance_role = PDRole.ROLE_U
        if instance_role != role:
            continue
        try:
            score = LoadBalancePolicy.calculate_endpoint_score(
                instance,
                endpoint,
                role=role,
                instance_score_weight=ctx.endpoint_instance_score_weight,
            )
        except Exception as e:
            logger.warning(
                "Failed to score affinity candidate instance_id=%s endpoint_id=%s: %s",
                cand[0],
                cand[1],
                e,
            )
            continue
        if best is None:
            best = (instance, endpoint, score)
        elif score < best[2]:
            best = (instance, endpoint, score)
    return best


def should_scan_global_load_balance(ctx: ArbitrationContext, candidate_policy: str | None) -> bool:
    """Return True when candidates were selected by load-balance semantics."""
    if candidate_policy == CANDIDATE_POLICY_LOAD_BALANCE:
        return True
    if candidate_policy in KNOWN_CANDIDATE_POLICIES:
        return False
    if candidate_policy is not None:
        logger.warning(
            "Unknown allocate candidate_policy=%s; falling back to scheduler_type",
            candidate_policy,
        )
    return ctx.is_load_balance_scheduler


def select_smetric_gated(
    ctx: ArbitrationContext,
    worker_candidate: tuple[int, int],
    gated_candidates: list[tuple[int, int, float, float]] | None,
    role: PDRole,
    required_engine_type: str | None = None,
    excluded: set[tuple[int, int]] | None = None,
    required_dispatch_capability: str | None = None,
    req_id: str | None = None,
) -> tuple[Instance, Endpoint, float] | None:
    """
    smetric_gated arbitration on the worker's fresh cache (SHM tokens + overlay fields).

    Used after a stale/blocked CAS, not on the first attempt. Resolve every scored endpoint
    that is still schedulable, sort by ``ledger.active_tokens + this_request_prefill`` and
    take the first one at or below both scaled ledger averages. The worker-supplied
    per-endpoint cost / cpu_blocks are the values stamped on commit. The returned score is
    the unified sort key.
    """
    if not gated_candidates:
        logger.warning(
            "smetric_gated: no endpoint costs; validating worker candidate %s req_id=%s",
            worker_candidate,
            req_id,
        )
        return select_valid_candidate(
            ctx, worker_candidate, role, required_engine_type, required_dispatch_capability
        )
    candidates: list[GatedCandidate] = []
    for instance_id, endpoint_id, prefill_cost, cpu_hits in gated_candidates:
        if excluded is not None and (instance_id, endpoint_id) in excluded:
            continue
        if ctx.is_instance_circuit_open(instance_id):
            continue
        found = find_available_instance_endpoint(ctx, instance_id, endpoint_id)
        if found is None:
            continue
        instance, endpoint = found
        if not matches_engine_type(instance, required_engine_type):
            continue
        if not matches_dispatch_capability(instance, required_dispatch_capability):
            continue
        try:
            instance_role = PDRole(instance.role)
        except ValueError:
            instance_role = PDRole.ROLE_U
        if instance_role != role:
            continue
        # npu_hit is not in the stamp tuple; GatedCandidate defaults it to 0.
        candidates.append(
            GatedCandidate(
                instance=instance,
                endpoint=endpoint,
                prefill_cost=prefill_cost,
                cpu_hit_blocks=cpu_hits,
            )
        )
    ranked = sort_candidates(candidates)
    picked = pick_gated(ranked, ctx.smetric_gated_active_factor, ctx.smetric_gated_cpu_factor)
    if picked is None:
        return None
    chosen, reason, active_threshold, cpu_threshold = picked
    logger.info(
        "smetric_gated: req_id=%s pick=%s-%s reason=%s active_threshold=%.1f cpu_threshold=%.1f "
        "factors=%.2f/%.2f ranked[ins-ep:ledger_prefill/active/cpu(+req_cost/+req_cpu)]=%s",
        req_id,
        chosen.instance.id,
        chosen.endpoint.id,
        reason,
        active_threshold,
        cpu_threshold,
        ctx.smetric_gated_active_factor,
        ctx.smetric_gated_cpu_factor,
        format_candidates(ranked),
    )
    return (chosen.instance, chosen.endpoint, chosen.unified_score)


def select_authoritative_allocate_candidate(
    ctx: ArbitrationContext,
    candidate: tuple[int, int],
    candidates: list[tuple[int, int]],
    role: PDRole,
    candidate_policy: str | None,
    affinity_candidates: list[tuple[int, int, float]] | None = None,
    prefill_load_scale: float | None = None,
    load_weight: float | None = None,
    required_engine_type: str | None = None,
    excluded: set[tuple[int, int]] | None = None,
    required_dispatch_capability: str | None = None,
    gated_candidates: list[tuple[int, int, float, float]] | None = None,
    req_id: str | None = None,
) -> tuple[Instance, Endpoint, float] | None:
    """
    Select the allocation target from a fresh workload view (the slow / re-rank path).

    Load-balance scans all endpoints. KV-cache affinity in unified mode re-ranks EVERY reported
    endpoint by ``prefill_load_scale*prefill_cost + load_weight*fresh_load``; older affinity callers
    without per-endpoint prefill_cost fall back to "least-loaded among the ranked alternates".
    smetric_gated re-sorts by ``active_tokens + this_request_prefill`` and applies the two
    mean gates (CHANGED / BLOCKED retry path; first CAS uses the policy winner like other
    policies).
    Other policies keep the proposed endpoint. ``excluded`` (pairs this CAS round already
    rejected) is forwarded to every branch that scans beyond ``candidates``.
    """
    if should_scan_global_load_balance(ctx, candidate_policy):
        selected = select_global_load_balance_candidate(
            ctx,
            role,
            required_engine_type,
            excluded=excluded,
            required_dispatch_capability=required_dispatch_capability,
        )
        if selected is not None:
            return selected
    if candidate_policy == CANDIDATE_POLICY_KV_CACHE_AFFINITY:
        if affinity_candidates:
            selected = select_affinity_global(
                ctx,
                affinity_candidates,
                role,
                prefill_load_scale,
                load_weight,
                required_engine_type,
                excluded=excluded,
                required_dispatch_capability=required_dispatch_capability,
            )
            if selected is not None:
                return selected
        elif len(candidates) > 1:
            selected = select_lowest_load_among_candidates(
                ctx, candidates, role, required_engine_type, required_dispatch_capability
            )
            if selected is not None:
                return selected
    if candidate_policy == CANDIDATE_POLICY_SMETRIC_GATED:
        selected = select_smetric_gated(
            ctx,
            candidate,
            gated_candidates,
            role,
            required_engine_type,
            excluded=excluded,
            required_dispatch_capability=required_dispatch_capability,
            req_id=req_id,
        )
        if selected is not None:
            return selected
    return select_valid_candidate(ctx, candidate, role, required_engine_type, required_dispatch_capability)
