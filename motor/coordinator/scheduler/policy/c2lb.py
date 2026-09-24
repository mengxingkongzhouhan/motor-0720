# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
C2LB scheduling policy: rank endpoints by their ledger, gate on two ledger averages.

1. Sort the endpoints of the request's role by the ledger ``workload.isl`` ascending,
   i.e. the in-flight request length currently outstanding on each endpoint (sum over its
   in-flight requests of ``max(0, isl)``).
2. Walk that order and commit the first endpoint whose ledger is at or below BOTH scaled
   averages over the ranked endpoints:
   ``active_tokens <= mean(active_tokens) * active_tokens_mean_factor`` and
   ``cpu_hit_blocks <= mean(cpu_hit_blocks) * cpu_hit_blocks_mean_factor``
   (factors from ``SchedulerConfig.c2lb``, default 1.0). ``<=`` so that an idle
   cluster (every ledger 0, mean 0) still passes the gates instead of relying on the fallback.

All three inputs are ledger fields, so the ranking itself needs no per-request affinity math.
The KV Conductor is queried once per request only to know what to ADD to the committed
endpoint's ledger: the request's own prompt length (``max(0, isl)``) and the CPU-tier KV
blocks it would pull there (``cpu_blocks``). RELEASE subtracts both again, so ``isl`` /
``cpu_hit_blocks`` track the in-flight prompt length and CPU->NPU KV transfer per endpoint.

When no endpoint passes both gates the policy degrades in order: first endpoint passing the
``active_tokens`` gate alone, then the head of the list (lowest ledger isl).

On motor-0924 the authoritative re-pick lives in worker-local ``allocate_arbitration``
(schema-4 SHM CAS). ``active_tokens`` is the cross-worker SHM ledger; ``isl`` and
``cpu_hit_blocks`` are a worker-local overlay (schema-4 does not carry them).
Prefill / encode / union use ``prefill_scheduler_type``; decode falls back to load_balance
when this policy is set on decode.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from motor.common.logger import get_logger
from motor.common.resources.endpoint import Endpoint
from motor.common.resources.instance import Instance, PDRole
from motor.common.utils.singleton import ThreadSafeSingleton
from motor.config.coordinator import CoordinatorConfig, SchedulerConfig
from motor.coordinator.api_client.conductor_api_client import (
    TENANT_ID,
    ConductorApiClient,
    conductor_instance_id,
)
from motor.coordinator.domain import InstanceProvider
from motor.coordinator.models.constants import DEFAULT_REQUEST_ID, OpenAIField
from motor.coordinator.models.request import RequestInfo
from motor.coordinator.scheduler.policy.base import BaseSchedulingPolicy
from motor.coordinator.scheduler.policy.utils import (
    preprocess_input,
    preprocess_messages_for_dsv4,
    preprocess_messages_for_standard,
)

logger = get_logger(__name__)

# Roles that do prefill, i.e. whose allocations have a conductor cost and CPU hit count.
C2LB_ROLES = frozenset({PDRole.ROLE_P, PDRole.ROLE_U})

# Gate threshold = candidate mean * factor; 1.0 is the plain average.
DEFAULT_MEAN_FACTOR = 1.0
_TOKENIZER_LOAD_RETRY_SECONDS = 30.0

# C2LB discounts a cached prefix 1:1 against prompt length. Not configurable; not shared with
# kv_cache_affinity's overlap_credit knob.
_C2LB_OVERLAP_CREDIT = 1


def _factor(value: float | None) -> float:
    """Normalize a mean factor: None -> default, negatives -> 0."""
    if value is None:
        return DEFAULT_MEAN_FACTOR
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return DEFAULT_MEAN_FACTOR


# How the final endpoint was chosen (returned for logging / tests).
PICK_BOTH_GATES = "both_gates"
PICK_ACTIVE_GATE = "active_gate"
PICK_MIN_LEDGER_PREFILL = "min_ledger_prefill"


class C2LBTokenizer(ThreadSafeSingleton):
    """Tokenizer owned by c2lb. Does not import other scheduling policies."""

    def __init__(self, config: CoordinatorConfig | None = None):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.config_lock = threading.RLock()
        if config is None:
            config = CoordinatorConfig()

        self.tokenizer = None
        self._is_dsv4 = False
        self._next_load_attempt_at = 0.0
        scheduler_config = getattr(config, "scheduler_config", None)
        kv_config = getattr(scheduler_config, "kv_conductor_config", None) if scheduler_config else None
        if kv_config is None:
            kv_config = getattr(config, "prefill_kv_event_config", None)
        self.model_path = getattr(kv_config, "model_path", "") if kv_config else ""
        self.engine_type = str(getattr(kv_config, "engine_type", "vllm") or "vllm").strip().lower()
        self.openai_standard = os.environ.get("OPENAI_STANDARD", "STANDARD")
        c2lb_enabled = isinstance(scheduler_config, SchedulerConfig) and scheduler_config.uses_c2lb()
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        if c2lb_enabled:
            self.get_tokenizer()
        logger.info(
            "C2LBTokenizer init.(model_path:%s, is_dsv4:%s, lazy_load:%s)",
            self.model_path,
            self._is_dsv4,
            not c2lb_enabled,
        )

    def get_tokenizer(self):
        """Load the local tokenizer lazily and retry transient failures after a cooldown."""
        if self.tokenizer is not None:
            return self.tokenizer
        if time.monotonic() < self._next_load_attempt_at:
            return None
        with self.config_lock:
            if self.tokenizer is not None:
                return self.tokenizer
            if time.monotonic() < self._next_load_attempt_at:
                return None
            if not getattr(self, "model_path", ""):
                return None
            try:
                if self.engine_type == "vllm" and self._is_deepseek_v4_model(self.model_path):
                    from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer  # pylint: disable=import-error,no-name-in-module

                    self.tokenizer = DeepseekV4Tokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                    self._is_dsv4 = True
                else:
                    from transformers import AutoTokenizer

                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            except Exception as exc:
                self._next_load_attempt_at = time.monotonic() + _TOKENIZER_LOAD_RETRY_SECONDS
                logger.warning(
                    "C2LBTokenizer load failed; retrying in %.0fs: %s",
                    _TOKENIZER_LOAD_RETRY_SECONDS,
                    exc,
                )
                return None
            self._next_load_attempt_at = 0.0
            return self.tokenizer

    def apply_chat_template(self, messages: list, tools: list | None = None, req_data: dict | None = None) -> list[int]:
        if self.get_tokenizer() is None:
            return []
        try:
            if self._is_dsv4:
                return self._apply_chat_template_dsv4(messages, tools, req_data)
            if self.openai_standard != "STANDARD":
                return self._apply_chat_template_with_preprocess(messages, tools, req_data)
            return self._apply_chat_template_standard(messages, tools, req_data)
        except Exception as exc:
            if self._is_dsv4:
                logger.error("c2lb dsv4 tokenize failed; returning []: %s", exc)
                return []
            logger.warning("c2lb primary tokenize path failed: %s; trying fallback", exc)
            return self._safe_fallback_encode(messages, tools, req_data)

    def encode(self, prompt: str) -> list[int]:
        tokenizer = self.get_tokenizer()
        return [] if tokenizer is None else tokenizer.encode(prompt)

    @staticmethod
    def _read_model_config_dict(model_path: str) -> dict | None:
        try:
            with open(Path(model_path) / "config.json", encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError) as exc:
            logger.debug("Could not read config.json from %s: %s", model_path, exc)
            return None

    @staticmethod
    def _is_deepseek_v4_model(model_path: str) -> bool:
        config_dict = C2LBTokenizer._read_model_config_dict(model_path)
        if not config_dict:
            return False
        return config_dict.get("model_type") == "deepseek_v4" or "DeepseekV4ForCausalLM" in (
            config_dict.get("architectures") or []
        )

    @staticmethod
    def _build_dsv4_chat_template_kwargs(req_data: dict | None) -> dict:
        kwargs: dict = {"tokenize": True, "drop_thinking": True}
        if not req_data:
            return kwargs
        reasoning_effort = req_data.get("reasoning_effort")
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        chat_template_kwargs = req_data.get("chat_template_kwargs") or {}
        if isinstance(chat_template_kwargs, dict):
            kwargs.update(chat_template_kwargs)
        if reasoning_effort is not None and "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = reasoning_effort != "none"
        return kwargs

    @staticmethod
    def _build_standard_chat_template_kwargs(req_data: dict | None, *, tokenize: bool) -> dict:
        kwargs: dict = {"add_generation_prompt": True, "tokenize": tokenize}
        if tokenize:
            kwargs["return_dict"] = False
        if not req_data:
            return kwargs
        if isinstance(req_data.get("add_generation_prompt"), bool):
            kwargs["add_generation_prompt"] = req_data["add_generation_prompt"]
        if req_data.get("continue_final_message"):
            kwargs["continue_final_message"] = True
            kwargs["add_generation_prompt"] = False
        if req_data.get("documents") is not None:
            kwargs["documents"] = req_data["documents"]
        template_kwargs = req_data.get("chat_template_kwargs") or {}
        if isinstance(template_kwargs, dict):
            reserved = {
                "tokenize",
                "return_dict",
                "conversation",
                "tools",
                "add_generation_prompt",
                "continue_final_message",
            }
            kwargs.update({key: value for key, value in template_kwargs.items() if key not in reserved})
        reasoning_effort = req_data.get("reasoning_effort")
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs.setdefault("enable_thinking", reasoning_effort != "none")
        thinking = req_data.get("thinking")
        if isinstance(thinking, dict) and "enable_thinking" not in kwargs:
            if thinking.get("type") == "enabled":
                kwargs["enable_thinking"] = True
            elif thinking.get("type") == "disabled":
                kwargs["enable_thinking"] = False
        return kwargs

    def _apply_chat_template_dsv4(self, messages: list, tools: list | None, req_data: dict | None) -> list[int]:
        messages, tools = preprocess_messages_for_dsv4(messages, tools)
        result = self.tokenizer.apply_chat_template(
            messages, tools=tools, **self._build_dsv4_chat_template_kwargs(req_data)
        )
        return result if isinstance(result, list) else self.tokenizer.encode(result, add_special_tokens=False)

    def _apply_chat_template_standard(self, messages: list, tools: list | None, req_data: dict | None) -> list[int]:
        return self.tokenizer.apply_chat_template(
            conversation=preprocess_messages_for_standard(messages),
            tools=tools,
            **self._build_standard_chat_template_kwargs(req_data, tokenize=True),
        )

    def _apply_chat_template_with_preprocess(
        self, messages: list, tools: list | None, req_data: dict | None
    ) -> list[int]:
        messages, tools = preprocess_input(messages, tools)
        prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            tools=tools,
            **self._build_standard_chat_template_kwargs(req_data, tokenize=False),
        )
        return self.tokenizer.encode(prompt)

    def _safe_fallback_encode(self, messages: list, tools: list | None, req_data: dict | None) -> list[int]:
        try:
            if self.openai_standard == "STANDARD":
                return self._apply_chat_template_with_preprocess(messages, tools, req_data)
            return self._apply_chat_template_standard(messages, tools, req_data)
        except Exception as exc:
            logger.error("c2lb tokenize failed on both primary and fallback paths; returning []: %s", exc)
            return []


def _prompt_token_ids(req_info: RequestInfo) -> list[int]:
    """Use cached token ids, otherwise tokenize with the local C2LBTokenizer."""
    engine_cached = getattr(req_info, "engine_token_ids", None)
    if isinstance(engine_cached, list) and engine_cached:
        return engine_cached
    cached = getattr(req_info, "token_ids", None)
    if isinstance(cached, list) and cached:
        return cached
    encoded_ids: list[int] = []
    req_data = getattr(req_info, "req_data", None) or {}
    messages = req_data.get(OpenAIField.MESSAGES, None)
    tools = req_data.get(OpenAIField.TOOLS, None)
    if messages is not None:
        encoded_ids = C2LBTokenizer().apply_chat_template(messages, tools, req_data=req_data)
    else:
        prompt = req_data.get(OpenAIField.PROMPT, None)
        if isinstance(prompt, str):
            encoded_ids = C2LBTokenizer().encode(prompt)
        elif (
            isinstance(prompt, list)
            and prompt
            and all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in prompt)
        ):
            encoded_ids = prompt.copy()
    try:
        req_info.token_ids = encoded_ids
    except Exception as e:
        logger.debug("Could not cache token_ids on req_info: %s", e)
    return encoded_ids


def _prefill_cost(isl: int, matched_tokens: int) -> float:
    """Remaining prefill tokens with overlap_credit fixed at 1: max(0, isl - matched)."""
    matched = max(0, min(matched_tokens, isl)) if isl > 0 else 0
    return float(max(0, isl - _C2LB_OVERLAP_CREDIT * matched))


def _matched_tokens(matched: object) -> int:
    """Read both legacy integer and DpBlocks conductor match formats."""
    if isinstance(matched, dict):
        matched = matched.get("matched_tokens", 0)
    try:
        return max(0, int(matched or 0))
    except (TypeError, ValueError):
        return 0


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
    npu_hit: float = 0.0

    @property
    def key(self) -> tuple[int, int]:
        return (self.instance.id, self.endpoint.id)

    @property
    def ledger_isl(self) -> float:
        return _ledger_value(self.endpoint, "isl")

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
    """Lowest ledger ``workload.isl`` first, ties by (instance_id, endpoint_id)."""
    endpoint_count = max(1, len(candidates[0].instance.get_all_endpoints()))
    return sorted(candidates, key=lambda c: (c.ledger_isl + 0.05 * (c.instance.gathered_workload.isl / endpoint_count)))


def sort_candidates_by_npu_hit(
    candidates: list[GatedCandidate],
    threshold: float,
) -> list[GatedCandidate]:
    return sorted(
        (c for c in candidates if c.npu_hit > threshold),
        key=lambda c: c.npu_hit,
        reverse=True,
    )


def pick_gated(
    candidates: list[GatedCandidate],
    active_tokens_mean_factor: float = DEFAULT_MEAN_FACTOR,
    cpu_hit_blocks_mean_factor: float = DEFAULT_MEAN_FACTOR,
) -> tuple[GatedCandidate, str, float, float] | None:
    """
    Walk ``candidates`` (already in ledger isl order) and return the first one whose
    ledger is at or below both scaled averages, plus the pick reason and the two thresholds
    actually used (``mean * factor``).

    Averages are taken over the candidates' current ledgers (``endpoint.workload``), so the
    caller decides which view is authoritative (worker SHM cache vs overlay).
    Fallback order when nothing passes both gates: active_tokens gate only, then the head of the
    list (lowest ledger isl).
    """
    if not candidates:
        return None

    n = len(candidates)
    active_avg = sum(c.ledger_active_tokens for c in candidates) / n
    cpu_avg = sum(c.ledger_cpu_hit_blocks for c in candidates) / n
    isl_avg = sum(c.ledger_isl for c in candidates) / n

    active_threshold = active_avg * _factor(active_tokens_mean_factor)
    cpu_threshold = cpu_avg * _factor(cpu_hit_blocks_mean_factor)
    isl_threshold = isl_avg * _factor(active_tokens_mean_factor)

    candidates_with_hight_npu_hit = sort_candidates_by_npu_hit(candidates, 0.8)
    if candidates_with_hight_npu_hit:

         for cand in candidates_with_hight_npu_hit:
            under_active = cand.ledger_active_tokens <= active_threshold
            under_cpu = cand.ledger_cpu_hit_blocks <= cpu_threshold
            under_isl = cand.ledger_isl <= isl_threshold

            if under_active and under_cpu and under_isl:
                return (cand, PICK_BOTH_GATES, active_threshold, cpu_threshold)
   
    active_only: GatedCandidate | None = None
    for cand in candidates:

        if cand.ledger_isl > isl_avg:
            return (candidates[0], PICK_MIN_LEDGER_PREFILL, active_threshold, cpu_threshold)
        under_active = cand.ledger_active_tokens <= active_threshold
        under_cpu = cand.ledger_cpu_hit_blocks <= cpu_threshold
        if under_active and under_cpu:
            return (cand, PICK_BOTH_GATES, active_threshold, cpu_threshold)

    return (candidates[0], PICK_MIN_LEDGER_PREFILL, active_threshold, cpu_threshold)


def format_candidates(candidates: list[GatedCandidate]) -> str:
    """``ins-ep:ledger_prefill/active/cpu(+req_cost/+req_cpu)`` per candidate, for the selection log."""
    return " ".join(
        f"{c.instance.id}-{c.endpoint.id}:{c.ledger_isl:.0f}/{c.ledger_active_tokens:.0f}/"
        f"{c.ledger_cpu_hit_blocks:.0f}(+{c.prefill_cost:.0f}/+{c.cpu_hit_blocks:.0f})"
        for c in candidates
    )


class C2LBPolicy(BaseSchedulingPolicy):
    """
    Rank by ledger isl, commit the first endpoint under both ledger load averages.

    Workers run the conductor query (for the stamp values) and re-rank / re-gate against the
    local cache (SHM ``active_tokens`` + worker-local overlay) before CAS-committing.
    """

    def __init__(self, instance_provider: InstanceProvider):
        super().__init__(instance_provider=instance_provider)
        self._active_tokens_mean_factor = DEFAULT_MEAN_FACTOR
        self._cpu_hit_blocks_mean_factor = DEFAULT_MEAN_FACTOR
        logger.info("C2LBPolicy started.")

    def set_mean_factors(self, active_tokens_mean_factor: float, cpu_hit_blocks_mean_factor: float) -> None:
        """Set the multipliers applied to the two candidate averages used as gate thresholds."""
        self._active_tokens_mean_factor = _factor(active_tokens_mean_factor)
        self._cpu_hit_blocks_mean_factor = _factor(cpu_hit_blocks_mean_factor)

    @property
    def mean_factors(self) -> tuple[float, float]:
        return (self._active_tokens_mean_factor, self._cpu_hit_blocks_mean_factor)

    @staticmethod
    def score_endpoints(instances: list[Instance], req_info: RequestInfo) -> list[GatedCandidate] | None:
        """
        Conductor lookup: every endpoint as a ``GatedCandidate``, sorted by its ledger isl.

        The conductor only supplies the per-endpoint stamp values (request cost, cpu_blocks).
        ``None`` means it had no data for our instances (caller falls back). Also caches
        ``{(instance_id, endpoint_id): (prefill_cost, cpu_hit_blocks)}`` on
        ``req_info.c2lb_debug`` for the allocate stamp.
        """
        encoded_ids = _prompt_token_ids(req_info)
        if not encoded_ids:
            logger.warning("c2lb: no cached token_ids; falling back")
            return None
        isl = len(encoded_ids)
        rsp = ConductorApiClient.query_conductor(instances, encoded_ids)
        req_id = getattr(req_info, "req_id", None) or DEFAULT_REQUEST_ID
        logger.debug("c2lb: req_id=%s conductor_rsp=%s", req_id, rsp)
        tenant = rsp.get(TENANT_ID, None) if isinstance(rsp, dict) else None
        if tenant is None:
            logger.warning(
                "c2lb: conductor query returned no tenant data (tenant_id=%s, instances=%d)",
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
            logger.warning("c2lb: no instance data")
            return None
        if not candidates:
            logger.warning("c2lb: no endpoint scored")
            return None

        ranked = sort_candidates(candidates)
        req_info.c2lb_debug = {c.key: (c.prefill_cost, c.cpu_hit_blocks) for c in ranked}
        logger.info(
            "c2lb: req_id=%s isl=%s ranked[ins-ep:ledger_prefill/active/cpu(+req_cost/+req_cpu)]=%s",
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
        active_tokens_mean_factor: float = DEFAULT_MEAN_FACTOR,
        cpu_hit_blocks_mean_factor: float = DEFAULT_MEAN_FACTOR,
    ) -> list[tuple[Instance, Endpoint, float]] | None:
        """
        Worker-side proposal: the gated pick first, then the rest in ledger isl order.

        ``allocate_arbitration`` re-ranks and re-gates on the cache after a SHM refresh.
        """
        ranked = C2LBPolicy.score_endpoints(instances, req_info)
        if not ranked:
            return None
        picked = pick_gated(ranked, active_tokens_mean_factor, cpu_hit_blocks_mean_factor)
        if picked is None:
            return None
        chosen, reason, active_threshold, cpu_threshold = picked
        logger.debug(
            "c2lb(worker): req_id=%s pick=%s-%s reason=%s active_threshold=%.1f cpu_threshold=%.1f",
            getattr(req_info, "req_id", None) or DEFAULT_REQUEST_ID,
            chosen.instance.id,
            chosen.endpoint.id,
            reason,
            active_threshold,
            cpu_threshold,
        )
        ordered = [chosen] + [c for c in ranked if c is not chosen]
        return [(c.instance, c.endpoint, c.ledger_isl) for c in ordered[: max(1, top_k)]]

    @staticmethod
    def select_endpoint_from_list(
        instances: list[Instance],
        req_info: RequestInfo,
        active_tokens_mean_factor: float = DEFAULT_MEAN_FACTOR,
        cpu_hit_blocks_mean_factor: float = DEFAULT_MEAN_FACTOR,
    ) -> tuple[Instance, Endpoint] | None:
        ranked = C2LBPolicy.select_endpoint_candidates_from_list(
            instances,
            req_info,
            top_k=1,
            active_tokens_mean_factor=active_tokens_mean_factor,
            cpu_hit_blocks_mean_factor=cpu_hit_blocks_mean_factor,
        )
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
        if role in C2LB_ROLES and req_info is not None:
            selected = C2LBPolicy.select_endpoint_from_list(
                instances,
                req_info,
                active_tokens_mean_factor=self._active_tokens_mean_factor,
                cpu_hit_blocks_mean_factor=self._cpu_hit_blocks_mean_factor,
            )
            if selected is not None:
                return selected
        from motor.coordinator.scheduler.policy.load_balance import LoadBalancePolicy

        return LoadBalancePolicy.select_endpoint_from_list(instances, role)
