# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Tokenizer shared by scheduler features that need prompt token IDs."""

import json
import os
import threading
from pathlib import Path

from motor.common.logger import get_logger
from motor.common.utils.singleton import ThreadSafeSingleton
from motor.config.coordinator import CONTEXT_BUDGET_ON, CoordinatorConfig
from motor.coordinator.scheduler.policy.utils import (
    preprocess_input,
    preprocess_messages_for_dsv4,
    preprocess_messages_for_standard,
)

logger = get_logger(__name__)


class TokenizerManager(ThreadSafeSingleton):
    """Process-wide tokenizer shared without coupling scheduling policies."""

    def __init__(self, config: CoordinatorConfig | None = None):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.config_lock = threading.RLock()
        if config is None:
            config = CoordinatorConfig()

        self.endpoint = config.tracer_config.endpoint
        self.tokenizer = None
        self._is_dsv4 = False
        scheduler_config = getattr(config, "scheduler_config", None)
        kv_config = (
            getattr(scheduler_config, "kv_conductor_config", None)
            if scheduler_config
            else None
        )
        if kv_config is None:
            kv_config = getattr(config, "prefill_kv_event_config", None)
        scheduler_type = (
            getattr(scheduler_config, "scheduler_type", None)
            if scheduler_config
            else None
        )
        scheduler_value = getattr(scheduler_type, "value", scheduler_type)
        needs_tokenizer = bool(
            (kv_config and getattr(kv_config, "conductor_service", ""))
            or scheduler_value in ("kv_cache_affinity", "smetric")
            or config.context_budget_mode == CONTEXT_BUDGET_ON
        )
        if not needs_tokenizer:
            logger.info(
                "Token-based scheduling and context budget are disabled; tokenizer disabled"
            )
            return

        model_path = getattr(kv_config, "model_path", "") if kv_config else ""
        if model_path:
            os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
            engine_type = (
                str(getattr(kv_config, "engine_type", "vllm") or "vllm").strip().lower()
            )
            if engine_type == "vllm" and self._is_deepseek_v4_model(model_path):
                from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer

                self.tokenizer = DeepseekV4Tokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
                self._is_dsv4 = True
            else:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
        self.openai_standard = os.environ.get("OPENAI_STANDARD", "STANDARD")
        logger.info(
            "TokenizerManager init.(model_path:%s, is_dsv4:%s)",
            model_path,
            self._is_dsv4,
        )

    def apply_chat_template(
        self, messages: list, tools: list | None = None, req_data: dict | None = None
    ) -> list[int]:
        if self.tokenizer is None:
            return []
        try:
            if self._is_dsv4:
                return self._apply_chat_template_dsv4(messages, tools, req_data)
            if self.openai_standard != "STANDARD":
                return self._apply_chat_template_with_preprocess(
                    messages, tools, req_data
                )
            return self._apply_chat_template_standard(messages, tools, req_data)
        except Exception as exc:  # noqa: BLE001 - tokenizer backends raise heterogeneous errors
            if self._is_dsv4:
                logger.error("dsv4 tokenize failed; returning []: %s", exc)
                return []
            logger.warning("primary tokenize path failed: %s; trying fallback", exc)
            return self._safe_fallback_encode(messages, tools, req_data)

    def encode(self, prompt: str) -> list[int]:
        return [] if self.tokenizer is None else self.tokenizer.encode(prompt)

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
        config_dict = TokenizerManager._read_model_config_dict(model_path)
        if not config_dict:
            return False
        return config_dict.get(
            "model_type"
        ) == "deepseek_v4" or "DeepseekV4ForCausalLM" in (
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
    def _build_standard_chat_template_kwargs(
        req_data: dict | None, *, tokenize: bool
    ) -> dict:
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
            kwargs.update(
                {
                    key: value
                    for key, value in template_kwargs.items()
                    if key not in reserved
                }
            )
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

    def _apply_chat_template_dsv4(
        self, messages: list, tools: list | None, req_data: dict | None
    ) -> list[int]:
        messages, tools = preprocess_messages_for_dsv4(messages, tools)
        result = self.tokenizer.apply_chat_template(
            messages, tools=tools, **self._build_dsv4_chat_template_kwargs(req_data)
        )
        return (
            result
            if isinstance(result, list)
            else self.tokenizer.encode(result, add_special_tokens=False)
        )

    def _apply_chat_template_standard(
        self, messages: list, tools: list | None, req_data: dict | None
    ) -> list[int]:
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

    def _safe_fallback_encode(
        self, messages: list, tools: list | None, req_data: dict | None
    ) -> list[int]:
        try:
            if self.openai_standard == "STANDARD":
                return self._apply_chat_template_with_preprocess(
                    messages, tools, req_data
                )
            return self._apply_chat_template_standard(messages, tools, req_data)
        except Exception as exc:  # noqa: BLE001 - fallback must contain backend-specific failures
            logger.error(
                "tokenize failed on both primary and fallback paths; returning []: %s",
                exc,
            )
            return []
