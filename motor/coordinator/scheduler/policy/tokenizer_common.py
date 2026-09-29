# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Shared tokenizer behavior for token-based scheduling policies."""

import json
import time
from pathlib import Path

from motor.common.logger import get_logger
from motor.coordinator.domain.block_offset_translator import (
    _DSV4_ASSISTANT_MARKER,
    _DSV4_EOS_MARKER,
    _DSV4_USER_MARKER,
)
from motor.coordinator.scheduler.policy.utils import (
    preprocess_input,
    preprocess_messages_for_dsv4,
    preprocess_messages_for_standard,
)

logger = get_logger(__name__)

TOKENIZER_LOAD_RETRY_SECONDS = 30.0


class SchedulingTokenizerMixin:
    """Tokenizer implementation shared by KV affinity and C2LB singleton wrappers."""

    tokenizer: object | None
    model_path: str
    engine_type: str
    openai_standard: str
    _is_dsv4: bool
    _next_load_attempt_at: float

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
                if self._is_deepseek_v4_model(self.model_path):
                    try:
                        from vllm.tokenizers.deepseek_v4 import (  # pylint: disable=import-error,no-name-in-module
                            DeepseekV4Tokenizer,
                        )
                    except ImportError as exc:
                        logger.warning(
                            "DeepseekV4Tokenizer unavailable for engine_type=%s (%s); "
                            "falling back to AutoTokenizer",
                            self.engine_type,
                            exc,
                        )
                        from transformers import AutoTokenizer

                        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                        self._is_dsv4 = True
                    else:
                        self.tokenizer = DeepseekV4Tokenizer.from_pretrained(
                            self.model_path, trust_remote_code=True
                        )
                        self._is_dsv4 = True
                else:
                    from transformers import AutoTokenizer

                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            except Exception as exc:
                self._next_load_attempt_at = time.monotonic() + TOKENIZER_LOAD_RETRY_SECONDS
                logger.warning(
                    "Scheduling tokenizer load failed; retrying in %.0fs: %s",
                    TOKENIZER_LOAD_RETRY_SECONDS,
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
                logger.error("DSV4 scheduling tokenize failed; returning []: %s", exc)
                return []
            logger.warning("Primary scheduling tokenize path failed: %s; trying fallback", exc)
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
        config_dict = SchedulingTokenizerMixin._read_model_config_dict(model_path)
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
        if not getattr(self.tokenizer, "chat_template", None):
            return self._encode_dsv4_messages(messages)
        result = self.tokenizer.apply_chat_template(
            messages, tools=tools, **self._build_dsv4_chat_template_kwargs(req_data)
        )
        return result if isinstance(result, list) else self.tokenizer.encode(result, add_special_tokens=False)

    def _encode_dsv4_messages(self, messages: list) -> list[int]:
        """Encode DSV4 markers when model weights do not provide a chat template."""
        if str(getattr(self, "engine_type", "vllm") or "vllm").strip().lower() == "sglang":
            return self._encode_dsv4_messages_sglang(messages)
        return self._encode_dsv4_messages_vllm(messages)

    def _encode_dsv4_messages_vllm(self, messages: list) -> list[int]:
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            if role in ("user", "system", "developer", "tool"):
                parts.append(f"{_DSV4_USER_MARKER}{content}")
            elif role == "assistant":
                parts.append(f"{_DSV4_ASSISTANT_MARKER}{content}{_DSV4_EOS_MARKER}")
        if not messages or (isinstance(messages[-1], dict) and messages[-1].get("role") != "assistant"):
            parts.append(_DSV4_ASSISTANT_MARKER)
        return self.tokenizer.encode("".join(parts), add_special_tokens=False)

    def _encode_dsv4_messages_sglang(self, messages: list) -> list[int]:
        """Match SGLang ``encoding_dsv4`` when no model chat template is available."""
        normalized = [message for message in messages if isinstance(message, dict)]
        parts: list[str] = ["<｜begin▁of▁sentence｜>"]
        generation_close = f"{_DSV4_ASSISTANT_MARKER}</think>"
        for index, message in enumerate(normalized):
            role = message.get("role")
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            next_role = normalized[index + 1].get("role") if index + 1 < len(normalized) else None
            if role == "system":
                parts.append(content)
            elif role in ("user", "developer", "tool"):
                parts.append(f"{_DSV4_USER_MARKER}{content}")
                if next_role is None or next_role in ("assistant", "latest_reminder"):
                    parts.append(generation_close)
            elif role == "assistant":
                parts.append(f"{content}{_DSV4_EOS_MARKER}")
        if not normalized:
            parts.append(generation_close)
        return self.tokenizer.encode("".join(parts), add_special_tokens=False)

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
            logger.error("Scheduling tokenize failed on primary and fallback paths; returning []: %s", exc)
            return []
