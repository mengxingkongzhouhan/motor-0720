# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Per-request vLLM local/remote prefix-cache hit logging for Motor correlation.

Targets **vLLM 0.23.0** (Motor's pinned engine; also vLLM-Ascend ``releases/v0.23.0``).
Does not patch vLLM source: wraps ``OutputProcessor.process_outputs`` in-process.

SMetric logs conductor match lengths keyed by ``req_id``. This module prints the
same request's vLLM hits so the two can be grepped together:

* ``local_hit`` — GPU/NPU prefix cache (``PrefillStats.num_local_cached_tokens``)
* ``remote_hit`` — external KV connector (``PrefillStats.num_external_cached_tokens``)

v0.23.0 contract used here:

* ``OutputProcessor.process_outputs(self, engine_core_outputs: list[EngineCoreOutput], ...)``
* ``EngineCoreOutput.prefill_stats: PrefillStats | None``
* ``EngineCoreOutput.request_id`` is internal/randomized; the external ID is read from
  ``OutputProcessor.request_states[request_id].external_req_id``
* ``PrefillStats`` fields ``num_local_cached_tokens`` / ``num_external_cached_tokens`` /
  ``num_cached_tokens`` / ``num_prompt_tokens`` (ints, default 0)

The OpenAI ``usage.prompt_tokens_details.cached_tokens`` field is only a total, so
the EngineCore ``prefill_stats`` path is preferred. The dispatch adapter then
re-emits the line with Motor's ``root_request_id``.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from motor.common.logger import get_logger

logger = get_logger(__name__)

UNKNOWN = "-"
_MAX_HIT_RECORDS = 4096
_HITS_LOCK = threading.Lock()
_INSTALLED = False
_STREAM_LOGGED_KEY = "vllm_cache_hit_logged"


@dataclass(frozen=True)
class CacheHitRecord:
    """One request's prefix-cache accounting. None means the source did not split that field."""

    local_hit: int | None = None
    remote_hit: int | None = None
    cached: int | None = None
    prompt: int | None = None

    def merge(self, other: CacheHitRecord) -> CacheHitRecord:
        """Fill missing fields from *other* without overwriting a split already recorded."""
        local_hit = self.local_hit if self.local_hit is not None else other.local_hit
        remote_hit = self.remote_hit if self.remote_hit is not None else other.remote_hit
        cached = self.cached if self.cached is not None else other.cached
        prompt = self.prompt if self.prompt is not None else other.prompt
        if cached is None and local_hit is not None and remote_hit is not None:
            cached = local_hit + remote_hit
        return CacheHitRecord(local_hit=local_hit, remote_hit=remote_hit, cached=cached, prompt=prompt)


_HITS_BY_ENGINE_REQ: OrderedDict[str, CacheHitRecord] = OrderedDict()


def format_hit(value: int | None) -> str:
    return UNKNOWN if value is None else str(int(value))


def log_vllm_cache_hit(
    *,
    req_id: str | None,
    engine_req_id: str | None,
    record: CacheHitRecord,
) -> None:
    logger.info(
        "vllm_cache_hit: req_id=%s engine_req_id=%s local_hit=%s remote_hit=%s cached=%s prompt=%s",
        req_id or UNKNOWN,
        engine_req_id or UNKNOWN,
        format_hit(record.local_hit),
        format_hit(record.remote_hit),
        format_hit(record.cached),
        format_hit(record.prompt),
    )


def remember_engine_hits(engine_req_id: str, record: CacheHitRecord) -> None:
    if not engine_req_id:
        return
    with _HITS_LOCK:
        existing = _HITS_BY_ENGINE_REQ.get(engine_req_id)
        merged = existing.merge(record) if existing is not None else record
        _HITS_BY_ENGINE_REQ[engine_req_id] = merged
        _HITS_BY_ENGINE_REQ.move_to_end(engine_req_id)
        while len(_HITS_BY_ENGINE_REQ) > _MAX_HIT_RECORDS:
            _HITS_BY_ENGINE_REQ.popitem(last=False)


def lookup_engine_hits(engine_req_id: str | None) -> CacheHitRecord | None:
    if not engine_req_id:
        return None
    with _HITS_LOCK:
        record = _HITS_BY_ENGINE_REQ.get(engine_req_id)
        if record is not None:
            return record
        for candidate in _engine_req_aliases(engine_req_id):
            record = _HITS_BY_ENGINE_REQ.get(candidate)
            if record is not None:
                return record
    return None


def hits_from_prefill_stats(stats: Any) -> CacheHitRecord | None:
    if stats is None:
        return None
    local_hit = _optional_int(getattr(stats, "num_local_cached_tokens", None))
    remote_hit = _optional_int(getattr(stats, "num_external_cached_tokens", None))
    cached = _optional_int(getattr(stats, "num_cached_tokens", None))
    prompt = _optional_int(getattr(stats, "num_prompt_tokens", None))
    if local_hit is None and remote_hit is None and cached is None:
        return None
    if cached is None and local_hit is not None and remote_hit is not None:
        cached = local_hit + remote_hit
    return CacheHitRecord(local_hit=local_hit, remote_hit=remote_hit, cached=cached, prompt=prompt)


def hits_from_usage(usage: Any) -> CacheHitRecord | None:
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details")
    cached = None
    if isinstance(details, dict):
        cached = _optional_int(details.get("cached_tokens"))
    prompt = _optional_int(usage.get("prompt_tokens"))
    if cached is None and prompt is None:
        return None
    return CacheHitRecord(cached=cached, prompt=prompt)


def log_from_engine_core_outputs(
    engine_core_outputs: Any,
    *,
    output_processor: Any = None,
) -> None:
    """Store+log PrefillStats from an EngineCore output batch (frontend process).

    vLLM 0.23.0 passes ``list[EngineCoreOutput]`` as ``process_outputs``' first
    argument. ``EngineCoreOutputs`` (``.outputs`` list) is also accepted.
    """
    for output in _iter_engine_core_outputs(engine_core_outputs):
        record = hits_from_prefill_stats(getattr(output, "prefill_stats", None))
        if record is None:
            continue
        internal_req_id = _output_request_id(output)
        engine_req_id = _external_request_id(output_processor, internal_req_id)
        if engine_req_id is None:
            engine_req_id = _output_external_request_id(output) or internal_req_id
        if engine_req_id:
            remember_engine_hits(engine_req_id, record)
        log_vllm_cache_hit(req_id=None, engine_req_id=engine_req_id, record=record)


def log_from_openai_body(
    body: dict[str, Any] | None,
    *,
    root_req_id: str | None,
    engine_req_id: str | None,
) -> None:
    """Emit the Motor-correlated line once the OpenAI/dispatch response is in hand."""
    record = lookup_engine_hits(engine_req_id)
    usage_record = hits_from_usage(_usage_from_body(body))
    if record is None:
        record = usage_record
    elif usage_record is not None:
        record = record.merge(usage_record)
    if record is None:
        return
    log_vllm_cache_hit(req_id=root_req_id, engine_req_id=engine_req_id, record=record)


def maybe_log_from_stream_chunk(
    chunk: bytes | str,
    *,
    root_req_id: str | None,
    engine_req_id: str | None,
    state: dict[str, Any],
) -> None:
    if state.get(_STREAM_LOGGED_KEY):
        return
    body = _parse_stream_chunk_json(chunk)
    if body is None:
        return
    usage_record = hits_from_usage(body.get("usage"))
    stored = lookup_engine_hits(engine_req_id)
    if stored is None and usage_record is None:
        return
    record = stored.merge(usage_record) if stored is not None and usage_record is not None else (stored or usage_record)
    if record is None:
        return
    state[_STREAM_LOGGED_KEY] = True
    log_vllm_cache_hit(req_id=root_req_id, engine_req_id=engine_req_id, record=record)


def install_vllm_cache_hit_logger() -> bool:
    """Wrap vLLM 0.23.0 ``OutputProcessor.process_outputs`` to capture PrefillStats."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from vllm.v1.engine.output_processor import OutputProcessor
    except ImportError:
        logger.debug("vLLM OutputProcessor not available; per-request cache hit hook skipped")
        return False
    original = OutputProcessor.process_outputs
    if getattr(original, "_motor_cache_hit_wrapped", False):
        _INSTALLED = True
        return True

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        engine_core_outputs = args[0] if args else kwargs.get("engine_core_outputs")
        if engine_core_outputs is not None:
            try:
                log_from_engine_core_outputs(engine_core_outputs, output_processor=self)
            except Exception as exc:  # pragma: no cover - never fail serving over a debug log
                logger.debug("vllm cache hit log failed: %s", exc)
        return original(self, *args, **kwargs)

    wrapped._motor_cache_hit_wrapped = True  # type: ignore[attr-defined]
    OutputProcessor.process_outputs = wrapped  # type: ignore[method-assign]
    _INSTALLED = True
    logger.info("Installed vLLM per-request local/remote cache hit logger")
    return True


def reset_cache_hit_logger_state() -> None:
    """Test helper: drop remembered hits. Does not uninstall the OutputProcessor wrap."""
    with _HITS_LOCK:
        _HITS_BY_ENGINE_REQ.clear()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_from_body(body: dict[str, Any] | None) -> Any:
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    if isinstance(usage, dict):
        return usage
    payload = body.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
        return payload["usage"]
    return None


def _iter_engine_core_outputs(engine_core_outputs: Any) -> list[Any]:
    """Yield EngineCoreOutput items from a v0.23.0 list or an ``.outputs`` wrapper."""
    if engine_core_outputs is None:
        return []
    outputs = getattr(engine_core_outputs, "outputs", engine_core_outputs)
    if outputs is None:
        return []
    if isinstance(outputs, list | tuple):
        return list(outputs)
    try:
        return list(outputs)
    except TypeError:
        return []


def _output_request_id(output: Any) -> str | None:
    value = getattr(output, "request_id", None)
    if isinstance(value, str) and value:
        return value
    return None


def _output_external_request_id(output: Any) -> str | None:
    """Compatibility fallback for output types that expose the external ID directly."""
    value = getattr(output, "external_req_id", None)
    if isinstance(value, str) and value:
        return value
    return None


def _external_request_id(output_processor: Any, internal_req_id: str | None) -> str | None:
    """Resolve vLLM 0.23.0's randomized internal ID through OutputProcessor state."""
    if output_processor is None or not internal_req_id:
        return None
    request_states = getattr(output_processor, "request_states", None)
    if request_states is None:
        return None
    try:
        request_state = request_states.get(internal_req_id)
    except (AttributeError, TypeError):
        return None
    value = getattr(request_state, "external_req_id", None)
    if isinstance(value, str) and value:
        return value
    return None


def _engine_req_aliases(engine_req_id: str) -> tuple[str, ...]:
    aliases: list[str] = []
    for prefix in ("chatcmpl-", "cmpl-"):
        if engine_req_id.startswith(prefix):
            aliases.append(engine_req_id.removeprefix(prefix))
        else:
            aliases.append(prefix + engine_req_id)
    return tuple(aliases)


def _parse_stream_chunk_json(chunk: bytes | str) -> dict[str, Any] | None:
    try:
        text = chunk.decode("utf-8").strip() if isinstance(chunk, bytes | bytearray) else chunk.strip()
    except UnicodeDecodeError:
        return None
    if text.startswith("data: "):
        text = text[len("data: ") :]
    if not text or text == "[DONE]":
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
