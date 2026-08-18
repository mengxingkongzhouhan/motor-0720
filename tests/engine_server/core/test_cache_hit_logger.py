# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Per-request vLLM local/remote cache-hit log extraction."""

import sys
from types import ModuleType, SimpleNamespace

import motor.engine_server.core.vllm.cache_hit_logger as cache_hit_logger_module
from motor.engine_server.core.vllm.cache_hit_logger import (
    CacheHitRecord,
    hits_from_prefill_stats,
    hits_from_usage,
    log_from_engine_core_outputs,
    log_from_openai_body,
    lookup_engine_hits,
    maybe_log_from_stream_chunk,
    remember_engine_hits,
    reset_cache_hit_logger_state,
)


def setup_function() -> None:
    reset_cache_hit_logger_state()


def test_hits_from_prefill_stats_splits_local_and_remote():
    stats = SimpleNamespace(
        num_local_cached_tokens=64,
        num_external_cached_tokens=32,
        num_cached_tokens=96,
        num_prompt_tokens=200,
    )
    record = hits_from_prefill_stats(stats)
    assert record == CacheHitRecord(local_hit=64, remote_hit=32, cached=96, prompt=200)


def test_hits_from_prefill_stats_fills_cached_from_split():
    stats = SimpleNamespace(
        num_local_cached_tokens=8,
        num_external_cached_tokens=4,
        num_cached_tokens=None,
        num_prompt_tokens=20,
    )
    record = hits_from_prefill_stats(stats)
    assert record is not None
    assert record.cached == 12


def test_hits_from_usage_is_total_only():
    record = hits_from_usage({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 40}})
    assert record == CacheHitRecord(cached=40, prompt=100)
    assert record.local_hit is None
    assert record.remote_hit is None


def test_log_from_engine_core_outputs_maps_vllm_internal_id_to_external_id(caplog):
    internal_req_id = "req#a1-deadbeef"
    output = SimpleNamespace(
        request_id=internal_req_id,
        prefill_stats=SimpleNamespace(
            num_local_cached_tokens=16,
            num_external_cached_tokens=8,
            num_cached_tokens=24,
            num_prompt_tokens=80,
        ),
    )
    output_processor = SimpleNamespace(
        request_states={
            internal_req_id: SimpleNamespace(external_req_id="req#a1"),
        }
    )
    with caplog.at_level("INFO"):
        log_from_engine_core_outputs([output], output_processor=output_processor)
    stored = lookup_engine_hits("req#a1")
    assert stored is not None
    assert stored.local_hit == 16
    assert stored.remote_hit == 8
    assert "vllm_cache_hit: req_id=- engine_req_id=req#a1 local_hit=16 remote_hit=8 cached=24 prompt=80" in caplog.text
    assert lookup_engine_hits(internal_req_id) is None


def test_randomized_vllm_id_correlates_split_with_motor_req_id(caplog):
    internal_req_id = "req#a1-deadbeef"
    output = SimpleNamespace(
        request_id=internal_req_id,
        prefill_stats=SimpleNamespace(
            num_local_cached_tokens=16,
            num_external_cached_tokens=8,
            num_cached_tokens=24,
            num_prompt_tokens=80,
        ),
    )
    output_processor = SimpleNamespace(
        request_states={
            internal_req_id: SimpleNamespace(external_req_id="req#a1"),
        }
    )

    log_from_engine_core_outputs([output], output_processor=output_processor)
    with caplog.at_level("INFO"):
        log_from_openai_body(
            {"usage": {"prompt_tokens": 80, "prompt_tokens_details": {"cached_tokens": 24}}},
            root_req_id="req-match-log",
            engine_req_id="req#a1",
        )

    assert (
        "vllm_cache_hit: req_id=req-match-log engine_req_id=req#a1 "
        "local_hit=16 remote_hit=8 cached=24 prompt=80" in caplog.text
    )


def test_install_wrapper_passes_output_processor_for_external_id_lookup(monkeypatch):
    class FakeOutputProcessor:
        def process_outputs(self, engine_core_outputs):
            return engine_core_outputs

    output_processor_module = ModuleType("vllm.v1.engine.output_processor")
    output_processor_module.OutputProcessor = FakeOutputProcessor
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", ModuleType("vllm.v1.engine"))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.output_processor", output_processor_module)
    monkeypatch.setattr(cache_hit_logger_module, "_INSTALLED", False)
    captured = {}

    def capture(engine_core_outputs, *, output_processor=None):
        captured["outputs"] = engine_core_outputs
        captured["processor"] = output_processor

    monkeypatch.setattr(cache_hit_logger_module, "log_from_engine_core_outputs", capture)

    assert cache_hit_logger_module.install_vllm_cache_hit_logger() is True
    processor = FakeOutputProcessor()
    outputs = [SimpleNamespace(request_id="req#a1-deadbeef")]
    assert processor.process_outputs(outputs) is outputs
    assert captured == {"outputs": outputs, "processor": processor}


def test_log_from_engine_core_outputs_accepts_outputs_container(caplog):
    output = SimpleNamespace(
        request_id="req#a1",
        prefill_stats=SimpleNamespace(
            num_local_cached_tokens=4,
            num_external_cached_tokens=2,
            num_cached_tokens=6,
            num_prompt_tokens=10,
        ),
    )
    with caplog.at_level("INFO"):
        log_from_engine_core_outputs(SimpleNamespace(outputs=[output]))
    stored = lookup_engine_hits("req#a1")
    assert stored is not None
    assert stored.local_hit == 4
    assert stored.remote_hit == 2
    assert "local_hit=4 remote_hit=2 cached=6 prompt=10" in caplog.text


def test_hits_from_prefill_stats_zero_defaults_are_hits_not_missing():
    """vLLM 0.23.0 PrefillStats fields default to int 0, not None."""
    stats = SimpleNamespace(
        num_local_cached_tokens=0,
        num_external_cached_tokens=0,
        num_cached_tokens=0,
        num_prompt_tokens=32,
    )
    record = hits_from_prefill_stats(stats)
    assert record == CacheHitRecord(local_hit=0, remote_hit=0, cached=0, prompt=32)


def test_log_from_openai_body_correlates_motor_req_id(caplog):
    remember_engine_hits(
        "req#a1",
        CacheHitRecord(local_hit=16, remote_hit=8, cached=24, prompt=80),
    )
    with caplog.at_level("INFO"):
        log_from_openai_body(
            {"usage": {"prompt_tokens": 80, "prompt_tokens_details": {"cached_tokens": 24}}},
            root_req_id="req-match-log",
            engine_req_id="req#a1",
        )
    assert (
        "vllm_cache_hit: req_id=req-match-log engine_req_id=req#a1 "
        "local_hit=16 remote_hit=8 cached=24 prompt=80" in caplog.text
    )


def test_log_from_openai_body_falls_back_to_usage_total(caplog):
    with caplog.at_level("INFO"):
        log_from_openai_body(
            {"usage": {"prompt_tokens": 50, "prompt_tokens_details": {"cached_tokens": 12}}},
            root_req_id="req-usage-only",
            engine_req_id="req#b2",
        )
    assert (
        "vllm_cache_hit: req_id=req-usage-only engine_req_id=req#b2 "
        "local_hit=- remote_hit=- cached=12 prompt=50" in caplog.text
    )


def test_stream_chunk_logs_once(caplog):
    remember_engine_hits("req#a1", CacheHitRecord(local_hit=4, remote_hit=2, cached=6, prompt=10))
    state: dict = {}
    chunk = b'data: {"usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 6}}}\n\n'
    with caplog.at_level("INFO"):
        maybe_log_from_stream_chunk(chunk, root_req_id="root", engine_req_id="req#a1", state=state)
        maybe_log_from_stream_chunk(chunk, root_req_id="root", engine_req_id="req#a1", state=state)
    assert caplog.text.count("vllm_cache_hit: req_id=root engine_req_id=req#a1") == 1
    assert "local_hit=4 remote_hit=2 cached=6 prompt=10" in caplog.text
