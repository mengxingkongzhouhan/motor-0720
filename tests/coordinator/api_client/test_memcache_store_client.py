# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Tests for coordinator-side MemCache SSD→DRAM prefetch."""

from unittest.mock import Mock, patch

import pytest

from motor.coordinator.api_client.memcache_store_client import MemcacheStoreClient


@pytest.fixture(autouse=True)
def _reset_store_client():
    MemcacheStoreClient.reset_for_tests()
    yield
    MemcacheStoreClient.reset_for_tests()


def test_hashes_to_keys_decimal_strings():
    assert MemcacheStoreClient.hashes_to_keys([201, "202", 203]) == ["201", "202", "203"]
    assert MemcacheStoreClient.hashes_to_keys(None) == []
    assert MemcacheStoreClient.hashes_to_keys(["x", None, 1.5]) == ["1"]


def test_prefetch_skips_empty_keys():
    assert MemcacheStoreClient.prefetch_disk_blocks([]) is False
    assert MemcacheStoreClient.prefetch_disk_blocks(None) is False


@patch.object(MemcacheStoreClient, "_is_memcache_backend", return_value=False)
def test_prefetch_skips_non_memcache_backend(_mock_backend):
    assert MemcacheStoreClient.prefetch_disk_blocks([201, 202]) is False


@patch.object(MemcacheStoreClient, "_is_memcache_backend", return_value=True)
def test_prefetch_calls_store_ssd_to_dram(_mock_backend):
    store = Mock()
    store.prefetch.return_value = 0
    MemcacheStoreClient._store = store

    assert MemcacheStoreClient.prefetch_disk_blocks([201, 202]) is True
    store.prefetch.assert_called_once_with(["201", "202"], src_media=2, dst_media=1, flags=0)


@patch.object(MemcacheStoreClient, "_is_memcache_backend", return_value=True)
def test_prefetch_nonzero_rc_is_fail_open(_mock_backend):
    store = Mock()
    store.prefetch.return_value = 7
    MemcacheStoreClient._store = store
    assert MemcacheStoreClient.prefetch_disk_blocks([201]) is False


@patch.object(MemcacheStoreClient, "_is_memcache_backend", return_value=True)
def test_prefetch_exception_is_fail_open(_mock_backend):
    store = Mock()
    store.prefetch.side_effect = RuntimeError("meta down")
    MemcacheStoreClient._store = store
    assert MemcacheStoreClient.prefetch_disk_blocks([201]) is False


@patch.object(MemcacheStoreClient, "_is_memcache_backend", return_value=True)
def test_prefetch_skips_when_store_unavailable(_mock_backend):
    MemcacheStoreClient._init_failed = True
    assert MemcacheStoreClient.prefetch_disk_blocks([201]) is False


def test_is_memcache_backend_case_insensitive():
    with patch(
        "motor.coordinator.api_client.conductor_api_client.ConductorApiClient._resolve_store_backend",
        return_value="Memcache",
    ):
        assert MemcacheStoreClient._is_memcache_backend() is True
    with patch(
        "motor.coordinator.api_client.conductor_api_client.ConductorApiClient._resolve_store_backend",
        return_value="Mooncake",
    ):
        assert MemcacheStoreClient._is_memcache_backend() is False


def test_meta_service_url_from_env(monkeypatch):
    monkeypatch.setenv("KVS_MASTER_SERVICE", "kv_store_service")
    monkeypatch.delenv("KV_CACHE_STORE_PORT", raising=False)
    assert MemcacheStoreClient._meta_service_url() == "tcp://kv_store_service:50088"


def test_meta_service_url_custom_port_and_ipv6(monkeypatch):
    monkeypatch.setenv("KVS_MASTER_SERVICE", "fd00::1")
    monkeypatch.setenv("KV_CACHE_STORE_PORT", "50099")
    assert MemcacheStoreClient._meta_service_url() == "tcp://[fd00::1]:50099"


def test_meta_service_url_empty_without_host(monkeypatch):
    monkeypatch.setenv("KVS_MASTER_SERVICE", "")
    with patch(
        "motor.coordinator.api_client.memcache_store_client.CoordinatorConfig.from_json",
        side_effect=RuntimeError("no config"),
    ):
        assert MemcacheStoreClient._meta_service_url() is None


def test_connect_uses_client_only_init(monkeypatch):
    monkeypatch.setattr(MemcacheStoreClient, "_meta_service_url", staticmethod(lambda: "tcp://host:50088"))
    store = Mock()
    store.setup.return_value = 0
    store.init.return_value = 0
    config = Mock()
    hybrid = Mock()
    hybrid.DistributedObjectStore.return_value = store
    hybrid.LocalConfig.return_value = config

    import sys

    with patch.dict(sys.modules, {"memcache_hybrid": hybrid}):
        connected = MemcacheStoreClient._connect()

    assert connected is store
    assert config.meta_service_url == "tcp://host:50088"
    store.setup.assert_called_once_with(config)
    store.init.assert_called_once_with(0, init_bm=False)


def test_connect_fails_when_init_returns_error(monkeypatch):
    monkeypatch.setattr(MemcacheStoreClient, "_meta_service_url", staticmethod(lambda: "tcp://host:50088"))
    store = Mock()
    store.setup.return_value = 0
    store.init.return_value = 3
    hybrid = Mock()
    hybrid.DistributedObjectStore.return_value = store
    hybrid.LocalConfig.return_value = Mock()

    import sys

    with patch.dict(sys.modules, {"memcache_hybrid": hybrid}):
        assert MemcacheStoreClient._connect() is None
