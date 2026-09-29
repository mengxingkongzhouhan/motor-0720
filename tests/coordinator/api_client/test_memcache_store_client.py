# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Tests for coordinator-side MemCache SSD-to-DRAM prefetch."""

import sys
from unittest.mock import Mock, patch

import pytest

from motor.coordinator.api_client.memcache_store_client import MemcacheStoreClient


@pytest.fixture(autouse=True)
def _reset_client():
    MemcacheStoreClient.reset_for_tests()
    yield
    MemcacheStoreClient.reset_for_tests()


def test_prefetch_skips_empty_or_non_memcache():
    with patch.object(MemcacheStoreClient, "_executor") as executor:
        MemcacheStoreClient.prefetch_disk_blocks([])
        with patch.object(MemcacheStoreClient, "_is_memcache_backend", return_value=False):
            MemcacheStoreClient.prefetch_disk_blocks(["object-key"])
    executor.submit.assert_not_called()


def test_prefetch_queues_background_work():
    with (
        patch.object(MemcacheStoreClient, "_is_memcache_backend", return_value=True),
        patch.object(MemcacheStoreClient, "_executor") as executor,
    ):
        MemcacheStoreClient.prefetch_disk_blocks(["key-a", "", 123, "key-b"])
    executor.submit.assert_called_once_with(
        MemcacheStoreClient._prefetch_disk_blocks_sync,
        ["key-a", "key-b"],
    )


def test_background_prefetch_uses_ssd_to_dram_and_is_fail_open():
    store = Mock()
    store.prefetch.return_value = 0
    MemcacheStoreClient._store = store
    MemcacheStoreClient._prefetch_disk_blocks_sync(["key-a"])
    store.prefetch.assert_called_once_with(["key-a"], src_media=2, dst_media=1, flags=0)

    store.prefetch.side_effect = RuntimeError("meta down")
    MemcacheStoreClient._prefetch_disk_blocks_sync(["key-b"])


def test_connect_uses_client_only_init(monkeypatch):
    monkeypatch.setattr(MemcacheStoreClient, "_meta_service_url", staticmethod(lambda: "tcp://host:50088"))
    store = Mock()
    store.setup.return_value = 0
    store.init.return_value = 0
    config = Mock()
    hybrid = Mock()
    hybrid.DistributedObjectStore.return_value = store
    hybrid.LocalConfig.return_value = config

    with patch.dict(sys.modules, {"memcache_hybrid": hybrid}):
        assert MemcacheStoreClient._connect() is store

    assert config.meta_service_url == "tcp://host:50088"
    store.setup.assert_called_once_with(config)
    store.init.assert_called_once_with(0, init_bm=False)


def test_meta_service_url_uses_env_and_formats_ipv6(monkeypatch):
    monkeypatch.setenv("KVS_MASTER_SERVICE", "fd00::1")
    monkeypatch.setenv("KV_CACHE_STORE_PORT", "50099")
    assert MemcacheStoreClient._meta_service_url() == "tcp://[fd00::1]:50099"
