# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Coordinator-side MemCache client for SSD→DRAM prefetch.

Wraps ``memcache_hybrid.DistributedObjectStore.prefetch``
(https://gitcode.com/Ascend/memcache docs: ``store.prefetch(keys, src_media=2,
dst_media=1)``). The coordinator uses client-only init (``init_bm=False``) so
it does not allocate a local DRAM/HBM pool.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable

from motor.common.logger import get_logger
from motor.common.utils.env import Env
from motor.common.utils.net import format_host
from motor.config.coordinator import CoordinatorConfig

logger = get_logger(__name__)

# MemCache MetaService RPC port (same default as KVCacheStoreConfig.port).
_DEFAULT_META_PORT = 50088
# SSD / DRAM media ids from the MemCache Python API.
_SSD_MEDIA = 2
_DRAM_MEDIA = 1


class MemcacheStoreClient:
    """Lazy MemCache client used by the coordinator to prefetch SSD-resident KV."""

    _lock = threading.Lock()
    _store: Any = None
    _init_failed: bool = False

    @staticmethod
    def hashes_to_keys(block_hashes: Iterable[Any] | None) -> list[str]:
        """Turn conductor ``disk_block_hashes`` (u64) into MemCache string keys."""
        keys: list[str] = []
        for raw in block_hashes or []:
            try:
                keys.append(str(int(raw)))
            except (TypeError, ValueError):
                continue
        return keys

    @classmethod
    def prefetch_disk_blocks(cls, block_hashes: Iterable[Any] | None) -> bool:
        """Prefetch exclusive SSD-hit blocks into DRAM.

        ``keys`` are the conductor ``disk_block_hashes`` values. Calls
        ``store.prefetch(keys, src_media=2, dst_media=1, flags=0)``
        (SSD→DRAM; the only combination MemCache currently supports).

        Fail-open: empty input, a non-memcache backend, a missing package, or
        a store error logs and returns ``False`` — scheduling is not affected.
        """
        keys = cls.hashes_to_keys(block_hashes)
        if not keys:
            return False
        if not cls._is_memcache_backend():
            logger.debug("skip memcache prefetch: store_backend is not memcache")
            return False
        store = cls._get_store()
        if store is None:
            return False
        try:
            result = store.prefetch(keys, src_media=_SSD_MEDIA, dst_media=_DRAM_MEDIA, flags=0)
        except Exception as exc:  # noqa: BLE001 — fail-open on the schedule path
            logger.warning("memcache prefetch raised keys=%d: %s", len(keys), exc)
            return False
        if result != 0:
            logger.warning("memcache prefetch failed rc=%s keys=%d", result, len(keys))
            return False
        logger.info("memcache prefetch submitted keys=%d", len(keys))
        return True

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop the cached store so unit tests can re-bind a mock."""
        with cls._lock:
            cls._store = None
            cls._init_failed = False

    @classmethod
    def _is_memcache_backend(cls) -> bool:
        from motor.coordinator.api_client.conductor_api_client import ConductorApiClient

        return ConductorApiClient._resolve_store_backend().lower() == "memcache"

    @classmethod
    def _get_store(cls) -> Any | None:
        if cls._store is not None:
            return cls._store
        if cls._init_failed:
            return None
        with cls._lock:
            if cls._store is not None or cls._init_failed:
                return cls._store
            store = cls._connect()
            if store is None:
                cls._init_failed = True
            else:
                cls._store = store
            return cls._store

    @classmethod
    def _connect(cls) -> Any | None:
        url = cls._meta_service_url()
        if not url:
            logger.debug("skip memcache prefetch: meta_service_url is empty")
            return None
        try:
            from memcache_hybrid import DistributedObjectStore, LocalConfig
        except Exception as exc:  # noqa: BLE001 — package is optional on coordinator
            logger.warning("memcache_hybrid unavailable, SSD prefetch disabled: %s", exc)
            return None
        try:
            config = LocalConfig()
            config.meta_service_url = url
            store = DistributedObjectStore()
            setup_rc = store.setup(config)
            if setup_rc != 0:
                logger.warning("memcache store.setup failed rc=%s url=%s", setup_rc, url)
                return None
            # Client-only: coordinator must not allocate a local DRAM/HBM pool.
            init_rc = store.init(0, init_bm=False)
            if init_rc != 0:
                logger.warning("memcache store.init(client) failed rc=%s url=%s", init_rc, url)
                return None
            logger.info("memcache prefetch client ready url=%s", url)
            return store
        except Exception as exc:  # noqa: BLE001
            logger.warning("memcache prefetch client init failed url=%s: %s", url, exc)
            return None

    @classmethod
    def _meta_service_url(cls) -> str | None:
        host = (Env.kvs_master_service or "").strip()
        if not host:
            try:
                host = (
                    CoordinatorConfig.from_json().prometheus_metrics_config.kv_store_service or ""
                ).strip()
            except Exception:  # noqa: BLE001
                host = ""
        if not host:
            return None
        raw_port = (Env.kv_cache_store_port or "").strip()
        try:
            port = int(raw_port) if raw_port else _DEFAULT_META_PORT
        except ValueError:
            port = _DEFAULT_META_PORT
        if port <= 0:
            port = _DEFAULT_META_PORT
        return f"tcp://{format_host(host)}:{port}"
