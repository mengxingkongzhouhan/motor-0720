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

import ipaddress
import os
import socket
import threading
from typing import Any, Iterable

from motor.common.logger import get_logger
from motor.common.utils.env import Env
from motor.common.utils.net import format_host, split_address
from motor.config.coordinator import CoordinatorConfig

logger = get_logger(__name__)

# MemCache MetaService RPC port (same default as KVCacheStoreConfig.port).
_DEFAULT_META_PORT = 50088
# SSD / DRAM media ids from the MemCache Python API.
_SSD_MEDIA = 2
_DRAM_MEDIA = 1
# Same label as examples/deployer kv-store templates. InferService may rewrite
# names (e.g. vllm-0-kv-store-0-*) so the name hint is a fallback.
_KV_STORE_LABEL = "app=mindie-motor-kv-store"
_KV_STORE_NAME_HINT = "kv-store"
_SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"


def _host_from_meta_url(url: str | None) -> str:
    """Extract the host from ``tcp://host:port`` / ``tcp://[v6]:port``."""
    if not url:
        return ""
    rest = url.split("://", 1)[-1]
    host, _port = split_address(rest)
    return host.strip("[]")


def _ip_literal(host: str) -> str | None:
    if not host:
        return None
    try:
        return str(ipaddress.ip_address(host.strip("[]")))
    except ValueError:
        return None


def _dns_ips(host: str) -> tuple[str, ...]:
    """Resolve ``host`` via DNS. ClusterIP Service → Service IP, not pod IP."""
    if not host or host == "-":
        return ()
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:  # noqa: BLE001 — diagnostics only
        return ()
    seen: list[str] = []
    for info in infos:
        addr = info[4][0] if info[4] else ""
        if addr and addr not in seen:
            seen.append(addr)
    return tuple(seen)


def _running_pod_ips(pods: Any, name_substr: str = "") -> tuple[str, ...]:
    ips: list[str] = []
    for pod in getattr(pods, "items", None) or []:
        name = getattr(getattr(pod, "metadata", None), "name", "") or ""
        if name_substr and name_substr not in name:
            continue
        status = getattr(pod, "status", None)
        phase = getattr(status, "phase", "") or ""
        if phase and phase != "Running":
            continue
        ip = getattr(status, "pod_ip", None) or getattr(status, "podIP", None)
        if ip and ip not in ips:
            ips.append(ip)
    return tuple(ips)


def _k8s_kv_store_pod_ips() -> tuple[str, ...]:
    """Best-effort ``status.podIP`` of Running kv-store pods.

    Same value as
    ``kubectl -n $POD_NAMESPACE get pod <kv-store> -o jsonpath='{.status.podIP}'``.

    Coordinator templates set ``automountServiceAccountToken: false``, so this
    usually returns empty and callers fall back to DNS of ``KVS_MASTER_SERVICE``.
    """
    if not os.path.exists(_SA_TOKEN_PATH):
        return ()
    namespace = os.getenv("POD_NAMESPACE", "").strip()
    if not namespace:
        return ()
    try:
        from kubernetes import client, config

        config.load_incluster_config()
        v1 = client.CoreV1Api()
        labeled = v1.list_namespaced_pod(namespace, label_selector=_KV_STORE_LABEL)
        ips = _running_pod_ips(labeled)
        if ips:
            return ips
        return _running_pod_ips(v1.list_namespaced_pod(namespace), name_substr=_KV_STORE_NAME_HINT)
    except Exception as exc:  # noqa: BLE001 — diagnostics only, never break prefetch
        logger.debug("kv-store pod IP lookup via k8s skipped: %s", exc)
        return ()


class MemcacheStoreClient:
    """Lazy MemCache client used by the coordinator to prefetch SSD-resident KV."""

    _lock = threading.Lock()
    _store: Any = None
    _init_failed: bool = False
    _meta_url: str | None = None
    _kv_store_ips: tuple[str, ...] = ()

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
            logger.warning(
                "memcache prefetch raised keys=%d url=%s kv_store_ip=%s: %s",
                len(keys),
                cls._cached_meta_url(),
                cls._kv_store_ip_csv(),
                exc,
            )
            return False
        if result != 0:
            logger.warning(
                "memcache prefetch failed rc=%s keys=%d url=%s kv_store_ip=%s",
                result,
                len(keys),
                cls._cached_meta_url(),
                cls._kv_store_ip_csv(),
            )
            return False
        logger.info(
            "memcache prefetch submitted keys=%d url=%s kv_store_ip=%s",
            len(keys),
            cls._cached_meta_url(),
            cls._kv_store_ip_csv(),
        )
        return True

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop the cached store so unit tests can re-bind a mock."""
        with cls._lock:
            cls._store = None
            cls._init_failed = False
            cls._meta_url = None
            cls._kv_store_ips = ()

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
    def _cached_meta_url(cls) -> str:
        return cls._meta_url or cls._meta_service_url() or "-"

    @classmethod
    def _kv_store_ip_csv(cls) -> str:
        if cls._kv_store_ips:
            return ",".join(cls._kv_store_ips)
        url = cls._meta_url or cls._meta_service_url()
        ips = cls._resolve_kv_store_ips(url)
        if ips:
            cls._kv_store_ips = ips
        return ",".join(ips) if ips else "-"

    @classmethod
    def _resolve_kv_store_ips(cls, url: str | None) -> tuple[str, ...]:
        """Best-effort kv-store addresses for prefetch diagnostics.

        Prefers in-cluster ``status.podIP`` (the value from
        ``kubectl get pod ... -o jsonpath='{.status.podIP}'``). Coordinator
        pods usually have no ServiceAccount token, so this falls back to:

        1. an IP literal in ``KVS_MASTER_SERVICE`` (docker / slurm)
        2. DNS of the MetaService host (ClusterIP Service → Service IP)
        """
        pod_ips = _k8s_kv_store_pod_ips()
        if pod_ips:
            return pod_ips
        host = _host_from_meta_url(url)
        literal = _ip_literal(host)
        if literal:
            return (literal,)
        return _dns_ips(host)

    @classmethod
    def _connect(cls) -> Any | None:
        url = cls._meta_service_url()
        if not url:
            logger.debug("skip memcache prefetch: meta_service_url is empty")
            return None
        cls._meta_url = url
        cls._kv_store_ips = cls._resolve_kv_store_ips(url)
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
                logger.warning(
                    "memcache store.setup failed rc=%s url=%s kv_store_ip=%s",
                    setup_rc,
                    url,
                    cls._kv_store_ip_csv(),
                )
                return None
            # Client-only: coordinator must not allocate a local DRAM/HBM pool.
            init_rc = store.init(0, init_bm=False)
            if init_rc != 0:
                logger.warning(
                    "memcache store.init(client) failed rc=%s url=%s kv_store_ip=%s",
                    init_rc,
                    url,
                    cls._kv_store_ip_csv(),
                )
                return None
            logger.info(
                "memcache prefetch client ready url=%s kv_store_ip=%s",
                url,
                cls._kv_store_ip_csv(),
            )
            return store
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "memcache prefetch client init failed url=%s kv_store_ip=%s: %s",
                url,
                cls._kv_store_ip_csv(),
                exc,
            )
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
