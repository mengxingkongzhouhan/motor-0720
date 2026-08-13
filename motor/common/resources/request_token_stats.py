# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Cluster-wide cumulative average of request token lengths (not per-endpoint)."""

from __future__ import annotations

import threading

_lock = threading.Lock()
_total_tokens = 0.0
_total_requests = 0


def record_request_token_length(token_length: float) -> None:
    """Accumulate one allocated request's token length into the cluster-wide average."""
    if token_length <= 0:
        return
    global _total_tokens, _total_requests
    with _lock:
        _total_tokens += float(token_length)
        _total_requests += 1


def avg_request_tokens() -> float:
    """Mean token length over all recorded requests; 0 when none have been recorded."""
    with _lock:
        if _total_requests <= 0:
            return 0.0
        return _total_tokens / _total_requests


def set_avg_request_tokens(avg: float) -> None:
    """Replace the local snapshot (worker copies the scheduler's cluster-wide average from SHM)."""
    global _total_tokens, _total_requests
    with _lock:
        if avg <= 0.0:
            _total_tokens = 0.0
            _total_requests = 0
            return
        _total_tokens = float(avg)
        _total_requests = 1


def reset_request_token_stats() -> None:
    """Clear cumulative stats. For tests only."""
    global _total_tokens, _total_requests
    with _lock:
        _total_tokens = 0.0
        _total_requests = 0
