# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Cluster-wide running average of request token lengths (not per-endpoint).

Policies read the current value with ``avg_request_tokens()`` or
``BaseSchedulingPolicy.get_avg_request_tokens()``. The mean is updated
incrementally on each allocation:

    new_avg = old_avg + (new_sample - old_avg) / (count + 1)

so only the mean and sample count are stored, not a cumulative token sum.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_avg = 0.0
_count = 0


def record_request_token_length(token_length: float) -> None:
    """Fold one allocated request's token length into the running average."""
    if token_length <= 0:
        return
    global _avg, _count
    sample = float(token_length)
    with _lock:
        _avg = _avg + (sample - _avg) / (_count + 1)
        _count += 1


def avg_request_tokens() -> float:
    """Cluster-wide running average of request token lengths; 0 when none recorded."""
    with _lock:
        return _avg


def request_token_sample_count() -> int:
    """Number of requests already folded into ``avg_request_tokens()``."""
    with _lock:
        return _count


def set_avg_request_tokens(avg: float) -> None:
    """Replace the local snapshot (worker copies the scheduler average from SHM)."""
    global _avg, _count
    with _lock:
        _avg = max(0.0, float(avg))
        if _avg <= 0.0:
            _count = 0


def reset_request_token_stats() -> None:
    """Clear running average state. For tests only."""
    global _avg, _count
    with _lock:
        _avg = 0.0
        _count = 0
