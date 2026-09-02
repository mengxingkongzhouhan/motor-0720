# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Scheduling policies and factory."""

from importlib import import_module

__all__ = [
    "BaseSchedulingPolicy",
    "LoadBalancePolicy",
    "RoundRobinPolicy",
    "SMetricPolicy",
    "SchedulingPolicyFactory",
]

_EXPORTS = {
    "BaseSchedulingPolicy": ("motor.coordinator.scheduler.policy.base", "BaseSchedulingPolicy"),
    "LoadBalancePolicy": ("motor.coordinator.scheduler.policy.load_balance", "LoadBalancePolicy"),
    "RoundRobinPolicy": ("motor.coordinator.scheduler.policy.round_robin", "RoundRobinPolicy"),
    "SMetricPolicy": ("motor.coordinator.scheduler.policy.smetric", "SMetricPolicy"),
    "SchedulingPolicyFactory": ("motor.coordinator.scheduler.policy.factory", "SchedulingPolicyFactory"),
}


def __getattr__(name: str):
    """Load policy exports on demand so importing one policy does not load the others."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
