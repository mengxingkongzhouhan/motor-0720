# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Scheduling: policies and runtime (ZMQ process)."""

from importlib import import_module

__all__ = [
    "BaseSchedulingPolicy",
    "LoadBalancePolicy",
    "RoundRobinPolicy",
    "SchedulerClient",
    "SchedulerClientConfig",
    "SchedulerConnectionManager",
    "Scheduler",
    "SchedulerRequest",
    "SchedulerRequestType",
    "SchedulerResponse",
    "SchedulerResponseType",
    "SchedulerServer",
    "SchedulingPolicyFactory",
]

_EXPORTS = {
    "Scheduler": ("motor.coordinator.scheduler.scheduler", "Scheduler"),
    "BaseSchedulingPolicy": ("motor.coordinator.scheduler.policy.base", "BaseSchedulingPolicy"),
    "LoadBalancePolicy": ("motor.coordinator.scheduler.policy.load_balance", "LoadBalancePolicy"),
    "RoundRobinPolicy": ("motor.coordinator.scheduler.policy.round_robin", "RoundRobinPolicy"),
    "SchedulingPolicyFactory": ("motor.coordinator.scheduler.policy.factory", "SchedulingPolicyFactory"),
    "SchedulerServer": ("motor.coordinator.scheduler.runtime", "SchedulerServer"),
    "SchedulerClient": ("motor.coordinator.scheduler.runtime", "SchedulerClient"),
    "SchedulerClientConfig": ("motor.coordinator.scheduler.runtime", "SchedulerClientConfig"),
    "SchedulerConnectionManager": ("motor.coordinator.scheduler.runtime", "SchedulerConnectionManager"),
    "SchedulerRequest": ("motor.coordinator.scheduler.runtime", "SchedulerRequest"),
    "SchedulerResponse": ("motor.coordinator.scheduler.runtime", "SchedulerResponse"),
    "SchedulerRequestType": ("motor.coordinator.scheduler.runtime", "SchedulerRequestType"),
    "SchedulerResponseType": ("motor.coordinator.scheduler.runtime", "SchedulerResponseType"),
}


def __getattr__(name: str):
    """Load public scheduler objects without eagerly importing every policy and runtime."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
