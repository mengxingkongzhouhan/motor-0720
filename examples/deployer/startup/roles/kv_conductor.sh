#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# Capture Deployment env before ConfigMap injection. `set_kv_conductor_env`
# copies motor_common_env / motor_kv_conductor_env and can overwrite RUST_LOG
# (e.g. leftover RUST_LOG=trace), which is why setting it only on the Pod spec
# often has no effect.
KV_CONDUCTOR_CONTAINER_RUST_LOG="${RUST_LOG:-}"

set_kv_conductor_env

KV_CONDUCTOR_PORT=${KV_CONDUCTOR_PORT:-13333}
KV_CONDUCTOR_HOST=${KV_CONDUCTOR_HOST:-0.0.0.0}

# If KV_CONDUCTOR_PORT is a full URL (e.g. "tcp://10.98.27.88:13333"),
# extract just the trailing port number for the --port argument.
if [[ "$KV_CONDUCTOR_PORT" == *":"* ]]; then
    KV_CONDUCTOR_PORT="${KV_CONDUCTOR_PORT##*:}"
fi

echo "Starting KV Conductor on ${KV_CONDUCTOR_HOST}:${KV_CONDUCTOR_PORT}"

# kv-conductor is bundled inside the motor Python package.
# Precedence: KV_CONDUCTOR_RUST_LOG > Deployment RUST_LOG > env.json > info.
if [ -n "${KV_CONDUCTOR_RUST_LOG:-}" ]; then
    export RUST_LOG="$KV_CONDUCTOR_RUST_LOG"
elif [ -n "${KV_CONDUCTOR_CONTAINER_RUST_LOG}" ]; then
    export RUST_LOG="$KV_CONDUCTOR_CONTAINER_RUST_LOG"
else
    export RUST_LOG="${RUST_LOG:-info}"
fi
exec python -m motor.kv_conductor \
    --host "$KV_CONDUCTOR_HOST" \
    --port "$KV_CONDUCTOR_PORT"
