# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
import os
import shutil
import subprocess

import lib.constant as C
from lib.utils import logger, load_yaml
from lib.generator.infer_service import get_infer_role, _find_infer_service_set_doc
from lib.generator.k8s_utils import get_accelerator_type_from_cluster, get_deploy_mode_from_config
from lib.update_config_whitelist import collect_changed_paths


PD_SEPARATION_DEPLOY_KEYS = {
    C.P_INSTANCES_NUM,
    C.D_INSTANCES_NUM,
    C.SINGER_P_INSTANCES_NUM,
    C.SINGER_D_INSTANCES_NUM,
    C.P_POD_NPU_NUM,
    C.D_POD_NPU_NUM,
}

PD_HYBRID_REQUIRED_DEPLOY_KEYS = {
    C.HYBRID_INSTANCES_NUM,
    C.SINGLE_HYBRID_INSTANCE_POD_NUM,
    C.HYBRID_POD_NPU_NUM,
}


reserved_labels: dict[str, bool] = {"app": True}  # True 表示禁止用户通过 additional_labels 覆盖该标签


def validate_reserved_labels(user_config: dict) -> None:
    """Reject user-defined labels that override deployer-managed labels."""
    if not isinstance(user_config, dict):
        return

    conflicts = []
    for config_name, component_config in user_config.items():
        if not isinstance(component_config, dict):
            continue
        additional_labels = component_config.get(C.ADDITIONAL_LABELS)
        if not isinstance(additional_labels, dict):
            continue
        conflicting_labels = sorted(label for label in additional_labels if reserved_labels.get(label, False))
        if conflicting_labels:
            conflicts.append(f"{config_name}.{C.ADDITIONAL_LABELS}: {', '.join(conflicting_labels)}")

    if conflicts:
        raise ValueError(f"User-defined labels contain reserved labels: {'; '.join(conflicts)}")


def resolve_config_paths(config_dir, user_config_path, env_config_path):
    if not config_dir and not user_config_path and not env_config_path:
        logger.error("No configuration provided. Please use one of the following options:")
        logger.error("  --config_dir <dir>     : Directory containing user_config.json and env.json")
        logger.error("  --config <file>        : Path to user_config.json (requires --env)")
        logger.error("  --env <file>           : Path to env.json (requires --config)")
        logger.error("Example:")
        logger.error("  python deploy.py --config_dir ../infer_engines/vllm")
        logger.error(
            "  python deploy.py --config ../infer_engines/vllm/user_config.json --env ../infer_engines/vllm/env.json"
        )
        raise ValueError("Missing required configuration. Use --config_dir or both --config and --env.")

    if config_dir:
        dir_user_config = os.path.join(config_dir, "user_config.json")
        dir_env_config = os.path.join(config_dir, "env.json")

        if not user_config_path:
            if os.path.exists(dir_user_config):
                user_config_path = dir_user_config
                logger.info(f"Using user_config.json from config_dir: {user_config_path}")
            else:
                logger.error(f"user_config.json not found in {config_dir}")
                raise FileNotFoundError(f"user_config.json not found in {config_dir}")

        if not env_config_path:
            if os.path.exists(dir_env_config):
                env_config_path = dir_env_config
                logger.info(f"Using env.json from config_dir: {env_config_path}")
            else:
                logger.error(f"env.json not found in {config_dir}")
                raise FileNotFoundError(f"env.json not found in {config_dir}")

    if user_config_path and not env_config_path:
        logger.error("--config is specified but --env is missing")
        raise ValueError("Both --config and --env must be specified together, or use --config_dir")

    if env_config_path and not user_config_path:
        logger.error("--env is specified but --config is missing")
        raise ValueError("Both --config and --env must be specified together, or use --config_dir")

    logger.info(f"{C.GREEN}User config path: {user_config_path}{C.RESET}")
    logger.info(f"{C.GREEN}Env config path: {env_config_path}{C.RESET}")

    return user_config_path, env_config_path


def strip_instance_nums(config_dict):
    cleaned = json.loads(json.dumps(config_dict))
    cleaned["motor_deploy_config"].pop(C.E_INSTANCES_NUM, None)
    cleaned["motor_deploy_config"].pop(C.P_INSTANCES_NUM, None)
    cleaned["motor_deploy_config"].pop(C.D_INSTANCES_NUM, None)
    cleaned["motor_deploy_config"].pop(C.HYBRID_INSTANCES_NUM, None)
    cleaned["motor_deploy_config"].pop(C.DEPLOY_MODE_CONFIG_KEY, None)
    return cleaned


def validate_only_instance_changed(current_config, baseline_config):
    current_stripped = strip_instance_nums(current_config)
    baseline_stripped = strip_instance_nums(baseline_config)

    changed_paths = collect_changed_paths(current_stripped, baseline_stripped)

    if changed_paths:
        logger.warning(
            "user_config changes detected beyond instance numbers: %s. "
            "Only e_instances_num/p_instances_num/d_instances_num/hybrid_instances_num "
            "can be modified for scaling.",
            ", ".join(sorted(changed_paths)),
        )


def validate_deploy_mode_consistency(deploy_config, baseline_config):
    """Validate that deploy_mode hasn't changed when updating config."""
    baseline_mode = get_deploy_mode_from_config(baseline_config)
    current_mode = get_deploy_mode_from_config(deploy_config)
    if baseline_mode != current_mode:
        raise ValueError(
            f"motor_deploy_config.{C.DEPLOY_MODE_CONFIG_KEY} cannot be changed when updating config. "
            f"Current deployment uses '{baseline_mode}', user_config has '{current_mode}'."
        )


def validate_deploy_mode_value(deploy_mode_arg):
    """Validate deploy_mode value is valid."""
    if deploy_mode_arg not in C.VALID_DEPLOY_MODES:
        raise ValueError(
            f"Baseline config has invalid {C.DEPLOY_MODE_CONFIG_KEY}: {deploy_mode_arg}. "
            f"Must be one of {list(C.VALID_DEPLOY_MODES)}."
        )


def validate_pd_hybrid_config(user_config):
    deploy_config = user_config.get(C.MOTOR_DEPLOY_CONFIG, {})
    if not isinstance(deploy_config, dict):
        raise ValueError("motor_deploy_config is required for PD hybrid.")

    missing_keys = PD_HYBRID_REQUIRED_DEPLOY_KEYS - deploy_config.keys()
    if missing_keys:
        raise ValueError(f"PD hybrid config missing required keys: {sorted(missing_keys)}")

    mixed_deploy_keys = PD_SEPARATION_DEPLOY_KEYS & deploy_config.keys()
    if mixed_deploy_keys:
        raise ValueError(f"PD hybrid config cannot include separation keys: {sorted(mixed_deploy_keys)}")

    if "engine_topology" in deploy_config:
        raise ValueError("PD hybrid config must not include motor_deploy_config.engine_topology.")

    if C.MOTOR_ENGINE_UNION_CONFIG not in user_config:
        raise ValueError("PD hybrid config requires motor_engine_union_config.")
    if C.MOTOR_ENGINE_PREFILL_CONFIG in user_config or "motor_engine_decode_config" in user_config:
        raise ValueError("PD hybrid config cannot include prefill/decode engine config sections.")


def validate_pd_hybrid_infer_service_template(user_config, infer_service_template_path):
    """Require union role in InferServiceSet template when PD hybrid uses CRD deploy mode."""
    deploy_config = user_config.get(C.MOTOR_DEPLOY_CONFIG, {})
    deploy_mode = deploy_config.get(C.DEPLOY_MODE_CONFIG_KEY, C.DEPLOY_MODE_INFER_SERVICE_SET)
    if deploy_mode == C.DEPLOY_MODE_MULTI_DEPLOYMENT_YAML:
        return
    if not os.path.exists(infer_service_template_path):
        raise FileNotFoundError(
            f"InferServiceSet template yaml not found for PD hybrid CRD validation: {infer_service_template_path}"
        )
    all_docs = load_yaml(infer_service_template_path, False)
    if not isinstance(all_docs, list):
        all_docs = [all_docs]
    infer_doc = _find_infer_service_set_doc(all_docs)
    if not get_infer_role(infer_doc, C.ROLE_UNION):
        raise ValueError("PD hybrid with infer_service_set requires a 'union' role in infer_service_template.yaml.")


def _get_pd_heterogeneous_config(deploy_config):
    """Extract PD heterogeneous config from deploy_config, returns None if disabled."""
    if deploy_config.get(C.ENABLE_PD_HETEROGENEOUS) is not True:
        return None
    label_key = deploy_config.get(C.PD_HETEROGENEOUS_LABEL_KEY, C.DEFAULT_PD_HETEROGENEOUS_LABEL_KEY)
    prefill_value = deploy_config.get(C.PD_HETEROGENEOUS_PREFILL_LABEL_VALUE, C.DEFAULT_PD_HETEROGENEOUS_PREFILL_VALUE)
    decode_value = deploy_config.get(C.PD_HETEROGENEOUS_DECODE_LABEL_VALUE, C.DEFAULT_PD_HETEROGENEOUS_DECODE_VALUE)
    return {
        "label_key": label_key,
        "prefill_value": prefill_value,
        "decode_value": decode_value,
    }


def _get_hardware_node_labels(hardware_type):
    """Extract nodeSelector labels determined by hardware_type.

    Returns dict of label key-value pairs. Raises ValueError for unknown types.
    """
    if hardware_type in C.HARDWARE_TYPE_A2 or hardware_type in C.HARDWARE_TYPE_A3:
        return {
            C.ACCELERATOR: C.ACCELERATOR_910,
            C.ACCELERATOR_TYPE: get_accelerator_type_from_cluster(hardware_type),
        }
    if hardware_type in C.HARDWARE_TYPE_950I_A5:
        return {
            C.ACCELERATOR: C.ACCELERATOR_A5,
            C.ACCELERATOR_TYPE: get_accelerator_type_from_cluster(hardware_type),
        }
    known = [*sorted(C.HARDWARE_TYPE_A2), *sorted(C.HARDWARE_TYPE_A3), *C.HARDWARE_TYPE_950I_A5]
    raise ValueError(f"Unknown hardware_type '{hardware_type}'. Supported values: {known}")


# Engine templates only tolerate these taints; control-plane NoSchedule is not among them.
_ENGINE_TOLERATED_TAINT_KEYS = {
    "node.kubernetes.io/not-ready",
    "node.kubernetes.io/unreachable",
}
_BLOCKING_TAINT_EFFECTS = {"NoSchedule", "NoExecute"}


def _npu_resource_name(hardware_type):
    if hardware_type in C.HARDWARE_TYPE_950I_A5:
        return C.ASCEND_950_NPU_NUM
    return C.ASCEND_910_NPU_NUM


def _pod_npu_request(deploy_config, npu_key, default=1):
    if npu_key not in deploy_config:
        return default
    try:
        return int(deploy_config[npu_key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{npu_key} must be an integer, got {deploy_config[npu_key]!r}") from exc


def _taint_blocks_engine(taint):
    if not isinstance(taint, dict):
        return False
    if taint.get("effect") not in _BLOCKING_TAINT_EFFECTS:
        return False
    return taint.get("key") not in _ENGINE_TOLERATED_TAINT_KEYS


def _node_blocking_taints(node):
    return [taint for taint in (node.get("spec") or {}).get("taints") or [] if _taint_blocks_engine(taint)]


def _node_allocatable_npu(node, resource_name):
    raw = ((node.get("status") or {}).get("allocatable") or {}).get(resource_name, 0)
    try:
        return int(str(raw).split(".")[0])
    except (TypeError, ValueError):
        return 0


def _node_name(node):
    return (node.get("metadata") or {}).get("name", "<unknown>")


def _get_matching_nodes(labels, node_desc):
    """Return node objects matching ALL labels. Raises if kubectl fails or none match."""
    label_selector = ",".join(f"{k}={v}" for k, v in labels.items())
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        raise RuntimeError("kubectl not found in PATH")
    try:
        result = subprocess.run(
            [kubectl, "get", "nodes", "-l", label_selector, "-o", "json"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to query cluster nodes for {node_desc} with labels {labels}: {e}") from e

    if result.returncode != 0:
        raise RuntimeError(
            f"kubectl get nodes failed for {node_desc} with labels {labels}. stderr: {result.stderr.strip()}"
        )

    try:
        items = (json.loads(result.stdout or "{}") or {}).get("items") or []
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"kubectl get nodes returned invalid JSON for {node_desc}: {exc}") from exc

    if not items:
        raise RuntimeError(
            f"No node in cluster matches nodeSelector for {node_desc}: {labels}. "
            f"Please ensure suitable nodes are labeled correctly."
        )
    return items


def _validate_nodes_schedulable_for_engine(nodes, node_desc, labels, npu_num, npu_resource):
    """Reject label matches that Volcano still cannot use (taints / allocatable NPU)."""
    schedulable = []
    tainted = []
    short_npu = []
    for node in nodes:
        name = _node_name(node)
        blocking = _node_blocking_taints(node)
        if blocking:
            keys = ",".join(str(taint.get("key")) for taint in blocking)
            tainted.append(f"{name}({keys})")
            continue
        allocatable = _node_allocatable_npu(node, npu_resource)
        if npu_num and allocatable < npu_num:
            short_npu.append(f"{name}({npu_resource}={allocatable}<{npu_num})")
            continue
        schedulable.append(name)

    if schedulable:
        logger.info(
            "Node selector validated for %s: %s -> %s schedulable node(s): %s",
            node_desc,
            labels,
            len(schedulable),
            schedulable,
        )
        return schedulable

    details = []
    if tainted:
        details.append(f"tainted (engine pods have no toleration): {', '.join(tainted)}")
    if short_npu:
        details.append(f"insufficient allocatable NPU: {', '.join(short_npu)}")
    raise RuntimeError(
        f"Nodes match nodeSelector for {node_desc}: {labels}, but none can schedule the engine pod. "
        + "; ".join(details)
        + ". Control-plane taints or pinning prefill+decode (8+8 NPU) onto one 8-card node "
        "will make Infer Operator / Volcano create InstanceSets without pods."
    )


def _validate_node_labels_exist(labels, node_desc, npu_num=None, npu_resource=None):
    """Assert that at least one node matches labels and can run an engine pod."""
    if not labels:
        return []
    nodes = _get_matching_nodes(labels, node_desc)
    if npu_num is None:
        names = [_node_name(node) for node in nodes]
        logger.info(f"Node selector validated for {node_desc}: {labels} -> {len(nodes)} node(s) found")
        return names
    return _validate_nodes_schedulable_for_engine(nodes, node_desc, labels, npu_num, npu_resource)


def cards_per_node_for_hardware(hardware_type):
    """Return cards-per-node for a known hardware_type, or None if unknown."""
    return C.HARDWARE_CARDS_PER_NODE.get(hardware_type)


def validate_pod_npu_against_hardware(deploy_config):
    """Reject per-pod NPU requests that no single node of this hardware can grant.

    InferServiceSet engine roles request ``huawei.com/Ascend910`` (or A5 NPU)
    on one pod. If ``p_pod_npu_num`` / ``d_pod_npu_num`` / ``hybrid_pod_npu_num``
    exceeds the cards on one node, Infer Operator + Volcano never create the
    engine pods — only controller / coordinator / kv-store come up.
    """
    if not isinstance(deploy_config, dict):
        return
    hardware_type = deploy_config.get(C.HARDWARE_TYPE)
    cards = cards_per_node_for_hardware(hardware_type)
    if cards is None:
        return

    npu_keys = (C.P_POD_NPU_NUM, C.D_POD_NPU_NUM, C.E_POD_NPU_NUM, C.HYBRID_POD_NPU_NUM)
    for key in npu_keys:
        if key not in deploy_config:
            continue
        try:
            npu_num = int(deploy_config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer, got {deploy_config[key]!r}") from exc
        if npu_num > cards:
            raise ValueError(
                f"{key}={npu_num} exceeds cards per node ({cards}) for hardware_type={hardware_type}. "
                "Each engine pod requests this many NPU on a single node; Infer Operator / Volcano "
                "will not create prefill/decode/union pods. Use a smaller *_pod_npu_num or split "
                "the instance across nodes with single_*_instance_pod_num."
            )


def _node_selector_override(deploy_config, selector_key):
    """Return user-defined nodeSelector extras for *selector_key*, or {}."""
    selector = deploy_config.get(selector_key, {})
    if not selector:
        return {}
    if not isinstance(selector, dict):
        raise ValueError(f"{C.MOTOR_DEPLOY_CONFIG}.{selector_key} must be a JSON object")
    return {key: value for key, value in selector.items() if key}


def validate_node_selectors(deploy_config):
    """Validate that cluster nodes exist for every nodeSelector combination to be used.

    Always validates base hardware labels (accelerator-type, accelerator).
    When PD heterogeneous deployment is enabled, additionally validates the
    combined prefill/decode labels per node type.
    Custom ``prefill_node_selector`` / ``decode_node_selector`` are AND-merged
    into the same check — validating only hardware labels would pass while
    Volcano still cannot create engine pods.
    """
    hardware_type = deploy_config.get(C.HARDWARE_TYPE)
    base_labels = _get_hardware_node_labels(hardware_type)
    prefill_override = _node_selector_override(deploy_config, C.PREFILL_NODE_SELECTOR)
    decode_override = _node_selector_override(deploy_config, C.DECODE_NODE_SELECTOR)
    npu_resource = _npu_resource_name(hardware_type)
    p_npu = _pod_npu_request(deploy_config, C.P_POD_NPU_NUM)
    d_npu = _pod_npu_request(deploy_config, C.D_POD_NPU_NUM)

    pd_config = _get_pd_heterogeneous_config(deploy_config)

    if pd_config is not None:
        label_key = pd_config["label_key"]
        prefill_labels = {**base_labels, label_key: pd_config["prefill_value"], **prefill_override}
        decode_labels = {**base_labels, label_key: pd_config["decode_value"], **decode_override}
        _validate_node_labels_exist(prefill_labels, "prefill(P)", p_npu, npu_resource)
        _validate_node_labels_exist(decode_labels, "decode(D)", d_npu, npu_resource)
        logger.info(
            f"PD heterogeneous node selectors validated: prefill -> {prefill_labels}, decode -> {decode_labels}"
        )
        return

    if prefill_override or decode_override:
        prefill_labels = {**base_labels, **prefill_override}
        decode_labels = {**base_labels, **decode_override}
        prefill_nodes = _validate_node_labels_exist(prefill_labels, "prefill(P)", p_npu, npu_resource)
        decode_nodes = _validate_node_labels_exist(decode_labels, "decode(D)", d_npu, npu_resource)
        if prefill_labels == decode_labels and prefill_nodes == decode_nodes and len(prefill_nodes) == 1:
            items = _get_matching_nodes(prefill_labels, "prefill+decode")
            schedulable = [node for node in items if _node_name(node) in prefill_nodes]
            allocatable = max((_node_allocatable_npu(node, npu_resource) for node in schedulable), default=0)
            if p_npu + d_npu > allocatable:
                raise RuntimeError(
                    f"prefill ({p_npu}) + decode ({d_npu}) NPU requests exceed allocatable "
                    f"{npu_resource}={allocatable} on the only matching node {prefill_nodes[0]} "
                    f"for nodeSelector {prefill_labels}. Pin each role to a different worker, "
                    f"or drop the extra nodeSelector."
                )
        return

    _validate_node_labels_exist(base_labels, "engine", max(p_npu, d_npu), npu_resource)
