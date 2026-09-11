// Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
// MindIE is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//         http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

//! Shared helpers for event application.

use std::sync::OnceLock;

use parking_lot::Mutex;
use rustc_hash::FxHashSet;

use crate::backend::MatchMode;
use crate::protocols::*;

/// Resolve the target storage media list from an event's optional `medium`
/// field, falling back to the subscriber's `default_media`.
pub(super) fn resolve_medium(
    event_medium: Option<&str>,
    default_media: &[StorageMedium],
) -> Vec<StorageMedium> {
    if let Some(m) = event_medium {
        vec![StorageMedium::parse(m)]
    } else {
        default_media.to_vec()
    }
}

/// Build the list of `WorkerKey` targets for an event.
///
/// - `MatchMode::None` (YuanRong): one worker per medium, using `backend_id`
///   directly as the instance identity.
/// - Other modes (Mooncake/Memcache): fans out via `hbm_ip_index`. An IP
///   with no HBM-registered DP (decode pool capacity) falls back to
///   `pool:<ip>` so the shared CPU/Disk graph still records the block.
pub(super) fn resolve_workers(
    match_mode: MatchMode,
    hbm_ip_index: &Option<HbmIpIndex>,
    backend_id: &str,
    dp_rank: u32,
    target_media: &[StorageMedium],
) -> Vec<WorkerKey> {
    if match_mode == MatchMode::None {
        return target_media
            .iter()
            .map(|&medium| WorkerKey {
                instance_id: backend_id.to_string(),
                backend_id: backend_id.to_string(),
                dp_rank,
                medium,
            })
            .collect();
    }

    let workers =
        match_mode.resolve_workers(hbm_ip_index.as_ref(), backend_id, dp_rank, target_media);
    if !workers.is_empty() {
        return workers;
    }

    // Store IP is not an HBM-registered engine node (decode LocalService
    // is the common case). The blocks are still in the shared pool, so
    // they must enter the continuation-edge index. Own them under
    // `pool:<ip>` — the query walk is ownership-blind, and known_dps
    // skips this prefix so the coordinator is not offered a fake instance.
    let (index_present, indexed_ip_count, ip_known) = match hbm_ip_index.as_ref() {
        Some(idx) => {
            let guard = idx.read();
            (true, guard.len(), guard.contains_key(backend_id))
        }
        None => (false, 0, false),
    };
    let reason = if !index_present {
        "hbm_ip_index_absent"
    } else if indexed_ip_count == 0 {
        "hbm_ip_index_empty"
    } else if !ip_known {
        "backend_id_not_in_hbm_ip_index"
    } else {
        "no_matching_dp"
    };
    let fallback_instance = pool_location_instance_id(backend_id);
    let fallback: Vec<WorkerKey> = target_media
        .iter()
        .map(|&medium| WorkerKey {
            instance_id: fallback_instance.clone(),
            backend_id: backend_id.to_string(),
            dp_rank: 0,
            medium,
        })
        .collect();

    // A decode LocalService emits one of these per pooled block, so the
    // per-event line would drown the info log. Say it once per store IP at
    // info — that is the fact an operator needs — and keep the rest at trace.
    if first_unmapped_sighting(backend_id) {
        tracing::info!(
            %backend_id,
            dp_rank,
            ?match_mode,
            index_present,
            indexed_ip_count,
            ip_known,
            fallback_instance = %fallback_instance,
            media = ?target_media.iter().map(|m| m.log_str()).collect::<Vec<_>>(),
            reason,
            "kv_event pool_unmapped (first sighting; further events from this ip at trace)"
        );
    } else {
        tracing::trace!(
            %backend_id,
            dp_rank,
            fallback_instance = %fallback_instance,
            reason,
            "kv_event pool_unmapped"
        );
    }
    fallback
}

/// Store IPs that have already fallen back to `pool:<ip>` at least once.
///
/// Bounded by the number of distinct pool nodes, so it is never cleared.
static UNMAPPED_BACKENDS_SEEN: OnceLock<Mutex<FxHashSet<String>>> = OnceLock::new();

/// `true` the first time `backend_id` falls back to a pool placeholder.
fn first_unmapped_sighting(backend_id: &str) -> bool {
    let seen = UNMAPPED_BACKENDS_SEEN.get_or_init(|| Mutex::new(FxHashSet::default()));
    let mut guard = seen.lock();
    if guard.contains(backend_id) {
        return false;
    }
    guard.insert(backend_id.to_string());
    true
}
