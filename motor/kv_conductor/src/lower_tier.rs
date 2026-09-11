// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// This file is a Derivative Work of NVIDIA Dynamo kv-router
// (https://github.com/ai-dynamo/dynamo), originally licensed under the
// Apache License, Version 2.0. Upstream source path:
//   lib/kv-router/src/indexer/lower_tier.rs
//
// You may obtain a copy of the Apache License at:
//   http://www.apache.org/licenses/LICENSE-2.0
// Local copy: licenses/Apache-2.0.txt
// Attribution: THIRD_PARTY_NOTICES.md
//
// Modified by Huawei Technologies Co., Ltd. for MindIE-PyMotor KV Conductor
// (RwLock + per-worker reverse index, ContiguousHit API, WorkerKey/medium
// integration). Huawei modifications are also available under Mulan PSL v2
// (http://license.coscl.org.cn/MulanPSL2). Redistribution of this file must
// still comply with Apache License 2.0.

//! Lower-tier (CPU / Disk) continuation-edge index.
//!
//! Derived from NVIDIA Dynamo kv-router `LowerTierIndexer`
//! (`lib/kv-router/src/indexer/lower_tier.rs`, Apache-2.0). See
//! `THIRD_PARTY_NOTICES.md`.
//!
//! Stores worker ownership over shared continuation edges:
//! ``(parent_sequence_hash, local_hash) -> child_sequence_hash``.
//!
//! Unlike the HBM radix tree, this index does **not** score from root by
//! default. Queries continue from caller-provided per-worker continuation
//! points (HBM → CPU; max(HBM, CPU) → Disk) and count how many
//! **consecutive** lower-tier blocks are present. The caller also issues an
//! unconditional root-walk candidate per worker owning the first edge, so a
//! longer full replica on this tier is never hidden by a shorter upstream
//! hit; the walk keeps the candidate with the farthest absolute end.

use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::protocols::{KvCacheStoreData, LocalBlockHash, SequenceBlockHash, WorkerKey};

type WorkerSet = FxHashSet<WorkerKey>;

/// Edge key in the continuation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TransitionKey {
    parent_hash: Option<SequenceBlockHash>,
    local_hash: LocalBlockHash,
}

#[derive(Debug, Clone)]
enum EdgeOwnersEntry {
    Single {
        child_hash: SequenceBlockHash,
        owner: WorkerKey,
    },
    Multi {
        child_hash: SequenceBlockHash,
        owners: WorkerSet,
    },
}

impl EdgeOwnersEntry {
    fn new(child_hash: SequenceBlockHash, owner: WorkerKey) -> Self {
        Self::Single { child_hash, owner }
    }

    fn child_hash(&self) -> SequenceBlockHash {
        match self {
            Self::Single { child_hash, .. } | Self::Multi { child_hash, .. } => *child_hash,
        }
    }

    /// Insert `owner` for this edge. Returns `false` if `child_hash` conflicts
    /// with the existing mapping (first-writer wins).
    fn insert(&mut self, child_hash: SequenceBlockHash, owner: WorkerKey) -> bool {
        match self {
            Self::Single {
                child_hash: existing_hash,
                owner: existing_owner,
            } => {
                if *existing_hash != child_hash {
                    return false;
                }
                if *existing_owner == owner {
                    return true;
                }
                let mut owners = WorkerSet::default();
                owners.insert(existing_owner.clone());
                owners.insert(owner);
                *self = Self::Multi { child_hash, owners };
                true
            }
            Self::Multi {
                child_hash: existing_hash,
                owners,
            } => {
                if *existing_hash != child_hash {
                    return false;
                }
                owners.insert(owner);
                true
            }
        }
    }

    /// Remove `owner`. Returns `true` if the edge should be deleted.
    fn remove(&mut self, owner: &WorkerKey) -> bool {
        match self {
            Self::Single {
                owner: existing_owner,
                ..
            } => existing_owner == owner,
            Self::Multi { child_hash, owners } => {
                if !owners.remove(owner) {
                    return false;
                }
                if owners.is_empty() {
                    return true;
                }
                if owners.len() == 1 {
                    let remaining = owners.iter().next().cloned().unwrap();
                    *self = Self::Single {
                        child_hash: *child_hash,
                        owner: remaining,
                    };
                }
                false
            }
        }
    }

    fn contains(&self, owner: &WorkerKey) -> bool {
        match self {
            Self::Single {
                owner: existing_owner,
                ..
            } => existing_owner == owner,
            Self::Multi { owners, .. } => owners.contains(owner),
        }
    }

    fn collect_workers(&self) -> Vec<WorkerKey> {
        match self {
            Self::Single { owner, .. } => vec![owner.clone()],
            Self::Multi { owners, .. } => owners.iter().cloned().collect(),
        }
    }
}

/// Where a lower-tier walk should resume for one worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LowerTierContinuation {
    pub start_pos: usize,
    pub last_matched_hash: Option<SequenceBlockHash>,
}

impl LowerTierContinuation {
    pub fn new(start_pos: usize, last_matched_hash: SequenceBlockHash) -> Self {
        Self {
            start_pos,
            last_matched_hash: Some(last_matched_hash),
        }
    }

    pub fn from_root(start_pos: usize) -> Self {
        Self {
            start_pos,
            last_matched_hash: None,
        }
    }
}

/// Result of a contiguous lower-tier walk for one worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContiguousHit {
    /// Number of blocks matched from ``start_pos``.
    pub count: usize,
    /// Absolute start index in the query hash sequence.
    pub start_pos: usize,
    /// Sequence hash of the last matched block (for the next tier).
    pub last_matched_hash: Option<SequenceBlockHash>,
}

impl ContiguousHit {
    /// Absolute end index (exclusive) — next tier continues here.
    pub fn end_pos(&self) -> usize {
        self.start_pos.saturating_add(self.count)
    }
}

/// A contiguous walk plus the blocks it passed through.
#[derive(Debug, Clone)]
pub struct ReachableChain {
    pub hit: ContiguousHit,
    /// `chain[i]` is the block at absolute position `hit.start_pos + i`.
    pub chain: Vec<SequenceBlockHash>,
}

impl ReachableChain {
    /// The walked blocks from absolute position `from` onwards.
    ///
    /// Callers pass the end of the higher-priority media so the slice covers
    /// exactly the blocks that still need fetching — earlier ones are already
    /// local in HBM. Empty when the walk ends at or before `from`, or when the
    /// chain was not collected.
    pub fn blocks_from(&self, from: usize) -> &[SequenceBlockHash] {
        let offset = from.saturating_sub(self.hit.start_pos);
        self.chain.get(offset..).unwrap_or(&[])
    }

    /// Sequence hash of the walked block at absolute position `pos`.
    ///
    /// `None` outside the walked span or when the chain was not collected.
    pub fn block_at(&self, pos: usize) -> Option<SequenceBlockHash> {
        pos.checked_sub(self.hit.start_pos)
            .and_then(|i| self.chain.get(i))
            .copied()
    }

    /// Whether a walk resumed at `start_pos` with parent `parent` would
    /// retrace this chain from that point on.
    ///
    /// Edges are keyed by `(parent, local_hash)`, so two walks that stand on
    /// the same block at the same position take identical steps afterwards:
    /// the resumed walk is exactly this chain's suffix from `start_pos`. That
    /// lets a DP whose HBM breakpoint lies on the shared root chain reuse it
    /// instead of walking again. Requires the chain to have been collected.
    pub fn retraces_from(&self, start_pos: usize, parent: SequenceBlockHash) -> bool {
        start_pos > self.hit.start_pos
            && start_pos <= self.hit.end_pos()
            && self.block_at(start_pos - 1) == Some(parent)
    }
}

/// Continuation-edge index for one lower-tier medium (CPU or Disk).
#[derive(Debug, Default)]
pub struct LowerTierIndexer {
    edges: RwLock<FxHashMap<TransitionKey, EdgeOwnersEntry>>,
    /// Per-worker reverse lookup: ``block_hash ->TransitionKey`` for O(1) remove.
    worker_blocks: RwLock<FxHashMap<WorkerKey, FxHashMap<SequenceBlockHash, TransitionKey>>>,
}

impl LowerTierIndexer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Workers owning the root edge for ``local_hash`` (parent = None).
    pub fn root_workers(&self, local_hash: LocalBlockHash) -> Vec<WorkerKey> {
        self.edge_owners(None, local_hash)
    }

    /// Workers owning edge ``(parent_hash, local_hash)``.
    pub fn edge_owners(
        &self,
        parent_hash: Option<SequenceBlockHash>,
        local_hash: LocalBlockHash,
    ) -> Vec<WorkerKey> {
        let key = TransitionKey {
            parent_hash,
            local_hash,
        };
        self.edges
            .read()
            .get(&key)
            .map(|e| e.collect_workers())
            .unwrap_or_default()
    }

    /// Insert a stored chain as continuation edges.
    pub fn store_blocks(&self, worker: &WorkerKey, store_data: &KvCacheStoreData) {
        let mut parent_hash = store_data.parent_hash.map(SequenceBlockHash);
        let mut worker_blocks = self.worker_blocks.write();
        let worker_map = worker_blocks.entry(worker.clone()).or_default();
        let mut edges = self.edges.write();

        for block in &store_data.blocks {
            let child = SequenceBlockHash(block.block_hash);
            let key = TransitionKey {
                parent_hash,
                local_hash: LocalBlockHash(block.tokens_hash),
            };

            // Conflicting reverse mapping for the same block_hash ->stop chain.
            if worker_map
                .get(&child)
                .is_some_and(|existing| *existing != key)
            {
                break;
            }

            let inserted = match edges.get_mut(&key) {
                Some(edge) => edge.insert(child, worker.clone()),
                None => {
                    edges.insert(key, EdgeOwnersEntry::new(child, worker.clone()));
                    true
                }
            };

            if !inserted {
                break;
            }

            worker_map.insert(child, key);
            parent_hash = Some(child);
        }
    }

    /// Remove blocks by engine sequence hash.
    pub fn remove_blocks(&self, worker: &WorkerKey, block_hashes: &[u64]) {
        let mut worker_blocks = self.worker_blocks.write();
        let Some(worker_map) = worker_blocks.get_mut(worker) else {
            return;
        };
        let mut edges = self.edges.write();

        for &h in block_hashes {
            let seq = SequenceBlockHash(h);
            let Some(key) = worker_map.remove(&seq) else {
                continue;
            };
            if let Some(edge) = edges.get_mut(&key) {
                if edge.remove(worker) {
                    edges.remove(&key);
                }
            }
        }

        if worker_map.is_empty() {
            worker_blocks.remove(worker);
        }
    }

    /// Drop all edges owned by ``worker``.
    pub fn clear_worker(&self, worker: &WorkerKey) {
        let mut worker_blocks = self.worker_blocks.write();
        let Some(worker_map) = worker_blocks.remove(worker) else {
            return;
        };
        let mut edges = self.edges.write();
        for (_, key) in worker_map {
            if let Some(edge) = edges.get_mut(&key) {
                if edge.remove(worker) {
                    edges.remove(&key);
                }
            }
        }
    }

    /// Look up `(parent_hash, tokens_hash)` for an engine `block_hash` owned
    /// by any worker in this tier.
    ///
    /// Used when a later pool medium (e.g. Disk) confirms a block that was
    /// already indexed on another lower tier (e.g. CPU): reuse the content
    /// mapping without requiring a fresh engine offload event.
    pub fn lookup_block(&self, block_hash: u64) -> Option<(Option<u64>, u64)> {
        let seq = SequenceBlockHash(block_hash);
        let worker_blocks = self.worker_blocks.read();
        for worker_map in worker_blocks.values() {
            if let Some(key) = worker_map.get(&seq) {
                return Some((key.parent_hash.map(|h| h.0), key.local_hash.0));
            }
        }
        None
    }

    /// Whether any worker currently owns ``block_hash``.
    pub fn contains_block(&self, block_hash: u64) -> bool {
        let seq = SequenceBlockHash(block_hash);
        self.worker_blocks
            .read()
            .values()
            .any(|m| m.contains_key(&seq))
    }

    /// Number of blocks tracked for ``worker``.
    pub fn worker_block_count(&self, worker: &WorkerKey) -> usize {
        self.worker_blocks
            .read()
            .get(worker)
            .map(|m| m.len())
            .unwrap_or(0)
    }

    /// Total blocks across all workers (sum of reverse-lookup sizes).
    pub fn total_blocks(&self) -> usize {
        self.worker_blocks.read().values().map(|m| m.len()).sum()
    }

    /// All workers that currently own at least one edge.
    pub fn worker_keys(&self) -> Vec<WorkerKey> {
        self.worker_blocks.read().keys().cloned().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.worker_blocks.read().is_empty()
    }

    /// Contiguous span reachable from `start_pos`, **ignoring** which worker
    /// owns each edge.
    ///
    /// Pooled blocks are fetchable from any node over the backend's transfer
    /// protocol (`device_rdma` / `device_sdma` / `device_urma`), so a block held
    /// by another DP still lets this DP skip recomputing it. Ownership therefore
    /// does not gate the walk — it only decides whether a block is *local*
    /// (free) or *fetched* (transfer cost), which the caller expresses by
    /// attributing the span to the NPU vs CPU/Disk tier.
    ///
    /// Contrast [`Self::query_contiguous_hits`], which does gate on ownership
    /// and answers "what does this worker hold locally".
    ///
    /// Returns `None` when the first edge is already missing, so a zero-length
    /// walk never reports its start position as an end.
    pub fn reachable_from(
        &self,
        local_hashes: &[LocalBlockHash],
        start_pos: usize,
        start_parent: Option<SequenceBlockHash>,
    ) -> Option<ContiguousHit> {
        self.walk(local_hashes, start_pos, start_parent, false)
            .map(|reached| reached.hit)
    }

    /// [`Self::reachable_from`], also returning the block identities it walked
    /// through.
    ///
    /// The chain is what makes per-DP attribution cheap: the walk itself is
    /// ownership-blind and so identical for every DP, but each DP still needs to
    /// know *which* of those blocks it can read locally. Walking once and then
    /// testing the chain against [`Self::count_owned`] answers that with one map
    /// lookup per DP instead of one walk per DP.
    pub fn reachable_chain(
        &self,
        local_hashes: &[LocalBlockHash],
        start_pos: usize,
        start_parent: Option<SequenceBlockHash>,
    ) -> Option<ReachableChain> {
        self.walk(local_hashes, start_pos, start_parent, true)
    }

    /// Ownership-blind contiguous walk; the one primitive behind
    /// [`Self::reachable_from`] and [`Self::reachable_chain`].
    ///
    /// `collect_chain = false` skips recording the block identities, leaving
    /// `chain` empty. Callers that only need the span (no per-DP ownership
    /// test) use that to avoid one allocation per walk.
    pub fn walk(
        &self,
        local_hashes: &[LocalBlockHash],
        start_pos: usize,
        start_parent: Option<SequenceBlockHash>,
        collect_chain: bool,
    ) -> Option<ReachableChain> {
        if start_pos >= local_hashes.len() {
            return None;
        }

        let edges = self.edges.read();
        let mut cur_pos = start_pos;
        let mut cur_hash = start_parent;
        let mut chain = Vec::new();

        while cur_pos < local_hashes.len() {
            let key = TransitionKey {
                parent_hash: cur_hash,
                local_hash: local_hashes[cur_pos],
            };
            let Some(edge) = edges.get(&key) else {
                break;
            };
            let child = edge.child_hash();
            if collect_chain {
                chain.push(child);
            }
            cur_hash = Some(child);
            cur_pos += 1;
        }

        if cur_pos > start_pos {
            Some(ReachableChain {
                hit: ContiguousHit {
                    count: cur_pos - start_pos,
                    start_pos,
                    last_matched_hash: cur_hash,
                },
                chain,
            })
        } else {
            None
        }
    }

    /// How many of `blocks` this worker owns.
    ///
    /// For a pooled medium this is the count of *local* hits: the pool-event
    /// fanout registers every DP in the reporting Pod as an owner, so owning a
    /// pooled block means holding a copy readable without a cross-machine
    /// transfer.
    pub fn count_owned(&self, worker: &WorkerKey, blocks: &[SequenceBlockHash]) -> u32 {
        let worker_blocks = self.worker_blocks.read();
        let Some(owned) = worker_blocks.get(worker) else {
            return 0;
        };
        blocks
            .iter()
            .filter(|block| owned.contains_key(*block))
            .count() as u32
    }

    /// For each worker, walk contiguous lower-tier hits from its continuations.
    ///
    /// A worker may have several candidate continuations (a root walk plus one
    /// or more upstream-breakpoint continuations). Each candidate is walked
    /// independently and the one with the **farthest absolute end** wins —
    /// coverage semantics stay correct no matter which candidate is longer.
    pub fn query_contiguous_hits(
        &self,
        local_hashes: &[LocalBlockHash],
        continuations: &FxHashMap<WorkerKey, Vec<LowerTierContinuation>>,
    ) -> FxHashMap<WorkerKey, ContiguousHit> {
        let mut hits = FxHashMap::default();
        let edges = self.edges.read();

        for (worker, conts) in continuations {
            let mut best: Option<ContiguousHit> = None;
            for cont in conts {
                let mut cur_pos = cont.start_pos;
                let mut cur_hash = cont.last_matched_hash;
                let start = cur_pos;

                while cur_pos < local_hashes.len() {
                    let key = TransitionKey {
                        parent_hash: cur_hash,
                        local_hash: local_hashes[cur_pos],
                    };
                    let Some(edge) = edges.get(&key) else {
                        break;
                    };
                    if !edge.contains(worker) {
                        break;
                    }
                    cur_hash = Some(edge.child_hash());
                    cur_pos += 1;
                }

                let hit = ContiguousHit {
                    count: cur_pos.saturating_sub(start),
                    start_pos: start,
                    last_matched_hash: if cur_pos > start { cur_hash } else { None },
                };
                // Keep the candidate with the farthest absolute end (ties:
                // later candidate wins — same end implies the same last hash).
                best = match best {
                    Some(b) if hit.end_pos() >= b.end_pos() => Some(hit),
                    Some(b) => Some(b),
                    None => Some(hit),
                };
            }
            if let Some(b) = best {
                hits.insert(worker.clone(), b);
            }
        }

        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{KvCacheStoredBlockData, StorageMedium};

    fn worker(id: &str) -> WorkerKey {
        WorkerKey {
            instance_id: id.into(),
            backend_id: id.into(),
            dp_rank: 0,
            medium: StorageMedium::Cpu,
        }
    }

    fn store(parent: Option<u64>, blocks: &[(u64, u64)]) -> KvCacheStoreData {
        KvCacheStoreData {
            parent_hash: parent,
            start_position: None,
            blocks: blocks
                .iter()
                .map(|&(bh, th)| KvCacheStoredBlockData {
                    block_hash: bh,
                    tokens_hash: th,
                })
                .collect(),
        }
    }

    #[test]
    fn root_chain_full_match() {
        let idx = LowerTierIndexer::new();
        let w = worker("w1");
        idx.store_blocks(&w, &store(None, &[(101, 11), (102, 12)]));

        let mut conts = FxHashMap::default();
        conts.insert(w.clone(), vec![LowerTierContinuation::from_root(0)]);
        let hits = idx.query_contiguous_hits(&[LocalBlockHash(11), LocalBlockHash(12)], &conts);
        assert_eq!(hits.get(&w).map(|h| h.count), Some(2));
    }

    #[test]
    fn mid_chain_continuation_from_parent() {
        let idx = LowerTierIndexer::new();
        let w = worker("w1");
        // Tail only: parent=999, then local 21,22
        idx.store_blocks(&w, &store(Some(999), &[(201, 21), (202, 22)]));

        let mut conts = FxHashMap::default();
        conts.insert(
            w.clone(),
            vec![LowerTierContinuation::new(2, SequenceBlockHash(999))],
        );
        let query = [
            LocalBlockHash(1),
            LocalBlockHash(2),
            LocalBlockHash(21),
            LocalBlockHash(22),
        ];
        let hits = idx.query_contiguous_hits(&query, &conts);
        assert_eq!(hits.get(&w).map(|h| h.count), Some(2));
        assert_eq!(
            hits.get(&w).and_then(|h| h.last_matched_hash),
            Some(SequenceBlockHash(202))
        );
        assert_eq!(hits.get(&w).map(|h| h.end_pos()), Some(4));
    }

    #[test]
    fn remove_breaks_contiguous_walk() {
        let idx = LowerTierIndexer::new();
        let w = worker("w1");
        idx.store_blocks(&w, &store(None, &[(101, 11), (102, 12), (103, 13)]));
        idx.remove_blocks(&w, &[102]);

        let mut conts = FxHashMap::default();
        conts.insert(w.clone(), vec![LowerTierContinuation::from_root(0)]);
        let hits = idx.query_contiguous_hits(
            &[LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)],
            &conts,
        );
        // First edge remains; walk stops at missing middle edge.
        assert_eq!(hits.get(&w).map(|h| h.count), Some(1));
    }

    #[test]
    fn shared_edge_remove_preserves_other_owner() {
        let idx = LowerTierIndexer::new();
        let a = worker("a");
        let b = worker("b");
        idx.store_blocks(&a, &store(None, &[(101, 11), (102, 12)]));
        idx.store_blocks(&b, &store(None, &[(101, 11), (102, 12)]));
        idx.remove_blocks(&a, &[101, 102]);

        let mut conts = FxHashMap::default();
        conts.insert(a.clone(), vec![LowerTierContinuation::from_root(0)]);
        conts.insert(b.clone(), vec![LowerTierContinuation::from_root(0)]);
        let hits = idx.query_contiguous_hits(&[LocalBlockHash(11), LocalBlockHash(12)], &conts);
        assert_eq!(hits.get(&a).map(|h| h.count), Some(0));
        assert_eq!(hits.get(&b).map(|h| h.count), Some(2));
    }

    #[test]
    fn walk_without_chain_matches_walk_with_chain() {
        let idx = LowerTierIndexer::new();
        idx.store_blocks(
            &worker("w1"),
            &store(None, &[(101, 11), (102, 12), (103, 13)]),
        );
        let query = [LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];

        let full = idx.walk(&query, 0, None, true).unwrap();
        let span_only = idx.walk(&query, 0, None, false).unwrap();

        assert_eq!(full.hit, span_only.hit);
        assert_eq!(
            full.chain,
            vec![
                SequenceBlockHash(101),
                SequenceBlockHash(102),
                SequenceBlockHash(103)
            ]
        );
        assert!(span_only.chain.is_empty(), "chain must not be collected");
        assert!(span_only.blocks_from(0).is_empty());
        assert_eq!(span_only.block_at(1), None);
    }

    #[test]
    fn retraces_from_matches_resumed_walk() {
        let idx = LowerTierIndexer::new();
        idx.store_blocks(
            &worker("w1"),
            &store(None, &[(101, 11), (102, 12), (103, 13), (104, 14)]),
        );
        let query = [
            LocalBlockHash(11),
            LocalBlockHash(12),
            LocalBlockHash(13),
            LocalBlockHash(14),
        ];
        let root = idx.reachable_chain(&query, 0, None).unwrap();
        assert_eq!(root.hit.end_pos(), 4);

        // A breakpoint standing on block 102 at position 2 retraces the root
        // chain; the real resumed walk agrees with the suffix view.
        assert!(root.retraces_from(2, SequenceBlockHash(102)));
        let resumed = idx
            .reachable_chain(&query, 2, Some(SequenceBlockHash(102)))
            .unwrap();
        assert_eq!(resumed.hit.end_pos(), root.hit.end_pos());
        assert_eq!(resumed.hit.last_matched_hash, root.hit.last_matched_hash);
        assert_eq!(resumed.chain.as_slice(), root.blocks_from(2));

        // Same position but a different upstream block (another engine's
        // sequence hash): no shortcut, the caller has to walk.
        assert!(!root.retraces_from(2, SequenceBlockHash(999)));
        // Resuming from the chain's own start is not a retrace of anything.
        assert!(!root.retraces_from(0, SequenceBlockHash(101)));
        // Standing on the last block: the resumed walk would start where the
        // root walk failed, which is still "retraced" (with an empty suffix).
        assert!(root.retraces_from(4, SequenceBlockHash(104)));
        assert!(idx
            .reachable_chain(&query, 4, Some(SequenceBlockHash(104)))
            .is_none());
        // Beyond the chain end there is nothing to retrace.
        assert!(!root.retraces_from(5, SequenceBlockHash(104)));
    }
}
