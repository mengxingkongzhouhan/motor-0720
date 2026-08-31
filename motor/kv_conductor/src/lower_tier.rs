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
// integration, content-addressed prefix-chain keys). Huawei modifications are
// also available under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
// Redistribution of this file must still comply with Apache License 2.0.

//! Pooled (CPU / Disk) block index, keyed by content prefix chain.
//!
//! Derived from NVIDIA Dynamo kv-router `LowerTierIndexer`
//! (`lib/kv-router/src/indexer/lower_tier.rs`, Apache-2.0). See
//! `THIRD_PARTY_NOTICES.md`.
//!
//! Every pooled block is stored under its [`PrefixChainHash`] — "this block
//! reached through exactly this prefix", derived from token content alone.
//! Consequences that matter for query results:
//!
//! - **One prefix, one identity.** Two engines that pooled the same prefix
//!   land on the same keys, so their copies merge into one walkable chain and
//!   both are reported as owners. Keying on the engine's own rolling
//!   `block_hash` instead would split them into per-engine chains, and vLLM
//!   seeds those chains with a per-process random value unless
//!   `PYTHONHASHSEED` is pinned.
//! - **A fragment is placed by content, not by who chained it.** A pooled
//!   range offloaded mid-sequence (the normal case: engines only save the
//!   blocks they just computed) is reachable from position 0 as soon as the
//!   preceding positions are pooled by *anyone*.
//! - **Positions are absolute.** A query recomputes the chain for its own
//!   token sequence, so a walk can start at any position — no engine-supplied
//!   anchor is needed to resume after a DP's HBM coverage ends.
//!
//! The index does not score from root by itself: the caller walks from root
//! and, per DP, from that DP's HBM coverage end, keeping whichever reaches
//! further. HBM is device memory and not poolable, so only the DP holding
//! those blocks may use them to bridge a gap in the pooled chain.

use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::protocols::{PrefixChainHash, SequenceBlockHash, WorkerKey};

type WorkerSet = FxHashSet<WorkerKey>;

/// One pooled block to record for a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PooledBlock {
    /// Engine sequence hash — the identity removal events use.
    pub block_hash: SequenceBlockHash,
    /// Content identity: this block plus its whole prefix.
    pub prefix_chain: PrefixChainHash,
    /// Engine content mapping, retained so a later medium (CPU → Disk) can
    /// resolve the block without waiting for a fresh engine offload event.
    pub parent_hash: Option<u64>,
    pub tokens_hash: u64,
}

/// What a worker's reverse index remembers about one pooled block.
#[derive(Debug, Clone, Copy)]
struct BlockRecord {
    prefix_chain: PrefixChainHash,
    parent_hash: Option<u64>,
    tokens_hash: u64,
}

/// Owners of one pooled position.
///
/// `One` avoids allocating a set per block: a pooled index can hold millions
/// of positions, and many are reported by a single Pod.
#[derive(Debug, Clone)]
enum Owners {
    One(WorkerKey),
    Many(WorkerSet),
}

impl Owners {
    fn insert(&mut self, owner: WorkerKey) {
        match self {
            Self::One(existing) => {
                if *existing == owner {
                    return;
                }
                let mut owners = WorkerSet::default();
                owners.insert(existing.clone());
                owners.insert(owner);
                *self = Self::Many(owners);
            }
            Self::Many(owners) => {
                owners.insert(owner);
            }
        }
    }

    /// Remove `owner`, reporting whether the position has no owners left.
    fn remove(&mut self, owner: &WorkerKey) -> bool {
        match self {
            Self::One(existing) => existing == owner,
            Self::Many(owners) => {
                owners.remove(owner);
                match owners.len() {
                    0 => true,
                    1 => {
                        let remaining = owners.iter().next().cloned().unwrap();
                        *self = Self::One(remaining);
                        false
                    }
                    _ => false,
                }
            }
        }
    }

    fn contains(&self, owner: &WorkerKey) -> bool {
        match self {
            Self::One(existing) => existing == owner,
            Self::Many(owners) => owners.contains(owner),
        }
    }
}

/// Per-worker reverse index.
///
/// `chain_refs` counts how many of the worker's block hashes point at each
/// position. An engine can re-offload the same content under a fresh
/// `block_hash` (its chain is reseeded on restart), and both hashes then map
/// to one position — the count keeps a removal of either from dropping
/// ownership the other still justifies.
#[derive(Debug, Default)]
struct WorkerTier {
    by_block: FxHashMap<SequenceBlockHash, BlockRecord>,
    chain_refs: FxHashMap<PrefixChainHash, u32>,
}

/// A block hash's mapping, plus how many workers hold it.
///
/// Kept alongside the per-worker indexes so resolving a block hash is one
/// lookup rather than a scan over every worker. That path is hot: placing a
/// pooled chain resolves each block against its predecessor.
#[derive(Debug, Clone, Copy)]
struct SharedRecord {
    record: BlockRecord,
    holders: u32,
}

/// Result of a contiguous pooled walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContiguousHit {
    /// Number of consecutive positions matched from ``start_pos``.
    pub count: usize,
    /// Absolute start index in the query sequence.
    pub start_pos: usize,
}

impl ContiguousHit {
    /// Absolute end index (exclusive) — the next tier continues here.
    pub fn end_pos(&self) -> usize {
        self.start_pos.saturating_add(self.count)
    }
}

/// Pooled block index for one medium (CPU or Disk).
#[derive(Debug, Default)]
pub struct LowerTierIndexer {
    /// Content position → the workers holding a copy.
    positions: RwLock<FxHashMap<PrefixChainHash, Owners>>,
    /// Engine block hash → its mapping, across all workers.
    blocks: RwLock<FxHashMap<SequenceBlockHash, SharedRecord>>,
    /// Per-worker reverse lookup for O(1) removal by engine block hash.
    workers: RwLock<FxHashMap<WorkerKey, WorkerTier>>,
}

impl LowerTierIndexer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record pooled blocks for ``worker``.
    pub fn store_blocks(&self, worker: &WorkerKey, blocks: &[PooledBlock]) {
        if blocks.is_empty() {
            return;
        }
        let mut workers = self.workers.write();
        let tier = workers.entry(worker.clone()).or_default();
        let mut positions = self.positions.write();
        let mut shared = self.blocks.write();

        for block in blocks {
            let record = BlockRecord {
                prefix_chain: block.prefix_chain,
                parent_hash: block.parent_hash,
                tokens_hash: block.tokens_hash,
            };
            match tier.by_block.insert(block.block_hash, record) {
                Some(previous) if previous.prefix_chain == block.prefix_chain => continue,
                Some(previous) => {
                    // Re-anchored to a different prefix: release the old position.
                    Self::release(&mut positions, tier, worker, previous.prefix_chain);
                    shared
                        .entry(block.block_hash)
                        .and_modify(|entry| entry.record = record);
                }
                None => {
                    shared
                        .entry(block.block_hash)
                        .and_modify(|entry| {
                            entry.record = record;
                            entry.holders += 1;
                        })
                        .or_insert(SharedRecord { record, holders: 1 });
                }
            }
            *tier.chain_refs.entry(block.prefix_chain).or_insert(0) += 1;
            positions
                .entry(block.prefix_chain)
                .and_modify(|owners| owners.insert(worker.clone()))
                .or_insert_with(|| Owners::One(worker.clone()));
        }
    }

    /// Drop one holder of `block_hash`, forgetting the mapping at zero.
    fn forget(
        shared: &mut FxHashMap<SequenceBlockHash, SharedRecord>,
        block_hash: SequenceBlockHash,
    ) {
        let Some(entry) = shared.get_mut(&block_hash) else {
            return;
        };
        entry.holders = entry.holders.saturating_sub(1);
        if entry.holders == 0 {
            shared.remove(&block_hash);
        }
    }

    /// Drop one reference to `chain` for `worker`, unindexing the position
    /// once the worker holds no block hashes pointing at it.
    fn release(
        positions: &mut FxHashMap<PrefixChainHash, Owners>,
        tier: &mut WorkerTier,
        worker: &WorkerKey,
        chain: PrefixChainHash,
    ) {
        let Some(refs) = tier.chain_refs.get_mut(&chain) else {
            return;
        };
        *refs = refs.saturating_sub(1);
        if *refs > 0 {
            return;
        }
        tier.chain_refs.remove(&chain);
        if let Some(owners) = positions.get_mut(&chain) {
            if owners.remove(worker) {
                positions.remove(&chain);
            }
        }
    }

    /// Remove blocks by engine sequence hash.
    pub fn remove_blocks(&self, worker: &WorkerKey, block_hashes: &[u64]) {
        let mut workers = self.workers.write();
        let Some(tier) = workers.get_mut(worker) else {
            return;
        };
        let mut positions = self.positions.write();
        let mut shared = self.blocks.write();

        for &h in block_hashes {
            let block_hash = SequenceBlockHash(h);
            let Some(record) = tier.by_block.remove(&block_hash) else {
                continue;
            };
            Self::release(&mut positions, tier, worker, record.prefix_chain);
            Self::forget(&mut shared, block_hash);
        }

        if tier.by_block.is_empty() {
            workers.remove(worker);
        }
    }

    /// Drop everything owned by ``worker``.
    pub fn clear_worker(&self, worker: &WorkerKey) {
        let mut workers = self.workers.write();
        let Some(tier) = workers.remove(worker) else {
            return;
        };
        let mut positions = self.positions.write();
        for chain in tier.chain_refs.into_keys() {
            if let Some(owners) = positions.get_mut(&chain) {
                if owners.remove(worker) {
                    positions.remove(&chain);
                }
            }
        }
        let mut shared = self.blocks.write();
        for block_hash in tier.by_block.into_keys() {
            Self::forget(&mut shared, block_hash);
        }
    }

    /// Look up `(parent_hash, tokens_hash)` for an engine `block_hash` held by
    /// any worker on this medium.
    ///
    /// Used when a later pool medium (e.g. Disk) confirms a block already
    /// indexed on another tier (e.g. CPU): reuse the content mapping without
    /// requiring a fresh engine offload event.
    pub fn lookup_block(&self, block_hash: u64) -> Option<(Option<u64>, u64)> {
        self.record_of(block_hash)
            .map(|record| (record.parent_hash, record.tokens_hash))
    }

    /// The content position of an engine `block_hash` on this medium.
    pub fn prefix_chain_of(&self, block_hash: u64) -> Option<PrefixChainHash> {
        self.record_of(block_hash).map(|record| record.prefix_chain)
    }

    fn record_of(&self, block_hash: u64) -> Option<BlockRecord> {
        self.blocks
            .read()
            .get(&SequenceBlockHash(block_hash))
            .map(|entry| entry.record)
    }

    /// Whether any worker currently holds ``block_hash``.
    pub fn contains_block(&self, block_hash: u64) -> bool {
        self.blocks
            .read()
            .contains_key(&SequenceBlockHash(block_hash))
    }

    /// Number of blocks tracked for ``worker``.
    pub fn worker_block_count(&self, worker: &WorkerKey) -> usize {
        self.workers
            .read()
            .get(worker)
            .map(|tier| tier.by_block.len())
            .unwrap_or(0)
    }

    /// Total blocks across all workers.
    pub fn total_blocks(&self) -> usize {
        self.workers
            .read()
            .values()
            .map(|tier| tier.by_block.len())
            .sum()
    }

    /// All workers holding at least one block.
    pub fn worker_keys(&self) -> Vec<WorkerKey> {
        self.workers.read().keys().cloned().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.workers.read().is_empty()
    }

    /// Longest run of consecutive pooled positions starting at ``start_pos``,
    /// **ignoring** which worker holds each one.
    ///
    /// Pooled blocks are fetchable from any node over the backend's transfer
    /// protocol (`device_rdma` / `device_sdma` / `device_urma`), so a block held
    /// by another DP still lets this DP skip recomputing it. Ownership only
    /// decides whether a block is *local* (free) or *fetched* (transfer cost),
    /// which the caller reports separately via [`Self::count_owned`].
    ///
    /// `None` when the position at ``start_pos`` is already missing, so a
    /// zero-length walk never reports its start as an end.
    pub fn reachable_span(
        &self,
        prefix_chain: &[PrefixChainHash],
        start_pos: usize,
    ) -> Option<ContiguousHit> {
        if start_pos >= prefix_chain.len() {
            return None;
        }

        let positions = self.positions.read();
        let count = prefix_chain[start_pos..]
            .iter()
            .take_while(|chain| positions.contains_key(chain))
            .count();

        (count > 0).then_some(ContiguousHit { count, start_pos })
    }

    /// How many of `prefix_chain`'s positions this worker holds itself.
    ///
    /// For a pooled medium this is the count of *local* hits: the pool-event
    /// fanout registers every DP in the reporting Pod as an owner, so holding a
    /// pooled block means having a copy readable without a cross-machine
    /// transfer.
    pub fn count_owned(&self, worker: &WorkerKey, prefix_chain: &[PrefixChainHash]) -> u32 {
        if prefix_chain.is_empty() {
            return 0;
        }
        let positions = self.positions.read();
        prefix_chain
            .iter()
            .filter(|chain| {
                positions
                    .get(chain)
                    .is_some_and(|owners| owners.contains(worker))
            })
            .count() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashing::{compute_prefix_chain_for_seq, extend_prefix_chain, PREFIX_CHAIN_ROOT};
    use crate::protocols::{LocalBlockHash, StorageMedium};

    fn worker(id: &str) -> WorkerKey {
        WorkerKey {
            instance_id: id.into(),
            backend_id: id.into(),
            dp_rank: 0,
            medium: StorageMedium::Cpu,
        }
    }

    /// The prefix chain of a token-content sequence, as a query computes it.
    fn chain_of(tokens_hashes: &[u64]) -> Vec<PrefixChainHash> {
        let local: Vec<LocalBlockHash> =
            tokens_hashes.iter().copied().map(LocalBlockHash).collect();
        compute_prefix_chain_for_seq(&local)
    }

    /// Pooled blocks covering `[from, from + block_hashes.len())` of `chain`.
    fn placements(
        chain: &[PrefixChainHash],
        from: usize,
        block_hashes: &[u64],
    ) -> Vec<PooledBlock> {
        block_hashes
            .iter()
            .enumerate()
            .map(|(offset, &block_hash)| PooledBlock {
                block_hash: SequenceBlockHash(block_hash),
                prefix_chain: chain[from + offset],
                parent_hash: None,
                tokens_hash: 0,
            })
            .collect()
    }

    #[test]
    fn root_chain_full_match() {
        let idx = LowerTierIndexer::new();
        let w = worker("w1");
        let chain = chain_of(&[11, 12]);
        idx.store_blocks(&w, &placements(&chain, 0, &[101, 102]));

        let hit = idx.reachable_span(&chain, 0).unwrap();
        assert_eq!(hit.count, 2);
        assert_eq!(hit.end_pos(), 2);
    }

    #[test]
    fn walk_resumes_mid_sequence() {
        let idx = LowerTierIndexer::new();
        let w = worker("w1");
        let chain = chain_of(&[1, 2, 21, 22]);
        // Tail only: positions 2 and 3 are pooled, 0 and 1 are not.
        idx.store_blocks(&w, &placements(&chain, 2, &[201, 202]));

        assert!(idx.reachable_span(&chain, 0).is_none());
        let hit = idx.reachable_span(&chain, 2).unwrap();
        assert_eq!(hit.count, 2);
        assert_eq!(hit.end_pos(), 4);
    }

    #[test]
    fn remove_breaks_contiguous_walk() {
        let idx = LowerTierIndexer::new();
        let w = worker("w1");
        let chain = chain_of(&[11, 12, 13]);
        idx.store_blocks(&w, &placements(&chain, 0, &[101, 102, 103]));
        idx.remove_blocks(&w, &[102]);

        // First position remains; the walk stops at the missing middle one.
        assert_eq!(idx.reachable_span(&chain, 0).unwrap().count, 1);
        assert_eq!(idx.reachable_span(&chain, 2).unwrap().count, 1);
    }

    #[test]
    fn shared_position_survives_one_owner_leaving() {
        let idx = LowerTierIndexer::new();
        let a = worker("a");
        let b = worker("b");
        let chain = chain_of(&[11, 12]);
        idx.store_blocks(&a, &placements(&chain, 0, &[101, 102]));
        idx.store_blocks(&b, &placements(&chain, 0, &[101, 102]));
        idx.remove_blocks(&a, &[101, 102]);

        assert_eq!(idx.reachable_span(&chain, 0).unwrap().count, 2);
        assert_eq!(idx.count_owned(&a, &chain), 0);
        assert_eq!(idx.count_owned(&b, &chain), 2);
    }

    #[test]
    fn different_engine_hashes_for_one_prefix_merge() {
        // Two engines pooled the same prefix. Their rolling `block_hash` values
        // are unrelated (vLLM reseeds per process), but the content is the same
        // — so the walk must see one 2-block chain owned by both, not two
        // chains where the first writer wins.
        let idx = LowerTierIndexer::new();
        let a = worker("engine-a");
        let b = worker("engine-b");
        let chain = chain_of(&[11, 12]);
        idx.store_blocks(&a, &placements(&chain, 0, &[0xA0, 0xA1]));
        idx.store_blocks(&b, &placements(&chain, 0, &[0xB0, 0xB1]));

        assert_eq!(idx.reachable_span(&chain, 0).unwrap().count, 2);
        assert_eq!(idx.count_owned(&a, &chain), 2);
        assert_eq!(idx.count_owned(&b, &chain), 2);
        assert_eq!(idx.total_blocks(), 4, "both engine hashes stay removable");
    }

    #[test]
    fn re_offloaded_content_keeps_ownership_until_last_hash_leaves() {
        // Same worker, same content, two engine hashes (engine restarted and
        // reseeded its chain). Removing the stale hash must not unindex the
        // position the fresh one still covers.
        let idx = LowerTierIndexer::new();
        let w = worker("w1");
        let chain = chain_of(&[11]);
        let (stale, fresh) = (0xDEAD, 0xBEEF);
        idx.store_blocks(&w, &placements(&chain, 0, &[stale]));
        idx.store_blocks(&w, &placements(&chain, 0, &[fresh]));

        idx.remove_blocks(&w, &[stale]);
        assert_eq!(idx.count_owned(&w, &chain), 1);

        idx.remove_blocks(&w, &[fresh]);
        assert_eq!(idx.count_owned(&w, &chain), 0);
        assert!(idx.reachable_span(&chain, 0).is_none());
        assert!(idx.is_empty());
    }

    #[test]
    fn clear_worker_drops_every_position() {
        let idx = LowerTierIndexer::new();
        let w = worker("w1");
        let chain = chain_of(&[11, 12]);
        idx.store_blocks(&w, &placements(&chain, 0, &[101, 102]));
        idx.clear_worker(&w);

        assert!(idx.is_empty());
        assert_eq!(idx.total_blocks(), 0);
        assert!(idx.reachable_span(&chain, 0).is_none());
    }

    #[test]
    fn lookup_block_returns_retained_content_mapping() {
        let idx = LowerTierIndexer::new();
        let w = worker("w1");
        idx.store_blocks(
            &w,
            &[PooledBlock {
                block_hash: SequenceBlockHash(0x501),
                prefix_chain: extend_prefix_chain(PREFIX_CHAIN_ROOT, LocalBlockHash(11)),
                parent_hash: Some(0x500),
                tokens_hash: 11,
            }],
        );

        assert_eq!(idx.lookup_block(0x501), Some((Some(0x500), 11)));
        assert!(idx.contains_block(0x501));
        assert_eq!(idx.worker_block_count(&w), 1);
        assert!(idx.lookup_block(0x999).is_none());
    }

    #[test]
    fn block_mapping_outlives_all_but_the_last_holder() {
        // The pool fanout records one block for every DP of a Pod, and resolving
        // the next block in a chain looks that mapping up. It must stay until the
        // last of them drops it.
        let idx = LowerTierIndexer::new();
        let a = worker("a");
        let b = worker("b");
        let chain = chain_of(&[11]);
        idx.store_blocks(&a, &placements(&chain, 0, &[101]));
        idx.store_blocks(&b, &placements(&chain, 0, &[101]));

        idx.remove_blocks(&a, &[101]);
        assert!(idx.contains_block(101));
        assert_eq!(idx.prefix_chain_of(101), Some(chain[0]));

        idx.clear_worker(&b);
        assert!(!idx.contains_block(101));
        assert!(idx.prefix_chain_of(101).is_none());
    }

    #[test]
    fn walk_past_the_end_reports_nothing() {
        let idx = LowerTierIndexer::new();
        let chain = chain_of(&[11]);
        assert!(idx.reachable_span(&chain, 1).is_none());
        assert!(idx.reachable_span(&[], 0).is_none());
    }
}
