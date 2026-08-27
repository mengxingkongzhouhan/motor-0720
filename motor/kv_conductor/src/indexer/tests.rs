// Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
// MindIE is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//         http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

use super::*;

#[test]
fn test_indexer_get_or_create_and_query() {
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-a", "tenant-1");

    // Compute the actual hash for the test token sequence
    let tokens: Vec<i64> = vec![10, 20, 30, 40];
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert!(!hashes.is_empty());
    let tokens_hash = hashes[0];

    // Insert a worker with NPU blocks using the real hash
    let wk_npu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "inst-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };

    let store = KvCacheEventData::Stored(KvCacheStoreData {
        parent_hash: None,
        start_position: None,
        blocks: vec![KvCacheStoredBlockData {
            block_hash: 100,
            tokens_hash: tokens_hash.0,
        }],
    });
    entry.apply_event(&wk_npu, &store).unwrap();

    // Query with the same tokens
    let resp = indexer.query("model-a", "tenant-1", &tokens, 4).unwrap();
    let tenant = &resp.tenants["tenant-1"];
    let imd = &tenant["inst-1"];
    let dp0 = &imd.dp["0"];
    assert!(dp0.npu_blocks > 0, "should have NPU match");
    assert_eq!(dp0.cpu_blocks, 0);
    assert_eq!(dp0.disk_blocks, 0);
    assert_eq!(
        dp0.matched_tokens,
        (dp0.npu_blocks + dp0.cpu_blocks + dp0.disk_blocks) * 4
    );
    assert_eq!(imd.longest_matched, dp0.matched_tokens);
}

#[test]
fn test_per_tier_aggregation() {
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-b", "t1");

    // Two different token sequences → different block hashes
    let tokens_a: Vec<i64> = vec![10, 20, 30, 40];
    let tokens_b: Vec<i64> = vec![50, 60, 70, 80];
    let hash_a = compute_block_hash_for_seq(&tokens_a, 4)[0];
    let hash_b = compute_block_hash_for_seq(&tokens_b, 4)[0];

    // Worker 1: NPU blocks
    let wk1 = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "inst-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    entry
        .apply_event(
            &wk1,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 100,
                    tokens_hash: hash_a.0,
                }],
            }),
        )
        .unwrap();

    // Worker 2: CPU blocks (different instance, different tokens)
    let wk2 = WorkerKey {
        instance_id: "inst-2".into(),
        backend_id: "mooncake-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Cpu,
    };
    entry
        .apply_event(
            &wk2,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 200,
                    tokens_hash: hash_b.0,
                }],
            }),
        )
        .unwrap();

    // Query with tokens_a — should match inst-1 (NPU) only
    let resp = indexer.query("model-b", "t1", &tokens_a, 4).unwrap();
    let tenant = &resp.tenants["t1"];

    let imd1 = &tenant["inst-1"];
    let dp0 = &imd1.dp["0"];
    assert!(
        dp0.npu_blocks > 0,
        "inst-1 should have NPU match for tokens_a"
    );
    assert_eq!(dp0.cpu_blocks, 0, "inst-1 should have no CPU match");

    // Query with tokens_b — should match inst-2 (CPU) only
    let resp = indexer.query("model-b", "t1", &tokens_b, 4).unwrap();
    let tenant = &resp.tenants["t1"];

    let imd2 = &tenant["inst-2"];
    let dp2 = &imd2.dp["0"];
    assert_eq!(dp2.npu_blocks, 0, "inst-2 should have no NPU match");
    assert!(
        dp2.cpu_blocks > 0,
        "inst-2 should have CPU match for tokens_b"
    );
}

#[test]
fn test_cpu_continuation_from_hbm_breakpoint() {
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-c", "t1");

    let tokens: Vec<i64> = (0..12).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 3);

    // HBM holds first two blocks.
    let wk_npu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "inst-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    entry
        .apply_event(
            &wk_npu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: 100,
                        tokens_hash: hashes[0].0,
                    },
                    KvCacheStoredBlockData {
                        block_hash: 200,
                        tokens_hash: hashes[1].0,
                    },
                ],
            }),
        )
        .unwrap();

    // CPU holds the tail continuing from HBM's last seq_hash=200.
    let wk_cpu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Cpu,
    };
    entry
        .apply_event(
            &wk_cpu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(200),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 300,
                    tokens_hash: hashes[2].0,
                }],
            }),
        )
        .unwrap();

    let overlap = entry.find_matches(&tokens, 4);
    assert_eq!(
        overlap.blocks.get(&wk_npu).copied().unwrap_or(0),
        2,
        "HBM should match first 2 blocks"
    );
    assert_eq!(
        overlap.blocks.get(&wk_cpu).copied().unwrap_or(0),
        1,
        "CPU should continue 1 block from HBM breakpoint"
    );

    // Exclusive: npu=2, cpu=1; default weights 1/1/0 → matched = 3 blocks.
    let resp = indexer.query("model-c", "t1", &tokens, 4).unwrap();
    let dp0 = &resp.tenants["t1"]["inst-1"].dp["0"];
    assert_eq!(dp0.npu_blocks, 2);
    assert_eq!(dp0.cpu_blocks, 1);
    assert_eq!(dp0.disk_blocks, 0);
    assert_eq!(dp0.matched_tokens, 3 * 4);
    assert_eq!(resp.tenants["t1"]["inst-1"].longest_matched, 12);
}

#[test]
fn test_cpu_replica_reported_when_hbm_hits_same_dp() {
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-d", "t1");

    let tokens: Vec<i64> = vec![1, 2, 3, 4];
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 1);

    let wk_npu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "inst-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    entry
        .apply_event(
            &wk_npu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 100,
                    tokens_hash: hashes[0].0,
                }],
            }),
        )
        .unwrap();

    // Same DP also has a full CPU root chain for the same prefix.
    let wk_cpu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Cpu,
    };
    entry
        .apply_event(
            &wk_cpu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 200,
                    tokens_hash: hashes[0].0,
                }],
            }),
        )
        .unwrap();

    let resp = indexer.query("model-d", "t1", &tokens, 4).unwrap();
    let dp0 = &resp.tenants["t1"]["inst-1"].dp["0"];
    assert_eq!(dp0.npu_blocks, 1);
    // Same-prefix CPU replica is exclusive-attributed to NPU.
    assert_eq!(dp0.cpu_blocks, 0);
    assert_eq!(dp0.matched_tokens, 4);
}

#[test]
fn test_cpu_root_used_when_no_hbm_on_dp() {
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-e", "t1");

    let tokens: Vec<i64> = vec![9, 8, 7, 6];
    let hashes = compute_block_hash_for_seq(&tokens, 4);

    let wk_cpu = WorkerKey {
        instance_id: "inst-cpu-only".into(),
        backend_id: "pool-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Cpu,
    };
    entry
        .apply_event(
            &wk_cpu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 300,
                    tokens_hash: hashes[0].0,
                }],
            }),
        )
        .unwrap();

    let resp = indexer.query("model-e", "t1", &tokens, 4).unwrap();
    let dp0 = &resp.tenants["t1"]["inst-cpu-only"].dp["0"];
    assert_eq!(dp0.npu_blocks, 0);
    assert_eq!(dp0.cpu_blocks, 1);
    assert_eq!(dp0.matched_tokens, 4);
}

#[test]
fn test_disk_continuation_from_cpu_breakpoint() {
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-f", "t1");

    let tokens: Vec<i64> = (0..12).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 3);

    // HBM: first block
    let wk_npu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "inst-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    entry
        .apply_event(
            &wk_npu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 100,
                    tokens_hash: hashes[0].0,
                }],
            }),
        )
        .unwrap();

    // CPU: second block, continuing from HBM seq=100
    let wk_cpu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-cpu".into(),
        dp_rank: 0,
        medium: StorageMedium::Cpu,
    };
    entry
        .apply_event(
            &wk_cpu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(100),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 200,
                    tokens_hash: hashes[1].0,
                }],
            }),
        )
        .unwrap();

    // Disk: third block, continuing from CPU seq=200
    let wk_disk = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-disk".into(),
        dp_rank: 0,
        medium: StorageMedium::Disk,
    };
    entry
        .apply_event(
            &wk_disk,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(200),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 300,
                    tokens_hash: hashes[2].0,
                }],
            }),
        )
        .unwrap();

    let resp = indexer.query("model-f", "t1", &tokens, 4).unwrap();
    let dp0 = &resp.tenants["t1"]["inst-1"].dp["0"];
    assert_eq!(dp0.npu_blocks, 1);
    assert_eq!(dp0.cpu_blocks, 1);
    assert_eq!(dp0.disk_blocks, 1);
    // Unweighted coverage = exclusive sum × block_size.
    assert_eq!(dp0.matched_tokens, 3 * 4);
}

#[test]
fn test_overlapping_npu_cpu_disk_replicas_do_not_inflate_matched_tokens() {
    // Same prefix on NPU + CPU + Disk: exclusive attribution keeps only NPU.
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-overlap", "t1");

    let tokens: Vec<i64> = (0..8).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 2);

    let wk_npu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "inst-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    entry
        .apply_event(
            &wk_npu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: 100,
                        tokens_hash: hashes[0].0,
                    },
                    KvCacheStoredBlockData {
                        block_hash: 200,
                        tokens_hash: hashes[1].0,
                    },
                ],
            }),
        )
        .unwrap();

    let wk_cpu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-cpu".into(),
        dp_rank: 0,
        medium: StorageMedium::Cpu,
    };
    entry
        .apply_event(
            &wk_cpu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: 100,
                        tokens_hash: hashes[0].0,
                    },
                    KvCacheStoredBlockData {
                        block_hash: 200,
                        tokens_hash: hashes[1].0,
                    },
                ],
            }),
        )
        .unwrap();

    let wk_disk = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-disk".into(),
        dp_rank: 0,
        medium: StorageMedium::Disk,
    };
    entry
        .apply_event(
            &wk_disk,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: 100,
                        tokens_hash: hashes[0].0,
                    },
                    KvCacheStoredBlockData {
                        block_hash: 200,
                        tokens_hash: hashes[1].0,
                    },
                ],
            }),
        )
        .unwrap();

    let resp = indexer.query("model-overlap", "t1", &tokens, 4).unwrap();
    let dp0 = &resp.tenants["t1"]["inst-1"].dp["0"];
    assert_eq!(dp0.npu_blocks, 2);
    assert_eq!(dp0.cpu_blocks, 0);
    assert_eq!(dp0.disk_blocks, 0);
    assert_eq!(dp0.matched_tokens, 2 * 4);
    assert!(
        dp0.matched_tokens <= tokens.len() as u32,
        "matched_tokens {} exceeds input {}",
        dp0.matched_tokens,
        tokens.len()
    );
}

#[test]
fn test_shorter_hbm_breakpoint_does_not_overcount_cpu_overlap() {
    // Two NPU workers on the same DP with different depths; CPU holds the
    // tail after the shorter breakpoint. Coverage must stay at the true
    // prefix end (2), not npu_max + cpu_segment.
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-short-break", "t1");

    let tokens: Vec<i64> = (0..8).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 2);

    let wk_npu_short = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "npu-short".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    entry
        .apply_event(
            &wk_npu_short,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 100,
                    tokens_hash: hashes[0].0,
                }],
            }),
        )
        .unwrap();

    let wk_npu_long = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "npu-long".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    entry
        .apply_event(
            &wk_npu_long,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: 100,
                        tokens_hash: hashes[0].0,
                    },
                    KvCacheStoredBlockData {
                        block_hash: 200,
                        tokens_hash: hashes[1].0,
                    },
                ],
            }),
        )
        .unwrap();

    let wk_cpu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-cpu".into(),
        dp_rank: 0,
        medium: StorageMedium::Cpu,
    };
    entry
        .apply_event(
            &wk_cpu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(100),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 200,
                    tokens_hash: hashes[1].0,
                }],
            }),
        )
        .unwrap();

    let resp = indexer
        .query("model-short-break", "t1", &tokens, 4)
        .unwrap();
    let dp0 = &resp.tenants["t1"]["inst-1"].dp["0"];
    assert_eq!(dp0.npu_blocks, 2);
    // CPU continuation ends at the same absolute position as the longer NPU
    // hit, so exclusive cpu_blocks is 0.
    assert_eq!(dp0.cpu_blocks, 0);
    assert_eq!(dp0.matched_tokens, 2 * 4);
}

#[test]
fn test_hbm_breakpoint_not_shared_across_instances() {
    // inst-a's HBM covers the first two blocks; inst-b holds ONLY the tail on
    // CPU, chained after inst-a's last HBM block. Reported coverage is an
    // absolute end position, so letting inst-b start at inst-a's breakpoint
    // would claim inst-b covers blocks 0..3 — it holds just block 2, and a
    // request routed there would recompute the whole prefix.
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-break-scope", "t1");

    let tokens: Vec<i64> = (0..12).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 3);

    let wk_a_npu = WorkerKey {
        instance_id: "inst-a".into(),
        backend_id: "inst-a".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    entry
        .apply_event(
            &wk_a_npu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: 100,
                        tokens_hash: hashes[0].0,
                    },
                    KvCacheStoredBlockData {
                        block_hash: 200,
                        tokens_hash: hashes[1].0,
                    },
                ],
            }),
        )
        .unwrap();

    // inst-b's DRAM chain is anchored at block_hash=200, which only inst-a ever
    // held. Block identities are content-derived, so this is a real anchor
    // inst-b's engine can emit after reusing the same prefix from the pool.
    let wk_b_cpu = WorkerKey {
        instance_id: "inst-b".into(),
        backend_id: "pool".into(),
        dp_rank: 0,
        medium: StorageMedium::Cpu,
    };
    entry
        .apply_event(
            &wk_b_cpu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(200),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 300,
                    tokens_hash: hashes[2].0,
                }],
            }),
        )
        .unwrap();

    let resp = indexer
        .query("model-break-scope", "t1", &tokens, 4)
        .unwrap();
    let tenant = &resp.tenants["t1"];

    // inst-a is unaffected — lending a breakpoint costs it nothing.
    let dp_a = &tenant["inst-a"].dp["0"];
    assert_eq!(dp_a.npu_blocks, 2);
    assert_eq!(dp_a.cpu_blocks, 0);
    assert_eq!(dp_a.matched_tokens, 2 * 4);

    // inst-b gets no candidate, so it never reaches medium_ends.
    assert!(
        !tenant.instances.contains_key("inst-b"),
        "inst-b holds only a mid-sequence segment and must not report coverage \
         borrowed from inst-a's breakpoint, got {:?}",
        tenant.instances.get("inst-b")
    );

    // The pool-wide entry does stitch across DPs: inst-a's HBM covers [0,2) and
    // inst-b's CPU covers [2,3), so the pool collectively holds all 3 blocks.
    assert_eq!(tenant.global.npu_blocks, 2);
    assert_eq!(tenant.global.cpu_blocks, 1);
    assert_eq!(tenant.global.matched_tokens, 3 * 4);
    assert_eq!(
        tenant.global.dp_ranges["inst-a"]["0"],
        vec![(0, 2)],
        "inst-a holds the first two blocks"
    );
    assert_eq!(
        tenant.global.dp_ranges["inst-b"]["0"],
        vec![(2, 3)],
        "inst-b holds only the third block"
    );
}

/// Store one lower-tier chain for `worker`, anchored at `parent`.
fn store_cpu_chain(
    entry: &IndexerEntry,
    worker: &WorkerKey,
    parent: Option<u64>,
    blocks: &[(u64, u64)],
) {
    entry
        .apply_event(
            worker,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent,
                start_position: None,
                blocks: blocks
                    .iter()
                    .map(|&(block_hash, tokens_hash)| KvCacheStoredBlockData {
                        block_hash,
                        tokens_hash,
                    })
                    .collect(),
            }),
        )
        .unwrap();
}

fn cpu_worker(instance_id: &str, dp_rank: u32) -> WorkerKey {
    WorkerKey {
        instance_id: instance_id.into(),
        backend_id: "pool".into(),
        dp_rank,
        medium: StorageMedium::Cpu,
    }
}

#[test]
fn test_global_span_stitches_cpu_chains_across_dps() {
    // The headline case for `_global`: inst-a's DRAM holds [0,2) and inst-b's
    // DRAM continues [2,4). No single DP covers the prefix, but the pool does —
    // and any DP can fetch any pooled block, so prefill is skippable for all 4.
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-global-cpu", "t1");

    let tokens: Vec<i64> = (0..16).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 4);

    let wk_a = cpu_worker("inst-a", 0);
    let wk_b = cpu_worker("inst-b", 0);
    store_cpu_chain(&entry, &wk_a, None, &[(10, hashes[0].0), (11, hashes[1].0)]);
    // inst-b's chain continues from inst-a's last block identity.
    store_cpu_chain(
        &entry,
        &wk_b,
        Some(11),
        &[(12, hashes[2].0), (13, hashes[3].0)],
    );

    let resp = indexer.query("model-global-cpu", "t1", &tokens, 4).unwrap();
    let tenant = &resp.tenants["t1"];

    // Pool-wide: the walk crosses the ownership boundary at position 2.
    assert_eq!(tenant.global.cpu_blocks, 4);
    assert_eq!(tenant.global.npu_blocks, 0);
    assert_eq!(tenant.global.matched_tokens, 4 * 4);

    // Per-DP is unchanged and stays strictly local: inst-a stops where inst-b's
    // ownership begins, and inst-b has no entry point of its own (it owns
    // neither the root edge nor any HBM breakpoint).
    assert_eq!(tenant["inst-a"].dp["0"].cpu_blocks, 2);
    assert_eq!(tenant["inst-a"].dp["0"].matched_tokens, 2 * 4);
    assert!(
        !tenant.instances.contains_key("inst-b"),
        "per-DP coverage must not inherit a foreign start position, got {:?}",
        tenant.instances.get("inst-b")
    );

    // The pool span therefore exceeds every individual DP — this is information
    // a per-DP max could never produce.
    let best_dp = tenant
        .instances
        .values()
        .map(|imd| imd.longest_matched)
        .max()
        .unwrap_or(0);
    assert_eq!(best_dp, 2 * 4);
    assert!(tenant.global.matched_tokens > best_dp);

    // Ownership map: contiguous runs collapse into a single range each.
    assert_eq!(tenant.global.dp_ranges["inst-a"]["0"], vec![(0, 2)]);
    assert_eq!(tenant.global.dp_ranges["inst-b"]["0"], vec![(2, 4)]);
}

#[test]
fn test_global_span_stops_at_gap_and_drops_unreachable_blocks() {
    // inst-a holds [0,2). inst-c holds a chain for position 3 anchored at the
    // position-2 block identity, but nobody holds position 2 itself — the prefix
    // is broken there, so the span stops at 2 and inst-c's blocks are not
    // reported: they sit behind a gap and cannot serve a contiguous prefix.
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-global-gap", "t1");

    let tokens: Vec<i64> = (0..16).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 4);

    let wk_a = cpu_worker("inst-a", 0);
    let wk_c = cpu_worker("inst-c", 0);
    store_cpu_chain(&entry, &wk_a, None, &[(10, hashes[0].0), (11, hashes[1].0)]);
    // Anchored at block identity 12 (position 2), which was never stored.
    store_cpu_chain(&entry, &wk_c, Some(12), &[(13, hashes[3].0)]);

    let resp = indexer.query("model-global-gap", "t1", &tokens, 4).unwrap();
    let tenant = &resp.tenants["t1"];

    assert_eq!(tenant.global.cpu_blocks, 2, "span must stop at the gap");
    assert_eq!(tenant.global.matched_tokens, 2 * 4);
    assert_eq!(tenant.global.dp_ranges["inst-a"]["0"], vec![(0, 2)]);
    assert!(
        !tenant.global.dp_ranges.contains_key("inst-c"),
        "blocks behind a gap are unreachable and must be dropped, got {:?}",
        tenant.global.dp_ranges.get("inst-c")
    );
}

#[test]
fn test_global_span_continues_from_pool_wide_hbm_end() {
    // The HBM prefix is itself assembled across DPs: inst-a holds block 0,
    // inst-b holds blocks 0..2 (so the pool-wide HBM end is 2, deeper than
    // inst-a). inst-c's DRAM continues from there. The global CPU walk must
    // resume at the pool-wide HBM end, not at any single DP's depth.
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-global-hbm", "t1");

    let tokens: Vec<i64> = (0..12).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 3);

    let npu = |instance_id: &str| WorkerKey {
        instance_id: instance_id.into(),
        backend_id: instance_id.into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    store_cpu_chain(&entry, &npu("inst-a"), None, &[(100, hashes[0].0)]);
    store_cpu_chain(
        &entry,
        &npu("inst-b"),
        None,
        &[(100, hashes[0].0), (200, hashes[1].0)],
    );
    // inst-c holds only the tail on CPU, chained after HBM block 200.
    store_cpu_chain(
        &entry,
        &cpu_worker("inst-c", 0),
        Some(200),
        &[(300, hashes[2].0)],
    );

    let resp = indexer.query("model-global-hbm", "t1", &tokens, 4).unwrap();
    let tenant = &resp.tenants["t1"];

    assert_eq!(tenant.global.npu_blocks, 2, "pool-wide HBM prefix");
    assert_eq!(tenant.global.cpu_blocks, 1, "CPU extends the pool-wide end");
    assert_eq!(tenant.global.matched_tokens, 3 * 4);

    // Both HBM holders are credited for the blocks they actually hold.
    assert_eq!(tenant.global.dp_ranges["inst-a"]["0"], vec![(0, 1)]);
    assert_eq!(tenant.global.dp_ranges["inst-b"]["0"], vec![(0, 2)]);
    assert_eq!(tenant.global.dp_ranges["inst-c"]["0"], vec![(2, 3)]);

    // Per-DP: inst-c still gets nothing (the breakpoint belongs to inst-b).
    assert_eq!(tenant["inst-b"].dp["0"].npu_blocks, 2);
    assert!(!tenant.instances.contains_key("inst-c"));
}

#[test]
fn test_hbm_breakpoint_not_shared_across_dp_ranks() {
    // Same instance, two DP ranks sharing one CPU continuation edge. dp0 has
    // HBM coverage and may continue from its own breakpoint; dp1 has none, so
    // it must not inherit dp0's start position.
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-break-dp", "t1");

    let tokens: Vec<i64> = (0..12).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 3);

    let wk_npu_dp0 = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "inst-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    entry
        .apply_event(
            &wk_npu_dp0,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: 100,
                        tokens_hash: hashes[0].0,
                    },
                    KvCacheStoredBlockData {
                        block_hash: 200,
                        tokens_hash: hashes[1].0,
                    },
                ],
            }),
        )
        .unwrap();

    // Both DPs of the node see the same pool block, so they co-own one edge
    // (same parent, same content hash, same child identity).
    for dp_rank in [0u32, 1u32] {
        let wk_cpu = WorkerKey {
            instance_id: "inst-1".into(),
            backend_id: "pool".into(),
            dp_rank,
            medium: StorageMedium::Cpu,
        };
        entry
            .apply_event(
                &wk_cpu,
                &KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: Some(200),
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: 300,
                        tokens_hash: hashes[2].0,
                    }],
                }),
            )
            .unwrap();
    }

    let resp = indexer.query("model-break-dp", "t1", &tokens, 4).unwrap();
    let imd = &resp.tenants["t1"]["inst-1"];

    // dp0 continues from its own breakpoint: 2 HBM + 1 CPU.
    let dp0 = &imd.dp["0"];
    assert_eq!(dp0.npu_blocks, 2);
    assert_eq!(dp0.cpu_blocks, 1);
    assert_eq!(dp0.matched_tokens, 3 * 4);

    // dp1 co-owns the edge but has no upstream coverage of its own.
    assert!(
        !imd.dp.contains_key("1"),
        "dp1 has no HBM prefix and must not inherit dp0's breakpoint, got {:?}",
        imd.dp.get("1")
    );
    assert_eq!(imd.longest_matched, 3 * 4);
}

#[test]
fn test_disk_continuation_from_hbm_when_cpu_miss() {
    // vLLM lookup: after NPU hit, Disk can hit even if CPU miss (then promote).
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-h", "t1");

    let tokens: Vec<i64> = (0..8).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 2);

    let wk_npu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "inst-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    entry
        .apply_event(
            &wk_npu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 100,
                    tokens_hash: hashes[0].0,
                }],
            }),
        )
        .unwrap();

    // Disk tail continues from HBM; CPU has nothing.
    let wk_disk = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-disk".into(),
        dp_rank: 0,
        medium: StorageMedium::Disk,
    };
    entry
        .apply_event(
            &wk_disk,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(100),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 300,
                    tokens_hash: hashes[1].0,
                }],
            }),
        )
        .unwrap();

    let resp = indexer.query("model-h", "t1", &tokens, 4).unwrap();
    let dp0 = &resp.tenants["t1"]["inst-1"].dp["0"];
    assert_eq!(dp0.npu_blocks, 1);
    assert_eq!(dp0.cpu_blocks, 0);
    assert_eq!(dp0.disk_blocks, 1);
    // Unweighted coverage includes exclusive disk extension.
    assert_eq!(dp0.matched_tokens, 2 * 4);
}

#[test]
fn test_disk_replica_reported_when_cpu_hits_same_dp() {
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-g", "t1");

    let tokens: Vec<i64> = vec![1, 2, 3, 4];
    let hashes = compute_block_hash_for_seq(&tokens, 4);

    let wk_cpu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-cpu".into(),
        dp_rank: 0,
        medium: StorageMedium::Cpu,
    };
    entry
        .apply_event(
            &wk_cpu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 100,
                    tokens_hash: hashes[0].0,
                }],
            }),
        )
        .unwrap();

    // Same DP also has a Disk root chain — reported as a real segment.
    let wk_disk = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-disk".into(),
        dp_rank: 0,
        medium: StorageMedium::Disk,
    };
    entry
        .apply_event(
            &wk_disk,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 200,
                    tokens_hash: hashes[0].0,
                }],
            }),
        )
        .unwrap();

    let resp = indexer.query("model-g", "t1", &tokens, 4).unwrap();
    let dp0 = &resp.tenants["t1"]["inst-1"].dp["0"];
    assert_eq!(dp0.cpu_blocks, 1);
    // Same-prefix Disk replica is exclusive-attributed to CPU.
    assert_eq!(dp0.disk_blocks, 0);
    assert_eq!(dp0.matched_tokens, 4);
}

#[test]
fn test_disk_replica_reported_when_hbm_hits_same_dp() {
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-i", "t1");

    let tokens: Vec<i64> = vec![1, 2, 3, 4];
    let hashes = compute_block_hash_for_seq(&tokens, 4);

    let wk_npu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "inst-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    entry
        .apply_event(
            &wk_npu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 100,
                    tokens_hash: hashes[0].0,
                }],
            }),
        )
        .unwrap();

    let wk_disk = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-disk".into(),
        dp_rank: 0,
        medium: StorageMedium::Disk,
    };
    entry
        .apply_event(
            &wk_disk,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 200,
                    tokens_hash: hashes[0].0,
                }],
            }),
        )
        .unwrap();

    let resp = indexer.query("model-i", "t1", &tokens, 4).unwrap();
    let dp0 = &resp.tenants["t1"]["inst-1"].dp["0"];
    assert_eq!(dp0.npu_blocks, 1);
    // Same-prefix Disk replica is exclusive-attributed to NPU.
    assert_eq!(dp0.disk_blocks, 0);
    assert_eq!(dp0.matched_tokens, 4);
}

/// Longer Disk root replica: exclusive disk_blocks captures the extension
/// beyond NPU; unweighted matched_tokens covers the full prefix.
#[test]
fn test_lower_tier_longer_replica_extends_coverage() {
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-long-replica", "t1");

    let tokens: Vec<i64> = (0..12).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 3);

    // NPU: only the first block.
    let wk_npu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "inst-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    entry
        .apply_event(
            &wk_npu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 100,
                    tokens_hash: hashes[0].0,
                }],
            }),
        )
        .unwrap();

    // Disk: full 3-block root chain.
    let wk_disk = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-disk".into(),
        dp_rank: 0,
        medium: StorageMedium::Disk,
    };
    entry
        .apply_event(
            &wk_disk,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: 200,
                        tokens_hash: hashes[0].0,
                    },
                    KvCacheStoredBlockData {
                        block_hash: 201,
                        tokens_hash: hashes[1].0,
                    },
                    KvCacheStoredBlockData {
                        block_hash: 202,
                        tokens_hash: hashes[2].0,
                    },
                ],
            }),
        )
        .unwrap();

    let resp = indexer
        .query("model-long-replica", "t1", &tokens, 4)
        .unwrap();
    let dp0 = &resp.tenants["t1"]["inst-1"].dp["0"];
    assert_eq!(dp0.npu_blocks, 1);
    assert_eq!(
        dp0.disk_blocks, 2,
        "exclusive Disk blocks are the extension beyond NPU"
    );
    assert_eq!(
        dp0.matched_tokens,
        3 * 4,
        "unweighted coverage extends to the longer replica"
    );
}

#[test]
fn test_no_indexer_error() {
    let indexer = Indexer::new();
    let err = indexer.query("no-such-model", "default", &[1, 2, 3, 4], 4);
    assert!(err.is_err());
    assert!(matches!(
        err.unwrap_err(),
        KvConductorError::NoIndexer { .. }
    ));
}

#[test]
fn test_hbm_cleared_removes_empty_worker_lookup() {
    let entry = IndexerEntry::new();
    let worker = WorkerKey {
        instance_id: "cleared-worker".into(),
        backend_id: "cleared-worker".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    entry
        .apply_event(
            &worker,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 1,
                    tokens_hash: 11,
                }],
            }),
        )
        .unwrap();
    assert!(entry.lookups.read().contains_key(&worker));

    entry
        .apply_event(&worker, &KvCacheEventData::Cleared)
        .unwrap();

    assert!(!entry.lookups.read().contains_key(&worker));
}

#[test]
fn test_maintenance_expires_offload_and_removes_empty_entry() {
    let indexer = Indexer::with_config(CacheMaintenanceConfig {
        offload_ttl: std::time::Duration::ZERO,
        ..CacheMaintenanceConfig::default()
    });
    let entry = indexer.get_or_create("stale-model", "stale-tenant");
    entry.ingest_offload_blocks(&[(1, 11, None)]);
    assert_eq!(entry.pending_count(), 1);

    let protected = FxHashSet::default();
    indexer.maintenance(&protected);

    assert!(indexer.get("stale-model", "stale-tenant").is_none());
}

#[test]
fn test_maintenance_keeps_registered_empty_entry() {
    let indexer = Indexer::new();
    indexer.get_or_create("active-model", "active-tenant");
    let mut protected = FxHashSet::default();
    protected.insert(IndexerKey {
        model_name: "active-model".into(),
        tenant_id: "active-tenant".into(),
    });

    indexer.maintenance(&protected);

    assert!(indexer.get("active-model", "active-tenant").is_some());
}

#[test]
fn test_exclusive_sum_is_unweighted_matched_tokens() {
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-coverage", "t1");

    let tokens: Vec<i64> = (0..12).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 3);

    let wk_npu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "inst-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Npu,
    };
    entry
        .apply_event(
            &wk_npu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 100,
                    tokens_hash: hashes[0].0,
                }],
            }),
        )
        .unwrap();

    let wk_cpu = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-cpu".into(),
        dp_rank: 0,
        medium: StorageMedium::Cpu,
    };
    entry
        .apply_event(
            &wk_cpu,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(100),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 200,
                    tokens_hash: hashes[1].0,
                }],
            }),
        )
        .unwrap();

    let wk_disk = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-disk".into(),
        dp_rank: 0,
        medium: StorageMedium::Disk,
    };
    entry
        .apply_event(
            &wk_disk,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(200),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 300,
                    tokens_hash: hashes[2].0,
                }],
            }),
        )
        .unwrap();

    let resp = indexer.query("model-coverage", "t1", &tokens, 4).unwrap();
    let dp0 = &resp.tenants["t1"]["inst-1"].dp["0"];
    assert_eq!(dp0.npu_blocks, 1);
    assert_eq!(dp0.cpu_blocks, 1);
    assert_eq!(dp0.disk_blocks, 1);
    assert_eq!(
        dp0.matched_tokens,
        (dp0.npu_blocks + dp0.cpu_blocks + dp0.disk_blocks) * 4
    );
}

#[test]
fn test_disk_only_coverage_matched_tokens() {
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-disk-only", "t1");

    let tokens: Vec<i64> = vec![1, 2, 3, 4];
    let hashes = compute_block_hash_for_seq(&tokens, 4);

    let wk_disk = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "pool-disk".into(),
        dp_rank: 0,
        medium: StorageMedium::Disk,
    };
    entry
        .apply_event(
            &wk_disk,
            &KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 100,
                    tokens_hash: hashes[0].0,
                }],
            }),
        )
        .unwrap();

    let resp = indexer.query("model-disk-only", "t1", &tokens, 4).unwrap();
    let dp0 = &resp.tenants["t1"]["inst-1"].dp["0"];
    assert_eq!(dp0.npu_blocks, 0);
    assert_eq!(dp0.cpu_blocks, 0);
    assert_eq!(dp0.disk_blocks, 1);
    assert_eq!(dp0.matched_tokens, 4);
}
