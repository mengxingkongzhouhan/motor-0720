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

/// `Indexer::query` with an empty topology: no DP locations are registered, so
/// nothing counts as pool-local. Tests that exercise locality build their own
/// topology and call `Indexer::query` directly.
fn query_default(
    indexer: &Indexer,
    model_name: &str,
    tenant_id: &str,
    token_ids: &[i64],
    block_size: u32,
) -> Result<QueryResponse, KvConductorError> {
    indexer.query(
        model_name,
        tenant_id,
        token_ids,
        block_size,
        &NodeTopology::default(),
    )
}

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
    let resp = query_default(&indexer, "model-a", "tenant-1", &tokens, 4).unwrap();
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
    let resp = query_default(&indexer, "model-b", "t1", &tokens_a, 4).unwrap();
    let tenant = &resp.tenants["t1"];

    let imd1 = &tenant["inst-1"];
    let dp0 = &imd1.dp["0"];
    assert!(
        dp0.npu_blocks > 0,
        "inst-1 should have NPU match for tokens_a"
    );
    assert_eq!(dp0.cpu_blocks, 0, "inst-1 should have no CPU match");

    // Query with tokens_b — should match inst-2 (CPU) only
    let resp = query_default(&indexer, "model-b", "t1", &tokens_b, 4).unwrap();
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
    // Lower-tier results are reported per DP, not per owning WorkerKey: the walk
    // is ownership-blind, so `backend_id` is the DP's own instance id rather
    // than the pool daemon that happened to store the block.
    let cpu_dp = WorkerKey {
        instance_id: "inst-1".into(),
        backend_id: "inst-1".into(),
        dp_rank: 0,
        medium: StorageMedium::Cpu,
    };
    assert_eq!(
        overlap.blocks.get(&cpu_dp).copied().unwrap_or(0),
        1,
        "CPU should continue 1 block from HBM breakpoint"
    );

    // Exclusive: npu=2, cpu=1; default weights 1/1/0 → matched = 3 blocks.
    let resp = query_default(&indexer, "model-c", "t1", &tokens, 4).unwrap();
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

    let resp = query_default(&indexer, "model-d", "t1", &tokens, 4).unwrap();
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

    let resp = query_default(&indexer, "model-e", "t1", &tokens, 4).unwrap();
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

    let resp = query_default(&indexer, "model-f", "t1", &tokens, 4).unwrap();
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

    let resp = query_default(&indexer, "model-overlap", "t1", &tokens, 4).unwrap();
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

    let resp = query_default(&indexer, "model-short-break", "t1", &tokens, 4).unwrap();
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

    let resp = query_default(&indexer, "model-break-scope", "t1", &tokens, 4).unwrap();
    let tenant = &resp.tenants["t1"];

    // inst-a resumes from its OWN breakpoint and then walks ownership-blind, so
    // it picks up inst-b's pooled block: it can fetch that block and serve all
    // three without recompute.
    let dp_a = &tenant["inst-a"].dp["0"];
    assert_eq!(dp_a.npu_blocks, 2, "blocks 0..2 are local to inst-a's HBM");
    assert_eq!(dp_a.cpu_blocks, 1, "block 2 is fetched from the pool");
    assert_eq!(dp_a.matched_tokens, 3 * 4);

    // inst-b is the asymmetric case and the reason breakpoints stay per-DP:
    // blocks 0..2 live only in inst-a's *HBM*, which is device memory and is
    // NOT fetchable across nodes. inst-b therefore cannot serve the prefix at
    // all, and must not inherit inst-a's start position to claim otherwise.
    assert!(
        !tenant.instances.contains_key("inst-b"),
        "inst-b cannot reach blocks 0..2 (HBM is not poolable) and must not \
         borrow inst-a's start position, got {:?}",
        tenant.instances.get("inst-b")
    );
}

/// Store one lower-tier chain for `worker`, anchored at `parent`.
fn store_chain(
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

fn worker_of(instance_id: &str, dp_rank: u32, medium: StorageMedium) -> WorkerKey {
    WorkerKey {
        instance_id: instance_id.into(),
        backend_id: instance_id.into(),
        dp_rank,
        medium,
    }
}

#[test]
fn test_pooled_walk_crosses_ownership_boundary() {
    // inst-a's DRAM holds [0,2), inst-b's DRAM continues [2,4). Pooled blocks
    // are fetchable from any node, so BOTH DPs can serve all four blocks
    // without recompute — the walk must not stop at the ownership boundary.
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-pooled-walk", "t1");

    let tokens: Vec<i64> = (0..16).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 4);

    let a = worker_of("inst-a", 0, StorageMedium::Cpu);
    let b = worker_of("inst-b", 0, StorageMedium::Cpu);
    store_chain(&entry, &a, None, &[(100, hashes[0].0), (101, hashes[1].0)]);
    store_chain(
        &entry,
        &b,
        Some(101),
        &[(102, hashes[2].0), (103, hashes[3].0)],
    );

    let resp = query_default(&indexer, "model-pooled-walk", "t1", &tokens, 4).unwrap();
    let tenant = &resp.tenants["t1"];

    for instance_id in ["inst-a", "inst-b"] {
        let dp0 = &tenant[instance_id].dp["0"];
        assert_eq!(
            dp0.cpu_blocks, 4,
            "{instance_id} can fetch every pooled block, so the walk must cross \
             the ownership boundary at position 2"
        );
        assert_eq!(dp0.npu_blocks, 0);
        assert_eq!(dp0.matched_tokens, 4 * 4);
    }
}

#[test]
fn test_own_hbm_bridges_pool_gap_and_differentiates_dps() {
    // The pooled chain is broken: [0,2) exists, position 2 is missing, [3,5)
    // exists anchored after position 2. Only inst-a's own HBM covers position 2,
    // and HBM is device memory — not fetchable across nodes. So inst-a bridges
    // the gap and reaches 5, while inst-b stops at the gap.
    //
    // This is why per-DP results still differ under an ownership-blind walk:
    // the walk's *start* is per-DP even though its *continuation* is not.
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-bridge", "t1");

    let tokens: Vec<i64> = (0..20).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 5);

    // inst-a HBM: positions 0..3.
    store_chain(
        &entry,
        &worker_of("inst-a", 0, StorageMedium::Npu),
        None,
        &[(100, hashes[0].0), (101, hashes[1].0), (102, hashes[2].0)],
    );
    // Pool head [0,2) and, after the gap at position 2, tail [3,5).
    let b_cpu = worker_of("inst-b", 0, StorageMedium::Cpu);
    store_chain(
        &entry,
        &b_cpu,
        None,
        &[(100, hashes[0].0), (101, hashes[1].0)],
    );
    store_chain(
        &entry,
        &b_cpu,
        Some(102),
        &[(103, hashes[3].0), (104, hashes[4].0)],
    );

    let resp = query_default(&indexer, "model-bridge", "t1", &tokens, 4).unwrap();
    let tenant = &resp.tenants["t1"];

    // inst-a: 3 blocks local in HBM, then fetches the tail across the gap.
    let dp_a = &tenant["inst-a"].dp["0"];
    assert_eq!(dp_a.npu_blocks, 3);
    assert_eq!(dp_a.cpu_blocks, 2, "tail fetched after bridging the gap");
    assert_eq!(dp_a.matched_tokens, 5 * 4);

    // inst-b: only the pooled head is reachable; position 2 lives in inst-a's
    // HBM, which it cannot fetch.
    let dp_b = &tenant["inst-b"].dp["0"];
    assert_eq!(dp_b.npu_blocks, 0);
    assert_eq!(dp_b.cpu_blocks, 2, "stops at the gap");
    assert_eq!(dp_b.matched_tokens, 2 * 4);

    // The affinity signal survives: the two DPs are not tied.
    assert!(dp_a.matched_tokens > dp_b.matched_tokens);
}

#[test]
fn test_dp_without_local_data_still_reports_pooled_reach() {
    // inst-c holds blocks for an unrelated prefix, so it matches nothing of its
    // own for this query. It can still fetch the pooled prefix, so reporting 0
    // would over-estimate its prefill cost.
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-idle-dp", "t1");

    let tokens: Vec<i64> = (0..8).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 2);
    let other: Vec<i64> = (900..908).collect();
    let other_hashes = compute_block_hash_for_seq(&other, 4);

    store_chain(
        &entry,
        &worker_of("inst-a", 0, StorageMedium::Cpu),
        None,
        &[(100, hashes[0].0), (101, hashes[1].0)],
    );
    // inst-c is known to the index but holds a different prefix.
    store_chain(
        &entry,
        &worker_of("inst-c", 0, StorageMedium::Npu),
        None,
        &[(900, other_hashes[0].0)],
    );

    let resp = query_default(&indexer, "model-idle-dp", "t1", &tokens, 4).unwrap();
    let tenant = &resp.tenants["t1"];

    let dp_c = &tenant["inst-c"].dp["0"];
    assert_eq!(dp_c.npu_blocks, 0, "no local HBM hit for this prefix");
    assert_eq!(
        dp_c.cpu_blocks, 2,
        "inst-c can still fetch both pooled blocks"
    );
    assert_eq!(dp_c.matched_tokens, 2 * 4);
}

/// Topology placing each `(instance, dp)` on a named node.
fn topology_of(entries: &[(&str, u32, &str, &str)]) -> NodeTopology {
    let mut topo = NodeTopology::default();
    for (instance_id, dp_rank, pod_ip, node_id) in entries {
        topo.record(pod_ip, Some(node_id), instance_id, *dp_rank);
    }
    topo
}

#[test]
fn test_pooled_hits_split_into_local_and_remote() {
    // inst-a and inst-b are different Pods on node-1; inst-c is on node-2.
    // inst-a's HBM covers [0,1); the pooled chain [1,4) is owned by inst-b (same
    // machine as inst-a) for block 1 and inst-c (another machine) for blocks 2,3.
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-locality", "t1");

    let tokens: Vec<i64> = (0..16).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 4);

    // Block 0 lives only in HBM, which is not poolable — so a DP needs its own
    // copy to have any entry point into the pooled chain that follows.
    for instance_id in ["inst-a", "inst-c"] {
        store_chain(
            &entry,
            &worker_of(instance_id, 0, StorageMedium::Npu),
            None,
            &[(100, hashes[0].0)],
        );
    }
    store_chain(
        &entry,
        &worker_of("inst-b", 0, StorageMedium::Cpu),
        Some(100),
        &[(101, hashes[1].0)],
    );
    store_chain(
        &entry,
        &worker_of("inst-c", 0, StorageMedium::Cpu),
        Some(101),
        &[(102, hashes[2].0), (103, hashes[3].0)],
    );

    let topo = topology_of(&[
        ("inst-a", 0, "10.0.0.5", "node-1"),
        ("inst-b", 0, "10.0.0.6", "node-1"),
        ("inst-c", 0, "10.0.1.7", "node-2"),
    ]);
    let resp = indexer
        .query("model-locality", "t1", &tokens, 4, &topo)
        .unwrap();
    let tenant = &resp.tenants["t1"];

    // inst-a: block 0 local in HBM; of the 3 pooled blocks, block 1 sits on its
    // own machine (inst-b) and blocks 2,3 are a machine away.
    let dp_a = &tenant["inst-a"].dp["0"];
    assert_eq!(dp_a.npu_blocks, 1);
    assert_eq!(dp_a.cpu_blocks, 3);
    assert_eq!(dp_a.cpu_local_blocks, 1, "block 1 is on node-1 via inst-b");
    assert_eq!(dp_a.cpu_remote_blocks, 2, "blocks 2,3 are on node-2");

    // inst-c is on node-2, so the split flips: blocks 2,3 are its own, block 1
    // has to come from node-1.
    let dp_c = &tenant["inst-c"].dp["0"];
    assert_eq!(dp_c.npu_blocks, 1);
    assert_eq!(dp_c.cpu_blocks, 3);
    assert_eq!(dp_c.cpu_local_blocks, 2, "blocks 2,3 are on node-2");
    assert_eq!(dp_c.cpu_remote_blocks, 1, "block 1 is on node-1");

    // inst-b owns pooled block 1 but has no HBM copy of block 0, so it has no
    // entry point at all — its own block sits behind a gap it cannot cross.
    assert!(!tenant.instances.contains_key("inst-b"));

    // The invariant holds for every DP: the split covers exactly this tier's
    // exclusive blocks, since blocks already in local HBM need no fetch.
    for imd in tenant.instances.values() {
        for dp in imd.dp.values() {
            assert_eq!(
                dp.cpu_local_blocks + dp.cpu_remote_blocks,
                dp.cpu_blocks,
                "local + remote must equal cpu_blocks: {dp:?}"
            );
            assert_eq!(dp.disk_local_blocks + dp.disk_remote_blocks, dp.disk_blocks);
        }
    }
}

#[test]
fn test_root_walk_reports_per_node_hit_counts() {
    // No HBM at all, so every DP walks from root. The histogram reports how many
    // of the shared span each machine holds — a DP-agnostic view the per-DP
    // split cannot express for a walk that is identical for everyone.
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-histogram", "t1");

    let tokens: Vec<i64> = (0..12).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);
    assert_eq!(hashes.len(), 3);

    // node-1 holds blocks 0,1; node-2 holds block 2 (and also co-owns block 1).
    store_chain(
        &entry,
        &worker_of("inst-a", 0, StorageMedium::Cpu),
        None,
        &[(100, hashes[0].0), (101, hashes[1].0)],
    );
    store_chain(
        &entry,
        &worker_of("inst-c", 0, StorageMedium::Cpu),
        Some(100),
        &[(101, hashes[1].0), (102, hashes[2].0)],
    );

    let topo = topology_of(&[
        ("inst-a", 0, "10.0.0.5", "node-1"),
        ("inst-c", 0, "10.0.1.7", "node-2"),
    ]);
    let resp = indexer
        .query("model-histogram", "t1", &tokens, 4, &topo)
        .unwrap();
    let tenant = &resp.tenants["t1"];

    let cpu_hits = &tenant.root_node_hits["cpu"];
    assert_eq!(cpu_hits["node-1"], 2, "node-1 holds blocks 0 and 1");
    assert_eq!(cpu_hits["node-2"], 2, "node-2 holds blocks 1 and 2");

    // Each DP's own split is consistent with the histogram: inst-a is on node-1
    // and holds 2 of the 3 root-walked blocks locally.
    assert_eq!(tenant["inst-a"].dp["0"].cpu_blocks, 3);
    assert_eq!(tenant["inst-a"].dp["0"].cpu_local_blocks, 2);
    assert_eq!(tenant["inst-a"].dp["0"].cpu_remote_blocks, 1);
}

#[test]
fn test_unlocated_owner_counts_as_remote() {
    // A block whose owners have no registered location must count as remote:
    // scoring an unknown location as local would understate the fetch cost.
    let indexer = Indexer::new();
    let entry = indexer.get_or_create("model-unlocated", "t1");

    let tokens: Vec<i64> = (0..8).collect();
    let hashes = compute_block_hash_for_seq(&tokens, 4);

    store_chain(
        &entry,
        &worker_of("inst-a", 0, StorageMedium::Cpu),
        None,
        &[(100, hashes[0].0), (101, hashes[1].0)],
    );

    // Only inst-b is registered; the owner inst-a has no location.
    let topo = topology_of(&[("inst-b", 0, "10.0.0.6", "node-1")]);
    let resp = indexer
        .query("model-unlocated", "t1", &tokens, 4, &topo)
        .unwrap();
    let tenant = &resp.tenants["t1"];

    let dp_b = &tenant["inst-b"].dp["0"];
    assert_eq!(dp_b.cpu_blocks, 2, "inst-b can still fetch both blocks");
    assert_eq!(dp_b.cpu_local_blocks, 0, "owner location unknown");
    assert_eq!(dp_b.cpu_remote_blocks, 2);
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

    let resp = query_default(&indexer, "model-break-dp", "t1", &tokens, 4).unwrap();
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

    let resp = query_default(&indexer, "model-h", "t1", &tokens, 4).unwrap();
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

    let resp = query_default(&indexer, "model-g", "t1", &tokens, 4).unwrap();
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

    let resp = query_default(&indexer, "model-i", "t1", &tokens, 4).unwrap();
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

    let resp = query_default(&indexer, "model-long-replica", "t1", &tokens, 4).unwrap();
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
    let err = query_default(&indexer, "no-such-model", "default", &[1, 2, 3, 4], 4);
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

    let resp = query_default(&indexer, "model-coverage", "t1", &tokens, 4).unwrap();
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

    let resp = query_default(&indexer, "model-disk-only", "t1", &tokens, 4).unwrap();
    let dp0 = &resp.tenants["t1"]["inst-1"].dp["0"];
    assert_eq!(dp0.npu_blocks, 0);
    assert_eq!(dp0.cpu_blocks, 0);
    assert_eq!(dp0.disk_blocks, 1);
    assert_eq!(dp0.matched_tokens, 4);
}
