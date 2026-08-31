# [2026-08-31] 池化索引按引擎私有 block_hash 建边，CPU 命中被外来短链截断

- **现象 (Symptom)**：`/query` 里 HBM 命中很深、CPU 池却几乎不命中，而且**所有没有自己 HBM
  前缀的 DP 报的是同一个很小的数**。生产 trace（`block_size=128`）：

  ```text
  vllm-prefill-3: longest_matched=12032   # dp0 npu=94
    dp3/dp6/dp7: matched_tokens=896  npu=0 cpu=7  cpu_local=3 cpu_remote=4
    dp2:         matched_tokens=10624 npu=82 cpu=1
  vllm-prefill-4: longest_matched=896     # 8 个 DP 全是 896 / cpu=7
  ```

  用户的疑问正确：这一轮算完的 KV 是会异步写一份进池子的，池子里确实有完整前缀，
  但索引报不出来。「所有 DP 都恰好是 7 块」这一点本身就说明瓶颈在**索引里的那条链**，
  而不是任何 DP 的缓存状态。
- **根因 (Root cause)**：`lower_tier.rs` 以 `TransitionKey { parent_hash: Option<SequenceBlockHash>,
  local_hash }` 建 continuation edge，`parent_hash` 用的是**引擎自己的滚动 `block_hash`**。
  vLLM 未固定 `PYTHONHASHSEED` 时 `NONE_HASH` 由 `os.urandom` 生成（每 EngineCore 进程一个），
  同一份内容在不同引擎/不同 DP 里编号完全不同。于是：
  1. 根边 `(None, tokens_hash[0])` 先写者独占。`EdgeOwnersEntry::insert`
     (`lower_tier.rs:77`) 在 child 不同时返回 false，`store_blocks` 随即 `break`
     (`lower_tier.rs:270-280`)——后来引擎那条 94 块的完整链整条悬空。
  2. `reachable_chain` (`lower_tier.rs:410`) 从 `parent=None` 出发后，每一步都拿上一步的
     `child_hash` 当 parent，等于**锁死在先写者的哈希空间里**，量到的是那条 7 块的短链。
  3. 中途 offload 的片段锚在引擎私有 parent 上，除非同一引擎从 0 号位起的整条链都在池中，
     否则任何 DP 都走不到（trace 里 dp2 的 `cpu=1` 就是这种孤立片段，恰好对上了它自己的断点）。
  4. HBM 侧没有这个问题：radix 树键是 `LocalBlockHash` 内容路径，`block_hash` 只是元数据，
     所以 HBM 能跨引擎累积到 94 块——这正是"NPU 很高、CPU 很低"这种不对称的来源。
- **为什么会写出 (Why)**：把 `SequenceBlockHash` 当成了全局稳定的块身份。它对 HBM 树是够用的
  （只做 per-worker 反查），但池化块是**跨引擎共享**的资源，用引擎私有的链去索引共享资源，
  就等于按进程把一份池化前缀切成互不连通的多条链。教训：判断一个哈希能不能做共享身份，要看它
  的**种子作用域**，而不是看它"看起来像内容哈希"。仓库里其实早有线索——
  `docs/zh/user_guide/deployment/cases/ModelArts&yuanrong/...md:1941` 的排障项就是
  "元戎 KV cache 命中率非预期为 0：验证 `PYTHONHASHSEED` 与启动设置一致"。
- **修复 (Fix)**：池化块改按 `PrefixChainHash` 落位——`tokens_hash` 的滚动前缀哈希，只由内容决定。
  - `protocols.rs`：新增 `PrefixChainHash`；`hashing.rs`：`PREFIX_CHAIN_ROOT` /
    `extend_prefix_chain` / `compute_prefix_chain_for_seq`。
  - `concurrent_tree.rs`：`Block` 增加 `prefix_chain`，`apply_store` 沿树维护，作为事件侧解析锚点。
  - `lower_tier.rs`：`edges: (parent, local) → child` 换成 `positions: chain → Owners`；
    `reachable_chain` → `reachable_span(chain, start_pos)`；删除 `query_contiguous_hits` /
    `LowerTierContinuation` / `edge_owners`；per-worker 反查表加 `chain_refs` 计数，
    同一份内容被同一 worker 用两个 engine hash 上报时不会被误删。
  - `indexer/mod.rs`：`resolve_pooled_blocks` 把 `parent_hash` 解析成内容位置（HBM 节点 →
    池化反查表 → offload/content 缓存链回溯）；`TierBreakpoint` 只剩 `end_pos`；
    `PrefixMatch` 退化成 `depth` 后删除。解析不出来的块丢弃并计入
    `unanchored_pooled_blocks`（`GET /workers` 暴露）。
- **测试拦截 (Test interception)**：
  - `indexer/tests.rs::test_pooled_prefix_is_not_truncated_by_a_shorter_foreign_chain`
    按上面的 trace 复原 16 个 DP + 两个哈希空间：修复前每个无 HBM 的 DP 报 896，修复后全部 12032。
  - `test_pooled_tail_anchored_on_own_hbm_extends_the_shared_walk`：挂在自己 HBM 块之后的
    offload 片段，对所有 DP 都可从 0 号位走到。
  - `test_unanchored_pooled_blocks_are_counted_and_not_reported`：定位不了前缀的块被丢弃并计数。
  - `lower_tier.rs::different_engine_hashes_for_one_prefix_merge` /
    `re_offloaded_content_keeps_ownership_until_last_hash_leaves`。
  - 原有 `test_hbm_breakpoint_not_shared_across_{instances,dp_ranks}` /
    `test_own_hbm_bridges_pool_gap_and_differentiates_dps` 未改动即通过，说明 per-DP 断点作用域
    与 HBM 桥接空洞的语义都保住了。
- **场景 (Scenario)**：Mooncake / Memcache 中心化池 + 多 prefill 实例（或多 DP，每 DP 一个
  EngineCore 进程）+ 引擎侧 offload 事件正常上报。未固定 `PYTHONHASHSEED` 时必然触发；即使固定了，
  只要有引擎在中途位置 offload 而更早的位置由别的引擎写入，旧实现同样走不通。
- **关键词 (Keywords)**：kv_conductor、lower_tier、PrefixChainHash、cpu_blocks 偏低、
  PYTHONHASHSEED、NONE_HASH、continuation edge、unanchored_pooled_blocks
