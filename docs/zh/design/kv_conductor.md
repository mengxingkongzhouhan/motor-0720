# KV Conductor 设计文档

## 架构总览

```text
                        ┌──────────────────────────────────────┐
                        │             KV Conductor             │
                        │                                      │
   Engine Worker        │  ┌──────────┐    ┌────────────────┐  │
   (vLLM/SGLang)        │  │ Registry │    │    Indexer     │  │
       |                │  │          │    │                │  │
       |  register      │  │ workers  │--->│ DashMap<       │  │
       +--------------->│  │ endpoints│    │ (model,tenant) │  │
       |                │  └──────────┘    │   -> Entry     │  │
       |  ZMQ / HTTP    │                  │                │  │
       |  KV events     │                  └───┬───────┬────┘  │
       +--------------->│                      |       |       │
       |                │               ┌──────┘       └──┐    │
       |  query         │               v                 v    │
       +--------------->│    ┌──────────────┐  ┌──────────────┐│
       |                │    │  HBM Tree    │  │  CPU/Disk    ││
   Coordinator          │    │ (RadixTree)  │  │ (LowerTier)  ││
       |                │    │              │  │              ││
       |  200 OK        │    │ prefix chain │  │ prefix-chain ││
       <----------------│    │ matched      │  │ positions    ││
                        │    │ block counts │  │ matched      ││
                        │    └──────────────┘  │ block counts ││
                        │                      └──────────────┘│
                        └──────────────────────────────────────┘
```

模块职责：

| 模块 | 文件 | 职责 |
|------|------|------|
| HTTP Server | `server.rs` | Axum 路由，CORS，TraceLayer |
| Worker Registry | `registry.rs` | 注册/注销，事件/查询路由，ZMQ 订阅管理 |
| Indexer | `indexer/mod.rs` | Per-(model, tenant) 索引生命周期、两阶段缓存、三层匹配聚合与维护 |
| HBM Tree | `concurrent_tree.rs` | 并发 Radix Tree，前缀链匹配 |
| CPU/Disk Index | `lower_tier.rs` | 按内容前缀哈希落位的池化块索引，位置 0 走查 + 断点续查 |
| Hashing | `hashing.rs` | XXH3 token → LocalBlockHash |
| Backend | `backend.rs` | 多后端适配（Mooncake/Memcache/YuanRong） |
| ZMQ | `zmq_subscriber.rs` | ZMQ SUB 事件接入 |
| Events | `events/` | vLLM/Pool 事件解析与规范化 |
| Protocols | `protocols.rs` | API 类型定义，wire format |
| Error | `error.rs` | `KvConductorError` 错误类型 |

**身份模型**：每个缓存持有者是一个 `WorkerKey` 四元组 `(instance_id, backend_id, dp_rank, medium)`。
同一实例同一 DP 的 HBM / CPU / Disk 块是三个不同的 WorkerKey，查询时按 `(instance_id, dp_rank)`
聚合跨介质命中。`backend_id` 是块的来源后端（引擎实例或 pool daemon），用于事件路由。

**匹配语义**：查询先收集每 DP 各介质的绝对覆盖终点，再按优先级 NPU > CPU > Disk
互斥切分为 `npu_blocks` / `cpu_blocks` / `disk_blocks`（同前缀副本只归最高优先级介质）。
`matched_tokens = (npu + cpu + disk) × block_size`（未加权真实覆盖）。亲和性评分由
Coordinator 调度器（`kv_cache_affinity`）按 `*_blocks` 与配置项
`scheduler_config.kv_affinity.w_npu/w_cpu/w_disk`（默认 `1.0/1.0/0.0`）加权后完成。

---

## 多级存储介质设计

### 三层模型

```text
   ┌─────────────────────────────────────────────────────┐
   │                     KV Conductor                    │
   │                                                     │
   │  ┌──────────┐   ┌──────────────┐   ┌──────────────┐ │
   │  │   HBM    │   │     CPU      │   │     DISK     │ │
   │  │  (NPU)   │   │  (Host DDR)  │   │  (SSD/NVMe)  │ │
   │  │          │   │              │   │              │ │
   │  │  Radix   │   │ Continuation │   │ Continuation │ │
   │  │   Tree   │   │    Edges     │   │    Edges     │ │
   │  │          │   │              │   │              │ │
   │  └────┬─────┘   └──────┬───────┘   └──────┬───────┘ │
   │       │                │                  │         │
   │       └────────────────┼──────────────────┘         │
   │                        ▼                            │
   │             ┌─────────────────────┐                 │
   │             │    Query Response   │                 │
   │             │ npu/cpu/disk_blocks │                 │
   │             │   + matched_tokens  │                 │
   │             └─────────────────────┘                 │
   └─────────────────────────────────────────────────────┘
```

- 下层介质对每个 DP **无条件从位置 0 走查**；上层断点更远的 DP 额外从断点再走一遍（断点不跨 DP 共享），取绝对终点最远者（如实报告更长副本）
- HBM 是前缀树（从 root 走）；CPU/Disk 是按内容前缀哈希索引的位置集合（位置 0 走查 + 断点续查）
- 同一 `(instance_id, dp_rank)` 的跨介质命中在响应中聚合到同一个 DP 条目

### HBM 索引：ConcurrentRadixTree

HBM 使用前缀链 Radix Tree，每个节点以 `LocalBlockHash`（XXH3 token-content hash）为键：

```text
  root
   |
   +-[H0]-- Block { workers: {W1, W2}, block_hash: seq100 }
   |    |
   |    +-[H1]-- Block { workers: {W1, W2}, block_hash: seq200 }
   |    |    |
   |    |    +-[H2]-- Block { workers: {W1}, block_hash: seq300 }
   |    |
   |    +-[H3]-- Block { workers: {W2}, block_hash: seq400 }
   |
   +-[H4]-- Block { workers: {W3} }
```

**为什么用 RadixTree？**

- `LocalBlockHash` 是独立 XXH3 内容哈希，不包含前缀信息
- 仅靠哈希值无法判断 "block 3 是否紧接 block 2"
- RadixTree 以 `parent → child` 的树结构显式编码前缀链
- 查询时从 root 逐层遍历，第一个缺失即停，天然保证最长连续前缀

**并发模型**：

- 查询路径（`find_matches_detailed`）：仅读锁，多个查询互不阻塞
- 变更路径（`apply_store`/`apply_remove`）：hand-over-hand 写锁，先锁父再锁子
- per-Worker 反向查找表（`WorkerLookup`）：`SequenceBlockHash → tree node`，O(1) 定位
- 遍历优化：active Worker 集合收缩到 1 个后改为单 Worker 成员检查，避免集合差集开销

**弱一致性语义**：RadixTree 不是 MVCC 结构。`workers` 使用 `Arc<FxHashSet>` 写时复制
（CoW）——查询热路径只 bump 引用计数，变更路径用 `Arc::make_mut`。当 Worker 被移除时
（`remove_worker`），若节点的所有 Worker 均离开，其 `children` map 一并清空以回收内存；
已有 `Arc` 引用的旧集合不受影响，正在并发遍历的查询可以安全完成。

**维护**：HBM `Removed` / `Cleared` 只做精确索引删除，不在事件热路径扫描整棵树。
后台 maintenance 按周期调用 `sweep_stale_nodes()`，统一回收无 Worker 且无子节点的空节点。
`Cleared` 会同时删除清空后的外层 `WorkerLookup` key；Worker 注销则由
`remove_worker_all_media` 清除该实例/DP 在所有介质上的索引。这样避免删除事件周期性出现
全树扫描延迟尖峰，同时保证停止 ingest 后孤儿节点仍会被回收。

### CPU/Disk 索引：LowerTierIndexer

CPU/DISK 不使用完整 RadixTree，而是把每个池化块登记在它的**内容前缀位置**上：

```text
  PrefixChainHash: chain(i) = fold(chain(i-1), tokens_hash(i)),  chain(-1) = ROOT

  positions: chain(i) -> {owner workers}

  Example (query [H0, H1, H2]):
    chain0 = fold(ROOT,   H0)   -> {inst-a/dp0, inst-a/dp1}
    chain1 = fold(chain0, H1)   -> {inst-a/dp0, inst-a/dp1}
    chain2 = fold(chain1, H2)   -> {inst-b/dp0}
```

**为什么不用 RadixTree？**

- CPU/Disk block 数量远大于 HBM（可能千万级），完整树内存开销过大
- 一层 `chain → owners` 哈希表就够：`chain` 本身已经编码了整条前缀，
  查询侧从自己的 token 序列直接算出每个位置的 `chain`，逐位查表到第一个缺失即停，
  连续性由构造保证，不需要物化前缀树，也不需要引擎给锚点

**为什么按内容而不是按引擎的 `block_hash`？**

vLLM 的 `block_hash` 是引擎私有的滚动哈希：未固定 `PYTHONHASHSEED` 时 `NONE_HASH`
每进程随机，同一份内容在不同引擎里编号完全不同。池化块是跨引擎共享的，按
`(parent_seq_hash, tokens_hash)` 建边会导致：

- 根边 `(None, H₀)` 先写者独占，后来引擎对同一内容给出的 child 不同而被拒，它那条
  链整条悬空；从位置 0 走查只能沿先写者的哈希空间前进
- 中途 offload 的片段锚在引擎私有的 parent 上，除非该引擎自己从 0 号位起的整条链都在池中，
  否则任何 DP 都走不到
- 于是"HBM 命中 94 块、池化只报 7 块"这种明显不对的结果会稳定出现（见回归测试
  `test_pooled_prefix_is_not_truncated_by_a_shorter_foreign_chain`）

按 `PrefixChainHash` 落位后，同一份前缀在所有引擎里落在同一批位置：副本自动去重合并，
彼此都算 owner，片段按内容拼接。与 HBM 树一样，它忽略 vLLM 的 `extra_keys`
（LoRA / 多模态 / cache salt）。

**位置所有权**：每个位置有一个或多个 owner Worker（`One` / `Many` 双形态存储，
省掉单 owner 的集合分配）。走查**不校验 owner**（池化块任意节点可取），owner 只用来把
命中拆成本地/远端。

**事件侧落位**：事件里的 `parent_hash` 需要解析成内容位置，顺序为 HBM 节点上记录的
`prefix_chain` → 池化反查表 → offload/content 缓存链（沿 `parent_hash` 回溯，
这样多段 offload 链在任何一段被 pool 确认之前也能定位）。解析不出来时该事件的块被丢弃
并计入 `unanchored_pooled_blocks`（`GET /workers` 暴露），而不是记在猜测的前缀上。

**续查语义**（位置 0 走查与断点续查并列，取绝对终点最远者）：

```text
  query: [H0, H1, H2, H3, H4]  ->  chain0..chain4
  HBM tree returns: W1 depth=2

  Candidate a) resume from position 2:
    // only granted to W1's own (instance_id, dp_rank)
    chain2 present  OK
    chain3 present  OK
    chain4 missing  -> stop
    -> absolute end = 4

  Candidate b) walk from position 0 (always, shared by every DP):
    chain0 -> ... walk until the first position with no pooled replica
    -> compare with a); keep farther absolute end

  Absolute ends: npu_end=2, cpu_end=<farthest absolute end>
  Exclusive: npu_blocks=npu_end, cpu_blocks=max(0, cpu_end-npu_end)
  matched_tokens = (npu + cpu + disk) × block_size   // unweighted coverage
  // Coordinator affinity: round((npu×w_npu + cpu×w_cpu + disk×w_disk) × block_size)
```

### 下层走查忽略 owner：报"能免重算服务多长"而非"本地有多长"

池化块通过后端传输协议（`device_rdma` / `device_sdma` / `device_urma`，见
`mmc-local-*.conf` 的 `ock.mmc.local_service.protocol`）**任意节点可取**，所以别的 DP
持有的块同样能让本 DP 跳过重算。下层走查因此**不校验 owner**，只在某个位置没有任何池化副本
时停止。

**关键的介质不对称**：

| 介质 | 跨节点可取? | 走查语义 |
|------|------------|---------|
| HBM | ❌ 设备显存，取不到别人的 | `find_matches_detailed` 逐层求交，**严格按 owner** |
| CPU / Disk | ✅ 池化，任意节点可取 | `reachable_span` **忽略 owner** |

那各 DP 的结果凭什么还不同?两处，都是 per-DP 的：

1. **走查起点**。所有 DP 都走一遍位置 0 起的池链；自己上层断点更远的 DP 额外从断点再走一遍。
   因为 HBM 取不到别人的，只有持有那些块的 DP 能用它们跨过池链中的缺口。
2. **归属切分**。互斥切分把 `[0, npu_end)` 记为 NPU（本地、免费）、其余记为 CPU/Disk
   （需搬运），于是 `kv_affinity.w_cpu` / `w_disk` 成为"优先选本地已有的节点"的旋钮。

```text
  池链: [0,2) 存在，位置 2 缺失，[3,5) 存在
  inst-a HBM: [0,3)     inst-b DRAM: [0,2) 与 [3,5)

  inst-a: 自己断点 pos=3 -> reachable_span(chain, 3) -> 取到 inst-b 的 [3,5) -> 终点 5
          npu_blocks=3（本地）+ cpu_blocks=2（搬运）= 5 块
  inst-b: 无 HBM 命中 -> 位置 0 走查 -> [0,2) 后位置 2 缺失 -> 终点 2
          npu_blocks=0 + cpu_blocks=2 = 2 块

  => inst-a 靠自己的 HBM 桥接缺口而走得更远；亲和信号未被抹平
```

**断点仍只归产生它的 DP**：若允许借用别的 DP 的断点，只持有中间段的 worker 会跨过自己
**取不到**的空洞谎报前缀——因为空洞那一段在别人的 HBM 里：

```text
  inst-a HBM: [0,2)            -> breakpoint end_pos=2
  inst-b DRAM: [2,3) only

  借用 inst-a 断点：inst-b 从 pos=2 起跑 -> 终点 3 -> npu=0, cpu=3
    声称能服务 3 块，但位置 0..2 只在 inst-a 的 HBM 里，inst-b 取不到 -> 实际得全量重算
  按 DP 索引断点后：inst-b 无起点，不进 medium_ends
    而 inst-a 从自己断点起跑、取到 inst-b 的池块 -> npu=2 + cpu=1 = 3 块（正确）
```

位置 0 起的走查不依赖任何上层覆盖，所以「HBM 全被驱逐、池中保有完整前缀」这一池化核心场景仍能
如实报告。**所有已知 DP 都参与下层走查**（`known_dps()`：HBM lookups ∪ 各 tier 的
`worker_keys()`），本地什么都没有的 DP 也能从池子取，报 0 会高估它的 prefill 代价。位置 0
走查忽略 owner，因此对所有 DP 结果相同，只计算一次后复用。

---

## 后端适配抽象

```text
                    ┌────────────────┐
                    │  StoreBackend  │  (enum)
                    └────────┬───────┘
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌───────────┐  ┌───────────┐  ┌───────────┐
        │  Mooncake │  │  Memcache │  │  YuanRong │
        │           │  │           │  │           │
        │  Central  │  │  Central  │  │   Per-DP  │
        │    Pool   │  │    Pool   │  │   Ports   │
        └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              ▼              ▼              ▼
        ┌───────────┐  ┌───────────┐  ┌───────────┐
        │   IpOnly  │  │   IpOnly  │  │    None   │
        │  IP->DPs  │  │  IP->DPs  │  │ port = DP │
        └───────────┘  └───────────┘  └───────────┘
```

**设计要点**：

- HBM 事件来自引擎 Worker，**不经过 Pool**。Worker 身份 = `(instance_id, dp_rank)`，后端无关。
- CPU/Disk 事件来自 Pool Master/Daemon，携带 `backend_id`（节点 IP 或端口）。
  - Mooncake/Memcache：`backend_id` = 节点 IP → 通过 `hbm_ip_index` 关联到该节点所有 DP
  - YuanRong：每个 DP 独立端口，`backend_id` = 端口号 → 精确匹配 DP

**MatchMode 策略**：

| Backend | MatchMode | Pool `backend_id` | 事件如何关联 Worker |
|---------|-----------|-------------------|-------------------|
| Mooncake | `IpOnly` | 节点 IP（如 `10.0.0.1`） | `hbm_ip_index[IP]` → 该节点所有 DP |
| Memcache | `IpOnly` | 节点 IP | 同 Mooncake |
| YuanRong | `None` | 端口号（如 `15558`） | ZMQ 订阅端口 → 唯一 DP |

注意：`MatchMode::None`（YuanRong）下，事件内的 `backend_id` 会被**忽略**，改用订阅者
注册时的 backend_id（即引擎 `instance_id`）——否则 pool daemon 的 IP:port 会产生与 HBM
块不同的实例标识，破坏跨介质聚合。枚举另有 `IpAndDpRank`（按 IP + DP 精确匹配），
当前后端未选用，保留作扩展。

**注册流程差异**：

```text
  Mooncake/Memcache registration:
    HBM:  medium_endpoints={"npu": "tcp://IP:50090"}  -> hbm_ip_index
    Pool: endpoint="tcp://master:5557"               -> one global ZMQ SUB

  YuanRong registration:
    HBM:  medium_endpoints={"npu": "tcp://IP:15557"}
    CPU:  medium_endpoints={"cpu": "tcp://IP:15558"}  -> per-endpoint ZMQ SUB
    Disk: medium_endpoints={"disk": "tcp://IP:15558"} -> dedupe if shared port
```

### 注册、重注册与注销生命周期

- 首次注册会创建 `(model_name, tenant_id)` 对应的 `IndexerEntry`，并按去重后的 endpoint
  创建 ZMQ subscriber；HTTP-only 注册允许 endpoint 为空。
- 同一 `(instance_id, dp_rank)` 重注册时，旧 subscriber 会先停止。后端未变化时保留已有
  索引，仅更新 endpoint；后端变化时清除该实例/DP 的 NPU/CPU/Disk 索引和旧 HBM IP 映射。
- `replay_endpoint` 只在该 `instance_id` 首次出现时通过 `spawn_blocking` 执行历史事件回放。
- 注销会停止该实例/DP 的全部 subscriber，删除 HBM IP 映射及三层索引；删除最后一个 DP
  时，使用**注册记录中的** model/tenant 回收空 `IndexerEntry`，不信任注销请求里的同名字段。
- `POST /events` 的 `shutdown=true` 当前只记录日志，完整释放仍需显式调用 `/unregister`。

### 节点拓扑：`dp → node` 与 `pod_ip → node`

`/register` 的可选字段 `node_id` 表示该 endpoint 所在的**机器**（K8s `status.hostIP`，
即 Pod 的宿主 Node）。注意与 `medium_endpoints` 里的 **Pod IP** 分属两个层级：

```text
  Node（一台机器，status.hostIP）
   ├── Pod（status.podIP）── DP 0, DP 1 ...
   └── Pod（另一个 podIP）── DP 0 ...
```

带上 `node_id` 后，`NodeTopology` 维护两张表（同一把锁，避免解析到半更新状态）。
**两张表的方向都是「指向 node」**：

| 表 | 键 → 值 | 用途 |
|----|---------|------|
| `dp_to_node` | `(instance_id, dp_rank)` → `{pod_ip, node_id}` | 给一个 DP（边的 owner），问它在哪台机器 |
| `pod_to_node` | Pod IP → node 标识 | 给一个 Pod IP（池事件的 `backend_id`），问它在哪台机器 |

**为什么是这个方向**：消费者要回答的问题是「这个池块的 owner 和我是不是同一台机器」，
用 `dp → node` 是两次 O(1) 查表加一次比较；反向的 `node → dps` 每次都要扫列表，而且
没有对应的消费场景（原本设想的「扇出到同机所有 DP」在走查改为忽略 owner 后已无效果——
走查不再读 owner，扇出只会让每条边的 owner 集合变大而不改变任何上报数字）。

`same_node(dp_a, dp_b)` 封装了这个判断。**任一方位置未知时返回 `false`** —— 位置不明
绝不能当作同机，否则远端块会被算成本地、低估搬运成本。

**生命周期**：写入时机完全跟随注册——`/register` 写、`/unregister` 删、后端类型变化的
重注册先删再写；**没有轮询也没有刷新**。因为 Pod 与机器的绑定是静态的（Pod 不迁移宿主机，
重调度产生的是新 Pod、新 IP），所以每个 Pod 只在注册时写一次。DP 条目在注销时立即删除；
`pod_to_node` 因为可能被同 Pod 的其他 DP 共用，只在该 Pod 最后一个 DP 离开后才删（DP 表
的条目里带着自己的 `pod_ip`，扫一遍即可判断，注销是低频操作，比维护单独的计数更简单）。

**只记录 HBM endpoint 的 IP**：`cpu` / `disk` endpoint 可能指向别处的池服务，其 IP 不是
本引擎的 Pod IP。这也保证 `pod_to_node` 与 `hbm_ip_index` 的键来自同一处
（`extract_ip_from_endpoint(medium_endpoints["npu"])`），两张索引对得上。

**当前状态**：Coordinator **尚未下发** `node_id`，因此两张表默认为空、行为与之前完全一致；
表也**尚未接入任何匹配或打分逻辑**，仅通过 `GET /workers` 的 `topology` 字段暴露，供外部
客户端和排查使用。

要让 Coordinator 填上这个字段，**不需要改 K8s 部署**：`engine_template.yaml` 已经通过
downward API 把 `status.hostIP` 注入成 `HOST_IP` 环境变量（与 `POD_IP` 并排），只是 Python
运行时从不读它（目前只有 HCCL 启动脚本在用）。缺的是数据透传，路径与 `pod_ip` 完全相同：

```text
  status.podIP  -> POD_IP  -> Env.pod_ip -> RegisterMsg.pod_ip
                -> Endpoint.ip -> InsEventMsg -> ConductorApiClient    已通
  status.hostIP -> HOST_IP -> ??? -> ??? -> ???                        缺中间四环
```

中间四环（`Env.host_ip`、`RegisterMsg`、`Endpoint`/`NodeManagerInfo`、
`instance_assembler` 组装，最后 `ConductorApiClient` 填字段）都只是加字段加赋值。

另一条路是复用 Controller 容错模块已有的 `K8sClient.get_node_hostname_by_pod_ip()`——它
已经在遍历 `instance.get_node_managers()` 建 `pod_ip → node_name` 映射（RBAC 里 `pods` /
`nodes` 的 list 权限也齐），只是结果留在 `NodeMetadata` 里，不进 `Instance`、不传 Coordinator。
这条能省掉 NodeManager 那两环，但有三个前提：`K8sClient` 现在挂在 `FaultManager` 上，
`enable_fault_tolerance=False` 时整个能力不存在，得先提级；解析只在 `INSTANCE_INITIAL`
触发，可能晚于首次 conductor 注册（靠定时重注册补，`NodeTopology::record` 是幂等覆盖的）；
标识格式变成 K8s node name 而非 hostIP，**必须全局统一**，conductor 只做字符串相等比较，
两种格式混用会把同一台机器认成两台。

注意：`NodeManagerInfo` 本身**没有**机器字段——它的 `pod_ip` 与 `Endpoint.ip` 来自同一个
`msg.pod_ip`（`instance_assembler` 里 `add_node_mgr` 与 `add_endpoints` 相邻两行），所以
`node_managers` 只是一份 per-Pod 列表，机器信息得靠外部查询才能得到。另外部分测试已经在
构造 `NodeManagerInfo(..., host_ip=...)` 和引用 `api_config.host_ip`，但生产模型里没有这些
字段，Pydantic 静默忽略了它们——动手前先确认这些测试的意图。

### 池命中的本地/远端拆分

池块任意节点可取，但**搬运代价不同**：本机 DRAM 几乎免费，跨机要走
`device_rdma` / `device_sdma` / `device_urma`。原先所有池命中都统一记作 `cpu_blocks`，
调度器看不出这个差异。

判断依据是**边的 owner**：一条池事件会广播给上报它的那个 Pod 里的所有 DP，所以「owner 里
有当前 DP」等价于「这块在当前 DP 自己的 Pod 里」，而同 Pod 必然同机 —— 于是这就是一次
免搬运的本地读。查询侧不需要任何拓扑信息：

```text
  走查覆盖 [start, end)，逐块问 owner 里有没有自己
    有   -> cpu_local_blocks  += 1     本 Pod DRAM
    没有 -> cpu_remote_blocks += 1     需要传输

  只统计 exclusive_from 之后的位置（CPU 层取 npu_end，Disk 层取 max(npu_end, cpu_end)）：
  之前的块已被更高优先级介质本地覆盖，不需要搬运
  => cpu_local_blocks + cpu_remote_blocks == cpu_blocks（不变式，有测试守护）
```

实现上，走查对所有 DP 相同（忽略 owner），所以只走一次并顺手收下经过的块标识
（`reachable_chain`）；每个 DP 再拿这串标识去查已有的 per-worker 反查表
（`count_owned`，这张表本来是为 O(1) 删除建的，这里白嫖），而不是每个 DP 各走一遍。

**这个拆分是刻意保守的**：同机不同 Pod 的块其实也是便宜的本地读，但 conductor 没有机器
标识就看不出来，只能记作远端。所以 `cpu_local_blocks` 是共置的**下界** —— 只会低估，不会
高估。方向很重要：高估会告诉调度器「这次搬运免费」，而实际要跨机，直接打在 TTFT 上。

要把「同 Pod」升级成「同机」，需要让广播覆盖整台机器上的所有 DP（跨 Pod），这依赖上面
`node_id` 那条尚未打通的链路；而且**在一机一 Pod 的部署下没有任何收益**（那时 Pod 级广播
已经等价于机器级）。是否值得做，取决于 `*_pod_npu_num` 与单机卡数的比值，以及本地与远端
池命中的实测 TTFT 差距。

---

## 哈希设计

### 两种哈希的职责分离

```text
                Engine computes                 Conductor computes
              ┌──────────────────┐             ┌──────────────────┐
              │ SequenceBlockHash│             │ LocalBlockHash   │
              │ (chained, parent │             │ (independent     │
              │ hash as seed)    │             │ XXH3, seed 1337) │
              ├──────────────────┤             ├──────────────────┤
              │ algo: engine def │             │ algo: XXH3       │
              │ seed: dynamic    │             │ seed: 1337       │
              │ deps: parent hash│             │ deps: tokens only│
              │ use:  reverse    │             │ use:  tree key   │
              │      lookup      │             │                  │
              └─────────┬────────┘             └─────────┬────────┘
                        │                                │
                        └───────────────┬────────────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │ KvCacheStored     │
                              │ BlockData         │
                              │ {block_hash,      │
                              │  tokens_hash}     │
                              └───────────────────┘
```

**为什么独立计算 XXH3？**

- 引擎的 `BlockHash` 是链式滚动哈希（`H_i = hash(H_{i-1}, tokens_i)`），编码了整个前缀
- Conductor 的 `LocalBlockHash` 是独立 XXH3（`H_i = XXH3(1337, tokens_i)`），仅编码本块内容
- 独立哈希使同一父节点下、相同 token 内容稳定映射到同一 child key，多个 Worker 可共享该节点
- 不同父路径仍对应不同树节点；若直接使用引擎链式哈希，节点 key 会额外绑定整个历史前缀，
  不适合作为 Conductor 的本地内容键

### 计算细节

```text
  fn compute_block_hash_for_seq(token_ids: &[i64], block_size: u32)
      -> Vec<LocalBlockHash>

  Input:  token_ids = [t0, t1, ..., tn]
          block_size = B

  Output: [XXH3(1337, [t0 .. t_B-1]),
           XXH3(1337, [t_B .. t_2B-1]),
           ...]

  Implementation:
    - Cast each block's tokens i64 -> u32 (engine wire width), then XXH3 over LE bytes
    - little-endian: from_raw_parts(&[u32]) as &[u8] (not whole i64 zero-copy)
    - big-endian: write to_le_bytes per token
    - Trailing partial block still counts (tokens.len().div_ceil(block_size))
    - >2048 blocks: rayon parallel, batches of 1024
```

---

## 匹配逻辑与查询流程

### find_matches_by_hash 完整流程

```text
  Input: [LocalBlockHash; N]   (query sequence)

  ┌─ Phase 1: HBM Prefix Match ──────────────────────────────┐
  │  tree.find_matches_detailed(hashes)                      │
  │                                                          │
  │  root -> H0 -> H1 -> H2 -> (H3 missing)                  │
  │                                                          │
  │  Result: { W1: depth 3, W2: depth 1 }                    │
  │  -> overlap.npu_blocks[worker] = depth                   │
  │  -> collect TierBreakpoint {instance, dp_rank,           │
  │       end_pos: depth}                                    │
  └──────────────────────────────────────────────────────────┘
                           │
                           ▼
  ┌─ Phase 2: CPU Continuation ──────────────────────────────┐
  │  chain = compute_prefix_chain_for_seq(hashes)            │
  │  lower_tier_lookup(chain, hbm_breaks, cpu_tiers)         │
  │                                                          │
  │  Candidates per DP:                                      │
  │    a) resume at its own end_pos (only when < N)          │
  │    b) walk from position 0: always, shared by every      │
  │       DP (longer pooled replicas are never masked by     │
  │       shorter upstream hits)                             │
  │                                                          │
  │  reachable_span: walk positions until the first with     │
  │    no pooled replica; keep the farthest absolute end     │
  │  -> overlap.cpu_blocks[worker] = winning length          │
  │    (a position-0 win = full span, may overlap NPU;       │
  │     not "tail continuation only")                        │
  └──────────────────────────────────────────────────────────┘
                           │
                           ▼
  ┌─ Phase 3: Disk Continuation ─────────────────────────────┐
  │  disk_breaks = merge_tier_breakpoints(hbm_breaks,        │
  │                                      cpu_breaks)         │
  │  # per (instance, dp_rank): keep farther end_pos         │
  │  # -> resume from max(HBM, CPU)                          │
  │                                                          │
  │  lower_tier_lookup(chain, disk_breaks, disk_tiers)       │
  │  -> overlap.disk_blocks[worker] = winning length         │
  └──────────────────────────────────────────────────────────┘
                           │
                           ▼
  ┌─ Phase 4: Aggregate ─────────────────────────────────────┐
  │  build_response:                                         │
  │    per (instance, dp_rank):                              │
  │      collect npu_end / cpu_end / disk_end                │
  │      exclusive *_blocks (NPU > CPU > Disk)               │
  │      matched_tokens = (npu + cpu + disk) × block_size    │
  │    longest_matched = max(matched_tokens) across          │
  │      DP ranks                                            │
  │  (tier weights applied later by Coordinator affinity)    │
  └──────────────────────────────────────────────────────────┘
```

### 为什么用 "断点续查" 而不是每层独立从 root 走？

| | HBM RadixTree | CPU/Disk 池化索引 |
|---|---|---|
| 匹配方式 | root 出发，树遍历 | 位置查表：位置 0 起 + 自己断点起，并列候选 |
| 缺失处理 | 第一个缺失即停 | 第一个**无池化副本的位置**即停（不因副本属于别人而停） |
| 是否校验 owner | ✅ 逐层求交（HBM 跨节点取不到） | ❌ 忽略（池块任意节点可取） |
| 位置 0 走查条件 | 总是 | **无条件**，且对所有 DP 相同（只算一次） |
| 断点作用域 | 不适用 | **仅本 `(instance_id, dp_rank)`**，不跨 DP 借用 |
| 报的是什么 | 该 worker 本地持有的前缀 | 该 DP **能免重算服务**的前缀 |
| 保证 | 匹配的块形成合法前缀链 | 同上：`PrefixChainHash` 本身编码整条前缀 |

两层都保证三层命中块是**同一条连续前缀**，与 vLLM prefix cache 的查找语义
（NPU → CPU → Disk 依次续接）一致。位置 0 起的走查无条件并行进行，是为了发现下层更长副本
或仅下层命中；响应侧再按绝对终点做互斥切分，同前缀副本不会重复计入 `*_blocks` /
加权 `matched_tokens`——这正是弃用旧 `skip_root` 规则的原因（旧规则只会制造低估）。

### 为什么是连续匹配而不是平铺索引？

如果不做连续匹配而只用平铺 `tokens_hash → workers`（更早实现），会出现"block 0, 1, 3, 4
命中，block 2 缺失"的虚高计数，而且同一段内容出现在不同前缀下时会互相冒认。
`PrefixChainHash` 把整条前缀折进键里，位置查表因此天然只认同一条连续前缀。

---

## 事件接入

### 双协议支持

```text
  ┌──────────────────────────────────────────────────────┐
  │                  KvEventWirePayload                  │
  │                     .normalize()                     │
  └─────────────┬───────────────────────────┬────────────┘
                │                           │
                ▼                           ▼
      ┌───────────────────┐    ┌────────────────────────┐
      │ Engine format     │    │ RFC #1527 format       │
      │ (vLLM msgspec)    │    │ (Mooncake pool)        │
      │                   │    │                        │
      │ {type: "stored",  │    │ {event_type: "stored", │
      │  blocks: [...],   │    │  seq_hashes: [...],    │
      │  parent_hash,     │    │  medium: "cpu",        │
      │  token_ids,       │    │  backend_id: "..."}    │
      │  block_size}      │    │                        │
      └─────────┬─────────┘    └────────────┬───────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
                   ┌─────────────────────┐
                   │   KvCacheEventData  │
                   │ (canonical internal)│
                   │                     │
                   │  Stored / Removed / │
                   │       Cleared       │
                   └─────────────────────┘
```

### 两阶段 Offload 匹配

CPU/Disk 事件分两阶段到达：引擎先上报 offload 事件（含 `token_ids`，可算 `tokens_hash`），
pool daemon 后上报 store 事件（含 `seq_hashes`，即 `block_hash`）。核心数据结构：

```rust
pub struct OffloadPoolState {
    // Phase 1 先到: block_hash → (tokens_hash, parent_hash, inserted_at)
    // 等待首次 pool 确认；默认 TTL 600s，无容量上限
    offload: FxHashMap<u64, OffloadCacheEntry>,
    // 首次确认后始终写入（无需配置）：供后续 Disk 复用
    // TTL 300s（CONTENT_TTL）清扫；跨 tier 移除存活，为 CPU→Disk 迁移窗口兜底
    content: FxHashMap<u64, ContentEntry>,   // BlockContent + content inserted_at
    // Phase 2 先到: block_hash → {waiting workers}，TTL 60s（PENDING_TTL）
    pending_pool: FxHashMap<u64, FxHashSet<PendingPoolEvent>>,
}
// 不变量: 一个 block_hash 不同时存在于 offload 与 content
```

- **Phase 1 先到**：引擎 offload 事件到达 → 计算 `tokens_hash` → 连同该块自己的
  `parent_hash` 缓存到 `offload` → 等待 Pool 确认。
- **Phase 2 先到**：Pool store 事件到达 → 排入 `pending_pool`（按 Worker 去重）→ 等待
  引擎 offload 事件。
- **双方到齐**：使用 block hash 关联两侧事件，取得 `tokens_hash` + `parent_hash` → 把
  `parent_hash` 解析成内容位置、算出该块的 `PrefixChainHash` → 插入 CPU/Disk 索引，
  同时将映射迁入 `content`。

**content 保留（无需配置，始终生效）**：

- 首次 pool 确认后，`(tokens_hash, parent_hash)` 始终迁入 `content`，供 Disk 晋升复用
  （block 仍在 CPU tier 上时也可直接 `cpu_tiers` 反查）。
- content **跨 tier 移除存活**（不随 CPU 驱逐清除）：即便 CPU 先驱逐、Disk store
  事件后到，仍可在 `content` 保留窗口（`CONTENT_TTL`，300s）内解析入 Disk。
- `content` 按 `CONTENT_TTL`（300s）TTL 清扫：迁移窗口关闭后自动清除，内存有界；
  条目是 tier 数据的拷贝加短暂迁移残留，无 Disk 部署的额外开销可忽略。

**为什么 offload 缓存要携带 parent_hash？**

Phase 1 的 offload 事件是一条链（引擎一次 offload 多个连续块）。首个块的 parent 是
事件携带的 `parent_block_hash`，其后每块的 parent 是链中前一个块的 `block_hash`。
携带 parent_hash 后，Phase 2 的确认事件才能算出每个块**在序列中的真实位置**，而不是统一
当成 0 号位——位置错了就等于把块记在了别的前缀上。

解析顺序：HBM 节点上记录的 `prefix_chain` → 池化反查表 → `offload` / `content` 缓存链
（沿 `parent_hash` 回溯，因此多段 offload 链在任何一段被 pool 确认之前也能定位）。三者都
解析不出来时该事件的块被丢弃并计入 `unanchored_pooled_blocks`。

**匹配后的应用**：命中条目**逐 block** 构造 `Stored` 事件（每个 block 自带
`parent_hash`）应用，绝不批量合并成单个 `parent_hash: None` 的事件——批量会静默丢失
块间续接。

**缓存清理语义**：ingest 路径不再触发惰性全量扫描。后台 maintenance 默认每 30 秒
调用 `sweep_stale_caches()`，分别清理 `pending_pool`（默认 60s）、`content`（默认 300s）
和 `offload`（默认 600s）。三类缓存均记录插入时间；实际最长驻留时间约为对应 TTL 加一个
maintenance 周期。`offload` 没有容量上限，但不会永久驻留，同时仍会在匹配成功或引擎侧
`evict_pending_blocks` 时提前删除。Worker 注销只按 Worker 清理 `pending_pool`；共享的
`offload` / `content` 由匹配、显式移除或 TTL 维护回收。

### 后台 Maintenance

服务启动时创建独立 Tokio 周期任务，默认每 30 秒执行一次完整维护：

1. 对每个 `IndexerEntry` 清理过期的 offload/pending/content；
2. 调用 `sweep_stale_nodes()` 回收 HBM 空节点；
3. 扫描顶层 `DashMap`，删除缓存已空且没有活跃注册引用的 `(model, tenant)` Entry。

维护任务通过 `Weak<WorkerRegistry>` 持有服务，Registry 释放后任务自行退出。扫描空 Entry 时
持有注册表读锁，避免注册过程与“活跃 Entry 保护集合”之间出现竞态。匹配缓存的 maintenance
与 ingest 共用 `RwLock<OffloadPoolState>` 写锁，因此不会并发修改 Map；代价是清扫期间 ingest
可能短暂等待。

| CLI 参数 | 默认值 | 说明 |
|----------|--------|------|
| `--maintenance-interval-secs` | `30` | 后台维护周期；实现至少按 1 秒执行 |
| `--pending-ttl-secs` | `60` | Pool-first 等待项 TTL |
| `--content-ttl-secs` | `300` | CPU→Disk promotion 映射保留 TTL |
| `--offload-ttl-secs` | `600` | 未确认 engine offload TTL；无容量上限 |

### vLLM 事件过滤

vLLM msgspec 事件按 attention group 过滤（`is_main_attention_kind`，deny-list）：

- **保留**：`FullAttention` / `MlaAttention` / `SinkFullAttention`（无 `kv_cache_spec_kind`
  的旧版本事件也保留，向后兼容）
- **丢弃**：`SlidingWindow` / `SlidingWindowMla` / `Mamba` / `ChunkedLocalAttention` /
  `EncoderOnlyAttention` / `CrossAttention`（与 Dynamo kv-router 一致）
- **未知 kind**：前向兼容，默认**保留**（`_ => true`），避免未来引擎新增主注意力类型被静默丢弃

匹配忽略大小写与下划线（`MlaAttention` 与 `mla_attention` 等价）。此外，`BlockStored`
事件携带的 `block_size` 与注册值不一致时丢弃，防止污染索引。

### ZMQ Wire Format

事件通过 ZMQ PUB 以 3 段消息送达：`[topic][seq: u64 BE][msgpack payload]`。

ZMQ 路径（`zmq_subscriber::process_payload`）依次尝试 **2** 种 payload：

1. **vLLM msgspec batch**（`parse_vllm_batch`）：
   - Format A：`[ts, events, dp_rank]`
   - Format B：`[ts, dp_rank, events]`
   - 单条事件为 array_like：`[tag, block_hashes, parent_hash?, token_ids, block_size, medium, ...]`
2. **Pool backend batch**：`(timestamp_ms, [PoolEvent...], dp_rank)`（Mooncake/Memcache 等）

二者均失败则记 parse error。事件中缺失的 `model_name` / `block_size` / `dp_rank` /
`medium` 使用注册时的默认值补齐。

### HTTP `/events` Wire Format

Coordinator / 引擎也可经 `POST /events` 推送 JSON（`KvEventBatch` / `KvEventWirePayload`）：
支持引擎 map 形态、RFC #1527 pool 形态，以及 `type` / `event_type` / legacy `BlockStored`
等字段别名（见 `protocols.rs`）。该路径与 ZMQ msgpack 解析相互独立，勿与上节混淆。

---

## 查询接口编码协商（JSON / MessagePack）

`/query` 与 `/query_by_hash` 通过请求 `Content-Type` 协商传输编码：

| Content-Type | 请求 | 响应 |
|---|---|---|
| `application/msgpack` / `application/x-msgpack` | `rmp_serde` 直解为 `QueryRequest` / `QueryByHashRequest` | `rmp::encode` 手工编码，错误/空结果同样按 msgpack 返回 |
| 其他（默认） | JSON（历史行为，不变） | JSON |

**为什么响应侧手工编码**：`QueryResponse` 使用 `#[serde(flatten)]`（`tenants` 展开到顶层
map），msgpack 序列化器不支持 flatten——`encode_query_response_msgpack`（`protocols.rs`）
手工编码嵌套 map，保证 msgpack wire 形状与 JSON 逐字节等价（有单元测试
rmpv→serde_json 结构化对比守护）。

Coordinator 侧通过 `kv_conductor_config.query_encoding`（默认 `"msgpack"`，合法值
`msgpack` / `json`）选择请求编码；响应按服务器 `Content-Type` 解析（msgpack → msgspec，
否则 JSON），旧版 JSON-only conductor 自动兼容。**滚动升级注意**：请求侧无自动降级——
须先升级 kv-conductor 再升级 Coordinator；混部（新版 Coordinator + 旧版 conductor）时
须显式配置 `query_encoding: "json"`。

---

## 错误处理

| 错误 | HTTP 状态码 | 场景 |
|------|-----------|------|
| `InstanceNotFound` | 404 | 对未注册实例执行操作 |
| `NoIndexer` | 404 | 查询时不存在对应的 (model, tenant) IndexerEntry |
| `NoWorkers` | 200 `{tenant_id: {}}` | 无缓存命中——正常，不视为错误 |
| `ParentBlockNotFound` | 500 | Store 事件引用了未知 parent hash |
| `InvalidBlockSequence` | 500 | 检测到自引用 block |

---

## 相关文件索引

| 文件 | 说明 |
|------|------|
| `motor/kv_conductor/src/indexer/mod.rs` | 顶层索引、两阶段缓存和 maintenance |
| `motor/kv_conductor/src/concurrent_tree.rs` | HBM 并发 Radix Tree |
| `motor/kv_conductor/src/lower_tier.rs` | CPU/Disk 池化块索引（按内容前缀哈希落位） |
| `motor/kv_conductor/src/registry.rs` | Worker 注册生命周期与 subscriber 管理 |
| `motor/kv_conductor/src/main.rs` | CLI、后台 maintenance 和 HTTP 服务启动 |
| `motor/kv_conductor/__init__.py` | Python 包入口，`is_available()`, `start()` |
| `motor/kv_conductor/__main__.py` | `python -m motor.kv_conductor` 入口 |
| `build.sh` | 条件编译，`KV_CONDUCTOR_PREBUILT` 支持预构建二进制 |
| `setup.py` | 条件 `package_data`，按需打包二进制到 wheel |
| `docs/zh/user_guide/features/kvcache_affinity.md` | 用户部署文档 |
| `motor/kv_conductor/README.md` | 功能简介 |
