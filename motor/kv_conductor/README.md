# KV Conductor

基于 Rust 的 KV Cache 索引服务。订阅引擎 KV 事件，维护前缀树索引，为 Coordinator 提供
缓存感知的请求路由——将请求导向已缓存最长 token 前缀的 Worker。已集成在 motor Python 包内。

## 快速开始

kv-conductor 以**可选组件**的形式随 motor wheel 发布。完整的构建和启动流程：

### 1. 编译二进制

```bash
cd motor/kv_conductor && cargo build --release
# 二进制产出：target/release/kv-conductor
```

仓库提交了 `Cargo.lock`；本地/CI 建议使用 `--locked` 以固定依赖版本。流水线加速请缓存：

- `~/.cargo/registry`、`~/.cargo/git`
- `motor/kv_conductor/target`

缓存 key 建议：`{os}-{rustc版本}-{Cargo.lock hash}`。

如果已有预编译的二进制，可跳过此步，后续 `build.sh` 会自动发现并打包。

### 2. 构建 motor wheel

```bash
# 在项目根目录执行
bash build.sh
```

`build.sh` 会自动检测 kv-conductor 二进制：

- `target/release/kv-conductor` 已存在 → 直接复制到 `bin/`，打包进 wheel
- 不存在但有 `cargo` → 自动编译
- 设置了 `KV_CONDUCTOR_PREBUILT=/path/to/binary` → 使用指定的预构建二进制
- 都没有 → 跳过，wheel 不含 kv-conductor（其他功能不受影响）

产物：`dist/motor-*.whl`

### 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `RUST_LOG` | `info` | 日志级别 |

### 3. 安装 wheel

```bash
pip install dist/motor-*.whl
```

安装后 `python -m motor.kv_conductor` 即可使用。验证：

```bash
python -c "from motor.kv_conductor import is_available; print(is_available())"
# True → kv-conductor 可用
```

### 4. 启动

```bash
python -m motor.kv_conductor --port 13333

# 或直接运行二进制
./motor/kv_conductor/target/release/kv-conductor --port 13333
```

缓存维护参数（均有默认值）：

```bash
kv-conductor \
  --maintenance-interval-secs 30 \
  --pending-ttl-secs 60 \
  --content-ttl-secs 300 \
  --offload-ttl-secs 600
```

后台维护会定期清理过期的 offload/pending/content 匹配缓存、HBM 空节点和无注册引用的空索引。
TTL 清理由后台周期任务完成，ingest 路径不再执行惰性全量扫描；因此即使事件流量停止，
过期数据也会被回收。实际最长驻留时间约为对应 TTL 加一个 maintenance 周期。

> **注意**：容器部署时，镜像内二进制路径为 `/usr/local/bin/kv-conductor`，
> 启动脚本 `kv_conductor.sh` 通过 `exec python -m motor.kv_conductor` 启动。

## 功能

KV Conductor 维护三层存储介质的 KV Cache 索引，按绝对覆盖终点做互斥切分后
返回各介质 `*_blocks` 与未加权覆盖长度 `matched_tokens`；介质亲和权重由
Coordinator 调度器（`kv_affinity_w_*`）在计分时应用。

### HBM（NPU）— 统一模型

引擎 Worker 通过 ZMQ PUB 或 HTTP 将 KV 事件发给 conductor。
ZMQ 模式下事件端点由引擎 Worker 绑定，conductor 作为 SUB **主动 connect 到引擎**
（连接方向 conductor → 引擎，事件数据流引擎 → conductor）；HTTP 模式下引擎 Worker
将 KV 事件 POST 到 conductor 的 `/events` 接口。
所有后端（Mooncake / Memcache / YuanRong）的 HBM 事件链路一致：

```text
Engine Worker                        KV Conductor
(vLLM/SGLang)
      │                                │
      │  ZMQ PUB（引擎绑定，conductor   │
      │  SUB 主动 connect）/ HTTP POST │
      │  {type: "stored",              │
      │   token_ids, block_hashes,     │
      │   parent_hash, medium: "npu"}  │
      │───────────────────────────────>│
      │                                ├─ XXH3(token_ids) -> LocalBlockHash
      │                                ├─ RadixTree.apply_store()
      │                                │    └─ build prefix chain via parent_hash
      │                                └─ query: tree walk -> longest contiguous prefix
```

- **索引结构**：`ConcurrentRadixTree`，按 token 内容哈希（XXH3）建前缀链
- **匹配语义**：最长连续前缀——从 root 走到第一个缺失即停
- **事件源**：Worker 自行上报，无需中心化 Pool
- **介质 key**：注册与事件中使用 `"npu"`（兼容旧值 `"gpu"` / `"xpu"`）

### CPU / DISK — 可选，后端相关

当启用 KV Cache 池化（Mooncake / Memcache / YuanRong）且配置了 CPU/DISK 副本时，
conductor 通过**两阶段匹配**索引二级缓存：

```text
Engine Worker           Pool Master               KV Conductor
      │                      │                      │
      │  [Phase 1]           │                      │
      │  offload event       │                      │
      │  {token_ids,         │                      │
      │   block_hashes,      │                      │
      │   parent_hash}       │                      │
      │──────────────────────┼─────────────────────>│
      │                      │                      │  cache hash(token_ids)
      │                      │                      │  (wait for pool confirm)
      │                      │                      │
      │                      │  [Phase 2]           │
      │                      │  pool store event    │
      │                      │  {seq_hashes,        │
      │                      │   medium: "cpu"}     │
      │                      │─────────────────────>│
      │                      │                      │  match -> insert CPU index
      │                      │                      │  keep content (TTL 300s)
      │                      │                      │
      │                      │  [Disk promote]      │
      │                      │  (optional)          │
      │                      │  {seq_hashes,        │
      │                      │   medium: "disk"}    │
      │                      │─────────────────────>│
      │                      │                      │  lookup CPU tier (or kept
      │                      │                      │   content; survives cross-
      │                      │                      │   tier remove) -> Disk index
```

- **索引结构**：`LowerTierIndexer`，按 `(parent_seq_hash, tokens_hash)` 记录 continuation edge
- **走查忽略 owner**：池化块通过后端传输协议（`device_rdma` / `device_sdma` / `device_urma`，见
  `mmc-local-*.conf` 的 `ock.mmc.local_service.protocol`）**任意节点可取**，所以别的 DP 持有的块
  同样能让本 DP 跳过重算。走查只在**边不存在**时停止，不因为边属于别人而停
- **每个 DP 报的是"我能免重算地服务多长前缀"**，不是"我本地有多长"

那为什么各 DP 的结果还会不同?两处:

1. **起点是 per-DP 的**。每个 DP 从**自己的**上层断点续接;自己上层没命中就从 root 走。
   **HBM 是设备显存、跨节点取不到**,所以只有持有那些块的 DP 能用它们跨过池链中的缺口
2. **归属是 per-DP 的**。互斥切分把 `[0, npu_end)` 记为 NPU(本地、免费)、其余记为 CPU/Disk
   (需搬运),于是 `kv_affinity.w_cpu` / `w_disk` 就是"优先选本地已有的节点"这个旋钮

举例(池链在位置 2 处断开,位置 3-4 另起一段):

```text
  inst-a HBM 覆盖 [0,3)，inst-b DRAM 覆盖 [0,2) 与 [3,5)

  inst-a：从自己断点 pos=3 起跑 → 取到 inst-b 的 [3,5) → 终点 5
          npu_blocks=3（本地）+ cpu_blocks=2（搬运）= 5 块
  inst-b：无 HBM 命中 → 从 root 走 → [0,2) 之后位置 2 缺失 → 终点 2
          npu_blocks=0 + cpu_blocks=2 = 2 块
```

`inst-a` 靠自己的 HBM 桥接了缺口,所以走得更远——差异化正来自这里。

- **断点不跨 DP**：断点只供产生它的 `(instance_id, dp_rank)` 使用。上报的是绝对覆盖终点，隐含
  「`[0, 终点)` 对该 DP 都可用」；这只有在 `[0, 起点)` 由该 DP **自己的**上层介质覆盖时才成立
  （上层是 HBM，取不到别人的）。若允许借用其他 DP 的断点，只持有中间段的 worker 会跨过自己
  取不到的空洞谎报前缀
- **所有已注册 DP 都参与**：从注册时维护的 Pod IP → DPs 索引汇总全局 DP 集合，并按查询的
  model/tenant 过滤；本地尚未产生任何缓存事件的 DP 也能得到 root 走查结果，否则会高估它的
  prefill 代价
- **连续匹配**：走到第一个缺失边即停；候选（root + 自己的断点）取绝对终点最远者
- **content 保留**：pool 确认后始终保留 `(tokens_hash, parent_hash)`（无需配置），跨 tier 移除存活，CPU 已驱逐后、保留窗口（300s TTL）内仍可解析 Disk store；窗口关闭自动清除，内存有界（条目为 tier 数据拷贝 + 短暂迁移残留）。未确认的 offload **无 TTL、无硬容量上限**，随未确认块增长，仅在匹配成功或引擎驱逐时清除

各后端的 CPU/Disk 适配差异：

| 后端 | Pool 模型 | Worker 识别 |
|------|----------|-------------|
| Mooncake | 中心化 master，一个 ZMQ PUB | IP 匹配 → 节点上所有 DP |
| Memcache | 中心化 master，一个 ZMQ PUB | 同 Mooncake |
| YuanRong | 每节点多端口 ZMQ PUB | Port 匹配 → 精确 DP |

### 查询

Coordinator 发起查询，conductor 汇总三层介质的**连续匹配 block 数**：

```text
Coordinator                                  KV Conductor
      │                                        │
      │  POST /query                           │
      │  {model, block_size,                   │
      │   token_ids, tenant_id?}               │
      │───────────────────────────────────────>│
      │                                        │
      │  200 {                                 │
      │    "default": {                        │  <- tenant_id (default "default")
      │      "inst-1": {                       │
      │        "longest_matched": 640,         │  <- max matched_tokens across DPs
      │        "DP": {                         │
      │          "0": {                        │
      │            "matched_tokens": 640,      │  <- exclusive sum × block_size
      │            "npu_blocks": 3,            │  <- exclusive NPU blocks
      │            "cpu_blocks": 2,            │  <- exclusive CPU beyond NPU
      │            "disk_blocks": 0            │  <- exclusive Disk beyond max(NPU,CPU)
      │          }                             │
      │        }                               │
      │      }                                 │
      │    }                                   │
      │  }                                     │
      │<───────────────────────────────────────│
```

字段计算（每 DP / rank）：

1. 收集各介质绝对覆盖终点 `npu_end` / `cpu_end` / `disk_end`
2. 互斥切分（优先级 NPU > CPU > Disk）：
   - `npu_blocks = npu_end`
   - `cpu_blocks = max(0, cpu_end - npu_end)`
   - `disk_blocks = max(0, disk_end - max(npu_end, cpu_end))`
3. `matched_tokens = (npu + cpu + disk) × block_size`（未加权覆盖）
4. `longest_matched = max(各 DP matched_tokens)`

| 字段 | 含义 |
|------|------|
| `npu_blocks` / `cpu_blocks` / `disk_blocks` | 该 DP 互斥真实命中块数（同前缀副本只归最高优先级介质） |
| `matched_tokens` | 互斥块数之和 × `block_size`（真实覆盖长度） |
| `longest_matched` | 该实例所有 DP 的 `matched_tokens` 最大值 |
| `cpu_local_blocks` / `cpu_remote_blocks` | `cpu_blocks` 按搬运代价拆开：本 Pod DRAM（几乎免费）/ 需要传输 |

### 池命中的本地/远端拆分

池块任意节点可取，但**搬运代价不同**：本机 DRAM 几乎免费，跨机要走
`device_rdma` / `device_sdma` / `device_urma`。原先所有池命中都统一记作 `cpu_blocks`，
调度器看不出这个差异。

判断依据是**边的 owner**：一条池事件会广播给上报它的那个 Pod 里的所有 DP，所以「owner 里有
当前 DP」等价于「这块在当前 DP 自己的 Pod 里」，而同 Pod 必然同机 —— 于是这就是一次免搬运
的本地读。查询侧不需要任何拓扑信息：

```text
  走查覆盖 [start, end)，逐块问 owner 里有没有自己
    有   -> cpu_local_blocks  += 1     本 Pod DRAM
    没有 -> cpu_remote_blocks += 1     需要传输

  只统计 npu_end 之后的位置：之前的块已在本地 HBM，不需要搬运
  => cpu_local_blocks + cpu_remote_blocks == cpu_blocks（不变式，有测试守护）
```

**这个拆分刻意保守**：同机不同 Pod 的块其实也便宜，但 conductor 没有机器标识就看不出来，
只能记作远端。所以 `cpu_local_blocks` 是共置的**下界** —— 只会低估不会高估。高估的后果更
严重：会告诉调度器「这次搬运免费」，而实际要跨机。

要升级成真正的「同机」语义，需要广播覆盖整台机器上的所有 DP（跨 Pod），依赖 `node_id`
那条尚未打通的链路；**在一机一 Pod 的部署下没有收益**（那时 Pod 级广播已等价于机器级）。

调度器读取 `DP[<dp_rank>]` 的 `*_blocks`，按 `scheduler_config.kv_affinity`
中的 `w_npu/w_cpu/w_disk`（默认 `1.0/1.0/0.0`）加权后再算亲和分（见亲和性调度文档）。

## 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` / `-p` | `13333` | HTTP 服务端口 |
| `--host` | `::` | 绑定地址（默认双栈） |

## API

| 端点 | 方法 | 用途 |
|------|------|------|
| `/register` | POST | 注册 Worker（`medium_endpoints`: `npu` / `cpu` / `disk`；可选 `node_id`） |
| `/unregister` | POST | 注销 Worker |
| `/query` | POST | 按 token_ids 查询各 Worker 命中 block 数 |
| `/query_by_hash` | POST | 使用预计算 hash 查询 |
| `/events` | POST | 接入 KV 事件 |
| `/health` | GET | 存活检查 |
| `/workers` | GET | 已注册 Worker 列表 |

详细 API 契约见 [设计文档](../../docs/zh/design/kv_conductor.md)。

## Motor 集成

kv-conductor 已随 motor wheel 打包。部署脚本 `kv_conductor.sh` 通过以下命令启动：

```bash
exec python -m motor.kv_conductor --host "$KV_CONDUCTOR_HOST" --port "$KV_CONDUCTOR_PORT"
```

Coordinator 通过 `ConductorApiClient` 与 conductor 通信。`user_config.json` 典型配置：

```json
{
  "motor_coordinator_config": {
    "scheduler_config": {
      "scheduler_type": "kv_cache_affinity"
    }
  },
  "kv_conductor_config": {
    "block_size": 128,
    "npu_endpoint": "tcp://*:5557",
    "http_server_port": 13333
  }
}
```

### 节点拓扑（`node_id`,可选）

`/register` 接受一个可选的 `node_id`，表示该 endpoint 跑在**哪台机器**上（K8s 的
`status.hostIP`，或任何稳定的单机标识）。它与 `medium_endpoints` 里的 **Pod IP** 是两个
不同层级——一台机器可以跑多个 Pod，每个 Pod 一个独立 Pod IP。

带上之后 conductor 会维护两张表，**方向都是"指向 node"**，可通过 `GET /workers` 的
`topology` 字段查看：

```json
{
  "topology": {
    "dp_to_node": {
      "vllm-prefill-1/0": {"pod_ip": "10.244.0.5", "node_id": "node-1"},
      "vllm-prefill-1/1": {"pod_ip": "10.244.0.5", "node_id": "node-1"},
      "vllm-prefill-2/0": {"pod_ip": "10.244.0.6", "node_id": "node-1"},
      "vllm-prefill-3/0": {"pod_ip": "10.244.1.7", "node_id": "node-2"}
    },
    "pod_to_node": {
      "10.244.0.5": "node-1",
      "10.244.0.6": "node-1",
      "10.244.1.7": "node-2"
    }
  }
}
```

| 表 | 键 → 值 | 用途 |
|----|---------|------|
| `dp_to_node` | `(instance_id, dp_rank)` → `{pod_ip, node_id}` | 给一个 DP（边的 owner），问它在哪台机器 |
| `pod_to_node` | Pod IP → node | 给一个 Pod IP（池事件的 `backend_id`），问它在哪台机器 |

**方向是刻意这样的**：消费者要问的是「这个 owner 和我是不是同一台机器」，两次 O(1) 查表
再比一下即可。反向的 `node → dps` 每次都要扫列表，而且没有对应的消费场景。
`same_node(dp_a, dp_b)` 直接封装了这个判断；任一方位置未知时返回 `false`——位置不明
绝不能当作同机，否则远端块会被算成本地。

写入时机跟随注册生命周期：`/register` 写入、`/unregister` 移除、后端类型变化的重注册
先移除再写入。**没有轮询或刷新** —— Pod 与机器的绑定是静态的（Pod 不会迁移宿主机，
重调度得到的是新 Pod、新 IP），所以每个 Pod 只在其注册时写一次。移除是精确的：DP 条目
立即删除，`pod_to_node` 则在该 Pod 的最后一个 DP 离开后才删。

**不带 `node_id` 时两张表为空**，其余行为完全不变——目前 Coordinator 尚未下发该字段，所以这
两张表当前仅供外部客户端与调试使用，**未接入任何匹配或打分逻辑**。

要接上不需要改 K8s 部署——`HOST_IP`（`status.hostIP`）已经注入到 engine 容器，只是 Python
侧从不读；缺的是从 NodeManager 经 Controller 到 Coordinator 的字段透传，路径与已经跑通的
`pod_ip` 完全相同。另一条路是复用 Controller 容错模块已有的 K8s 查询。详见设计文档。

`npu_endpoint` 必须与引擎 `--kv-events-config` 的 `endpoint` 一致（`tcp://*:5557` 为 vLLM 常用值）；
模式中的 `*` 会被替换为 endpoint IP，端口会加上 `dp_rank`，conductor 主动 connect 到各引擎节点绑定的事件端口。
注册时写入 conductor 的 `medium_endpoints` key 为 `"npu"`。

详见 [KV Cache 亲和性调度文档](../../docs/zh/user_guide/features/kvcache_affinity.md)。

## 详细设计

架构细节、多介质适配、哈希与匹配算法见 [设计文档](../../docs/zh/design/kv_conductor.md)。

## 许可证与第三方声明

本组件主体采用 **Mulan PSL v2**。

以下文件（或部分）为 NVIDIA Dynamo kv-router 的 **Apache-2.0 衍生作品 / 策略对齐**，
已保留 NVIDIA 版权与 SPDX / 归因声明；分发时须同时提供 Apache-2.0 许可证文本：

| 本地文件 | 上游路径 |
|----------|----------|
| `src/lower_tier.rs` | `lib/kv-router/src/indexer/lower_tier.rs` |
| `src/concurrent_tree.rs` | `lib/kv-router/src/indexer/concurrent_radix_tree.rs` |
| `src/hashing.rs` | `lib/kv-router/src/protocols.rs`（XXH3 哈希） |
| `src/protocols.rs`（部分） | `lib/kv-router/src/protocols.rs` |
| `src/events/vllm.rs`（attention 过滤策略） | `lib/kv-router/src/zmq_wire/filter.rs` |

详见：

- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt)
