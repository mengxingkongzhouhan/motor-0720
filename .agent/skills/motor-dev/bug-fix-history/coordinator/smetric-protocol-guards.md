# [2026-08-18] SMetric 协议不兼容导致错误路由或空分配

- **现象 (Symptom)**：Conductor 返回 DpBlocks 字典时所有端点被当作零命中；缺失成本候选时返回空分配；拓扑替换后继续使用旧平均值；转储路径可能选中未评分端点并记录零成本；滚动升级时新旧 SHM 条目格式无法区分；客户端可能用本地缓存覆盖服务端已提交的成本。
- **根因 (Root cause)**：`smetric.py` 只解析整数匹配值；`scheduler_server.py` 缺少候选降级、评分集合边界和 SET 重置；`workload_shm/layout.py` 改写条目尾部字段却未升级 `SCHEMA_VERSION`；`scheduler_client.py` 未区分服务端权威字段与旧协议回退。
- **为什么会写出 (Why)**：实现只覆盖理想的同版本、完整候选、整数响应流程，遗漏外部响应演进、拓扑生命周期和进程滚动升级的协议边界。
- **修复 (Fix)**：兼容 `matched_tokens` 字典格式；缺失成本时校验 worker 候选；转储仅在已评分端点中选择；SET 变更重置平均值；SHM schema 升级到 4；客户端仅在服务端未返回成本字段时使用本地缓存回退。
- **测试拦截 (Test interception)**：SMetric 策略测试覆盖 DpBlocks、空 token 与负匹配；分配仲裁覆盖缺失成本和未评分端点；SchedulerServer 覆盖 SET 重置；SHM reader 覆盖 schema 4 与非零 prefill cost；SchedulerClient 覆盖服务端成本优先级。
- **场景 (Scenario)**：新 Conductor 响应格式、worker/scheduler 视图暂时不一致、实例集合全量替换，或 Scheduler 与 Worker 滚动升级。
- **关键词 (Keywords)**：smetric, DpBlocks, prefill_cost, shared memory schema, topology SET
