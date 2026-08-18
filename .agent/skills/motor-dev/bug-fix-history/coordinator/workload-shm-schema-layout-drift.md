# [2026-08-18] Workload SHM entry 变更未升级 schema

- **现象 (Symptom)**：`prefill_cost` 占用了 schema v3 entry 尾部原有的 4B padding，但 header 仍声明 v3，混合版本进程无法识别布局变化。
- **根因 (Root cause)**：`motor/coordinator/scheduler/runtime/workload_shm/layout.py` 修改 `ENTRY_FMT` 时遗漏同步递增 `SCHEMA_VERSION`。
- **为什么会写出 (Why)**：entry 总大小仍为 24B，误把“stride 不变”当成“二进制语义兼容”；实际上相同字节在 v3 是 padding，在 v4 是 float32。
- **修复 (Fix)**：将 schema 升级到 v4，并在组件参考中记录 v3→v4 的字段语义变化。
- **测试拦截 (Test interception)**：reader 测试显式构造 schema v3 header 并断言拒绝读取；writer 测试验证非零 `prefill_cost` 的二进制 roundtrip。
- **场景 (Scenario)**：Scheduler writer 和 Inference worker reader 版本不一致，或滚动升级期间共享同一 SHM segment。
- **关键词 (Keywords)**：workload-shm, schema-version, prefill-cost, binary-layout, rolling-upgrade
