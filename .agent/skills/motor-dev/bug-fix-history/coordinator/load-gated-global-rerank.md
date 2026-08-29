# [2026-08-21] load_gated 候选被误当作 unified 全局重排

- **现象 (Symptom)**：`test_allocate_only_load_gated_prefill_cost_does_not_trigger_global_rank` 期望保留 Worker 的 load-gated 候选，实际被 Scheduler 改选到候选成本更低的另一实例。
- **根因 (Root cause)**：适配冲突把 `scheduler_server.py:_select_authoritative_allocate_candidate` 的 unified 判断简化为“存在 `affinity_candidates`”；load-gated 也会携带 `prefill_cost` 用于账本提交，因此误入全局重排。
- **为什么会写出 (Why)**：把候选中存在成本字段误认为 unified 模式标识，忽略了同一字段在 load-gated 模式下还有账本记录用途。
- **修复 (Fix)**：仅当请求同时携带 `prefill_load_scale` 时调用 `_select_affinity_global`；继续向该路径传递 `required_engine_type`。
- **测试拦截 (Test interception)**：`tests/coordinator/scheduler/test_scheduler_allocate_arbitration.py::test_allocate_only_load_gated_prefill_cost_does_not_trigger_global_rank` 验证携带 prefill cost 的 load-gated 请求不会越过 Worker 候选边界。
- **场景 (Scenario)**：`kv_cache_affinity.mode=load_gated`，Worker 候选携带 prefill cost，候选外或低优先级端点成本更低。
- **关键词 (Keywords)**：coordinator, kv_cache_affinity, load_gated, prefill_cost, global rerank
