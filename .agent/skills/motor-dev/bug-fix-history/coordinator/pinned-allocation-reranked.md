# [2026-08-18] 固定实例分配被全局策略重选或拒绝

- **现象 (Symptom)**：配置 `load_balance` 时，带 `target_instance_id` 的请求可能被 SchedulerServer 改派到其他实例；配置 `smetric` 时，同类请求因没有 Conductor 成本候选而返回空分配。
- **根因 (Root cause)**：`motor/coordinator/scheduler/runtime/scheduler_client.py` 的固定实例分支仍将全局 `scheduler_type` 写入 `candidate_policy`。服务端据此执行全局负载重排，或进入要求 SMetric 成本列表的分支。
- **为什么会写出 (Why)**：把 `candidate_policy` 同时当成“客户端如何选出候选”和“服务端是否可以重选”的标记，遗漏了固定实例约束优先于调度策略的协议语义。
- **修复 (Fix)**：固定实例分支统一发送 `round_robin` 候选策略，使服务端只校验并提交已固定的单一候选，不再全局重排。
- **测试拦截 (Test interception)**：`test_pinned_allocation_does_not_request_global_reranking` 覆盖 `smetric` 与 `load_balance`，断言固定实例请求发送单候选校验策略。
- **场景 (Scenario)**：请求包含内部调度 pin（如精度探测），且全局调度器配置为 `load_balance` 或 `smetric`。
- **关键词 (Keywords)**：coordinator, scheduling pin, target_instance_id, smetric, load_balance
