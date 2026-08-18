# [2026-08-18] vLLM 随机内部 ID 导致命中拆分无法关联

- **现象 (Symptom)**：EngineCore hook 能看到 local/remote hit，但 Motor `req_id` 关联日志只能回退到 usage 总量，split 显示 `-`。
- **根因 (Root cause)**：vLLM 0.23.0 将外部 request ID 保存到 `RequestState.external_req_id`，并给 `EngineCoreOutput.request_id` 追加 8 位随机后缀；hook 按内部 ID 缓存，adapter 按外部 `engine_request_id` 查询。
- **为什么会写出 (Why)**：测试用相同的 fake `request_id` 模拟 hook 和 adapter，没有复现 vLLM 的 internal/external ID 双层语义。
- **修复 (Fix)**：hook 通过 `OutputProcessor.request_states[internal_id].external_req_id` 解析外部 ID，再缓存 PrefillStats；无法解析时保留直接输出 ID 的兼容回退。
- **测试拦截 (Test interception)**：使用 `req#a1-deadbeef` 内部 ID 和 `req#a1` 外部 ID 验证 split 能关联到 Motor root ID，并验证 monkey-patch 把 OutputProcessor 实例传给采集函数。
- **场景 (Scenario)**：vLLM 0.23.0 默认启用 request ID randomization 的所有请求。
- **关键词 (Keywords)**：vllm-0.23.0, request-id-randomization, PrefillStats, cache-hit, correlation
