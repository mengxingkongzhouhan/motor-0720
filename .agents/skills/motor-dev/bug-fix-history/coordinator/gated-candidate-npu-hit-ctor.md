# [2026-09-16] 给 GatedCandidate 加必填字段后 CAS 重选 TypeError

- **现象 (Symptom)**：`Unified PD exception 0/5, because of GatedCandidate.__init__() missing 1 required positional argument: 'npu_hit', retry=True`。
- **根因 (Root cause)**：`GatedCandidate` 是 frozen dataclass。只在 `smetric_gated.py::score_endpoints` 补了 `npu_hit=`，但 `allocate_arbitration.select_smetric_gated`（以及测试里的 4 参数构造）仍按旧签名构造。第一次 CAS 走 CHANGED/BLOCKED 或 `select_valid_candidate` 失败后会走这条路径，于是 attempt 0 就 TypeError。
- **为什么会写出 (Why)**：把「策略打分处的构造」当成唯一构造点。CAS 重选会从 stamp 四元组重建 candidate，不会复用 `score_endpoints` 的对象。
- **修复 (Fix)**：`npu_hit: float = 0.0`；`score_endpoints` 用 `_npu_hit_ratio(matched, isl)`（`npu_blocks / isl`，isl<=0 为 0）。
- **测试拦截 (Test interception)**：`test_gated_candidate_npu_hit_defaults_to_zero`、`test_npu_hit_ratio_divides_by_isl_and_guards_zero`、`test_score_endpoints_reads_cost_and_cpu_hits_and_orders_by_ledger` 断言 ep 10 的 `npu_hit==0.01`。
- **场景 (Scenario)**：给 `GatedCandidate` 增加无默认值字段，却不改 `allocate_arbitration.py` 的构造。
- **关键词 (Keywords)**：GatedCandidate npu_hit, missing positional argument, allocate_arbitration, select_smetric_gated
