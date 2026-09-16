# [2026-09-15] 切到 schema-5 分支后 Mgmt 因 ABI 2 .so 无法启动

- **现象 (Symptom)**：`cursor/shm-prefill-cpu-ledger-30c3` 启动 Management server 失败。`NativeWorkloadShmUnavailable: Could not load libmindie_workload_shm.so ... lib/...so: ABI 2 < 3; target/release/...so: ABI 2 < 3`。
- **根因 (Root cause)**：Python `MIN_ABI_VERSION = 3`（`native.py`）拒绝 schema-4 的 ABI 2 库。`.so` 在 `.gitignore`（`workload_shm_rs/lib/`），git checkout 只换 Python。`build.sh` 原先只要 `lib/*.so` 存在就跳过 cargo，把上一分支的 ABI 2 库原样留下。`load_native_library` 在 `lib/` 与 `target/release` 都是 ABI 2 时于 `WorkloadShm.create_v4` 失败，Mgmt lifespan 退出。
- **为什么会写出 (Why)**：把「目录里有 .so」当成「和当前 checkout 的 ABI 匹配」。ABI 升级后，默认 reuse 会让「只拉分支、不重编 native」的部署在启动期硬失败。
- **修复 (Fix)**：`probe_so_abi` / `so_abi_is_current`；`build.sh` 把 ABI 过旧的 `.so` 当缺失并重编；最终 gate 拒绝 ABI-stale 出包。ABI 失败异常带 `SKIP_WORKLOAD_SHM_BUILD=0` 提示。
- **测试拦截 (Test interception)**：`test_stale_abi_library_is_refused`（假 ABI 2 必须含 leftover / SKIP 提示）、`test_probe_so_abi_rejects_missing_and_garbage`、`test_workload_shm_so_needs_rebuild_missing_or_garbage`。
- **场景 (Scenario)**：先在 ABI 2 树编过 `libmindie_workload_shm.so`，再 checkout 到要求 ABI 3 的分支后直接启动 Coordinator，或不带 `SKIP_WORKLOAD_SHM_BUILD=0` 跑旧逻辑的 `bash build.sh`。
- **关键词 (Keywords)**：ABI 2 < 3, libmindie_workload_shm.so, SKIP_WORKLOAD_SHM_BUILD, NativeWorkloadShmUnavailable, schema-5
