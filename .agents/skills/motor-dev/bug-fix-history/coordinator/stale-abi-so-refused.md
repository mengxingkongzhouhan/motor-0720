# [2026-09-24] checkout 残留 ABI 2 `.so` 无法绑定 schema-5

- **现象 (Symptom)**：只 checkout schema-5 代码、不重编 `libmindie_workload_shm.so` 时，Mgmt `create_v4` / Worker attach 在旧 ABI 2 库上失败或 ctypes 参数错位，Coordinator 起不来。
- **根因 (Root cause)**：`lib/*.so` 被 gitignore。schema-5 把 `cas_add` / `cas_sub_floor0` / `load_entry` 多了 overlay 参数（ABI 2 → 3），残留 ABI 2 库仍会被 `load_native_library` 搜到。
- **为什么会写出 (Why)**：以为 `pip install -e .` 或复用上次 `lib/` 产物即可；忽略了 ABI 与 layout 一起升级。
- **修复 (Fix)**：`MIN_ABI_VERSION=3` 拒绝旧库；`build.sh` 把 ABI 过旧的 `.so` 当缺失并重编。
- **测试拦截 (Test interception)**：`test_stale_abi_library_is_refused`、`test_workload_shm_so_needs_rebuild_missing_or_garbage`。
- **场景 (Scenario)**：从 schema-4 分支切到 schema-5 后未 `SKIP_WORKLOAD_SHM_BUILD=0 bash build.sh`。
- **关键词 (Keywords)**：ABI 2, leftover .so, schema-5, MIN_ABI_VERSION, build.sh
