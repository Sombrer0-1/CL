# 阶段4接手

阶段 4 已于 2026-10-11 结项：Orion 不借鉴，结束投入。先读 [G5_CLAIMS](../reports/fullmem_v4/G5_CLAIMS.md) 和 [STATUS](../STATUS.md)。

整机已还原为原始 `primary` 启动项（无 `mem=`，约 122.8 GiB）。执行器已停用。没有未完成的批次，也没有待办的训练。

不要重开。若用户明确要求重开：

1. 回到实验启动配置与执行器的步骤见 [STATUS](../STATUS.md) 的「回到实验配置」。
2. H5b 必须用 `plugin_policy: fixed_advanced` 另 ID 重发；原记录不改，不用同一 task_id。
3. 下一批 `kind=train` 先登记进 `reports/fullmem_v4/g3/FREEZE.json` 的 `admitted_batches`，再提交给执行器。不要用 v3 队列发射。
4. 先读 [MEMORY](MEMORY.md) 的「已知坑」。
