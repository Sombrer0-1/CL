# 阶段4接手

先读 [STATUS](../STATUS.md)、[G4_STATIC_TIGHT](../reports/fullmem_v4/G4_STATIC_TIGHT.md)、[G4_STATIC_LOOSE](../reports/fullmem_v4/G4_STATIC_LOOSE.md)、[MEMORY](MEMORY.md)。

当前 `mem=5G`，boot_id `13f1512f-57ca-4ce5-8054-f415f168afd3`。执行器正在跑 `g2-h2-tune-select-nc-mem5g` 的 18 条开发选择。不要停，不要换档，不要在选出之前发射调优静态的正式 seed。

回到 8G：`sudo cp /boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20261007-mem8g /boot/extlinux/extlinux.conf` 后重启。这批结束前不换档。mid 未建立。

下一批 `kind=train` 先登记进 `reports/fullmem_v4/g3/FREEZE.json` 的 `admitted_batches`，再提交给执行器。不要用 v3 队列发射。失败记录保留，重试换新的 task_id。
