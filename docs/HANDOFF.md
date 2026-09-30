# 阶段4接手

依次阅读 [STATUS](../STATUS.md)、[PLAN](../PLAN.md)、[MEMORY](MEMORY.md)、[G4_H4](../reports/fullmem_v4/G4_H4.md)、[G5_CLAIMS](../reports/fullmem_v4/G5_CLAIMS.md)、[G3 冻结](../reports/fullmem_v4/g3/FREEZE.json)、[A30](decisions/A30.md)。

单一入口仍是根 PLAN §0。当前 Thor 在 `mem=8G`（MemTotal ≈ 7.80 GiB，boot_id `18f419b5-dd78-4c2a-bd52-4dd84d9b38da`）。已准入的正式批次 `g4-h4-mem8g` 已全部终态：42 完成，6 条 NIC O-recon 资源失败。裁决已写。不要跑 v3 队列。不要把 8GiB 写成 tight 或三档。不要发射冻结文件以外的 `kind=train`。

查询：

```bash
export XDG_RUNTIME_DIR=/run/user/1001
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1001/bus
/home/zhuzetong/miniconda3/envs/orion/bin/python -m orion_repro.stages.fullmem_v4 status
```

回退 8G 之前的 16G：`sudo cp /boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20260925-mem16g /boot/extlinux/extlinux.conf` 后重启。回退 primary：`...bak.orion-fullmem-v4-20260920`。不改 nvpmodel。换档必须先暂停批次。
