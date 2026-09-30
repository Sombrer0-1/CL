# 当前状态：已准入的正式 H4 已终态，其余块未建立

更新：2026-09-30 10:40（UTC+8）。当前阶段 `fullmem_v4`。**8G 不是 tight。只放行过 `g4-h4-mem8g`，该批次已全部终态。**

## 现场

- `mem=8G`，boot_id=`18f419b5-dd78-4c2a-bd52-4dd84d9b38da`（2026-09-29 06:34 起）。MemTotal ≈ 7.80 GiB，swap=0，nvpmodel 120W/mode 1。队列 idle，没有暂停。
- 正式 H4：48 格里 42 completed，6 条 NIC 完整 O-recon 为 `resource_failure`（systemd oom-kill）。NC 四种方法以及 NIC 的 S0、S*、BR-adapt 都完成。指标与逐 seed 差值见 [G4_H4](reports/fullmem_v4/G4_H4.md)。
- 裁决见 [G5_CLAIMS](reports/fullmem_v4/G5_CLAIMS.md)。已准入场景不支持借鉴整体 Orion。静态三档、H1–H3、H5–H8 是覆盖缺口，不是负结果。
- 其它块的 `kind=train` 仍拒绝。不把 8G 写成紧档，不放大模型，不另开 `mem=`。

查询：

```bash
export XDG_RUNTIME_DIR=/run/user/1001
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1001/bus
/home/zhuzetong/miniconda3/envs/orion/bin/python -m orion_repro.stages.fullmem_v4 status
```

## 历史结果，不作为当前队列

| 阶段 | 结项状态 |
|---|---|
| light24_v1 / RTX5060 Ti | 已结项，证据保留 |
| pressure_v2 / RTX5060 Ti | 中断收尾，未验收 |
| effectiveness_v3 / Thor thor_r2 | 129 终态：75 完成、54 CUDA OOM；主比较不支持 Orion 优势 |
| fullmem_v4 / Thor | G1 四档启动已准入；静态压力未建立；正式只跑完 H4。见 G5 |

第三阶段裁决：[CLAIMS](reports/effectiveness_v3/thor_r2/CLAIMS.md)。不与 v4 跨阶段配对。
