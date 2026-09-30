# 阶段4长程记忆

更新：2026-09-30 10:40（UTC+8）。指针：[STATUS](../STATUS.md)、[G4_H4](../reports/fullmem_v4/G4_H4.md)、[G5_CLAIMS](../reports/fullmem_v4/G5_CLAIMS.md)。

- `fullmem_v4` / design-v1。正式只放行过 H4 批次 `g4-h4-mem8g`，现已全部终态。8G 不是 tight。
- 48 格：42 completed；NIC 完整 O-recon 6/6 为 systemd oom-kill，记 `resource_failure`。NC 上 O-recon 相对动态 S* 没有实用优势。
- 静态宽/中/紧未建立。H1–H3 与 H5–H8 保持未验证，不发射。研究决策：现有证据不支持整体借鉴 Orion，未跑块不能写成方法无效。
- boot_id=`18f419b5-dd78-4c2a-bd52-4dd84d9b38da`，`mem=8G`，执行器 idle。
