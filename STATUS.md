# 当前状态：阶段 4 已结项，整机已还原

更新：2026-10-11 23:30（UTC+8）。

**裁决：Orion（O-recon）没有可信的实用优势，不借鉴，结束投入。** 完整裁决、证据表与归因见 [G5_CLAIMS](reports/fullmem_v4/G5_CLAIMS.md)。不再发射 `kind=train`。

## 环境（已还原）

- 启动项：`/boot/extlinux/extlinux.conf` 已还原为实验前原件（与 `extlinux.conf.bak.orion-fullmem-v4-20260920` 逐字节一致），`DEFAULT primary`，没有 `mem=`，也没有 `orion-mem*` 标签。
- 重启验收，2026-10-11 23:10：boot_id `68b3d186-454f-44a1-adaa-eaee2be1f369`，`MemTotal` 128766588 kB（约 122.8 GiB，128G 机型的整机内存），swap 0，`/proc/cmdline` 无 `mem=`，IP 仍 `10.5.229.37`。nvpmodel 仍是 120W/mode 1，驱动 595.78 未动，`orion` 环境里 torch 2.14.0+cu132 的 CUDA 前向与 1 GiB 分配正常，根盘 12%。
- 执行器：`orion-fullmem-v4-executor.service` 已 stop 并 disable，重启后保持 inactive/disabled。停止时 inbox、running 都空，`mode=idle`，completed_n 557、failed_n 39。单元文件 `~/.config/systemd/user/orion-fullmem-v4-executor.service` 还在，不再自启。
- Linger 实验前就是 yes，保持不动。
- 回到实验配置（只有用户明确要求才做）：`sudo cp /boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20261011-pre-restore /boot/extlinux/extlinux.conf` 后重启，得到 `DEFAULT orion-mem5g` 和 64/32/16/8/5G 标签；再 `systemctl --user enable --now orion-fullmem-v4-executor.service`。其它档的备份仍在 `/boot/extlinux/`。
- 没有改内核、驱动、nvpmodel。

## 终态

设计分母 1320 个名额，约 165 格有终态，其余是覆盖缺口，不写成无效。已准入批次全部终态。

| 范围 | 结果 |
|---|---|
| 5 个数据流 × 5G/8G 的 S0 / S* / O-recon | O-recon 相对 S*：ΔP −0.17 到 −0.37，online 2.1 到 4.5 倍，没有实用优势。见 [G4_STATIC_LOOSE](reports/fullmem_v4/G4_STATIC_LOOSE.md)、[G4_STATIC_TIGHT](reports/fullmem_v4/G4_STATIC_TIGHT.md) |
| H4 共运行压力（NC/NIC，8G + 2.5 GiB 持有） | NC 完成但 O-recon 更差更慢；NIC 的 O-recon 6/6 oom-kill。见 [G4_H4](reports/fullmem_v4/G4_H4.md) |
| H2（NC 上 GSS/GEM/AGEM，5G） | Orion 对同算法基础静态 ΔP −0.26 到 −0.39；固定 EWC 更差更慢；调优静态没有拉开。见 [G4_H2_NC](reports/fullmem_v4/G4_H2_NC.md) |
| H5b（插件拆分，NC 与 CIFAR-100） | **无效**：配置 `fixed_default` 使插件全程关闭，等同 ER。见 [G4_H5B](reports/fullmem_v4/G4_H5B.md) |

未建立：中间档、Endless 官方协议、Oracle 全搜索、LR/MAX-A/MAX-P、H1–H3、H5–H8 的正式矩阵。

## 位置

- Git 仓库在 Thor：`/home/zhuzetong/research/cl/Reproduce-Orion`，推送目标 `origin/master`（`Sombrer0-1/CL`）。`runs/` 被 gitignore，原始 `summary.json` 与控制轨迹只在 Thor 本机。
- 控制机工作区 `C:\Users\admin\Desktop\ssh` 不是 git 仓库，里面有 SSH 脚本，密码只在这些脚本里，不进远端仓库。
