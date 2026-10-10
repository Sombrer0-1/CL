# 阶段4长程记忆

更新：2026-10-10 11:00（UTC+8）。指针：[STATUS](../STATUS.md)、[G4_STATIC_TIGHT](../reports/fullmem_v4/G4_STATIC_TIGHT.md)、[G4_STATIC_LOOSE](../reports/fullmem_v4/G4_STATIC_LOOSE.md)。

- 当前 `mem=5G`，boot_id=`13f1512f-57ca-4ce5-8054-f415f168afd3`，MemTotal ≈ 4.856 GiB。tight 候选。8G 是 loose。mid 未建立。回退这次 8G：`extlinux.conf.bak.orion-fullmem-v4-20261007-mem8g`。
- 控制机连接：这台控制机的 `id_ed25519` 已在 2026-10-09 追加进 `authorized_keys`。密钥和密码脚本并存。
- 五流在 8G 上的 S0/S*/O-recon 已齐，O-recon 没有实用优势。5G 上 CIFAR-100、NC、NIC、CIFAR-10 已齐；NIC 的 O-recon 三条 oom-kill。5G CIFAR-10 ΔP −0.266、ΔS +0.046、时间 4.2 倍。
- NI 的 O-recon seed 0/1 在 5G 上完成，ΔP 约 −0.17、ΔS 约 −0.13。seed 2 的原 attempt 与 b2 都是热身 CUDA OOM，约 11 秒，`resource_failure`。没有第三条质量结果。
- H2 基础静态 `g4-h2-basic-nc-mem5g` 已于 2026-10-08 13:36 终态。GSS 与 GEM 各 3 条 completed；GSS seed 2 的 P 为 0.011、S 为 1。AGEM seed 0/1 completed，seed 2 热身 CUDA OOM。这只是基础静态，不是 108 格。
- 2026-10-09 14:32 那 11 条全部 completed。H2 Orion 相对基础静态没有实用优势：GEM ΔP 约 −0.32，AGEM 约 −0.39，GSS 未崩的 seed 约 −0.26，时间更长。NI O-recon seed 2 的 b3 也 completed，ΔP −0.157，ΔS −0.137。见 `reports/fullmem_v4/G4_H2_NC.md`。
- `g4-h2-fixedopt-nc-mem5g` 于 20:54 终态。8/9 completed，GSS seed 0 热身 CUDA OOM。固定 EWC 相对基础静态更差更慢，GEM ΔP 约 −0.44，AGEM 约 −0.45。
- 设计分母 1320 里大约 165 格有终态，约 12%。已准入批次都终态。中间档、Endless 官方协议、Oracle 全搜索没有跑完日期。
- 调优静态开发选择 18 条已于 2026-10-09 22:29 终态。GSS 选 b16/r200，复用基础静态正式结果。GEM 选 b256/r200。AGEM 选 b16/r2000。
- 2026-10-10 11:00 已发射 GEM 与 AGEM 的正式 seed 0/1/2，共 6 条，大约 1 到 2 小时。不要换档，不要把 5G/8G 写成已冻结三档。
