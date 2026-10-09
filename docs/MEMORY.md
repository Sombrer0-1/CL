# 阶段4长程记忆

更新：2026-10-09 11:40（UTC+8）。指针：[STATUS](../STATUS.md)、[G4_STATIC_TIGHT](../reports/fullmem_v4/G4_STATIC_TIGHT.md)、[G4_STATIC_LOOSE](../reports/fullmem_v4/G4_STATIC_LOOSE.md)。

- 当前 `mem=5G`，boot_id=`13f1512f-57ca-4ce5-8054-f415f168afd3`，MemTotal ≈ 4.856 GiB。tight 候选。8G 是 loose。mid 未建立。回退这次 8G：`extlinux.conf.bak.orion-fullmem-v4-20261007-mem8g`。
- 控制机连接：这台控制机的 `id_ed25519` 已在 2026-10-09 追加进 `authorized_keys`。密钥和密码脚本并存。
- 五流在 8G 上的 S0/S*/O-recon 已齐，O-recon 没有实用优势。5G 上 CIFAR-100、NC、NIC、CIFAR-10 已齐；NIC 的 O-recon 三条 oom-kill。5G CIFAR-10 ΔP −0.266、ΔS +0.046、时间 4.2 倍。
- NI 的 O-recon seed 0/1 在 5G 上完成，ΔP 约 −0.17、ΔS 约 −0.13。seed 2 的原 attempt 与 b2 都是热身 CUDA OOM，约 11 秒，`resource_failure`。没有第三条质量结果。
- H2 基础静态 `g4-h2-basic-nc-mem5g` 已于 2026-10-08 13:36 终态。GSS 与 GEM 各 3 条 completed；GSS seed 2 的 P 为 0.011、S 为 1。AGEM seed 0/1 completed，seed 2 热身 CUDA OOM。这只是基础静态，不是 108 格。
- 2026-10-09 11:40 已发射 11 条：AGEM seed 2 的 b2、NI O-recon seed 2 的 b3，然后 `g4-h2-orecon-nc-mem5g` 的 9 条。预计大约 3 到 5 小时。不要换档，不要把 5G/8G 写成已冻结三档。
