# G1 `mem=8G` 准入

日期：2026-09-25。开发启动验收，**不是冻结**。

## 启动项

- 新增 LABEL `orion-mem8g`，DEFAULT 指向它。
- 保留 `primary`（无 `mem=`）、`orion-mem64g`、`orion-mem32g`、`orion-mem16g`。
- 未改 nvpmodel、内核、驱动。
- 16G 现场备份：`/boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20260925-mem16g`
- 32G 回退：`/boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20260922-mem32g`
- 64G 回退：`/boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20260922-mem64g`
- primary 回退：`/boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20260920`
- 方案目录：`reports/fullmem_v4/boot/2026-09-25T040359Z/`

## 实测

| 项 | 值 |
|---|---|
| boot_id | `cc389c7a-b286-4600-ae93-71c273599497` |
| IP | `10.5.229.37` |
| cmdline `mem=` | `8G` |
| MemTotal | 8178064 kB ≈ 7.799 GiB |
| 空闲 MemAvailable（刚启动） | ≈ 5.3–6.0 GiB |
| SwapTotal | 0 |
| nvpmodel | 120W / mode 1 |
| Linger | yes |
| CUDA | NVIDIA Thor，torch 2.14.0+cu132，device total ≈ 7.80 GiB |
| 共享池 ≤2GiB | cpu/cuda/pinned/pagecache/child/mixed 均 ok |
| mixed gpu_drop | ≈ 0.49 GiB |
| mixed cpu_extra_drop | ≈ 0.50 GiB |

共享池原文：`reports/fullmem_v4/admission/20260925T120521Z/shared_pool.json`。

请求 8G、实际 MemTotal ≈ 7.799 GiB，差额视为 firmware/carveout。空载三窗 MemAvailable 最低 ≈ 5.17 GiB。训练期本档最少仍剩 ≈3.06 GiB（batch256），峰值占用 ≈4.74 GiB。16G 上 NIC 高资源峰值占用曾到 ≈7.88 GiB，那是更长流；本批 CIFAR100/NC 静态没有压到 8G 边界，也没有资源失败。

## 队列

`g2-dev-mem8g` 9/9 completed。见 [G2_MEM8](G2_MEM8.md)。`required_capacity=mem8g`。`kind=probe`。未冻结。8G 不是 tight。
