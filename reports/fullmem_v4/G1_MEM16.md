# G1 `mem=16G` 准入

日期：2026-09-22。开发启动验收，**不是冻结**。

## 启动项

- 新增 LABEL `orion-mem16g`，DEFAULT 指向它。
- 保留 `primary`（无 `mem=`）、`orion-mem64g`、`orion-mem32g`。
- 未改 nvpmodel、内核、驱动。
- 32G 现场备份：`/boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20260922-mem32g`
- 64G 回退：`/boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20260922-mem64g`
- primary 回退：`/boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20260920`

## 实测

| 项 | 值 |
|---|---|
| boot_id | `7b3d596d-3424-4a78-854c-18fd5cc5833e` |
| IP | `10.5.229.37` |
| cmdline `mem=` | `16G` |
| MemTotal | 16435592 kB ≈ 15.674 GiB |
| 空闲 MemAvailable（刚启动） | ≈ 13.1–13.5 GiB |
| SwapTotal | 0 |
| nvpmodel | 120W / mode 1 |
| Linger | yes |
| CUDA | NVIDIA Thor，torch 2.14.0+cu132，device total ≈ 15.67 GiB |
| 共享池 ≤4GiB | cpu/cuda/pinned/pagecache/child/mixed 均 ok |
| mixed gpu_drop | ≈ 1.02 GiB |
| mixed cpu_extra_drop | ≈ 1.02 GiB |

共享池原文：`reports/fullmem_v4/admission/20260922T110041Z/shared_pool.json`。

## 队列

`g2-dev-mem16g`：G1 单经验 + CIFAR100 方法卡（S0/GSS/GEM/AGEM/ER+GEM/ER+EWC/ER+GEM+EWC/LR/S*/O-recon/batch256/replay2000）+ CORe50-NC（S0/GSS/ER+GEM+EWC/O-recon）+ occupancy/idle。开发 seed=17。不含 MAX-A。`required_capacity=mem16g`。未冻结。
