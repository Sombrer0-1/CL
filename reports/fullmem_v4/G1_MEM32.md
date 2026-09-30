# G1 `mem=32G` 准入

日期：2026-09-22。不是冻结，不是 16G。

## 启动项

- 新增 LABEL `orion-mem32g`，DEFAULT 指向它。
- 保留 LABEL `primary`（无 `mem=`）与 `orion-mem64g`。
- 未改 nvpmodel、内核、驱动。
- 现场备份：`/boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20260922-mem64g`
- primary 回退：`/boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20260920`

## 实测

| 项 | 值 |
|---|---|
| boot_id | `6ac10659-f2df-44e7-a713-6bb21074bdd4` |
| IP | `10.5.229.37` |
| cmdline `mem=` | `32G` |
| MemTotal | 32924032 kB ≈ 31.399 GiB |
| 空闲 MemAvailable（刚启动） | ≈ 30.6 GiB |
| SwapTotal | 0 |
| nvpmodel | 120W / mode 1 |
| Linger | yes |
| CUDA | NVIDIA Thor，torch 2.14.0+cu132 |
| 共享池 ≤8GiB | cpu/cuda/pinned/pagecache/child/mixed 均 ok |
| mixed gpu_drop | ≈ 1.01 GiB |
| mixed cpu_extra_drop | ≈ 0.82 GiB |

共享池原文：`reports/fullmem_v4/admission/20260922T101842Z/shared_pool.json`。

## 队列

`g2-dev-mem32g`：g1 单经验 → CIFAR100 S0 / batch256 / replay2000 / ER+GEM+EWC → CORe50-NC S0 → occupancy → idle。开发 seed=17。`required_capacity=mem32g`。
