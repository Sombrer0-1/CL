# G2 `mem=16G` 剩余流补盖（开发，未冻结）

批次：`g2-dev-mem16g-c`。40 条记录：**39 completed**，1 条实现错误保留。`kind=probe`。不是正式矩阵，也不是 G3 冻结。

收尾：idle 基线 2026-09-23T23:06:58Z。NIC GSS/GEM 在 9-22 已完成；AGEM 原 attempt 当晚失败后暂停；9-23 15:31 修代码后 resume，9-24 07:07 左右空闲。

CIFAR100/NC 本档前半见 [G2_MEM16](G2_MEM16.md)，CIFAR10/NI/NIC 前半见 [G2_MEM16B](G2_MEM16B.md)。

## 实现错误（已修，不记科学负结果）

`g2-dev-core50-nic-agem-seed17-mem16g`：第 65/79 经验 Avalanche 0.6.0 `sample_size // n_buffers == 0`（sample_size=64）。attempt 保留，`implementation_error`。

`AdaptiveAGEMPlugin.update_memory` 改为每组至少 batch=1。重试 `...-agem-...-b2` completed，墙时 1900.9 s，P=0.023，S=0.991。

## 成绩（非正式）

墙时单位秒。

### CORe50-NIC nicv2_79

| 运行 | 墙时 | P | S |
|---|---:|---:|---:|
| GSS | 1830.7 | 0.010 | 1.000 |
| GEM | 5710.2 | 0.055 | 1.019 |
| AGEM（原） | — | — | — |
| AGEM（b2） | 1900.9 | 0.023 | 0.991 |
| ER+GEM | 5811.5 | 0.120 | 1.064 |
| ER+EWC | 3413.8 | 0.010 | 1.000 |
| MAX-P reconstructed | 1600.1 | 0.011 | 0.992 |

此前 NIC ER+GEM+EWC 是 6882 s / P=0.011。长流上插件与 GSS/MAX-P 的 P 都很低，是学习问题不是整机 OOM。**未发 NIC MAX-A**（batch=1 × 79 经验，按 NC 外推多日）。

### CORe50-NI

| 运行 | 墙时 | P | S |
|---|---:|---:|---:|
| GSS | 715.3 | 0.069 | 1.044 |
| GEM | 500.3 | 0.210 | 1.128 |
| AGEM | 419.4 | 0.165 | 1.084 |
| MAX-A reconstructed | 11735.9 | 0.028 | 0.999 |
| MAX-P reconstructed | 142.9 | 0.122 | 1.090 |

MAX-A 约 3.3 小时，塑性接近塌掉，与 64G NC MAX-A 同方向。不是把 S* 改名。

### Endless A22 grouped（不是官方协议）

| 流 | GSS P/S（墙时） | GEM | AGEM | ER+GEM+EWC | MAX-P |
|---|---|---|---|---|---|
| IC 4-exp | 0.941 / 0.668（155） | 0.832 / 0.305（49） | 0.998 / 0.003（56） | 0.141 / 0.993（112） | 0.978 / 0.003（18） |
| IL 5-exp | 0.313 / 1.000（276） | 0.895 / 1.075（202） | 0.874 / 1.141（247） | 0.350 / 0.953（561） | 0.904 / 1.047（66） |
| WC 5-exp | 0.840 / 0.981（518） | 0.892 / 0.945（208） | 0.817 / 1.095（262） | 0.550 / 0.708（560） | 0.873 / 0.926（73） |

IC AGEM/MAX-P 的 S≈0 只记诊断。这些不能填进 H3 官方 IC/IL/WC 名额。

### CIFAR10 / CIFAR100 / CORe50-NC 本档补格

| 运行 | 墙时 | P | S |
|---|---:|---:|---:|
| CIFAR100 MAX-A | 5898.3 | 0.017 | 0.992 |
| CIFAR100 MAX-P | 25.7 | 0.093 | 0.917 |
| CIFAR10 MAX-A | 5853.3 | 0.219 | 0.868 |
| CIFAR10 MAX-P | 25.6 | 0.402 | 0.665 |
| NC MAX-A | 11879.8 | 0.035 | 0.985 |
| NC MAX-P | 119.5 | 0.228 | 0.769 |
| NC GEM | 468.5 | 0.669 | 0.440 |
| NC AGEM | 367.5 | 0.612 | 0.400 |
| NC ER+GEM | 522.2 | 0.667 | 0.530 |
| NC ER+EWC | 667.5 | 0.173 | 0.830 |
| NC LR reconstructed | 170.2 | 0.623 | 0.485 |

64G CIFAR100 MAX-A 曾是 5835 / 0.026。档位对照只作诊断。

## 占用

`pressure_scope: run_id_contains_capacity`。最少仍剩 **8374267904 B ≈ 7.80 GiB**（仍是更早的 NIC ER+GEM+EWC）。本批新峰值次紧是 NIC ER+GEM ≈ 7.91 GiB 空闲。`factor_m` 对本档成员仍全部 1.0。`pressure_scene: unestablished_at_mem16g`。

不放大模型。不把 16G 写成冻结预算。

## 仍不做

- `kind=train` / 1320 正式格 / 把 16G 写成紧档
- Endless 官方机器人协议
- NIC MAX-A
- 硬整机 OOM
