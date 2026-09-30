# G2 `mem=16G` 剖析（开发，未冻结）

批次：`g2-dev-mem16g`。**19/19 completed，失败 0。** 2026-09-22T03:00Z 提交，约 04:18Z 空闲基线收尾。`kind=probe`。不是正式矩阵，也不是 G3 冻结。

## 准入

见 [G1_MEM16](G1_MEM16.md)。MemTotal ≈ 15.674 GiB。空闲刚启动 ≈ 13.1–13.5 GiB。空闲窗最少 MemAvailable ≈ 13.03 GiB。

## 成绩（非正式）

墙时单位秒。与 64G seed=17 对照，不是冻结比较。

### CIFAR100

| 运行 | 墙时 | P | S |
|---|---:|---:|---:|
| S0 | 69.9 | 0.447 | 0.578 |
| GSS | 137.1 | 0.010 | 1.000 |
| GEM | 155.7 | 0.446 | 0.570 |
| AGEM | 125.0 | 0.396 | 0.604 |
| ER+GEM | 177.2 | 0.420 | 0.605 |
| ER+EWC | 262.4 | 0.319 | 0.716 |
| ER+GEM+EWC | 310.2 | 0.154 | 0.840 |
| LR reconstructed | 43.0 | 0.379 | 0.658 |
| S* b16/r2000 | 74.4 | 0.413 | 0.695 |
| O-recon full hash-off | 253.2 | 0.197 | 0.792 |
| batch=256 | 25.8 | 0.091 | 0.912 |
| replay=2000 | 72.3 | 0.396 | 0.691 |

64G 对照：S0 71.5/0.442；ER+GEM+EWC 326/0.174；S* 71.1/0.410。GSS 在 16G 上 P=0.01（64G seed=17 曾是 0.174），记为开发诊断，不写成已修好或已证实档位因果。

### CORe50-NC

| 运行 | 墙时 | P | S |
|---|---:|---:|---:|
| S0 | 229.8 | 0.699 | 0.407 |
| GSS | 575.2 | 0.393 | 0.673 |
| ER+GEM+EWC | 982.2 | 0.160 | 0.845 |
| O-recon full hash-off | 799.8 | 0.180 | 0.823 |

64G NC GSS 曾是 P=0.022。16G 这次 P=0.393，同样只记诊断，不拿单次差改协议。

## 占用

`pressure_scope: run_id_contains_capacity`，16 个本档成员。最少仍剩 **11681619968 B ≈ 10.88 GiB**（CIFAR100 batch256）。峰值 MemTotal−MemAvailable ≈ **4.79 GiB**。`factor_m` 全部 1.0。`pressure_scene: unestablished_at_mem16g`。

64G 最少 ≈52.3 GiB，32G 最少 ≈26.49 GiB，16G 最少 ≈10.88 GiB。档位在变，ResNet-20 32×32 训练峰仍约 5 GiB，内存因子仍饱和。**不要放大模型造压。不要把 16G 写成已冻结预算。**

副本：`reports/fullmem_v4/g2/occupancy_mem16g.json`。

## 仍不做

- `kind=train` / 1320 正式格 / G3 冻结
- 把 16GiB 写成已冻结
- 改 nvpmodel / 删 LABEL `primary`
- 硬整机 OOM、盘满、运行中重启
