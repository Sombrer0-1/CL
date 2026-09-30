# G2 `mem=32G` 剖析（开发，未冻结）

批次：`g2-dev-mem32g`。**8/8 completed，失败 0。** 2026-09-22T02:18Z 提交，约 10:34 CST 收尾。`kind=probe`。不是正式矩阵。

## 准入

见 [G1_MEM32](G1_MEM32.md)。MemTotal ≈ 31.399 GiB。空闲刚启动 ≈ 30.6 GiB。

## 成绩（非正式）

墙时单位秒。与 64G seed=17 对照，学习指标同方向，不是冻结比较。

| 运行 | 墙时 | P | S |
|---|---:|---:|---:|
| G1 CIFAR100 单经验 | 7.5 | 0.440 | 1.000 |
| CIFAR100 S0 | 72.3 | 0.438 | 0.586 |
| CIFAR100 batch=256 | 24.9 | 0.092 | 0.917 |
| CIFAR100 replay=2000 | 70.5 | 0.406 | 0.699 |
| CIFAR100 ER+GEM+EWC | 319.2 | 0.174 | 0.818 |
| CORe50-NC S0 | 230.4 | 0.669 | 0.441 |

64G 对照：S0 71.5/0.442；ER+GEM+EWC 326/0.174；NC S0 227/0.641。32G 没有把学习问题变成资源失败。

## 占用

占用扫描 171 个 G2 成员；其中本档 `mem32g` 运行最少仍剩 **28439089152 B ≈ 26.49 GiB**（CIFAR100 batch256）。峰值 MemTotal−MemAvailable ≈ **4.91 GiB**。`factor_m` 全部 1.0。`pressure_scene: unestablished_at_mem32g`。

不要放大模型造压。用户已授权更紧档；下一窗口是 `mem=16G` 准入，不是冻结。

## 仍不做

- `kind=train` / 1320 正式格 / G3 冻结
- 把 16GiB 写成已冻结
- 改 nvpmodel / 删 LABEL `primary`
