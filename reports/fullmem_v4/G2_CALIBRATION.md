# G2 64GiB 开发校准留档（2026-09-21）

状态：**G2 64GiB 全流开发剖析已齐，非正式冻结。** 压力场景未建立。没有正式训练命令。

批次：`g2-dev-mem64g`。执行器在 2026-09-20 20:42 空闲收尾第二批。失败 2：故意注入 `g1-impl-error-inject`，以及 `g2-dev-cifar100-prefetch` YAML notes 未加引号（attempt 保留；`prefetch-b` 成功）。

## 数据

| 流 | 论文 experiences | 本机 | 备注 |
|---|---:|---|---|
| SplitCIFAR10 | 10 | raw 341 MiB，类序 4 1 7 5 3 9 0 8 6 2 | 未改 git manifest |
| SplitCIFAR100 | 10 | raw 501 MiB | 已有 |
| CORe50-NC/NI/NIC | 9 / 8 / 79 | raw 4.1 GiB | NC 用历史 processed；NI/NIC filelist 写在 `reports/fullmem_v4/data/`，计数与历史 manifest 一致 |
| Endless IC/IL/WC | 4 / 5 / 5 | zip+解压约 14 GiB | 未改 `data/manifests/endless_cl_sim.json`；A22 分组不是官方机器人协议 |

对齐表：`docs/fullmem_v4_alignment.csv`。

## 全流墙时与 P/S（开发 seed，非正式）

| 运行 | 流 | 方法 | exp | 墙时 s | p_diag | s_initial |
|---|---|---|---:|---:|---:|---:|
| cifar10-s0 | CIFAR10 | S0 | 10 | 69.9 | 0.958 | 0.057 |
| cifar100-s0 | CIFAR100 | S0 | 10 | 67.8 | 0.402 | 0.611 |
| cifar100-prefetch | CIFAR100 | S0 预取开 | 10 | 62.1 | 0.430 | 0.583 |
| cifar100-batch256 | CIFAR100 | 高 batch | 10 | 25.6 | 0.068 | 0.939 |
| cifar100-replay2000 | CIFAR100 | 高 replay | 10 | 69.3 | 0.395 | 0.712 |
| cifar100-er-gem | CIFAR100 | ER+GEM | 10 | 170.4 | 0.424 | 0.596 |
| cifar100-er-ewc | CIFAR100 | ER+EWC | 10 | 263.6 | 0.233 | 0.752 |
| cifar100-er-gem-ewc | CIFAR100 | ER+GEM+EWC | 10 | 329.8 | 0.156 | 0.837 |
| core50-nc-s0 | CORe50-NC | S0 | 9 | 225.7 | 0.641 | 0.460 |
| core50-ni-s0 | CORe50-NI | S0 | 8 | 254.0 | 0.232 | 1.162 |
| core50-nic-s0 | CORe50-NIC | S0 | 79 | 1750.0 | 0.117 | 1.021 |
| endless-ic-s0 | Endless IC | S0 | 4 | 36.4 | 0.994 | 0.488 |
| endless-il-s0 | Endless IL | S0 | 5 | 151.1 | 0.991 | 0.995 |
| endless-wc-s0 | Endless WC | S0 | 5 | 151.4 | 0.975 | 0.968 |
| cifar100-lr-seed17 | CIFAR100 | LR_reconstructed | 10 | 42.4 | 0.395 | 0.636 |
| core50-nc-lr-seed17 | CORe50-NC | LR_reconstructed | 9 | 171.5 | 0.622 | 0.443 |

S>1 表示旧域相对首次学习平均上升（设计允许）。Endless 高 P 是 5 类补丁流，不能写成机器人闭环成功。

## L_cal（本机重测）

`experience_metrics.learning_s` 中位数：CIFAR100 S0 **5.513 s/经验**（旧配置占位 30 s）。NC S0 15.706 s；NI 19.308 s；NIC 1.871 s；Endless IL/WC 约 23 s。相对 30 s 阈值，CIFAR 上 latency 因子会饱和「够快」。不沿用 v3 L_cal。

## 物理占用

`reports/fullmem_v4/g2/occupancy.json`：14 个完成流。系统最少仍剩 **61788577792 B ≈ 57.55 GiB**。RSS 高峰约 2.0–2.5 GiB。GPU reserved 高峰除 batch256≈0.91 GiB 外约 0.15 GiB。没有资源失败。

**压力场景未建立。** 不放大模型。不能把 64GiB 写成紧/中/宽三档。

## 公式

km=0.25、32 MiB slack 时 `factor_m≈0.9997`。O-recon 会饱和；不静默改成归一化。

## 成本草图

见 [G2_COST](G2_COST.md) 与 `reports/fullmem_v4/g2/cost_estimate.json`。1320 设计名额按 CIFAR100 S0 粗算约 25 小时下限；带 GEM/EWC 或 NIC 会长很多。Oracle 42 格仅 CIFAR100 约 0.8 小时下限。32/16GiB 未验收，失败率未知。

## 明确未完成 / 下一步

- **完整 O-recon 接线已复核**，见 [G2_WIRING](G2_WIRING.md)。旧 `g2_*_orecon` 仍只是 batch/replay 探针。
- S* 六格与 Oracle 42 的 **seed=17** 搜索已完成，开发选择见 [G2_SELECTION](G2_SELECTION.md)：S* b16/r2000，Oracle b16/r10000。**未按协议冻结。** seed=0 探索不能选参。
- 预取 H7：seed=17 供给哈希一致、参数更新不一致，收益未建立。见 [G2_PREFETCH_HASH](G2_PREFETCH_HASH.md)。
- LR reconstructed seed=17 已完成，见 [G2_LR](G2_LR.md)。MAX-A/MAX-P 仍未接线。
- Endless 仍是 A22 自建分组。
- 32/16GiB 启动与选档；硬整机 OOM / 盘满 / 运行中重启 / terminate-user。
- G3 冻结与正式矩阵。不要把 64GiB 旧跑通写成主方法已校准。
