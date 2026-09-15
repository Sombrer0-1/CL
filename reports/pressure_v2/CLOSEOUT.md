# pressure_v2 收尾：正式矩阵中断，未验收

日期：2026-09-15。平台：原 WSL2 + RTX 5060 Ti。本文件关闭第二阶段作为**未完成、未验收**的阶段记录；不是正向结论，也不是下一阶段的待跑队列。

第一阶段 `light24-v1-complete` 证据不变。原始 run 目录仍在 `runs/`（不纳入 Git）；Git 内进度摘要为同目录 `progress.json` 与 `attempts.csv`。

## 判定

- **不得声称第二阶段验收通过。**
- 正式 54 格未跑完，未做分组报告与配对分析。
- 已完成格、OOM 与中断都是有效记录，保留；不以完成率或 URGE 触发代替方法有效性。
- 仓库已迁到 Linux + 双 RTX 5090。**禁止**在新机器上续跑剩余格并并入同一 54 格比较。`L_cal≈5.94s`、IO `wait_ratio≈0.50` 及已有耗时均绑定原 5060 Ti。后续工作另开阶段、先在本机重建 `orion` 环境并重新校准。

## 已交付（实现与校准）

- 冻结协议 `experiments/pressure_v2/frozen_protocol.json`，`frozen_hash=863ac7b0f08570d985be0c2a5e0dd4e692088d0812854ce57455f58a1907471a`。
- 独立执行器 `python -m orion_repro.pressure_study`；进度文件 `runs/pressure_v2_progress.json`，不写 light24。
- 开发校准（不按正式 P/S 选参）：eval_batch=32；Q_tight=176 MiB；Q_loose=352 MiB；L_cal≈5.94s；各预算 S* 均为 b16/r2000；DYN 在 k=3..5 预留 32505856 B；IO 开发 wait_ratio≈0.50。
- 开发扫描 40 格：25 completed + 15 cuda_oom。正式前要点：CIFAR100 eval_batch=128 评价 OOM，32/8 可完成；CORe50 128 MiB OOM，176 MiB 训练 reserved/配额≈0.886；O00（L=30）可完成且收缩；O10/O11 开发路径启用 GEM/EWC 后于 experience 6 训练 OOM。

## 正式 54 格进度（中断时）

已记录 45 / 54：completed 30、cuda_oom 14、interrupted 1。累计子进程墙时约 11229s。`active=null`。

| 分组 | 计划 | 已记录 | 未启动 |
|---|---:|---:|---|
| tight | 21 | 21 | 0 |
| loose | 9 | 9 | 0 |
| dyn | 9 | 9 | 0 |
| decay | 3 | 3 | 0 |
| pref | 6 | 3 | 3 |
| io | 6 | 0 | 6 |

- 中断格：`configs/pressure_v2/formal/pref/O11_ps_s1.yaml`（`20260915T122513Z_pressure_v2_016f9552`）。
- 未启动：`pref/O11_latency_s1`、`pref/O11_latency_s2`、`pref/O11_ps_s2`，以及全部 6 个 IO 格。
- 若在原平台续跑，中断格需重跑，加上 9 个未启动格，共 10 格。本收尾选择不续跑。

14 例正式 cuda_oom 与开发期插件路径 OOM 一样，是配额内资源失败证据，不是实现错误清零。

## 下一阶段边界

1. 在本机新建独立 Conda 环境 `orion`（Python 3.11），不要使用 `/home/admin/miniconda3` 或 mineru。
2. 重新确认 GPU/驱动/RAM；历史 5060 Ti 校准值不自动沿用。
3. 新阶段使用新 study id 与新进度文件；不覆盖 `reports/light24/`，也不把本中断矩阵改写成已完成。
