# G3 冻结包清点（非正式，未冻结）

日期：2026-09-25。对照设计契约 §5。本文件只登记缺什么，**不构成 G3 冻结**，不授权 `kind=train`。

## 已有、可写入冻结包草稿

| 项 | 现状 |
|---|---|
| 设计版本 | design-v1，见 PLAN / `docs/fullmem_v4_design.md` |
| 启动档准入 | `mem=64G` / `32G` / `16G` / `8G` 机制已验收；DEFAULT=`orion-mem8g`；LABEL `primary` 保留。准入 ≠ 冻结 |
| 当前 boot | boot_id=`cc389c7a-b286-4600-ae93-71c273599497`，`mem=8G`，MemTotal ≈ 7.799 GiB，swap=0，nvpmodel 120W/mode1，IP `10.5.229.37` |
| G1 故障 | 重复提交、SSH 断、agent 可退出、worker 异常、资源失败后续、linger、盘满模拟、运行中重启已测 |
| 数据 | CIFAR10/100、CORe50 NI/NC/NIC nicv2_79、Endless A22 grouped 已有开发划分 seed=17 |
| 开发选择 | S* b16/r2000，Oracle b16/r10000，seed=17；**未冻结** |
| H4 共运行 | CIFAR100/NC、NIC S0、BR-adapt 和六格静态搜索的 seed=17 开发结果已齐。动态 S* 按序列记下，未冻结。NIC O-recon 两条是资源失败。见 [G2_H4](G2_H4.md)。**不是 48 格正式块** |
| 预算草稿 | [G3_BUDGET](G3_BUDGET.md)：一档 8G + H4 动态占用。**未冻结** |
| 矩阵分母 | [G2_GAPS](G2_GAPS.md) / [G3_MATRIX](G3_MATRIX.md)：1320 设计名额；今日正式可执行格 0 |
| 成本草图 | [G2_COST](G2_COST.md)，非正式 |

## 明确未齐（挡住 G3）

1. **源码身份**：远端 Git HEAD 仍是 v3 `b59b4c4`。v4 执行代码在工作区未作为冻结 revision 提交。冻结包需要提交哈希或完整文件快照，不能继续绑 v3 frozen_hash。
2. **orion 锁 / 环境快照**：已有 live 草稿 `reports/fullmem_v4/g3/`（pip freeze / conda export / CUDA / 内核），**未冻结为锁**。
3. **实际预算**：四档 `mem=` 已准入，静态压力均未建立，不能写成宽/中/紧三档。一档 8G + H4 只是草稿，见 [G3_BUDGET](G3_BUDGET.md)。
4. **场景门槛**：静态 tight 未建立。H4 共运行开发场景已可辨识，**未冻结**，正式 48 格未做。Endless 官方协议未建立。
5. **可执行矩阵**：见 [G3_MATRIX](G3_MATRIX.md)；今日正式可执行格 0。未冻结预算身份前，即使只承认 1 档容量，H1 的 3 seeds 也不能进正式队列。
6. **硬整机 OOM**：按设计报告未确认，不主动撞机。
7. **H7**：确定性路径下供给与参数哈希已对齐；预取净收益未建立。正式 H7 块未做。
8. **控制机备份**：v4 报告 Markdown 已复制到控制机 `backup/fullmem_v4/`。原始 run 目录仍只在 Thor。

现场草稿落在 `reports/fullmem_v4/g3/`（identity / source snapshot / pip freeze）。那是清点产物，**不是冻结包**。

已记录（仍非正式）：

- Git HEAD `b59b4c47d6de0eaf0626568c670ee170aff34468`，工作区脏，不能绑冻结。
- 内核 `6.8.12-1021-tegra`，R39.2.1，torch `2.14.0+cu132`，设备 NVIDIA Thor。
- 静态 8G 最少仍约 3.06 GiB；H4 对 S0 最少约 1.24 GiB。NIC O-recon + 2.5 GiB 高压段曾把 MemAvailable 压到 ≈0.083 GiB 后 worker oom-kill。
- H5c pause-keep-state 已测；release-rebuild 不可并入 O-recon。
- H6 开发探针 30/30 完成。非正式。

在 1–5 未关闭前，执行器继续拒绝 `kind=train`。未建立项留在覆盖分母，不删名额充完成。
