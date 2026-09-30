# G2 完整 O-recon 接线复核（2026-09-21）

状态：**接线已在 mem64g 开发探针上复核，不是 G3 冻结，不是正式主比较。** 64 GiB 压力仍未建立。

批次：`g2-dev-orecon-full`。执行器 2026-09-21T01:15:28Z 开始，01:32:21Z 两条均 `completed` exit 0。

## 复核了什么

对照用户 2026-09-21 指出的缺口，以及设计契约 §3.1 / §3.2。

| 项 | 旧 `g2_*_orecon` | 新 `g2_*_orecon_full` | 结论 |
|---|---|---|---|
| 内存观测 | `controlled_resource: device`，CIFAR100 首轮 `memory_mib≈91` | `board`，CIFAR100 5309–5574 MiB；NC 5118–5427 MiB | 已接到训练窗 MemTotal−MemAvailable 峰值 |
| 侧通道 | 未分列 | 同窗 GPU allocated ≈91–182 MiB，RSS ≈2.1–2.5 GiB，**不相加** | 符合「不把 RSS/allocated 当整机总量」 |
| 映射 | `m_batch=m_frame=1`，`mb0=16`，`mr0=200` | 32×32 uint8 **3072** 字节，`mb0=49152`，`mr0=614400` | 个数空间不再冒充字节 |
| 插件 | `optional_plugins: none`，`fixed_default` | `gem_ewc` + `adaptive` | 第 0 经验 default（插件关），第 1 经验起 advanced（GEM/EWC 开） |
| 开发 seed | model/stream/replay/aug=**0** | **17** | 符合设计；划分 seed=17 不能替代这项 |
| source 身份 | 会吃 `executor.json` 心跳 | 快照 1053 文件含 `docs/fullmem_v4_design.md`，不含 `state/executor.json` 与 queue | 冻结闭包仍差 git commit；HEAD `b59b4c4` 不能冒充 |

CIFAR100 经验 0：`optimizer_mode=default`，`gem/ewc=False`。经验 1：`advanced`，GEM `before_backward=250`，插件张量驻留。这是完整方法，不是只调 batch/replay。

旧两条 `g2_*_orecon` **保留为 batch/replay 开发探针**，不进入 O-recon 冻结依据。

## 原式在 64 GiB 上的行为（预期，不是实现错误）

`factor_m=1.0` 全程。km=0.25 时 32 MiB 余量已饱和；本机最少仍剩约 50 GiB+，内存因子不会触发收缩。URGE 只受 P/S/L 驱动，batch 16→18、replay 200→280。这是设计已要求诊断的饱和，**不能因此改成归一化或加隐藏余量**（那是 O-eng）。

CIFAR100 P 从旧探针 0.411 降到完整方法 0.194（墙时 63s→246s）；NC 0.664→0.261（193s→751s）。插件打开后塑性下降，与 v3 插件归因同方向，**这是学习问题不是整机 OOM**。

## 明确仍不是冻结

- 压力场景未建立。
- S*/Oracle seed=0 是探索；seed=17 开发选择见 [G2_SELECTION](G2_SELECTION.md)，未冻结。
- 预取：seed=17 开了 `record_hashes` 后供给哈希 10/10 一致，参数哈希 10/10 不一致。见 [G2_PREFETCH_HASH](G2_PREFETCH_HASH.md)。**不能宣称预取净收益。**
- Endless 仍是 A22 开发分组。
- 执行器恢复探针已过（2026-09-21T04:29–04:31Z）：监督 SIGTERM 后 `g1-adopt-a` 以 `finalize_dead` 采纳（marker 在、unit 已死），inbox 的 `g1-adopt-b` 直到 `resume` 才启动；`g1-resource-fail` 记 `resource_failure`，health=`ok` 后 `g1-resource-after` completed。paused 路径会收尾 recorded worker，但全局故障仍保持 paused，不自动取 inbox。
- 开发 seed=17 的 S* 六格 + Oracle 42 已另开批次 `g2-dev-seed17-select`，**选择尚未冻结**。
- 32/16GiB 未启动。`kind=train` 仍拒绝。
