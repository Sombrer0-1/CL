# 正式协议 paper_feedback_diag_initial_v1

> 历史协议/选择规则参考。当前实验范围与新有限搜索规则由 PLAN.md 和 light24_v1.md 规定；不要据此恢复旧全矩阵。
状态：**2026-09-14 冻结为 protocol v1。** 开发 val 结果不得当作本协议主表。Oracle 获选格子按 A15 在搜索完成后填入，不在正式测试上重选。

## 冻结项

| 项 | 采用值 | 分类 |
|---|---|---|
| 反馈 | `official_test_seen`（CIFAR/CORe50 官方 train 全集 + 官方 test 已见域） | 论文口径重建（A06） |
| 指标 | `P_diag` / `S_initial`，见 `metric_definitions.md` | A05 |
| 骨干/优化 | ResNet-20 option A，SGD 0.01/0.9，1 epoch，FP32 eager | A02/A07/A18 |
| CIFAR10 类序 | `[4, 1, 7, 5, 3, 9, 0, 8, 6, 2]` | 开发 shuffle 冻结，非论文列出 |
| CIFAR100 类序 | `configs/development/cifar100_er_static.yaml` 中的 100 类列表 | 同上 |
| CORe50 | mini 32×32，run=0，object_level，NC 9 / NI 8 / NIC nicv2_79 | A14 |
| 主预算 B0 | `observed_only`，URGE `M_max=4096 MiB`，无 allocator/cgroup 硬上限 | 平台：host cgroup blocked |
| URGE 主协议 | balanced k=0.25；`L_th=30 s`；`Thr0=0.05`；`delta=0`；`α=0.1`；`β=0.2`；`MB0=16`；`MR0=200` | `L_th=30` 为本机非轨迹先验；本机 learning≈2 s 时 latency 因子近 0 |
| 预取 | 主比较默认 **关闭** | 三次 prefetch-off 重复 P∈[0.944,0.979]、learning_sum∈[16.38,17.97]s；prefetch-on learning_sum=36.11s（约 2.1×）超出 off 方差。单次 on/off 的 P 差落在 off 重复范围内，不得据此声称准确率不等价或把历史 P 差全部归因于 GPU 噪声。E07 仍需更多 on 重复与固定轨迹配对 |
| Seeds | 主比较 `{0,1,2}`，四类 seed 同值 | PLAN §8 |
| Oracle 搜索 | 开发 val、model seed=0、split seed=17；规则 A15 | 搜索耗时与获选单次训练分账 |
| MAX-A/MAX-P/LR | `_reconstructed` 名称保留 | A11 |
| E04 LR×GEM/AGEM/GSS | **不实现、不静默替换** | `experiments/formal_skipped.json` |

## 明确不在本协议内

- Jetson 能耗、原设备耗时、机器人闭环。
- host cgroup 硬限制（本 WSL 命名空间 `host_enforced=blocked`）。
- 用正式测试挑选超参。历史官方测试暴露不能通过换 seed 消除。
- `--resume` 不是 checkpoint 恢复。
- Endless 数字 patch ID 的源帧时间映射。

## 敏感性（E08，不进入主表选参）

- `L_th_profiled=2.1`（CIFAR10 开发 learning 量级），`M_max=160`。
- 预取 on 作为命名变体，不得与主表混比，除非 E07 验收等价。

## 失败与重跑

- 实现错误：修复后重跑受影响 identity，保留失败记录。
- `cuda_oom` / `budget_exceeded` / `preflight_infeasible`：计入失败分布，不改成零准确率。
- 去重仅当 resolved config + 源码快照 + 环境锁 + 数据 manifest 全同。
