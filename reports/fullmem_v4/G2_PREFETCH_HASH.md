# G2 H7 预取供给哈希（seed=17）

更新：2026-09-21 22:10（UTC+8）。批次 `g2-dev-prefetch-hash`。**不是冻结，不能凭墙时差宣称预取净收益。** 后续确定性路径见 [G2_PREFETCH_HASH_DET](G2_PREFETCH_HASH_DET.md)。

对照：同一 S0（CIFAR100，batch 16，replay 200，eval 32，开发 seed=17）。off 为串行 `main_thread_indices_serial`；on 为队列 `main_thread_indices`、`queue_depth=2`。两边都开了 `data_supply.record_hashes`。

| 项 | prefetch off | prefetch on |
|---|---:|---:|
| status | completed | completed |
| 墙时 s | 71.5 | 64.2 |
| P | 0.4418 | 0.4234 |
| S | 0.5752 | 0.5959 |
| 每经验 batch 数 | 250 | 250 |

## 供给侧（10/10 experience 一致）

`rolling_sha256`、`first_x_hash`、`first_y_hash`、`last_x_hash`、`last_y_hash` 全部相同，没有 null。样本张量/标签在消费点上等价。plan_mode 不同是实现预期（串行 vs 有界队列），不是样本不同。

## 参数更新（10/10 不一致）

每经验结束后的 `param_sha256` 全部不同，从经验 0 就开始分叉。设计要求 H7 静态 off/on 还要参数更新一致。**这条没过。** 因此 P/S 差和约 7 s 墙时差都不能记成预取净收益。

经验 0 的新流 batch 哈希已经对齐，却仍得到不同权重，更像是 CUDA/非确定性数值路径，而不是预取改了样本。不加 sleep 造瓶颈。真实供给在 mem64g 上也不构成可隔离的 I/O 瓶颈。

结论：**供给等价已建立；学习轨迹不等价；H7 预取收益未建立。** 旧 seed=0 那对 P/S 不同且当时哈希为 null，不能回写成本次结论。

占用重汇总：`occupancy-d` 120 条，最少仍剩 ≈52.3 GiB。压力场景仍未建立。
