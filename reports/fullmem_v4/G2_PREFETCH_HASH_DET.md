# G2 H7 确定性路径下的预取哈希（开发，未冻结）

批次：`g2-dev-prefetch-hash-det`。对照与 [G2_PREFETCH_HASH](G2_PREFETCH_HASH.md) 相同：CIFAR100 S0，seed=17，batch 16，replay 200，`record_hashes`。当前档 `mem=16G`。`kind=probe`。

唯一新增：`training.deterministic_algorithms: true`。`configure_torch` 在该标志下设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。不是正式 H7 块，也不把 16G 写成冻结。

| 项 | prefetch off | prefetch on |
|---|---:|---:|
| status | completed | completed |
| 墙时 s | 82.2 | 77.1 |
| P | 0.4424 | 0.4424 |
| S | 0.5782 | 0.5782 |
| 经验数 | 10 | 10 |

## 供给与参数

`rolling_sha256`、`first_x_hash`、`first_y_hash`、`last_x_hash`、`last_y_hash`、`param_sha256` **10/10 全同**。`plan_mode` 仍是 `main_thread_indices_serial` vs `main_thread_indices`，这是实现预期。

上一对 mem64g 无确定性标志时，供给已对齐、参数从经验 0 分叉。本对说明分叉来自 CUDA 非确定数值路径，不是预取改了样本。

## 仍不能记成预取净收益

P/S 完全相同。消费者 `wait_s` 合计：off **0.0 s**，on **0.147 s**（2500 batch）。on 的 `produce_s` 更高（39.4 vs 27.2）是生产者侧计量，不是消费等待。约 5 s 墙时差不能记成可隔离 I/O 瓶颈，也不满足设计「净 online 收益」。不加 sleep 造瓶颈。正式 H7 若要比较 off/on，必须冻结同一套确定性标志；不能把本诊断写成 1320 里的 H7 已执行。
