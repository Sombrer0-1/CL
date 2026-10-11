# H5b：固定 batch/replay 的插件拆分（结论作废）

更新：2026-10-11 23:30（UTC+8）。`mem=5G`。NC 与 CIFAR-100 两批都已终态。

## 作废说明

**这两批的插件全程是关的，结果等同于 ER，不是插件拆分。下面的表只作原始记录，不得当作「GEM/EWC 对质量没有影响」的证据。**

- 配置 `controller.plugin_policy: fixed_default`。`runner/loop.py` 在起跑时执行 `plugin.enabled = (initial_mode == "advanced")`，`initial_optimizer_mode` 对 `fixed_default` 恒返回 `default`，所以 `optional_start_enabled: true` 被覆盖，GEM 和 EWC 的 `TogglePlugin` 全程 `enabled=false`。
- 运行里的痕迹（核对了 NC 的 ER+EWC seed 0 与 CIFAR-100 的 ER+GEM+EWC seed 0）：`plugin_activity.jsonl` 里 `enabled_history` 全是 `false`，`before_backward=0`；NC 的 ER+GEM、ER+EWC、ER+GEM+EWC 和 CIFAR-100 的三组在 `experience_metrics.csv` 里 `optimizer_mode` 一直是 `default`、`auxiliary_visits=0`、`SparseGEMPlugin` 与 `EWCPlugin` 的 `plugin_state_bytes` 全是 0；P、S、时间都落在 S0 的噪声内。
- 要打开插件须用 `fixed_advanced`。H2 的固定 EWC 批次用的是打开的配置，结果也确实不同（见 [G4_H2_NC](G4_H2_NC.md)）。
- 按用户 2026-10-11 的决定，Orion 结项，不用 `fixed_advanced` 重发。所有记录保留，不改写，不用同一 task_id 重跑。插件对 O-recon 退化的归因改由控制轨迹和 seed 17 的开发对照承担，见 [G5_CLAIMS](G5_CLAIMS.md)。

## NC（batch 16，replay 200，控制器关闭，插件实际关闭）

ER 复用已完成的 S0。ER+EWC seed 1 的原 attempt 是热身 CUDA OOM，`resource_failure`，b2 已 completed。

| 方法（名义） | seed | 状态 | P | S | online 秒 |
|---|---:|---|---:|---:|---:|
| ER（S0） | 0 | completed | 0.601 | 0.462 | 337 |
| ER（S0） | 1 | completed | 0.609 | 0.454 | 338 |
| ER（S0） | 2 | completed | 0.602 | 0.488 | 341 |
| ER+GEM | 0 | completed | 0.599 | 0.481 | 332 |
| ER+GEM | 1 | completed | 0.605 | 0.469 | 333 |
| ER+GEM | 2 | completed | 0.609 | 0.472 | 333 |
| ER+EWC | 0 | completed | 0.596 | 0.482 | 338 |
| ER+EWC | 1 | resource_failure | — | — | 热身约 11 秒 |
| ER+EWC | 1（b2） | completed | 0.596 | 0.479 | 340 |
| ER+EWC | 2 | completed | 0.618 | 0.458 | 321 |
| ER+GEM+EWC | 0 | completed | 0.597 | 0.473 | 325 |
| ER+GEM+EWC | 1 | completed | 0.615 | 0.447 | 310 |
| ER+GEM+EWC | 2 | completed | 0.626 | 0.453 | 333 |

## CIFAR-100（同配置，插件实际关闭）

2026-10-11 18:16 到 18:37（UTC+8）全部 9 条 completed。P、S、online 为均值±样本标准差，n=3。ER 用 CIFAR-100 的 S0（来自 [G4_STATIC_TIGHT](G4_STATIC_TIGHT.md)）。

| 方法（名义） | P | S | online 秒 |
|---|---:|---:|---:|
| ER（S0） | 0.462±0.013 | 0.552±0.007 | 87.3±0.5 |
| ER+GEM | 0.477±0.016 | 0.536±0.022 | 84±4 |
| ER+EWC | 0.466±0.016 | 0.546±0.017 | 83±3 |
| ER+GEM+EWC | 0.455±0.021 | 0.560±0.018 | 83±4 |

四行之间没有差别，正是因为插件没有运行。对照开发阶段插件真正打开的结果（seed 17）：ER+GEM+EWC 的 P 约 0.16，online 约 320 秒，见 [G5_CLAIMS](G5_CLAIMS.md)。
