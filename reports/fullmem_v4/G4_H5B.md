# H5b：NC 上固定 batch/replay 的插件拆分

更新：2026-10-11 18:20（UTC+8）。`mem=5G`。只覆盖 NC，batch 16，replay 200，控制器关闭。ER 复用已完成的 S0。不是 CIFAR-100 的结果，也不是第二档预算。CIFAR-100 的同一组已另行发射。

P = `p_diag`，S = `s_initial`。

| 方法 | seed | 状态 | P | S | online 秒 |
|---|---:|---|---:|---:|---:|
| ER（S0） | 0 | completed | 0.601 | 0.462 | 337 |
| ER（S0） | 1 | completed | 0.609 | 0.454 | 338 |
| ER（S0） | 2 | completed | 0.602 | 0.488 | 341 |
| ER+GEM | 0 | completed | 0.599 | 0.481 | 332 |
| ER+GEM | 1 | completed | 0.605 | 0.469 | 333 |
| ER+GEM | 2 | completed | 0.609 | 0.472 | 333 |
| ER+EWC | 0 | completed | 0.596 | 0.482 | 338 |
| ER+EWC | 1 | resource_failure | — | — | 热身约 11 秒 |
| ER+EWC | 2 | completed | 0.618 | 0.458 | 321 |
| ER+GEM+EWC | 0 | completed | 0.597 | 0.473 | 325 |
| ER+GEM+EWC | 1 | completed | 0.615 | 0.447 | 310 |
| ER+GEM+EWC | 2 | completed | 0.626 | 0.453 | 333 |

在这个固定的 b16/r200 上，加上 GEM、EWC 或两者，P 仍在 0.60 附近，online 仍在 330 秒附近，和单独的 ER 没有拉开。ER+EWC seed 1 的 b2 已另发。
