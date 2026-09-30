# G2 H5c 插件生命周期诊断（开发，未冻结）

批次：`g2-dev-h5c-plugin-lifecycle`。CIFAR100 ER+GEM+EWC，seed=17，`mem=16G`。脚本化经验日程 `AAA DD AA DDD`。**不是 O-recon，不是正式 H5。**

## pause-keep-state（默认解释，已测）

`g2-dev-cifar100-h5c-pause-keep-seed17-mem16g-b2` completed。墙时 139.3 s，P=0.324，S=0.714。

| 经验 | 模式 | GEM bytes | EWC bytes | enabled |
|---:|---|---:|---:|---|
| 0–2 | advanced | 246080 → 738240 | 2.20 MiB → 6.31 MiB | true |
| 3–4 | default | **738240 不变** | **6.31 MiB 不变** | false |
| 5–6 | advanced | 984320 → 1230400 | 8.41 → 10.52 MiB | true |
| 7–9 | default | **1230400 不变** | **10.52 MiB 不变** | false |

关掉计算后张量仍驻留；再打开时从保留状态继续累加，不是从零重建。这就是对齐表 `plugin_pause` 的 pause-keep-state 默认解释。不能把它写成「停插件就释放内存」。

原 attempt `...-pause-keep-...` 因 spec 不允许脚本化 `plugin_policy` 在启动前失败，attempt 保留。

## release-rebuild（不可并入 O-recon）

`g2-dev-cifar100-h5c-release-rebuild-seed17-mem16g`：经验 0–2 之后清空 EWC `saved_params`，再打开时 Avalanche 0.6.0 `EWCPlugin.before_backward` 对缺失 experience key 抛 `KeyError: 0`。attempt 保留，`implementation_error`。

设计契约：若无法给出符合算法的重建语义，记该变体不可实现，不并入 O-recon。朴素 `dict.clear()` 不是合法 EWC/GEM 重建。正式 H5c 的 release-rebuild 仍未建立。

## 仍不做

- 不把脚本化日程写成 URGE 在真实压力下停插件
- 不把 16G / 本诊断写成 H5 正式块
- `kind=train` 仍拒绝
