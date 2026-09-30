# G2 LR reconstructed 开发探针（seed=17）

状态：**开发探针已完成，不是冻结，不是作者 Orion 代码。** 名称 `LR_reconstructed`（A11：ResNet-20 layer2 latent，首经验后冻 stem）。MAX-A / MAX-P 仍未接线。

批次：`g2-dev-lr`。CIFAR100 `2026-09-21T14:14:07Z` completed；CORe50-NC `14:17:06Z` completed。exit 0。

## 成绩（非正式）

| 运行 | 流 | exp | 墙时 s | P | S |
|---|---|---:|---:|---:|---:|
| `fullmem_v4_g2_cifar100_lr_seed17` | CIFAR100 | 10 | 42.4 | 0.395 | 0.636 |
| `fullmem_v4_g2_cifar100_s0_hash_off_seed17` | CIFAR100 S0 | 10 | 71.5 | 0.442 | 0.575 |
| `fullmem_v4_g2_core50_nc_lr_seed17` | CORe50-NC | 9 | 171.5 | 0.622 | 0.443 |
| `fullmem_v4_g2_core50_nc_s0` | NC S0（seed=0） | 9 | 225.7 | 0.641 | 0.460 |

P/S 差和墙时差都不是冻结比较。NC S0 还是 seed=0，不能直接当配对。

## 接线核对

| 项 | 证据 |
|---|---|
| 算法 | `algorithm.base: lr`，插件序含 `LatentReplayPlugin` |
| 存储 | `replay.policy: random_latent`，`representation: layer2_float32`，容量 200 |
| 占用 | 经验 0 准入时 `latent_occupancy=0`；经验 1 起一直是 200 |
| 驻留 | `LatentReplayPlugin` `plugin_state_bytes=6555200`（约 6.25 MiB；200×32776 B）全程不变 |
| 冻 stem | 源码 `conv1/bn1/layer1/layer2` 在首经验后 `requires_grad=False`。CIFAR100 `learning_s` 经验 0 ≈5.18 s，之后 ≈2.2–2.5 s；S0 对照全程 ≈5.2–6.0 s。NC 经验 0 因抽特征更慢（26 s vs S0 21 s），之后约 6–7 s vs S0 12–17 s |
| 辅助访问 | 每经验 `auxiliary_visits=256`，scope=`latent_cache_feature_forward`。不是 GEM/EWC |
| 控制器 | `skipped / disabled` |

## 占用

CIFAR100 LR：`min system_available ≈ 62.36 GiB`，RSS 峰 ≈2.36 GiB，GPU reserved 峰 ≈132 MiB。NC 类似。压力场景仍未建立。不要放大模型。

## 明确不是

- 不是 G3 冻结，不是正式主比较。
- 不能把墙时下降写成论文 LR 已复现优势。
- MAX-A / MAX-P 不能从 S* 或高 batch 改名冒充。
