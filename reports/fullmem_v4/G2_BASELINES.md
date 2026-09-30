# G2 对照方法接线（开发，未冻结）

更新：2026-09-21。H1 七方法里，S0 / S* / Oracle / O-recon / GSS / GEM / AGEM / LR 已有 mem64g 开发探针。剩下：

| 方法 | 状态 |
|---|---|
| LR | 已完成 seed=17 CIFAR100/NC，见 [G2_LR](G2_LR.md)。其它流在 [G2_WIDE](G2_WIDE.md) 大队列。名称 `LR_reconstructed`。 |
| MAX-A | 大队列里按 A11 重建为 `MAX-A_reconstructed`（batch=1 replay=200 GEM+EWC，Avalanche 0.6.0）。**不是**把 S* 改名。缺论文数字，重建口径已记录。 |
| MAX-P | 大队列里按 A11 重建为 `MAX-P_reconstructed`（batch=256 replay=10）。**不是**把高 batch 格子改名。 |

H2 插件兼容（已在物理审计里）：GEM/AGEM 不作可选开关、不叠第二套 GEM；GSS 选择机制不能当可选插件关掉。

PLAN C 分组件：seed=0 校准保留；CIFAR100 seed=17 插件在 `g2-dev-plugins-seed17`；其它流的插件/GSS/GEM/AGEM/MAX-A/MAX-P 在 `g2-dev-seed17-wide`。旧 seed=0 运行不覆盖。
