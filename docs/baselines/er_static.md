# er_static

- 用途：开发、消融和静态对照，**不是**论文 MAX-A。
- 来源：Avalanche 0.6.0 `Replay` + 本仓库 CIFAR ResNet-20。
- 配置起点：`new_batch=16`（PLAN 调试 starter，不是库默认 1），`mem_size=200`（库默认），`replay_batch=new_batch`，SGD lr=0.01 momentum=0.9，单 epoch，无可选 GEM/EWC，无预取。
- 存储：当前使用 Avalanche `ExperienceBalancedBuffer`；不能以逻辑容量代替实际物化字节；正式预算实验需验证 backing storage（A09）。
- 与 MAX-A 的差异：MAX-A reconstructed 使用库默认 `train_mb_size=1` 并启用 GEM+EWC，见 M5 基线卡。
