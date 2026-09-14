# lr_reconstructed

- 用途：Latent Replay 的 **ResNet-20 适配重建**（A11）。不是 Orion 作者代码，也不是 Avalanche AR1+MobileNet。
- 来源：Pellegrini et al. 2020；Avalanche AR1 / FeatureReplay 行为作参考。
- 本实现：`CifarResNet.lower` = stem+layer1+layer2（32×16×16 float32）；`upper` = layer3+pool+fc。首个 experience 结束后冻结 lower（含 BN eval）；buffer 在 `after_training_exp` 物化当前 experience 的 layer2 激活，容量 `mem_size`，随机子采样。后续 minibatch 在 `before_forward` 拼接 replay latent，并扩展 `mb_y`。不对存储的 latent 反传。
- 配置起点：`mem_size=200`，`replay_batch=new_batch`。
- 禁止：把 raw ER 改名为 LR，或换骨干后仍称同骨干对比。
