# max_a_reconstructed

- 用途：论文 MAX-A 的**具名重建**，不是作者配置快照。
- 论文依据：§5.1 文字，固定小 batch、默认 replay，并启用高级优化插件。论文未给出数值。
- 采用值（A11，`source_inferred`）：Avalanche 0.6.0 默认 `train_mb_size=1`、`mem_size=200`；在 ER `ReplayPlugin` 上叠加 `GEMPlugin(patterns_per_exp=50, memory_strength=0.5)` 与 `EWCPlugin(ewc_lambda=100)`，训练开始即开启（`optional_start_enabled: true`）。
- 不自适应、不预取。EWC 在每个 experience 结束后对当前训练集额外遍历以估计 Fisher。
- 禁止：把 `er_static`（batch 16、无 GEM/EWC）叫作 MAX-A。
