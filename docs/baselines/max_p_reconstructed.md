# max_p_reconstructed

- 用途：论文 MAX-P 的**具名重建**。
- 论文依据：§5.1 固定大 batch、小 replay 的高效静态配置。论文未给出数值。
- 采用值（A11，开发吞吐起点）：`new_batch=256`、`replay_capacity=10`、`replay_batch=256`，ER，无 GEM/EWC，无预取，不自适应。
- 256 取自 Oracle batch 网格中偏大的可执行点；10 取自 Oracle replay 网格最小档。这是本平台重建，不是作者实验值。
- 与 Orion 比较时必须使用同一加载后端；不能只给 MAX-P 关预取/降学习率。
