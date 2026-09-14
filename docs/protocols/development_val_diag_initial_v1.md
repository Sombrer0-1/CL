# 开发协议 development_val_diag_initial_v1

状态：**开发用，未冻结为正式 protocol v1。** 不得用于选超参后直接当作 paper_feedback 主表。

- 反馈：`development_val_seen`。CIFAR 官方 train 分层 val，split_seed=17，20%。CORe50 官方 train filelist 按 `s*/o*` 整段，split_seed=17。
- 指标：`docs/protocols/metric_definitions.md` 的 P_diag / S_initial。
- 骨干：ResNet-20 option A，SGD 0.01/0.9，1 epoch，FP32 eager。
- CIFAR10 类顺序：`[4, 1, 7, 5, 3, 9, 0, 8, 6, 2]`。
- CORe50：mini 32×32，run=0，object_level，50 类。NC 9 / NI 8 / NIC nicv2_79。
- Endless-Sim：采用独立的 `development_endless_grouped_v2`，见 A22。IC/IL/WC 为 4/5/5 experiences，32×32，数字 patch ID 连续留出加 32-patch embargo；真实时间映射未确认。
- 控制器默认 balanced k=0.25；E05 用 A21 偏好。URGE slot `m_batch=m_frame=1`。
- 正式 `paper_feedback_diag_initial_v1` 在 M6 冻结后才启用官方 test；开发选参不得看该协议结果。
