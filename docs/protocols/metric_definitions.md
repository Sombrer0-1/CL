# 指标协议 paper_feedback_diag_initial_v1

- `A[k,i]`：完成第 k 个训练 experience 后，在第 i 个已见测试域上的 accuracy（correct/total）。
- Plasticity `P_diag`：`mean_i A[i,i]`。
- Forgetting `F_initial[i] = A[i,i] - A[k,i]`，允许为负。
- Stability `S_initial`：k=1 时为 1，否则 `1 - mean_{i<k} F_initial[i]`，不裁剪到 [0,1]。
- 同时记录 `avg_seen_accuracy`、`forgetting_max`、`stability_max`。
- `paper_feedback` / `official_test_seen`：官方测试集已见域。仅用于 smoke 接线或冻结后的正式 paper_feedback。
- `development_val_seen`：CIFAR 从官方训练集分层划出的固定验证域（split_seed=17）；CORe50 从官方 train filelist 按 `s*/o*` 整段序列划出的验证域。开发阶段必须使用该源。
- 正式测试曾在早期探索中暴露，后续校准仅使用开发验证集。
- 这是 A05 的操作性定义，不是已确认的作者代码实现。
