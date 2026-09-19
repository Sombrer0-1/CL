# thor_r2 G2 审查笔记（2026-09-18）

本文件是冻结前审查记录，不是 Orion 有效性结论。`pass` 表示该项已审核并附证据，允许负结果与场景未建立。

## 身份

- revision：`thor_r2`，campaign：`identity_recalibrate`
- 当前树 `source_hash`：`264eb322e88c490030523262362cc66e6a7548acb183003363b79f892a6c49e1`
- probe_manifest 61 行（含 host）；校准用行单一 source_hash，与当前树一致
- evidence_closure：`n_ok=61 / n_mismatch=0`
- `thor_r1` 95 条保留为过程证据，未写入本冻结数字
- 无 `implementation_error`

## 开发覆盖（决定 Q/L/S* 的探针）

两数据集 eval_batch、S0×2、配额扫描、Q_loose 验证、L_cal×2、紧/宽各 6 格静态搜索均已终态。跳过 plugin_cost / plugin_loose / 完整 control_2x2 / 敏感性 / AGEM / DYN 六候选。

只读 `calibrate()`：

| 数据集 | eval_batch | Q_tight | Q_loose | Q_mid | L_cal（秒） | 紧 S* | 宽 S* |
|---|---:|---:|---:|---:|---:|---|---|
| CORe50-NC | 128 | 160 | 320 | 240 | 15.872 | b16/r2000（2/6 可行） | b16/r2000（4/6 可行） |
| SplitCIFAR100 | 128 | 160 | 320 | 240 | 5.842 | b16/r2000（2/6 可行） | b16/r2000（4/6 可行） |

紧档 b64/b256 均 cuda_oom；宽档 b256 均 cuda_oom。Q_tight=160 是 S0 reserved≈152 MiB 落入 80–95% 网格的结果，不是为了让 Orion 跑完而改档。DYN 无独立 S*。

## 资源场景

- S01：NC O11@160 身份盖章 `cuda_oom`（trained=2/9，failure_phase=training）→ `failed_to_construct`。A 组仍按 153 执行，不因此删格。
- S04：本 revision 放弃建设（A29）。身份盖章 DYN R=4 MiB@160 `cuda_oom`（trained=2/9）；窗口 `mutation_round_failed`，无可观测反馈区间。C 组 15 格 `scenario_not_realized`。
- S07：io_off wait_ratio≈0.520，cpu%≈9.49；io_on 完成。`realized`（供给瓶颈可测，不是预取净加速或 Orion 有效）。
- host / H 组：`unavailable`（无授权子 cgroup）。
- 240：只服务 G 组 quota_mid，不加一套 A/B。

## 报告

开发报告表已写出（attempts/paired/coverage 等）。判定栏在正式结果前保持未验证。配对表不挑最快 attempt。

## 完整性

`pytest tests` 在冻结前重跑通过。混 `source_hash` 拒绝与缺 g2_review 拒绝已有回归测试。实现错误未当作资源失败。
