# 结果入口

已结束阶段：light24_v1。第二阶段 pressure_v2 已中断收尾、未验收。范围见 [PLAN](../PLAN.md)，实时工作状态见 [STATUS](../STATUS.md)。

## 第一阶段交付

以下为已冻结第一阶段交付。源码已经修补，不应直接运行当前study_report覆写旧阶段结果；复核使用结项标签及 `light24/stage_manifest.json`：

- `light24/coverage.csv`：当前源码与配置身份逐格覆盖，未跑/失败不填零。
- `light24/attempts.csv`：新study所有可读取的运行尝试，保留失败和中断。
- `light24/means.csv`：按阶段与变体报告均值、样本标准差和已完成/计划数。
- `light24/progress.json`：已执行96格，90完成、6个评价阶段OOM。
- `light24/selection.json`：已冻结的开发搜索静态配置选择依据。
- `light24/readiness.json`：本轮代码与执行入口验证摘要，不是科学结论。

## 历史结果

根目录的 `formal_context_table.csv`、`formal_context_means.csv` 及其他 `formal_*.csv` 是旧协议结果；`results.csv` 是更早开发/校准清单。它们保留为研究背景，不能直接和新study混合成均值或配对加速。旧Oracle42格不是当前有限搜索。

旧 `tabulate_context.py`、`pair_speedup.py`、`extract_*` 等仅汇总非light24运行；当前study使用上面的新入口。原始记录仍在 `runs/<run_id>/`，保留失败、低准确率、GSS崩塌和预取负收益。

`validation.json` 是有日期的历史验证摘要，不代表当前全部代码已验收。`legacy_includes_setup` 与 `online_loop_v2` 不混算。Endless grouped辅助结果不混称官方测试结果。来源、原始矩阵和资源轨迹是后续图表的依据；尚无结果不代表零值。

结项报告为 `light24/RESULTS.md`；OOM原始堆栈阶段索引见 `light24/budget_failure_stages.csv`。第一阶段已结束。

## 第二阶段收尾（未验收）

- `pressure_v2/CLOSEOUT.md`：正式 54 格中断说明、不得跨机续跑。
- `pressure_v2/progress.json`、`pressure_v2/attempts.csv`：开发+正式记录摘要；原始 runs 仍在 `runs/`。
- `pressure_v2/readiness.json`：实现准备摘要，不是正式科学结论。

