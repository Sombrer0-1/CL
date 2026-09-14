# 结果入口

已结束阶段：light24_v1，范围见 [PLAN](../PLAN.md)，实时工作状态见 [STATUS](../STATUS.md)。

## 第一阶段交付

运行 `/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.study_report` 更新：

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

结项报告为 `light24/RESULTS.md`；OOM原始堆栈阶段索引见 `light24/budget_failure_stages.csv`。本阶段已结束，下一阶段尚未立项。
