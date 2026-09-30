# 结果与证据入口

当前阶段fullmem_v4尚无正式结果。见 [STATUS](../STATUS.md) 与 [PLAN](../PLAN.md)。

| 路径 | 定位 |
|---|---|
| `host_feasibility/20260920/` | CPU/cgroup/CUDA限制能力实测；不是阶段4正式结果 |
| `effectiveness_v3/thor_r2/CLAIMS.md` | 第三阶段最终人工裁决：主比较不支持Orion优势 |
| `effectiveness_v3/thor_r2/` | 正式表、日志及冻结结果；RESULTS自动判定占位不能取代CLAIMS |
| `effectiveness_v3/thor_r1/`、`readiness/` | 第三阶段过程与迁移证据，非当前准入 |
| `light24/RESULTS.md` | 第一阶段结项 |
| `pressure_v2/CLOSEOUT.md` | 第二阶段中断收尾、未验收 |
| `restart_readiness.md` | 2026-09-17历史快照，不作为当前下一步 |

根目录的 `formal_*.csv`、`results.csv`、`oracle_*.csv/json`、`validation.json`、`envcheck.json`、`registry_summary.json` 和提取脚本为早期探索/检查记录，保留原路径避免破坏引用，不代表当前配置或结论。不要运行这些旧汇总脚本覆盖历史报告。

原始日志与失败位于Git忽略的runs目录，须单独备份。Thor第三阶段证据确实存在；“effectiveness_v3已删除”仅指A26放弃的旧5090运行，不适用于Thor。不同阶段、平台、预算和时间口径不得合并。
