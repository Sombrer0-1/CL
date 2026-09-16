# 配置用途

当前没有可迁移的第三阶段正式配置。冻结后的 YAML 只在执行宿主生成，见 [PLAN](../PLAN.md) 与 [A26](../docs/decisions/A26.md)。

| 目录 / 文件 | 状态 |
|---|---|
| `effectiveness_v3/` | 仅 README；`dev/` `formal/` 在新宿主 freeze/emit 后出现 |
| `light24/` | 第一阶段已结束，保留原配置 |
| `pressure_v2/` | 第二阶段原平台开发与中断正式配置，不能在新机续入同一矩阵 |
| `development/`、`formal/` | 早期配置；部分仍是生成器输入模板，保留路径 |
| `oracle/` | 旧42格搜索背景，不是当前任务 |
| `smoke*.yaml` | 功能/小样本检查，不能代替正式复现或视为冻结协议 |

按研究语义复用模板时，须重新核对数据域、插件、预算、study/protocol ID 和 provenance，不能只修改文件名。
