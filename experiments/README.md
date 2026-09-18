# 实验入口状态

当前无活动的正式矩阵。下一步见 [PLAN](../PLAN.md)。当前执行宿主是 Jetson AGX Thor（[A27](../docs/decisions/A27.md)）。5090 上的 effectiveness_v3 冻结与跑数已按 [A26](../docs/decisions/A26.md) 删除。

- `effectiveness_v3/design.yaml`：新阶段不可执行设计，153 名额及校准规则；没有可迁移的冻结参数。
- `effectiveness_v3/revisions/`：仅在执行宿主生成；换机后重建。
- `light24/`：第一阶段已结项，矩阵与选择证据保留。
- `pressure_v2/`：第二阶段原平台冻结设计、校准和54格中断矩阵；未验收，不续跑。
- `reference/`：更早方案与队列参考，不批量恢复。
- `registry.jsonl`：跨时期运行注册账本。本机写入的 effectiveness_v3 行已撤回；汇总必须按阶段、环境与协议身份筛选。

新阶段使用独立目录与 ID。现有 pressure_* 代码的内部固定路径必须先隔离，不能仅复制 YAML 后直接执行。
