# 实验入口状态

当前fullmem_v4仅有 [计划](../PLAN.md)，没有可执行正式矩阵。

- `effectiveness_v3/revisions/thor_r2/`：第三阶段冻结身份和已完成队列；129终态，不续跑。
- `effectiveness_v3/revisions/thor_r1/`：混源码身份的开发过程，不作为冻结来源。
- `effectiveness_v3/design.yaml`：第三阶段153名额旧设计，不是阶段4设计。
- `light24/`、`pressure_v2/`、`reference/`：历史证据与参考。
- `registry.jsonl`：跨阶段原始账本，保留，不作为当前任务清单；统计按阶段/身份过滤。

阶段4使用独立目录、任务ID与闭包。旧生成器存在内部路径和默认revision，不调用它们发射v4。
