# 配置用途

当前阶段为fullmem_v4，尚无冻结或正式可执行配置；见 [PLAN](../PLAN.md)。

- `effectiveness_v3/thor_r2/`：已结项第三阶段的开发/正式配置，保留溯源，不用于阶段4。
- `effectiveness_v3/thor_r1/`、`thor_readiness/`：第三阶段过程/准入证据，不能混作冻结配置。
- `light24/`、`pressure_v2/`：历史阶段；后者中断未验收。
- `development/`、`formal/`、`oracle/`：旧模板/探索配置，部分仍被代码引用，不清空。
- `smoke*.yaml`：功能检查，不是正式复现。

阶段4另建fullmem_v4目录和身份；不复制旧配额或L_cal冒充校准。所有现存执行配置均不作为当前待跑队列。
