# effectiveness_v3 接手入口

2026-09-16。[A26](decisions/A26.md) 之后：只迁移第三阶段**计划、SDD 和实现**。Linux + RTX 5090 上的校准、冻结和正式跑数已删除，不要寻找或续跑 `frozen_protocol.json` / `progress.json`。

## 最短阅读

1. `AGENTS.md`、`STATUS.md`、`PLAN.md`
2. [SDD](SDD_effectiveness_v3.md)、[design.yaml](../experiments/effectiveness_v3/design.yaml)、[验收表](effectiveness_v3_acceptance.csv)
3. [A25](decisions/A25.md)（设计）、[A26](decisions/A26.md)（换机放弃执行证据）
4. [实现审计](../reports/effectiveness_v3/IMPLEMENTATION_AUDIT.md)（源码修补，不是有效性结论）

## 当前事实

- 无可执行冻结协议；`design.yaml` 仍为 `designed_not_calibrated`
- 153 设计名额（含条件 H 组 9 格）不变；配额、L_cal、S* 必须在新宿主测量
- G0/G1 代码在 `src/orion_repro/stages/effectiveness_v3/`，换机后重跑测试
- 历史 light24 / pressure_v2 不得续跑或改写
- 根目录环境锁是 5090 快照，另存 [linux_rtx5090](environments/linux_rtx5090/)；新宿主重建 `orion` 并换锁

## 换机后顺序

1. 独立 Conda 环境 `orion`（不要 mineru，不要套用 5090/WSL 锁当新机安装清单）。
2. `ORION_PY=... python -m orion_repro.envcheck` 与 `pytest tests -ra`。
3. 若硬件/驱动与 PLAN 假设冲突，先记平台适配，再改最小必要项。
4. `python -m orion_repro.stages.effectiveness_v3 develop --revision r1 --execute`
5. 审查后再 `freeze` / `emit` / `run` / `report`。失败保留。H 组未建立则标 `scenario_not_realized`，不填 0。

## 接手 prompt

> 请接手 effectiveness_v3。先读 STATUS.md、PLAN.md、docs/SDD_effectiveness_v3.md 和 docs/decisions/A26.md。5090 上的校准和正式跑数已放弃，不要复用任何旧冻结数字。在新宿主重建 orion、重验 G0/G1，再按 PLAN 做 G2 校准并冻结。仅用 orion 环境；禁止覆盖 light24 / pressure_v2 证据。不以正向结论为目标。
