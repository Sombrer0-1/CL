# effectiveness_v3：Thor 接手步骤

2026-09-17。先读 [STATUS](../STATUS.md)、[PLAN](../PLAN.md)、[SDD §10](SDD_effectiveness_v3.md#10-thor-准入实施清单2026-09-17)、[A28](decisions/A28.md)。当前硬件/环境见[A27](decisions/A27.md)。历史5090校准已放弃，没有可移植冻结协议。

在仓库根目录明确使用独立环境：

```bash
ORION_PY=/home/zhuzetong/miniconda3/envs/orion/bin/python
WANDB_MODE=disabled "$ORION_PY" -m orion_repro.envcheck
WANDB_MODE=disabled "$ORION_PY" -m pytest tests -ra
"$ORION_PY" -m orion_repro.stages.effectiveness_v3.readiness
```

数据未完成时先运行以下入口，它不会改历史manifest：

```bash
WANDB_MODE=disabled "$ORION_PY" -m orion_repro.stages.effectiveness_v3.prepare
```

准备完成后先检查生成的开发波次，再执行。`--max-waves 1`只限制启动波次数，不超时终止训练；`--plan-only`会写开发配置，不是只读检查。

```bash
"$ORION_PY" -m orion_repro.stages.effectiveness_v3 inspect --revision thor_r1
"$ORION_PY" -m orion_repro.stages.effectiveness_v3 develop --revision thor_r1 --plan-only
WANDB_MODE=disabled "$ORION_PY" -m orion_repro.stages.effectiveness_v3 develop --revision thor_r1 --execute --max-waves 1
```

`inspect`检查受保护文件相对Git HEAD是否变化。目前会指出接手前已有的`reports/pressure_v2/CLOSEOUT.md`迁移说明修改；这是需要保留/审阅的工作树事实，不表示历史数字被本轮重跑。不要删历史文件或取消保护来取得绿灯。

G2完整开发之前/期间完成SDD §10的实现清单；用实际运行估计耗时。预算、L_cal、S*、场景状态全部重新测量。G3前补齐审计与报告并生成有证据的g2_review.json，随后才能freeze→emit→run→report。153为设计分母，条件不成立保留缺口。禁止将pressure_*入口用于本阶段，禁止用测试通过代替Orion有效性结论。
