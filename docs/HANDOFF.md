# effectiveness_v3：结项后接手

2026-09-19。先读 [STATUS](../STATUS.md)、[MEMORY](MEMORY.md)、[后续选项](plans/after_effectiveness_v3.md)、[CLAIMS](../reports/effectiveness_v3/thor_r2/CLAIMS.md)、[A29](decisions/A29.md)。

**第三阶段已结项。** `thor_r2` 正式 129/129 终态。C02 = 不支持（本协议范围）。不要把宽档跑完说成 Orion 有效。未选定 T0–T3 前，没有训练命令。

判定在 `CLAIMS.md`，不在自动 `RESULTS.md`。不要为填判定重跑 `report`。

只读已有表：

```bash
ls reports/effectiveness_v3/thor_r2/CLAIMS.md \
   reports/effectiveness_v3/thor_r2/means.csv \
   reports/effectiveness_v3/thor_r2/paired.csv
```

环境核验（不替代有效性结论）：

```bash
ORION_PY=/home/zhuzetong/miniconda3/envs/orion/bin/python
WANDB_MODE=disabled "$ORION_PY" -m orion_repro.envcheck
WANDB_MODE=disabled "$ORION_PY" -m pytest tests -ra
```

过程证据在 `thor_r1`（保留）。冻结与正式身份是 **`thor_r2`**。CLI 默认仍是 `r1`，只读 inspect 时必须带 `--revision thor_r2`。禁止 `pressure_*`。另开阶段必须新 revision 名并重校准。
