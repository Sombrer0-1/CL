# 当前状态：第三阶段已结项

更新：2026-09-19。执行设备为 Jetson AGX Thor，解释器必须显式使用 `/home/zhuzetong/miniconda3/envs/orion/bin/python`。决定见 [A27](docs/decisions/A27.md)、[A28](docs/decisions/A28.md)、**[A29](docs/decisions/A29.md)**。长程指针见 [docs/MEMORY.md](docs/MEMORY.md)。

**effectiveness_v3 / `thor_r2` 已结项。** 正式 129 格终态；C 与 H 未执行；人工裁决见 [CLAIMS.md](reports/effectiveness_v3/thor_r2/CLAIMS.md)。**没有 Orion 有效结论。** 主比较 C02 = 不支持（本协议范围）。覆盖缺口与外推仍在，见 [后续选项](docs/plans/after_effectiveness_v3.md)。未选定新轨道前，没有下一条训练命令。

## 已执行（结项摘要）

- 身份：单一 `source_hash=264eb322e88c490030523262362cc66e6a7548acb183003363b79f892a6c49e1`。
- `g2_review` → `freeze` → `emit`。`frozen_hash=9289a239c1f10ff7d32d9b6422b8dc1a467ded2516658a3b3ea668abea798e6c`。设计 153，可执行 129，排除 24（C 15 + H 9）。
- 正式 **129/129**，`formal_run_exit=0`。completed **75**，cuda_oom **54**，无 `implementation_error`。
- 自动表：`reports/effectiveness_v3/thor_r2/RESULTS.md`（判定栏是占位）。裁决以 `CLAIMS.md` 为准。
- light24 的 C01–C08 表未覆盖。

主比较：宽档 O11 相对 S0/S*/R11 的 ΔP 六对全负；紧档 O11 训练 OOM。

## 冻结数字（thor_r2，不是 thor_r1）

| 数据集 | eval_batch | Q_tight | Q_loose | Q_mid | L_cal（秒） | 紧/宽 S* |
|---|---:|---:|---:|---:|---:|---|
| CORe50-NC | 128 | 160 | 320 | 240 | 15.872 | 均为 b16/r2000 |
| SplitCIFAR100 | 128 | 160 | 320 | 240 | 5.842 | 均为 b16/r2000 |

## 明确不做的事

- 不改 `thor_r2` 会进 source_hash 的文件；不覆盖已冻结协议。
- 不执行 C/H，不填 0；不在 240 加一套 A/B；不为了让 Orion 跑完而改 Q_tight。
- 禁止 `pressure_*`；不覆盖 light24 / pressure_v2 / 本阶段 CLAIMS。
- 不把 OOM、URGE、格数或宽档跑完说成 Orion 有效。
- 不要重跑 `report --revision thor_r2` 来「填判定」。

## 下一步

读 [docs/plans/after_effectiveness_v3.md](docs/plans/after_effectiveness_v3.md)。默认先做 T0（只读已有表，不训练）。T1–T3 必须另开阶段与 revision。

## 历史边界

| 阶段 | 状态 |
|---|---|
| light24_v1 / RTX5060 Ti | 已结项，原证据保留 |
| pressure_v2 / RTX5060 Ti | 中断收尾、未验收，不跨机续跑 |
| effectiveness_v3 / RTX5090 | A26 放弃，不混入 Thor |
| effectiveness_v3 / Thor `thor_r1` | G2 过程证据，混 source_hash，不 freeze |
| effectiveness_v3 / Thor `thor_r2` | **第三阶段已结项**；C01–C08 见 CLAIMS.md；无 Orion 有效结论 |
