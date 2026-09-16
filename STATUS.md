# 当前状态：换机整理，effectiveness_v3 只保留计划与实现

更新：2026-09-16。[A26](docs/decisions/A26.md)：放弃本机 Linux + 双 RTX 5090 上的第三阶段校准、冻结和正式跑数。Git 基线仍为 `ecc004c`；工作树含可迁移的第三阶段入口，尚未提交。

## 接手入口

先读 `AGENTS.md`、`PLAN.md`、[SDD](docs/SDD_effectiveness_v3.md)、[HANDOFF](docs/HANDOFF.md)。独立入口：`python -m orion_repro.stages.effectiveness_v3`。禁止 pressure_* 续跑旧矩阵，禁止把任何 5090 配额/L_cal/frozen_hash 当作可执行协议。

换机后先重建 `orion` 环境并重跑 envcheck / pytest，再从 **G2 开发校准** 开始。解释器用 `ORION_PY` 或 `~/.conda/envs/orion/bin/python`，不要写死旧宿主路径。

## 已完成（可迁移）

- 第三阶段研究设计：S01–S08、153 名额、校准规则、报告口径。见 PLAN / SDD / `experiments/effectiveness_v3/design.yaml`。
- G0/G1 **代码与测试**：独立 stage、runtime 修补（B01–B06）、隔离检查。换机后必须重验，不把本机 pytest/GPU 诊断当正式证据。

## 已放弃（不迁移）

- 本机 G2 开发校准、G3 冻结协议、G4 未完成正式矩阵，以及对应 configs/revisions/runs。
- `configs/effectiveness_v3/<revision>/`、`experiments/effectiveness_v3/revisions/`、`runs/effectiveness_v3*`、本机写入 registry 的 v3 行。

## 正在进行

- 仓库已按换机整理。下一步在**新宿主**上：环境 → G0/G1 重验 → G2 校准 → G3 冻结 → G4 正式矩阵。
- 允许按新硬件对平台参数做小幅适配并记入新决策；不以让 Orion 胜出为选参目标。约 24 小时仅估算，不杀正常训练。

## 历史阶段（保持冻结）

| 阶段 | 状态 | 证据 |
|---|---|---|
| light24_v1，WSL2 / RTX5060 Ti | 96格：90完成+6评价OOM；未证明Orion有效性 | [结果](reports/light24/RESULTS.md) |
| pressure_v2，同原平台 | 正式54格中断未验收 | [收尾](reports/pressure_v2/CLOSEOUT.md) |
| effectiveness_v3 本机 5090 执行 | 按 A26 放弃，不验收、不混入新机 | 无保留跑数 |
