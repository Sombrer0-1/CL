# Orion 论文复现

依据本地 `12_Orion_2605.26473v1.pdf` 重建 Orion，检验持续学习、自适应内存管理与预取的机制、收益和局限。本仓库不是作者官方实现。

**当前状态：effectiveness_v3 的计划、SDD 与实现可迁移；本机 RTX 5090 上的校准和正式跑数已按 A26 放弃。** 换机后按新宿主重建环境，从 G2 重新校准，不要沿用任何 5090 冻结数字。

| 阶段 | 平台 | 状态与结论 |
|---|---|---|
| light24_v1 | WSL2 / RTX 5060 Ti | 96 格已执行：90 完成、6 个评价 OOM；未证明 Orion 有效性 |
| pressure_v2 | WSL2 / RTX 5060 Ti | 54 格中记录 45：30 完成、14 OOM、1 中断；9 未启动，已收尾、未验收 |
| effectiveness_v3 | 设计固定；执行宿主待定 | 八类场景，153 设计名额。实现入口已在树中。5090 跑数不保留 |

## 接手入口

- [STATUS.md](STATUS.md)：当前事实、验证结果与待办。
- [PLAN.md](PLAN.md)：场景、对照、敏感性、矩阵及有效性判据。
- [SDD](docs/SDD_effectiveness_v3.md)：详细架构与实施契约；[HANDOFF](docs/HANDOFF.md)：换机后步骤及简洁 prompt。
- [A26](docs/decisions/A26.md)：放弃 5090 执行证据的决定。
- [实现审计](reports/effectiveness_v3/IMPLEMENTATION_AUDIT.md)：源码修补记录；其中本机 GPU 测量不能当作新宿主证据。
- [仓库地图](docs/REPO_MAP.md)：实现链路、目录职责、历史入口和已知缺口。
- [AGENTS.md](AGENTS.md)：稳定研究约束；§4–5 的 5090 路径是离机快照，换机后重写可执行细节。
- [方法定义](docs/METHODS.md)、[对齐记录](docs/alignment.csv)、[数据说明](docs/DATA.md)：实现依据与假设。
- [结果入口](reports/README.md)、[结论边界](docs/claims_status.md)：两阶段证据与适用范围。

## 环境

换机后在仓库根目录用**新宿主**的 `orion` 解释器：

```bash
ORION_PY="${ORION_PY:-$HOME/.conda/envs/orion/bin/python}"
WANDB_MODE=disabled "$ORION_PY" -m orion_repro.envcheck
WANDB_MODE=disabled "$ORION_PY" -m pytest tests -ra
```

不得使用 mineru、向 base 安装依赖或未经说明升级驱动。根目录 `environment.lock.yml` / `requirements.lock.txt` 是 5090 快照，归档于 [docs/environments/linux_rtx5090/](docs/environments/linux_rtx5090/)；WSL 锁见 [docs/environments/wsl2_rtx5060ti/](docs/environments/wsl2_rtx5060ti/)。新宿主应重新锁定。

`data/raw`、`data/processed` 在本机曾指向 `/mnt/data/zzt/CL/Reproduce-Orion/`；换机后按新数据盘重建链接。`runs/` 不纳入 Git，且本机 effectiveness_v3 跑数已删除。远端 Git 不构成完整证据备份。

## 历史材料

第一阶段：[计划](docs/plans/light24_v1.md)、[协议](docs/protocols/light24_v1.md)、[报告](reports/light24/RESULTS.md)、[冻结索引](reports/light24/stage_manifest.json)，结项标签 `light24-v1-complete`。

第二阶段：[计划](docs/plans/pressure_v2.md)、[架构契约](docs/SDD_pressure_v2.md)、[收尾报告](reports/pressure_v2/CLOSEOUT.md)。`experiments/pressure_v2/formal.yaml` 保留原平台中断矩阵，不能作为待跑队列。

历史配置、生成器输入和结果保留原路径，以免破坏运行溯源。`study` / `light24` / `pressure_*` 是阶段专用入口；执行或重新汇总前先查[仓库地图](docs/REPO_MAP.md)，避免覆写冻结材料。
