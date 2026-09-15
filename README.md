# Orion 代表性方法复现

依据本地 `12_Orion_2605.26473v1.pdf` 验证 Orion 的自适应机制、性能权衡和局限。本仓库是依据论文重建的实现，不是作者官方代码。

第一、二阶段实验在 **WSL2 + RTX 5060 Ti** 上执行。仓库已迁到 Linux + 双 RTX 5090；**后续工作在本机新环境另开阶段**，不得把未完成的 pressure_v2 正式格续跑后与原 5060 Ti 结果混为同一矩阵。

**light24_v1 已完成**：96 格全部执行（90 completed + 6 个 128MiB 评价阶段 OOM 资源失败），结果与 C01–C08 判定见 [reports/light24/RESULTS.md](reports/light24/RESULTS.md) 与 [docs/claims_status.md](docs/claims_status.md)。结项标签 `light24-v1-complete`。

**pressure_v2 已收尾、未验收**：实现、开发校准与冻结已完成；正式 54 格记录 45（30 completed + 14 cuda_oom + 1 interrupted），其余 9 格未启动。见 [STATUS.md](STATUS.md) 与 [reports/pressure_v2/CLOSEOUT.md](reports/pressure_v2/CLOSEOUT.md)。原计划保存在 [docs/plans/light24_v1.md](docs/plans/light24_v1.md)。

## 当前接手入口

1. [STATUS.md](STATUS.md)：当前状态、已验证内容、下一步。
2. [PLAN.md](PLAN.md)：第二阶段资源实验设计、论文启用场景、开发校准门槛及54格正式比较。
3. [AGENTS.md](AGENTS.md)：稳定环境与研究约束。
4. [当前协议](docs/protocols/light24_v1.md)、[验收表](docs/acceptance.csv)、[数据说明](docs/DATA.md)。

第二阶段设计与冻结协议位于 `experiments/pressure_v2/`。正式矩阵 `formal.yaml` 保留为原平台中断记录，**不是本机待执行队列**。准备验收见 [docs/pressure_v2_acceptance.csv](docs/pressure_v2_acceptance.csv)。不直接把 `design.yaml` 传给训练执行器。

架构/实现契约：[docs/SDD_pressure_v2.md](docs/SDD_pressure_v2.md)。新源码包含缺陷修补，第一阶段应使用结项标签及 `reports/light24/stage_manifest.json` 复核；不要用当前身份汇总覆写其冻结覆盖表。

## 环境

当前工作区：`/home/zhuzetong/research/CL/Reproduce-Orion`。解释器：`/home/zhuzetong/.conda/envs/orion/bin/python`（Python 3.11.16，`torch==2.11.0+cu128`）。共享 Conda CLI 为 `/opt/miniconda3`（base 只读）；包缓存在 `~/.conda/pkgs`。`data/raw`、`data/processed`、`runs` 经符号链接指向 `/mnt/data/zzt/CL/Reproduce-Orion/`。不得使用 mineru 或向 base 安装项目依赖。

重建步骤见 [environment.yml](environment.yml)；精确清单为本机 [requirements.lock.txt](requirements.lock.txt)、[environment.lock.yml](environment.lock.yml)。原 WSL/5060 Ti 锁文件在 [docs/environments/wsl2_rtx5060ti/](docs/environments/wsl2_rtx5060ti/)，版本选择见 [A24](docs/decisions/A24.md)。已有 `orion` 环境先检查，不删除重建。源码：`conda run -n orion python -m pip install -e . --no-deps`。

本机无 Clash TUN 时，访问 PyTorch 官方轮子可导出 `http_proxy`/`https_proxy=http://127.0.0.1:7890`。不要改系统驱动或 Toolkit。

验收：

```bash
WANDB_MODE=disabled /home/zhuzetong/.conda/envs/orion/bin/python -m orion_repro.envcheck
WANDB_MODE=disabled /home/zhuzetong/.conda/envs/orion/bin/python -m pytest tests -ra
```

## 第一阶段复核与重跑命令（非当前待办）

以下命令用于确认 light24 配置能在**本机解释器**上解析；**不要**用新环境身份覆盖 `reports/light24/`。原 WSL 解释器是 `/home/admin/miniconda3/envs/orion/bin/python`，本机结果不能并入该阶段矩阵。

```bash
ORION_PY=/home/zhuzetong/.conda/envs/orion/bin/python
# 校验与预览，不训练（本机解释器；结果身份与原阶段不同）
WANDB_MODE=disabled "$ORION_PY" -m pytest tests -ra
WANDB_MODE=disabled "$ORION_PY" -m orion_repro.study --matrix experiments/light24/core.yaml --dry-run

# 下列矩阵是第一阶段队列，不是本机待执行任务
# WANDB_MODE=disabled "$ORION_PY" -m orion_repro.study --matrix experiments/light24/core.yaml
# WANDB_MODE=disabled "$ORION_PY" -m orion_repro.study --matrix experiments/light24/search.yaml
# WANDB_MODE=disabled "$ORION_PY" -m orion_repro.light24 --select
# WANDB_MODE=disabled "$ORION_PY" -m orion_repro.study --matrix experiments/light24/selected.yaml
# WANDB_MODE=disabled "$ORION_PY" -m orion_repro.study --matrix experiments/light24/extensions.yaml
# WANDB_MODE=disabled "$ORION_PY" -m orion_repro.study_report
```

原平台解释器为 `/home/admin/miniconda3/envs/orion/bin/python`，仅作历史对照。配置已生成，接手无需重新生成。`python -m orion_repro.light24` 是重建静态配置的入口。`all.yaml` 包含90个预生成格，不包含搜索完成后才产生的6格selected。已有资源失败不会被无限重试；实现错误暂停后需修复。Ctrl-C保留部分产物，下一次从头重跑未完成格；不是完整训练断点恢复。

`runs/light24_progress.json` 记录累计执行耗时与中断；不设置时间杀进程规则。新identity包含源码、实际配置、依赖锁与数据manifest，新增无关配置/文档不使旧格失效。源码修改仍需重新验收受影响配对。

## 目录

| 路径 | 用途 |
|---|---|
| `src/orion_repro/`、`tests/` | 实现与验证 |
| `configs/light24/`、`experiments/light24/` | 已结束阶段的配置与矩阵 |
| `configs/pressure_v2/`、`experiments/pressure_v2/` | 第二阶段配置、冻结协议与中断矩阵（未验收） |
| `reports/pressure_v2/` | 第二阶段准备摘要与收尾记录 |
| `configs/formal/`、`configs/development/` | 既有配置与新生成器输入模板 |
| `experiments/reference/` | 历史矩阵，仅供追溯，不是待恢复队列 |
| `docs/` | 方法、对齐依据、协议与验收 |
| `reports/light24/` | 第一阶段结果、尝试清单和统计 |
| `reports/` 其他表 | 历史结果，见[结果说明](reports/README.md) |
| `data/`、`runs/` | 数据与不可伪造的原始运行证据 |
| PDF、论文Markdown、`images/` | 原文与辅助材料 |

当前范围不覆盖全部原论文实验、Jetson能耗、原设备耗时或机器人闭环；结果以证据判定，不承诺正向结论。

## 版本基线

当前 light24_v1 代码、测试、配置、协议和必要 manifests 已纳入 Git。初始提交名称为 `Establish light24 reproduction baseline`；接手先执行 `git status --short` 和 `git log -1 --oneline`，确认当前版本及本地改动。

后续按逻辑单元提交实现或协议变更，提交前检查 diff 并执行相关验证。每次实验的源码哈希和配置仍保留；哈希用于核对身份，Git 保存源码历史，二者用途不同。不要把旧运行的哈希当作可以恢复源码的完整备份。

`data/raw/`、`data/processed/`、`runs/`、权重和缓存不纳入 Git；这些内容仍保存在本机，重要原始结果需另外备份。远端仓库为 `git@github.com:Sombrer0-1/CL.git`（origin），当前分支为 master。远端仅保存 Git 纳入的源码、配置、文档和必要研究记录，不包含原始数据、runs 或模型文件；这些本地材料仍需单独备份。
