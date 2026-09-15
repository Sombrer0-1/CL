# Orion 代表性方法复现

依据本地 `12_Orion_2605.26473v1.pdf`，在 WSL2 + RTX 5060 Ti 上验证 Orion 的自适应机制、性能权衡和局限。本仓库是依据论文重建的实现，不是作者官方代码。

第一阶段目标是约24小时以内的代表性复现工作量，**时间只做估算，不是硬截止**。保留完整数据流和主要比较3 seeds，缩减昂贵的全组合搜索。历史71次正式运行保留为背景，新study独立验收。

**light24_v1 已完成**：96 格全部执行（90 completed + 6 个 128MiB 评价阶段 OOM 资源失败），结果与 C01–C08 判定见 [reports/light24/RESULTS.md](reports/light24/RESULTS.md) 与 [docs/claims_status.md](docs/claims_status.md)。

**第一阶段已结项；第二阶段 pressure_v2 的PLAN/SDD与局部修补已完成，尚未正式训练。** 当前[PLAN.md](PLAN.md)聚焦真实GPU资源约束与控制变量，原计划保存在[docs/plans/light24_v1.md](docs/plans/light24_v1.md)。下方light24命令只供第一阶段复核，结项标签仍为 `light24-v1-complete`。

## 当前接手入口

1. [STATUS.md](STATUS.md)：当前状态、已验证内容、下一步。
2. [PLAN.md](PLAN.md)：第二阶段资源实验设计、论文启用场景、开发校准门槛及54格正式比较。
3. [AGENTS.md](AGENTS.md)：稳定环境与研究约束。
4. [当前协议](docs/protocols/light24_v1.md)、[验收表](docs/acceptance.csv)、[数据说明](docs/DATA.md)。

第二阶段设计清单位于 `experiments/pressure_v2/design.yaml`（executable=false），准备验收见 [docs/pressure_v2_acceptance.csv](docs/pressure_v2_acceptance.csv)。预算与阈值须先由开发证据冻结；不直接把设计文件传给训练执行器。

架构/实现契约：[docs/SDD_pressure_v2.md](docs/SDD_pressure_v2.md)。新源码包含缺陷修补，第一阶段应使用结项标签及 `reports/light24/stage_manifest.json` 复核；不要用当前身份汇总覆写其冻结覆盖表。

## 环境

只使用 `/home/admin/miniconda3/envs/orion/bin/python`。依赖见 `requirements.lock.txt`、`environment.lock.yml`；不得使用 mineru 或向 base 安装项目依赖。WSL 复用宿主显卡驱动，不安装 Linux NVIDIA 驱动。

环境重建见 [环境定义](environment.yml) 和锁文件；已有环境先检查，不删除重建。源码可用 `orion/bin/python -m pip install -e . --no-deps` 安装（解释器使用上面的绝对路径）。

## 第一阶段复核与重跑命令（非当前待办）

先确认没有其他项目训练进程：`pgrep -af 'orion_repro.run|orion_repro.study'`，并查看 `nvidia-smi`。新执行器会防止两个study同时启动；不要与旧run_matrix或其他训练并行。

```bash
# 校验与预览，不训练
WANDB_MODE=disabled /home/admin/miniconda3/envs/orion/bin/python -m pytest tests -ra
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.study --matrix experiments/light24/core.yaml --dry-run

# 核心比较和开发搜索：自动复用新study身份匹配的完成结果
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.study --matrix experiments/light24/core.yaml
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.study --matrix experiments/light24/search.yaml

# 12格开发搜索完成后，冻结获选静态配置并正式评价
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.light24 --select
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.study --matrix experiments/light24/selected.yaml

# 核心后按计划完成扩展（亦可分别运行budget/preference/prefetch/sensitivity/endless.yaml）
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.study --matrix experiments/light24/extensions.yaml

# 当前identity覆盖、全部尝试、均值与标准差
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.study_report
```

配置已生成，接手无需重新生成。`python -m orion_repro.light24` 是重建静态配置的入口。`all.yaml` 包含90个预生成格，不包含搜索完成后才产生的6格selected。已有资源失败不会被无限重试；实现错误暂停后需修复。Ctrl-C保留部分产物，下一次从头重跑未完成格；不是完整训练断点恢复。

`runs/light24_progress.json` 记录累计执行耗时与中断；不设置时间杀进程规则。新identity包含源码、实际配置、依赖锁与数据manifest，新增无关配置/文档不使旧格失效。源码修改仍需重新验收受影响配对。

## 目录

| 路径 | 用途 |
|---|---|
| `src/orion_repro/`、`tests/` | 实现与验证 |
| `configs/light24/`、`experiments/light24/` | 已结束阶段的配置与矩阵 |
| `experiments/pressure_v2/` | 第二阶段设计清单，尚非可运行矩阵 |
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
