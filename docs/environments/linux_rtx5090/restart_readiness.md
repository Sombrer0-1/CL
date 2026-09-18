# 迁移后仓库整理验收（5090 宿主快照，已归档）

**不是当前 Thor 检查清单。** 当前环境见 [A27](../../decisions/A27.md) 与仓库根 `reports/envcheck.json`。

本记录早于 A26 换机整理。源码修补见 [实现审计](../../../reports/effectiveness_v3/IMPLEMENTATION_AUDIT.md)。正文保留 5090 当时的整理事实。

日期：2026-09-16。检查基线：`ecc004c`，本轮变更为文档整理，未修改训练源码、YAML 配置、依赖锁或历史结果；未启动新阶段训练。本记录只证明仓库整理及环境基础检查，不是下一阶段正式实验验收。

## 整理结果

- 根目录 README、STATUS、PLAN 分别负责导航、当前事实和下一阶段准备；移除重复历史命令和过期活动矩阵说明。
- 原根目录 pressure_v2 计划归档到 [docs/plans/pressure_v2.md](../../plans/pressure_v2.md)，正文保留，仅调整相对链接。
- 新增[仓库地图](../../REPO_MAP.md)及 configs / experiments 目录索引，解释生成器模板依赖和历史材料用途。历史配置原路径保留。
- 修正 SDD 中“仅局部修补已实现”的过时状态；METHODS 对齐当前插件保留状态行为、可选扩展范围及不设训练硬截止约定。
- 修正 AGENTS / README 的存储描述：两个 data 目录为符号链接，runs 实为普通目录。未搬动或删除原始证据。
- 新阶段入口隔离、端到端验收、开发校准和冻结仍待开展，见 [PLAN](../../../PLAN.md)。

## 本次实际验证

| 检查 | 结果 | 边界 |
|---|---|---|
| orion 环境检查 | Python 3.11.16 / torch 2.11.0+cu128 / torchvision 0.26.0+cu128 / Avalanche 0.6.0；sm_120；cuda:0 前向、反向、优化器更新流程通过 | 小型环境检查，不是完整训练资源校准 |
| 全套现有测试 | **121 passed，9 warnings，5.31s** | 告警来自 Avalanche 无 logger 和 replay update 弃用提示；不证明方法收益 |
| 当前与 WSL 锁文件 | docs/environment_hashes.txt 的 4 项 SHA256 全部一致 | 未重装依赖；历史锁保留 |
| 配置数据路径 | 381 个 YAML 可解析；其中 dataset.root / split_dir / split_manifest 引用路径均存在 | 路径检查，未重新哈希全量数据或验证每个样本 |
| 第一阶段索引 | stage_manifest 的 96 个运行目录均存在，90 completed + 6 cuda_oom | 未重新计算历史指标 |
| 第二阶段索引 | progress 的 85 条记录引用目录均存在；开发25完成/15 OOM，正式30完成/14 OOM/1中断 | 9 个未启动正式格仍未启动 |
| 历史运行进度 | 两个 runs/*_progress.json 的 active 均为 null | 检查时状态，不保证未来无并发任务 |
| runs 双份内容 | 工作区与 /mnt/data/zzt/CL/Reproduce-Orion/runs 各5814个文件，相对路径及逐文件 SHA256 全部相同 | 只是当时一致；没有自动同步，未删除任一副本 |
| 历史证据保护 | 整理前后对照：源码、测试、两阶段报告、原 YAML/JSON 配置、实验记录、数据 manifests 内容未改变 | configs/formal/README.md、experiments/reference/README.md 属有意更新的目录说明 |
| 文档检查 | 本轮新增/修改文档的本地 Markdown 链接存在；git diff --check 通过 | 历史文档中的纯文本路径/外部 URL 未作全量验证 |

本机检查时 GPU0 / GPU1 分别占用约348 / 24 MiB（各32607 MiB），驱动570.133.20；RAM 可用约105 GiB，swap 已用约49 MiB。根分区可用约849 GiB，/mnt/data 可用约13 TiB。以上是资源快照，不是运行预算或未来可用量保证。

环境与测试复核命令（仓库根目录）：

```bash
ORION_PY=/home/zhuzetong/.conda/envs/orion/bin/python
WANDB_MODE=disabled "$ORION_PY" -m orion_repro.envcheck
WANDB_MODE=disabled "$ORION_PY" -m pytest tests -ra
```

本次不重新汇总 light24 / pressure_v2，也不运行旧队列。特别是 `pressure_*` 内仍绑定旧 ID、冻结文件及输出路径，独立新阶段入口必须先实现并验证。
