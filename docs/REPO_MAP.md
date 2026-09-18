# 仓库地图与接手说明

2026-09-16 整理。当前状态以 [STATUS](../STATUS.md) 为准，下一阶段任务以 [PLAN](../PLAN.md) 为准；本文描述代码和材料之间的关系。新阶段架构见 [SDD_effectiveness_v3.md](SDD_effectiveness_v3.md)。当前执行宿主是 Jetson AGX Thor（[A27](decisions/A27.md)）。源码修补见[实现审计](../reports/effectiveness_v3/IMPLEMENTATION_AUDIT.md)。

## 实现链路

```text
阶段设计 / 开发校准 → 配置与冻结协议 → 阶段执行器（独立子进程）
  → run.py / runner.loop.run_from_spec
  → 数据流 + ResNet20 + Avalanche strategy + replay / 可选插件 / 预取
  → 每 experience：训练一次 → 评价 → URGE → 下一 experience 配置
  → runs 中原始产物 + experiments/registry.jsonl → 阶段报告
```

| 代码 | 责任与边界 |
|---|---|
| `benchmarks/`、`prepare_*.py` | CIFAR、CORe50、Endless 准备与开发划分；真实基准数据，不用合成检查替代正式实验 |
| `models/resnet20.py` | 骨干模型 |
| `strategies/builder.py`、`capacity.py`、`toggles.py` | Avalanche ER/GEM/AGEM/GSS、replay 调整及可选插件；无插件时拒绝 advanced；停用插件保留历史状态 |
| `control/urge.py`、`controller.py` | 四项 sigmoid、衰减阈值、连续预算与离散映射、模式建议；低于阈值仍有收缩更新 |
| `prefetch/` | 新数据与 replay 的有界供给、顺序与消费记录 |
| `memory/` | host/GPU 观测、allocator 配额、分阶段峰值、资源预留；各自范围不能相加冒充共享内存 |
| `evaluation/` | 反馈域、准确率矩阵及 P/S 定义；paper_feedback 包含官方测试反馈 |
| `runner/loop.py`、`spec.py`、`artifacts.py`、`failures.py` | 验证配置、训练/评价/控制时序、产物、失败阶段及上下文 |
| `provenance.py`、`runner/reuse.py`、`locks.py` | 源码/环境/数据身份、复用判定、执行互斥；旧结果不能仅因配置名相同就复用 |
| `tests/` | 公式、指标、replay、插件生命周期、顺序、配额、执行器和冻结协议检查；不证明方法收益 |

## 阶段入口与依赖

| 入口 | 绑定对象 | 当前使用方式 |
|---|---|---|
| `light24.py`、`study.py`、`study_report.py` | 第一阶段配置、队列、报告 | 历史复核；不要以当前源码身份重新生成冻结覆盖表 |
| `pressure_calibrate.py`、`pressure_protocol.py` | 第二阶段开发、冻结、54格发射 | 历史实现，可参考；默认写入 pressure_v2 |
| `pressure_study.py`、`pressure_report.py` | 第二阶段进度、执行、报告 | 历史入口；新阶段需先隔离内部 ID 与固定路径 |
| `run.py` | 单配置底层执行 | 可复用基础，但阶段身份、产物路径和配置须由新入口明确管理 |
| `run_matrix.py`、`emit_formal.py`、`oracle_grid.py`、`reports/*.py` | 早期探索/全覆盖方案 | 追溯用，不批量恢复旧队列 |

`configs/development/` 与 `configs/formal/e03/` 仍被阶段生成器读取为模板，不能按“旧文件”整体删除或搬走。`experiments/registry.jsonl` 和配置内路径参与溯源。整理保留这些路径，使用目录说明区分用途。

## 材料分层

| 材料 | 定位 |
|---|---|
| 根目录 README / STATUS / PLAN | 导航 / 当前事实 / 下一阶段准备，避免重复展开历史执行命令 |
| `docs/SDD_effectiveness_v3.md`、`docs/HANDOFF.md`、`experiments/effectiveness_v3/design.yaml` | 第三阶段架构/接手/不可执行设计；冻结协议在本机生成 |
| `AGENTS.md` | 稳定约束，用户后续明确指示优先 |
| `docs/METHODS.md`、`alignment.csv`、`decisions/` | 定义、来源及假设；旧 E 编号是历史技术背景，不代表全部实现或待执行任务 |
| `docs/plans/`、`docs/protocols/light24_v1.md`、`docs/SDD_pressure_v2.md` | 阶段计划与契约；不当作本机冻结协议 |
| `configs/light24/`、`experiments/light24/`、`reports/light24/` | 第一阶段完整证据，标签 light24-v1-complete |
| `configs/pressure_v2/`、`experiments/pressure_v2/`、`reports/pressure_v2/` | 第二阶段原平台开发/冻结/中断记录，未验收 |
| `configs/formal/`、`configs/oracle/`、`experiments/reference/`、`reports/` 中旧表 | 更早探索结果和模板，不能混入阶段均值 |
| `data/manifests/` | 已纳入 Git 的数据版本和划分证据，旧绝对路径保留为原始溯源 |
| `data/raw`、`data/processed` | 本机普通目录；未纳入 Git |
| `runs/` | 工作区普通目录，原始日志/配置/矩阵/失败等；未纳入 Git |
| `requirements.lock.txt`、`environment.lock.yml` | 本机 Thor 锁，归档于 docs/environments/jetson_agx_thor/；历史宿主见同目录其它子目录 |
| PDF、论文 Markdown、`images/` | 原文及辅助材料；公式与图表以 PDF 为准 |

## 已知限制

- 第一阶段完整配置未安装可切换插件，且核心轨迹未进 advanced；保留 P/S，不将旧字符串模式检查算作真实插件验收。
- 第二阶段有真实插件路径和资源失败记录，但未完成正式比较；迁移后需重新验证开发可行区。
- 暂停插件计算不释放 GEM/EWC 状态；这可能限制内存自适应，必须作为实测因素。
- 更快 GPU 会改变以秒计的延迟反馈，其它宿主的 L_cal 和预算不能直接移植；不能为让 Orion 胜出而选参。
- GPU allocator 限额仅约束对应分配器范围，不是 cgroup 或板级硬限额。host 探测与端到端校准须在本机重做。
- 新运行默认写工作区 `runs/`。历史 effectiveness_v3 跑数已删除。

## Thor 第三阶段新增入口（2026-09-17）

- `stages/effectiveness_v3/prepare.py`：恢复本机公开数据、重建划分并核对历史manifest，输出当前宿主数据证据。
- `stages/effectiveness_v3/readiness.py`：只读数据准入检查；提供CLI冻结前的G2审查校验。
- `tests/test_v3_migration.py`：因素阈值、DYN对照、候选完整性、冻结审查、源码身份自引用回归。
- `reports/effectiveness_v3/readiness/`：本次准备验收；不进入正式统计。
- `docs/decisions/A28.md`、SDD §10：本轮修补及剩余G2/G3实施条件。
