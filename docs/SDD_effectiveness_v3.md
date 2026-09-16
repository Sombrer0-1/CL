# effectiveness_v3 软件设计说明（SDD）

版本：2026-09-16。依据：[PLAN](../PLAN.md) §1–8、本地原PDF §4.1–4.5 / 式(1)–(5) / Algorithm 1，以及现有源码审查。**本文是第三阶段架构契约，随仓库迁移。** G0/G1 独立入口与测试已在 `src/orion_repro/stages/effectiveness_v3/`；G2–G4 必须在新宿主重新校准与冻结。[A26](decisions/A26.md) 放弃 RTX 5090 上的开发冻结和正式跑数，不得把那些数字写进新冻结协议。源码修补见[审计记录](../reports/effectiveness_v3/IMPLEMENTATION_AUDIT.md)，其中本机 GPU 测量不是新宿主证据。下文部分“待实现”条目以换机后重验为准，不把设计文档写成能力已经验收。

本文所有代码/产物路径相对仓库根。study_id 固定为 `effectiveness_v3`，实验协议版本为 `effectiveness_v3_paper_feedback_v1`，开发协议为 `effectiveness_v3_development_v1`。内容变更通过 revision 和内容哈希追踪，不改写旧冻结对象。

## 1. 架构与边界

```text
ExperimentDesign（不可执行）
    ↓ G0/G1：身份隔离、回归与真实数据小型验收
DevPlanner → 独立开发子进程 → ProbeManifest + 全部原始证据
    ↓ ScenarioAuditor / ResourceCalibrator / StaticSelector
CalibrationResult → G2审查 → ProtocolFreezer
    ↓ FrozenProtocol（不可变、含哈希）
MatrixEmitter → G3完整验证 → StudyExecutor（串行fresh process）
    ↓ run_from_spec
Benchmark + Strategy + Replay/Plugins + DataSupply
    ↓ train → evaluate → URGE → apply next config
PhaseRecorder / FailureRecorder / ConsumptionAudit
    ↓ Runs + Registry + Progress
CoverageResolver → PairingValidator → ReportBuilder → S01–S08判定
```

计算公式、资源执行和场景判定分离：控制器只读本轮反馈；资源预留器持有真实张量，不把未来序列传入控制器；报告器依据已冻结规则解释证据，不在汇总时重新选最优配置。测试反馈不进入人工搜索器。

## 2. 模块布局与复用决策

新建 `src/orion_repro/stages/effectiveness_v3/` 包，保持旧阶段入口可追溯，不批量重命名 pressure_*。

| 新模块（待实现） | 主要接口 | 责任与可复用部分 |
|---|---|---|
| `schema.py` | `load_design(path) -> Design`；`validate_frozen(...)` | 严格区分设计/开发配置/冻结/正式矩阵，拒绝未知必需字段与非法数值；复用基础spec字段验证 |
| `context.py` | `StageContext(root, study_id, revision)`；`assert_output_path(path)` | 新阶段目录、进度、锁、平台身份；禁止输出解析到历史目录或工作区外 |
| `development.py` | `plan_probes(design, evidence) -> ProbeBatch` | 依赖顺序生成资源扫描、6候选静态搜索、控制因素和敏感性；失败与原始config都保存 |
| `calibration.py` | `calibrate(manifest) -> CalibrationResult` | 源域、完整性、平台与测量验证，按PLAN固定规则选Q/L/eval/S*；不读取formal P/S |
| `coverage.py` | `audit_scenarios(records, rules) -> ScenarioCoverage` | S01–S08区间、实际动作和资源证据；区分不可用/未建立/未运行/已建立 |
| `protocol.py` | `freeze(design, calibration, identity) -> FrozenProtocol` | canonical内容hash、输入闭包、不可变写入；拒绝隐含默认值和未决必需参数 |
| `matrix.py` | `emit(frozen) -> MatrixManifest` | A–H矩阵、唯一cell_id、3 seeds、交错顺序、不可执行格的原因；先写临时目录再验证提交 |
| `executor.py` | `dry_run(matrix)`；`execute(matrix, context)` | 身份检查、互斥锁、子进程、失败/中断账本、复用；借鉴旧study但不引用旧常量 |
| `report.py` | `collect(manifest)`；`validate_pairs(...)`；`render(...)` | 从矩阵定义左连接结果，保留所有尝试、缺失及失败；禁止glob后按method静默覆盖 |
| `__main__.py` | 子命令 `inspect / develop / freeze / emit / run / report` | 只有完成模块与测试后才发布可执行命令；帮助默认只读，不默认运行旧formal |

通用底层继续使用 `runner/loop.py`、`control/`、`strategies/`、`prefetch/`、`memory/`。必要扩展局部化并回归历史接口，不为目录整洁改变训练语义。

必须扩展的现有接口：

- `runner/spec.py` 与 `control/ablation.py` 增加 `plugin_policy=fixed_advanced`，返回实际always-on策略但保留URGE原建议。F11起始插件开启；R11起始关闭；adaptive起始关闭。**当前仅支持adaptive/fixed_default，F11尚不可运行。**
- `runner/reuse.py` 的新阶段身份加入硬件、实际依赖、完整引用文件hash和冻结hash；旧v2/v3身份规则保持可解释。新阶段采用具名新身份schema，不能仅靠当前requirements文件代表实际环境。
- `memory/phase_recorder.py` 的host峰值补充采样聚合；当前PhaseRecord中的RSS/PSS是阶段结束快照，不能改标签称峰值。异步采样与phase切换需加phase token或锁，防止旧phase样本写入新phase峰值。
- `runner/loop.py` 将训练/评价/控制/重配置/失败的当前阶段显式传给记录器；零值不是unknown；实际预算来源、插件状态、消费计数持久化。
- `memory/host_enforcement.py` 当前只提供探测，不提供已验证的host执行器。H组需要下述专属cgroup执行接口，不能复用“检测到可写”作为执行证明。

## 3. 目录与身份契约

| 路径 | 用途 |
|---|---|
| `experiments/effectiveness_v3/design.yaml` | 不可执行设计清单；场景、分组、候选规则，无冻结预算 |
| `experiments/effectiveness_v3/revisions/<revision>/` | probe_manifest、calibration、frozen_protocol、matrix、资源序列、IO描述及哈希闭包 |
| `configs/effectiveness_v3/<revision>/{dev,formal}/` | 实际配置，formal只在冻结后生成 |
| `runs/effectiveness_v3_<UTC>_<uuid>/` | 一个fresh process的一次尝试；不能覆盖既有目录 |
| `runs/effectiveness_v3/<revision>/progress.json` | 原子持久化进度，与旧progress完全分离 |
| `runs/effectiveness_v3/<revision>/executor/` | 子进程临时配置、stdout/stderr及启动身份 |
| `reports/effectiveness_v3/<revision>/` | 正式报告与覆盖、配对、图表 |
| `reports/effectiveness_v3/IMPLEMENTATION_AUDIT.md` | 本次实现审查/补丁证据，与正式结果分开 |

既有全局 `experiments/registry.jsonl` 允许**追加**含study/revision/attempt_id字段的记录，不重写历史行。使用同一 `runs/.orion_project.lock` 防止跨入口并发；锁存在不等于锁被占用，不删除锁文件绕过互斥。数据盘与工作区runs不同步问题在STATUS中保留；新入口以工作区runs为唯一写入点。

`cell_id = group/dataset/scenario/method/seed/variant`。`attempt_id`每次启动唯一。配置比较用resolved spec的canonical JSON hash，排除运行时间/UUID等非语义字段；全部路径解析并记录，但身份依赖文件内容，不依赖路径字符串。static S*的选择ID必须进入正式配置。

平台身份至少含 GPU型号/UUID/总显存、逻辑device到物理卡映射、driver、torch/torchvision/Avalanche、CUDA runtime、Python、CPU/OS、数值模式；实际资源占用是快照，不能加入每次变化就失效的稳定身份。新平台或关键依赖变化拒绝复用并要求新冻结revision。

## 4. 数据结构与验证不变量

### 4.1 ExperimentDesign（不可执行）

字段：`kind=experiment_design`、`executable=false`、`study_id`、`design_version`、`scenario_ids`、`groups`、`seeds`、`calibration_rules`、`parameter_sources`、`gates`。不得提供可以被旧执行器误认为正式队列的`configs`键；TBD只允许存在于设计描述或显式pending字段。

### 4.2 ProbeManifest

每条包含 `probe_id / config_hash / run_id / role / dataset / source_hash / environment_hash / platform_hash / split_hash / feedback_source / phase / status / full_stream / n_expected / n_trained / n_evaluated / artifact_hashes`。

校准必须从原始resolved config、summary、phase_trace和experience_metrics交叉核对这些字段，不能信任一行人工填写的completed。反馈源缺失、冲突或非`development_val_seen`立即拒绝。完整流校准要求n_trained=n_evaluated=n_expected；单experience评价诊断显式标记，只用于eval诊断，不能进入L_cal或静态选择。

各候选具有唯一ID和终态。implementation_error会阻止冻结；OOM允许保留为资源不可行；interrupted/not_run不能被当作失败候选静默丢弃。重复尝试全部保留，选择首次符合当前身份的完整成功尝试；若因计时异常重测，以预先规则整组追加、不替换最快值。

### 4.3 CalibrationResult

每数据集/资源场景独立字段：`eval_batch`、`quota_tight_bytes / quota_loose_bytes`、`latency_cal_s`、`static_selections`、`memory_feedback_definition`、`replay_representation`、`phase_resource_evidence`、`parameter_sensitivity`、`scenario_coverage`、`probe_run_ids`、`search_cost_s`。

DYN必须有N长非负整数预留序列、过渡当轮训练/评价及之后反馈窗口证据；IO必须分串行供给工作、队列消费者等待、producer工作。**不能把异步producer耗时与consumer wait相加当阻塞耗时。** H组必须有cgroup限制验证、memory.events和swap policy证据。

Q网格按PLAN；中预算定义为16 MiB网格上最接近(Q_tight+Q_loose)/2且严格在两者之间的值（等距取较低者）。无合法中值则冻结失败并解释，不能复用端点冒充新格。L_cal只来自同dataset/宽预算/S0完整开发2次的合并中位数。F组另验宽配额下AGEM/EWC可行性，主A/B资源预算不能按F正式结果调整。

### 4.4 FrozenProtocol

字段闭包：schema/study/protocol/revision、design hash、source/env/platform/data hashes、calibration hash、输入artifact hashes、method_specs、三seed、groups、cell_order、resource_schedules、data_supply、feedback定义、统计规则、失败策略、scenario_status。

采用排序canonical JSON计算SHA256，排除`frozen_hash`自身及生成时间；写入采用临时文件+原子rename，目标存在且内容不同则拒绝。冻结中正式启用的字段不允许null、NaN、Inf、TBD；未建立组应有reason和证据引用，不能生成无效配置占位。`memory.max`等bytes为正整数，不接受bool/float；比例与accuracy单位显式检查，S_initial可>1不得悄悄裁剪。

### 4.5 MatrixManifest / RunSpec

矩阵字段：kind、study/revision、frozen_hash、cell definitions、order、eligible_cells、excluded_cells。组名额固定A48/B30/C15/D9/E12/F6/G24/H9；manifest同时报告设计153、具备执行条件数、未建立数，禁止只显示较小分母假装全面完成。

RunSpec必须满足通用spec+阶段扩展：cell_id、source/env/platform/frozen hashes、new epochs=1、完整流、数据变换/评价batch、控制四因子单位、plugin_policy与实际安装集合、物理quota/host机制、reservation/IO引用hash、测量和输出规则。S*例外只能改变开发获选batch/replay，不能顺便改模型/学习率/变换。

输出安全检查对resolve后的真实路径执行，防止`..`或符号链接绕到旧目录。执行前重新计算全部hash；不一致时明确停止该revision，不能更新hash后继续拼接比较。

## 5. 运行时状态机与时序

状态：`planned → validating → launching → running → completed / cuda_oom / host_oom / budget_exceeded / numerical_error / implementation_error / interrupted`；`scenario_not_realized`是矩阵覆盖状态，不是假run。retry产生新attempt，原终态不可覆盖。

每次运行：

1. 验证身份及可用资源，取得项目锁；创建新run目录，保存启动配置/环境。cgroup子进程在导入CUDA和载入数据前进入专属组。
2. 安装进程allocator配额，setup记录模型、数据/变换、策略及插件顺序。训练前确认样本/类映射及标签。
3. 第k轮：`resource_transition → admission → training → evaluation → controller → reconfiguration`。外部预留变化真实分配并记录，控制器只见当轮已发生的测量。
4. `training`仅一次strategy.train，包含replay、GEM投影/EWC辅助遍历及buffer更新；记录new/replay/auxiliary visits及时间。训练成功后立即增加trained计数。
5. 评价在独立eval模式、不更新BN/优化器/RNG，保存A[k,i]与correct/total，再增加evaluated计数。P/S定义复用协议，列明索引，未知不填0。
6. 控制读取训练阶段allocated峰值（device主轨）、实际learning_s、P/S；记录U、Thr、因子、MB/MR前后、floor结果、建议和实际plugin模式、保护干预。
7. k<N−1才应用配置；末轮记录建议但不apply。若选择experience_boundary checkpoint，末轮也必须保存完整边界；正式主比较默认none，不主张跨中断计时恢复等价。
8. 无异常则完成summary及所有闭合phase。任意异常先记录失败phase的当前峰值/阶段上下文，再清理预留张量/采样器/loader/log；测量失败记录null+reason，不遮盖原异常。

OOM不能读取`recorder.last`替代仍开放的失败phase；本轮已修补。OOM后的CUDA进程退出，不尝试在同run继续。SIGTERM/INT只处理自己创建的进程组，保存已知产物与墙时；再次执行从头新attempt。

进度采用原子写，父进程记录pid及进程start-time/命令签名，不能只用pid存在判断是否旧任务（避免PID重用）。中断重启先协调summary、registry与progress，矛盾留待检查，不默默覆盖。异常退出的证据不足时不得仅凭returncode判host_oom。

## 6. 资源与数据供给契约

### 6.1 测量

`PhaseRecord`输出phase/index、monotonic start/end、duration、allocated/reserved current/peak、quota、reservation bytes、RSS/PSS end、采样RSS/children RSS峰值、system available/swap、sample interval/count、missing_reason。峰值采样范围和下界性质写入schema；children RSS可能共享页重复，不能与PSS混合求和。

采样phase token与snapshot一致；阶段内无嵌套GPU peak reset。初始warmup、cache政策所有方法相同；控制/重配置边界同步费用既在分项也可能包含实际online墙时，解释口径，不重复求和。监测开销用独立开发测量，不把监测关闭run并入主表。

失败摘要：failure_phase/index、trained/evaluated、request bytes、allocated/reserved峰值、quota、external reservation、unannounced_transition、异常栈及measurement_error。训练成功但评价失败不是启动失败；采样未知不是0。

### 6.2 GPU与host执行

GPU：使用现有allocator fraction接口，验证实际device与limit；范围不含driver/non-Torch。`ResourceEnvelope.transition`只在量变化时调整持有张量；无变化时保留，避免每轮无意义释放/重分配扰动allocator。释放不默认调用empty_cache，保留真实reserved表现。

host（H组条件实现）：`HostEnvelope.create(delegated_root, quota, swap_policy)`只建实验专属子cgroup；`attach_before_exec(pid)`；`read_events()`；`close()`只清理自己创建且已空的组。父级/全局memory.max禁止修改。记录memory.current/peak/events、swap.max、child pids；硬OOM可杀掉子进程，因此父进程必须回收证据。权限不足返回capability unavailable，不换成RLIMIT_AS冒充RAM。mock与小进程真实限制验收后才允许H正式配置。

主轨replay保持原存储表示，单独测owned tensor bytes与数据集引用。若探索materialized host replay，需新representation ID、共同静态对照与独立协议，不能混入本次A–G；H默认仍用已对齐表示，无法建立有辨识度压力时保留缺口。

### 6.3 插件与访问

完整ER安装GEM/EWC；AGEM扩展只能选EWC。每轮输出base algorithm、installed、requested/applied enabled、hook counts、辅助访问、GEM经验keys/样本数、EWC状态字节、QP fallback计数。未知EWC辅助访问不得填0；实现审计counter后校对实际数据访问。关闭后状态保留并继续计入资源；扩容只能用合法现有/后续样本。

F11增加fixed_advanced语义需测试从第0轮enabled，不覆写controller原建议；R11固定default。测试基础AGEM/GEM不因禁用optional而停止自身投影。

### 6.4 预取

索引在主线程、独立RNG下规划；producer只取已授权当前流及replay样本，禁止提前将未来experience样本参与更新。新/replay统一计数，transform需按sample/experience/seed确定，避免线程消耗共享随机源造成训练变化。

新队列版本校验experience_id与config_version；producer异常传回，消费者早停明确close；join超时不能静默丢弃仍活动线程后重用队列。每轮记录线程退出、丢弃批次、queue depth、planned/consumed摘要。重复迭代时先重置本轮统计或明确累计口径。

静态off/on：消费索引、标签、tensor hash、逐步loss及最终参数小例通过预先容差；确定性CPU小例要求完全一致，GPU确定性小例建议atol=1e-6/rtol=1e-5并在首次开发前冻结。正式完整流保存hash/访问数和完整A矩阵，不能仅靠最终准确率相同证明等价。

供给指标：串行blocked_s=load_next_s；预取blocked_s=consumer_get_wait_s；producer_work_s单列并可与训练重叠。IO门槛只用串行blocked_s/learning_s，不能用on端producer+wait。on/off所有输入后端、线程/worker、pin_memory和hash记录开关一致。Orion配对允许因latency反馈分化，另报告控制轨迹交互。

## 7. 场景判定器与统计

`ScenarioEvidence`字段：scenario_id、implementation_status、construction_status、evidence_run_ids、experience_windows、predicate_values、actions_applied、failure_category、effectiveness_verdict、limitations。

construction_status枚举 `not_run / realized / failed_to_construct / unavailable`；implementation_status为`pending / pass / fail`。效能判定为支持/部分支持/不支持/未验证，不由建设状态直接推导。

- S01：资源余量存在，P/S不足或L超目标；同时保存全部因子以避免“一个指标差必定高U”的错误推断。实际扩张用下一轮applied整数或真实插件启用计，不把末轮建议算事件。
- S02：资源有余量且(P≥Pth且S≥Sth)或L≤Lth，分别标注充分质量/低延迟；可与其他不足项并存，不要求所有场景互斥。完整区间输出，不后选最好seed。
- S03：真实紧配额与训练压力证据，控制后轮次可观测。allocated/reserved、model/reservation分别看；“接近reserved配额”本身不证明内存因子下降。
- S04：突变当轮与后续窗口分开；反应与恢复必须有实际可用轮次，不将一次成功reservation事件当整轮可行。
- S05：资源/计算量随经验变化的实测斜率与原值；delta版本轨迹差异及收益独立判定。
- S06：按预冻结顺序计算权重，报告期望方向和真实差异，不按结果改名称。
- S07：开发自然瓶颈、输入等价、正式净收益三项独立。
- S08：R11/F11/静态/预取对照签名合法，报告组件开销和系统净变化。

先按matrix列出所有名额再解析attempt。一个cell多次完成要显式选择规则/审计，不用字典赋值吞掉旧run。配对签名包含dataset/split/classes/transforms/model/optimizer/seed/eval/precision/platform/source/quota/资源序列/IO后端；method差异使用预定义allowlist。敏感性组只放行声明的单因素差异。不同protocol、预算场景、修补前后代码不自动配对。

结果文件：`coverage.csv`、`attempts.csv`、`means.csv`、`paired.csv`、`phase_resources.csv`、`control_events.csv`、`plugin_activity.csv`、`supply.csv`、`scenario_coverage.json`、`selection.json`、`RESULTS.md`及图。搜索成本单列。失败率用所有计划可执行格；质量/加速明确成功配对n，不用缺失补0或只写幸存者均值。原始run_id贯穿所有表。

## 8. 验证矩阵与实施门槛

| Gate / 测试ID | 实施任务 | 验收证据 |
|---|---|---|
| G0 / V01 | StageContext、独立CLI与路径 | dry-run无写入；拒绝旧stage、目录穿越、旧硬件/锁hash；历史保护hash一致 |
| G1 / V02 | 公式/指标/floor/单位 | 单元边界、sigmoid单调、持续小更新跨整数、末轮不apply、P/S矩阵核算 |
| G1 / V03 | Replay/插件/访问 | 真实strategy 4 experiences的off→on→off→on、收缩再扩张、无丢失历史复活、基础插件保持、state bytes及aux计数 |
| G1 / V04 | 评价与时序 | 训练一次、评价前后模型/BN/optimizer/RNG不变；trained/evaluated分开；末轮checkpoint完整 |
| G1 / V05 | 阶段资源/失败 | GPU真实配额分配/过渡/训练与评价OOM；失败phase峰值；测量失败不掩盖原错误；host端快照与采样峰值区分 |
| G1 / V06 | 数据供给 | CPU/GPU小例off/on更新比较、早停/异常/迭代/陈旧项/线程退出；串行next计时、producer不重复相加 |
| G1 / V07 | 校准/冻结/报告 | 缺少反馈拒绝、partial完整流拒绝、候选缺失拒绝、相同种子不同身份禁止配对、重复attempt保留、修改引用hash失效 |
| G1 / V08 | 进程与状态 | 子进程异常、信号中断、PID重用、原子progress恢复、锁争用、日志/summary不一致；无超时杀正常训练 |
| G2 / V09 | 真实完整开发流 | 两数据集/各预算、因素分解、静态搜索、敏感性及S01–S08证据；所有候选状态可审计 |
| G2 / V10 | host能力 | 本机专属cgroup与replay存储测量；有权限则真小进程OOM，否则unavailable证据；不影响GPU线推进 |
| G3 / V11 | 不可变冻结/发射 | A–H名额计数153、eligible/excluded分母、3 seeds、控制allowlist、无TBD正式配置 |
| G4 / V12 | 正式与报告 | 可执行格终态齐全、失败保留、场景判定与C01–C08映射、3 seeds原值/均值/离散和配对n |

实现顺序：context/schema → 测量与runtime缺口/回归 → development/calibration/coverage → protocol/matrix → executor/report → 完整开发与冻结 → 正式。G1小型功能检查可用合成数据，但V09和正式必须用完整约定真实基准，不能互相代替。

当前代码补丁覆盖 V02–V07 中已写入测试的部分；V01 入口已在树中。所有未在新宿主重验的项保持 pending；不能把旧宿主测试全绿写成 V01–V12 全部通过，也不能把 5090 校准当作已冻结协议。

## 9. 源码变更与交付治理

- 每个实现单元记录变更原因、测试、影响及源码身份；协议变化产生新revision，影响到的正式配对全组重验，旧attempt保留。
- 每次正式发射前保存Git commit（若有未提交改动，同时保存patch与可恢复源码包）、resolved config、环境锁/实际版本、完整引用hash；只有hash而无对应源码并不足以重跑。
- 不改`reports/light24/`、`reports/pressure_v2/`、旧配置/冻结协议/data manifests；发现历史缺陷以新审计文档限定结论，不重写旧数字。
- G1/G2 在新宿主通过后，README 再加入该宿主实际 CLI；此前只链接设计与 STATUS，不留貌似可跑的旧 formal 命令。
- 整理后的接手文档见 [HANDOFF](HANDOFF.md)。工作量遵循 PLAN 估算与非硬截止约定，长运行定期记录进度，不因实验负结果停止留证。换机后从 G2 重新校准，见 [A26](decisions/A26.md)。
