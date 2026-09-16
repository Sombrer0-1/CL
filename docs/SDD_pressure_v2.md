# pressure_v2 软件设计说明（SDD）

依据：[第二阶段归档计划](plans/pressure_v2.md)，原PDF §4.1–4.5、式(1)–(5)、Algorithm 1。本文为历史设计契约：实现、开发校准和冻结已完成，正式矩阵中断、未验收，详见[收尾记录](../reports/pressure_v2/CLOSEOUT.md)。下文接口及验收要求不代表每项均已通过；下一阶段安排见[当前计划](../PLAN.md)。所有代码/产物路径相对仓库根目录。

## 1. 架构与责任

```text
设计文件 → 开发探测/校准 → FrozenProtocol → MatrixEmitter
                                               ↓
StudyExecutor → run_from_spec → ResourceEnvelope → Strategy + DataSupply
                          ↓             ↓             ↓
                   PhaseRecorder ← 实际资源/插件/消费证据
                          ↓
              原始运行产物 → 按冻结manifest汇总 → 有效性/覆盖判定
```

- `control/urge.py`和`control/controller.py`：保持公式与连续状态，不能加入临时调参、配额裁剪或IO造负载逻辑。
- `memory/enforcement.py`：保留真实allocator配额；ResourceEnvelope负责可控外部占用，与模型/回放记账分开。
- `strategies/builder.py`：实际插件构建、启停及buffer调整，不接受没有插件却声称advanced。
- 执行器负责进程/注册/重启，不能改变算法。分析器负责完整性与比较分组，不能把实现错误筛掉后宣称全部完成。

## 2. 已实现修补及影响

| 修补 | 实现位置 | 验证 |
|---|---|---|
| 空advanced切换在任何状态修改前失败 | strategies/builder.py apply_runtime_config | 无TogglePlugin时batch/buffer不变，明确UnsupportedAdaptationError |
| 空经验buffer缩放不再触发库IndexError | strategies/builder.py | 未训练时调整容量后可真实训练；非空缩放保留原路径 |
| 返回真实插件状态 | 同上 applied_optional_plugins；loop控制轨迹 | 显式GEM/EWC对象启停与返回字典一致 |
| OOM阶段上下文 | runner/failures.py + loop.py | failure_phase、failure_experience、trained/evaluated数量分开；evaluation OOM不是startup |

第一阶段完整配置未安装插件；default-only运行数值仍是当时实现的真实结果，不能改名为已验证的完整插件行为。新插件配置不能写回旧run或旧配置。旧readiness模式字符串检查有局限，不能用作第二阶段门槛。

## 3. 模块与接口（待实现）

### 3.1 协议冻结与矩阵生成

新增`pressure_protocol.py`：

- `validate_design(design) -> Design`：只接受kind=experiment_design；不得直接送入训练器。
- `calibrate(probe_manifest) -> CalibrationResult`：验证开发数据域、完整经验、资源证据和全部候选outcome。按PLAN唯一确定Q_tight/Q_loose、eval_batch、L_cal、各情形S*。
- `freeze(calibration, source_identity) -> FrozenProtocol`：输出`experiments/pressure_v2/frozen_protocol.json`，schema_version、protocol_id、来源run列表、selection_rule、dataset_hash、source_hash、已确定数值、执行顺序和输入文件hash必须完整。
- `emit(frozen) -> list[RunSpec]`：生成`configs/pressure_v2/`及各阶段矩阵；method/场景/预算/偏好/decay/IO标识互不覆盖；恰好54个正式格，开发诊断单独计数。

`FrozenProtocol`关键字段：quota_bytes（整数）、reserved_bytes_by_experience（整数列表）、latency_threshold_s、memory_threshold_mib、thr0/delta/alpha/beta、eval_batch、initial_counts、optional_plugins、static_selections、seeds、run_order、calibration_evidence。未冻结不允许null/TBD绕过验证。

现有study只支持light24；不得仅改字符串继续写runs/light24_progress.json。新执行器必须独立路径，并在加载配置时比较冻结hash。

### 3.2 ResourceEnvelope

新增`memory/resource_envelope.py`，接口：

- `install_quota(bytes, device) -> QuotaRecord`，委托现有quota模块。
- `transition(k, reservation_bytes) -> ReservationRecord`：持有本进程CUDA uint8张量并实际分配；GPU同步后记录requested、actual tensor bytes、allocated/reserved前后值；分配失败记resource_transition。
- `close()`：只释放自身张量，不重置其他策略/进程资源，不调用全局缓存清除作为默认优化。

DYN默认三段low/high/low：k=0..2、3..5、6..8。新外部占用在k=3/6训练前进入，此时上一轮控制决策基于旧情形；**突变首轮无法提前规避的失败记unannounced_transition，不算已反馈调节失败**。为检验反应，开发必须选保证这一过渡首轮可完成的预留量；k=3训练反馈后，k=4/5使用新决策且占用保持。释放阶段同理。不要伪称提前感知；不把未来序列告诉控制器，静态方法使用同序列。

当预留张量被释放，allocator可能保留reserved缓存；allocated变化与reserved驻留分别记录。URGE继续使用原训练allocated峰值（含该可见同进程外部占用）；同时输出扣除reservation的model相关观测值，不把预留量混成模型内存。此设定模拟同进程共享资源占用，不宣称真实多进程调度或Jetson。

### 3.3 PhaseRecorder与失败模型

扩展`memory/probe.py`/`runner/artifacts.py`，避免另建第二套计时器：

`begin(phase,k)`先同步并重置CUDA allocated/reserved峰值；`end(phase,k)`同步后读取最大值并合并host采样。记录setup、resource_transition、training、evaluation、controller、reconfiguration。峰值为观测范围，不把轮询RSS叫绝对瞬时峰值。

每阶段记录：monotonic起止、duration_s、allocated_current/peak、reserved_current/peak、quota_bytes、external_reservation_bytes、RSS/PSS/children RSS、系统swap（明确不是进程swap）。失败摘要含失败阶段、experience、已训练数、已评价数、请求与配额、traceback；取不到的字段用null不填0。若异常文本包含异常巨大nonTorch值，不参与资源统计。

禁止在OOM后复用损坏CUDA上下文重试同run；新子进程、新run_id，原失败保留。OOM策略对所有方法一致。

### 3.4 插件和访问审计

完整O00/O10/O01/O11显式`optional_plugins=gem_ewc`、`optional_start_enabled=false`；R11同样安装但policy=fixed_default。S0/S*无额外插件，是方法定义差异，须报告常驻状态开销。

新增或扩展插件审计：before_backward/after_training_exp的实际调用次数、enabled区间、GEM参考样本数、EWC额外前后向访问、plugin_state_bytes；未知计数不得记0。真实小型strategy.train至少2个经验，验证开→关期间hook确实执行/停止，历史状态按约定保留。

TogglePlugin目前是pause-compute-keep-state；停用不等于释放历史GEM/EWC张量。这是工程重建限制，压力实验应验证常驻内存，不能把“mode=default”写成“释放MO”。若常驻状态导致无法缓解压力，先报告原重建失败；不擅自清除历史状态改善结果。释放/迁移策略须单独设计变体并验证恢复语义。

MB0/MR0与m_batch/m_frame必须同单位。开发记录真实样本/缓冲表示和边际资源剖析；固定线性计数映射只是局部模型，不能称完整成本模型。若采用现有count-space proxy（1.0），显式标记而非称MiB物理分配。raw replay的CPU引用/数据集驻留不能作为新增GPU样本内存；host只观测不约束，保留覆盖缺口。

### 3.5 数据供给与预取

新增可选DataSupplyProfile，仅针对IO组；主表后端不变。使用同一真实图像源、相同解码结果、变换与顺序的按需供给。不得重复增强或人为sleep制造优势。将new+replay消费顺序、输入tensor/label hash按批滚动摘要持久化，记录生产者/消费者耗时、队列等待和废弃批次。

小型确定性验收记录逐步loss和最终参数；全流以消费摘要+矩阵+访问计数作为有限证据。预取开关是IO对照唯一差别，CPU/GPU设置及pin_memory/depth/worker提前冻结。若自然IO未达瓶颈门槛，不扩展人为昂贵计算，报告该适用场景在本平台未建立。

### 3.6 执行与报告

新`pressure_study.py`可复用study底层进程函数，独立`runs/pressure_v2_progress.json`和`runs/_pressure_v2_executor/`；项目级锁防止两个阶段并发训练。读取冻结manifest，不在运行中生成配置或修改源码。超时不杀训练；信号仅终止自己创建的子进程组。

reuse身份必须包含实际配置及所有引用manifest/资源序列/数据供给描述内容hash，不能只hash路径。完成复用与资源失败跳过有明确状态；修复代码后不自动放行旧身份。中断保留部分产物，从头新run；完整checkpoint恢复不在本期。

新`pressure_report.py`按照study/protocol/source/dataset/scenario/quota/reservation/method/seed分组。run级失败和每阶段峰值独立表；配对前检查同一控制变量签名，S*明确允许batch/replay差异。动态和固定预算、不同IO后端不混表。

输出：coverage、attempts、means、paired、resource_phases、control_events、plugin_activity、selection、scenario_coverage、RESULTS。scenario_coverage区分realized/failed_to_construct/not_run；不能用已跑格数替代场景覆盖或方法有效性。

## 4. 训练时序契约

1. 校验冻结协议、数据和源码身份；初始化共同环境及quota。
2. 安装/调整本轮外部占用，阶段记录；不得把未来变化输入控制器。
3. 用上轮已应用配置训练一次，记录真实样本访问和插件hook。
4. 用共同eval_batch评价，保存准确率矩阵。
5. 从训练阶段反馈计算URGE/Thr，保存连续提议及整数映射。
6. 若非末经验，应用下一轮配置并记录实际插件/容量；末经验仅记录不应用。
7. 失败按所在阶段持久化，不用评价完成数掩盖已训练事实。

## 5. 验证与实施顺序

| 顺序 | 交付 | 放行条件 |
|---|---|---|
| I1 | 当前缺陷修补、冻结第一阶段证据 | 回归测试通过；不重写原运行 |
| I2 | PhaseRecorder、ResourceEnvelope、真实插件/访问审计 | CPU单元测试+小型GPU实际分配/过渡/OOM/插件hook验收 |
| I3 | 独立配置schema、校准器、选择器及manifest | 12/18候选含失败测试；非开发反馈拒绝；hash更改不可复用 |
| I4 | 新执行器与报告 | 进程重启/信号/失败/分组隔离测试；不覆盖light24 |
| I5 | D0/D1开发诊断、IO等价与场景覆盖 | 论文启用情形有真实证据，不能只看阈值字符串 |
| I6 | 冻结并执行54格 | 当前同源码、3seeds、预算对照齐全；保留全部失败 |
| I7 | 限定结论和研究启发 | 逐论文情形判定；无证据不宣称优越性 |

在I5之前不生成貌似可运行但含待定值的正式矩阵。若关键情形构造失败，保持阶段未完整验证并向用户解释，不无限补实验、不假称验证完成。

## 6. 时间与范围控制

正式54格；开发只做预定的有辨识力诊断和最多18格静态候选。主要场景NC，CIFAR100仅OOM复核，不追加全数据集/全算法/Oracle。预计总工作量约6–14小时，先实测advanced开销再更新，无硬截止。host硬限额、共享内存硬件、任意插件迁移、全checkpoint以及新方法研发不顺带实现。
